"""Profile one training iteration to find where time is spent.

Usage: python scout-bot/profile_iteration.py [checkpoint_path]
Default: scout-bot/v3_4/latest.pt
"""
import sys
import time
import torch
import random
from collections import defaultdict
from contextlib import contextmanager

from game import Game, Phase
from encoding import (
	encode_state, encode_hand_both_orientations, get_action_type_mask,
	get_play_start_mask, get_play_end_mask, get_scout_insert_mask,
	get_sns_insert_mask, get_legal_plays, decode_action_type,
	decode_slot_to_hand_index, HAND_SLOTS, PLAY_SLOTS, SCOUT_INSERT_SIZE,
	INPUT_SIZE,
)
from network import ScoutNetwork, masked_sample
from training import play_game, compute_gae, ppo_update, OpponentPool

# --- Timing infrastructure ---
_timers = defaultdict(float)
_counts = defaultdict(int)

@contextmanager
def timer(name):
	t0 = time.perf_counter()
	yield
	_timers[name] += time.perf_counter() - t0
	_counts[name] += 1

def print_timers(total_seconds):
	print(f"\n{'='*60}")
	print(f"Total wall time: {total_seconds:.3f}s")
	print(f"{'='*60}\n")
	# Sort by time descending
	items = sorted(_timers.items(), key=lambda x: -x[1])
	for name, t in items:
		pct = t / total_seconds * 100
		count = _counts[name]
		per_call = t / count * 1000 if count else 0
		print(f"  {name:30s}  {t:8.3f}s  ({pct:5.1f}%)  n={count:6d}  {per_call:.3f}ms/call")

# --- Instrumented game play ---
def profiled_play_game(network, num_players, opponent_pool=None, training_seats=1):
	"""Play one game with fine-grained timing."""
	game = Game(num_players)
	networks = []
	for i in range(num_players):
		if i < training_seats:
			networks.append(network)
		elif opponent_pool:
			networks.append(random.choice(opponent_pool))
		else:
			networks.append(network)

	all_records = []
	for round_idx in range(game.total_rounds):
		with timer("game_logic"):
			game.start_round()

		# Flip decisions
		for p in range(game.num_players):
			net = networks[p]
			with timer("encoding"):
				ho = random.randint(0, HAND_SLOTS - 1)
				po = random.randint(0, PLAY_SLOTS - 1)
				t_normal, t_flipped = encode_hand_both_orientations(game, p, ho, po)
			with timer("forward_pass"):
				with torch.no_grad():
					h_normal = net(t_normal)
					h_flipped = net(t_flipped)
					v_normal = net.value(h_normal).item()
					v_flipped = net.value(h_flipped).item()
			with timer("game_logic"):
				did_flip = v_flipped > v_normal
				game.submit_flip_decision(p, do_flip=did_flip)

		# Play turns
		while game.phase == Phase.TURN:
			p = game.current_player
			net = networks[p]

			with timer("encoding"):
				hand_offset = random.randint(0, HAND_SLOTS - 1)
				play_offset = random.randint(0, PLAY_SLOTS - 1)
				state_tensor = encode_state(game, p, hand_offset, play_offset)

			with timer("forward_pass"):
				with torch.no_grad():
					hidden = net(state_tensor)
					value = net.value(hidden).item()

			with timer("masks"):
				hand = game.players[p].hand
				legal_plays = get_legal_plays(hand, game.current_play)
				at_mask = get_action_type_mask(game, legal_plays)

			if not at_mask.any():
				with timer("game_logic"):
					game._advance_turn()
				continue

			with timer("forward_pass"):
				with torch.no_grad():
					at_logits = net.action_type_logits(hidden)

			with timer("sampling"):
				action_type, _ = masked_sample(at_logits, at_mask)

			action_info = decode_action_type(action_type)

			if action_info["type"] == "play":
				with timer("forward_pass"):
					with torch.no_grad():
						ps_logits = net.play_start_logits(hidden, action_type)
				with timer("masks"):
					ps_mask = get_play_start_mask(legal_plays, hand_offset)
				with timer("sampling"):
					start_slot, _ = masked_sample(ps_logits, ps_mask)
				start_idx = decode_slot_to_hand_index(start_slot, hand_offset)

				with timer("forward_pass"):
					with torch.no_grad():
						pe_logits = net.play_end_logits(hidden, action_type, start_slot)
				with timer("masks"):
					pe_mask = get_play_end_mask(legal_plays, start_idx, hand_offset)
				with timer("sampling"):
					end_slot, _ = masked_sample(pe_logits, pe_mask)
				end_idx = decode_slot_to_hand_index(end_slot, hand_offset)

				with timer("game_logic"):
					game.apply_play(start_idx, end_idx)

			elif action_info["type"] == "scout":
				with timer("forward_pass"):
					with torch.no_grad():
						si_logits = net.scout_insert_logits(hidden, action_type)
				with timer("masks"):
					si_mask = get_scout_insert_mask(game, hand_offset)
				with timer("sampling"):
					insert_slot, _ = masked_sample(si_logits, si_mask)
				insert_pos = (insert_slot - hand_offset) % SCOUT_INSERT_SIZE

				with timer("scout_quality"):
					left_end = action_info["left_end"]
					play_cards = game.current_play.cards
					scouted = play_cards[0] if left_end else play_cards[-1]
					if action_info["flip"]:
						scouted = (scouted[1], scouted[0])
					new_hand = list(hand[:insert_pos]) + [scouted] + list(hand[insert_pos:])
					for s, e in get_legal_plays(new_hand, None):
						pass  # just timing the call

				with timer("game_logic"):
					game.apply_scout(left_end, action_info["flip"], insert_pos)

			elif action_info["type"] == "sns":
				with timer("forward_pass"):
					with torch.no_grad():
						si_logits = net.scout_insert_logits(hidden, action_type)
				with timer("masks"):
					si_mask = get_sns_insert_mask(game, action_info["left_end"], action_info["flip"], hand_offset)
				with timer("sampling"):
					insert_slot, _ = masked_sample(si_logits, si_mask)
				insert_pos = (insert_slot - hand_offset) % SCOUT_INSERT_SIZE

				with timer("scout_quality"):
					left_end = action_info["left_end"]
					play_cards = game.current_play.cards
					scouted = play_cards[0] if left_end else play_cards[-1]
					if action_info["flip"]:
						scouted = (scouted[1], scouted[0])
					new_hand = list(hand[:insert_pos]) + [scouted] + list(hand[insert_pos:])
					for s, e in get_legal_plays(new_hand, None):
						pass

				with timer("game_logic"):
					game.apply_sns_scout(left_end, action_info["flip"], insert_pos)
					# S&S play portion
				if game.phase == Phase.SNS_PLAY:
					# Simplified: just time the rest as game_logic
					# (recursive _play_turn would need same instrumentation)
					sp = game.current_player
					snet = networks[sp]
					with timer("encoding"):
						sho = random.randint(0, HAND_SLOTS - 1)
						spo = random.randint(0, PLAY_SLOTS - 1)
						sstate = encode_state(game, sp, sho, spo)
					with timer("forward_pass"):
						with torch.no_grad():
							shidden = snet(sstate)
							snet.value(shidden)
							sat_logits = snet.action_type_logits(shidden)
					with timer("masks"):
						shand = game.players[sp].hand
						slegal = get_legal_plays(shand, game.current_play)
						sat_mask = get_action_type_mask(game, slegal)
					# Force a play action for S&S
					if sat_mask.any():
						with timer("sampling"):
							sat, _ = masked_sample(sat_logits, sat_mask)
						sai = decode_action_type(sat)
						if sai["type"] == "play":
							with timer("forward_pass"):
								with torch.no_grad():
									sps_logits = snet.play_start_logits(shidden, sat)
							with timer("masks"):
								sps_mask = get_play_start_mask(slegal, sho)
							with timer("sampling"):
								ss_slot, _ = masked_sample(sps_logits, sps_mask)
							ss_idx = decode_slot_to_hand_index(ss_slot, sho)
							with timer("forward_pass"):
								with torch.no_grad():
									spe_logits = snet.play_end_logits(shidden, sat, ss_slot)
							with timer("masks"):
								spe_mask = get_play_end_mask(slegal, ss_idx, sho)
							with timer("sampling"):
								se_slot, _ = masked_sample(spe_logits, spe_mask)
							se_idx = decode_slot_to_hand_index(se_slot, sho)
							with timer("game_logic"):
								game.apply_play(ss_idx, se_idx)
						else:
							with timer("game_logic"):
								game._advance_turn()
					else:
						with timer("game_logic"):
							game._advance_turn()

	_counts["games"] += 1


def main():
	ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "v3_4/latest.pt"
	print(f"Loading {ckpt_path}...")
	checkpoint = torch.load(ckpt_path, weights_only=False)
	cfg = checkpoint.get("config", {})
	layer_sizes = cfg.get("layer_sizes", [512, 256, 256, 128, 128, 128])
	network = ScoutNetwork(input_size=INPUT_SIZE, layer_sizes=layer_sizes)
	network.load_state_dict(checkpoint["model_state"])
	network.eval()

	num_players = cfg.get("num_players", 4)
	num_games = cfg.get("games_per_iteration", 100)
	training_seats = cfg.get("training_seats", 4)
	ppo_epochs = cfg.get("ppo_epochs", 4)

	# Build a small opponent pool
	pool = OpponentPool(max_size=3)
	pool.add(network)
	opponents = pool.sample(num_players - training_seats) or None

	print(f"Profiling: {num_games} games, {num_players} players, {training_seats} training seats")
	print(f"Network: layers={layer_sizes}\n")

	# --- Phase 1: Game generation ---
	print("Phase 1: Game generation...")
	t_play_start = time.perf_counter()
	all_records = []
	for game_idx in range(num_games):
		profiled_play_game(network, num_players, opponent_pool=opponents,
						   training_seats=training_seats)
	t_play = time.perf_counter() - t_play_start

	# --- Phase 2: GAE computation ---
	# Need real records for this, use the un-instrumented version
	print("Phase 2: Collecting real records for GAE + PPO timing...")
	real_records = []
	for game_idx in range(num_games):
		records = play_game(network, num_players, opponent_pool=opponents,
							training_seats=training_seats,
							reward_distribution=cfg.get("reward_distribution", "terminal"),
							reward_mode=cfg.get("reward_mode", "game_score"),
							shaped_bonus_scale=cfg.get("shaped_bonus_scale", 0.0))
		for r in records:
			r.game_id = game_idx
		real_records.extend(records)

	with timer("gae"):
		advantages, returns, adv_std = compute_gae(real_records,
			gamma=cfg.get("gamma", 0.99), lam=cfg.get("gae_lambda", 0.95))

	# --- Phase 3: PPO update ---
	print(f"Phase 3: PPO update ({ppo_epochs} epochs, {len(real_records)} steps)...")
	network.train()
	optimizer = torch.optim.Adam(network.parameters(), lr=cfg.get("learning_rate", 3e-4))
	for epoch in range(ppo_epochs):
		with timer("ppo_update"):
			ppo_update(network, optimizer, real_records, advantages,
					   clip_epsilon=cfg.get("clip_epsilon", 0.2),
					   entropy_bonus=cfg.get("entropy_bonus", 0.01),
					   value_loss_coeff=cfg.get("value_loss_coeff", 0.25),
					   returns=returns,
					   entropy_floors=cfg.get("entropy_floors"),
					   entropy_floor_coeff=cfg.get("entropy_floor_coeff", 1.0))

	total = time.perf_counter() - t_play_start
	print_timers(total)
	print(f"\n  Game generation total: {t_play:.3f}s")
	print(f"  Steps collected: {len(real_records)}")
	print(f"  Steps per game: {len(real_records) / num_games:.1f}")


if __name__ == "__main__":
	main()
