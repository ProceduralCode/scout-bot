"""Bridging probe: test whether GAE can propagate credit through multi-step episodes.

Creates controlled episodes where decision 1 has a known good/bad action (established via
rollouts) and decisions 2+ are played by the network. Measures whether GAE assigns the
correct advantage sign to decision 1, and compares with rollout-based ground truth.

This bridges the gap between single-decision probes (value head can rank states) and real
training (GAE must attribute terminal reward through 3-10 intermediate decisions).

Usage: python -u bridging_probe.py [checkpoint_path]"""

import sys
import os
import time
import random
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from network import FlatScoutNetwork, masked_sample, masked_log_prob
from encoding import (INPUT_SIZE_V6, HAND_SLOTS_V6, encode_state_v6,
	get_flat_action_mask, encode_hand_both_orientations_v6, decode_flat_action,
	get_legal_plays)
from training import (StepRecordV6, rollout_from_states_batched_v6,
	compute_gae, _assign_round_rewards, _play_turn_v6)
from game import Game, Phase

NUM_PLAYERS = 4
ROLLOUTS_PER_ACTION = 20
STATE_GEN_TIME = 3.0  # seconds — keep short, candidates are cheap
MAX_SCREEN_STATES = 30  # cap candidates sent to rollout screening
MIN_TURNS_BEFORE_SNAPSHOT = 3  # skip early turns
VALUE_GAP_THRESHOLD = 0.3  # min |play_value - scout_value| to keep a state
MIN_EPISODE_DECISIONS = 3  # test player must make 3+ decisions for GAE to propagate

H = HAND_SLOTS_V6


# --- Phase 1: Generate candidate mid-game states ---

def generate_candidate_states(net, time_limit=STATE_GEN_TIME):
	"""Play games, snapshot mid-round positions where both play and scout are legal."""
	candidates = []
	deadline = time.time() + time_limit
	game_num = 0
	while time.time() < deadline:
		game = Game(NUM_PLAYERS)
		game.starting_player = random.randint(0, NUM_PLAYERS - 1)
		game.total_rounds = 1
		networks = [net] * NUM_PLAYERS
		game.start_round()
		# Flip decisions
		with torch.no_grad():
			for p in range(NUM_PLAYERS):
				ho = random.randint(0, H - 1)
				t_normal, t_flipped = encode_hand_both_orientations_v6(game, p, ho)
				h_normal = net(t_normal)
				h_flipped = net(t_flipped)
				if net.value(h_flipped).item() > net.value(h_normal).item():
					game.submit_flip_decision(p, do_flip=True)
				else:
					game.submit_flip_decision(p, do_flip=False)
			# Play turns, snapshot qualifying positions
			turn = 0
			while game.phase in (Phase.TURN, Phase.SNS_PLAY):
				p = game.current_player
				hand = game.players[p].hand
				legal_plays = get_legal_plays(hand, game.current_play)
				forced_play = game.phase == Phase.SNS_PLAY
				if turn >= MIN_TURNS_BEFORE_SNAPSHOT and not forced_play:
					# Check if both play and scout/S&S actions exist
					mask = get_flat_action_mask(game, p, legal_plays, 0)
					has_play = mask[:256].any()
					has_scout = mask[256:].any()
					if has_play and has_scout:
						candidates.append({
							'snapshot': game.clone(),
							'player': p,
							'game_num': game_num,
							'turn': turn,
						})
				# Play the turn normally
				records = _play_turn_v6(game, networks)
				turn += 1
				if time.time() >= deadline:
					break
		game_num += 1
	return candidates


# --- Phase 2: Screen states via rollouts ---

def find_best_action_by_type(net, game, player, legal_plays, action_type):
	"""Find the network's preferred action of a given type ('play' or 'scout').
	Returns (action_idx, hand_offset) or None if no actions of that type are legal."""
	ho = 0  # use fixed offset for comparison
	state = encode_state_v6(game, player, ho, forced_play=False)
	hidden = net(state)
	logits = net.policy_logits(hidden)
	mask = get_flat_action_mask(game, player, legal_plays, ho)
	# Mask to only the requested type
	type_mask = mask.clone()
	if action_type == 'play':
		type_mask[256:] = False
	else:  # scout/S&S
		type_mask[:256] = False
	if not type_mask.any():
		return None
	# Greedy selection among this type
	masked_logits = logits.clone()
	masked_logits[~type_mask] = float('-inf')
	action_idx = masked_logits.argmax().item()
	return action_idx, ho


def apply_action_to_game(game, player, action_idx, hand_offset):
	"""Apply a decoded flat action to a game. Returns the action dict."""
	action = decode_flat_action(action_idx, hand_offset)
	if action['type'] == 'play':
		game.apply_play(action['start'], action['end'])
	elif action['type'] == 'scout':
		game.apply_scout(action['left_end'], action['flip'], action['insert_pos'])
	elif action['type'] == 'sns':
		game.apply_sns_scout(action['left_end'], action['flip'], action['insert_pos'])
	return action


def compute_rollout_value(snapshots, player, net):
	"""Run rollouts from snapshots, return mean margin for the given player."""
	expanded = [s for s in snapshots for _ in range(ROLLOUTS_PER_ACTION)]
	all_scores = rollout_from_states_batched_v6(expanded, net)
	values = []
	for i in range(len(snapshots)):
		margins = []
		base = i * ROLLOUTS_PER_ACTION
		for r in range(ROLLOUTS_PER_ACTION):
			scores = all_scores[base + r]
			opp = [scores[j] for j in range(NUM_PLAYERS) if j != player]
			margin = (scores[player] - sum(opp) / len(opp)) / 10.0
			margins.append(margin)
		values.append(np.mean(margins))
	return values


def screen_states(candidates, net):
	"""For each candidate, find play vs scout actions, rollout both, keep divergent pairs."""
	test_states = []
	# Batch: collect all post-action snapshots first, then rollout together
	snapshot_tasks = []  # (candidate_idx, action_type, action_idx, hand_offset, post_snapshot)
	for i, cand in enumerate(candidates):
		game = cand['snapshot']
		player = cand['player']
		legal_plays = get_legal_plays(game.players[player].hand, game.current_play)
		play_result = find_best_action_by_type(net, game, player, legal_plays, 'play')
		scout_result = find_best_action_by_type(net, game, player, legal_plays, 'scout')
		if play_result is None or scout_result is None:
			continue
		# Create post-action snapshots
		for action_type, (action_idx, ho) in [('play', play_result), ('scout', scout_result)]:
			post = game.clone()
			apply_action_to_game(post, player, action_idx, ho)
			snapshot_tasks.append((i, action_type, action_idx, ho, post))
	if not snapshot_tasks:
		return []
	# Batch rollout all post-action snapshots
	all_snapshots = [t[4] for t in snapshot_tasks]
	all_players = [candidates[t[0]]['player'] for t in snapshot_tasks]
	print(f"  Running {len(all_snapshots) * ROLLOUTS_PER_ACTION} rollouts for {len(all_snapshots)} post-action states...")
	expanded = [s for s in all_snapshots for _ in range(ROLLOUTS_PER_ACTION)]
	all_scores = rollout_from_states_batched_v6(expanded, net)
	# Compute per-snapshot values
	snapshot_values = []
	for si in range(len(all_snapshots)):
		player = all_players[si]
		margins = []
		base = si * ROLLOUTS_PER_ACTION
		for r in range(ROLLOUTS_PER_ACTION):
			scores = all_scores[base + r]
			opp = [scores[j] for j in range(NUM_PLAYERS) if j != player]
			margin = (scores[player] - sum(opp) / len(opp)) / 10.0
			margins.append(margin)
		snapshot_values.append(np.mean(margins))
	# Pair up play/scout results per candidate
	task_by_cand = {}
	for ti, (ci, atype, aidx, ho, _) in enumerate(snapshot_tasks):
		task_by_cand.setdefault(ci, {})[atype] = (aidx, ho, snapshot_values[ti])
	for ci, actions in task_by_cand.items():
		if 'play' not in actions or 'scout' not in actions:
			continue
		play_idx, play_ho, play_val = actions['play']
		scout_idx, scout_ho, scout_val = actions['scout']
		gap = abs(play_val - scout_val)
		if gap < VALUE_GAP_THRESHOLD:
			continue
		if play_val >= scout_val:
			good = ('play', play_idx, play_ho, play_val)
			bad = ('scout', scout_idx, scout_ho, scout_val)
		else:
			good = ('scout', scout_idx, scout_ho, scout_val)
			bad = ('play', play_idx, play_ho, play_val)
		cand = candidates[ci]
		test_states.append({
			'snapshot': cand['snapshot'],
			'player': cand['player'],
			'good': good,  # (type, action_idx, hand_offset, rollout_value)
			'bad': bad,
			'value_gap': good[3] - bad[3],
		})
	return test_states


# --- Phase 3: Generate controlled episodes ---

def generate_episode(snapshot, player, forced_action_idx, forced_ho, net, game_id):
	"""Force decision 1, then play round to completion with network.
	Returns list of StepRecordV6 for the test player, or None if too short."""
	game = snapshot.clone()
	networks = [net] * NUM_PLAYERS
	records = []
	# Decision 1: forced
	with torch.no_grad():
		hand = game.players[player].hand
		legal_plays = get_legal_plays(hand, game.current_play)
		state = encode_state_v6(game, player, forced_ho, forced_play=False)
		hidden = net(state)
		value = net.value(hidden).item()
		logits = net.policy_logits(hidden)
		mask_t = get_flat_action_mask(game, player, legal_plays, forced_ho)
		old_lp = masked_log_prob(logits, mask_t, forced_action_idx).item()
		action = decode_flat_action(forced_action_idx, forced_ho)
		rec = StepRecordV6(
			state=state, action=forced_action_idx, mask=mask_t.numpy(),
			old_log_prob=old_lp, value=value, reward=0.0,
			player=player, round_num=0, game_id=game_id,
			hand_offset=forced_ho, play_length=None, scout_quality=None,
			predicted_value=value,
		)
		if action['type'] == 'play':
			rec.play_length = action['end'] - action['start'] + 1
		elif action['type'] in ('scout', 'sns'):
			# Compute scout quality (max run length after insertion)
			left_end, flip = action['left_end'], action['flip']
			insert_pos = action['insert_pos']
			play_cards = game.current_play.cards
			scouted = play_cards[0] if left_end else play_cards[-1]
			if flip:
				scouted = (scouted[1], scouted[0])
			new_hand = list(hand[:insert_pos]) + [scouted] + list(hand[insert_pos:])
			max_len = 1
			for s, e in get_legal_plays(new_hand, None):
				if s <= insert_pos <= e:
					max_len = max(max_len, e - s + 1)
			rec.scout_quality = max_len
		records.append(rec)
		apply_action_to_game(game, player, forced_action_idx, forced_ho)
	# Play remaining turns with network
	all_round_records = list(records)  # includes forced decision for reward assignment
	with torch.no_grad():
		while game.phase in (Phase.TURN, Phase.SNS_PLAY):
			turn_records = _play_turn_v6(game, networks)
			for tr in turn_records:
				tr.game_id = game_id
				tr.round_num = 0
				all_round_records.append(tr)
				if tr.player == player:
					records.append(tr)
	# Assign terminal reward
	_assign_round_rewards(all_round_records, game, round_idx=0,
		reward_mode="game_score", reward_distribution="terminal")
	if len(records) < MIN_EPISODE_DECISIONS:
		return None
	return records


def generate_all_episodes(test_states, net):
	"""Generate good and bad episodes for each test state."""
	episodes = []  # (state_idx, 'good'/'bad', records)
	game_id = 0
	for si, ts in enumerate(test_states):
		for label, (atype, aidx, aho, rval) in [('good', ts['good']), ('bad', ts['bad'])]:
			recs = generate_episode(ts['snapshot'], ts['player'], aidx, aho, net, game_id)
			if recs is not None:
				episodes.append((si, label, recs))
			game_id += 1
	return episodes


# --- Phase 4: Analysis ---

def analyze(test_states, episodes):
	"""Run GAE, compare decision-1 advantages against rollout ground truth."""
	# Flatten all records for GAE
	all_records = []
	episode_offsets = []  # (start_idx, length) for each episode
	for si, label, recs in episodes:
		start = len(all_records)
		all_records.extend(recs)
		episode_offsets.append((start, len(recs)))
	if not all_records:
		print("No valid episodes generated.")
		return
	advantages, returns = compute_gae(all_records)
	# Extract decision-1 advantages per episode
	results = []  # per test state
	ep_by_state = {}
	for ei, (si, label, recs) in enumerate(episodes):
		start, length = episode_offsets[ei]
		d1_advantage = advantages[start]
		d1_value = recs[0].value
		terminal_reward = recs[-1].reward
		ep_by_state.setdefault(si, {})[label] = {
			'gae_adv': d1_advantage,
			'd1_value': d1_value,
			'terminal_reward': terminal_reward,
			'num_decisions': length,
		}
	# Compare
	print(f"\n{'='*80}")
	print(f"BRIDGING PROBE RESULTS")
	print(f"{'='*80}")
	correct = 0
	total = 0
	all_gaps_gae = []
	all_gaps_rollout = []
	all_lengths = []
	for si, ts in enumerate(test_states):
		if si not in ep_by_state:
			continue
		eps = ep_by_state[si]
		if 'good' not in eps or 'bad' not in eps:
			continue
		good_ep = eps['good']
		bad_ep = eps['bad']
		gae_gap = good_ep['gae_adv'] - bad_ep['gae_adv']
		rollout_gap = ts['value_gap']  # always positive by construction
		gae_correct = gae_gap > 0
		total += 1
		if gae_correct:
			correct += 1
		avg_len = (good_ep['num_decisions'] + bad_ep['num_decisions']) / 2
		all_gaps_gae.append(gae_gap)
		all_gaps_rollout.append(rollout_gap)
		all_lengths.append(avg_len)
		marker = "OK" if gae_correct else "WRONG"
		good_type = ts['good'][0]
		bad_type = ts['bad'][0]
		print(f"\nState {si}: good={good_type} bad={bad_type}  rollout_gap={rollout_gap:+.3f}  [{marker}]")
		print(f"  Good: GAE_adv={good_ep['gae_adv']:+.4f}  V={good_ep['d1_value']:+.3f}  "
			f"R_term={good_ep['terminal_reward']:+.3f}  steps={good_ep['num_decisions']}")
		print(f"  Bad:  GAE_adv={bad_ep['gae_adv']:+.4f}  V={bad_ep['d1_value']:+.3f}  "
			f"R_term={bad_ep['terminal_reward']:+.3f}  steps={bad_ep['num_decisions']}")
		print(f"  GAE gap={gae_gap:+.4f}  avg_episode_len={avg_len:.1f}")
	# Summary
	print(f"\n{'='*80}")
	print(f"SUMMARY")
	print(f"{'='*80}")
	if total == 0:
		print("No complete test state pairs.")
		return
	accuracy = correct / total
	print(f"GAE sign accuracy:    {correct}/{total} = {accuracy:.1%}")
	gaps_gae = np.array(all_gaps_gae)
	gaps_rollout = np.array(all_gaps_rollout)
	lengths = np.array(all_lengths)
	print(f"GAE gap:              mean={gaps_gae.mean():+.4f}  std={gaps_gae.std():.4f}")
	print(f"Rollout gap:          mean={gaps_rollout.mean():+.4f}  std={gaps_rollout.std():.4f}")
	if len(gaps_gae) > 2:
		corr = np.corrcoef(gaps_gae, gaps_rollout)[0, 1]
		print(f"GAE-rollout corr:     {corr:.4f}")
	print(f"Episode length:       mean={lengths.mean():.1f}  min={lengths.min():.0f}  max={lengths.max():.0f}")
	# Accuracy by episode length
	if len(all_lengths) >= 10:
		median_len = np.median(lengths)
		short_correct = sum(1 for g, l in zip(all_gaps_gae, all_lengths) if g > 0 and l <= median_len)
		short_total = sum(1 for l in all_lengths if l <= median_len)
		long_correct = sum(1 for g, l in zip(all_gaps_gae, all_lengths) if g > 0 and l > median_len)
		long_total = sum(1 for l in all_lengths if l > median_len)
		if short_total > 0 and long_total > 0:
			print(f"  Short episodes (len<={median_len:.0f}): {short_correct}/{short_total} = {short_correct/short_total:.1%}")
			print(f"  Long episodes  (len>{median_len:.0f}):  {long_correct}/{long_total} = {long_correct/long_total:.1%}")


def main():
	ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "v7_3/latest.pt"
	random.seed(42)
	torch.manual_seed(42)
	np.random.seed(42)
	print(f"Loading {ckpt_path}...")
	checkpoint = torch.load(ckpt_path, weights_only=False)
	cfg = checkpoint.get("config", {})
	ls = cfg.get("layer_sizes", [512, 256, 128])
	net = FlatScoutNetwork(INPUT_SIZE_V6, ls, encoding_version=6,
		attention=cfg.get("attention"))
	net.load_state_dict(checkpoint["model_state"])
	net.eval()
	iteration = checkpoint.get("iteration", "?")
	print(f"Loaded (iteration {iteration})")

	# Phase 1: Generate candidate states
	print(f"\nPhase 1: Generating candidate states ({STATE_GEN_TIME}s)...")
	candidates = generate_candidate_states(net, STATE_GEN_TIME)
	print(f"  Found {len(candidates)} candidate states")
	if not candidates:
		print("No candidate states found. Exiting.")
		return

	# Phase 2: Screen via rollouts
	if len(candidates) > MAX_SCREEN_STATES:
		random.shuffle(candidates)
		candidates = candidates[:MAX_SCREEN_STATES]
	print(f"\nPhase 2: Screening {len(candidates)} states (rollouts, gap threshold={VALUE_GAP_THRESHOLD})...")
	test_states = screen_states(candidates, net)
	print(f"  {len(test_states)} states passed screening (of {len(candidates)} candidates)")
	if not test_states:
		print("No states with sufficient play/scout value gap. Try lowering VALUE_GAP_THRESHOLD.")
		return

	# Phase 3: Generate episodes
	print(f"\nPhase 3: Generating controlled episodes...")
	episodes = generate_all_episodes(test_states, net)
	good_count = sum(1 for _, l, _ in episodes if l == 'good')
	bad_count = sum(1 for _, l, _ in episodes if l == 'bad')
	total_decisions = sum(len(recs) for _, _, recs in episodes)
	print(f"  {len(episodes)} episodes ({good_count} good, {bad_count} bad), {total_decisions} total decisions")

	# Phase 4: Analyze
	print(f"\nPhase 4: Analyzing GAE credit assignment...")
	analyze(test_states, episodes)


if __name__ == "__main__":
	main()
