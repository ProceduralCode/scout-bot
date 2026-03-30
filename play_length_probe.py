"""Play length advantage probe: does playing longer sets actually improve your position?

Collects mid-game states, forces plays of different lengths, rollouts before and after
to measure the empirical advantage (v_after - v_before) by play length.

No value head involved — pure rollout ground truth.

Usage: python -u play_length_probe.py [checkpoint_path]"""

import sys
import os
import time
import random
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from network import FlatScoutNetwork, masked_sample
from encoding import (INPUT_SIZE_V6, HAND_SLOTS_V6, encode_state_v6,
	get_flat_action_mask, encode_hand_both_orientations_v6, decode_flat_action,
	get_legal_plays)
from training import rollout_from_states_batched_v6
from game import Game, Phase

NUM_PLAYERS = 4
ROLLOUTS = 20
STATE_GEN_TIME = 3.0
MAX_STATES = 40

H = HAND_SLOTS_V6


def generate_states(net, time_limit=STATE_GEN_TIME):
	"""Play games, snapshot mid-round positions where plays of multiple lengths are legal."""
	candidates = []
	deadline = time.time() + time_limit
	while time.time() < deadline:
		game = Game(NUM_PLAYERS)
		game.starting_player = random.randint(0, NUM_PLAYERS - 1)
		game.total_rounds = 1
		game.start_round()
		with torch.no_grad():
			for p in range(NUM_PLAYERS):
				ho = random.randint(0, H - 1)
				t_normal, t_flipped = encode_hand_both_orientations_v6(game, p, ho)
				h_n = net(t_normal)
				h_f = net(t_flipped)
				game.submit_flip_decision(p, do_flip=net.value(h_f).item() > net.value(h_n).item())
			turn = 0
			networks = [net] * NUM_PLAYERS
			while game.phase in (Phase.TURN, Phase.SNS_PLAY):
				p = game.current_player
				hand = game.players[p].hand
				legal_plays = get_legal_plays(hand, game.current_play)
				forced_play = game.phase == Phase.SNS_PLAY
				if turn >= 3 and not forced_play and legal_plays:
					# Group legal plays by length
					lengths = {}
					for s, e in legal_plays:
						l = e - s + 1
						lengths.setdefault(l, []).append((s, e))
					if len(lengths) >= 2:
						candidates.append({
							'snapshot': game.clone(),
							'player': p,
							'plays_by_length': lengths,
						})
				# Play turn normally
				ho = random.randint(0, H - 1)
				state = encode_state_v6(game, p, ho, forced_play=forced_play)
				hidden = net(state)
				logits = net.policy_logits(hidden)
				mask = get_flat_action_mask(game, p, legal_plays, ho)
				if not mask.any():
					game._advance_turn()
				else:
					action_idx, _ = masked_sample(logits, mask)
					action = decode_flat_action(action_idx, ho)
					if action['type'] == 'play':
						game.apply_play(action['start'], action['end'])
					elif action['type'] == 'scout':
						game.apply_scout(action['left_end'], action['flip'], action['insert_pos'])
					elif action['type'] == 'sns':
						game.apply_sns_scout(action['left_end'], action['flip'], action['insert_pos'])
				turn += 1
				if time.time() >= deadline:
					break
	return candidates


def compute_margin(scores, player):
	opp = [scores[j] for j in range(NUM_PLAYERS) if j != player]
	return (scores[player] - sum(opp) / len(opp)) / 10.0


def main():
	ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "bots/v7_3/latest.pt"
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

	# Generate states
	print(f"\nGenerating states ({STATE_GEN_TIME}s)...")
	candidates = generate_states(net, STATE_GEN_TIME)
	if len(candidates) > MAX_STATES:
		random.shuffle(candidates)
		candidates = candidates[:MAX_STATES]
	print(f"  {len(candidates)} states with multiple play lengths")

	# For each state, pick one play per available length, rollout before and after
	print(f"\nRunning rollouts...")
	# Collect all snapshots: (before, after) pairs tagged with metadata
	tasks = []  # (cand_idx, play_length, 'before'/'after', snapshot)
	for ci, cand in enumerate(candidates):
		game = cand['snapshot']
		player = cand['player']
		# Before snapshot (same for all play lengths)
		tasks.append((ci, 0, 'before', game.clone()))
		# After snapshot for each play length
		for length, plays in cand['plays_by_length'].items():
			s, e = plays[0]  # pick first legal play of this length
			post = game.clone()
			post.apply_play(s, e)
			tasks.append((ci, length, 'after', post))

	# Batch rollout everything
	all_snapshots = [t[3] for t in tasks]
	expanded = [s for s in all_snapshots for _ in range(ROLLOUTS)]
	print(f"  {len(expanded)} total rollouts ({len(all_snapshots)} snapshots x {ROLLOUTS})")
	all_scores = rollout_from_states_batched_v6(expanded, net)

	# Compute margins per snapshot
	snapshot_values = []
	for si in range(len(all_snapshots)):
		player = candidates[tasks[si][0]]['player']
		margins = []
		base = si * ROLLOUTS
		for r in range(ROLLOUTS):
			margins.append(compute_margin(all_scores[base + r], player))
		snapshot_values.append((np.mean(margins), np.std(margins)))

	# Group results by candidate, then compute advantages per play length
	# advantage = v_after(play of length L) - v_before
	by_length = {}  # length -> list of advantages
	for si, (ci, length, role, _) in enumerate(tasks):
		if role == 'before':
			continue
		# Find the before snapshot for this candidate
		before_val = None
		for sj, (cj, _, rj, _) in enumerate(tasks):
			if cj == ci and rj == 'before':
				before_val = snapshot_values[sj][0]
				break
		after_val = snapshot_values[si][0]
		advantage = after_val - before_val
		by_length.setdefault(length, []).append(advantage)

	# Print results
	print(f"\n{'='*60}")
	print(f"PLAY LENGTH ADVANTAGE (rollout ground truth)")
	print(f"{'='*60}")
	print(f"{'Length':>8}  {'N':>5}  {'Mean Adv':>10}  {'Std':>8}  {'% Positive':>12}")
	print(f"{'-'*60}")
	for length in sorted(by_length.keys()):
		advs = np.array(by_length[length])
		pct_pos = np.mean(advs > 0) * 100
		print(f"{length:>8}  {len(advs):>5}  {advs.mean():>+10.4f}  {advs.std():>8.4f}  {pct_pos:>11.1f}%")

	# Overall: does longer = better?
	print(f"\n{'='*60}")
	print(f"SUMMARY")
	print(f"{'='*60}")
	all_lengths = []
	all_advs = []
	for length, advs in by_length.items():
		for a in advs:
			all_lengths.append(length)
			all_advs.append(a)
	all_lengths = np.array(all_lengths)
	all_advs = np.array(all_advs)
	if len(all_lengths) > 2:
		corr = np.corrcoef(all_lengths, all_advs)[0, 1]
		print(f"Length-advantage correlation: {corr:.4f}")
	print(f"Total samples: {len(all_advs)}")


if __name__ == "__main__":
	main()
