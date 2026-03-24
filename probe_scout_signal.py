"""Measure the actual rollout value signal from scout insert position choices.

Uses a trained checkpoint's play policy to run rollouts. For each scoutable state,
tries every possible insert position and measures whether position choice materially
affects game outcomes.

Usage: python scout-bot/probe_scout_signal.py [--checkpoint PATH] [--states N] [--rollouts N]
"""
import sys
import os
import copy
import random
import numpy as np
import torch

from game import Game, Phase
from encoding import (
	get_legal_plays,
	INPUT_SIZE, HAND_SLOTS, PLAY_SLOTS, SCOUT_INSERT_SIZE,
	PLAY_START_SIZE, PLAY_END_SIZE,
	INPUT_SIZE_V2, HAND_SLOTS_V2, SCOUT_INSERT_SIZE_V2,
	PLAY_START_SIZE_V2, PLAY_END_SIZE_V2,
)
from network import ScoutNetwork
from training import rollout_from_states_batched

NUM_PLAYERS = 4


def load_checkpoint(path):
	checkpoint = torch.load(path, weights_only=False)
	cfg = checkpoint.get("config", {})
	layer_sizes = cfg.get("layer_sizes", [512, 256, 256, 128, 128, 128])
	ev = cfg.get("encoding_version", 1)
	if ev == 2:
		network = ScoutNetwork(INPUT_SIZE_V2, layer_sizes,
			play_start_size=PLAY_START_SIZE_V2, play_end_size=PLAY_END_SIZE_V2,
			scout_insert_size=SCOUT_INSERT_SIZE_V2, encoding_version=2)
	elif ev == 1:
		network = ScoutNetwork(INPUT_SIZE, layer_sizes)
	else:
		raise ValueError(f"Unsupported encoding version {ev}")
	network.load_state_dict(checkpoint["model_state"])
	network.eval()
	return network, ev


def mid_round_state():
	"""Create a state where the current player can scout."""
	for _ in range(100):
		game = Game(NUM_PLAYERS)
		game.start_round()
		for p in range(NUM_PLAYERS):
			game.submit_flip_decision(p, do_flip=random.random() < 0.5)
		# Player 0 plays to establish current_play
		hand = game.players[0].hand
		legal_plays = get_legal_plays(hand, game.current_play)
		if legal_plays:
			start, end = random.choice(legal_plays)
			game.apply_play(start, end)
			if game.current_play and len(game.current_play.cards) > 0:
				return game
	return None


def measure_scout_signal(network, num_states=100, rollouts_per_pos=20):
	"""For each scoutable state, try all insert positions and measure
	the rollout value difference between best and worst positions."""
	deltas = []  # best - worst per state
	best_vs_random = []  # best - mean per state
	adj_vs_nonadj = []  # mean(adjacent match positions) - mean(non-adjacent) per state
	states_with_adj = 0

	for state_idx in range(num_states):
		game = mid_round_state()
		if game is None:
			continue

		player = game.current_player
		hand = game.players[player].hand
		card = game.current_play.cards[0]  # left end card
		card_val = card[0]  # showing value
		num_positions = len(hand) + 1

		# Which positions place card adjacent to a matching value?
		adj_positions = set()
		for pos in range(num_positions):
			if pos > 0 and hand[pos - 1][0] == card_val:
				adj_positions.add(pos)
			if pos < len(hand) and hand[pos][0] == card_val:
				adj_positions.add(pos)

		# Create a game snapshot for each possible insert position
		position_games = []
		for pos in range(num_positions):
			g_copy = copy.deepcopy(game)
			g_copy.apply_scout(True, False, pos)  # left_end, no flip
			position_games.append(g_copy)

		# Expand for rollouts and run batched
		expanded = [g for g in position_games for _ in range(rollouts_per_pos)]
		all_scores = rollout_from_states_batched(expanded, network)

		# Aggregate per position
		position_values = []
		for pos_idx in range(num_positions):
			base = pos_idx * rollouts_per_pos
			margins = []
			for r in range(rollouts_per_pos):
				scores = all_scores[base + r]
				opp_scores = [scores[j] for j in range(NUM_PLAYERS) if j != player]
				margin = (scores[player] - sum(opp_scores) / len(opp_scores)) / 10.0
				margins.append(margin)
			position_values.append(sum(margins) / len(margins))

		best_val = max(position_values)
		worst_val = min(position_values)
		mean_val = sum(position_values) / len(position_values)

		deltas.append(best_val - worst_val)
		best_vs_random.append(best_val - mean_val)

		# Adjacent vs non-adjacent comparison
		if adj_positions and len(adj_positions) < num_positions:
			states_with_adj += 1
			adj_vals = [position_values[p] for p in range(num_positions) if p in adj_positions]
			non_adj_vals = [position_values[p] for p in range(num_positions) if p not in adj_positions]
			adj_mean = sum(adj_vals) / len(adj_vals)
			non_adj_mean = sum(non_adj_vals) / len(non_adj_vals)
			adj_vs_nonadj.append(adj_mean - non_adj_mean)

		if (state_idx + 1) % 25 == 0:
			print(f"  {state_idx + 1}/{num_states} states | "
				  f"avg best-worst={sum(deltas)/len(deltas):.4f}  "
				  f"avg best-rand={sum(best_vs_random)/len(best_vs_random):.4f}"
				  + (f"  avg adj-nonadj={sum(adj_vs_nonadj)/len(adj_vs_nonadj):.4f}" if adj_vs_nonadj else ""))

	return deltas, best_vs_random, adj_vs_nonadj, states_with_adj


def main():
	import argparse
	parser = argparse.ArgumentParser(description="Measure scout insert position signal strength")
	parser.add_argument("--checkpoint", default="v4_2/latest.pt")
	parser.add_argument("--states", type=int, default=100)
	parser.add_argument("--rollouts", type=int, default=20, help="Rollouts per insert position")
	args = parser.parse_args()

	print(f"Checkpoint: {args.checkpoint}")
	network, ev = load_checkpoint(args.checkpoint)
	print(f"  encoding=v{ev}, measuring {args.states} states x {args.rollouts} rollouts/pos\n")

	deltas, bvr, adj_nonadj, n_adj_states = measure_scout_signal(
		network, args.states, args.rollouts)

	se = 0.55 / args.rollouts ** 0.5  # approx standard error per position estimate

	print(f"\n{'='*60}")
	print(f"Results ({len(deltas)} states, {args.rollouts} rollouts/pos)")
	print(f"{'='*60}")
	print(f"\nBest - Worst position value:")
	print(f"  mean={np.mean(deltas):.4f}  median={np.median(deltas):.4f}  std={np.std(deltas):.4f}")
	print(f"\nBest - Random (mean) position value:")
	print(f"  mean={np.mean(bvr):.4f}  median={np.median(bvr):.4f}  std={np.std(bvr):.4f}")
	if adj_nonadj:
		print(f"\nAdjacent-match vs non-adjacent positions ({n_adj_states} states with matches):")
		print(f"  mean={np.mean(adj_nonadj):.4f}  median={np.median(adj_nonadj):.4f}  std={np.std(adj_nonadj):.4f}")
		t_stat = np.mean(adj_nonadj) / (np.std(adj_nonadj) / len(adj_nonadj)**0.5) if len(adj_nonadj) > 1 else 0
		print(f"  t-stat={t_stat:.2f} (>2 = significant)")

	print(f"\nNoise floor: std of each position estimate ≈ {se:.3f}")
	print(f"  Best-worst includes max/min selection bias from ~14 positions")
	print(f"  Signal needs to be >> {se:.3f} to be learnable with 5 rollouts/state in training")

	# What fraction of states have meaningful signal?
	for thresh in [0.02, 0.05, 0.10, 0.20]:
		frac = sum(1 for d in deltas if d > thresh) / len(deltas)
		print(f"  States with best-worst > {thresh}: {frac*100:.0f}%")


if __name__ == "__main__":
	main()
