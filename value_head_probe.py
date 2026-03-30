"""Value head probe: compare predicted V(s) to empirical V(s) from rollouts.
Plays games from a checkpoint, takes snapshots at each decision, runs rollouts,
and plots timeline comparison."""

import sys
import os
import torch
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from network import FlatScoutNetwork, masked_sample
from encoding import (INPUT_SIZE_V6, HAND_SLOTS_V6, encode_state_v6,
	get_flat_action_mask, encode_hand_both_orientations_v6, decode_flat_action,
	get_legal_plays)
from training import rollout_from_states_batched_v6
from game import Game, Phase

NUM_GAMES = 5
NUM_PLAYERS = 4
ROLLOUTS = 40
PROBE_OFFSETS = [0, 4, 8, 12]
PLAYER_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def load_checkpoint(path):
	checkpoint = torch.load(path, weights_only=False)
	cfg = checkpoint.get("config", {})
	ls = cfg.get("layer_sizes", [512, 256, 128])
	net = FlatScoutNetwork(INPUT_SIZE_V6, ls, encoding_version=6,
		attention=cfg.get("attention"))
	net.load_state_dict(checkpoint["model_state"])
	net.eval()
	iteration = checkpoint.get("iteration", "?")
	return net, iteration

def play_game_with_snapshots(net, game_idx):
	"""Play one game, collecting snapshots and value predictions at each decision."""
	H = HAND_SLOTS_V6
	game = Game(NUM_PLAYERS)
	game.starting_player = random.randint(0, NUM_PLAYERS - 1)
	game.total_rounds = 1
	decisions = []
	game.start_round()
	# Flip decisions
	with torch.no_grad():
		for p in range(NUM_PLAYERS):
			ho = random.randint(0, H - 1)
			t_normal, t_flipped = encode_hand_both_orientations_v6(game, p, ho)
			h_normal = net(t_normal)
			h_flipped = net(t_flipped)
			v_normal = net.value(h_normal).item()
			v_flipped = net.value(h_flipped).item()
			game.submit_flip_decision(p, do_flip=v_flipped > v_normal)
		# Play turns
		step = 0
		while game.phase in (Phase.TURN, Phase.SNS_PLAY):
			p = game.current_player
			hand = game.players[p].hand
			legal_plays = get_legal_plays(hand, game.current_play)
			forced_play = game.phase == Phase.SNS_PLAY
			# Value predictions at multiple offsets
			offset_values = []
			for ho in PROBE_OFFSETS:
				state = encode_state_v6(game, p, ho, forced_play=forced_play)
				hidden = net(state)
				offset_values.append(net.value(hidden).item())
			# Use offset 0 for action selection
			ho = 0
			state = encode_state_v6(game, p, ho, forced_play=forced_play)
			hidden = net(state)
			logits = net.policy_logits(hidden)
			mask_t = get_flat_action_mask(game, p, legal_plays, ho)
			if not mask_t.any():
				game._advance_turn()
				continue
			snapshot = game.clone()
			decisions.append({
				'game_id': game_idx,
				'step': step,
				'player': p,
				'offset_values': offset_values,
				'snapshot': snapshot,
			})
			# Execute action
			action_idx, _ = masked_sample(logits, mask_t)
			action = decode_flat_action(action_idx, ho)
			if action['type'] == 'play':
				game.apply_play(action['start'], action['end'])
			elif action['type'] == 'scout':
				game.apply_scout(action['left_end'], action['flip'], action['insert_pos'])
			elif action['type'] == 'sns':
				game.apply_sns_scout(action['left_end'], action['flip'], action['insert_pos'])
			step += 1
	return decisions

def compute_empirical_values(decisions, net):
	"""Run rollouts from each decision's snapshot and compute empirical V."""
	snapshots = [d['snapshot'] for d in decisions]
	expanded = [s for s in snapshots for _ in range(ROLLOUTS)]
	print(f"Running {len(expanded)} rollouts...")
	all_scores = rollout_from_states_batched_v6(expanded, net)
	for i, d in enumerate(decisions):
		p = d['player']
		margins = []
		base = i * ROLLOUTS
		for r in range(ROLLOUTS):
			scores = all_scores[base + r]
			opp = [scores[j] for j in range(NUM_PLAYERS) if j != p]
			margin = (scores[p] - sum(opp) / len(opp)) / 10.0
			margins.append(margin)
		d['empirical_v'] = sum(margins) / len(margins)
		d['empirical_std'] = (sum((m - d['empirical_v'])**2 for m in margins) / len(margins)) ** 0.5

def plot_results(all_decisions, save_path, iteration):
	games = {}
	for d in all_decisions:
		games.setdefault(d['game_id'], []).append(d)
	num_games = len(games)
	fig, axes = plt.subplots(num_games, 1, figsize=(14, 4 * num_games), squeeze=False)
	fig.suptitle(f"Value Head Probe — v6_5 (iter {iteration})", fontsize=14, y=0.995)
	for row, (game_id, decs) in enumerate(sorted(games.items())):
		ax = axes[row, 0]
		decs.sort(key=lambda d: d['step'])
		steps = [d['step'] for d in decs]
		players_in_game = sorted(set(d['player'] for d in decs))
		# Plot per player
		for p in players_in_game:
			p_decs = [d for d in decs if d['player'] == p]
			p_steps = [d['step'] for d in p_decs]
			color = PLAYER_COLORS[p]
			# Empirical V (bold dashed)
			empirical = [d['empirical_v'] for d in p_decs]
			ax.plot(p_steps, empirical, 'o--', color=color, alpha=0.8,
				markersize=4, linewidth=1.5, label=f'P{p} empirical' if row == 0 else None)
			# Value head predictions at each offset (thin solid)
			for oi, ho in enumerate(PROBE_OFFSETS):
				preds = [d['offset_values'][oi] for d in p_decs]
				label = None
				if row == 0 and oi == 0:
					label = f'P{p} predicted (offsets 0,4,8,12)'
				ax.plot(p_steps, preds, '-', color=color, alpha=0.3,
					linewidth=0.8, label=label)
		ax.set_ylabel('Margin')
		ax.set_title(f'Game {game_id} ({len(decs)} decisions)')
		ax.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')
		ax.grid(True, alpha=0.3)
	axes[-1, 0].set_xlabel('Decision step')
	axes[0, 0].legend(loc='upper right', fontsize=7, ncol=2)
	plt.tight_layout()
	plt.savefig(save_path, dpi=150)
	print(f"Saved plot to {save_path}")

def print_stats(all_decisions):
	predicted = [d['offset_values'][0] for d in all_decisions]
	empirical = [d['empirical_v'] for d in all_decisions]
	pred_arr = np.array(predicted)
	emp_arr = np.array(empirical)
	corr = np.corrcoef(pred_arr, emp_arr)[0, 1]
	# Explained variance
	var_emp = emp_arr.var()
	ev = 1 - (emp_arr - pred_arr).var() / var_emp if var_emp > 1e-8 else 0.0
	# Rotation invariance: std across offsets
	offset_stds = [np.std(d['offset_values']) for d in all_decisions]
	mean_offset_std = np.mean(offset_stds)
	print(f"\n=== Statistics ===")
	print(f"Decisions:           {len(all_decisions)}")
	print(f"Correlation:         {corr:.4f}")
	print(f"Explained variance:  {ev:.4f}")
	print(f"Pred mean:           {pred_arr.mean():.4f}  std: {pred_arr.std():.4f}")
	print(f"Empirical mean:      {emp_arr.mean():.4f}  std: {emp_arr.std():.4f}")
	print(f"Empirical noise (avg rollout std): {np.mean([d['empirical_std'] for d in all_decisions]):.4f}")
	print(f"Rotation spread (avg offset std):  {mean_offset_std:.4f}")

def main():
	ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "v6_5/latest.pt"
	random.seed(42)
	torch.manual_seed(42)
	np.random.seed(42)
	print(f"Loading {ckpt_path}...")
	net, iteration = load_checkpoint(ckpt_path)
	print(f"Loaded (iteration {iteration})")
	all_decisions = []
	for g in range(NUM_GAMES):
		print(f"Playing game {g}...")
		decs = play_game_with_snapshots(net, g)
		all_decisions.extend(decs)
		print(f"  {len(decs)} decisions")
	compute_empirical_values(all_decisions, net)
	print_stats(all_decisions)
	save_path = os.path.splitext(ckpt_path)[0] + "_value_probe.png"
	plot_results(all_decisions, save_path, iteration)

if __name__ == "__main__":
	main()
