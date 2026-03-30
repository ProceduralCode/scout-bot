"""Supervised value head warmup test with train/test split.
Generates rollout-based value targets, freezes the trunk, trains only the value head,
and measures whether it can learn the correct magnitude. OLS provides a linear ceiling.
Train/test split validates whether OLS generalizes or is overfitting.
Runs multiple optimizer/init variants to diagnose why Adam can't reach the OLS ceiling."""

import sys
import os
import copy
import torch
import torch.nn as nn
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

NUM_GAMES = 40
NUM_PLAYERS = 4
ROLLOUTS = 40
TRAIN_EPOCHS = 500
TEST_FRACTION = 0.3

DARK_BG = '#1e1e2e'
DARK_FG = '#cdd6f4'
DARK_GRID = '#45475a'
COLORS = {
	'adam_ppo': '#89b4fa',
	'adam_fresh': '#fab387',
	'sgd_ppo': '#a6e3a1',
	'sgd_fresh': '#f38ba8',
	'ols_train': '#cba6f7',
	'ols_test': '#f5c2e7',
}

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

def collect_data(net):
	"""Play games and collect (state, empirical_V) via rollouts."""
	H = HAND_SLOTS_V6
	entries = []  # (state_tensor, player, snapshot)
	for game_idx in range(NUM_GAMES):
		game = Game(NUM_PLAYERS)
		game.starting_player = random.randint(0, NUM_PLAYERS - 1)
		game.total_rounds = 1
		game.start_round()
		with torch.no_grad():
			# Flip decisions
			for p in range(NUM_PLAYERS):
				ho = random.randint(0, H - 1)
				t_normal, t_flipped = encode_hand_both_orientations_v6(game, p, ho)
				h_normal = net(t_normal)
				h_flipped = net(t_flipped)
				v_normal = net.value(h_normal).item()
				v_flipped = net.value(h_flipped).item()
				game.submit_flip_decision(p, do_flip=v_flipped > v_normal)
			# Play turns
			while game.phase in (Phase.TURN, Phase.SNS_PLAY):
				p = game.current_player
				hand = game.players[p].hand
				legal_plays = get_legal_plays(hand, game.current_play)
				forced_play = game.phase == Phase.SNS_PLAY
				state = encode_state_v6(game, p, 0, forced_play=forced_play)
				hidden = net(state)
				logits = net.policy_logits(hidden)
				mask_t = get_flat_action_mask(game, p, legal_plays, 0)
				if not mask_t.any():
					game._advance_turn()
					continue
				snapshot = game.clone()
				entries.append((state, p, snapshot))
				action_idx, _ = masked_sample(logits, mask_t)
				action = decode_flat_action(action_idx, 0)
				if action['type'] == 'play':
					game.apply_play(action['start'], action['end'])
				elif action['type'] == 'scout':
					game.apply_scout(action['left_end'], action['flip'], action['insert_pos'])
				elif action['type'] == 'sns':
					game.apply_sns_scout(action['left_end'], action['flip'], action['insert_pos'])
		if (game_idx + 1) % 10 == 0:
			print(f"  Played {game_idx + 1}/{NUM_GAMES} games ({len(entries)} decisions)")
	# Run rollouts
	snapshots = [e[2] for e in entries]
	expanded = [s for s in snapshots for _ in range(ROLLOUTS)]
	print(f"Running {len(expanded)} rollouts...")
	all_scores = rollout_from_states_batched_v6(expanded, net)
	# Compute empirical V
	states = []
	targets = []
	for i, (state, player, _) in enumerate(entries):
		margins = []
		base = i * ROLLOUTS
		for r in range(ROLLOUTS):
			scores = all_scores[base + r]
			opp = [scores[j] for j in range(NUM_PLAYERS) if j != player]
			margin = (scores[player] - sum(opp) / len(opp)) / 10.0
			margins.append(margin)
		states.append(state)
		targets.append(sum(margins) / len(margins))
	return torch.stack(states), torch.tensor(targets, dtype=torch.float32)

def compute_ev(preds, targets):
	"""Explained variance: 1 - Var(residual) / Var(targets)."""
	var_t = targets.var()
	if var_t < 1e-8:
		return 0.0
	return (1 - (targets - preds).var() / var_t).item()

def train_linear(train_hiddens, train_targets, test_hiddens, test_targets,
				 init_weights=None, optimizer_type='adam', lr=0.001, epochs=200,
				 label=''):
	"""Train a linear head on pre-computed hiddens. Returns history dict."""
	hidden_dim = train_hiddens.shape[1]
	head = nn.Linear(hidden_dim, 1)

	if init_weights is not None:
		head.load_state_dict(copy.deepcopy(init_weights))
	else:
		# Kaiming init (PyTorch default for Linear)
		nn.init.kaiming_uniform_(head.weight)
		nn.init.zeros_(head.bias)

	if optimizer_type == 'adam':
		optimizer = torch.optim.Adam(head.parameters(), lr=lr)
	elif optimizer_type == 'sgd':
		optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9)

	history = {'epoch': [], 'train_ev': [], 'test_ev': []}

	for epoch in range(epochs + 1):
		with torch.no_grad():
			tr_preds = head(train_hiddens).squeeze(-1)
			te_preds = head(test_hiddens).squeeze(-1)
			tr_ev = compute_ev(tr_preds, train_targets)
			te_ev = compute_ev(te_preds, test_targets)
			history['epoch'].append(epoch)
			history['train_ev'].append(tr_ev)
			history['test_ev'].append(te_ev)

		if epoch % 50 == 0:
			tr_loss = torch.nn.functional.mse_loss(tr_preds, train_targets).item()
			print(f"  [{label}] Epoch {epoch:>4}: train_EV={tr_ev:.4f}  test_EV={te_ev:.4f}  "
				f"loss={tr_loss:.4f}  pred_std={tr_preds.std():.4f}")

		if epoch < epochs:
			head.train()
			pred = head(train_hiddens).squeeze(-1)
			loss = torch.nn.functional.mse_loss(pred, train_targets)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			head.eval()

	with torch.no_grad():
		final_test_preds = head(test_hiddens).squeeze(-1)

	print(f"  [{label}] Final: train_EV={tr_ev:.4f}  test_EV={te_ev:.4f}  "
		f"pred_std={final_test_preds.std():.4f}  target_std={test_targets.std():.4f}")

	return history, final_test_preds

def compute_ols(train_hiddens, train_targets, test_hiddens, test_targets):
	"""OLS closed-form on train, evaluate on both."""
	H_tr = train_hiddens.detach().numpy()
	T_tr = train_targets.numpy()
	H_tr_bias = np.column_stack([H_tr, np.ones(len(H_tr))])
	w_ols, _, _, _ = np.linalg.lstsq(H_tr_bias, T_tr, rcond=None)

	ols_train_preds = H_tr_bias @ w_ols
	ols_train_ev = 1 - np.var(T_tr - ols_train_preds) / T_tr.var() if T_tr.var() > 1e-8 else 0.0

	H_te = test_hiddens.detach().numpy()
	T_te = test_targets.numpy()
	H_te_bias = np.column_stack([H_te, np.ones(len(H_te))])
	ols_test_preds = H_te_bias @ w_ols
	ols_test_ev = 1 - np.var(T_te - ols_test_preds) / T_te.var() if T_te.var() > 1e-8 else 0.0

	n_params = H_tr.shape[1] + 1
	print(f"\n  OLS ({len(H_tr)} train samples, {n_params} params, "
		f"{len(H_tr)/n_params:.1f} samples/param):")
	print(f"    Train: EV={ols_train_ev:.4f}  pred_std={ols_train_preds.std():.4f}")
	print(f"    Test:  EV={ols_test_ev:.4f}  pred_std={ols_test_preds.std():.4f}")

	return ols_train_ev, ols_test_ev, torch.tensor(ols_test_preds, dtype=torch.float32)

def plot_results(variants, ols_train_ev, ols_test_ev, ols_test_preds,
				 test_targets, save_path, iteration, n_train, n_test):
	"""Plot EV curves for all variants + OLS scatter."""
	n_variants = len(variants)
	# Layout: top row = EV curves (wide) + OLS scatter, bottom row = test scatters for each variant
	n_cols = max(n_variants, 2)
	fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9), facecolor=DARK_BG)
	fig.suptitle(f'Value Head Warmup — v6_5 (iter {iteration}) — '
		f'{n_train} train / {n_test} test',
		color=DARK_FG, fontsize=14)

	for ax in axes.flat:
		ax.set_facecolor(DARK_BG)
		ax.tick_params(colors=DARK_FG)
		ax.xaxis.label.set_color(DARK_FG)
		ax.yaxis.label.set_color(DARK_FG)
		ax.title.set_color(DARK_FG)
		for spine in ax.spines.values():
			spine.set_color(DARK_GRID)

	t_np = test_targets.numpy()

	# Collect all predictions for shared axis limits
	all_preds = [ols_test_preds.numpy()]
	for name, hist, preds in variants:
		all_preds.append(preds.numpy())
	all_preds = np.concatenate(all_preds)
	lim = max(abs(t_np.min()), abs(t_np.max()), abs(all_preds.min()), abs(all_preds.max())) * 1.1

	# Top left: EV curves for all variants
	ax = axes[0, 0]
	for name, hist, preds in variants:
		color = COLORS.get(name, '#cdd6f4')
		ax.plot(hist['epoch'], hist['test_ev'], color=color, linewidth=2, label=name)
		# Show train as thin dotted
		ax.plot(hist['epoch'], hist['train_ev'], color=color, linewidth=1,
			linestyle=':', alpha=0.5)
	ax.axhline(y=ols_train_ev, color=COLORS['ols_train'], linewidth=1,
		linestyle='--', alpha=0.7, label=f'OLS train ({ols_train_ev:.3f})')
	ax.axhline(y=ols_test_ev, color=COLORS['ols_test'], linewidth=1,
		linestyle='--', alpha=0.7, label=f'OLS test ({ols_test_ev:.3f})')
	ax.set_xlabel('Epoch')
	ax.set_ylabel('Explained Variance (test=solid, train=dotted)')
	ax.set_title('EV During Training')
	ax.legend(facecolor=DARK_BG, edgecolor=DARK_GRID, labelcolor=DARK_FG, fontsize=7)
	ax.grid(True, color=DARK_GRID, alpha=0.5)

	# Top right: OLS test scatter
	ax = axes[0, 1]
	ax.scatter(t_np, ols_test_preds.numpy(), alpha=0.4, s=10, color=COLORS['ols_test'])
	ax.plot([-lim, lim], [-lim, lim], '--', color=DARK_FG, alpha=0.3, linewidth=1)
	ax.set_xlim(-lim, lim)
	ax.set_ylim(-lim, lim)
	ax.set_xlabel('Empirical V (rollouts)')
	ax.set_ylabel('Predicted V')
	ax.set_title(f'OLS Test (EV={ols_test_ev:.3f})')
	ax.set_aspect('equal')
	ax.grid(True, color=DARK_GRID, alpha=0.5)

	# Hide extra top-row axes if n_cols > 2
	for i in range(2, n_cols):
		axes[0, i].set_visible(False)

	# Bottom row: test scatter for each variant
	for i, (name, hist, preds) in enumerate(variants):
		ax = axes[1, i]
		color = COLORS.get(name, '#cdd6f4')
		final_ev = hist['test_ev'][-1]
		ax.scatter(t_np, preds.numpy(), alpha=0.4, s=10, color=color)
		ax.plot([-lim, lim], [-lim, lim], '--', color=DARK_FG, alpha=0.3, linewidth=1)
		ax.set_xlim(-lim, lim)
		ax.set_ylim(-lim, lim)
		ax.set_xlabel('Empirical V (rollouts)')
		ax.set_ylabel('Predicted V')
		ax.set_title(f'{name} Test (EV={final_ev:.3f})')
		ax.set_aspect('equal')
		ax.grid(True, color=DARK_GRID, alpha=0.5)

	# Hide extra bottom-row axes
	for i in range(n_variants, n_cols):
		axes[1, i].set_visible(False)

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, facecolor=DARK_BG)
	print(f"Saved plot to {save_path}")

def main():
	ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "v6_5/latest.pt"
	random.seed(42)
	torch.manual_seed(42)
	np.random.seed(42)

	print(f"Loading {ckpt_path}...")
	net, iteration = load_checkpoint(ckpt_path)
	print(f"Loaded (iteration {iteration})")

	print(f"\nCollecting data ({NUM_GAMES} games, {ROLLOUTS} rollouts/state)...")
	states, targets = collect_data(net)
	n = len(states)
	print(f"Dataset: {n} states, target range [{targets.min():.3f}, {targets.max():.3f}]")

	# Shuffle and split
	indices = torch.randperm(n)
	split = int(n * (1 - TEST_FRACTION))
	train_idx, test_idx = indices[:split], indices[split:]
	train_states, train_targets = states[train_idx], targets[train_idx]
	test_states, test_targets = states[test_idx], targets[test_idx]
	print(f"Split: {split} train, {n - split} test (test fraction={TEST_FRACTION})")

	# Pre-compute hiddens (trunk is frozen for all variants)
	net.eval()
	with torch.no_grad():
		train_hiddens = net(train_states)
		test_hiddens = net(test_states)

	# Save original value head weights
	ppo_weights = copy.deepcopy(net.value_head.state_dict())

	# OLS baseline
	print(f"\n--- OLS ---")
	ols_train_ev, ols_test_ev, ols_test_preds = compute_ols(
		train_hiddens, train_targets, test_hiddens, test_targets)

	# Variant 1: Adam with PPO-trained weights (baseline from last session)
	print(f"\n--- Adam + PPO weights (lr=0.001) ---")
	h1, p1 = train_linear(train_hiddens, train_targets, test_hiddens, test_targets,
		init_weights=ppo_weights, optimizer_type='adam', lr=0.001,
		epochs=TRAIN_EPOCHS, label='adam_ppo')

	# Variant 2: Adam with fresh random weights
	print(f"\n--- Adam + fresh weights (lr=0.001) ---")
	h2, p2 = train_linear(train_hiddens, train_targets, test_hiddens, test_targets,
		init_weights=None, optimizer_type='adam', lr=0.001,
		epochs=TRAIN_EPOCHS, label='adam_fresh')

	# Variant 3: SGD with PPO-trained weights
	print(f"\n--- SGD + PPO weights (lr=0.01, momentum=0.9) ---")
	h3, p3 = train_linear(train_hiddens, train_targets, test_hiddens, test_targets,
		init_weights=ppo_weights, optimizer_type='sgd', lr=0.01,
		epochs=TRAIN_EPOCHS, label='sgd_ppo')

	# Variant 4: SGD with fresh random weights
	print(f"\n--- SGD + fresh weights (lr=0.01, momentum=0.9) ---")
	h4, p4 = train_linear(train_hiddens, train_targets, test_hiddens, test_targets,
		init_weights=None, optimizer_type='sgd', lr=0.01,
		epochs=TRAIN_EPOCHS, label='sgd_fresh')

	variants = [
		('adam_ppo', h1, p1),
		('adam_fresh', h2, p2),
		('sgd_ppo', h3, p3),
		('sgd_fresh', h4, p4),
	]

	# Summary
	print(f"\n{'='*60}")
	print(f"{'Variant':<15} {'Train EV':>10} {'Test EV':>10}")
	print(f"{'-'*60}")
	print(f"{'OLS':<15} {ols_train_ev:>10.4f} {ols_test_ev:>10.4f}")
	for name, hist, preds in variants:
		print(f"{name:<15} {hist['train_ev'][-1]:>10.4f} {hist['test_ev'][-1]:>10.4f}")
	print(f"{'='*60}")

	save_path = os.path.splitext(ckpt_path)[0] + "_value_warmup.png"
	plot_results(variants, ols_train_ev, ols_test_ev, ols_test_preds,
		test_targets, save_path, iteration, split, n - split)

if __name__ == "__main__":
	main()
