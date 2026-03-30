"""Trunk feature analysis: what does PC1 represent, and who's driving trunk learning?

Investigation 1 — PC1 semantic identification:
  Play games, extract interpretable features at each decision, run PCA on trunk
  activations, correlate PC1 with game features to identify what it encodes.

Investigation 2 — Gradient magnitude comparison:
  Compute policy loss, value loss, entropy loss separately, backward each through
  the shared trunk, compare gradient norms to see which head dominates.
"""

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from network import FlatScoutNetwork, masked_sample
from encoding import (
	INPUT_SIZE_V6, HAND_SLOTS_V6, HAND_DIM_V6, SCOUT_CARDS_DIM_V6,
	GLOBAL_START_V6, GLOBAL_DIM_V6, METADATA_DIM_V6, PLAY_BUFFER_DIM_V6,
	encode_state_v6, get_flat_action_mask, encode_hand_both_orientations_v6,
	decode_flat_action, get_legal_plays,
)
from game import Game, Phase

NUM_GAMES = 60
NUM_PLAYERS = 4
H = HAND_SLOTS_V6

DARK_BG = '#1e1e2e'
DARK_FG = '#cdd6f4'
DARK_GRID = '#45475a'


def load_checkpoint(path):
	checkpoint = torch.load(path, weights_only=False)
	cfg = checkpoint.get("config", {})
	ls = cfg.get("layer_sizes", [512, 256, 128])
	net = FlatScoutNetwork(INPUT_SIZE_V6, ls, encoding_version=6,
		attention=cfg.get("attention"))
	net.load_state_dict(checkpoint["model_state"])
	net.eval()
	iteration = checkpoint.get("iteration", "?")
	return net, iteration, cfg


def collect_data(net):
	"""Play games, collect (state_tensor, game_features_dict) at each decision."""
	entries = []
	for game_idx in range(NUM_GAMES):
		game = Game(NUM_PLAYERS)
		game.starting_player = random.randint(0, NUM_PLAYERS - 1)
		game.total_rounds = 1
		game.start_round()
		turn_number = 0
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

				# Extract interpretable features
				features = {
					'hand_size': len(hand),
					'collected': len(game.players[p].collected),
					'scout_tokens': game.players[p].scout_tokens,
					'sns_available': 1.0 if game.players[p].sns_available else 0.0,
					'has_current_play': 1.0 if game.current_play is not None else 0.0,
					'play_length': len(game.current_play.cards) if game.current_play else 0,
					'play_strength': game.current_play.strength if game.current_play else 0,
					'num_legal_actions': mask_t.sum().item(),
					'num_legal_plays': mask_t[:256].sum().item(),
					'num_legal_scouts': mask_t[256:320].sum().item(),
					'turn_number': turn_number,
					'forced_play': 1.0 if forced_play else 0.0,
					'scouts_since_play': game.scouts_since_play,
					# Opponent hand sizes (sum)
					'opp_hand_total': sum(len(game.players[(p + 1 + j) % NUM_PLAYERS].hand)
						for j in range(NUM_PLAYERS - 1)),
					'opp_collected_total': sum(len(game.players[(p + 1 + j) % NUM_PLAYERS].collected)
						for j in range(NUM_PLAYERS - 1)),
				}
				entries.append((state, features, mask_t, None))  # action filled below

				action_idx, _ = masked_sample(logits, mask_t)
				entries[-1] = (state, features, mask_t, action_idx)
				action = decode_flat_action(action_idx, 0)
				if action['type'] == 'play':
					game.apply_play(action['start'], action['end'])
				elif action['type'] == 'scout':
					game.apply_scout(action['left_end'], action['flip'], action['insert_pos'])
				elif action['type'] == 'sns':
					game.apply_sns_scout(action['left_end'], action['flip'], action['insert_pos'])
				turn_number += 1

		if (game_idx + 1) % 20 == 0:
			print(f"  Played {game_idx + 1}/{NUM_GAMES} games ({len(entries)} decisions)")

	return entries


def analyze_pc1_semantics(net, entries):
	"""PCA on trunk activations, correlate PC1 with interpretable features."""
	print(f"\n{'='*60}")
	print("INVESTIGATION 1: What does PC1 represent?")
	print(f"{'='*60}")

	states = torch.stack([e[0] for e in entries])
	feature_names = list(entries[0][1].keys())
	feature_matrix = np.array([[e[1][k] for k in feature_names] for e in entries])
	print(f"  ({len(entries)} decisions from {NUM_GAMES} games)")

	with torch.no_grad():
		fwd = net(states)
		hiddens = (fwd[0] if isinstance(fwd, tuple) else fwd).numpy()

	n, d = hiddens.shape
	mean = hiddens.mean(axis=0)
	centered = hiddens - mean

	# PCA via SVD
	U, S, Vt = np.linalg.svd(centered, full_matrices=False)
	total_var = (S ** 2).sum()
	pc1_var = S[0] ** 2 / total_var
	print(f"\nTrunk activations: {n} samples, {d} dims")
	print(f"PC1 explains {pc1_var*100:.2f}% of variance")
	print(f"Top 5 singular values: {S[:5].round(1)}")

	# Dead neurons
	dead = (hiddens == 0).all(axis=0).sum()
	print(f"Dead ReLU neurons: {dead}/{d} ({dead/d*100:.0f}%)")

	# Project onto PCs
	pc1_proj = centered @ Vt[0]
	pc2_proj = centered @ Vt[1]
	pc3_proj = centered @ Vt[2]

	# Correlate PC1 (and PC2, PC3) with each feature
	print(f"\n{'Feature':<25} {'PC1 corr':>10} {'PC2 corr':>10} {'PC3 corr':>10} {'mean':>8} {'std':>8}")
	print("-" * 75)
	correlations = {}
	for i, name in enumerate(feature_names):
		feat = feature_matrix[:, i]
		if feat.std() < 1e-8:
			correlations[name] = (0.0, 0.0, 0.0)
			print(f"{name:<25} {'(const)':>10} {'':>10} {'':>10} {feat.mean():>8.2f} {feat.std():>8.3f}")
			continue
		c1 = np.corrcoef(pc1_proj, feat)[0, 1]
		c2 = np.corrcoef(pc2_proj, feat)[0, 1]
		c3 = np.corrcoef(pc3_proj, feat)[0, 1]
		correlations[name] = (c1, c2, c3)
		# Highlight strong correlations
		marker = " ***" if abs(c1) > 0.5 else " **" if abs(c1) > 0.3 else ""
		print(f"{name:<25} {c1:>10.3f} {c2:>10.3f} {c3:>10.3f} {feat.mean():>8.2f} {feat.std():>8.3f}{marker}")

	# Also check: correlation of raw activation mean with PC1
	act_mean = hiddens.mean(axis=1)
	act_norm = np.linalg.norm(hiddens, axis=1)
	c_mean = np.corrcoef(pc1_proj, act_mean)[0, 1]
	c_norm = np.corrcoef(pc1_proj, act_norm)[0, 1]
	print(f"\n{'activation_mean':<25} {c_mean:>10.3f}")
	print(f"{'activation_norm':<25} {c_norm:>10.3f}")

	# What fraction of PC1 variance is explained by the best single feature?
	best_feat = max(correlations.items(), key=lambda x: abs(x[1][0]))
	print(f"\nBest single feature for PC1: {best_feat[0]} (r={best_feat[1][0]:.3f}, r²={best_feat[1][0]**2:.3f})")

	# Multiple regression: how much of PC1 is explained by all features?
	X = np.column_stack([feature_matrix, np.ones(n)])
	w, _, _, _ = np.linalg.lstsq(X, pc1_proj, rcond=None)
	pred = X @ w
	r2_all = 1 - np.var(pc1_proj - pred) / np.var(pc1_proj) if np.var(pc1_proj) > 1e-8 else 0
	print(f"All features combined explain {r2_all*100:.1f}% of PC1 variance (R²)")

	return hiddens, pc1_proj, feature_matrix, feature_names, S, Vt


def analyze_gradients(net, entries, cfg):
	"""Measure gradient magnitudes from policy vs value loss on shared trunk params."""
	print(f"\n{'='*60}")
	print("INVESTIGATION 2: Policy vs value gradient magnitudes")
	print(f"{'='*60}")

	# Use a random subset for gradient analysis
	n_samples = min(500, len(entries))
	indices = random.sample(range(len(entries)), n_samples)
	states = torch.stack([entries[i][0] for i in indices])
	masks = torch.stack([entries[i][2] for i in indices])
	actions = torch.tensor([entries[i][3] for i in indices], dtype=torch.long)

	net.train()  # need gradients
	hidden = net(states)

	# Value loss: MSE against zero (arbitrary target — we just need gradient magnitude)
	v_pred = net.value(hidden).squeeze(-1)
	v_target = torch.zeros_like(v_pred)
	value_loss = F.mse_loss(v_pred, v_target)

	# Policy loss: log prob of the actions actually taken, with real masks
	logits = net.policy_logits(hidden)
	masked_logits = logits.masked_fill(~masks, float('-inf'))
	log_probs = F.log_softmax(masked_logits, dim=-1)
	selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(-1)
	policy_loss = -selected_log_probs.mean()

	# Entropy (using masked log probs)
	probs = torch.softmax(masked_logits, dim=-1)
	# Zero out -inf log probs to avoid nan in probs * log_probs
	safe_log_probs = log_probs.masked_fill(~masks, 0.0)
	entropy = -(probs * safe_log_probs).sum(dim=-1).mean()
	entropy_loss = -entropy  # negative because we maximize entropy

	# Identify shared trunk parameters
	trunk_params = list(net.shared.parameters())
	policy_head_params = list(net.policy_head.parameters())
	value_head_params = list(net.value_head.parameters())

	# Measure gradient norms from each loss component
	results = {}
	for loss_name, loss_val, coeff in [
		("policy_loss", policy_loss, 1.0),
		("value_loss", value_loss, 0.25),  # value_loss_coeff from PARAMS
		("entropy_loss", entropy_loss, 0.25),  # entropy_bonus from PARAMS
	]:
		net.zero_grad()
		scaled_loss = coeff * loss_val
		scaled_loss.backward(retain_graph=True)

		# Gradient norms on trunk
		trunk_grad_norms = []
		for p in trunk_params:
			if p.grad is not None:
				trunk_grad_norms.append(p.grad.norm().item())
		trunk_total = sum(g**2 for g in trunk_grad_norms) ** 0.5

		# Per-layer breakdown
		layer_grads = {}
		for name, param in net.shared.named_parameters():
			if param.grad is not None:
				layer_grads[name] = param.grad.norm().item()

		results[loss_name] = {
			'loss_value': loss_val.item(),
			'coeff': coeff,
			'scaled_loss': scaled_loss.item(),
			'trunk_grad_norm': trunk_total,
			'layer_grads': layer_grads,
		}
		print(f"\n{loss_name} (coeff={coeff}):")
		print(f"  Loss value: {loss_val.item():.4f}, scaled: {scaled_loss.item():.4f}")
		print(f"  Trunk gradient norm (total): {trunk_total:.6f}")
		for lname, gnorm in sorted(layer_grads.items()):
			print(f"    {lname}: {gnorm:.6f}")

	# Summary comparison
	print(f"\n--- Gradient magnitude ratio ---")
	p_norm = results['policy_loss']['trunk_grad_norm']
	v_norm = results['value_loss']['trunk_grad_norm']
	e_norm = results['entropy_loss']['trunk_grad_norm']
	print(f"Policy / Value trunk gradient ratio: {p_norm / v_norm:.1f}x" if v_norm > 1e-10 else "Value grad ~0")
	print(f"Entropy / Value trunk gradient ratio: {e_norm / v_norm:.1f}x" if v_norm > 1e-10 else "Value grad ~0")
	print(f"(Policy + Entropy) / Value: {(p_norm + e_norm) / v_norm:.1f}x" if v_norm > 1e-10 else "Value grad ~0")

	net.eval()
	return results


def analyze_input_pc1(entries):
	"""Check: does the input encoding itself have a dominant direction?"""
	print(f"\n{'='*60}")
	print("BONUS: Input encoding PCA")
	print(f"{'='*60}")

	states = torch.stack([e[0] for e in entries]).numpy()
	n, d = states.shape
	centered = states - states.mean(axis=0)
	U, S, Vt = np.linalg.svd(centered, full_matrices=False)
	total_var = (S ** 2).sum()

	print(f"Input: {n} samples, {d} dims")
	print(f"Top 5 singular values: {S[:5].round(2)}")
	cum_var = np.cumsum(S**2) / total_var
	for k in [1, 3, 5, 10, 20]:
		print(f"  PC1..{k}: {cum_var[k-1]*100:.1f}% variance")

	# What's the dominant input direction? Check which encoding regions have highest loading
	pc1_loadings = Vt[0]
	regions = [
		("hand_top (one-hot+scalar)", 0, HAND_DIM_V6 - H),
		("hand_bottom (scalars)", HAND_DIM_V6 - H, HAND_DIM_V6),
		("scout_cards", HAND_DIM_V6, HAND_DIM_V6 + SCOUT_CARDS_DIM_V6),
		("play_buffer", GLOBAL_START_V6, GLOBAL_START_V6 + PLAY_BUFFER_DIM_V6),
		("metadata", GLOBAL_START_V6 + PLAY_BUFFER_DIM_V6, INPUT_SIZE_V6),
	]
	print(f"\nPC1 loading magnitude by encoding region:")
	for name, start, end in regions:
		loading_norm = np.linalg.norm(pc1_loadings[start:end])
		loading_frac = loading_norm**2 / (np.linalg.norm(pc1_loadings)**2)
		print(f"  {name:<30} norm={loading_norm:.3f} ({loading_frac*100:.1f}%)")


def plot_results(hiddens, pc1_proj, feature_matrix, feature_names, S, Vt,
				 grad_results, save_path, iteration):
	"""Visualization of both investigations."""
	fig = plt.figure(figsize=(18, 14), facecolor=DARK_BG)

	def style_ax(ax):
		ax.set_facecolor(DARK_BG)
		ax.tick_params(colors=DARK_FG)
		ax.xaxis.label.set_color(DARK_FG)
		ax.yaxis.label.set_color(DARK_FG)
		ax.title.set_color(DARK_FG)
		for spine in ax.spines.values():
			spine.set_color(DARK_GRID)

	# 1. Singular value spectrum
	ax1 = fig.add_subplot(3, 3, 1)
	style_ax(ax1)
	ax1.semilogy(S[:20], 'o-', color='#89b4fa', markersize=4)
	ax1.set_xlabel('Component')
	ax1.set_ylabel('Singular value')
	ax1.set_title('Trunk singular values')
	ax1.grid(True, color=DARK_GRID, alpha=0.5)

	# 2. Cumulative variance
	ax2 = fig.add_subplot(3, 3, 2)
	style_ax(ax2)
	cum_var = np.cumsum(S**2) / (S**2).sum()
	ax2.plot(cum_var[:20], 'o-', color='#a6e3a1', markersize=4)
	ax2.axhline(y=0.99, color='#f38ba8', linestyle='--', alpha=0.5, label='99%')
	ax2.set_xlabel('Components')
	ax2.set_ylabel('Cumulative variance')
	ax2.set_title('Variance explained')
	ax2.legend(facecolor=DARK_BG, edgecolor=DARK_GRID, labelcolor=DARK_FG)
	ax2.grid(True, color=DARK_GRID, alpha=0.5)

	# 3. PC1 vs top correlated features (scatter grid)
	# Find top 4 correlations with PC1
	corrs = []
	for i, name in enumerate(feature_names):
		feat = feature_matrix[:, i]
		if feat.std() < 1e-8:
			continue
		c = np.corrcoef(pc1_proj, feat)[0, 1]
		corrs.append((abs(c), c, i, name))
	corrs.sort(reverse=True)

	colors = ['#89b4fa', '#fab387', '#a6e3a1', '#f38ba8', '#cba6f7', '#f5c2e7']
	for plot_idx, (_, c, feat_idx, name) in enumerate(corrs[:6]):
		ax = fig.add_subplot(3, 3, 4 + plot_idx)
		style_ax(ax)
		feat = feature_matrix[:, feat_idx]
		ax.scatter(feat, pc1_proj, alpha=0.3, s=8, color=colors[plot_idx % len(colors)])
		ax.set_xlabel(name)
		ax.set_ylabel('PC1 projection')
		ax.set_title(f'r={c:.3f}')
		ax.grid(True, color=DARK_GRID, alpha=0.5)

	# Bottom: gradient comparison bar chart (use remaining subplot space)
	ax_grad = fig.add_subplot(3, 1, 3)
	style_ax(ax_grad)
	loss_names = list(grad_results.keys())
	# Get per-layer gradient norms for each loss component
	all_layers = sorted(grad_results[loss_names[0]]['layer_grads'].keys())
	x = np.arange(len(all_layers))
	width = 0.25
	grad_colors = ['#89b4fa', '#a6e3a1', '#fab387']
	for i, loss_name in enumerate(loss_names):
		values = [grad_results[loss_name]['layer_grads'].get(l, 0) for l in all_layers]
		ax_grad.bar(x + i * width, values, width, label=loss_name, color=grad_colors[i], alpha=0.8)
	ax_grad.set_xticks(x + width)
	ax_grad.set_xticklabels(all_layers, rotation=45, ha='right', color=DARK_FG, fontsize=8)
	ax_grad.set_ylabel('Gradient norm')
	ax_grad.set_title('Per-layer gradient norm by loss component (with coefficients applied)')
	ax_grad.legend(facecolor=DARK_BG, edgecolor=DARK_GRID, labelcolor=DARK_FG)
	ax_grad.grid(True, axis='y', color=DARK_GRID, alpha=0.5)

	fig.suptitle(f'Trunk Analysis — iteration {iteration} — {len(pc1_proj)} decisions',
		color=DARK_FG, fontsize=14)
	plt.tight_layout()
	plt.savefig(save_path, dpi=150, facecolor=DARK_BG)
	print(f"\nSaved plot to {save_path}")


def main():
	ckpt_arg = sys.argv[1] if len(sys.argv) > 1 else "v6_5/latest.pt"
	ckpt_path = os.path.join(SCRIPT_DIR, ckpt_arg)
	random.seed(42)
	torch.manual_seed(42)
	np.random.seed(42)

	print(f"Loading {ckpt_path}...")
	net, iteration, cfg = load_checkpoint(ckpt_path)
	print(f"Loaded (iteration {iteration})")

	print(f"\nCollecting data ({NUM_GAMES} games)...")
	entries = collect_data(net)
	print(f"Collected {len(entries)} decisions")

	# Investigation 1: PC1 semantics
	hiddens, pc1_proj, feat_matrix, feat_names, S, Vt = analyze_pc1_semantics(net, entries)

	# Investigation 2: Gradient magnitudes
	grad_results = analyze_gradients(net, entries, cfg)

	# Bonus: input encoding PCA
	analyze_input_pc1(entries)

	# Plot
	save_path = ckpt_path.replace('.pt', '_trunk_analysis.png') if ckpt_path.endswith('.pt') \
		else os.path.join(SCRIPT_DIR, 'trunk_analysis.png')
	plot_results(hiddens, pc1_proj, feat_matrix, feat_names, S, Vt,
		grad_results, save_path, iteration)


if __name__ == "__main__":
	main()
