import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
import argparse
import math
import time
import textwrap
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from encoding import (
	INPUT_SIZE, INPUT_SIZE_V2, INPUT_SIZE_V6,
	PLAY_START_SIZE_V2, PLAY_END_SIZE_V2, SCOUT_INSERT_SIZE_V2,
)
from network import ScoutNetwork, FlatScoutNetwork, RandomBot
from training import (
	play_game, play_games_batched, play_games_with_rollouts, play_eval_game,
	OpponentPool, compute_gae,
	prepare_ppo_batch, subsample_batch, concatenate_batches, ppo_update, direct_pg_update,
	play_games_v6, play_games_with_rollouts_v6, prepare_ppo_batch_v6, subsample_batch_v6,
	concatenate_batches_v6, ppo_update_v6, augment_rotation_v6,
)
from game_log import GameLog
from probe import eval_scout_quality

PARAMS = {
	"num_players": 4,
	# "layer_sizes": [256, 128],  # old shallow network
	"layer_sizes": [512, 256, 128],
	# "layer_sizes": [512, 256, 256, 128, 128, 128],
	# "learning_rate": 0.0001,
	"learning_rate": 0.0003, # base
	# "learning_rate": 0.0006,
	# "learning_rate": 0.001,
	# "learning_rate": 0.003,
	# "mini_batch_size": 2**12,
	# "mini_batch_size": 2**13,
	"mini_batch_size": 2**14,
	# "mini_batch_size": 2**15,
	"games_per_iteration": 100,
	# "games_per_iteration": 4,
	"ppo_epochs": 1,
	# "ppo_epochs": 2,
	# "ppo_epochs": 4, # passes over the batch per iteration
	# "ppo_epochs": 8,
	# "replay_past": [0.4, 0.2, 0.1, 0.1, 0.1],  # fraction of current batch to keep from past iterations ([] = no buffer)
	"replay_past": [],
	"clip_epsilon": 0.2,
	# "entropy_bonus": 0.01,
	"entropy_bonus": 0.03,
	# "entropy_bonus": 0.05,
	# "entropy_bonus": 0.08,
	# "entropy_floors": {
	# 	"action_type": 0.05,
	# 	"play_start": 0.05,
	# 	"play_end": 0.05, # 91% of steps have 1 option; floor targets the 9% with 2+
	# 	"scout_insert": 0.05,
	# },
	# v6 entropy floors: quadratic penalty when region entropy drops below floor
	# "entropy_floors": {
	# 	"play": 1.0,
	# 	"scout": 1.0,
	# },
	"entropy_floors": None,
	"entropy_floor_coeff": 1.0,
	# Ablation: zero all gradient on policy_head rows 256-319 (scout logits)
	"zero_scout_policy_grad": False,
	"kl_target": 0.015,
	"reward_mode": "game_score",  # "game_score", "play_length", or "play_and_scout"
	"reward_distribution": 0.7,  # "terminal", "uniform", or 0-1 uniform fraction (game_score mode only)
	# Used this for a while to start training like 2,000 iterations and then turned it off.
	# "shaped_bonus_scale": 0.05,  # per-action bonus for play_length and scout_quality
	"shaped_bonus_scale": 0,
	# "value_loss_coeff": 0.25,
	"value_loss_coeff": 0.5,
	# "gamma": 0.99,
	"gamma": 0.995,
	"gae_lambda": 0.95,
	"training_seats": 4,
	"opponent_pool_size": 10,
	"snapshot_interval": 30, # add to pool every N iterations
	"total_iterations": 1_000_000,
	# "total_iterations": 100,
	"log_interval": 1,
	# "save_interval": 1000,  # iterations between snapshots (ignored if save_interval_hours is set)
	"save_interval_hours": 3,
	# "eval_interval": 30,
	"eval_interval": 5,
	# "encoding_version": 2,
	"encoding_version": 6,
	"num_values": 10,
	# Rollout-based advantage estimation (replaces GAE)
	# "use_rollouts": True,
	"use_rollouts": False,
	"rollout_games": 100,  # real games per iteration (rollouts are per-state within these)
	"rollouts_per_state": 20,  # N rollout games from each decision point
	# "rollout_fraction": 0.1,  # fraction of games_per_iteration to use rollouts instead of GAE
	# "rollout_fraction": 0.05,
	"rollout_fraction": 0.25,
	# "sampling_temperature": 1.5,  # >1.0 flattens sampling for exploration; recorded in old_log_prob so PPO ratios are correct
	"sampling_temperature": 2.5,
	"augment_rotations": 16,  # 1 = no augmentation, 16 = all rotations
	"use_direct_pg": False,  # vanilla policy gradient instead of PPO (forces 1 epoch)
	"diagnose": False,  # per-iteration diagnostics: raw advantages by action type, policy prefs, value accuracy
	"attention": {"dim": 32, "heads": 2, "layers": 1},
	# "save_dir": "bots/v7_5",
	"save_dir": "bots/v7_7",
	"eval_opponents": {
		# "random": "random", # magic word → uses RandomBot
		"v1_4": "bots/v1_4/latest.pt",
		"v2_5": "bots/v2_5/latest.pt",
		"v3_4": "bots/v3_4/latest.pt",
		"v4_2": "bots/v4_2/latest.pt",
	}, # name → checkpoint path (or "random" for RandomBot)
}

def _save_checkpoint(network, optimizer, iteration, cfg, metrics_history, save_dir, filename, pool=None, extra=None):
	path = os.path.join(save_dir, filename)
	tmp_path = path + ".tmp"
	data = {
		"model_state": network.state_dict(),
		"optimizer_state": optimizer.state_dict(),
		"iteration": iteration,
		"config": cfg,
		"metrics_history": metrics_history,
	}
	if pool is not None:
		data["opponent_pool"] = pool.state_dicts()
	if extra:
		data.update(extra)
	torch.save(data, tmp_path)
	for attempt in range(5):
		try:
			os.replace(tmp_path, path)
			return path
		except OSError:
			if attempt < 4:
				time.sleep(1)
	# Last resort: tmp file is still valid, just warn
	print(f"  WARNING: could not rename {tmp_path} → {filename}, checkpoint saved as .tmp")
	return tmp_path

def _count_dormant_neurons(network, states, threshold=0.01):
	"""Count dormant neurons per trunk layer (mean |activation| < threshold across batch).
	Returns dict with per-layer counts and total: {"layer_0": n, ..., "total": n, "total_neurons": n}."""
	if not hasattr(network, 'shared'):
		return None
	# Find activation layers (ReLU or GELU) and hook them
	act_outputs = {}
	hooks = []
	act_idx = 0
	for i, module in enumerate(network.shared):
		if isinstance(module, (torch.nn.ReLU, torch.nn.GELU)):
			idx = act_idx
			def make_hook(layer_idx):
				def hook_fn(mod, inp, out):
					act_outputs[layer_idx] = out.detach()
				return hook_fn
			hooks.append(module.register_forward_hook(make_hook(idx)))
			act_idx += 1
	if not hooks:
		return None
	# Forward pass on a sample of states
	network.eval()
	with torch.no_grad():
		if isinstance(states, torch.Tensor):
			batch = states[:2000]
		else:
			batch = torch.stack(states[:2000])
		dev = next(network.parameters()).device
		network(batch.to(dev))
	# Count dormant neurons (mean |activation| below threshold)
	result = {"total": 0, "total_neurons": 0}
	for idx in sorted(act_outputs):
		out = act_outputs[idx]  # [batch, neurons]
		mean_abs = out.abs().mean(dim=0)  # per-neuron mean |activation|
		dormant = (mean_abs < threshold).sum().item()
		total = out.shape[1]
		result[f"layer_{idx}"] = dormant
		result[f"layer_{idx}_size"] = total
		result["total"] += dormant
		result["total_neurons"] += total
	for h in hooks:
		h.remove()
	return result

def _run_eval(network, eval_opponents, metrics_history, iteration, cfg, save_dir):
	"""Run eval vs all opponents, update metrics, save charts."""
	# Eval runs infrequently — move to CPU to avoid batch-1 GPU overhead
	# and device mismatches in probe code that expects CPU tensors
	was_cuda = next(network.parameters()).is_cuda
	if was_cuda:
		network.cpu()
	try:
		network.eval()
		n_eval = 40
		metrics_history["eval_iteration"].append(iteration)
		for name, eval_net in eval_opponents.items():
			total_margin = 0.0
			for _ in range(n_eval):
				nets = [network] + [eval_net for _ in range(cfg["num_players"] - 1)]
				scores = play_eval_game(nets, cfg["num_players"])
				mean_opponent = sum(scores[1:]) / len(scores[1:])
				total_margin += scores[0] - mean_opponent
			avg_margin = total_margin / n_eval
			metrics_history[f"eval_margin_{name}"].append(avg_margin)
			print(f"  Eval vs {name}: margin={avg_margin:+.1f}")
		scout_len, scout_n = eval_scout_quality(network, n_samples=200)
		metrics_history["scout_play_len"].append(scout_len)
		print(f"  Scout play_len: {scout_len:.2f} (n={scout_n})")
	except Exception as e:
		print(f"  WARNING: eval failed at iter {iteration}: {e}")
		expected_keys = [f"eval_margin_{n}" for n in eval_opponents]
		expected_keys.append("scout_play_len")
		if (metrics_history["eval_iteration"]
				and metrics_history["eval_iteration"][-1] == iteration):
			metrics_history["eval_iteration"].pop()
			for k in expected_keys:
				if k in metrics_history and len(metrics_history[k]) > len(metrics_history["eval_iteration"]):
					metrics_history[k].pop()
	finally:
		if was_cuda:
			network.cuda()
	_save_charts(metrics_history, save_dir, set(eval_opponents), cfg=cfg)
	if cfg.get("diagnose"):
		_save_diagnostic_charts(metrics_history, save_dir)

def _smooth(vals, window):
	"""Centered moving average for chart smoothing."""
	smoothed = []
	half = window // 2
	for i in range(len(vals)):
		start = max(0, i - half)
		end = min(len(vals), i + half + 1)
		smoothed.append(sum(vals[start:end]) / (end - start))
	return smoothed

def _save_diagnostic_charts(metrics_history: dict, save_dir: str):
	"""Generate diagnostics PNG from accumulated diagnostic metrics."""
	iters = metrics_history["iteration"]
	if len(iters) < 2 or not metrics_history.get("diag_adv_std"):
		return
	trim = 30 if len(iters) > 400 else 10 if len(iters) > 100 else 0
	all_iters = iters
	iters = iters[trim:]
	# Trim and smooth diagnostic metrics (same logic as _save_charts)
	trimmed = {}
	smoothed = {}
	for k, vals in metrics_history.items():
		if not k.startswith("diag_") or not vals:
			continue
		start = len(all_iters) - len(vals)
		t = max(trim - start, 0)
		trimmed[k] = vals[t:]
		w = max(len(trimmed[k]) // 10, 3)
		smoothed[k] = _smooth(trimmed[k], w) if len(trimmed[k]) >= w else trimmed[k]
	BG = "#1a1a2e"
	PANEL = "#16213e"
	TEXT = "#e0e0e0"
	SUBTEXT = "#a0a0a0"
	GRID = "#ffffff"
	with plt.style.context("dark_background"):
		fig, axes = plt.subplots(2, 2, figsize=(14, 10))
		fig.patch.set_facecolor(BG)
		fig.suptitle("Diagnostics", fontsize=16, color=TEXT, y=0.98)
		def _style(ax, title, desc):
			ax.set_facecolor(PANEL)
			ax.set_title(title, color=TEXT, fontsize=11)
			ax.text(0.5, -0.15, textwrap.fill(desc, 50), transform=ax.transAxes,
					ha="center", va="top", fontsize=7, color=SUBTEXT, style="italic")
			ax.tick_params(colors=SUBTEXT, labelsize=8)
			ax.grid(True, alpha=0.15, color=GRID)
		# [0,0] Signal vs Noise — advantage spread vs estimated rollout noise
		ax = axes[0, 0]
		for key, label, color, ls in [
			("diag_adv_std", "Advantage Std (total)", "#69db7c", "-"),
			("diag_rollout_noise", "Rollout Noise (est.)", "#ff6b6b", "--"),
		]:
			if key in trimmed:
				ax.plot(iters[-len(trimmed[key]):], trimmed[key], alpha=0.25, color=color, linewidth=0.8)
				ax.plot(iters[-len(smoothed[key]):], smoothed[key], color=color, linewidth=2,
						label=label, linestyle=ls)
		if ax.get_legend_handles_labels()[1]:
			ax.legend(fontsize=7, loc="upper right")
		_style(ax, "Signal vs Noise",
			"Advantage std = total spread. Rollout noise = expected noise from finite rollouts. "
			"Gap between them = real signal. If they overlap, advantages are pure noise.")
		# [0,1] Policy preference when pairs are legal
		ax = axes[0, 1]
		for key, label, color in [
			("diag_policy_p_single", "P(single)", "#ff6b6b"),
			("diag_policy_p_pair", "P(pair)", "#69db7c"),
			("diag_policy_p_3plus", "P(3+)", "#5dadec"),
		]:
			if key in trimmed:
				ax.plot(iters[-len(trimmed[key]):], trimmed[key], alpha=0.25, color=color, linewidth=0.8)
				ax.plot(iters[-len(smoothed[key]):], smoothed[key], color=color, linewidth=2, label=label)
		ax.set_ylim(0, 1)
		if ax.get_legend_handles_labels()[1]:
			ax.legend(fontsize=7, loc="upper right")
		_style(ax, "Policy Preference (when pairs legal)",
			"Avg probability mass on singles vs pairs vs 3+ plays, only at states where pair+ actions are legal.")
		# [1,0] Advantage distribution — p10/p90 band + abs_mean
		ax = axes[1, 0]
		if "diag_adv_p10" in smoothed and "diag_adv_p90" in smoothed:
			n = min(len(smoothed["diag_adv_p10"]), len(smoothed["diag_adv_p90"]))
			x = iters[-n:]
			ax.fill_between(x, smoothed["diag_adv_p10"][-n:], smoothed["diag_adv_p90"][-n:],
							alpha=0.25, color="#5dadec", label="P10-P90 range")
			ax.plot(x, smoothed["diag_adv_p10"][-n:], color="#5dadec", linewidth=1, alpha=0.5)
			ax.plot(x, smoothed["diag_adv_p90"][-n:], color="#5dadec", linewidth=1, alpha=0.5)
		if "diag_adv_abs_mean" in trimmed:
			ax.plot(iters[-len(trimmed["diag_adv_abs_mean"]):], trimmed["diag_adv_abs_mean"],
					alpha=0.25, color="#ffa552", linewidth=0.8)
			ax.plot(iters[-len(smoothed["diag_adv_abs_mean"]):], smoothed["diag_adv_abs_mean"],
					color="#ffa552", linewidth=2, label="Mean |advantage|")
		ax.axhline(y=0, color="#666666", linestyle="--", alpha=0.5)
		if ax.get_legend_handles_labels()[1]:
			ax.legend(fontsize=7, loc="upper right")
		_style(ax, "Advantage Distribution",
			"Shaded band = 10th-90th percentile of raw advantages. "
			"Orange = mean absolute advantage. Shows what signal the policy sees.")
		# [1,1] Value head quality — MAE + correlation on twin axes
		ax = axes[1, 1]
		key = "diag_value_mae"
		if key in trimmed:
			ax.plot(iters[-len(trimmed[key]):], trimmed[key], alpha=0.25, color="#e0aaff", linewidth=0.8)
			ax.plot(iters[-len(smoothed[key]):], smoothed[key], color="#e0aaff", linewidth=2, label="MAE")
		key = "diag_value_corr"
		if key in trimmed:
			ax2 = ax.twinx()
			ax2.plot(iters[-len(trimmed[key]):], trimmed[key], alpha=0.25, color="#74c0fc", linewidth=0.8)
			ax2.plot(iters[-len(smoothed[key]):], smoothed[key], color="#74c0fc", linewidth=2, label="Correlation")
			ax2.set_ylim(-0.1, 1.05)
			ax2.tick_params(colors="#74c0fc", labelsize=8)
			ax2.set_ylabel("Correlation", color="#74c0fc", fontsize=8)
		ax.set_ylabel("MAE", color="#e0aaff", fontsize=8)
		# Manual legend for twin axes
		from matplotlib.lines import Line2D
		ax.legend([Line2D([0],[0], color="#e0aaff", lw=2), Line2D([0],[0], color="#74c0fc", lw=2)],
				  ["MAE (left)", "Corr (right)"], fontsize=7, loc="upper right")
		_style(ax, "Value Head Quality",
			"MAE (purple, left axis): prediction error, lower=better. "
			"Correlation (blue, right axis): ranking quality, higher=better.")
		fig.subplots_adjust(left=0.06, right=0.92, top=0.93, bottom=0.06, hspace=0.40, wspace=0.25)
		try:
			fig.savefig(os.path.join(save_dir, "diagnostics.png"), dpi=100,
						facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.15)
		except OSError as e:
			print(f"  WARNING: failed to save diagnostic charts: {e}")
		plt.close(fig)

def _save_charts(metrics_history: dict, save_dir: str, eval_opponent_names: set[str] | None = None, cfg: dict | None = None):
	"""Generate training charts PNG from accumulated metrics."""
	iters = metrics_history["iteration"]
	# Trim noisy early iterations from charts when there's enough data
	trim = 30 if len(iters) > 400 else 10 if len(iters) > 100 else 0
	iters = iters[trim:]
	all_eval_iters = metrics_history.get("eval_iteration", [])
	eval_trim = sum(1 for ei in all_eval_iters if ei <= trim) if trim else 0
	eval_iters = all_eval_iters[eval_trim:]
	chart_path = os.path.join(save_dir, "charts.png")

	# Precompute trimmed and smoothed data for all metrics.
	# Metrics may have fewer entries than iteration (added mid-training),
	# so right-align to iteration list before trimming.
	all_iters = metrics_history["iteration"]
	trimmed = {}
	smoothed = {}
	for k, vals in metrics_history.items():
		if k in ("iteration", "eval_iteration") or not vals:
			continue
		# Metrics aligned to eval_iteration x-axis
		if k.startswith("eval_") or k == "scout_play_len":
			start = len(all_eval_iters) - len(vals)
			t = max(eval_trim - start, 0)
		else:
			start = len(all_iters) - len(vals)
			t = max(trim - start, 0)
		trimmed[k] = vals[t:]
		w = max(len(trimmed[k]) // 10, 3)
		smoothed[k] = _smooth(trimmed[k], w) if len(trimmed[k]) >= w else trimmed[k]

	BG = "#1a1a2e"
	PANEL = "#16213e"
	TEXT = "#e0e0e0"
	SUBTEXT = "#a0a0a0"
	GRID = "#ffffff"

	with plt.style.context("dark_background"):
		fig, axes = plt.subplots(4, 4, figsize=(18, 20))
		fig.patch.set_facecolor(BG)
		fig.suptitle("Scout Bot Training", fontsize=16, color=TEXT, y=0.98)

		def plot_line(ax, key, title, desc, color):
			ax.set_facecolor(PANEL)
			if key in trimmed:
				ax.plot(iters[-len(trimmed[key]):], trimmed[key], alpha=0.25, color=color, linewidth=0.8)
				ax.plot(iters[-len(smoothed[key]):], smoothed[key], color=color, linewidth=2)
			ax.set_title(title, color=TEXT, fontsize=11)
			ax.text(0.5, -0.15, textwrap.fill(desc, 45), transform=ax.transAxes,
					ha="center", va="top", fontsize=7, color=SUBTEXT, style="italic")
			ax.tick_params(colors=SUBTEXT, labelsize=8)
			ax.grid(True, alpha=0.15, color=GRID)

		def plot_multi(ax, series, title, desc, ylim=None):
			ax.set_facecolor(PANEL)
			for key, label, color in series:
				if key in trimmed:
					ax.plot(iters[-len(smoothed[key]):], smoothed[key],
							color=color, linewidth=1.5, label=label)
			if ylim:
				ax.set_ylim(*ylim)
			if ax.get_legend_handles_labels()[1]:
				ax.legend(fontsize=7, loc="upper right")
			ax.set_title(title, color=TEXT, fontsize=11)
			ax.text(0.5, -0.15, textwrap.fill(desc, 45), transform=ax.transAxes,
					ha="center", va="top", fontsize=7, color=SUBTEXT, style="italic")
			ax.tick_params(colors=SUBTEXT, labelsize=8)
			ax.grid(True, alpha=0.15, color=GRID)

		def _style_eval_ax(ax, title, desc):
			"""Style helper for charts using eval_iteration x-axis."""
			ax.set_title(title, color=TEXT, fontsize=11)
			ax.text(0.5, -0.15, textwrap.fill(desc, 45), transform=ax.transAxes,
					ha="center", va="top", fontsize=7, color=SUBTEXT, style="italic")
			ax.tick_params(colors=SUBTEXT, labelsize=8)
			ax.grid(True, alpha=0.15, color=GRID)

		# Row 0: Game performance (highest priority)
		ax_eval = axes[0, 0]
		ax_eval.set_facecolor(PANEL)
		opponent_colors = ["#b197fc", "#ff6b6b", "#69db7c", "#ffa552", "#5dadec"]
		opponent_keys = sorted(k for k in smoothed if k.startswith("eval_margin_")
			and (eval_opponent_names is None or k[len("eval_margin_"):] in eval_opponent_names))
		for i, key in enumerate(opponent_keys):
			name = key[len("eval_margin_"):]
			c = opponent_colors[i % len(opponent_colors)]
			ax_eval.plot(eval_iters[-len(trimmed[key]):], trimmed[key],
						color=c, alpha=0.25, linewidth=0.8)
			ax_eval.plot(eval_iters[-len(smoothed[key]):], smoothed[key],
						color=c, linewidth=2, label=f"vs {name}")
		ax_eval.axhline(y=0, color="#666666", linestyle="--", alpha=0.5)
		if ax_eval.get_legend_handles_labels()[1]:
			ax_eval.legend(fontsize=7, loc="upper left")
		_style_eval_ax(ax_eval, "Score Margin",
			"P0 score minus mean opponent, averaged over eval games. Positive = winning.")
		plot_line(axes[0, 1], "steps_per_game", "Steps Per Game",
			"Average decisions per game. Shorter games may indicate more decisive play.", "#e0aaff")
		plot_multi(axes[0, 2], [
			("play_len_1_pct", "1", "#ff6b6b"),
			("play_len_2_pct", "2", "#ffa552"),
			("play_len_3_pct", "3", "#69db7c"),
			("play_len_4_pct", "4", "#5dadec"),
			("play_len_5_pct", "5", "#b197fc"),
			("play_len_6_pct", "6", "#74c0fc"),
			("play_len_7plus_pct", "7+", "#ffd43b"),
		], "Play Length Distribution",
			"Fraction of plays by length. Shift from 1-card to longer = learning combos.")
		plot_line(axes[0, 3], "avg_play_length", "Avg Play Length",
			"Mean cards per play action. Higher = learning longer sequences instead of 1-card plays.", "#69db7c")

		# Row 1: Play behavior + training losses
		ax_sq = axes[1, 0]
		ax_sq.set_facecolor(PANEL)
		if "scout_play_len" in trimmed:
			ax_sq.plot(eval_iters[-len(trimmed["scout_play_len"]):], trimmed["scout_play_len"],
					   color="#e0aaff", alpha=0.25, linewidth=0.8)
			ax_sq.plot(eval_iters[-len(smoothed["scout_play_len"]):], smoothed["scout_play_len"],
					   color="#e0aaff", linewidth=2)
		_style_eval_ax(ax_sq, "Scout Play Length",
			"Avg longest set/run containing scouted card after insertion. 1.0 = no play, 2.0 = pairs. Random ~1.5.")
		plot_multi(axes[1, 1], [
			("play_pct", "Play", "#69db7c"),
			("scout_pct", "Scout", "#5dadec"),
			("sns_pct", "S&S", "#ff6b6b"),
		], "Action Type Distribution",
			"Fraction of each action type. Shows how strategy evolves over training.",
			ylim=(0, 1))
		plot_line(axes[1, 2], "policy_loss", "Policy Loss",
			"PPO clipped surrogate loss. Watch for instability (spikes or divergence).", "#ff6b6b")
		plot_line(axes[1, 3], "value_loss", "Value Loss",
			"MSE between predicted and actual returns. Should decrease as value function improves.", "#ffa552")

		# Row 2: Entropy + KL
		if "entropy_play" in trimmed and trimmed["entropy_play"]:
			plot_multi(axes[2, 0], [
				("entropy", "Total", "#69db7c"),
				("entropy_play", "Play Only", "#5dadec"),
				("entropy_scout", "Scout Only", "#ff6b6b"),
			], "Conditional Entropies",
				"Flat-head entropy. Play/Scout are conditional on masking other regions. Low = converging.")
		else:
			plot_multi(axes[2, 0], [
				("entropy_action_type", "Action Type", "#69db7c"),
				("entropy_play_start", "Play Start", "#5dadec"),
				("entropy_play_end", "Play End", "#ffa552"),
				("entropy_scout_insert", "Scout Insert", "#ff6b6b"),
			], "Per-Head Entropy",
				"Entropy per head (steps with 2+ options only). Collapsing = premature convergence.")
		plot_line(axes[2, 1], "entropy_floor_penalty", "Entropy Floor Penalty",
			"Quadratic penalty when head entropy drops below floor. >0 = floor active. 0 = heads above floor.", "#ff922b")
		ax_kl = axes[2, 2]
		plot_line(ax_kl, "approx_kl", "Approx KL",
			"How far policy moved from collection policy. Dashed line = early stop threshold.", "#74c0fc")
		kl_tgt = cfg.get("kl_target", 0.015) if cfg else 0.015
		ax_kl.axhline(y=kl_tgt, color="#ff6b6b", linestyle="--", alpha=0.7, linewidth=1)
		plot_line(axes[2, 3], "kl_batch_frac", "KL Early Stop",
			"Fraction of mini-batches used before KL early stop. 1.0 = no early stop triggered.", "#ffa552")

		# Row 3: Network health
		plot_multi(axes[3, 0], [
			("dormant_neurons_layer_0", "Layer 0", "#ff6b6b"),
			("dormant_neurons_layer_1", "Layer 1", "#ffa552"),
			("dormant_neurons_layer_2", "Layer 2", "#69db7c"),
			("dormant_neurons_total", "Total", "#e0aaff"),
		], "Dormant Neurons",
			"Neurons with mean |activation| < 0.01 across batch. High count = underutilized capacity.")
		plot_line(axes[3, 1], "clip_fraction", "Clip Fraction",
			"Fraction of samples clipped by PPO. <0.01 typical with masked multi-head actions.", "#ff922b")
		plot_multi(axes[3, 2], [
			("explained_variance", "GAE", "#69db7c"),
			("rollout_ev", "Rollout", "#5dadec"),
		], "Explained Variance",
			"Value head accuracy. GAE = vs GAE returns (circular). Rollout = vs empirical ground truth.")
		axes[3, 3].set_visible(False)

		fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.03,
						   hspace=0.40, wspace=0.25)
		try:
			fig.savefig(chart_path, dpi=100, facecolor=fig.get_facecolor(),
						bbox_inches='tight', pad_inches=0.15)
		except OSError as e:
			print(f"  WARNING: failed to save charts: {e}")
		plt.close(fig)

	# Write text summary with smoothed values
	summary_path = os.path.join(save_dir, "summary.txt")
	lines = [f"=== Run: {len(iters)} iterations (trimmed first {trim}) ===\n"]

	def _snap_idx(n):
		count = min(n, 5)
		step = (n - 1) / (count - 1) if count > 1 else 0
		return [round(i * step) for i in range(count)]

	# Training metrics
	idx = _snap_idx(len(iters))
	lines.append("iters: " + ", ".join(str(iters[i]) for i in idx))
	for k in smoothed:
		if k.startswith("eval_"):
			continue
		kidx = idx if len(smoothed[k]) >= len(iters) else _snap_idx(len(smoothed[k]))
		vals = [f"{smoothed[k][i]:.4f}" for i in kidx]
		lines.append(f"  {k}: {', '.join(vals)}")
	lines.append("")

	# Eval metrics
	eval_sm_keys = [k for k in smoothed if k.startswith("eval_")]
	if eval_sm_keys:
		lines.append("=== Eval ===\n")
		eidx = _snap_idx(len(eval_iters))
		lines.append("iters: " + ", ".join(str(eval_iters[i]) for i in eidx))
		for k in eval_sm_keys:
			kidx = eidx if len(smoothed[k]) >= len(eval_iters) else _snap_idx(len(smoothed[k]))
			fmt = "+.2f" if "margin" in k else ".4f"
			vals = [f"{smoothed[k][i]:{fmt}}" for i in kidx]
			lines.append(f"  {k}: {', '.join(vals)}")
		lines.append("")

	try:
		with open(summary_path, "w") as f:
			f.write("\n".join(lines))
	except OSError as e:
		print(f"  WARNING: failed to save summary: {e}")

def _compute_diagnostics(network, records, raw_advantages,
						  rollout_margin_std: float, rollouts_per_state: int) -> dict:
	"""Compute per-iteration diagnostics from pre-augmentation records and raw advantages.
	Network must be in eval mode with pre-update weights."""
	diag = {}
	# 1. Advantage distribution stats — signal strength
	n_adv = len(raw_advantages)
	if n_adv > 0:
		mean_adv = sum(raw_advantages) / n_adv
		diag["adv_std"] = (sum((a - mean_adv)**2 for a in raw_advantages) / n_adv) ** 0.5
		diag["adv_abs_mean"] = sum(abs(a) for a in raw_advantages) / n_adv
		sorted_adv = sorted(raw_advantages)
		diag["adv_p10"] = sorted_adv[max(0, int(n_adv * 0.1))]
		diag["adv_p90"] = sorted_adv[min(n_adv - 1, int(n_adv * 0.9))]
	else:
		diag["adv_std"] = diag["adv_abs_mean"] = diag["adv_p10"] = diag["adv_p90"] = 0.0
	# 2. Rollout noise estimate
	# Each advantage = V_after - V_before; each V is mean of rollouts_per_state rollouts
	# Expected noise std of the advantage = sqrt(2) * margin_std / sqrt(rollouts_per_state)
	diag["rollout_noise"] = math.sqrt(2) * rollout_margin_std / math.sqrt(max(rollouts_per_state, 1))
	# Signal-to-noise: how much of the advantage spread is real vs rollout noise
	# total_var = signal_var + noise_var → signal = sqrt(total^2 - noise^2)
	if diag["adv_std"] > diag["rollout_noise"]:
		signal = (diag["adv_std"]**2 - diag["rollout_noise"]**2) ** 0.5
		diag["snr"] = signal / diag["rollout_noise"] if diag["rollout_noise"] > 1e-8 else float('inf')
	else:
		diag["snr"] = 0.0
	# 3. Policy preference: P(single) vs P(pair) vs P(3+) when pairs are legal
	dev = next(network.parameters()).device
	states = torch.stack([r.state for r in records]).to(dev)
	with torch.no_grad():
		hidden = network(states)
		all_logits = network.policy_logits(hidden)
	p_single_total, p_pair_total, p_3plus_total = 0.0, 0.0, 0.0
	n_with_choice = 0
	for i, rec in enumerate(records):
		if rec.play_length is None:
			continue  # scout/sns decision, skip
		mask = torch.from_numpy(rec.mask)
		logits = all_logits[i]
		# Check if any pair+ actions are legal
		has_pair = False
		for a in range(256):
			if mask[a]:
				length = (a % 16) - (a // 16) + 1
				if length >= 2:
					has_pair = True
					break
		if not has_pair:
			continue
		# Compute masked softmax over all legal actions (play + scout + sns)
		masked_logits = logits.clone()
		masked_logits[~mask] = float('-inf')
		probs = torch.softmax(masked_logits, dim=0)
		# Sum probability mass by play length (play actions only)
		p_single, p_pair, p_3plus = 0.0, 0.0, 0.0
		for a in range(256):
			if mask[a]:
				length = (a % 16) - (a // 16) + 1
				p = probs[a].item()
				if length == 1:
					p_single += p
				elif length == 2:
					p_pair += p
				else:
					p_3plus += p
		p_single_total += p_single
		p_pair_total += p_pair
		p_3plus_total += p_3plus
		n_with_choice += 1
	if n_with_choice > 0:
		diag["policy_p_single"] = p_single_total / n_with_choice
		diag["policy_p_pair"] = p_pair_total / n_with_choice
		diag["policy_p_3plus"] = p_3plus_total / n_with_choice
	else:
		diag["policy_p_single"] = 0.0
		diag["policy_p_pair"] = 0.0
		diag["policy_p_3plus"] = 0.0
	# 3. Value prediction accuracy: predicted_value vs rollout value
	preds = [r.predicted_value for r in records]
	actuals = [r.value for r in records]
	n = len(preds)
	if n > 1:
		mae = sum(abs(p - a) for p, a in zip(preds, actuals)) / n
		mean_p = sum(preds) / n
		mean_a = sum(actuals) / n
		cov = sum((p - mean_p) * (a - mean_a) for p, a in zip(preds, actuals)) / n
		std_p = (sum((p - mean_p) ** 2 for p in preds) / n) ** 0.5
		std_a = (sum((a - mean_a) ** 2 for a in actuals) / n) ** 0.5
		corr = cov / (std_p * std_a) if std_p > 1e-8 and std_a > 1e-8 else 0.0
		diag["value_mae"] = mae
		diag["value_corr"] = corr
	else:
		diag["value_mae"] = 0.0
		diag["value_corr"] = 0.0
	return diag

def train(config: dict | None = None, profile_iters: int | None = None):
	cfg = {**PARAMS, **(config or {})}
	save_dir = os.path.join(SCRIPT_DIR, cfg["save_dir"])
	os.makedirs(save_dir, exist_ok=True)
	ev = cfg.get("encoding_version", 1)
	if ev == 6:
		network = FlatScoutNetwork(INPUT_SIZE_V6, cfg["layer_sizes"],
			encoding_version=6, attention=cfg.get("attention"))
	elif ev == 2:
		network = ScoutNetwork(INPUT_SIZE_V2, cfg["layer_sizes"],
			play_start_size=PLAY_START_SIZE_V2, play_end_size=PLAY_END_SIZE_V2,
			scout_insert_size=SCOUT_INSERT_SIZE_V2, encoding_version=2)
	else:
		network = ScoutNetwork(input_size=INPUT_SIZE, layer_sizes=cfg["layer_sizes"])
	optimizer = torch.optim.Adam(network.parameters(), lr=cfg["learning_rate"])
	metrics_history = {
		"iteration": [], "reward": [], "value": [],
		"policy_loss": [], "value_loss": [], "entropy": [],
		"clip_fraction": [], "approx_kl": [], "explained_variance": [],
		"entropy_action_type": [], "entropy_play_start": [],
		"entropy_play_end": [], "entropy_scout_insert": [],
		"entropy_floor_penalty": [],
		"entropy_play": [], "entropy_scout": [],
		"play_pct": [], "scout_pct": [], "sns_pct": [],
		"steps_per_game": [],
		"avg_play_length": [], "reward_std": [],
		"play_len_1_pct": [], "play_len_2_pct": [], "play_len_3_pct": [],
		"play_len_4_pct": [], "play_len_5_pct": [], "play_len_6_pct": [],
		"play_len_7plus_pct": [],
		"eval_iteration": [],
		"scout_play_len": [],
		# Diagnostics (populated when diagnose=True)
		"diag_adv_std": [], "diag_adv_abs_mean": [],
		"diag_adv_p10": [], "diag_adv_p90": [],
		"diag_rollout_noise": [], "diag_snr": [],
		"diag_policy_p_single": [], "diag_policy_p_pair": [],
		"diag_policy_p_3plus": [],
		"diag_value_mae": [], "diag_value_corr": [],
		# Dormant neuron tracking (mean |activation| < threshold)
		"dormant_neurons_total": [],
		"dormant_neurons_layer_0": [], "dormant_neurons_layer_1": [], "dormant_neurons_layer_2": [],
		# KL early stopping
		"kl_batch_frac": [],
		# Rollout value accuracy (EV against rollout ground truth)
		"rollout_ev": [],
	}
	start_iter = 1
	# Auto-resume if save dir has a checkpoint
	resume_path = os.path.join(save_dir, "latest.pt")
	if os.path.exists(resume_path):
		checkpoint = torch.load(resume_path, weights_only=False, map_location='cpu')
		network.load_state_dict(checkpoint["model_state"])
		optimizer.load_state_dict(checkpoint["optimizer_state"])
		start_iter = checkpoint["iteration"] + 1
		if "metrics_history" in checkpoint:
			saved_metrics = checkpoint["metrics_history"]
			# Migrate old eval_margin → eval_margin_random
			if "eval_margin" in saved_metrics and "eval_margin_random" not in saved_metrics:
				saved_metrics["eval_margin_random"] = saved_metrics.pop("eval_margin")
			# Merge saved metrics, keeping new keys with empty defaults
			for k, v in saved_metrics.items():
				metrics_history[k] = v
		saved_cfg = checkpoint.get("config", {})
		# Backward compat: convert old hidden_size/first_hidden_size to layer_sizes
		if "layer_sizes" not in saved_cfg and "hidden_size" in saved_cfg:
			saved_cfg["layer_sizes"] = [
				saved_cfg.get("first_hidden_size", 256),
				saved_cfg["hidden_size"],
			]
		# PARAMS overrides saved config; architecture params always come from checkpoint
		cfg = {**saved_cfg, **PARAMS, **(config or {})}
		if "layer_sizes" in saved_cfg:
			cfg["layer_sizes"] = saved_cfg["layer_sizes"]
		if "encoding_version" in saved_cfg:
			cfg["encoding_version"] = saved_cfg["encoding_version"]
		if "attention" in saved_cfg:
			cfg["attention"] = saved_cfg["attention"]
		print(f"Resumed from iteration {start_iter - 1}")
	pool = OpponentPool(max_size=cfg["opponent_pool_size"])
	if os.path.exists(resume_path):
		checkpoint = torch.load(resume_path, weights_only=False, map_location='cpu')
		if "opponent_pool" in checkpoint:
			pool.load_state_dicts(checkpoint["opponent_pool"], network)
			print(f"  Restored opponent pool ({len(pool.versions)} versions)")
	if not pool.versions:
		pool.add(network)
	# Load eval opponents
	eval_opponents = {}
	for name, path in cfg.get("eval_opponents", {}).items():
		if path == "random":
			eval_opponents[name] = RandomBot()
		else:
			ckpt = torch.load(os.path.join(SCRIPT_DIR, path), weights_only=False, map_location='cpu')
			ckpt_cfg = ckpt.get("config", {})
			if "layer_sizes" in ckpt_cfg:
				ls = ckpt_cfg["layer_sizes"]
			else:
				ls = [ckpt_cfg.get("first_hidden_size", 256), ckpt_cfg.get("hidden_size", 128)]
			eval_ev = ckpt_cfg.get("encoding_version", 1)
			if eval_ev == 6:
				eval_net = FlatScoutNetwork(INPUT_SIZE_V6, ls,
					encoding_version=6, attention=ckpt_cfg.get("attention"))
			elif eval_ev == 2:
				eval_net = ScoutNetwork(INPUT_SIZE_V2, ls,
					play_start_size=PLAY_START_SIZE_V2, play_end_size=PLAY_END_SIZE_V2,
					scout_insert_size=SCOUT_INSERT_SIZE_V2, encoding_version=2)
			else:
				eval_net = ScoutNetwork(layer_sizes=ls)
			eval_net.load_state_dict(ckpt["model_state"])
			eval_net.eval()
			eval_opponents[name] = eval_net
			print(f"  Loaded eval opponent '{name}' from {path} (v{eval_ev})")
		key = f"eval_margin_{name}"
		if key not in metrics_history:
			metrics_history[key] = []
	input_size = INPUT_SIZE_V6 if ev == 6 else INPUT_SIZE_V2 if ev == 2 else INPUT_SIZE
	print(f"Training Scout bot: {cfg['num_players']} players, "
		  f"input_size={input_size}, encoding=v{ev}, layers={cfg['layer_sizes']}")
	rp = cfg.get("replay_past", [])
	replay_str = f", replay_past={rp}" if rp else ""
	print(f"Games/iter={cfg['games_per_iteration']}, "
		  f"PPO epochs={cfg['ppo_epochs']}{replay_str}")
	print(f"Output: {save_dir}/")
	profiler = None
	if profile_iters:
		from pyinstrument import Profiler
		profile_stop = start_iter + profile_iters
		profiler = Profiler()
		print(f"\nProfiling {profile_iters} iterations ({start_iter} to {profile_stop - 1})...")
		profiler.start()
	iteration = start_iter
	replay_past = cfg.get("replay_past", [])
	replay_buffer = deque(maxlen=len(replay_past) + 1) if replay_past else None
	last_snapshot_time = time.time()
	if torch.cuda.is_available():
		network.cuda()
		for state in optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.cuda()
	# Pre-training eval (init weights baseline)
	if start_iter <= 1:
		_run_eval(network, eval_opponents, metrics_history, 0, cfg, save_dir)
	try:
		for iteration in range(start_iter, cfg["total_iterations"] + 1):
			t0 = time.time()
			# Self-play: collect games and compute advantages
			network.eval()
			rollout_time = 0.0
			if ev == 6 and cfg.get("use_rollouts", False) and not cfg.get("rollout_fraction"):
				iteration_records, advantages, rollout_margin_std = play_games_with_rollouts_v6(
					network, cfg.get("rollout_games", 10), cfg["num_players"],
					rollouts_per_state=cfg.get("rollouts_per_state", 50),
					training_seats=cfg["training_seats"],
					temperature=cfg.get("sampling_temperature", 1.0))
				raw_advantages = list(advantages)
				original_records = iteration_records  # pre-augmentation copy for metrics
				if cfg.get("augment_rotations", 1) > 1:
					iteration_records, advantages = augment_rotation_v6(
						iteration_records, advantages, network)
				returns = None  # prepare_ppo_batch_v6 uses record.value
			elif ev == 6:
				rollout_frac = cfg.get("rollout_fraction", 0.0)
				n_total = cfg["games_per_iteration"]
				n_rollout = int(n_total * rollout_frac)
				n_gae = n_total - n_rollout
				# GAE games
				gae_records = play_games_v6(
					network, n_gae, cfg["num_players"],
					training_seats=cfg["training_seats"],
					opponent_pool=pool.versions or None,
					reward_distribution=cfg.get("reward_distribution", "terminal"),
					reward_mode=cfg.get("reward_mode", "game_score"),
					shaped_bonus_scale=cfg.get("shaped_bonus_scale", 0.0),
					temperature=cfg.get("sampling_temperature", 1.0))
				gae_advantages, gae_returns = compute_gae(
					gae_records, gamma=cfg["gamma"], lam=cfg["gae_lambda"])
				raw_advantages = [ret - rec.value for rec, ret in zip(gae_records, gae_returns)]
				for rec, ret in zip(gae_records, gae_returns):
					rec.value = ret  # overwrite with GAE return (value target)
				# Rollout games (high-quality value targets and advantages)
				rollout_time = 0.0
				if n_rollout > 0:
					t_ro = time.time()
					ro_records, ro_advantages, rollout_margin_std = play_games_with_rollouts_v6(
						network, n_rollout, cfg["num_players"],
						rollouts_per_state=cfg.get("rollouts_per_state", 20),
						training_seats=cfg["training_seats"],
						temperature=cfg.get("sampling_temperature", 1.0))
					# EV of value head predictions vs rollout ground truth
					ro_targets = [r.value for r in ro_records]
					ro_preds = [r.predicted_value for r in ro_records]
					ro_var = sum((t - sum(ro_targets)/len(ro_targets))**2 for t in ro_targets) / len(ro_targets)
					ro_resid_var = sum((p - t)**2 for p, t in zip(ro_preds, ro_targets)) / len(ro_targets)
					ro_ev = 1.0 - ro_resid_var / (ro_var + 1e-8)
					metrics_history["rollout_ev"].append(ro_ev)
					# Offset game_ids to avoid collision with GAE games
					for rec in ro_records:
						rec.game_id += n_gae
					advantages = gae_advantages + ro_advantages
					iteration_records = gae_records + ro_records
					raw_advantages = raw_advantages + ro_advantages
					rollout_time = time.time() - t_ro
				else:
					iteration_records = gae_records
					advantages = gae_advantages
					rollout_margin_std = 0.0
				original_records = iteration_records
				if cfg.get("augment_rotations", 1) > 1:
					iteration_records, advantages = augment_rotation_v6(
						iteration_records, advantages, network)
				returns = None  # prepare_ppo_batch_v6 uses record.value
			elif cfg.get("use_rollouts"):
				iteration_records, advantages = play_games_with_rollouts(
					network, cfg.get("rollout_games", 40), cfg["num_players"],
					rollouts_per_state=cfg.get("rollouts_per_state", 10),
					training_seats=cfg["training_seats"])
				original_records = iteration_records
				returns = [r.value for r in iteration_records]
			else:
				iteration_records = play_games_batched(
					network, cfg["games_per_iteration"], cfg["num_players"],
					training_seats=cfg["training_seats"],
					opponent_pool=pool.versions or None,
					reward_distribution=cfg.get("reward_distribution", "terminal"),
					reward_mode=cfg.get("reward_mode", "game_score"),
					shaped_bonus_scale=cfg.get("shaped_bonus_scale", 0.0))
				original_records = iteration_records
				advantages, returns = compute_gae(
					iteration_records, gamma=cfg["gamma"], lam=cfg["gae_lambda"])
			play_time = time.time() - t0
			if any(math.isnan(r.value) for r in original_records):
				print(f"[iter {iteration:>5}] WARNING: NaN in forward pass, skipping update")
				continue
			# Diagnostics: raw advantage signal, policy preferences, value accuracy
			diag = None
			if ev == 6 and cfg.get("diagnose"):
				diag = _compute_diagnostics(network, original_records, raw_advantages,
											rollout_margin_std, cfg.get("rollouts_per_state", 25))
			# PPO training — on-policy, all steps from this iteration
			network.train()
			# LR annealing: linear decay to 0 (disabled for short diagnostic runs)
			if cfg["total_iterations"] <= 1000:
				lr = cfg["learning_rate"]
			else:
				lr = cfg["learning_rate"] * (1 - iteration / cfg["total_iterations"])
			optimizer.param_groups[0]["lr"] = lr
			use_dpg = cfg.get("use_direct_pg", False)
			if ev == 6:
				batch = prepare_ppo_batch_v6(iteration_records, advantages)
				if batch is not None and replay_buffer is not None:
					replay_buffer.append(batch)
					buf = list(replay_buffer)
					sampled = [buf[-1]]
					current_n = batch["n"]
					for i, b in enumerate(reversed(buf[:-1])):
						keep_n = max(1, int(replay_past[i] * current_n))
						sampled.append(subsample_batch_v6(b, keep_n))
					training_batch = concatenate_batches_v6(sampled)
				else:
					training_batch = batch
			else:
				batch = prepare_ppo_batch(iteration_records, advantages, returns=returns)
				if not use_dpg and batch is not None and replay_buffer is not None:
					replay_buffer.append(batch)
					buf = list(replay_buffer)
					sampled = [buf[-1]]
					current_n = batch["n"]
					for i, b in enumerate(reversed(buf[:-1])):
						keep_n = max(1, int(replay_past[i] * current_n))
						sampled.append(subsample_batch(b, keep_n))
					training_batch = concatenate_batches(sampled)
				else:
					training_batch = batch
			n_epochs = 1 if use_dpg else cfg["ppo_epochs"]
			ppo_sums = {}
			for epoch in range(n_epochs):
				if ev == 6:
					m = ppo_update_v6(
						network, optimizer, training_batch,
						clip_epsilon=cfg["clip_epsilon"],
						entropy_bonus=cfg["entropy_bonus"],
						value_loss_coeff=cfg["value_loss_coeff"],
						mini_batch_size=cfg.get("mini_batch_size"),
						entropy_floors=cfg.get("entropy_floors"),
						entropy_floor_coeff=cfg.get("entropy_floor_coeff", 1.0),
						zero_scout_policy_grad=cfg.get("zero_scout_policy_grad", False),
					kl_target=cfg.get("kl_target", 0.015),
					)
					# Epoch 0: ratios must be ~1.0 (catches old_log_prob recording bugs)
					if epoch == 0 and (not replay_buffer or len(replay_buffer) <= 1) and abs(m["mean_ratio"] - 1.0) > 0.01:
						print(f"  WARNING: epoch 0 mean ratio={m['mean_ratio']:.4f} (expected ~1.0)")
				elif use_dpg:
					m = direct_pg_update(
						network, optimizer, training_batch,
						entropy_bonus=cfg["entropy_bonus"],
						value_loss_coeff=cfg["value_loss_coeff"],
						entropy_floors=cfg.get("entropy_floors"),
						entropy_floor_coeff=cfg.get("entropy_floor_coeff", 1.0),
						play_start_size=network.play_start_size,
					)
				else:
					m = ppo_update(
						network, optimizer, training_batch,
						clip_epsilon=cfg["clip_epsilon"],
						entropy_bonus=cfg["entropy_bonus"],
						value_loss_coeff=cfg["value_loss_coeff"],
						entropy_floors=cfg.get("entropy_floors"),
						entropy_floor_coeff=cfg.get("entropy_floor_coeff", 1.0),
						play_start_size=network.play_start_size,
					)
					# Epoch 0: ratios must be ~1.0 (stale buffer data shifts this)
					if epoch == 0 and (not replay_buffer or len(replay_buffer) <= 1) and abs(m["mean_ratio"] - 1.0) > 0.01:
						print(f"  WARNING: epoch 0 mean ratio={m['mean_ratio']:.4f} (expected ~1.0)")
				for k, v in m.items():
					ppo_sums[k] = ppo_sums.get(k, 0.0) + v
				# KL early stopping across epochs
				if m.get("approx_kl", 0) > cfg.get("kl_target", 0.015):
					break
			actual_epochs = epoch + 1
			ppo_avg = {k: v / actual_epochs for k, v in ppo_sums.items()}
			train_time = time.time() - t0 - play_time
			# Snapshot to opponent pool
			if iteration % cfg["snapshot_interval"] == 0:
				pool.add(network)
			# Logging + metrics + latest save
			if iteration % cfg["log_interval"] == 0:
				# Use original (pre-augmentation) records for behavioral metrics
				metric_records = original_records
				p0_records = [r for r in metric_records if r.player == 0]
				# Per-round reward: sum steps within each round so metric is
				# comparable across terminal vs uniform reward distribution
				round_totals: dict[tuple[int,int], float] = {}
				for r in p0_records:
					key = (r.game_id, r.round_num)
					round_totals[key] = round_totals.get(key, 0.0) + r.reward
				p0_rewards = list(round_totals.values())
				avg_reward = sum(p0_rewards) / max(len(p0_rewards), 1)
				reward_std = (sum((r - avg_reward)**2 for r in p0_rewards) / max(len(p0_rewards), 1)) ** 0.5
				avg_value = sum(r.value for r in p0_records) / max(len(p0_records), 1)
				# Behavioral metrics (from original records, not augmented)
				n_steps = len(metric_records)
				if ev == 6:
					n_play = sum(1 for r in metric_records if r.action < 256)
					n_scout = sum(1 for r in metric_records if 256 <= r.action < 320)
					n_sns = sum(1 for r in metric_records if r.action >= 320)
				else:
					n_play = sum(1 for r in metric_records if r.action_type == 0)
					n_scout = sum(1 for r in metric_records if 1 <= r.action_type <= 4)
					n_sns = sum(1 for r in metric_records if 5 <= r.action_type <= 8)
				play_pct = n_play / max(n_steps, 1)
				scout_pct = n_scout / max(n_steps, 1)
				sns_pct = n_sns / max(n_steps, 1)
				num_games = cfg["rollout_games"] if cfg.get("use_rollouts") or ev == 6 else cfg["games_per_iteration"]
				steps_per_game = n_steps / num_games
				play_lengths = [r.play_length for r in metric_records if r.play_length is not None]
				avg_play_length = sum(play_lengths) / max(len(play_lengths), 1)
				n_plays = max(len(play_lengths), 1)
				play_len_counts = [0] * 8 # indices 1-7, 0 unused
				for l in play_lengths:
					play_len_counts[min(l, 7)] += 1
				play_len_pcts = [c / n_plays for c in play_len_counts]
				# Dormant neuron count
				dormant_info = _count_dormant_neurons(network, training_batch["states"])
				if dormant_info:
					metrics_history["dormant_neurons_total"].append(dormant_info["total"])
					for li in range(3):
						k = f"dormant_neurons_layer_{li}"
						metrics_history[k].append(dormant_info.get(f"layer_{li}", 0))
				metrics_history["iteration"].append(iteration)
				metrics_history["reward"].append(avg_reward)
				metrics_history["value"].append(avg_value)
				if ev == 6:
					for k in ("policy_loss", "value_loss", "entropy",
							  "clip_fraction", "approx_kl", "explained_variance",
							  "entropy_play", "entropy_scout"):
						metrics_history[k].append(ppo_avg.get(k, 0.0))
					used = ppo_avg.get("kl_batches_used", 1)
					total = ppo_avg.get("kl_batches_total", 1)
					metrics_history["kl_batch_frac"].append(used / total if total > 0 else 1.0)
				else:
					for k in ("policy_loss", "value_loss", "entropy",
							  "clip_fraction", "approx_kl", "explained_variance",
							  "entropy_action_type", "entropy_play_start",
							  "entropy_play_end", "entropy_scout_insert",
							  "entropy_floor_penalty"):
						metrics_history[k].append(ppo_avg[k])
				metrics_history["play_pct"].append(play_pct)
				metrics_history["scout_pct"].append(scout_pct)
				metrics_history["sns_pct"].append(sns_pct)
				metrics_history["steps_per_game"].append(steps_per_game)
				metrics_history["avg_play_length"].append(avg_play_length)
				for i in range(1, 7):
					metrics_history[f"play_len_{i}_pct"].append(play_len_pcts[i])
				metrics_history["play_len_7plus_pct"].append(play_len_pcts[7])
				metrics_history["reward_std"].append(reward_std)
				if diag is not None:
					for k in ("adv_std", "adv_abs_mean", "adv_p10", "adv_p90",
							  "rollout_noise", "snr",
							  "policy_p_single", "policy_p_pair", "policy_p_3plus",
							  "value_mae", "value_corr"):
						metrics_history[f"diag_{k}"].append(diag[k])
				buf_str = f"({training_batch['n']})" if training_batch and replay_buffer and len(replay_buffer) > 1 else ""
				print(f"[iter {iteration:>5}] "
					  f"reward={avg_reward:+.3f}  value={avg_value:+.3f}  "
					  f"ploss={ppo_avg['policy_loss']:.4f}  vloss={ppo_avg['value_loss']:.4f}  "
					  f"ent={ppo_avg['entropy']:.3f}  clip={ppo_avg['clip_fraction']:.2f}  "
					  f"kl={ppo_avg['approx_kl']:.4f}  ev={ppo_avg['explained_variance']:.2f}  "
					  f"steps={n_steps}{buf_str}  pool={len(pool.versions)}  "
					  f"play={play_time:.1f}s  train={train_time:.1f}s"
					  f"{'  ro=' + f'{rollout_time:.1f}s' if rollout_time > 0 else ''}"
					  f"{'  dormant=' + str(dormant_info['total']) + '/' + str(dormant_info['total_neurons']) if dormant_info else ''}")
				if diag is not None:
					print(f"  diag: adv_std={diag['adv_std']:.4f} noise={diag['rollout_noise']:.4f} "
						  f"SNR={diag['snr']:.2f}  "
						  f"P(1)={diag['policy_p_single']:.2f} P(2)={diag['policy_p_pair']:.2f} "
						  f"P(3+)={diag['policy_p_3plus']:.2f}  "
						  f"v_mae={diag['value_mae']:.3f} v_corr={diag['value_corr']:.2f}")
				_save_checkpoint(network, optimizer, iteration, cfg, metrics_history, save_dir, "latest.pt", pool=pool)
			# Periodic snapshots (time-based or iteration-based)
			now = time.time()
			save_hours = cfg.get("save_interval_hours")
			save_snapshot = False
			if save_hours is not None:
				save_snapshot = (now - last_snapshot_time) >= save_hours * 3600
			elif iteration % cfg.get("save_interval", 1000) == 0:
				save_snapshot = True
			if save_snapshot:
				last_snapshot_time = now
				_save_checkpoint(network, optimizer, iteration, cfg, metrics_history,
								save_dir, f"iter_{iteration}.pt")
				# Save a sample game log for replay
				log = GameLog(num_players=cfg["num_players"])
				network.eval()
				opponents = pool.sample(cfg["num_players"] - cfg["training_seats"]) or None
				play_game(network, cfg["num_players"], opponent_pool=opponents, game_log=log,
						  training_seats=cfg["training_seats"])
				try:
					log.save(os.path.join(save_dir, f"iter_{iteration}_game.json"))
				except OSError as e:
					print(f"  WARNING: failed to save game log: {e}")
				print(f"  Saved snapshot + game log (iter {iteration})")
			# Eval vs all opponents
			if iteration % cfg["eval_interval"] == 0:
				_run_eval(network, eval_opponents, metrics_history, iteration, cfg, save_dir)
			# Profile exit
			if profiler and iteration >= profile_stop - 1:
				profiler.stop()
				profile_path = os.path.join(save_dir, "profile.txt")
				output = profiler.output_text(unicode=False, color=False)
				print(f"\n{output}")
				with open(profile_path, "w") as f:
					f.write(output)
				html_path = os.path.join(save_dir, "profile.html")
				with open(html_path, "w") as f:
					f.write(profiler.output_html())
				print(f"Profile saved to {profile_path} and {html_path}")
				return
	except KeyboardInterrupt:
		print(f"\nInterrupted at iteration {iteration}.")
		if profiler:
			profiler.stop()
			print(profiler.output_text(unicode=False, color=False))
		return
	# Final save
	_save_checkpoint(network, optimizer, iteration, cfg, metrics_history, save_dir, "latest.pt", pool=pool)
	_save_charts(metrics_history, save_dir, set(eval_opponents), cfg=cfg)
	if cfg.get("diagnose"):
		_save_diagnostic_charts(metrics_history, save_dir)
	print(f"Training complete. Saved to {save_dir}/latest.pt")

def main():
	parser = argparse.ArgumentParser(description="Train a Scout card game bot")
	parser.add_argument("--players", type=int, default=None, choices=[3, 4, 5])
	parser.add_argument("--save-dir", type=str, default=None)
	parser.add_argument("--replay", type=str, default=None,
		help="Path to a game log JSON file to replay")
	parser.add_argument("--match", nargs="+", metavar="AGENT",
		help="Play agents against each other (e.g., random path/to/model.pt)")
	parser.add_argument("--games", type=int, default=100,
		help="Number of games for --match (default: 100)")
	parser.add_argument("--profile", type=int, default=None, metavar="N",
		help="Profile N training iterations with pyinstrument, then exit")
	args = parser.parse_args()
	if args.replay:
		from game_log import GameLog, print_replay
		log = GameLog.load(args.replay)
		print_replay(log)
		return
	if args.match:
		from matchup import load_agent, run_matchup
		num_players = args.players or len(args.match)
		agents = [load_agent(spec) for spec in args.match]
		if len(agents) != num_players:
			print(f"Error: --match needs {num_players} agents (got {len(agents)})")
			return
		run_matchup(agents, args.games)
		return
	config = {
		"num_players": args.players,
		"save_dir": args.save_dir,
	}
	train({k: v for k, v in config.items() if v is not None},
		  profile_iters=args.profile)

if __name__ == "__main__":
	main()
