import torch
import argparse
import math
import time
import os
import textwrap
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from encoding import (
	INPUT_SIZE, INPUT_SIZE_V2,
	PLAY_START_SIZE_V2, PLAY_END_SIZE_V2, SCOUT_INSERT_SIZE_V2,
)
from network import ScoutNetwork, RandomBot
from training import play_game, play_games_batched, play_games_with_rollouts, play_eval_game, OpponentPool, compute_gae, prepare_ppo_batch, concatenate_batches, ppo_update, direct_pg_update
from game_log import GameLog
from probe import eval_scout_quality

PARAMS = {
	"num_players": 4,
	# "layer_sizes": [256, 128],  # old shallow network
	"layer_sizes": [512, 256, 256, 128, 128, 128],
	# "learning_rate": 0.0001, # didn't help
	"learning_rate": 0.0003, # base
	# "learning_rate": 0.0006, # seems slightly worse
	# "learning_rate": 0.001, # too much
	# "learning_rate": 0.003, # too much
	# "batch_size": 256,  # TODO: implement mini-batching within PPO epochs
	"games_per_iteration": 400,
	# "ppo_epochs": 4, # passes over the batch per iteration
	"ppo_epochs": 8,
	"replay_buffer_size": 20,  # keep last N iterations of data for PPO (1 = no buffer)
	"clip_epsilon": 0.2,
	"entropy_bonus": 0.01,
	# "entropy_bonus": 0.05,
	# "entropy_bonus": 0.25,
	"entropy_floors": {
		"action_type": 0.05,
		"play_start": 0.05,
		"play_end": 0.05, # 91% of steps have 1 option; floor targets the 9% with 2+
		"scout_insert": 0.05,
	},
	"entropy_floor_coeff": 1.0,
	"reward_mode": "game_score",  # "game_score", "play_length", or "play_and_scout"
	"reward_distribution": 0.7,  # "terminal", "uniform", or 0-1 uniform fraction (game_score mode only)
	# Used this for a while to start training like 2,000 iterations and then turned it off.
	# "shaped_bonus_scale": 0.05,  # per-action bonus for play_length and scout_quality
	"shaped_bonus_scale": 0,  # per-action bonus for play_length and scout_quality
	"value_loss_coeff": 0.25,
	"gamma": 0.99,
	"gae_lambda": 0.95,
	"training_seats": 4,
	"opponent_pool_size": 10,
	"snapshot_interval": 30, # add to pool every N iterations
	"total_iterations": 1_000_000,
	"log_interval": 1,
	"save_interval": 1000,
	# "save_interval": 100,
	# "eval_interval": 30,
	"eval_interval": 5,
	"encoding_version": 2,
	# "encoding_version": 4,
	# Rollout-based advantage estimation (replaces GAE)
	"use_rollouts": True,
	"rollout_games": 10,  # real games per iteration (rollouts are per-state within these)
	"rollouts_per_state": 50,  # N rollout games from each decision point
	"use_direct_pg": False,  # vanilla policy gradient instead of PPO (forces 1 epoch)
	"save_dir": "v5_5",
	"eval_opponents": {
		# "random": "random", # magic word → uses RandomBot
		"v1_4": "v1_4/latest.pt",
		"v2_5": "v2_5/latest.pt",
		"v3_4": "v3_4/latest.pt",
		"v4_2": "v4_2/latest.pt",
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

def _smooth(vals, window):
	"""Centered moving average for chart smoothing."""
	smoothed = []
	half = window // 2
	for i in range(len(vals)):
		start = max(0, i - half)
		end = min(len(vals), i + half + 1)
		smoothed.append(sum(vals[start:end]) / (end - start))
	return smoothed

def _save_charts(metrics_history: dict, save_dir: str, eval_opponent_names: set[str] | None = None):
	"""Generate training charts PNG from accumulated metrics."""
	iters = metrics_history["iteration"]
	if len(iters) < 2:
		return
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

		# Row 1: Play behavior + reward
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
		plot_line(axes[1, 2], "reward", "Avg Reward (P0)",
			"Mean per-round reward for the training player. Positive = winning more than losing.", "#5dadec")
		plot_line(axes[1, 3], "reward_std", "Reward Std",
			"Std of per-round rewards. High = noisy signal, low = consistent outcomes.", "#74c0fc")

		# Row 2: Training losses + entropy
		plot_line(axes[2, 0], "policy_loss", "Policy Loss",
			"PPO clipped surrogate loss. Watch for instability (spikes or divergence).", "#ff6b6b")
		plot_line(axes[2, 1], "value_loss", "Value Loss",
			"MSE between predicted and actual returns. Should decrease as value function improves.", "#ffa552")
		plot_multi(axes[2, 2], [
			("entropy_action_type", "Action Type", "#69db7c"),
			("entropy_play_start", "Play Start", "#5dadec"),
			("entropy_play_end", "Play End", "#ffa552"),
			("entropy_scout_insert", "Scout Insert", "#ff6b6b"),
		], "Per-Head Entropy",
			"Entropy per head (steps with 2+ options only). Collapsing = premature convergence. Compare to floors in config.")
		plot_line(axes[2, 3], "entropy_floor_penalty", "Entropy Floor Penalty",
			"Quadratic penalty when head entropy drops below floor. >0 = floor active. 0 = heads above floor.", "#ff922b")

		# Row 3: PPO diagnostics
		plot_line(axes[3, 0], "value", "Value Prediction",
			"Mean value function output. Should track reward. Divergence = value function miscalibrated.", "#e0aaff")
		plot_line(axes[3, 1], "clip_fraction", "Clip Fraction",
			"Fraction of samples clipped by PPO. <0.01 typical with masked multi-head actions.", "#ff922b")
		plot_line(axes[3, 2], "approx_kl", "Approx KL Divergence",
			"How far policy moved from collection policy. >0.05 = aggressive updates, risk of instability.", "#74c0fc")
		plot_line(axes[3, 3], "explained_variance", "Explained Variance",
			"How well value function predicts returns. ~0 = useless baseline, ~1 = perfect.", "#69db7c")

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

def train(config: dict | None = None, profile_iters: int | None = None):
	cfg = {**PARAMS, **(config or {})}
	save_dir = cfg["save_dir"]
	os.makedirs(save_dir, exist_ok=True)
	ev = cfg.get("encoding_version", 1)
	if ev == 2:
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
		"play_pct": [], "scout_pct": [], "sns_pct": [],
		"steps_per_game": [],
		"avg_play_length": [], "reward_std": [],
		"play_len_1_pct": [], "play_len_2_pct": [], "play_len_3_pct": [],
		"play_len_4_pct": [], "play_len_5_pct": [], "play_len_6_pct": [],
		"play_len_7plus_pct": [],
		"eval_iteration": [],
		"scout_play_len": [],
	}
	start_iter = 1
	# Auto-resume if save dir has a checkpoint
	resume_path = os.path.join(save_dir, "latest.pt")
	if os.path.exists(resume_path):
		checkpoint = torch.load(resume_path, weights_only=False)
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
		print(f"Resumed from iteration {start_iter - 1}")
	pool = OpponentPool(max_size=cfg["opponent_pool_size"])
	if os.path.exists(resume_path):
		checkpoint = torch.load(resume_path, weights_only=False)
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
			ckpt = torch.load(path, weights_only=False)
			ckpt_cfg = ckpt.get("config", {})
			if "layer_sizes" in ckpt_cfg:
				ls = ckpt_cfg["layer_sizes"]
			else:
				ls = [ckpt_cfg.get("first_hidden_size", 256), ckpt_cfg.get("hidden_size", 128)]
			eval_ev = ckpt_cfg.get("encoding_version", 1)
			if eval_ev == 2:
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
	input_size = INPUT_SIZE_V2 if ev == 2 else INPUT_SIZE
	print(f"Training Scout bot: {cfg['num_players']} players, "
		  f"input_size={input_size}, encoding=v{ev}, layers={cfg['layer_sizes']}")
	print(f"Games/iter={cfg['games_per_iteration']}, "
		  f"PPO epochs={cfg['ppo_epochs']}")
	print(f"Output: {save_dir}/")
	profiler = None
	if profile_iters:
		from pyinstrument import Profiler
		profile_stop = start_iter + profile_iters
		profiler = Profiler()
		print(f"\nProfiling {profile_iters} iterations ({start_iter} to {profile_stop - 1})...")
		profiler.start()
	iteration = start_iter
	replay_buffer = deque(maxlen=cfg.get("replay_buffer_size", 1))
	try:
		for iteration in range(start_iter, cfg["total_iterations"] + 1):
			t0 = time.time()
			# Self-play: collect games and compute advantages
			network.eval()
			if cfg.get("use_rollouts"):
				iteration_records, advantages = play_games_with_rollouts(
					network, cfg.get("rollout_games", 40), cfg["num_players"],
					rollouts_per_state=cfg.get("rollouts_per_state", 10),
					training_seats=cfg["training_seats"])
				# Value targets: use rollout V estimates stored in record.value
				returns = [r.value for r in iteration_records]
				adv_std_val = 1.0  # already normalized
			else:
				iteration_records = play_games_batched(
					network, cfg["games_per_iteration"], cfg["num_players"],
					training_seats=cfg["training_seats"],
					opponent_pool=pool.versions or None,
					reward_distribution=cfg.get("reward_distribution", "terminal"),
					reward_mode=cfg.get("reward_mode", "game_score"),
					shaped_bonus_scale=cfg.get("shaped_bonus_scale", 0.0))
				advantages, returns, adv_std_val = compute_gae(
					iteration_records, gamma=cfg["gamma"], lam=cfg["gae_lambda"])
			play_time = time.time() - t0
			if any(math.isnan(r.value) for r in iteration_records):
				print(f"[iter {iteration:>5}] WARNING: NaN in forward pass, skipping update")
				continue
			# PPO training — on-policy, all steps from this iteration
			network.train()
			# LR annealing: linear decay to 0
			lr = cfg["learning_rate"] * (1 - iteration / cfg["total_iterations"])
			optimizer.param_groups[0]["lr"] = lr
			batch = prepare_ppo_batch(iteration_records, advantages, returns=returns)
			use_dpg = cfg.get("use_direct_pg", False)
			# Replay buffer: accumulate batches for PPO (not used with direct PG)
			if not use_dpg and batch is not None:
				replay_buffer.append(batch)
				training_batch = concatenate_batches(list(replay_buffer))
			else:
				training_batch = batch
			n_epochs = 1 if use_dpg else cfg["ppo_epochs"]
			ppo_sums = {}
			for epoch in range(n_epochs):
				if use_dpg:
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
					if epoch == 0 and len(replay_buffer) <= 1 and abs(m["mean_ratio"] - 1.0) > 0.01:
						print(f"  WARNING: epoch 0 mean ratio={m['mean_ratio']:.4f} (expected ~1.0)")
				for k, v in m.items():
					ppo_sums[k] = ppo_sums.get(k, 0.0) + v
			ppo_avg = {k: v / n_epochs for k, v in ppo_sums.items()}
			train_time = time.time() - t0 - play_time
			# Snapshot to opponent pool
			if iteration % cfg["snapshot_interval"] == 0:
				pool.add(network)
			# Logging + metrics + latest save
			if iteration % cfg["log_interval"] == 0:
				p0_records = [r for r in iteration_records if r.player == 0]
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
				# Behavioral metrics
				n_steps = len(iteration_records)
				n_play = sum(1 for r in iteration_records if r.action_type == 0)
				n_scout = sum(1 for r in iteration_records if 1 <= r.action_type <= 4)
				n_sns = sum(1 for r in iteration_records if 5 <= r.action_type <= 8)
				play_pct = n_play / max(n_steps, 1)
				scout_pct = n_scout / max(n_steps, 1)
				sns_pct = n_sns / max(n_steps, 1)
				num_games = cfg["rollout_games"] if cfg.get("use_rollouts") else cfg["games_per_iteration"]
				steps_per_game = n_steps / num_games
				play_lengths = [r.play_length for r in iteration_records if r.play_length is not None]
				avg_play_length = sum(play_lengths) / max(len(play_lengths), 1)
				n_plays = max(len(play_lengths), 1)
				play_len_counts = [0] * 8 # indices 1-7, 0 unused
				for l in play_lengths:
					play_len_counts[min(l, 7)] += 1
				play_len_pcts = [c / n_plays for c in play_len_counts]
				metrics_history["iteration"].append(iteration)
				metrics_history["reward"].append(avg_reward)
				metrics_history["value"].append(avg_value)
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
				buf_str = f"({training_batch['n']})" if training_batch and len(replay_buffer) > 1 else ""
				print(f"[iter {iteration:>5}] "
					  f"reward={avg_reward:+.3f}  value={avg_value:+.3f}  "
					  f"ploss={ppo_avg['policy_loss']:.4f}  vloss={ppo_avg['value_loss']:.4f}  "
					  f"ent={ppo_avg['entropy']:.3f}  clip={ppo_avg['clip_fraction']:.2f}  "
					  f"kl={ppo_avg['approx_kl']:.4f}  ev={ppo_avg['explained_variance']:.2f}  "
					  f"steps={n_steps}{buf_str}  pool={len(pool.versions)}  "
					  f"play={play_time:.1f}s  train={train_time:.1f}s")
				_save_checkpoint(network, optimizer, iteration, cfg, metrics_history, save_dir, "latest.pt", pool=pool)
			# Periodic snapshots
			if iteration % cfg["save_interval"] == 0:
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
				try:
					network.eval()
					n_eval = 40
					metrics_history["eval_iteration"].append(iteration)
					for name, eval_net in eval_opponents.items():
						total_margin = 0.0
						for _ in range(n_eval):
							nets = [network] + [eval_net for _ in range(cfg["num_players"] - 1)]
							scores = play_eval_game(nets, cfg["num_players"])
							# Compare vs mean opponent score (not max, which biases negative)
							mean_opponent = sum(scores[1:]) / len(scores[1:])
							total_margin += scores[0] - mean_opponent
						avg_margin = total_margin / n_eval
						metrics_history[f"eval_margin_{name}"].append(avg_margin)
						print(f"  Eval vs {name}: margin={avg_margin:+.1f}")
					# Scout placement quality
					scout_len, scout_n = eval_scout_quality(network, n_samples=200)
					metrics_history["scout_play_len"].append(scout_len)
					print(f"  Scout play_len: {scout_len:.2f} (n={scout_n})")
				except Exception as e:
					print(f"  WARNING: eval failed at iter {iteration}: {e}")
					# Remove the partial eval_iteration entry if metrics are incomplete
					expected_keys = [f"eval_margin_{n}" for n in eval_opponents]
					expected_keys.append("scout_play_len")
					if (metrics_history["eval_iteration"]
							and metrics_history["eval_iteration"][-1] == iteration):
						metrics_history["eval_iteration"].pop()
						for k in expected_keys:
							if k in metrics_history and len(metrics_history[k]) > len(metrics_history["eval_iteration"]):
								metrics_history[k].pop()
				_save_charts(metrics_history, save_dir, set(eval_opponents))
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
	_save_charts(metrics_history, save_dir, set(eval_opponents))
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
