import torch
import argparse
import time
import os
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from encoding import INPUT_SIZE
from network import ScoutNetwork, RandomBot
from training import play_game, play_eval_game, OpponentPool, compute_gae, ppo_update
from game_log import GameLog

# TODO: All hyperparameters are initial guesses — tune empirically
DEFAULT_CONFIG = {
	"num_players": 4,
	"layer_sizes": [512, 256, 256, 128, 128, 128],
	# "layer_sizes": [256, 128],  # old shallow network
	"learning_rate": 3e-4,
	# "batch_size": 256,  # TODO: implement mini-batching within PPO epochs
	"games_per_iteration": 200,
	"ppo_epochs": 4, # passes over the batch per iteration
	"clip_epsilon": 0.2,
	"entropy_bonus": 0.01,
	# "entropy_bonus": 0.05,
	# "entropy_bonus": 0.25,
	"value_loss_coeff": 0.25,
	"gamma": 0.99,
	"gae_lambda": 0.95,
	"training_seats": 3,
	"opponent_pool_size": 10,
	"snapshot_interval": 35, # add to pool every N iterations
	"total_iterations": 1_000_000,
	"log_interval": 1,
	"save_interval": 100,
	"eval_interval": 8,
	"save_dir": "v2_4",
	"eval_opponents": {
		"v1_3": "v1_3/latest.pt",
		"v1_4": "v1_4/latest.pt",
		"v2_2": "v2_2/latest.pt",
		"v2_3": "v2_3/latest.pt",
	}, # name → checkpoint path
}

def _save_checkpoint(network, optimizer, iteration, cfg, metrics_history, save_dir, filename, pool=None, extra=None):
	path = os.path.join(save_dir, filename)
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
	torch.save(data, path)
	return path

def _smooth(vals, window):
	"""Centered moving average for chart smoothing."""
	smoothed = []
	half = window // 2
	for i in range(len(vals)):
		start = max(0, i - half)
		end = min(len(vals), i + half + 1)
		smoothed.append(sum(vals[start:end]) / (end - start))
	return smoothed

def _save_charts(metrics_history: dict, save_dir: str):
	"""Generate training charts PNG from accumulated metrics."""
	iters = metrics_history["iteration"]
	if len(iters) < 2:
		return
	chart_path = os.path.join(save_dir, "charts.png")
	BG = "#1a1a2e"
	PANEL = "#16213e"
	TEXT = "#e0e0e0"
	SUBTEXT = "#a0a0a0"
	GRID = "#ffffff"

	with plt.style.context("dark_background"):
		fig, axes = plt.subplots(4, 3, figsize=(14, 20))
		fig.patch.set_facecolor(BG)
		fig.suptitle("Scout Bot Training", fontsize=16, color=TEXT, y=0.98)

		def plot_line(ax, key, title, desc, color):
			vals = metrics_history.get(key, [])
			ax.set_facecolor(PANEL)
			if vals:
				ax.plot(iters[:len(vals)], vals, alpha=0.25, color=color, linewidth=0.8)
				window = max(len(vals) // 10, 10)
				if len(vals) >= window:
					ax.plot(iters[:len(vals)], _smooth(vals, window), color=color, linewidth=2)
			ax.set_title(title, color=TEXT, fontsize=11)
			ax.text(0.5, -0.15, textwrap.fill(desc, 50), transform=ax.transAxes,
					ha="center", va="top", fontsize=7, color=SUBTEXT, style="italic")
			ax.tick_params(colors=SUBTEXT, labelsize=8)
			ax.grid(True, alpha=0.15, color=GRID)

		def plot_multi(ax, series, title, desc, ylim=None):
			ax.set_facecolor(PANEL)
			for key, label, color in series:
				vals = metrics_history.get(key, [])
				if vals:
					window = max(len(vals) // 10, 10)
					if len(vals) >= window:
						ax.plot(iters[:len(vals)], _smooth(vals, window),
								color=color, linewidth=1.5, label=label)
					else:
						ax.plot(iters[:len(vals)], vals,
								color=color, linewidth=1, alpha=0.5, label=label)
			if ylim:
				ax.set_ylim(*ylim)
			ax.legend(fontsize=7, loc="upper right")
			ax.set_title(title, color=TEXT, fontsize=11)
			ax.text(0.5, -0.15, textwrap.fill(desc, 50), transform=ax.transAxes,
					ha="center", va="top", fontsize=7, color=SUBTEXT, style="italic")
			ax.tick_params(colors=SUBTEXT, labelsize=8)
			ax.grid(True, alpha=0.15, color=GRID)

		# Row 0: Core performance
		plot_line(axes[0, 0], "reward", "Avg Reward (P0)",
			"Mean reward per step for the training player. Positive = winning more than losing.", "#5dadec")
		plot_line(axes[0, 1], "value", "Value Prediction",
			"Mean value function output. Should track reward. Divergence = value function miscalibrated.", "#e0aaff")
		# Eval chart (special handling for different x-axis, multiple opponents)
		ax_eval = axes[0, 2]
		ax_eval.set_facecolor(PANEL)
		eval_iters = metrics_history.get("eval_iteration", [])
		eval_margin = metrics_history.get("eval_margin", [])
		if eval_iters and eval_margin:
			ax_eval.plot(eval_iters, eval_margin, color="#b197fc",
						linewidth=2, marker="o", markersize=3, label="vs Random")
		opponent_colors = ["#ff6b6b", "#69db7c", "#ffa552", "#5dadec"]
		opponent_keys = sorted(k for k in metrics_history if k.startswith("eval_margin_"))
		for i, key in enumerate(opponent_keys):
			name = key[len("eval_margin_"):]
			vals = metrics_history.get(key, [])
			if vals:
				ax_eval.plot(eval_iters[:len(vals)], vals,
							color=opponent_colors[i % len(opponent_colors)],
							linewidth=2, marker="o", markersize=3, label=f"vs {name}")
		ax_eval.axhline(y=0, color="#666666", linestyle="--", alpha=0.5)
		if ax_eval.get_legend_handles_labels()[1]:
			ax_eval.legend(fontsize=7, loc="upper left")
		ax_eval.set_title("Score Margin", color=TEXT, fontsize=11)
		ax_eval.text(0.5, -0.15,
			textwrap.fill("P0 score minus best opponent, averaged over eval games. Positive = winning.", 50),
			transform=ax_eval.transAxes, ha="center", va="top", fontsize=7, color=SUBTEXT, style="italic")
		ax_eval.tick_params(colors=SUBTEXT, labelsize=8)
		ax_eval.grid(True, alpha=0.15, color=GRID)

		# Row 1: Loss & entropy
		plot_line(axes[1, 0], "policy_loss", "Policy Loss",
			"PPO clipped surrogate loss. Not directly interpretable; watch for instability (spikes or divergence).", "#ff6b6b")
		plot_line(axes[1, 1], "value_loss", "Value Loss",
			"MSE between predicted and actual returns. Should decrease as value function improves.", "#ffa552")
		plot_multi(axes[1, 2], [
			("entropy_action_type", "Action Type", "#69db7c"),
			("entropy_play_start", "Play Start", "#5dadec"),
			("entropy_play_end", "Play End", "#ffa552"),
			("entropy_scout_insert", "Scout Insert", "#ff6b6b"),
		], "Per-Head Entropy",
			"Entropy per decision head. High = uncertain, low = confident. Collapsing entropy may signal premature convergence.")

		# Row 2: PPO health
		plot_line(axes[2, 0], "clip_fraction", "Clip Fraction",
			"Fraction of samples clipped by PPO. Values <0.01 are typical with masked multi-head actions.", "#ff922b")
		plot_line(axes[2, 1], "approx_kl", "Approx KL Divergence",
			"How far the policy moved from the collection policy. >0.05 = aggressive updates, risk of instability.", "#74c0fc")
		plot_line(axes[2, 2], "explained_variance", "Explained Variance",
			"How well the value function predicts returns. ~0 = useless baseline, ~1 = perfect predictions.", "#69db7c")

		# Row 3: Behavioral
		plot_multi(axes[3, 0], [
			("play_pct", "Play", "#69db7c"),
			("scout_pct", "Scout", "#5dadec"),
			("sns_pct", "S&S", "#ff6b6b"),
		], "Action Type Distribution",
			"Fraction of each action type chosen per iteration. Shows how strategy evolves over training.",
			ylim=(0, 1))
		plot_line(axes[3, 1], "steps_per_game", "Steps Per Game",
			"Average decisions per game for the training player. Shorter games may indicate more decisive play.", "#e0aaff")
		plot_line(axes[3, 2], "advantage_std", "Advantage Std (pre-norm)",
			"Std of raw advantages before normalization. <0.01 = all actions look equally good, weak learning signal.", "#ffd43b")

		fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.03,
						   hspace=0.40, wspace=0.25)
		fig.savefig(chart_path, dpi=100, facecolor=fig.get_facecolor(),
					bbox_inches='tight', pad_inches=0.15)
		plt.close(fig)

def train(config: dict | None = None):
	cfg = {**DEFAULT_CONFIG, **(config or {})}
	save_dir = cfg["save_dir"]
	os.makedirs(save_dir, exist_ok=True)
	network = ScoutNetwork(input_size=INPUT_SIZE, layer_sizes=cfg["layer_sizes"])
	optimizer = torch.optim.Adam(network.parameters(), lr=cfg["learning_rate"])
	metrics_history = {
		"iteration": [], "reward": [], "value": [],
		"policy_loss": [], "value_loss": [], "entropy": [],
		"clip_fraction": [], "approx_kl": [], "explained_variance": [],
		"entropy_action_type": [], "entropy_play_start": [],
		"entropy_play_end": [], "entropy_scout_insert": [],
		"play_pct": [], "scout_pct": [], "sns_pct": [],
		"steps_per_game": [], "advantage_std": [],
		"eval_iteration": [], "eval_margin": [],
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
			# Merge saved metrics, keeping new keys with empty defaults
			for k, v in checkpoint["metrics_history"].items():
				metrics_history[k] = v
		saved_cfg = checkpoint.get("config", {})
		# Backward compat: convert old hidden_size/first_hidden_size to layer_sizes
		if "layer_sizes" not in saved_cfg and "hidden_size" in saved_cfg:
			saved_cfg["layer_sizes"] = [
				saved_cfg.get("first_hidden_size", 256),
				saved_cfg["hidden_size"],
			]
		cfg = {**DEFAULT_CONFIG, **saved_cfg, **(config or {})}
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
		ckpt = torch.load(path, weights_only=False)
		ckpt_cfg = ckpt.get("config", {})
		if "layer_sizes" in ckpt_cfg:
			ls = ckpt_cfg["layer_sizes"]
		else:
			ls = [ckpt_cfg.get("first_hidden_size", 256), ckpt_cfg.get("hidden_size", 128)]
		eval_net = ScoutNetwork(layer_sizes=ls)
		eval_net.load_state_dict(ckpt["model_state"])
		eval_net.eval()
		eval_opponents[name] = eval_net
		key = f"eval_margin_{name}"
		if key not in metrics_history:
			metrics_history[key] = []
		print(f"  Loaded eval opponent '{name}' from {path}")
	print(f"Training Scout bot: {cfg['num_players']} players, "
		  f"input_size={INPUT_SIZE}, layers={cfg['layer_sizes']}")
	print(f"Games/iter={cfg['games_per_iteration']}, "
		  f"PPO epochs={cfg['ppo_epochs']}")
	print(f"Output: {save_dir}/")
	iteration = start_iter
	stopped = False
	try:
		for iteration in range(start_iter, cfg["total_iterations"] + 1):
			t0 = time.time()
			# Self-play: collect games
			network.eval()
			iteration_records = []
			for game_idx in range(cfg["games_per_iteration"]):
				opponents = pool.sample(cfg["num_players"] - cfg["training_seats"]) or None
				records = play_game(network, cfg["num_players"], opponent_pool=opponents,
									training_seats=cfg["training_seats"])
				for r in records:
					r.game_id = game_idx
				iteration_records.extend(records)
			play_time = time.time() - t0
			# PPO training — on-policy, all steps from this iteration
			network.train()
			advantages, returns, adv_std_val = compute_gae(
				iteration_records, gamma=cfg["gamma"], lam=cfg["gae_lambda"])
			# LR annealing: linear decay to 0
			lr = cfg["learning_rate"] * (1 - iteration / cfg["total_iterations"])
			optimizer.param_groups[0]["lr"] = lr
			ppo_sums = {}
			for epoch in range(cfg["ppo_epochs"]):
				m = ppo_update(
					network, optimizer, iteration_records, advantages,
					clip_epsilon=cfg["clip_epsilon"],
					entropy_bonus=cfg["entropy_bonus"],
					value_loss_coeff=cfg["value_loss_coeff"],
					returns=returns,
				)
				# Epoch 0: ratios must be ~1.0 (policy hasn't changed yet)
				if epoch == 0 and abs(m["mean_ratio"] - 1.0) > 0.01:
					print(f"  WARNING: epoch 0 mean ratio={m['mean_ratio']:.4f} (expected ~1.0)")
				for k, v in m.items():
					ppo_sums[k] = ppo_sums.get(k, 0.0) + v
			n_epochs = cfg["ppo_epochs"]
			ppo_avg = {k: v / n_epochs for k, v in ppo_sums.items()}
			train_time = time.time() - t0 - play_time
			# Snapshot to opponent pool
			if iteration % cfg["snapshot_interval"] == 0:
				pool.add(network)
			# Logging + metrics + latest save
			if iteration % cfg["log_interval"] == 0:
				p0_records = [r for r in iteration_records if r.player == 0]
				# Per-round reward (only non-zero steps are round-end rewards)
				p0_rewards = [r.reward for r in p0_records if r.reward != 0.0]
				avg_reward = sum(p0_rewards) / max(len(p0_rewards), 1)
				avg_value = sum(r.value for r in p0_records) / max(len(p0_records), 1)
				# Behavioral metrics
				n_steps = len(iteration_records)
				n_play = sum(1 for r in iteration_records if r.action_type == 0)
				n_scout = sum(1 for r in iteration_records if 1 <= r.action_type <= 4)
				n_sns = sum(1 for r in iteration_records if 5 <= r.action_type <= 8)
				play_pct = n_play / max(n_steps, 1)
				scout_pct = n_scout / max(n_steps, 1)
				sns_pct = n_sns / max(n_steps, 1)
				steps_per_game = n_steps / cfg["games_per_iteration"]
				adv_std = adv_std_val
				metrics_history["iteration"].append(iteration)
				metrics_history["reward"].append(avg_reward)
				metrics_history["value"].append(avg_value)
				for k in ("policy_loss", "value_loss", "entropy",
						  "clip_fraction", "approx_kl", "explained_variance",
						  "entropy_action_type", "entropy_play_start",
						  "entropy_play_end", "entropy_scout_insert"):
					metrics_history[k].append(ppo_avg[k])
				metrics_history["play_pct"].append(play_pct)
				metrics_history["scout_pct"].append(scout_pct)
				metrics_history["sns_pct"].append(sns_pct)
				metrics_history["steps_per_game"].append(steps_per_game)
				metrics_history["advantage_std"].append(adv_std)
				print(f"[iter {iteration:>5}] "
					  f"reward={avg_reward:+.3f}  value={avg_value:+.3f}  "
					  f"ploss={ppo_avg['policy_loss']:.4f}  vloss={ppo_avg['value_loss']:.4f}  "
					  f"ent={ppo_avg['entropy']:.3f}  clip={ppo_avg['clip_fraction']:.2f}  "
					  f"kl={ppo_avg['approx_kl']:.4f}  ev={ppo_avg['explained_variance']:.2f}  "
					  f"steps={n_steps}  pool={len(pool.versions)}  "
					  f"play={play_time:.1f}s  train={train_time:.1f}s")
				_save_checkpoint(network, optimizer, iteration, cfg, metrics_history, save_dir, "latest.pt", pool=pool)
				_save_charts(metrics_history, save_dir)
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
				log.save(os.path.join(save_dir, f"iter_{iteration}_game.json"))
				print(f"  Saved snapshot + game log (iter {iteration})")
			# Eval vs random + named opponents
			if iteration % cfg["eval_interval"] == 0:
				network.eval()
				n_eval = 40
				total_margin = 0.0
				for _ in range(n_eval):
					nets = [network] + [RandomBot() for _ in range(cfg["num_players"] - 1)]
					scores = play_eval_game(nets, cfg["num_players"])
					# Compare vs mean opponent score (not max, which biases negative)
					mean_opponent = sum(scores[1:]) / len(scores[1:])
					total_margin += scores[0] - mean_opponent
				avg_margin = total_margin / n_eval
				metrics_history["eval_iteration"].append(iteration)
				metrics_history["eval_margin"].append(avg_margin)
				print(f"  Eval vs random: margin={avg_margin:+.1f}")
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
				_save_charts(metrics_history, save_dir)
	except KeyboardInterrupt:
		stopped = True
		print(f"\nInterrupted at iteration {iteration}. Saving...")
	# Final save
	_save_checkpoint(network, optimizer, iteration, cfg, metrics_history, save_dir, "latest.pt", pool=pool)
	_save_charts(metrics_history, save_dir)
	if stopped:
		print(f"Saved to {save_dir}/latest.pt")
	else:
		print(f"Training complete. Saved to {save_dir}/latest.pt")

def main():
	parser = argparse.ArgumentParser(description="Train a Scout card game bot")
	parser.add_argument("--players", type=int, default=None, choices=[3, 4, 5])
	parser.add_argument("--iterations", type=int, default=None)
	parser.add_argument("--lr", type=float, default=None)
	parser.add_argument("--batch-size", type=int, default=None)
	parser.add_argument("--games-per-iter", type=int, default=None)
	parser.add_argument("--entropy-bonus", type=float, default=None)
	parser.add_argument("--save-dir", type=str, default=None)
	parser.add_argument("--replay", type=str, default=None,
		help="Path to a game log JSON file to replay")
	parser.add_argument("--match", nargs="+", metavar="AGENT",
		help="Play agents against each other (e.g., random path/to/model.pt)")
	parser.add_argument("--games", type=int, default=100,
		help="Number of games for --match (default: 100)")
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
		"total_iterations": args.iterations,
		"learning_rate": args.lr,
		"batch_size": args.batch_size,
		"games_per_iteration": args.games_per_iter,
		"entropy_bonus": args.entropy_bonus,
		"save_dir": args.save_dir,
	}
	train({k: v for k, v in config.items() if v is not None})

if __name__ == "__main__":
	main()
