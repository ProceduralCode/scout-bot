import torch
import argparse
import time
from encoding import INPUT_SIZE
from network import ScoutNetwork
from training import play_game, ReplayBuffer, OpponentPool, compute_advantages, ppo_update

# TODO: All hyperparameters are initial guesses — tune empirically
DEFAULT_CONFIG = {
	"num_players": 4,
	"hidden_size": 128,
	"first_hidden_size": 256,
	"learning_rate": 3e-4,
	"batch_size": 256,
	"games_per_iteration": 50,
	"ppo_epochs": 4, # passes over the batch per iteration
	"clip_epsilon": 0.2,
	"entropy_bonus": 0.01,
	"value_loss_coeff": 0.5,
	"replay_buffer_size": 500, # max games in buffer
	"opponent_pool_size": 10,
	"snapshot_interval": 10, # add to pool every N iterations
	"total_iterations": 1000,
	"log_interval": 10,
	"save_interval": 100,
	"save_path": "scout_model.pt",
}

def train(config: dict | None = None):
	cfg = {**DEFAULT_CONFIG, **(config or {})}
	network = ScoutNetwork(
		input_size=INPUT_SIZE,
		hidden_size=cfg["hidden_size"],
		first_hidden_size=cfg["first_hidden_size"],
	)
	optimizer = torch.optim.Adam(network.parameters(), lr=cfg["learning_rate"])
	replay = ReplayBuffer(max_games=cfg["replay_buffer_size"])
	pool = OpponentPool(max_size=cfg["opponent_pool_size"])
	# Seed the pool with the initial network
	pool.add(network)
	print(f"Training Scout bot: {cfg['num_players']} players, "
		  f"input_size={INPUT_SIZE}, hidden={cfg['hidden_size']}")
	print(f"Games/iter={cfg['games_per_iteration']}, "
		  f"PPO epochs={cfg['ppo_epochs']}, batch={cfg['batch_size']}")
	for iteration in range(1, cfg["total_iterations"] + 1):
		t0 = time.time()
		# Self-play: collect games
		network.eval()
		iteration_records = []
		for _ in range(cfg["games_per_iteration"]):
			opponents = pool.sample(cfg["num_players"] - 1) or None
			records = play_game(network, cfg["num_players"], opponent_pool=opponents)
			replay.add_game(records)
			iteration_records.extend(records)
		play_time = time.time() - t0
		# PPO training
		network.train()
		avg_policy_loss = 0.0
		avg_value_loss = 0.0
		avg_entropy = 0.0
		for epoch in range(cfg["ppo_epochs"]):
			steps = replay.sample_steps(cfg["batch_size"])
			advantages = compute_advantages(steps)
			pl, vl, ent = ppo_update(
				network, optimizer, steps, advantages,
				clip_epsilon=cfg["clip_epsilon"],
				entropy_bonus=cfg["entropy_bonus"],
				value_loss_coeff=cfg["value_loss_coeff"],
			)
			avg_policy_loss += pl
			avg_value_loss += vl
			avg_entropy += ent
		avg_policy_loss /= cfg["ppo_epochs"]
		avg_value_loss /= cfg["ppo_epochs"]
		avg_entropy /= cfg["ppo_epochs"]
		train_time = time.time() - t0 - play_time
		# Snapshot to opponent pool
		if iteration % cfg["snapshot_interval"] == 0:
			pool.add(network)
		# Logging
		if iteration % cfg["log_interval"] == 0:
			# Compute average reward for player 0 across this iteration's games
			p0_records = [r for r in iteration_records if r.player == 0]
			avg_reward = sum(r.reward for r in p0_records) / max(len(p0_records), 1)
			avg_value = sum(r.value for r in p0_records) / max(len(p0_records), 1)
			print(f"[iter {iteration:>5}] "
				  f"reward={avg_reward:+.3f}  value={avg_value:+.3f}  "
				  f"policy_loss={avg_policy_loss:.4f}  value_loss={avg_value_loss:.4f}  "
				  f"entropy={avg_entropy:.3f}  "
				  f"buffer={replay.total_steps()}  pool={len(pool.versions)}  "
				  f"play={play_time:.1f}s  train={train_time:.1f}s")
		# Save
		if iteration % cfg["save_interval"] == 0:
			torch.save({
				"model_state": network.state_dict(),
				"optimizer_state": optimizer.state_dict(),
				"iteration": iteration,
				"config": cfg,
			}, cfg["save_path"])
			print(f"  Saved to {cfg['save_path']}")
	# Final save
	torch.save({
		"model_state": network.state_dict(),
		"optimizer_state": optimizer.state_dict(),
		"iteration": cfg["total_iterations"],
		"config": cfg,
	}, cfg["save_path"])
	print(f"Training complete. Model saved to {cfg['save_path']}")

def main():
	parser = argparse.ArgumentParser(description="Train a Scout card game bot")
	parser.add_argument("--players", type=int, default=4, choices=[3, 4, 5])
	parser.add_argument("--iterations", type=int, default=1000)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--games-per-iter", type=int, default=50)
	parser.add_argument("--save-path", type=str, default="scout_model.pt")
	args = parser.parse_args()
	config = {
		"num_players": args.players,
		"total_iterations": args.iterations,
		"learning_rate": args.lr,
		"batch_size": args.batch_size,
		"games_per_iteration": args.games_per_iter,
		"save_path": args.save_path,
	}
	train(config)

if __name__ == "__main__":
	main()
