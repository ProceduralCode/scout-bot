"""Profile play_games_batched to find where time is spent.

Usage: python scout-bot/profile_batched.py [checkpoint_path]
Default: scout-bot/v3_4/latest.pt
"""
import sys
import torch
from pyinstrument import Profiler

from game import Game
from encoding import INPUT_SIZE
from network import ScoutNetwork
from training import play_games_batched, OpponentPool

def main():
	ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "scout-bot/v3_4/latest.pt"
	print(f"Loading {ckpt_path}...")
	checkpoint = torch.load(ckpt_path, weights_only=False)
	cfg = checkpoint.get("config", {})
	layer_sizes = cfg.get("layer_sizes", [512, 256, 256, 128, 128, 128])
	network = ScoutNetwork(input_size=INPUT_SIZE, layer_sizes=layer_sizes)
	network.load_state_dict(checkpoint["model_state"])
	network.eval()

	num_players = cfg.get("num_players", 4)
	num_games = cfg.get("games_per_iteration", 100)
	training_seats = cfg.get("training_seats", 4)

	pool = OpponentPool(max_size=3)
	pool.add(network)
	opponents = pool.sample(num_players - training_seats) or None

	print(f"Profiling: {num_games} games, {num_players} players, {training_seats} training seats")
	print(f"Network: layers={layer_sizes}\n")

	profiler = Profiler()
	profiler.start()
	records = play_games_batched(
		network, num_games, num_players,
		training_seats=training_seats,
		opponent_pool=opponents,
		reward_distribution=cfg.get("reward_distribution", "terminal"),
		reward_mode=cfg.get("reward_mode", "game_score"),
		shaped_bonus_scale=cfg.get("shaped_bonus_scale", 0.0),
	)
	profiler.stop()

	print(f"Steps collected: {len(records)}")
	print(profiler.output_text(unicode=False, color=False))

if __name__ == "__main__":
	main()
