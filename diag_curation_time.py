"""Time the curate_samples function."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from training import play_games_q_v6, curate_samples
from network import FlatScoutNetwork
from encoding import INPUT_SIZE_V6

network = FlatScoutNetwork(INPUT_SIZE_V6, [256, 128],
	encoding_version=6, attention={"dim": 32, "heads": 2, "layers": 1})
ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots", "q_v1", "latest.pt")
if os.path.exists(ckpt_path):
	ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
	network.load_state_dict(ckpt["model_state"])
network.cuda()
network.eval()

for mult in [5, 10]:
	print(f"\n--- curation_multiplier={mult} ---")
	games = 100 * mult
	t0 = time.time()
	samples = play_games_q_v6(network, games, 4, training_seats=4,
		temperature=0.0, epsilon=0.05)
	play_time = time.time() - t0
	print(f"  Play {games} games: {play_time:.1f}s  ({len(samples)} samples)")
	t1 = time.time()
	curated = curate_samples(samples, mult)
	curate_time = time.time() - t1
	print(f"  Curate {len(samples)} -> {len(curated)}: {curate_time:.1f}s")
	print(f"  Total (play + curate): {play_time + curate_time:.1f}s")
