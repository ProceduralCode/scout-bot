"""Benchmark rollout_multi_action_v6 with different chunk_pairs values."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from training import play_games_q_v6, rollout_multi_action_v6, attach_snapshots
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

# Warmup Numba JIT
print("Warmup...")
warmup, warmup_replays = play_games_q_v6(network, 5, 4, training_seats=4, temperature=0.0, epsilon=0.05)
attach_snapshots(warmup, warmup_replays)
rollout_multi_action_v6(warmup, network, 4,
	rollout_actions_per_sample=3, rollout_actions_random_extra=1,
	rollouts_per_action=5, rollout_temperature=1.0, chunk_pairs=64)
torch.cuda.synchronize()

# Collect samples
print("Playing 100 games...")
samples_orig, game_replays = play_games_q_v6(network, 100, 4, training_seats=4,
	temperature=0.0, epsilon=0.05)
attach_snapshots(samples_orig, game_replays)
n_samples = len(samples_orig)
print(f"  {n_samples} samples")

ROLLOUT_ACTIONS = 10
ROLLOUT_EXTRA = 2
ROLLOUTS_PER = 30

# Estimate total pairs
total_pairs = sum(
	min(ROLLOUT_ACTIONS, int(s.action_mask.sum().item())) + ROLLOUT_EXTRA
	for s in samples_orig)
print(f"  ~{total_pairs} action pairs, ~{total_pairs * ROLLOUTS_PER} rollout games")

chunk_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]
# Also test "all at once"
chunk_sizes.append(total_pairs + 1000)

print(f"\n{'chunk_pairs':>12} {'B (games)':>12} {'grid':>8} {'chunks':>8} {'time':>8} {'games/s':>10}")
print("-" * 72)

for cp in chunk_sizes:
	# Reset samples
	for s in samples_orig:
		s.rolled_actions = None
		s.rollout_margins = None
		s.rollout_stds = None

	torch.cuda.synchronize()
	t0 = time.time()
	rollout_multi_action_v6(
		samples_orig, network, 4,
		rollout_actions_per_sample=ROLLOUT_ACTIONS,
		rollout_actions_random_extra=ROLLOUT_EXTRA,
		rollouts_per_action=ROLLOUTS_PER,
		rollout_temperature=1.0,
		chunk_pairs=cp,
	)
	torch.cuda.synchronize()
	elapsed = time.time() - t0

	b = min(cp, total_pairs) * ROLLOUTS_PER
	grid = (b + 255) // 256
	n_chunks = (total_pairs + cp - 1) // cp
	total_games = total_pairs * ROLLOUTS_PER
	label = f"{cp}" if cp <= 16384 else "all"
	print(f"{label:>12} {b:>12,} {grid:>8,} {n_chunks:>8} {elapsed:>7.1f}s {total_games / elapsed:>9,.0f}")
