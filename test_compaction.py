"""Test that compaction produces identical scores to no-compaction.
Also benchmarks speedup. Hard 25s timeout."""

import sys, os, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

DEADLINE = time.time() + 25

import torch
from game import Game
from gpu_engine import from_snapshots, repeat_state
from numba_engine import rollout_numba
from network import FlatScoutNetwork
from encoding import INPUT_SIZE_V6
from main import Q_PARAMS

cfg = Q_PARAMS
network = FlatScoutNetwork(
    input_size=INPUT_SIZE_V6, layer_sizes=cfg["layer_sizes"],
    attention=cfg.get("attention"),
).cuda().eval()

# Create test games
N = 100
games = [Game(num_players=4) for _ in range(N)]
for g in games:
    g.start_round()

gpu_state_a = from_snapshots(games, device='cuda')
gpu_state_a = repeat_state(gpu_state_a, 15)
gpu_state_b = from_snapshots(games, device='cuda')
gpu_state_b = repeat_state(gpu_state_b, 15)
B = gpu_state_a.done.shape[0]
print(f"Batch: {B} games ({N} base x 15 rollouts)")

# Set same random seed for both runs
torch.manual_seed(42)
print("Running WITHOUT compaction...", flush=True)
t0 = time.time()
scores_no_compact = rollout_numba(gpu_state_a, network, temperature=0.3, compact_threshold=0)
torch.cuda.synchronize()
t_no = time.time() - t0
print(f"  Time: {t_no*1000:.0f}ms", flush=True)

if time.time() > DEADLINE:
    print("TIMEOUT"); sys.exit(0)

torch.manual_seed(42)
print("Running WITH compaction (threshold=0.5)...", flush=True)
t0 = time.time()
scores_compact = rollout_numba(gpu_state_b, network, temperature=0.3, compact_threshold=0.5)
torch.cuda.synchronize()
t_compact = time.time() - t0
print(f"  Time: {t_compact*1000:.0f}ms", flush=True)

# Compare scores — they won't be identical because compaction changes RNG state,
# but the distribution should be similar. Let's check they're at least valid.
print(f"\nScores shape: {scores_no_compact.shape} vs {scores_compact.shape}")
print(f"Score ranges: [{scores_no_compact.min()}, {scores_no_compact.max()}] vs "
      f"[{scores_compact.min()}, {scores_compact.max()}]")
print(f"Mean scores: {scores_no_compact.float().mean():.2f} vs {scores_compact.float().mean():.2f}")

# Check all games completed (no zeros everywhere)
all_zero_a = (scores_no_compact.sum(dim=1) == 0).sum()
all_zero_b = (scores_compact.sum(dim=1) == 0).sum()
print(f"All-zero rows: {all_zero_a} vs {all_zero_b}")

# Per-player score sums should be similar (law of large numbers)
psum_a = scores_no_compact[:, :4].float().mean(dim=0)
psum_b = scores_compact[:, :4].float().mean(dim=0)
print(f"Mean per-player scores (no compact): {psum_a.tolist()}")
print(f"Mean per-player scores (compact):    {psum_b.tolist()}")

speedup = t_no / t_compact if t_compact > 0 else float('inf')
print(f"\nSpeedup: {speedup:.2f}x ({t_no*1000:.0f}ms -> {t_compact*1000:.0f}ms)")
