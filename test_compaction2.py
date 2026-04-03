"""Test compaction speedup at production batch size (~7000 games).
Hard 25s timeout."""

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

# Create ~500 base games (x15 rollouts = 7500, matching production)
N = 500
print(f"Creating {N} games...", flush=True)
games = [Game(num_players=4) for _ in range(N)]
for g in games:
    g.start_round()
active_games = games  # freshly started rounds are fine
print(f"  {len(active_games)} games", flush=True)

# JIT warmup with small batch
small = from_snapshots(active_games[:10], device='cuda')
small = repeat_state(small, 15)
_ = rollout_numba(small, network, temperature=0.3)
torch.cuda.synchronize()
print("JIT warm", flush=True)

if time.time() > DEADLINE:
    print("TIMEOUT"); sys.exit(0)

# Production-size batch
gpu_a = from_snapshots(active_games, device='cuda')
gpu_a = repeat_state(gpu_a, 15)
gpu_b = from_snapshots(active_games, device='cuda')
gpu_b = repeat_state(gpu_b, 15)
B = gpu_a.done.shape[0]
print(f"\nBatch: {B} games", flush=True)

# Without compaction
print("No compaction...", flush=True)
t0 = time.time()
s_a = rollout_numba(gpu_a, network, temperature=0.3, compact_threshold=0)
torch.cuda.synchronize()
t_no = time.time() - t0
print(f"  {t_no*1000:.0f}ms", flush=True)

if time.time() > DEADLINE:
    print("TIMEOUT"); sys.exit(0)

# With compaction
print("With compaction (0.5)...", flush=True)
t0 = time.time()
s_b = rollout_numba(gpu_b, network, temperature=0.3, compact_threshold=0.5)
torch.cuda.synchronize()
t_compact = time.time() - t0
print(f"  {t_compact*1000:.0f}ms", flush=True)

# Scores won't match exactly (RNG differs) but should be valid
print(f"\nScore ranges: [{s_a.min()},{s_a.max()}] vs [{s_b.min()},{s_b.max()}]")
print(f"Means: {s_a[:,:4].float().mean():.2f} vs {s_b[:,:4].float().mean():.2f}")
speedup = t_no / t_compact if t_compact > 0 else float('inf')
print(f"Speedup: {speedup:.2f}x ({t_no*1000:.0f}ms -> {t_compact*1000:.0f}ms)")
