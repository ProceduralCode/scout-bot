"""Test optimized from_snapshots + compaction. Hard 25s timeout."""

import sys, os, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
import numpy as np
import random
from training import play_games_q_v6, attach_snapshots, _apply_action_to_game
from encoding import decode_flat_action, INPUT_SIZE_V6
from gpu_engine import from_snapshots, repeat_state
from numba_engine import rollout_numba
from network import FlatScoutNetwork
from main import Q_PARAMS

cfg = Q_PARAMS
network = FlatScoutNetwork(
    input_size=INPUT_SIZE_V6, layer_sizes=cfg["layer_sizes"],
    attention=cfg.get("attention"),
).cuda().eval()

# Play 2 games, get samples
samples, game_replays = play_games_q_v6(
    network, 2, cfg["num_players"],
    training_seats=cfg.get("training_seats", cfg["num_players"]),
    temperature=cfg["temperature"], epsilon=cfg["epsilon"],
)
attach_snapshots(samples, game_replays)
del game_replays

# Action selection
ra, re = cfg["rollout_actions_per_sample"], cfg["rollout_actions_random_extra"]
rpa, rt = cfg["rollouts_per_action"], cfg["rollout_temperature"]
for sample in samples:
    legal = np.where(sample.action_mask)[0]
    outputs = sample.network_outputs[legal]
    k = min(ra, len(legal))
    top_idx = legal[np.argsort(outputs)[-k:][::-1]]
    selected = set(top_idx.tolist())
    selected.add(sample.action_taken)
    remaining = [a for a in legal if a not in selected]
    n_extra = min(re, len(remaining))
    if n_extra > 0:
        selected.update(random.sample(remaining, n_extra))
    sample.rolled_actions = sorted(selected)

# Build all games for 1 chunk
chunk = []
for si, sample in enumerate(samples):
    for ai, action_idx in enumerate(sample.rolled_actions):
        chunk.append((si, ai, action_idx))
        if len(chunk) >= 512:
            break
    if len(chunk) >= 512:
        break

games = []
for si, ai, action_idx in chunk:
    sample = samples[si]
    g = sample.snapshot.clone()
    action = decode_flat_action(action_idx, sample.hand_offset)
    _apply_action_to_game(g, action)
    if g.phase.value < 3:
        games.append(g)

print(f"{len(games)} games needing rollout", flush=True)

# JIT warmup
gpu_w = from_snapshots(games[:10], device='cuda')
gpu_w = repeat_state(gpu_w, rpa)
_ = rollout_numba(gpu_w, network, temperature=rt)
torch.cuda.synchronize()
print("JIT warm\n", flush=True)

# Benchmark from_snapshots
times = []
for _ in range(5):
    t0 = time.time()
    gs = from_snapshots(games, device='cuda')
    torch.cuda.synchronize()
    times.append(time.time() - t0)
print(f"from_snapshots ({len(games)} games): {min(times)*1000:.0f}ms best, {np.mean(times)*1000:.0f}ms avg")

# Full pipeline benchmark
t0 = time.time()
gs = from_snapshots(games, device='cuda')
gs = repeat_state(gs, rpa)
scores = rollout_numba(gs, network, temperature=rt, compact_threshold=0.5)
torch.cuda.synchronize()
t_total = time.time() - t0
B = len(games) * rpa
print(f"\nFull pipeline ({len(games)} games x {rpa} = {B} batch):")
print(f"  Total: {t_total*1000:.0f}ms")
print(f"  Scores: mean={scores[:,:4].float().mean():.2f}, range=[{scores.min()},{scores.max()}]")

# Correctness: verify from_snapshots produces valid state
gs2 = from_snapshots(games[:5], device='cuda')
print(f"\n  State check: hands_show range [{gs2.hands_show.min()},{gs2.hands_show.max()}]")
print(f"  hand_len: {gs2.hand_len[:5].tolist()}")
print(f"  num_players: {gs2.num_players[:5].tolist()}")
print(f"  done: {gs2.done[:5].tolist()}")
