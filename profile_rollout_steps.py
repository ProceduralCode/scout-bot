"""Profile per-step breakdown inside rollout_numba.

Reproduces the rollout loop with timing per phase.
Hard 25s timeout."""

import sys, os, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

DEADLINE = time.time() + 25

import torch
import numpy as np
from numba import cuda as nb_cuda
from game import Game
from training import play_games_q_v6, attach_snapshots, _apply_action_to_game
from encoding import decode_flat_action, INPUT_SIZE_V6
from gpu_engine import from_snapshots as gpu_from_snapshots, repeat_state
from numba_engine import (
    rollout_numba, compute_legal_plays_kernel, compute_action_masks_kernel,
    encode_states_kernel, apply_actions_kernel, _grid, TPB, H, FLAT_ACTION_SIZE
)
from network import FlatScoutNetwork, batched_masked_sample
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

# Select actions, build pairs
ra, re = cfg["rollout_actions_per_sample"], cfg["rollout_actions_random_extra"]
rpa = cfg["rollouts_per_action"]
rt = cfg["rollout_temperature"]
import random
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

# Build games for 1 chunk
chunk_pairs = []
for si, sample in enumerate(samples):
    for ai, action_idx in enumerate(sample.rolled_actions):
        chunk_pairs.append((si, ai, action_idx))
        if len(chunk_pairs) >= 512:
            break
    if len(chunk_pairs) >= 512:
        break

games = []
for si, ai, action_idx in chunk_pairs:
    sample = samples[si]
    g = sample.snapshot.clone()
    action = decode_flat_action(action_idx, sample.hand_offset)
    _apply_action_to_game(g, action)
    if g.phase.value < 3:
        games.append(g)

# Warm up JIT
print(f"Warming up JIT with {len(games)} games x {rpa} rollouts...", flush=True)
gpu_state = gpu_from_snapshots(games, device='cuda')
gpu_state = repeat_state(gpu_state, rpa)
_ = rollout_numba(gpu_state, network, temperature=rt)
torch.cuda.synchronize()
print("JIT warm, now profiling...", flush=True)

if time.time() > DEADLINE:
    print("TIMEOUT after JIT warmup")
    sys.exit(0)

# Now profile step-by-step
gpu_state = gpu_from_snapshots(games, device='cuda')
gpu_state = repeat_state(gpu_state, rpa)
B = gpu_state.done.shape[0]
dev = gpu_state.done.device
print(f"Batch size: {B}")

legal_buf = torch.zeros(B, H, H, dtype=torch.bool, device=dev)
mask_buf = torch.zeros(B, FLAT_ACTION_SIZE, dtype=torch.bool, device=dev)
encode_buf = torch.zeros(B, 309, dtype=torch.float32, device=dev)
logits = torch.zeros(B, FLAT_ACTION_SIZE, device=dev)

grid = _grid(B)
step_data = []  # (n_active, t_kernel, t_forward, t_sample_apply)

network.eval()
with torch.no_grad():
    step = 0
    while True:
        active_idx = torch.where(~gpu_state.done)[0]
        n_active = active_idx.shape[0]
        if n_active == 0:
            break
        step += 1

        hand_offsets = torch.randint(0, H, (B,), device=dev, dtype=torch.long)
        d_offsets = nb_cuda.as_cuda_array(hand_offsets)

        # --- Kernels ---
        torch.cuda.synchronize()
        t0 = time.time()
        compute_legal_plays_kernel[grid, TPB](
            nb_cuda.as_cuda_array(gpu_state.hands_show),
            nb_cuda.as_cuda_array(gpu_state.hand_len),
            nb_cuda.as_cuda_array(gpu_state.current_player),
            nb_cuda.as_cuda_array(gpu_state.play_len),
            nb_cuda.as_cuda_array(gpu_state.play_type),
            nb_cuda.as_cuda_array(gpu_state.play_strength),
            nb_cuda.as_cuda_array(gpu_state.done),
            nb_cuda.as_cuda_array(legal_buf), B,
        )
        compute_action_masks_kernel[grid, TPB](
            nb_cuda.as_cuda_array(gpu_state.hands_show),
            nb_cuda.as_cuda_array(gpu_state.hand_len),
            nb_cuda.as_cuda_array(gpu_state.current_player),
            nb_cuda.as_cuda_array(gpu_state.play_show),
            nb_cuda.as_cuda_array(gpu_state.play_hide),
            nb_cuda.as_cuda_array(gpu_state.play_len),
            nb_cuda.as_cuda_array(gpu_state.play_type),
            nb_cuda.as_cuda_array(gpu_state.phase),
            nb_cuda.as_cuda_array(gpu_state.sns_available),
            nb_cuda.as_cuda_array(gpu_state.num_players),
            nb_cuda.as_cuda_array(legal_buf), d_offsets,
            nb_cuda.as_cuda_array(mask_buf), B,
        )
        encode_states_kernel[grid, TPB](
            nb_cuda.as_cuda_array(gpu_state.hands_show),
            nb_cuda.as_cuda_array(gpu_state.hands_hide),
            nb_cuda.as_cuda_array(gpu_state.hand_len),
            nb_cuda.as_cuda_array(gpu_state.current_player),
            nb_cuda.as_cuda_array(gpu_state.play_show),
            nb_cuda.as_cuda_array(gpu_state.play_hide),
            nb_cuda.as_cuda_array(gpu_state.play_len),
            nb_cuda.as_cuda_array(gpu_state.play_owner),
            nb_cuda.as_cuda_array(gpu_state.play_type),
            nb_cuda.as_cuda_array(gpu_state.play_strength),
            nb_cuda.as_cuda_array(gpu_state.phase),
            nb_cuda.as_cuda_array(gpu_state.scouts_since_play),
            nb_cuda.as_cuda_array(gpu_state.sns_available),
            nb_cuda.as_cuda_array(gpu_state.num_players),
            nb_cuda.as_cuda_array(gpu_state.collected),
            nb_cuda.as_cuda_array(gpu_state.scout_tokens),
            d_offsets, nb_cuda.as_cuda_array(encode_buf), B,
        )
        torch.cuda.synchronize()
        t_kernel = time.time() - t0

        # --- Forward ---
        t0 = time.time()
        CHUNK = 1024
        if n_active <= CHUNK:
            h = network(encode_buf[active_idx])
            logits[active_idx] = network.policy_logits(h)
        else:
            for start in range(0, n_active, CHUNK):
                end = min(start + CHUNK, n_active)
                idx = active_idx[start:end]
                h_chunk = network(encode_buf[idx])
                logits[idx] = network.policy_logits(h_chunk)
        torch.cuda.synchronize()
        t_forward = time.time() - t0

        # --- Sample + apply ---
        t0 = time.time()
        has_action = mask_buf.any(dim=1)
        no_action = (~gpu_state.done) & ~has_action
        if no_action.any():
            adv_cp = ((gpu_state.current_player.long() + 1) %
                gpu_state.num_players.long()).to(torch.int8)
            gpu_state.current_player = torch.where(
                no_action, adv_cp, gpu_state.current_player)

        if rt == 0.0:
            actions = logits.masked_fill(~mask_buf, float('-inf')).argmax(dim=1)
        elif rt != 1.0:
            actions = batched_masked_sample(logits / rt, mask_buf)
        else:
            actions = batched_masked_sample(logits, mask_buf)

        apply_active = (~gpu_state.done) & has_action
        apply_actions_kernel[grid, TPB](
            nb_cuda.as_cuda_array(gpu_state.hands_show),
            nb_cuda.as_cuda_array(gpu_state.hands_hide),
            nb_cuda.as_cuda_array(gpu_state.hand_len),
            nb_cuda.as_cuda_array(gpu_state.play_show),
            nb_cuda.as_cuda_array(gpu_state.play_hide),
            nb_cuda.as_cuda_array(gpu_state.play_len),
            nb_cuda.as_cuda_array(gpu_state.play_owner),
            nb_cuda.as_cuda_array(gpu_state.play_type),
            nb_cuda.as_cuda_array(gpu_state.play_strength),
            nb_cuda.as_cuda_array(gpu_state.current_player),
            nb_cuda.as_cuda_array(gpu_state.phase),
            nb_cuda.as_cuda_array(gpu_state.scouts_since_play),
            nb_cuda.as_cuda_array(gpu_state.sns_available),
            nb_cuda.as_cuda_array(gpu_state.num_players),
            nb_cuda.as_cuda_array(gpu_state.collected),
            nb_cuda.as_cuda_array(gpu_state.scout_tokens),
            nb_cuda.as_cuda_array(gpu_state.round_ender),
            nb_cuda.as_cuda_array(gpu_state.done),
            nb_cuda.as_cuda_array(actions), d_offsets,
            nb_cuda.as_cuda_array(apply_active), B,
        )
        torch.cuda.synchronize()
        t_sa = time.time() - t0

        step_data.append((n_active, t_kernel, t_forward, t_sa))

        if time.time() > DEADLINE:
            print(f"TIMEOUT at step {step}")
            break

# Summary
tot_k = sum(d[1] for d in step_data)
tot_f = sum(d[2] for d in step_data)
tot_s = sum(d[3] for d in step_data)
tot = tot_k + tot_f + tot_s
print(f"\n--- {len(step_data)} steps, {B} batch ---")
print(f"  Kernels:       {tot_k*1000:6.0f}ms  ({tot_k/tot*100:4.1f}%)  avg {tot_k/len(step_data)*1000:.2f}ms/step")
print(f"  Forward:       {tot_f*1000:6.0f}ms  ({tot_f/tot*100:4.1f}%)  avg {tot_f/len(step_data)*1000:.2f}ms/step")
print(f"  Sample+apply:  {tot_s*1000:6.0f}ms  ({tot_s/tot*100:4.1f}%)  avg {tot_s/len(step_data)*1000:.2f}ms/step")
print(f"  Total:         {tot*1000:6.0f}ms")

# Show first/last 5 steps
print(f"\n  Step breakdown (n_active, kernel_ms, fwd_ms, sa_ms):")
for i, (na, tk, tf, ts) in enumerate(step_data[:5]):
    print(f"    step {i+1:3d}: active={na:5d}  kern={tk*1000:5.1f}  fwd={tf*1000:5.1f}  sa={ts*1000:5.1f}")
if len(step_data) > 10:
    print(f"    ...")
    for i in range(max(5, len(step_data)-5), len(step_data)):
        na, tk, tf, ts = step_data[i]
        print(f"    step {i+1:3d}: active={na:5d}  kern={tk*1000:5.1f}  fwd={tf*1000:5.1f}  sa={ts*1000:5.1f}")

# Forward passes per step
fwd_per_step = [((d[0] + 1023) // 1024) for d in step_data]
print(f"\n  Avg forward chunks/step: {np.mean(fwd_per_step):.1f}")
print(f"  Total forward passes: {sum(fwd_per_step)}")
