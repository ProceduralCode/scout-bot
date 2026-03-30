"""Profile Numba CUDA rollout — breakdown of time per phase within the rollout loop.

Two approaches:
1. Deferred events: record CUDA events without syncing, read all timings after loop.
   Preserves natural GPU pipelining. Measures GPU-side time per phase.
2. Wall-time bracket: measures total wall time including all Python overhead.
   The gap between sum-of-events and wall time = Python dispatch overhead.

Also measures from_snapshots (CPU->GPU transfer) and compute_scores.
"""
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import random
import time
import glob
import torch
from numba import cuda as numba_cuda

from game import Game, Phase
from encoding import get_legal_plays
from gpu_engine import GpuGameState, from_snapshots, compute_scores
from numba_engine import (
    rollout_numba, compute_legal_plays_kernel, compute_action_masks_kernel,
    encode_states_kernel, apply_actions_kernel,
    H, FLAT_ACTION_SIZE, MAX_STEPS, TPB, _grid,
)
from network import FlatScoutNetwork, batched_masked_sample

DEVICE = 'cuda'

PHASE_NAMES = [
    'active_check',
    'offsets',
    'legal_plays',
    'action_masks',
    'encode',
    'network_fwd',
    'no_action',
    'sample',
    'apply',
]
NUM_PHASES = len(PHASE_NAMES)


def make_games(num_players, count, seed_start=0, turns=1):
    games = []
    for i in range(seed_start, seed_start + count * 30):
        random.seed(i)
        g = Game(num_players)
        g.start_round()
        for p in range(num_players):
            g.submit_flip_decision(p, do_flip=False)
        for _ in range(turns):
            if g.phase != Phase.TURN:
                break
            p = g.current_player
            legal = get_legal_plays(g.players[p].hand, g.current_play)
            if not legal:
                g._advance_turn()
                continue
            g.apply_play(*random.choice(legal))
        if g.phase in (Phase.TURN, Phase.SNS_PLAY):
            games.append(g)
        if len(games) >= count:
            break
    return games


def load_network():
    for pattern in [
        os.path.join(SCRIPT_DIR, 'checkpoints', 'checkpoint_*.pt'),
        os.path.join(SCRIPT_DIR, 'bots', 'v7_*', 'latest.pt'),
    ]:
        files = glob.glob(pattern)
        if files:
            break
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    data = torch.load(latest, map_location='cpu', weights_only=False)
    config = data.get('config', {})
    net = FlatScoutNetwork(
        input_size=309,
        layer_sizes=config.get('layer_sizes', [512, 256, 128]),
        attention=config.get('attention', None),
    )
    net.load_state_dict(data['model_state'])
    net = net.to(DEVICE)
    net.eval()
    print(f"Loaded: {os.path.basename(latest)}")
    return net


def profiled_rollout(state: GpuGameState, network, max_steps: int = MAX_STEPS):
    """Rollout with deferred CUDA event timing — no mid-loop syncs except the
    unavoidable active.any() check."""
    B = state.done.shape[0]
    dev = state.done.device

    legal_buf = torch.zeros(B, H, H, dtype=torch.bool, device=dev)
    mask_buf = torch.zeros(B, FLAT_ACTION_SIZE, dtype=torch.bool, device=dev)
    encode_buf = torch.zeros(B, 309, dtype=torch.float32, device=dev)

    d_hands_show = numba_cuda.as_cuda_array(state.hands_show)
    d_hands_hide = numba_cuda.as_cuda_array(state.hands_hide)
    d_hand_len = numba_cuda.as_cuda_array(state.hand_len)
    d_play_show = numba_cuda.as_cuda_array(state.play_show)
    d_play_hide = numba_cuda.as_cuda_array(state.play_hide)
    d_play_len = numba_cuda.as_cuda_array(state.play_len)
    d_play_owner = numba_cuda.as_cuda_array(state.play_owner)
    d_play_type = numba_cuda.as_cuda_array(state.play_type)
    d_play_strength = numba_cuda.as_cuda_array(state.play_strength)
    d_current_player = numba_cuda.as_cuda_array(state.current_player)
    d_phase = numba_cuda.as_cuda_array(state.phase)
    d_scouts_since_play = numba_cuda.as_cuda_array(state.scouts_since_play)
    d_sns_available = numba_cuda.as_cuda_array(state.sns_available)
    d_num_players = numba_cuda.as_cuda_array(state.num_players)
    d_collected = numba_cuda.as_cuda_array(state.collected)
    d_scout_tokens = numba_cuda.as_cuda_array(state.scout_tokens)
    d_round_ender = numba_cuda.as_cuda_array(state.round_ender)
    d_done = numba_cuda.as_cuda_array(state.done)
    d_legal = numba_cuda.as_cuda_array(legal_buf)
    d_mask = numba_cuda.as_cuda_array(mask_buf)
    d_encode = numba_cuda.as_cuda_array(encode_buf)

    grid = _grid(B)

    # Pre-allocate all CUDA events: [step][phase] boundary markers
    # For each step we record NUM_PHASES+1 events (fenceposts around NUM_PHASES phases)
    all_events = []
    for _ in range(max_steps):
        step_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_PHASES + 1)]
        all_events.append(step_events)

    total_steps = 0

    network.eval()
    with torch.no_grad():
        for step in range(max_steps):
            evs = all_events[step]

            # ── phase 0: active check (has unavoidable sync) ──
            evs[0].record()
            active = ~state.done
            any_active = active.any().item()  # sync
            evs[1].record()

            if not any_active:
                # Record a dummy end event so we can still read this step
                total_steps = step
                break
            total_steps = step + 1

            # ── phase 1: offsets ──
            hand_offsets = torch.randint(0, H, (B,), device=dev, dtype=torch.long)
            d_offsets = numba_cuda.as_cuda_array(hand_offsets)
            evs[2].record()

            # ── phase 2: legal plays kernel ──
            compute_legal_plays_kernel[grid, TPB](
                d_hands_show, d_hand_len, d_current_player,
                d_play_len, d_play_type, d_play_strength,
                d_done, d_legal, B,
            )
            evs[3].record()

            # ── phase 3: action masks kernel ──
            compute_action_masks_kernel[grid, TPB](
                d_hands_show, d_hand_len, d_current_player,
                d_play_show, d_play_hide, d_play_len, d_play_type,
                d_phase, d_sns_available, d_num_players,
                d_legal, d_offsets, d_mask, B,
            )
            evs[4].record()

            # ── phase 4: encode kernel ──
            encode_states_kernel[grid, TPB](
                d_hands_show, d_hands_hide, d_hand_len, d_current_player,
                d_play_show, d_play_hide, d_play_len, d_play_owner,
                d_play_type, d_play_strength, d_phase,
                d_scouts_since_play, d_sns_available, d_num_players,
                d_collected, d_scout_tokens, d_offsets, d_encode, B,
            )
            evs[5].record()

            # ── phase 5: network forward ──
            h = network(encode_buf)
            logits = network.policy_logits(h)
            evs[6].record()

            # ── phase 6: no-action turn advance ──
            has_action = mask_buf.any(dim=1)
            no_action = active & ~has_action
            if no_action.any():
                adv_cp = ((state.current_player.long() + 1) %
                    state.num_players.long()).to(torch.int8)
                state.current_player = torch.where(
                    no_action, adv_cp, state.current_player)
                d_current_player = numba_cuda.as_cuda_array(state.current_player)
            evs[7].record()

            # ── phase 7: sample ──
            actions = batched_masked_sample(logits, mask_buf)
            d_actions = numba_cuda.as_cuda_array(actions)
            apply_active = active & has_action
            d_apply_active = numba_cuda.as_cuda_array(apply_active)
            evs[8].record()

            # ── phase 8: apply actions kernel ──
            apply_actions_kernel[grid, TPB](
                d_hands_show, d_hands_hide, d_hand_len,
                d_play_show, d_play_hide, d_play_len, d_play_owner,
                d_play_type, d_play_strength, d_current_player,
                d_phase, d_scouts_since_play, d_sns_available,
                d_num_players, d_collected, d_scout_tokens,
                d_round_ender, d_done, d_actions, d_offsets,
                d_apply_active, B,
            )
            evs[9].record()

    # Single sync at the end — read all event timings
    torch.cuda.synchronize()

    phase_totals = [0.0] * NUM_PHASES
    for s in range(total_steps):
        evs = all_events[s]
        for p in range(NUM_PHASES):
            phase_totals[p] += evs[p].elapsed_time(evs[p + 1])

    return {
        'steps': total_steps,
        'phase_totals_ms': dict(zip(PHASE_NAMES, phase_totals)),
    }


def main():
    net = load_network()
    if net is None:
        print("No checkpoint found.")
        return

    batch_sizes = [1000, 5000]

    for B in batch_sizes:
        print(f"\n{'='*60}")
        print(f"  B = {B}")
        print(f"{'='*60}")

        games = make_games(4, B, seed_start=B * 7, turns=1)
        while len(games) < B:
            games.extend(games[:B - len(games)])
        games = games[:B]

        # Warmup (full unsync'd rollout)
        state = from_snapshots(games, device=DEVICE)
        torch.manual_seed(0)
        rollout_numba(state, net, max_steps=100)
        torch.cuda.synchronize()

        # ── Measure from_snapshots ──
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        state = from_snapshots(games, device=DEVICE)
        torch.cuda.synchronize()
        t_from_snap = time.perf_counter() - t0

        # ── Profiled rollout (deferred events) ──
        torch.manual_seed(42)
        t_wall_start = time.perf_counter()
        prof = profiled_rollout(state, net, max_steps=100)
        t_wall = time.perf_counter() - t_wall_start

        # ── Measure compute_scores ──
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        compute_scores(state)
        torch.cuda.synchronize()
        t_scores = time.perf_counter() - t0

        # ── Also run the real (unmodified) rollout for wall-time comparison ──
        state2 = from_snapshots(games, device=DEVICE)
        torch.manual_seed(42)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rollout_numba(state2, net, max_steps=100)
        torch.cuda.synchronize()
        t_real = time.perf_counter() - t0

        steps = prof['steps']
        phases = list(prof['phase_totals_ms'].items())
        total_gpu = sum(v for _, v in phases)

        print(f"\n  Steps: {steps}")
        print(f"  Real rollout wall time:     {t_real:.3f}s  ({B/t_real:.0f} games/s)")
        print(f"  Profiled rollout wall time: {t_wall:.3f}s")
        print(f"  from_snapshots:             {t_from_snap*1000:.1f} ms")
        print(f"  compute_scores:             {t_scores*1000:.1f} ms")

        print(f"\n  GPU event timings (deferred, no extra syncs):\n")
        print(f"  {'Phase':<16} {'Total ms':>10} {'Per step ms':>12} {'% GPU':>8}")
        print(f"  {'-'*50}")
        for name, ms in phases:
            pct = ms / total_gpu * 100 if total_gpu > 0 else 0
            print(f"  {name:<16} {ms:>10.1f} {ms/steps:>12.3f} {pct:>7.1f}%")
        print(f"  {'-'*50}")
        print(f"  {'GPU total':<16} {total_gpu:>10.1f} {total_gpu/steps:>12.3f}")

        wall_ms = t_wall * 1000
        python_overhead = wall_ms - total_gpu
        real_ms = t_real * 1000
        real_python = real_ms - total_gpu

        print(f"\n  Wall vs GPU:")
        print(f"    Profiled wall:  {wall_ms:>8.1f} ms")
        print(f"    GPU total:      {total_gpu:>8.1f} ms")
        print(f"    Difference:     {python_overhead:>8.1f} ms  (Python dispatch + event overhead)")
        print(f"    Real wall:      {real_ms:>8.1f} ms")
        print(f"    Real - GPU:     {real_python:>8.1f} ms  (Python dispatch, no event overhead)")

        # Group summary
        kernel_ms = sum(prof['phase_totals_ms'][k] for k in ['legal_plays', 'action_masks', 'encode', 'apply'])
        network_ms = prof['phase_totals_ms']['network_fwd']
        overhead_ms = sum(prof['phase_totals_ms'][k] for k in ['active_check', 'offsets', 'no_action', 'sample'])

        print(f"\n  Summary (GPU time):")
        print(f"    Kernels (4):    {kernel_ms:>8.1f} ms  ({kernel_ms/total_gpu*100:>5.1f}%)")
        print(f"    Network fwd:    {network_ms:>8.1f} ms  ({network_ms/total_gpu*100:>5.1f}%)")
        print(f"    PyTorch ops:    {overhead_ms:>8.1f} ms  ({overhead_ms/total_gpu*100:>5.1f}%)")
        print(f"    Python glue:    {real_python:>8.1f} ms  (wall - GPU, est.)")


if __name__ == '__main__':
    main()
