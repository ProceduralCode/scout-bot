# Eval Fix & Post-GPU-Migration Profiling

## Task & State

Completed the three remaining items from session 063: fixed the eval path, re-profiled, and verified training correctness. All working cleanly.

## What Changed

### Modified files
- `scout-bot/main.py`:
  - `_run_eval()` now moves network to CPU before eval, restores to CUDA in a `finally` block. Fixes both the device mismatch error (`indices should be either on cpu or on the same device`) and the performance issue (batch-1 GPU inference with per-call `.to(dev)` overhead).

## Profiling Results (5 iterations, per-iter averages, steady state)

| Component | Session 062 (CPU) | Session 063 (GPU, pre-fix) | Now (GPU, post-fix) |
|---|---|---|---|
| `play_games_v6` | 10.7s | 3.1s | **0.8s** |
| `augment_rotation_v6` | 4.8s | 12.0s (regressed) | **1.6s** |
| `ppo_update_v6` | 12.1s | 1.2s | **1.0s** |
| Rollouts | 30.0s | 27.6s | **23.7s** |
| Eval (per occurrence) | — | 37.0s (error) | **31.5s** (works) |

Training-only time per iteration: **~26.5s** (down from 58.1s in session 062). 54% speedup.

## Rollout Bottleneck Analysis

Rollouts dominate at 23.7s/iter (90% of training time). Breakdown:
- `rollout_numba`: 17s — 15,000 games (750 snapshots × 20 rollouts_per_state), network forward passes are 66% of this
- `from_snapshots`: 2.2s — Python Game → tensor packing
- Source game play: 2.4s — 25 games × ~30 turns of batch-1 GPU inference

Options discussed (not implemented):
- **Batch source game inference** (~2.4s savings, moderate effort) — same approach as `play_games_v6` rewrite, but snapshot logic per turn adds complexity
- **Reduce `rollouts_per_state`** (20 → 10 saves ~8s, but noisier value estimates — empirical test needed)
- **Reduce `rollout_fraction`** (0.25 → 0.15 saves proportionally)
- Core `rollout_numba` bottleneck (network forward) is not compressible without fewer rollouts or a smaller network

## Decisions

- **Eval runs on CPU.** Network moved to CPU before `_run_eval`, back to CUDA after. Eval is infrequent (every 5 iters) so the ~0.3s transfer cost is negligible. All eval code paths (`_play_turn_v6`, `_sample_scout`, `play_eval_game`) detect device via `next(net.parameters()).device`.

## Watch Out

- **`ppo_update_v6` has 0.5s/iter of `.item()` inside PyTorch's Adam internals** (`_get_value` in `adam.py`). This is internal PyTorch code, not fixable on our side.
- **Numba JIT compilation** adds ~4.4s on first iteration after restart (cached thereafter).
