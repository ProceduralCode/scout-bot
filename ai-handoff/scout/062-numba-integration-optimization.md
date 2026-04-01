# Numba Engine Integration & Rollout Optimization

## Task & State

Integrated `rollout_numba` into the training pipeline and optimized the data marshaling bottleneck. Changes compile and run but the optimized version (`repeat_state` + vectorized margins) has not been validated in training yet — the user should run and check for errors or incorrect training metrics.

## What Changed

### Modified files
- `scout-bot/training.py` — `play_games_with_rollouts_v6()` restructured:
  - Rollout call hoisted outside the per-source-game loop (one batched call instead of N)
  - Unique snapshots packed once via `from_snapshots`, then `repeat_state()` duplicates on GPU
  - Margin computation vectorized on GPU tensors instead of Python loops
  - `rollout_scores` is now a `[B, MAX_P]` tensor, not `list[list[int]]`
- `scout-bot/gpu_engine.py` — added `repeat_state()` and `compute_scores_tensor()`
- `scout-bot/numba_engine.py` — `rollout_numba` returns `[B, MAX_P]` tensor via `compute_scores_tensor`; suppressed Numba grid size warnings
- `scout-bot/main.py` — guarded 4 `ax.legend()` calls to suppress matplotlib warning when no labeled artists exist

### New files
- `scout-bot/profile_numba.py` — profiling script for the rollout loop (diagnostic, can be deleted)

## Decisions

- **No new version number (still v7).** The Numba engine is a pure performance optimization — same network, encoding, training algorithm, checkpoint format.
- **`rollout_numba` return type changed from `list[list[int]]` to `Tensor`.** This breaks the drop-in interface with `rollout_gpu` / `rollout_from_states_batched_v6`. The CPU fallback path in the caller converts its list output to a tensor for uniform handling.
- **`repeat_state` uses `repeat_interleave` not `repeat`.** `repeat_interleave` keeps rollouts contiguous per snapshot (snap0_r0, snap0_r1, ..., snap1_r0, ...) which matches the `.view(n_snaps, rollouts_per_state, n_p)` reshape in margin computation.

## Next Steps

1. **Run training and verify correctness.** Check that training metrics (reward, value loss, advantages, EV) look normal. The vectorized margin computation should produce identical results to the old Python loop, but verify empirically.
2. **Profile again with `--profile 1`.** The old profile showed `from_snapshots` at 50.6s and `compute_scores` at 14.8s. These should both drop dramatically. Expected iteration time ~30-40s (down from ~117s).
3. **If `from_snapshots` is still slow**, the remaining cost is the ~750 unique snapshot iterations (Python Game objects → tensor packing). Further optimization would require changing how snapshots are stored (e.g., numpy arrays instead of Game objects).

## Watch Out

- `rollout_numba`'s return type is now `Tensor` (was `list[list[int]]`). Existing tests (`test_numba_rollout.py`) and benchmarks (`bench_numba.py`) expect the old return type and will need updating.
- The CPU fallback path in `play_games_with_rollouts_v6` still expands snapshots in Python (no `repeat_state`) since it works with Game objects, not GPU tensors.
- The vectorized margin formula `(score * n - total) / ((n-1) * 10)` assumes all games have the same `num_players`. This is true for training (set by PARAMS) but would break if mixed player counts were ever used.
- `snap_means` and `avg_std_per_snap` are moved to CPU before the per-game advantage loop to avoid per-element GPU→CPU transfers.

## Profiling Data (from this session)

Pyinstrument profile of one iteration (before optimization, with batched rollout call but without repeat_state/vectorized margins):

| Component | Time | % of 117s |
|---|---|---|
| `from_snapshots` | 50.6s | 43% |
| `rollout_numba` [self] (GPU compute) | 15.2s | 13% |
| `compute_scores` (GPU→Python) | 14.8s | 13% |
| Numba JIT compilation (one-time) | 5.1s | 4% |
| PPO update | 11.6s | 10% |
| GAE games | 7.9s | 7% |
| Augmentation | 4.7s | 4% |

Numba rollout loop breakdown (CUDA events, B=5000): network forward 66%, kernels 15%, PyTorch ops 19%, Python dispatch 25% of wall time.
