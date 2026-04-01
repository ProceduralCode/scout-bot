# Numba Engine Implementation

## Task & State

Implemented all 4 Numba CUDA kernels and the `rollout_numba` entry point. All kernel tests pass against the PyTorch reference (`gpu_engine.py`). Full rollout works end-to-end with a trained network.

Integration into `training.py` not yet done.

## What Changed

### New files
- `scout-bot/numba_engine.py` — all 4 kernels + `rollout_numba` entry point
- `scout-bot/test_numba_legal_plays.py` — tests compute_legal_plays_kernel
- `scout-bot/test_numba_encode.py` — tests encode_states_kernel
- `scout-bot/test_numba_masks.py` — tests compute_action_masks_kernel (including S&S)
- `scout-bot/test_numba_apply.py` — tests apply_actions_kernel (single-action + multi-step + full game)
- `scout-bot/test_numba_rollout.py` — end-to-end rollout test with trained network
- `scout-bot/bench_numba.py` — benchmark at various batch sizes
- `scout-bot/spike_numba.py` — initial spike (can be deleted)

## Benchmarks

Numba kernel launch overhead: ~11us no-op, ~23us at B=5,000.

Rollout throughput (vs CPU Cython 270 g/s, PyTorch compiled 960 g/s peak):

| B | Numba (g/s) | vs CPU | vs PyTorch |
|---|---|---|---|
| 100 | 385 | 1.4x | — |
| 500 | 1,339 | 5.0x | 3.0x |
| 1,000 | 1,955 | 7.2x | — |
| 2,000 | 2,307 | 8.5x | — |
| 5,000 | 2,569 | 9.5x | 2.7x |

Throughput still climbing at B=5,000 — hasn't plateaued. Spec estimated 20,000 g/s; actual is ~2,600. Gap is likely network inference (bigger than the spec's 2ms estimate) and Python loop overhead (`active.any()` sync, `torch.where` for no-action, per-step `as_cuda_array` wrapping).

## Next Steps

1. **Profile the rollout loop** to identify where time is actually spent (kernel compute vs network inference vs Python overhead). This determines whether optimization is worthwhile before integration.
2. **Integration into training.py** — hook `rollout_numba` as the rollout backend. `rollout_numba` is a drop-in replacement for `rollout_gpu` / `rollout_from_states_batched_v6` (same `from_snapshots` → `rollout_numba` → scores interface). Network must be on CUDA before calling.
3. Potential optimizations if profiling shows room:
   - Remove `active.any()` sync point (run all steps unconditionally, or check periodically)
   - Pre-allocate `hand_offsets`, `actions`, `apply_active` tensors to avoid per-step allocation + `as_cuda_array`
   - The `torch.where` for no-action turn advance creates a new tensor requiring re-wrapping

## Watch Out

- `rollout_numba` imports `batched_masked_sample` from `network` module (local import to avoid circular dependency).
- `apply_actions_kernel` sets `play_type = PLAY_SET` when play is emptied (new_plen == 0), matching the reference. This is semantically meaningless (no play exists) but needed for exact state match.
- The `torch.where` for no-action turn advancement creates a new `current_player` tensor, requiring re-wrapping with `as_cuda_array`. If this becomes a bottleneck, move the no-action logic into a Numba kernel.
