# Rollout Profiling and Active-Game Compaction

## Task & State

Completed the two remaining items from session 074 (4×4 charts, mode default). Then ran first training run, noticed ~380s rollout time per iteration, and profiled to find the cause. Implemented active-game compaction in `rollout_numba` for a 2.1× speedup. Training pipeline is working end-to-end.

## What Changed

### main.py — Charts and mode default

- **`_save_q_charts` rewritten to 4×4 grid** (was 3×3). 15 charts + 1 hidden slot:
  - Row 0: Score Margin, MSE Loss, Pred vs Target Margin, Steps Per Game
  - Row 1: Play Length Dist, Avg Play Length, Scout Play Length, Action Type Dist
  - Row 2: Conditional Entropies, Dormant Neurons, Replay Buffer, Learning Rate
  - Row 3: Margin Predictions (hist line), Rollout Margins (hist line), Replay Buffer Age (hist line)
  - Added `plot_hist_snapshot` helper for `_hist_*` snapshot keys (line chart with fill, x=bin centers)
  - Added `_hist_` prefix skip in trim/smooth loop
- **`--mode` default changed to `"q"`** (was `"ppo"`)
- **Added `"scout_play_len": []`** to Q metrics_history init — was missing (present in PPO init but not Q), would have caused KeyError when `_run_eval` appended to it

### numba_engine.py — Active-game compaction

- `rollout_numba` now only forward-passes active (non-done) games each step
- Uses `torch.where(~state.done)[0]` to get active indices, gathers `encode_buf[active_idx]`, scatters logits back
- Pre-allocates logits buffer before the step loop (was allocated per-step)
- Numba kernels still run on all B games (they're only 6% of time)
- Stale logits for done games are harmless — masked out by `apply_active`

### Benchmark/profiling scripts created (not part of training)

- `profile_rollout.py` — pyinstrument profile of rollout
- `bench_chunk_pairs.py` — tested chunk_pairs 256–all-at-once
- `bench_rollout_step.py` — instrumented per-step timing with CUDA sync
- `bench_compaction.py` — before/after comparison

## Profiling Results

### Per-step breakdown (bench_rollout_step.py, B=15,300, one chunk)

| Phase | Per-step | % |
|---|---|---|
| Forward pass | 23.1ms | 79% |
| Sampling + no_action | 2.4ms | 8% |
| Action masks kernel | 1.5ms | 5% |
| Encode states kernel | 0.8ms | 3% |
| Legal plays kernel | 0.6ms | 2% |
| Apply actions kernel | 0.4ms | 2% |
| Active check + randint | 0.4ms | 1% |

### chunk_pairs benchmark

256–8192 all within 4.5% of each other (375–393s). 16384+ dramatically worse due to tail problem (longest game forces forward passes on all games). chunk_pairs is not a meaningful lever.

### Compaction result

383s → 182s (2.1× speedup). The savings come from not forward-passing done games. With 100-step rollouts where active fraction averages ~50%, this roughly halves forward pass work.

## Decisions

- chunk_pairs stays at 512 (default). Benchmarked 256–all-at-once; no meaningful difference in 256–8192 range. Larger values worse due to tail problem.
- Numba kernels left uncompacted — only 6% of time, not worth the complexity.

## Next Steps

### Training run

The pipeline works end-to-end (verified through iteration 5 + eval + chart generation). Rollout time is now ~180s/iter instead of ~380s. Ready for a real training run.

### Remaining rollout performance observations

- **100-step rounds**: every chunk hits MAX_STEPS=100. At step 100, 11% of games still active. With temperature=1.0 rollout policy and a near-random early network, games scout heavily and rounds last very long. This should improve as the network trains.
- **Forward pass still dominates** (79% of per-step time). Within that, `nn.MultiheadAttention` is 75% of forward pass time (63s/85s pre-compaction). Whether replacing it with manual QKV ops would help is untested.
- **`from_snapshots` is 28s** (Python loop converting Game objects → GPU tensors). Proportional to total pairs, not chunk_pairs.

## Watch Out

- The `_hist_*` keys in metrics_history are snapshots (overwritten each iter), not time series. The chart function handles them via `plot_hist_snapshot` which reads bin edges + counts directly, not through the trim/smooth pipeline.
- The 4×4 chart grid has slot (3,3) hidden via `set_visible(False)`. If a 16th chart is needed later, just unhide it.
- Rollout margins for games that don't finish within MAX_STEPS=100 are computed from the unfinished game state (partial round scores). These margins are inaccurate. Currently 11% of rollout games hit this.
