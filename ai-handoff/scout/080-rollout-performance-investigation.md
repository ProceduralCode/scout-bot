# Rollout Performance Investigation

## Task & State

Investigated why rollouts take 86% of iteration time (~26s/30s). Profiled the full pipeline, identified bottlenecks, implemented two optimizations, and discovered a third (largest) opportunity. No training run done yet with changes — needs end-to-end verification.

## What Changed

### numba_engine.py — Batch compaction in `rollout_numba`

Added `compact_threshold` param (default 0.5). When active games drop below this fraction of batch size, gathers only active games into new compact tensors. Eliminates wasted kernel launches on the long tail (last ~100 steps often have <100 active games but kernels launch on full 7K+ grid). ~7% rollout speedup.

New helper functions: `_wrap_state_for_numba()`, `_compact_state()`. Score tracking via `scores_out` tensor + `idx_map` for mapping compacted indices back to original positions.

### gpu_engine.py — Vectorized `from_snapshots`

Replaced per-element Python loop (22K iterations for 500 games × 4 players × 11 cards) with numpy batch assignment for hand/play cards. 300ms → 10ms per chunk (30x). Saves ~2.5s across 9 chunks per iteration.

### network.py — Replaced `nn.MultiheadAttention` with manual qkv/out + bmm

Replaced `nn.MultiheadAttention` (which launches 8+ CUDA kernels per layer) with manual `qkv` Linear + `bmm` + `softmax` + `out` Linear. Currently single-head (not multi-head). Added `load_state_dict` override to map old checkpoint keys (`attn.in_proj_weight` → `qkv.weight`, etc.).

**This is the biggest win: 4.93ms → 2.40ms per 1024 samples (2.1x forward pass speedup).** Forward pass is 64% of rollout time, so this potentially halves total rollout time.

**BUT: semantics changed from 4-head to single-head.** Checkpoint loads (shapes match) but produces different attention patterns. Not equivalent to old model.

## Decisions

- **Compaction threshold 0.5**: triggers when half the batch is done. Lower thresholds compact more often but overhead per compaction is non-trivial (gather all state tensors, reallocate buffers, re-wrap Numba arrays).
- **Numpy for from_snapshots**: `np.array(hand, dtype=np.int8)` then slice-assign is much faster than per-element torch tensor assignment in Python loops.
- **Single-head attention for now**: simpler, faster, fewer reshape operations. But not checkpoint-compatible with multi-head training.

## Next Steps

- **Decide on attention approach**:
  - Multi-head bmm (compatible with v8_3 checkpoint, probably ~3-4ms vs 4.93ms)
  - Single-head bmm (2.40ms, requires retraining or adaptation period)
  - No attention / FC-only (0.22ms, ~22x faster, but loses entity-level modeling)
- Run a training iteration to verify end-to-end speedup
- The v8_3 checkpoint is still flat at -71 to -77 margins — restarting might be fine anyway

## Watch Out

- **Single-head ≠ multi-head**: old checkpoint loads but attention computation differs. Model will behave differently until retrained. The `load_state_dict` mapping is shape-compatible but not semantically equivalent.
- **`compact_threshold=0` disables compaction** for A/B testing
- **Test files created**: `profile_rollouts.py`, `profile_rollout_steps.py`, `bench_chunk_size.py`, `bench_attention.py`, `test_compaction.py`, `test_compaction2.py`, `test_optimizations.py` — all diagnostic, safe to delete

## Profiling Data

Per-chunk breakdown (512 pairs × 15 rollouts = ~7K batch):

| Component | Time | % |
|-----------|------|---|
| `rollout_numba` | 3000ms | 91% |
| `from_snapshots` | 300ms → 10ms | 9% → 0.3% |
| clone+apply | 3ms | <1% |
| repeat_state | 1ms | <1% |

Inside `rollout_numba` (288 steps, 7215 batch):

| Phase | Time | % | Notes |
|-------|------|---|-------|
| Forward pass | 2775ms | 64% | GPU throughput flat at ~207K samples/sec regardless of batch size |
| Numba kernels | 892ms | 21% | 3 kernels per step on full batch |
| Sample+apply | 659ms | 15% | Gumbel sampling + apply_actions_kernel |

Forward pass throughput is identical from B=512 to B=8192 (~207K/sec). Network is small enough to saturate GPU at any batch size. Chunk size (1024) doesn't matter.

Attention is 96% of forward pass time (4.58ms of 4.93ms). FC trunk is 0.22ms. The `nn.MultiheadAttention` overhead comes from launching 8+ CUDA kernels per layer for tiny tensors (dim=20, seq_len=20, head_dim=5).
