# Attention Optimization and Probe Rewards

## Task & State

Optimized attention implementation for faster rollouts, scaled up attention dimension, and added a probe reward system for architecture validation. Play-length probe is implemented and smoke-tested. Scout-quality probe is not yet implemented.

## What Changed

### network.py — SDPA attention, single-head d=64

Replaced manual `bmm` + `softmax` + `bmm` (4 kernel launches) with `F.scaled_dot_product_attention` (1 fused kernel). Changed from multi-head to single-head. Kept `load_state_dict` mapping for old `nn.MultiheadAttention` keys (shapes won't match at d=64 though).

The SDPA path does `view/transpose` into `(B, H, S, HD)` format even for H=1 — this is the format SDPA expects, and benchmarks showed H=1 SDPA is faster than raw bmm (3.51ms vs 4.12ms at d=64, B=1024). The reshape is trivial for H=1.

### main.py — Q_PARAMS config changes

- `attention`: `{"dim": 64, "heads": 1, "layers": 2}` (was `{"dim": 20, "heads": 4, "layers": 2}`)
- `layer_sizes`: changed by user to `[512, 256, 256, 128, 128]` (was `[256, 128]`)
- `save_dir`: changed by user to `"bots/v8_5"`
- Added `probe_reward` param (currently set to `"play_length"`, needs to be set to `None` for normal training)
- `total_iterations`: currently set to 2 (TEMP smoke test value, needs to be restored to 1_000_000)

### training.py — `rollout_multi_action_v6` probe_reward param

Added `probe_reward: str | None = None` parameter. When set to `"play_length"`, skips all GPU rollouts and assigns deterministic targets: `play_len / 5.0` for play actions (0-255), `0.0` for scout/S&S actions (256-383). Play length from flat action index: `(action_idx % 16 - action_idx // 16) % 16 + 1`.

### Benchmark scripts created

- `bench_attention.py` — dims × implementations (bmm vs SDPA)
- `bench_multihead.py` — single-head vs multi-head bmm vs SDPA overhead isolation
- `bench_trunk.py` — FC trunk size timing
- `bots/v8_perf_test/` and `bots/probe_smoke_test/` — test outputs, safe to delete

## Decisions

- **Single-head over multi-head**: Multi-head reshape is expensive (+66% at h=4, +138% at h=8 for bmm; SDPA reduces this but still slower). Scout card relationships (runs vs sets) are ~1-2 similarity patterns, not 4-8. Two attention layers compensate for single-head — layer 2 can attend based on layer 1's output.
- **SDPA over manual bmm**: At h=1 d=64, SDPA is 15% faster than bmm (3.51ms vs 4.12ms) because the fused kernel avoids materializing the intermediate attention matrix.
- **d=64 over d=20**: 3.2x more attention capacity for comparable speed to old nn.MHA (4.93ms). d=128 was slower than old; d=32 was faster but less capacity.
- **FC-only rejected**: Can't learn position-invariant card-to-card relations. Needs exponentially more capacity to approximate what attention does naturally with circular slot augmentation.
- **torch.compile rejected**: Crashes on Windows with file cache race condition.

## Benchmark Data

Forward pass at B=1024, d=64 h=1 L=2:
- Old nn.MultiheadAttention (d=20 h=4): 4.93ms, 207K/sec
- SDPA single-head (d=64 h=1): 3.51ms, 292K/sec (1.4x faster)
- FC only (no attention): 0.21ms

Per-chunk rollout time: ~2.0s/chunk (was ~3.0s), 33% faster.

Trunk timing at d=64 attention:
- `[256, 128]`: 3.35ms
- `[512, 256, 128]`: 4.23ms (+0.88ms)
- `[512, 256, 256, 128, 128]`: 4.32ms (+0.97ms)

## Next Steps

- **Restore Q_PARAMS**: Set `total_iterations` back to 1_000_000, set `probe_reward` to `None` (or remove) for normal training
- **Implement `scout_quality` probe reward**: For each scout action, compute longest legal play containing scouted card after insertion. Requires cloning game state and calling `get_legal_plays`. Reference: `eval_scout_quality` in `probe.py` lines 232-255.
- **Run probe training**: Play-length probe first (~20 iterations should show clear convergence), then scout-quality probe
- **Clean up**: Delete benchmark scripts and test bot dirs when done

## Watch Out

- **Q_PARAMS has TEMP values**: `total_iterations=2` and `probe_reward="play_length"` are smoke test values. Must be changed before any real training run.
- **Probe targets are normalized**: play_length / 5.0, so a 5-card play = target 1.0. This puts targets in a different range than normal margins (-8 to +8).
- **Old checkpoints incompatible**: d=64 weights won't load from d=20 checkpoints (shape mismatch). The `load_state_dict` key mapping handles old nn.MHA→qkv renaming but not dimension changes.
