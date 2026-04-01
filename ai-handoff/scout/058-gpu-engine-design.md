# GPU Engine Design and Performance Analysis

## Task & State

Extended performance investigation from session 057. No training changes made. Session was exploratory — profiling, benchmarking, and architectural design discussion. v7_4 is the current checkpoint.

## What Changed

### New files
- `scout-bot/profile_rollouts.py` — cProfile-based profiler for `play_games_with_rollouts_v6`. Loads v7_3, runs 5 rollout games, prints cumulative and tottime breakdowns. Run: `python -u scout-bot/profile_rollouts.py`
- `scout-bot/bench_gpu.py` — CPU vs GPU inference benchmark at various batch sizes (1–1024). Measures raw GPU time and GPU+transfer (realistic). Run: `python -u scout-bot/bench_gpu.py`

### context.md
- Added **Self-Play Purity Principle** section (user-authored, do not edit)
- GPU compute profile note corrected (was wrong — see Decisions)

## Decisions

### GPU IS faster at rollout batch sizes
The earlier "GPU is slower" finding was measured at batch=1 (sequential per-turn inference). The actual average batch size during `rollout_from_states_batched_v6` is ~346 states/call (all active games batched per step). At that batch size, GPU+transfer is **3.91x faster** than CPU. At batch=1024, 5.45x.

Benchmark results (RTX 3060 Laptop GPU):
| Batch | CPU (ms) | GPU+xfer (ms) | Speedup |
|-------|----------|---------------|---------|
| 1     | 0.56     | 0.77          | 0.73x   |
| 64    | 1.32     | 0.93          | 1.42x   |
| 256   | 3.12     | 1.09          | 2.87x   |
| 346   | 4.37     | 1.12          | 3.91x   |
| 1024  | 12.19    | 2.24          | 5.45x   |

### Profiling breakdown (5 rollout games, 15.5s total)
- Python/Cython loop in rollout function: 6.3s (41%) — Cython calls hidden here, not separately visible
- Model inference (717 calls, ~346 avg batch): 3.4s (22%)
- Game engine apply_play/scout: 3.0s (19%)
- random.randint (248k calls): 1.0s (6%)
- Other: 1.8s (12%)

### Vectorized GPU game engine: the main architectural direction
The highest-leverage path is replacing `rollout_from_states_batched_v6` with a GPU-native implementation where game state lives in GPU tensors and never round-trips to CPU between steps. Python calls one function per iteration; GPU handles all game sim + inference internally.

Scout is ~65-70% cleanly tensorizable:
- All scalar state, scoring, phase transitions: trivial
- Hand manipulation (apply_play removes a slice, apply_scout inserts): doable with batched gather/scatter over fixed 16-slot tensors
- Legal play computation: O(n²) over bounded 16-slot hands, fully parallelizable
- Action masking (384-dim flat space): expressible as tensor ops, complex but bounded
- S&S two-phase logic: the trickiest part, requires per-game phase tracking

Fixed-step execution (MAX_STEPS=40, mask done games) eliminates all sync points except one final scores download.

Estimated speedup: 20–40x over current for rollout completions.

### Value head
The value head exists to enable GAE without running rollouts. If rollouts become significantly cheaper, its role diminishes. Decision deferred — assess after GPU engine is built. Flip decisions currently use the value head directly; would need an alternative if value head is dropped.

### Performance options ranked
1. GPU inference only (2-3 lines): ~1.2x — useful quick win
2. Reduce rollouts_per_state 20→10 (1 param): ~1.9x — trades signal quality
3. Python multiprocessing: ~6-8x, 1-2 weeks
4. Multiprocessing + shared GPU: ~10x, 2-4 weeks
5. Multithreaded C + GPU: ~10-15x, 3-5 weeks
6. Full GPU vectorized engine: ~20-40x, 3-4 weeks (Scout is tensorizable)

## Next Steps

Decision needed: which option to pursue. Full GPU engine is highest leverage but significant work. Multiprocessing is lower risk, faster to validate.

If pursuing GPU engine: start with rollout completions only (no records, just scores), keeping existing Python engine for main game play. Cuts scope roughly in half.

Quick win available now: move model to GPU + `.to('cuda')` in rollout loop = ~1.2x for 2 lines of code.

## Watch Out

- The "GPU is slower" note in context.md was updated — future sessions should not revert to the old conclusion
- Cython functions are invisible to cProfile; their time is folded into the 6.3s "self time" of the rollout loop
- `bench_gpu.py` checkpoint loading uses `model_state` key (not `model_state_dict`)
- The vectorized engine design is for rollout completions only (play-to-end, return scores). The main game play path (which records states/actions for PPO) is more complex and would be a separate effort.
