# Performance and Memory Fixes

## Task & State

Fixed several performance and memory issues blocking the first real training run with curation + attention config. All changes made, not yet verified with a full training run.

## What Changed

### training.py — Deferred snapshot creation

`play_games_q_v6` no longer creates 93K Game clones during game play. Instead:
- Records replay data per game: initial state clone, flip decisions, and action list
- Returns `(samples, game_replays)` tuple instead of just `samples`
- New `attach_snapshots(samples, game_replays)` replays only games with surviving curated samples, creating ~4.7K clones instead of 93K

QSample changes: `snapshot` field now `Game | None = None`, new `_turn_index: int = -1` field. Old checkpoints load fine (`QSample(**sd)` uses defaults for missing fields). `_turn_index` is not serialized — only used within one iteration.

### training.py — Active-index tracking

Game-playing turn loop uses `active = set(range(game_count))` instead of scanning all games every turn. Only iterates active games; removes finished games from the set. With 2000 games, this eliminates scanning ~1950 finished games per turn in the late phase.

### training.py — Batched revalidation

`ReplayBuffer.revalidate()` now batches rollouts in chunks of 512 pairs through GPU, same pattern as `rollout_multi_action_v6`. Was previously launching the full GPU rollout pipeline individually for every (sample, action) pair — ~3,200 individual GPU launches per cohort.

### training.py — Empty mask fix

`_select_action_q_batched` epsilon-greedy loop: added `if len(legal) == 0: continue` to handle game states with no legal actions (passing). The crash was `index 4294967295` (uint32 -1 from `% 0`).

### training.py — Progress bars

Added tqdm progress bars:
- `games` bar in `play_games_q_v6` turn loop (tracks finished games)
- `rollouts` bar in `rollout_multi_action_v6` chunk loop

### numba_engine.py — Warning suppression

Changed `warnings.filterwarnings` to `warnings.simplefilter("ignore", NumbaPerformanceWarning)` — the filterwarnings version wasn't sticking due to other libraries resetting filters.

### main.py — Memory diagnostics

Added `psutil.Process().memory_info().rss` logging at 4 points: after play, after curation, after snapshots, after rollouts. Import `psutil` added. Import `attach_snapshots` added.

### Bench/diag scripts

All 7 scripts updated for new `play_games_q_v6` return type:
- bench_chunk_pairs.py, bench_check_interval.py, bench_compaction.py, bench_rollout_step.py, profile_rollout.py: unpack tuple + call attach_snapshots before rollouts
- diag_curation_time.py, diag_output_coverage.py: unpack with `samples, _ = ...`

## Decisions

- **Replay-based deferred snapshots over pickle serialization**: Recording game actions and replaying is cleaner than serializing/deserializing Game objects. The replay data (2000 initial clones + action dicts) is much smaller than 93K Game clones.
- **Active set uses `sorted(active)` iteration**: Maintains deterministic game processing order matching the original sequential iteration.
- **Eval not changed**: Runs on CPU, 160 games total, infrequent (every 5 iters). Not worth optimizing.

## Next Steps

- Start training run, verify memory with [mem] prints, check iteration timing is reasonable
- Remove [mem] diagnostic prints once satisfied with memory behavior
- The current Q_PARAMS config: `curation_multiplier: 20`, `attention: {dim: 20, heads: 4, layers: 2}`, `rollout_actions: 6+2`, `rollouts_per_action: 20`

## Watch Out

- Checkpoint at iteration 17 exists from a previous run with different config (3-layer attention, different rollout params). Since the attention architecture changed (3→2 layers), this checkpoint is incompatible with the current config. The training should start fresh or the old checkpoint should be cleared.
- `curation_multiplier: 20` was bumped from the originally-discussed 10. User's choice but means 2000 games played per iteration (100 × 20).
