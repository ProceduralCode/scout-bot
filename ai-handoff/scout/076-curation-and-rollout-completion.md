# Sample Curation and Rollout Completion

## Task & State

Three changes made, all working and verified:

1. **Rollout completion** — games now run to completion instead of truncating at 100 steps
2. **Sample curation** — new system to equalize per-output-neuron training signal
3. **Q charts summary.txt** — `_save_q_charts` now writes summary.txt (was PPO-only)

No training run started yet. Pipeline was verified end-to-end through iteration 5 in session 075.

## What Changed

### numba_engine.py — Rollout completion

- `MAX_STEPS` changed from 100 to 1000 (safety cap)
- `rollout_numba` loop changed from `for step in range(max_steps)` to `while True` with `max_steps` check
- Benchmarked check-every-step vs check-every-N: checking every step is optimal (compaction benefit outweighs `torch.where` cost)

### training.py — `curate_samples()`

- New function between `play_games_q_v6` and `rollout_multi_action_v6` in the pipeline
- Scores each sample by inverse frequency of its legal actions across all 16 rotations
- Vectorized using numpy (45K samples in 5.4s at multiplier=10)
- Uses `FULL_PERM` permutation tables for rotation-aware frequency counting

### main.py — Config and integration

- `curation_multiplier: 1` added to `Q_PARAMS` (default = no curation)
- `train_q` loop plays `game_count * curation_multiplier` games, curates down before rollouts
- `curate_samples` imported and called when multiplier > 1, logs curated count
- `_save_q_charts` now writes `summary.txt` with smoothed metrics, eval margins, and config dump

### Diagnostic/bench scripts created (not part of training)

- `diag_output_coverage.py` — compares per-output training signal: baseline vs curated
- `diag_curation_time.py` — times game-playing and curation at different multipliers
- `bench_check_interval.py` — benchmarked compaction check frequency

## Decisions

- **Rollout completion over truncation**: 11% of games were unfinished at MAX_STEPS=100, producing inaccurate margins. With compaction, continuing is cheap (only forward-passing the shrinking active set). MAX_STEPS=1000 as safety cap.
- **Legal-action rarity scoring** (not rollout-action rarity): scores based on which actions are legal in each game state, independent of the network's current predictions. Simpler and more stable than coupling to top-K selection.
- **Vectorized curation**: initial Python-loop implementation took 60-120s; vectorized with numpy takes 2-5s.

## Next Steps

### Training run with new config

Discussed but not committed to config yet:
- `curation_multiplier: 10` — play 1000 games, curate to ~4500 samples
- Attention config change: `{"dim": 20, "heads": 4, "layers": 3}` (was `{"dim": 32, "heads": 2, "layers": 1}`)
- This is an architecture change — requires fresh training run, incompatible with existing checkpoints

### Rationale for attention change

- dim 32→20: reduce unnecessary projection overhead. Raw entity features are 13 dims + 20-dim positional = 33 input. 20 is closer to the useful information than 32.
- heads 2→4: more parallel attention perspectives (5 dims/head)
- layers 1→3: deeper relational reasoning (iterative card-to-card interactions). Prior probes showed FC networks can't learn card comparisons efficiently — attention depth should help more than attention width.

## Watch Out

- Curation coverage diagnostic showed 176/384 outputs still get zero signal even at 10x multiplier — these are play-5+ which are never legal in 1000 games. Genuinely rare game states, not a bug.
- The `max_steps is not None` check in the rollout loop is vestigial from when default was `None` — now that default is `MAX_STEPS`, it's always not-None. Harmless but could be simplified to just `step >= max_steps`.
- Attention config change + curation multiplier are not yet applied to Q_PARAMS — they were discussed but need to be set before starting the run.
