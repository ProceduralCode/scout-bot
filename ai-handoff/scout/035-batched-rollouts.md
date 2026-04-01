# Batched Rollouts

## Task & State

Added batched forward passes to rollout-based advantage estimation. Previously rollouts ran sequentially (one `rollout_from_state()` call at a time). Now all rollouts for a game's snapshots run through a single batched turn loop. Complete and ready to test.

## What Changed

- `scout-bot/training.py` — Added `rollout_from_states_batched()` (line 168): stripped-down version of `play_games_batched()`'s turn loop that takes a list of game snapshots, deepcopies them, plays all to completion with batched network forward passes, returns scores. No StepRecords, no opponent pool, no flip phase, no rewards — just plays and returns scores.
- `scout-bot/training.py` — Modified `play_games_with_rollouts()` (line 360): instead of looping over snapshots and calling `rollout_from_state()` N times each, expands all snapshots × `rollouts_per_state` into a flat list and sends them through one `rollout_from_states_batched()` call. Results mapped back by index.
- `rollout_from_state()` (sequential version) kept in place but no longer called. Could be removed later.

## Decisions

- **Option B batching** — all rollouts from all snapshots of one game batched together (not just per-snapshot, not cross-game). A game with ~15 decision points × 10 rollouts/state = ~160 rollout games batched simultaneously. Sweet spot between complexity and parallelism.
- **Source games still sequential** — the outer `for game_idx in range(num_games)` loop in `play_games_with_rollouts` is unchanged. Batching across source games (option C) would require restructuring snapshot collection since snapshots are gathered as each game plays.

## Next Steps

1. Run a training session with `use_rollouts: True` to verify correctness and measure speedup.
2. The rollout approach itself hasn't been validated for learning quality yet (see handoff 032).
