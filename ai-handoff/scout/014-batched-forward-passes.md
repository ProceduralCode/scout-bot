# Scout: Batched Forward Passes and Handoff System Redesign

## Task

Two main threads: (1) batch forward passes during game generation for speedup, (2) redesign the AI handoff system with thread context files.

## What Changed

### Modified Files

- **`scout-bot/main.py`**
  - `DEFAULT_CONFIG` renamed to `PARAMS` — now always overrides saved checkpoint config on resume. Layer sizes always come from checkpoint regardless of PARAMS (prevents shape mismatch crashes).
  - CLI training args removed (`--lr`, `--batch-size`, `--games-per-iter`, `--entropy-bonus`, `--iterations`). Only `--save-dir`, `--players`, `--match`, `--replay`, `--games` remain.
  - Config merge on resume is now: `saved_cfg → PARAMS → CLI overrides`, with `layer_sizes` forced from checkpoint.
  - Game generation loop now calls `play_games_batched()` instead of per-game `play_game()`.
  - LR already changed to 6e-4 in PARAMS (user made this change).

- **`scout-bot/training.py`**
  - `_assign_round_rewards()` — extracted from `play_game()` as shared function. Both `play_game` and `play_games_batched` use it.
  - `_process_turn_from_hidden()` — per-game turn logic using a pre-computed hidden state. Same as `_play_turn` body but without the forward pass or S&S recursion (S&S leaves game in SNS_PLAY for batch loop).
  - `play_games_batched()` — plays N games simultaneously with batched shared-layer forward passes. Handles flip phase (one batch of 800 states) and turn phase (batched per iteration of turn loop). Splits training vs opponent network calls.

- **`ai-context/shared/workflows/handoff.md`** — rewritten with thread system (directories, numbered handoffs, context.md, rewrite discipline, no-overlap rule).
- **`ai-context/ProceduralCode/init/misc.md`** — added "don't use auto-memory system" instruction.

### Migrated Files

- All 27 handoff files reorganized from flat `ai-handoff/` into `ai-handoff/scout/`, `ai-handoff/cathedral/`, `ai-handoff/claude-code/` with numbered names.

## Current State

Working but the speedup from batching is modest. Batched version runs correctly and produces equivalent StepRecords.

### v3_4 Training (summary.txt at iter 2180, LR bumped to 6e-4 around iter ~2100)

Early signs of LR change: clip_fraction 0.014 → 0.038, approx_kl 0.0014 → 0.004 (both healthy). Eval margins against named opponents dipped slightly — likely LR turbulence, needs more iterations to tell.

## Decisions

- **PARAMS override instead of CLI args** — user wanted to edit main.py directly rather than pass training hyperparameters via command line. Checkpoint still records config for future use.
- **Layer sizes forced from checkpoint** — prevents crash when PARAMS layer_sizes doesn't match saved model weights.
- **Batching approach**: batch the shared-layer forward pass, keep sub-heads and game logic per-game. This was the right first step, but the sub-heads turned out to be the new bottleneck.

## Next Steps

- **Watch v3_4 with 6e-4 LR** — check if eval margins recover and continue climbing after initial turbulence. If unstable (clip_fraction > 20%, KL > 0.02), restore from iter_2000.pt and use a lower LR. User suggested binary search (overshoot then bisect) to find optimal LR.
- **Batch sub-heads too** — the 1.3x speedup is because `_process_turn_from_hidden` is now 87% of time (sub-head logits are individual PyTorch calls per game, same dispatch overhead problem). Next step: after batched shared forward, batch action_type_logits for all games, sample per-game, then group by action type and batch play_start/scout_insert, etc.
- **Profile again after sub-head batching** — see if encoding (10%) or masking becomes the new bottleneck.

## Watch Out

- **S&S handling differs between paths**: `_play_turn` recurses for the forced play after S&S. `_process_turn_from_hidden` doesn't — the game stays in SNS_PLAY and the batch loop picks it up on the next iteration. Functionally equivalent but structurally different.
- **Opponent pool forwarding**: `play_games_batched` receives `pool.versions` directly (the full list), not a pre-sampled subset. It does `random.choice()` per seat per game to match the original distribution.
- **Summary.txt** in `scout-bot/v3_4/summary.txt` has live smoothed metrics — check it alongside charts for current training state.
