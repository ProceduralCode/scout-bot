# Scout Bot — game.py Review Complete, encoding.py Updated

## Task

Continuing review and refactoring of scout-bot. This session: finished game.py review, updated encoding.py to match new API.

## Current State

`game.py` — reviewed, refactored, committed.
`encoding.py` — rewritten to match new game.py API, uncommitted.
`training.py`, `network.py`, `main.py` — still use old imports (`from Scout.*`) and old API (`Phase.PLAY`). Not yet updated.

## What Changed This Session

### game.py (committed as `28c6251`)

All changes from prior session plus:

1. **Removed dead `count == 1` branch in `beats()`** — redundant after SINGLE→SET change.
2. **Merged double `current_play is not None` check in `apply_play()`** (lines 131-133).
3. **Removed unreachable `current_play_owner is not None` guard in `_do_scout()`** (line 175) — callers already assert `current_play` exists.
4. **`_end_round()` calls `get_round_scores()`** instead of duplicating score calculation (line 225).
5. **`get_state_for_player` uses `list()` not tuple reconstruction** (line 264).
6. **Fixed `flip_hand`** — no longer reverses hand order, just swaps values (line 60).
7. **S&S asserts legal play** instead of degrading to regular scout (line 166). Encoding must only offer S&S when a legal play exists after scouting.
8. **`_has_any_legal_play` optimized** — incremental set/run/desc tracking, no Play object construction (lines 186-218).
9. **`_do_scout` copies cards list** before mutating (line 170).
10. **Added `collected_counts` and `scout_tokens`** to `get_state_for_player` (lines 267-268).
11. **Removed `get_relative_position`** — unused.
12. **Updated `get_round_scores` docstring** — removed stale "call before start_round()" note.

### encoding.py (uncommitted)

Full rewrite of mask functions. Key changes:

1. **Import path** — `from game import Game, Play, PlayType, Phase` (was `from Scout.game`).
2. **Flip fix** — `encode_hand_both_orientations` no longer reverses hand.
3. **Metadata expanded** — added `collected_counts` (/20) and `scout_tokens` (/5). METADATA_SIZE 23→33, INPUT_SIZE 463→473.
4. **New helper functions**:
   - `_get_legal_plays(hand, current_play)` — returns all (start, end) pairs
   - `_any_legal_play(hand, current_play)` — short-circuits on first match
   - `_sns_variant_legal(hand, play_cards, left_end, flip)` — simulates scout, checks all insert positions for legal play. Has fast path: if original hand already beats reduced play, skips per-position loop.
5. **`get_action_type_mask`** — computes legality locally. S&S variants individually checked via `_sns_variant_legal`. In SNS_PLAY phase, only play (index 0) is legal.
6. **`get_play_start_mask` / `get_play_end_mask`** — use `_get_legal_plays`.
7. **`get_scout_insert_mask`** — all positions 0..len(hand) are legal.

### Hook added

Created `.claude/hooks/block-cd.sh` and updated `.claude/settings.json` to block `cd` in bash commands via PreToolUse hook. Existing deny rules removed (redundant with hook).

## Decisions Made

- **S&S can't degrade to regular scout** — if you can't play after scouting, the action was illegal. Encoding must pre-check. Game engine asserts.
- **Scouting from a 1-card play is legal** — play gets consumed, `current_play` clears, next player can play anything.
- **`collected_counts` and `scout_tokens` exposed in state** — visible information in real game, useful for network.
- **Masking is correct approach for illegal actions** — discussed alternatives (rejection sampling, partial masks). Masking is standard, cheap for this game, avoids biased PPO gradients.

## Remaining Downstream Updates

These files still use `from Scout.*` imports and `Phase.PLAY`:

- **`network.py`** — imports from `Scout.encoding`. Needs import fix. INPUT_SIZE changed 463→473 so input layer size changes.
- **`training.py`** — imports from `Scout.game`, `Scout.encoding`, `Scout.network`. Uses `Phase.PLAY` (→ `Phase.TURN`). Line 78: `while game.phase == Phase.PLAY`.
- **`main.py`** — imports from `Scout.encoding`, `Scout.network`, `Scout.training`.

Also: `design.md` is untracked in the repo.

## How to Verify

```bash
# From workspace root (bash cwd is currently inside scout-bot/)
python -c "import sys; sys.path.insert(0, 'scout-bot'); from game import Game, Phase, Play; print('game OK')"

# encoding.py needs torch:
python -c "import sys; sys.path.insert(0, 'scout-bot'); from encoding import encode_state, INPUT_SIZE; print(f'encoding OK, INPUT_SIZE={INPUT_SIZE}')"
```

## Next Steps

1. Update `network.py` — fix imports, adjust input size for INPUT_SIZE 473
2. Update `training.py` — fix imports, `Phase.PLAY` → `Phase.TURN`
3. Update `main.py` — fix imports
4. Review each file for design issues (like we did with game.py)
5. Commit encoding.py + downstream fixes

## Watch Out

- **Bash working directory is inside `scout-bot/`** due to an earlier `cd`. All relative paths in bash are relative to `scout-bot/`, not workspace root. Read/Edit/Glob/Grep tools still use workspace root paths.
- **`get_play_end_mask` recomputes all legal plays** (same work as `get_play_start_mask`). Acceptable for now but could be refactored to compute once if perf matters.
- **S&S legality checking in encoding is O(variants × insert_positions × hand²)** — fine for hand sizes ~12 but most expensive mask operation.
