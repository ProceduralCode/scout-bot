# Scout Bot — game.py Review Session

## Task

Reviewing `scout-bot/game.py` structure and architectural decisions. Going through the file top-to-bottom, discussing each design choice and making changes.

## Current State

game.py has been significantly refactored and is in a good state. Changes are uncommitted. The other files (encoding.py, network.py, training.py, main.py) have NOT been updated to match — they still reference the old API (classify_play, play_beats, get_legal_action_types, get_legal_plays, etc.). They will need their own review pass.

## Repo Setup

- Created `scout-bot/` git repo (GitHub: ProceduralCode/scout-bot)
- Cloned from `git@github.com:ProceduralCode/scout-bot.git`
- Initial commit pushed with all first-pass files + `.gitignore`
- Old `Scout/` directory deleted; `design.md` copied into `scout-bot/`

## Changes to game.py

1. **Removed `PlayType.SINGLE`** — single cards now classify as `SET`. A single value trivially satisfies "all values equal." The `SINGLE` type was never consulted by comparison logic (`play_beats` had a count==1 early return before checking type).

2. **Renamed `Phase.PLAY` → `Phase.TURN`** — "Play" was overloaded (the phase, the action, the dataclass). `TURN` means "it's your turn, pick an action." `SNS_PLAY` kept as-is since it literally means "you must play cards now."

3. **Added `count` to `Play` dataclass and `classify_play` return** — count, play_type, and strength are all derived properties of a play. Storing count is consistent with storing the other two. Ordered as `count, play_type, strength` to match comparison priority in `play_beats`.

4. **Moved `classify_play` → `Play.from_cards()` classmethod** — A Play is defined by its cards; construction logic belongs on the class. Returns `Play | None`.

5. **Moved `play_beats` → `Play.beats()` instance method** — `new_play.beats(current_play)` reads naturally. Eliminated 6-parameter function.

6. **Removed `get_legal_action_types`, `get_legal_plays`, `get_legal_play_starts`, `get_legal_play_ends`, `get_legal_insert_positions`** — These were the game engine pre-chewing data for the network's action decomposition. The game engine's job is to validate and enforce rules, not enumerate legal moves for consumers. Encoding will import `Play.from_cards` and `Play.beats` to do its own enumeration.

7. **Rewrote `_has_any_legal_play` to short-circuit** — Returns `True` on the first legal play found instead of building the full list.

## Where We Left Off

We were partway through the top-to-bottom review of game.py. Completed review of:
- Types/enums (PlayType, Phase)
- Play dataclass (from_cards, beats)
- PlayerState
- flip_hand, create_deck
- Game.__init__, start_round, submit_flip_decision
- apply_play, apply_scout, apply_sns_scout, _do_scout
- _has_any_legal_play

Still need to review:
- `_end_round` (line 212)
- `get_round_scores` (line 224)
- `get_rewards` (line 234)
- `get_state_for_player` (line 248)
- `get_relative_position` (line 270)

## Downstream Impact

These files still use the old API and will break:
- `encoding.py` — calls `game.get_legal_action_types()`, `game.get_legal_play_starts()`, `game.get_legal_play_ends()`, `game.get_legal_insert_positions()`, references `classify_play`
- `network.py` — likely unaffected (doesn't call game directly)
- `training.py` — calls game methods, uses `Phase.PLAY`
- `main.py` — likely uses `Phase` references

Decision: update downstream files only after game.py review is complete.

## How to Verify

```bash
# Syntax check (no torch needed)
python -c "from scout_bot.game import Game, Phase, Play; print('OK')"

# Note: can't run as a package yet — files are flat in scout-bot/, not in a Python package subdirectory. Use:
python -c "import sys; sys.path.insert(0, 'scout-bot'); from game import Game, Phase, Play; print('OK')"
```

## Watch Out

- `_do_scout` still mutates `self.current_play.cards` in place via `.pop()`. Works because the Play object is immediately replaced, but would break if anyone held a reference.
- `get_state_for_player` returns `current_play` as a raw card tuple list, not using the Play dataclass. May want to revisit when reviewing that method.
