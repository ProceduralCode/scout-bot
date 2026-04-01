# Training Improvements v2 Handoff

## Task

Implemented four changes to break the Scout bot's training plateau: GAE, deeper network with residuals, LR annealing, and multi-seat training. Also added eval opponent tracking. A GAE bug was found and fixed during first training run.

## Current State

All changes implemented, bug fixed, training running on `v2_2` save dir. Previous handoff (`ai-handoff/training-improvements.md`) contains the original plan and root cause analysis — still accurate for context.

## What Changed

### `scout-bot/network.py`
- Added `ResidualBlock` class (line 10-17): `relu(linear(x) + x)` skip connection for same-width layers
- `ScoutNetwork.__init__` now takes `layer_sizes: list[int]` instead of `hidden_size`/`first_hidden_size`
- Builds shared layers dynamically: `Linear+ReLU` for width changes, `ResidualBlock` for same-width consecutive layers
- Stores `self.layer_sizes` for checkpoint serialization
- Backward compat: `hidden_size`/`first_hidden_size` kwargs still work (converted to `layer_sizes` internally)

### `scout-bot/training.py`
- `StepRecord`: added `game_id: int = 0` field (line 41)
- `play_game()`: added `training_seats` param; first N seats use training network, rest from opponent pool
- Reward restructuring: only last step per player per round gets `round_score/10`, intermediate steps get 0 (for GAE)
- `compute_gae()` replaces `compute_advantages()`: groups by `(game_id, round_num, player)`, walks backward with TD errors, returns `(normalized_advantages, returns, raw_std)`
- `ppo_update()`: added `returns` param for value targets (falls back to `s.reward` if None)
- `OpponentPool.state_dicts()`: now stores `{layer_sizes, state_dict}` per member
- `OpponentPool.load_state_dicts()`: handles both new format and old bare state dicts
- Removed dead `ReplayBuffer` class

### `scout-bot/main.py`
- Config: `layer_sizes=[512,256,256,128,128,128]`, `gamma=0.99`, `gae_lambda=0.95`, `training_seats=3`, `eval_opponents={}`. Removed `hidden_size`, `first_hidden_size`, `replay_buffer_size`
- LR annealing: `lr = initial_lr * (1 - iteration/total_iterations)` before each PPO update
- `compute_gae` call with `game_id` assignment per game in the collection loop
- `advantage_std` metric now uses GAE raw std instead of `reward - value`
- Eval opponents: loads checkpoints at startup (detects old vs new architecture format), runs eval games against each, tracks `eval_margin_<name>` metrics
- Chart: eval chart shows multiple lines with auto-legend (only when labeled artists exist)
- Centered moving average for chart smoothing
- Auto-resume handles old config format → `layer_sizes` conversion
- Argparse defaults changed to `None` so DEFAULT_CONFIG values aren't silently overridden

### `scout-bot/matchup.py`
- `load_agent()`: detects `layer_sizes` in checkpoint config for new-format architecture

## Bug Found and Fixed

**GAE cross-game contamination**: `compute_gae` originally grouped records by `(round_num, player)`. Since `round_num` is the round index within a game (0-3), records from different games with the same round number were merged into one GAE trajectory. Step B from game 1 was bootstrapped from step C of game 2 — completely unrelated states. Manifested as dramatic loss swings then quick plateau.

**Fix**: Added `game_id` field to `StepRecord`, assigned in the collection loop, and included in the GAE grouping key: `(game_id, round_num, player)`.

**Lesson**: The original handoff plan specified grouping by `(round_num, player)` which was wrong — the plan didn't account for multi-game batches. Unit testing `compute_gae` with records from multiple games would have caught this immediately.

## Next Steps

- Monitor `v2_2` training run for healthy dynamics (steady improvement in eval margin, explained variance trending toward 1, no dramatic loss swings)
- If training looks healthy past ~200 iterations (where v1 plateaued), the changes are working
- If still plateauing, possible next investigations: entropy bonus tuning, batch size, GAE lambda sensitivity
- Consider adding assertions to `compute_gae` as safety nets (e.g., max group size sanity check)

## Files to Read First
- `scout-bot/training.py` — GAE implementation, reward structure, multi-seat
- `scout-bot/main.py` — training loop, config, eval opponents
- `scout-bot/network.py` — ResidualBlock, layer_sizes architecture

## Watch Out
- `avg_reward` metric (main.py) is now diluted — most records have reward=0 (sparse for GAE). Still trends correctly but at lower absolute scale than v1.
- Must start fresh save_dir for new architecture — can't resume v1 checkpoints
- Old v1 checkpoints work as eval opponents (architecture auto-detected from saved config)
