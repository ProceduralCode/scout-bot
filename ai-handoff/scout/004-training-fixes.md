# Scout Bot Training Pipeline Fixes

## Task

Diagnosed why the Scout bot was playing randomly after 300 iterations of training, then fixed the critical issues in the training pipeline.

## What Was Wrong

The bot played randomly because:
1. **PPO was broken by a replay buffer** — PPO is on-policy but was sampling from a stale replay buffer. Policy loss was literally 0 (clipping zeroed out all gradients from stale data).
2. **Game-end reward was too sparse** — A single reward applied to every decision in a 4-round, 20-60 turn game. No credit assignment possible.
3. **Play owner encoding bug** — `play_owner_relative_pos=0` (self owns play) was encoded identically to "no play on table" because the one-hot only covered positions 1-4.

## What Changed

### `scout-bot/training.py`
- Removed replay buffer from training flow (class still exists, unused)
- `play_game()`: per-round rewards via `game.get_round_scores() / 10.0` instead of single game-end reward
- `compute_advantages()`: now normalizes advantages (zero-mean, unit-variance) for gradient stability. Removed unused `gamma` param.
- `ppo_update()`: entropy bonus now computed across ALL active heads (action_type + play_start/play_end/scout_insert), not just action_type. `value_loss_coeff` default 0.5→0.25.
- Added `round_num: int = 0` field to `StepRecord`

### `scout-bot/encoding.py`
- Play owner one-hot: 4 slots (1-4) → 5 slots (0-4), where 0 = "I own the play"
- `METADATA_SIZE` 34→35, changing `INPUT_SIZE` by +1. **Must train from scratch.**

### `scout-bot/main.py`
- Removed `ReplayBuffer` import and usage. PPO now trains on all ~1500 steps from the current iteration for K epochs (standard on-policy).
- Advantages computed once before epoch loop, same batch passed to `ppo_update` each epoch.
- Auto-resume: if `{save_dir}/latest.pt` exists, resumes automatically. `--resume` flag removed.
- Graceful Ctrl+C: signal handler, first press finishes iteration + saves, second forces exit.
- Eval: replaced binary win% with average score margin vs random (continuous, more informative). Chart updated.
- `value_loss_coeff` 0.5→0.25 in DEFAULT_CONFIG.

### `scout-bot/game_log.py`
- Table column moved after Action column (was at the end)
- Turn 0 row added showing all players' post-flip starting hands

## Decisions Made

- **Per-round rewards over game-end** — Round scores are the natural dense signal. Game-end reward removed entirely (cumulative score = sum of round scores, so winning rounds ≈ winning game). Can add game-end bonus later if needed.
- **value_loss_coeff 0.25** — Halved to prevent value loss from dominating shared backbone gradients early in training when value predictions are way off.
- **No within-round temporal credit assignment yet** — All steps in a round still share the same reward. GAE or discounting within rounds would help but was deferred to keep changes focused.
- **No learning rate schedule yet** — Constant 3e-4. Can add linear decay later.

## Known Issues (Not Yet Fixed)

From Opus review (`agent a0325f4d3f978c071`):

1. **No within-round credit assignment** — 10-30 decisions per round all get the same reward. High-variance advantages. GAE would help significantly.
2. **Per-step gradient accumulation is slow** — Each of ~1500 steps does individual forward+backward, 4 epochs = ~6000 passes. Batching the forward pass would be a major speedup.
3. **No cross-round value bootstrapping** — Each round treated as independent episode. The value function can't learn to sacrifice current-round score for long-term advantage.
4. **No learning rate schedule** — Constant LR over 1M iterations may cause oscillation later.
5. **Dead config keys** — `batch_size` and `replay_buffer_size` in DEFAULT_CONFIG are unused.

## Next Steps

1. **Run training from scratch** (new save-dir, not resuming old checkpoint — INPUT_SIZE changed).
2. **Monitor**: policy_loss should be non-zero now. Watch for reward trending upward and eval margin improving.
3. **If training works but is slow**: batch the forward pass in `ppo_update` (stack state tensors, single forward pass per batch).
4. **If learning plateaus**: add within-round temporal credit assignment (GAE or discount factor).

## Files to Read First

- `scout-bot/training.py` — core training logic, PPO update
- `scout-bot/main.py` — training loop, hyperparameters
- `scout-bot/encoding.py` — state encoding (INPUT_SIZE changed)

## Memory Update Needed

MEMORY.md still says `INPUT_SIZE changed 463→473`. It should reflect the current value (now 475 = 220 + 220 + 35).
