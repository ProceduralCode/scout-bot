# Entropy Floors and Training Diagnostics

## Task & State

Investigated probe 10 failure, then shifted to diagnosing v6_7 training run which collapsed overnight due to entropy collapse. Implemented entropy floors in `_ppo_step_v6` (training.py complete, main.py not yet wired). User adjusted PARAMS to reduce gradient steps per iteration.

## What Changed

- `scout-bot/training.py` — `_ppo_step_v6()` now accepts `entropy_floors` and `entropy_floor_coeff` params. Computes per-region entropy (play actions 0-255, scout actions 256-319) with gradients before the loss, applies quadratic penalty when mean entropy drops below the floor (only for samples with 2+ legal options in that region). `ppo_update_v6()` passes these through and aggregates `entropy_floor_penalty` metric.

- `scout-bot/main.py` — User changed PARAMS: `mini_batch_size` 8192 → 32768, `ppo_epochs` 2 → 1, `replay_past` [0.4, 0.2, 0.1, 0.1, 0.1] → [], `entropy_bonus` 0.05 → 0.08, `value_loss_coeff` 0.25 → 0.5. `entropy_floors` is still None — needs to be wired to `ppo_update_v6` call (line ~786) and set to non-None value in PARAMS.

## Measurements

### Probe 10 investigation

Supervised learning test: FlatScoutNetwork with attention trained to predict best insertion position (cross-entropy loss, 2000 unique scenarios, 16x rotation augmentation = 32K samples):
- Test accuracy peaked at 40.6% (vs 8.3% random baseline) at epoch 40, then overfitted
- Plain FC [128, 64] without attention: 9.4% (barely above random)
- Conclusion: attention CAN learn insertion quality patterns, but needs ~2000+ unique scenarios. The probe generates only 50 per iteration — massively data-starved for this task.

### v6_7 overnight run (1155 iterations)

Peaked around iter 593 (eval_margin_v1_4: -1.14), then degraded. Entropy collapsed: overall 1.94 → 0.06, scout 2.11 → 0.01, play 1.10 → 0.05. Policy loss and KL exploded around iter 874. Model unrecoverable after that.

At peak (iter 593): clip_fraction=0.29, approx_kl=0.18 — both far above healthy PPO ranges. With mini_batch_size=8192 on ~120K records (64K augmented + replay), that was ~30 gradient steps per iteration. The old v1-v4 training did 4 full-batch steps.

entropy_bonus=0.25 was tried earlier and "completely disabled the network" (user's words). entropy_bonus=0.05 was too weak to prevent collapse over 1000+ iterations.

## Decisions

- **Probe 10 is a data regime problem, not architecture.** Supervised learning proves the attention network can learn insertion quality, but needs far more data than the probe provides. Not an architecture failure.

- **Entropy floors over entropy bonus for collapse prevention.** Entropy bonus is a blunt instrument (constant pressure toward uniform, disrupts learning at high values). Floors only activate when entropy drops below threshold — surgical safety net. Already existed for old ScoutNetwork, now added to v6 path.

## Next Steps

1. **Finish wiring entropy floors in main.py.** The `ppo_update_v6` call at line ~786 needs `entropy_floors=cfg.get("entropy_floors")` and `entropy_floor_coeff=cfg.get("entropy_floor_coeff", 1.0)`. Set PARAMS `entropy_floors` to `{"play": 0.5, "scout": 0.3}` (or discuss values). Update the commented-out example to show v6 keys.

2. **Start a fresh v6_8 run** with the new PARAMS (32K mini-batch, 1 epoch, no replay, entropy floors). This should give ~2-4 gradient steps per iteration, similar to the old training dynamics.

3. **Monitor KL and clip fraction** as the primary health indicators. Healthy PPO: KL < 0.03, clip < 0.15.

## Watch Out

- The floor values (play: 0.5, scout: 0.3) are proposed but untested. At peak v6_7 performance, play entropy was 0.69 and scout was 0.46. The floors should be below peak values to avoid fighting useful learning.

- `entropy_floors` is not in the checkpoint-preserved params list (like `layer_sizes`, `encoding_version`, `attention`). It overrides from PARAMS on resume, which is the intended behavior.

- The v6_7 best checkpoint (~iter 500-600) still exists but was trained with unstable dynamics (high KL). Resuming from it may carry over a corrupted policy. Fresh start is cleaner for testing new settings.
