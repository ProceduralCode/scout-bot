# Pipeline Diagnostic & GAE Value Loss Weight

## Task & State

Built a diagnostic script to examine signal flow inside a real training iteration. Findings led to implementing `gae_vloss_weight` — a parameter controlling how much GAE-sourced samples contribute to the value head's training loss.

## What Changed

### New files
- `scout-bot/pipeline_diagnostic.py` — loads a checkpoint, generates one iteration of games (GAE + rollout), and reports on:
  1. Episode structure (decisions/episode, action type census by step position)
  2. Credit assignment (advantage magnitude by step position, TD residuals)
  3. Signal dilution (advantage stats split by play/scout/sns action type)
  4. Value function accuracy by game stage (V_pred vs V_target, GAE vs rollout)
  5. Reward distribution (where reward lands relative to episode end)
  6. Policy entropy by action type (full, play-region, scout-region)
  7. Gradient contribution by action type (per-type gradient norms from one PPO step)

  Usage: `python -u scout-bot/pipeline_diagnostic.py [checkpoint_dir]`

### Modified files
- `scout-bot/main.py`:
  - New PARAMS entry: `"gae_vloss_weight": 1.0` (default preserves current behavior)
  - Training loop creates `v_weights` list: GAE samples get `gae_vloss_weight`, rollout samples get `1.0`. Passed through augmentation → batch prep → PPO update.

- `scout-bot/training.py`:
  - `augment_rotation_v6` — new `v_weights` param, replicates weights like advantages. Returns 3-tuple now: `(steps, advantages, v_weights)`.
  - `prepare_ppo_batch_v6` — new `v_weights` param, includes `v_weight` tensor in batch dict.
  - `subsample_batch_v6` / `concatenate_batches_v6` — carry `v_weight` through.
  - `_ppo_step_v6` — new `v_weight` param. When provided, value loss uses weighted MSE: `(per_sample_loss * weight).sum() / weight.sum()`. When `v_weight=None`, unchanged `F.mse_loss`.
  - `ppo_update_v6` — extracts `v_weight` from batch, passes to `_ppo_step_v6` mini-batches.

## Diagnostic Findings (v7_8, iteration 116)

- Scout is 29% of decisions with advantage magnitude equal to play's (0.24 vs 0.23). Signal dilution is not a problem.
- Value head vs rollout ground truth: corr=0.80, explained variance=0.55. vs GAE targets: corr=0.45, EV=0.16. The value head can learn — the issue is GAE target quality.
- V_pred range [-0.83, 0.67] vs actual outcome range [-1.45, 1.43]. Value head predictions are compressed toward zero, especially for extreme positions.
- Play and scout gradient norms partially cancel when combined (play=0.09, scout=0.44, combined=0.14).

## Decisions

- `augment_rotation_v6` now returns a 3-tuple. Both call sites in `main.py` updated. The third element is `None` when `v_weights` is not provided.
- Value loss weighting divides by `v_weight.sum()` not batch size, so setting `gae_vloss_weight=0` computes value loss only over rollout samples with correct normalization.

## Next Steps

1. Run a training experiment with `gae_vloss_weight: 0.0` (rollout-only value training) and compare value head accuracy trajectory against the default.
2. Consider increasing `rollout_fraction` to give the value head more ground-truth data per iteration.
3. Re-run `pipeline_diagnostic.py` on a more-trained checkpoint to see if the value compression and gradient conflict resolve with training.

## Watch Out

- The `augment_rotation_v6` return type changed from 2-tuple to 3-tuple. Any other callers (probes, scripts) that destructure the return will need updating. Currently only `main.py` calls it.
- With `gae_vloss_weight=0` and `rollout_fraction=0`, the value head gets zero gradient. This is a valid configuration (user chose no rollouts and no GAE value training) but worth noting.
