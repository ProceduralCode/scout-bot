# Pipeline Verification & Post-Bugfix Hyperparameter Reset

## Task & State

Implemented the three follow-up items from the temperature bugfix session: ratio=1.0 assertion, pipeline verification script, and hyperparameter reset. Training run v7_8 is in progress with the new params.

## What Changed

### Modified files
- `scout-bot/main.py`:
  - Added epoch-0 first-batch ratio check after `ppo_update_v6` call (~line 973). Uses `first_batch_ratio` from the return dict — checks only the very first mini-batch before any gradient step, not the full-epoch average. Threshold: 0.01 deviation from 1.0.
  - PARAMS updated for v7_8: `ppo_epochs` 1->3, `gae_lambda` 0.95->0.98, `sampling_temperature` 2.5->1.0, `save_dir` -> "bots/v7_8". `entropy_bonus` kept at 0.03.

- `scout-bot/training.py`:
  - `ppo_update_v6` now returns `first_batch_ratio` — the mean ratio from the very first mini-batch, before any gradient step has been applied. This is the clean signal for the ratio=1.0 invariant.

### New files
- `scout-bot/verify_pipeline.py` — standalone pipeline verification script. 4 checks:
  1. Permutation table group properties (FULL_PERM is cyclic, inverses work, HAND_SHIFT valid)
  2. State/mask round-trip (rotate by k then -k recovers original)
  3. Policy equivariance under rotation (informational — measures how well the network treats rotations consistently; not architecturally guaranteed, improves with training)
  4. Initial-state value near zero (no inherent player advantage)

  Usage: `python -u scout-bot/verify_pipeline.py [checkpoint_path]`

### context.md
- Updated ratio=1.0 invariant section to reflect the check is now in place.

## Decisions

- **Ratio check targets first mini-batch only.** The original check compared the full-epoch mean_ratio, but with 3 PPO epochs and ~5 mini-batches per epoch, within-epoch gradient drift pulled the mean to ~0.966 even with correct old_log_probs. The first mini-batch (before any gradient step) is the clean signal.

- **No entropy annealing.** Decided to keep fixed entropy bonus for now. Temperature + entropy bonus = double exploration pressure (discussed in session). At T=1.0, entropy bonus is the sole exploration mechanism. Annealing is a tuning optimization for later, not a fundamental dynamics change.

- **Equivariance check is informational, not pass/fail.** The network isn't architecturally equivariant — augmentation encourages learning equivariance through data, but a random init or early-training network will have high deviation. Fresh network: max_dev ~2.3, v7_7 at 267 iters: max_dev ~77 (moved away from symmetric init but hasn't learned symmetry back yet).

## Next Steps

1. Monitor v7_8 training — watch kl_batch_frac (declining toward 0.6), entropy (settling around 1.5), eval margins
2. If entropy drops too fast or eval margins stall, bump entropy_bonus to 0.05
3. Run `verify_pipeline.py` on v7_8 checkpoint periodically to track equivariance improvement
4. Once v7_8 has enough iterations for comparison (~200+), assess whether the param changes helped vs v7_7

## Watch Out

- The v7_8 run was producing `WARNING: epoch 0 mean ratio=0.9661` with the old check (full-epoch average). This was a false alarm from within-epoch gradient drift, not a bug. The fix (first-batch-only check) should silence it. If the WARNING still appears after this fix, there's a real old_log_prob recording issue.
- kl_batch_frac at iter 170 is 0.617 — the 3rd PPO epoch is mostly getting cut by KL early stopping. This means ~2/3 epoch utilization, which is expected with KL as a safety valve. If it drops below 0.5, the extra epochs aren't buying much.
