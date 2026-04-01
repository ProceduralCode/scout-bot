# GAE Verification and Value Head Probing

## Task & State

Verified GAE correctness, then built diagnostic tooling to probe value head quality. GAE is correct. Value head predictions are magnitude-compressed. OLS shows the trunk features contain more signal than the value head is extracting, but the OLS result may be overfit (877 samples, 129 parameters). No code changes to the training pipeline.

## What Changed

New diagnostic scripts (not part of training pipeline):

- `scout-bot/verify_gae.py` — Replays `compute_gae()` against an independent reimplementation. Confirmed zero numerical difference across 275 records from 3 games. GAE is correct.
- `scout-bot/value_head_probe.py` — Plays games from a checkpoint, runs 40 rollouts per decision point, compares value head predictions to empirical V. Plots per-game timelines with multiple rotation offsets. Usage: `python value_head_probe.py [checkpoint_path]`
- `scout-bot/value_warmup_test.py` — Freezes trunk, trains value head on rollout targets, compares Adam convergence to OLS closed-form ceiling. Usage: `python value_warmup_test.py [checkpoint_path]`

Charts saved to `scout-bot/v6_5/`:

- `latest_value_probe.png` — timeline of predicted V vs empirical V across 5 games
- `latest_value_warmup.png` — supervised warmup results with OLS comparison

## Measurements (v6_5, iteration 224)

Value head probe (5 games, 209 decisions, 40 rollouts each):

- Correlation between predicted V and empirical V: 0.73
- EV against rollout targets: 0.35
- Predicted range: [-0.1, 0.3], empirical range: [-1.0, 1.0]
- Rotation invariance spread (avg offset std): 0.011
- Empirical rollout noise (avg std): 0.37

Supervised warmup test (20 games, 877 decisions, 40 rollouts each):

- Value head EV before supervised training: 0.41
- Value head EV after 200 epochs Adam (lr=0.001): 0.41 (no improvement)
- Adam pred_std: 0.155 vs target_std: 0.432
- OLS closed-form on same data: EV=0.66, pred_std=0.352, range [-1.30, 1.20]
- Caveat: OLS has 877 samples / 129 parameters = 6.8 samples/param. Possible overfitting — not validated with held-out data.

## Open Questions

- Why does Adam plateau at EV 0.41 when OLS achieves 0.66 on the same data? Loss was still slowly decreasing at epoch 200. Could be slow convergence, could be OLS overfitting.
- How much of the OLS 0.66 is overfitting? A train/test split would answer this.
- During actual PPO training, EV is 0.22. The gap from 0.22 to 0.41 (supervised) is unexplained — noisy GAE targets, shared optimizer, policy gradient interference are all possible factors but untested.
- Would a deeper value head (MLP) help? Untested.

## Next Steps

1. **Validate OLS result** — split data into train/test to check if 0.66 is inflated by overfitting.
2. **Debug Adam convergence** — try higher LR, more epochs, or SGD to see if Adam is the problem or the 0.41 plateau is real.
3. **If OLS result holds** — the trunk features genuinely support better value prediction than the value head achieves during training. Investigate why the value head can't reach its ceiling during PPO.
4. Items 3-4 from the verification plan (fix probes 10-11, PPO direction probe) are still pending.
