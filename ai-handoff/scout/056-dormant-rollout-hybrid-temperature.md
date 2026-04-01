# Dormant Neurons, Rollout Hybrid, Unnormalized Advantages, Temperature Scaling

## Task & State

Continued from session 055 (GELU, KL early stopping, detached value head — code-complete but untested). This session made four additional changes: fixed the dead neuron metric for GELU, added hybrid rollout/GAE training, removed advantage normalization, and added temperature scaling for exploration. All changes are code-complete. A training run was started mid-session (v7_2 or v7_3) but the code was modified further after that — a fresh run with the final code has not been verified.

## What Changed

### Dormant neuron metric (`main.py`)

`_count_dead_neurons()` → `_count_dormant_neurons(threshold=0.01)`. Old metric counted neurons with all-negative output (meaningful for ReLU where gradient is exactly 0). New metric computes `mean(|activation|)` per neuron across a 2000-sample batch and flags neurons below threshold. This captures "functionally not contributing" regardless of sign — correct for GELU where negative outputs still carry gradient.

All references renamed: `dead_neurons_*` → `dormant_neurons_*` in metrics_history, charts, and console log.

### Rollout fraction hybrid (`main.py`, `training.py`)

New param `rollout_fraction` (default 0.1). Splits `games_per_iteration` into GAE games and rollout games. The `elif ev == 6:` block now:
1. Plays `n_gae` games with `play_games_v6()`, computes GAE advantages
2. Plays `n_rollout` games with `play_games_with_rollouts_v6()` — these get rollout-based value targets and advantages
3. Offsets rollout game_ids by `n_gae` to avoid collision
4. Combines records and advantages into a single batch

Rollout games provide high-quality value targets (empirical rollout means) for the detached value head, and high-quality advantages for the policy. This breaks the chicken-and-egg loop where the value head trains on noisy GAE targets.

Rollout timing is tracked separately and printed as `ro=X.Xs` in the console log.

**Rollout EV metric**: For rollout games, computes explained variance of value head predictions (`predicted_value`) against rollout ground truth (`value`). Stored in `metrics_history["rollout_ev"]` and overlaid on the Explained Variance chart as a second line (blue "Rollout" vs green "GAE"). The GAE EV is partially circular (value head evaluated against its own bootstrapped targets), while rollout EV measures accuracy against empirical outcomes.

### Removed advantage normalization (`training.py`, `main.py`)

`compute_gae()` now returns raw unnormalized advantages (2 values instead of 3 — `advantages, returns`). `play_games_with_rollouts_v6()` also returns unnormalized advantages (3 values — `records, advantages, avg_margin_std`).

Rationale: the value function predicts margin (centered at 0 = neutral). Positive advantages mean the action genuinely improved expected margin. Zero-centering forces half the batch to have negative advantages regardless of actual quality — distorting the signal. The reward structure provides natural scaling, and clip epsilon + KL early stopping + LR decay already control update magnitude.

All callers across ~8 script files were updated for the new return signatures. Scripts that printed `adv_std` now compute it locally via `np.std(advantages)`.

### Temperature scaling (`training.py`, `main.py`)

New param `sampling_temperature` (default 1.5). During data collection, logits are divided by temperature before `masked_sample` and `masked_log_prob`. This flattens the sampling distribution — good plays are still more likely, but second/third choices happen more often. The tempered log prob is recorded as `old_log_prob`, so PPO importance ratios are correct.

Applied in:
- `play_games_v6()` — only for training seats (`p < training_seats`); opponent pool networks play at T=1
- `play_games_with_rollouts_v6()` — all seats (all use the training network)
- NOT applied in `rollout_from_states_batched_v6()` — rollout completions play at T=1 for accurate value estimates

### Charts fix (`main.py`)

Removed `if len(iters) < 2: return` guard from `_save_charts()` so charts render even before training data exists (e.g., after iteration 0 eval). All plotting helpers already guard with `if key in trimmed` before drawing, so empty panels are fine.

### Chart layout change (`main.py`)

The hidden `axes[3, 3]` panel was removed. Explained Variance panel (`axes[3, 2]`) changed from `plot_line` to `plot_multi` with two series: "GAE" (green, existing) and "Rollout" (blue, new).

## Decisions

- **No entropy floors or zero_scout_policy_grad for v7**: Session 055 ablations showed these were protective against ReLU dead neurons. With GELU eliminating the dead neuron mechanism, there's no reason to carry them forward. Both stay off.
- **Eval interval stays at 5**: User preference, not diagnostic-run pacing.
- **No advantage normalization**: Deliberate removal, not an oversight. The margin-based reward structure is naturally centered and scaled.
- **Temperature 1.5 as starting point**: Moderate flattening. If the peak-then-regress pattern disappears, the regression was entropy-driven.
- **Dormant threshold 0.01**: Small enough that the neuron is barely contributing, but not requiring exact zero. Empirically reasonable for GELU where outputs are in the range of typical LayerNorm-scaled activations.

## Next Steps

1. **Start a fresh training run** with all changes. PARAMS need: fresh `save_dir`, verify `total_iterations` is 1M, check `rollout_fraction` and `sampling_temperature` are set to desired values.
2. **Watch for**: dormant neuron counts (should be low with GELU), rollout EV (should be better signal than GAE EV), entropy staying healthy (temperature should prevent collapse), `ro=` timing in console (cost of rollout fraction).
3. **If temperature works**: the peak-then-regress pattern from previous runs should disappear. If it still regresses, the cause isn't entropy-driven.

## Watch Out

- **`compute_gae` return signature changed**: 2 values now, not 3. All known callers fixed, but any new script importing it will need to match.
- **`play_games_with_rollouts_v6` return signature changed**: 3 values now, not 4 (no separate `raw_advantages` since advantages are no longer normalized).
- **No checkpoint compatibility** with pre-GELU runs (from session 055 — still applies).
- **`rollout_ev` only recorded when `n_rollout > 0`**: The metric has fewer entries than `iteration` in metrics_history. The chart trimming/alignment code handles this (right-aligns shorter metrics).
