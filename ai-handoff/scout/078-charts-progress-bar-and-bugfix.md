# Charts, Progress Bar, and Bugfix

## Task & State

Added new charts and metrics, training progress bar, cleaned up output, and fixed a CUDA assertion crash. All changes are in main.py and training.py. Two training runs in progress: v8_1 (~134 iterations) and v8_2 (fresh start, hit the CUDA bug, needs restart with fix).

## What Changed

### main.py — New charts and chart reorganization

4×4 chart grid reorganized. New layout for rows 2-3:

- **Row 2** (time series): Conditional Entropies, Dormant Neurons, Rollout Margins histogram, Rollout Signal vs Noise
- **Row 3** (snapshots + replay): Margin Predictions histogram, Replay Buffer (cohorts+samples over time), Replay Cohorts by Age, (hidden)

New/changed charts:
- **Rollout Signal vs Noise** (`axes[2,3]`): plots `rollout_margin_spread` (std of all target margins = signal) and `mean_rollout_std` (mean per-target std = noise) on same axes. Red above green = training on noise.
- **Replay Cohorts by Age** (`axes[3,2]`): line chart with fill showing total vs effective (weighted) samples per cohort age. X-axis inverted so oldest on left. Replaces the separate "Replay Buffer Age" histogram and "Cohort Composition" bar chart that were previously two charts.
- Bumped all chart description text from fontsize 7 to 8.
- Replaced LR chart (was `axes[2,3]`) — LR is constant at 0.0003 and not useful to plot.

New metrics in `metrics_history`:
- `mean_rollout_std`: mean of per-action rollout standard deviations across all samples
- `rollout_margin_spread`: std of all rollout margin values (target spread)
- `_cohort_ages`, `_cohort_total_samples`, `_cohort_eff_samples`: snapshot data for cohort chart

### main.py — Training progress bar

- Added `from tqdm import tqdm` import
- Outer training loop wrapped with `tqdm(range(...), desc="training", unit="iter")`
- `pbar.clear()` at iteration start, `pbar.refresh()` after training — prevents two bars showing simultaneously when inner bars (games, rollouts) are active
- `pbar.set_postfix(mse=...)` updated each iteration
- All `print()` calls inside the training loop and `_run_eval()` changed to `pbar.write()` / `tqdm.write()` to avoid breaking the progress bar display

### main.py — Output cleanup

- Removed 4 individual `[mem]` diagnostic prints (after play, curation, snapshots, rollouts)
- Removed "Curated N from M samples" print
- Added `mem=X.XGB` to the iteration summary line instead
- Removed the interrupt checkpoint save — was redundant with `log_interval: 1` (every iteration already saves), and had a bug where it saved with the incomplete iteration's number

### training.py — CUDA assertion fix

`_select_action_q_batched`: games with no legal actions (all-False mask) caused `argmax` over all-inf values, triggering CUDA assertion `input[0] != 0` in TensorCompare.cu. Fix: detect empty-mask rows and set a dummy legal action before argmax/softmax. Caller already checks `has_action` and skips those games.

## Decisions

- **Removed interrupt save rather than fixing it**: with `log_interval: 1`, checkpoints save every iteration. The interrupt save was redundant and introduced an iteration-numbering bug (saved incomplete iteration number, causing a skip on resume).
- **Kept iteration numbering 1-based**: user clarified they wanted pre-training eval data to show as iteration 0 on charts (which it already does), not 0-based iteration numbering throughout.
- **Merged two replay age charts into one**: "Replay Buffer Age" histogram and "Cohort Composition" were showing similar data (total vs effective samples by age). Combined into single chart with two lines.

## Next Steps

- Restart v8_2 with CUDA fix applied
- Address replay buffer revalidation problem (see below)

## Watch Out

- **Replay buffer never prunes**: all cohorts settle to ~0.5 weight regardless of age. The revalidation MAE (~0.2) is dominated by rollout noise (SE ≈ 0.65/sqrt(15) ≈ 0.17), not policy drift. The 0.3 kill threshold is never reached. Buffer grows without bound — 131 alive cohorts at iteration 134. This needs a design fix: either age-based cutoff, higher rollout count during revalidation, or a different staleness metric.
- **v8_1 eval margins flat at -68 to -76**: bot is losing badly to all eval opponents and not visibly improving. MSE is still dropping (0.051) but this isn't translating to better play yet. Could be too early, or could indicate a problem.
- **Rollout noise is high**: mean_rollout_std ≈ 0.65, rollout_margin_spread not yet measured in v8_1 (metric was just added). The signal-to-noise chart will show if target rankings are reliable.
