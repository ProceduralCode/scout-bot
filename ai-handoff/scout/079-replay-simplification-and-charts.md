# Replay Simplification and Chart Updates

## Task & State

Simplified the replay buffer from revalidation-based to sliding window, reorganized charts, fixed signal-vs-noise metric, and various chart improvements. All changes in main.py and training.py. v8_3 training in progress (~50 iterations, 30s/iter).

## What Changed

### training.py — ReplayBuffer rewrite

Replaced the revalidation-based replay buffer (~170 lines) with a sliding-window buffer (~50 lines). New class takes `window` param — keeps the last N cohorts, trims on `add_cohort`. No revalidation, no weighting, no alive/dead tracking. `sample_training_data(samples)` replaced with `all_samples()`. `load_state_dict` is backward-compatible (skips dead cohorts from old checkpoints, applies window trim).

Removed: `revalidate`, `check_and_prune`, `get_alive_cohorts`, `sample_training_data`, weight/alive/last_validated fields.

### main.py — Config

Removed: `cohort_check_interval`, `replay_check_perc`, `replay_margin_max_diff`, `min_replay_perc`.
Added: `replay_window: 1` (v8_3 running with 4).

### main.py — Signal vs Noise metric fix

`mean_rollout_std` now stores SE (std/√n) instead of raw std. The raw std doesn't change with rollout count — SE is what actually reflects target uncertainty. The `rollout_margin_spread` now subtracts `mean(SE²)` to remove noise inflation from observed spread (corrected spread ≈ true spread regardless of rollout count).

Labels changed to "Spread of means" and "SE of means".

### main.py — Chart reorganization

Row 2: Margin Predictions, Rollout Margins, Signal vs Noise, (hidden)
Row 3: Conditional Entropies, Dormant Neurons, (hidden), (hidden)

Removed: Replay Buffer time-series chart, Replay Cohorts by Age chart.

### main.py — Other chart changes

- Description fontsize 8 → 10
- `ylim=(0, None)` on MSE, Steps/Game, Conditional Entropies, Signal vs Noise
- Signal vs Noise legend moved to lower left
- `plot_line` gained optional `ylim` parameter
- `plot_multi` gained optional `legend_loc` parameter

### main.py — Chart/eval decoupling

Charts no longer saved inside `_run_eval`. Separate step after eval: every iteration for first 10, then every `eval_interval`. `_run_eval` no longer takes `chart_fn` param.

### main.py — Progress bar fix

`pbar.clear()` before eval, `pbar.refresh()` after — prevents bar showing during eval output.

## Decisions

- **Sliding window over revalidation**: revalidation MAE was dominated by rollout noise, not policy drift. Cohorts never got pruned. Simpler model: keep N iterations, drop old ones. Equivalent to a moving-window epoch.
- **SE instead of raw std for chart**: raw std doesn't change with rollout count, making the signal-vs-noise chart misleading. SE reflects actual target uncertainty.
- **Corrected spread**: observed spread of means is inflated by noise. Subtracting mean(SE²) removes the inflation, giving consistent spread values regardless of rollout count.
- **Removed replay buffer chart**: cohort-by-age chart showed the same info. Removed replay buffer time-series too — with sliding window it's just a constant.

## Next Steps

- Monitor v8_3 eval margins — still flat at -71 to -77 after 50 iters
- Rollouts are 86% of iteration time (26.4s/30.5s). Levers: `rollout_actions_per_sample` (currently 4+1), `rollouts_per_action` (currently 15), `curation_multiplier` (currently 20)
- Config has accumulated commented-out alternatives — could use cleanup

## Watch Out

- **v8_3 plays very differently from v8_1**: ~50/40 play/scout split (vs 76/16), 1.5 avg play length (vs 2.0), 78 steps/game (vs 46). Driven by temperature 0.5 (vs 0.0) and epsilon 0.1 (vs 0.05).
- **SE correction imprecise at low rollout counts**: std estimated from N samples has high variance when N is small. At 15+ rollouts it's fine.
- **`replay_window: 4` means each sample gets trained on ~4 times** (once per iteration it's in the window). With 16x augmentation, effective exposure is 64x per sample.
