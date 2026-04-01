# Diagnostics Overhaul and Data Starvation Analysis

## Task & State

Started by interpreting v6_3 diagnostic output, discovered the original diagnostics were measuring the wrong thing, replaced them, then identified the likely core training bottleneck: data starvation from rollout compute overhead. No decision yet on path forward.

## What Changed

- **scout-bot/training.py** — `play_games_with_rollouts_v6()` now returns 4 values: `(records, advantages, raw_advantages, avg_rollout_margin_std)`. Added sum-of-squares tracking during snapshot value computation to calculate per-snapshot margin variance.

- **scout-bot/main.py** — Multiple changes:
  - `_compute_diagnostics()` — new signature: `(network, records, raw_advantages, rollout_margin_std, rollouts_per_state)`. Replaced per-type advantage means with: `adv_std`, `adv_abs_mean`, `adv_p10`, `adv_p90`, `rollout_noise` (estimated noise from finite rollouts), `snr` (signal-to-noise ratio). Kept policy preference and value head accuracy metrics.
  - `_save_diagnostic_charts()` — new 2x2 layout: Signal vs Noise (adv_std and rollout_noise lines), Policy Preference (unchanged), Advantage Distribution (p10-p90 shaded band + abs_mean), Value Head Quality (MAE + correlation on twin y-axes).
  - `metrics_history` initialization — new `diag_*` keys replace old `diag_raw_adv_*` keys. Old keys from checkpoint are loaded harmlessly but not referenced.
  - Console output — `diag: adv_std=X noise=X SNR=X ...` replaces old per-type format.
  - Training loop — unpacks 4th return value from rollout function, passes to diagnostics.

**Note:** PARAMS in main.py currently show `rollouts_per_state: 100` and `diagnose: False`. These may differ from what the running v6_3 process was started with (25 and True). PARAMS override saved config on resume except for architecture params.

## Decisions

- **Removed per-type advantage means** (diag_raw_adv_play1/2/3p/scout). These averaged advantages by action type across all game states, which washes out per-state signal. Action types aren't systematically better/worse regardless of state — singles are sometimes better, pairs sometimes better. The original signal tests worked because they compared actions within the same state.

## Key Finding: Data Starvation

Compared training data throughput across versions:
- v4_2: 400 games/iter, no rollouts, ~25s/iter → ~57,600 unique decisions/hour
- v6: 10 games/iter, 50 rollouts/state, ~82s/iter → ~440 unique decisions/hour
- That's ~130x less unique training data per wall-clock hour.

×16 rotation augmentation doesn't compensate — it teaches hand rotation equivalence but doesn't add new game situations. The rollout compute (95% of iteration time) produces high-quality per-decision advantages but from only 10 unique games.

v4_2 reached +18.66 vs v1 after 19,200 iterations. v6_3 at 535 iterations is at -0.16 vs v1 with entropy collapsing (0.90) and clip/KL elevated (0.20/0.035).

## Discussion (No Resolution)

Three approaches discussed for fixing the throughput problem:

1. **Go back to GAE** — 100x more games/hour. Signal quality per decision drops (depends on value head) but total signal per hour increases massively. v3/v4 trained successfully this way.

2. **Value-head search (expert iteration)** — Evaluate all legal actions at each decision using V(after). 30-150x cheaper than rollouts per decision. User doesn't trust the value head to capture advanced strategy.

3. **Hybrid: GAE bulk + rollout calibration** — Play 200 games/iter with GAE (cheap, ~15s). Play 5 games with full rollouts (~40s). GAE games provide volume, rollout games provide strategy discovery and value head calibration. ~55s/iter, 20x more unique decisions.

User's key concern: the value head won't pick up advanced strategies. Rollouts have a genuine advantage for strategy discovery — during rollouts, the policy sometimes accidentally plays strong combos, and those rollouts score higher, providing signal for the setup move. The value head can only reflect what it's already learned, not discover new strategy.

No decision made on which approach to pursue.

## v6_3 State at Session End

~535 iterations, training still running with old code (pre-diagnostic changes). Summary.txt shows:
- Entropy: 1.52 → 0.90 (collapsing)
- Clip fraction: 0.07 → 0.20, approx KL: 0.006 → 0.035 (elevated)
- Explained variance: 0.41 → 0.51 (modest)
- Eval: v1_4 -0.16, v2_5 -3.74, v3_4 -12.71, v4_2 -19.28
- Play length trending toward more singles (53%), fewer pairs (42%)
