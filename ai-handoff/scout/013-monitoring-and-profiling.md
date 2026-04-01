# Scout Bot: v3_4 Monitoring, Chart Improvements, and Profiling

## Task

Monitor v3_4 overnight run, improve chart/monitoring infrastructure, analyze training dynamics, and profile for speedup opportunities.

## Current State

v3_4 is running at ~1880 iterations. It's the strongest run so far — beating all prior bots and still improving. No code is broken or in-progress.

### v3_4 Training Status (iter 1881, smoothed)

- **eval_margin_v2_5**: -3.8 → +3.2 (steadily improving, best ever)
- **eval_margin_v3_1**: +0.4 → +7.0
- **avg_play_length**: 1.51 → 1.60 (slow but steady)
- **scout_play_len**: 1.52 → 1.55 (barely moving)
- **explained_variance**: 0.24 → 0.28 (appears flat, but see analysis below)
- **value_loss**: 0.088 → 0.080 (still declining)
- **entropy_scout_insert**: 1.51 → 0.87 (still the highest-entropy head)
- **steps_per_game**: 170 → 162 (stable, no over-scouting)

## What Changed

### Modified Files

- **`scout-bot/main.py`** — `_save_charts()` function reworked:
  - **Chart trimming**: first 10 iters trimmed when >100 iters, first 30 when >400 (`trim` variable, line ~89)
  - **Precomputed smoothing**: `trimmed` and `smoothed` dicts computed once at top of function (line ~93-102), shared by both charts and summary.txt. No re-smoothing.
  - **Moving averages on eval charts**: score margin and scout_play_len now show raw (alpha=0.25) + smoothed line, same as training charts.
  - **`summary.txt` generation**: written alongside `charts.png` with smoothed trajectory snapshots. 5 evenly-spaced data points, iteration header printed once, then just values per metric. Uses the precomputed `smoothed` dict.
  - **Ctrl+C behavior**: `KeyboardInterrupt` now prints iteration number and returns immediately — no saving on interrupt. Periodic saves (every `save_interval`) still happen during training.

- **`scout-bot/profile_iteration.py`** — New file. Profiles one training iteration with fine-grained timing breakdown. Usage: `python scout-bot/profile_iteration.py scout-bot/v3_4/latest.pt`

### Deleted Files

- `scout-bot/inspect_run.py` — replaced by `summary.txt`
- `scout-bot/inspect_metrics.py` — replaced by `summary.txt`

## Key Findings

### Explained Variance Is Misleading

EV looks plateaued at ~0.28, but value_loss is still declining (0.088 → 0.080). This isn't contradictory — as the agent improves via self-play, games get more even and `reward_std` shrinks (0.58 → 0.56). The value function is improving in absolute terms, but predicting a target with shrinking variance, so the ratio (EV) stays flat. The system is still learning.

### Reward Distribution: divide-by-N Is Correct

We explored changing uniform reward from `round_reward * 0.7 / N` (per step) to constant `round_reward * 0.7` (every step gets same amount). The constant version fixes signal dilution in long games, but breaks the value function — V would need to predict `margin × steps_remaining` instead of just `margin`. The return at each step becomes position-dependent. Divide-by-N keeps total return ≈ round_margin, so V has a clean prediction target. Reverted to divide-by-N.

### Profiling Results

Game generation: **15.1s** (88% of iteration). PPO: **2.0s** (12%). GAE: **14ms**.

Within game generation (100 games, ~16K steps):

| Component | Time | % of gen | Calls | Per-call |
|-----------|------|----------|-------|----------|
| forward_pass | 8.8s | 58% | 58,776 | 0.15ms |
| sampling | 2.3s | 15% | 42,711 | 0.054ms |
| masks | 1.8s | 12% | 42,711 | 0.043ms |
| encoding | 1.1s | 7.5% | 17,468 | 0.065ms |
| game_logic | 0.2s | 1.3% | 17,868 | 0.011ms |

The bottleneck is 58K tiny individual forward passes — all Python/PyTorch dispatch overhead, not actual compute. Game engine itself is negligible (1.3%).

## Next Steps

1. **Let v3_4 continue running** — eval margins are still trending up at 1880 iters. Worth running to 3-5K to see if it plateaus or keeps climbing.
2. **Batch forward passes during game play** — the 10x speedup path. Play all 100 games simultaneously, gather all pending decisions, do one batched forward pass, distribute results. Would require refactoring `_play_round`/`_play_turn` into a vectorized game loop. The game engine is fast; it's the 58K per-step forward passes that kill throughput.
3. **If v3_4 plateaus** — representation bottleneck is the hypothesis. Architectural changes (attention, CNN, match hints in encoding) are the next direction.

## Watch Out

- **`summary.txt` metrics with fewer entries**: some metrics may have fewer entries than `iters` (e.g., added mid-run). The summary writer handles this with per-metric `_snap_idx` fallback (line ~251 in main.py).
- **Ctrl+C no longer saves**: if training is interrupted, latest.pt reflects the last periodic save, not the current iteration. This was intentional — user preferred immediate exit.
- **`profile_iteration.py` S&S handling is simplified**: the S&S path doesn't produce StepRecords (it's just timing the operations), so it won't affect profiling accuracy but isn't usable for training data collection.

## Files to Read First

- `scout-bot/v3_4/summary.txt` — current training metrics (smoothed)
- `scout-bot/main.py:83-102` — precomputed smoothing and chart trimming
- `scout-bot/main.py:236-270` — summary.txt generation
- `scout-bot/profile_iteration.py` — profiling script and results
