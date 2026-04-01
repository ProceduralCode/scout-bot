# Scout Bot: Diagnostics, Pool Persistence, and Tuning

## What Was Done

### 1. Added comprehensive training diagnostics

**`scout-bot/training.py`** — `ppo_update()` now returns a dict with 10 metrics (was a 3-tuple):
- Original: `policy_loss`, `value_loss`, `entropy`
- New PPO health: `clip_fraction`, `approx_kl`, `explained_variance`
- New per-head entropy: `entropy_action_type`, `entropy_play_start`, `entropy_play_end`, `entropy_scout_insert`
- Per-head entropies are captured before being scattered into the combined entropy tensor (lines ~325-390)

**`scout-bot/main.py`** — Training loop and charts:
- PPO loop consumes the dict return, averages all metrics across epochs (`ppo_sums` / `ppo_avg` pattern)
- Computes behavioral metrics per iteration: action type distribution (play/scout/S&S %), steps per game, raw advantage std
- Console log includes `clip`, `kl`, `ev` (explained variance)
- Charts expanded from 3x2 (6 panels) to 4x3 (12 panels), each with italic description underneath
- Old checkpoints handled gracefully — new metric keys default to empty lists via merge logic

Chart layout:
- Row 0: Avg Reward, Value Prediction, Score Margin vs Random
- Row 1: Policy Loss, Value Loss, Per-Head Entropy (4 lines)
- Row 2: Clip Fraction, Approx KL, Explained Variance
- Row 3: Action Type Distribution (3 lines), Steps Per Game, Advantage Std (pre-norm)

Helper function `_smooth()` extracted from inline smoothing logic.

### 2. Opponent pool persistence

**`scout-bot/training.py`** — Added `state_dicts()` and `load_state_dicts()` methods to `OpponentPool`

**`scout-bot/main.py`**:
- `_save_checkpoint()` takes optional `pool=` param, saves pool state dicts
- Pool is saved to `latest.pt` (both in-loop and final save), NOT to `best.pt` or periodic snapshots
- On resume, pool is restored from checkpoint before training starts
- Falls back to seeding with current network if no saved pool exists

Note: checkpoint is loaded twice on resume (once for network/optimizer, once for pool). Minor inefficiency, works correctly.

### 3. Performance profiling

Profiled 50 games with pyinstrument. Key findings:
- ~60% of play time is network inference + sampling (trunk 14.6%, heads 23%, conditioning 13%, masked_sample 19%)
- ~25% is encoding + mask computation
- ~15% is misc Python overhead
- Batching the trunk alone would only save ~14% — the cost is spread across many small PyTorch calls
- PPO update is already batched and fast (0.3s vs 13.6s play time)

## Key Decisions

- **Diagnostics over optimization**: Prioritized visibility into training dynamics before making architectural changes (GAE, batching, etc.)
- **Dict return from ppo_update**: Chose dict over named tuple for extensibility. Breaking change — old code destructuring `pl, vl, ent =` won't work.
- **Pool saved only to latest.pt**: Best.pt and periodic snapshots don't include pool to keep checkpoint sizes reasonable.

## Analysis of Third Run (scout-bot/third/charts.png)

- Reward declined from 1.3→0.85 over 400 iters — self-play artifact (opponents improving), NOT regression
- Eval margin vs random climbed to ~55 — real skill improving
- Value loss dropped fast (2.0→0.25) but explained variance was ~0 — value function predicts a constant (mean reward), doesn't distinguish states
- Clip fraction was 0.00 — policy barely moves per iteration, LR too low
- Entropy settled at ~1.6 from ~2.8 — reasonable, not collapsed

## Recommended Hyperparameter Changes for Next Run

Two changes, both addressing the same core issue (value function not learning, policy updates too small):

1. `learning_rate`: 3e-4 → **1e-3** — should push clip fraction to 0.05-0.2 range
2. `value_loss_coeff`: 0.25 → **0.5** — give value head more gradient influence on shared trunk

Pass via CLI: `python main.py --lr 1e-3` (value_loss_coeff needs to be changed in DEFAULT_CONFIG or added as a CLI arg)

Note: `value_loss_coeff` is not currently a CLI argument — only in DEFAULT_CONFIG. May want to add `--value-loss-coeff` to argparse.

## What's Left / Future Directions

Discussed but not implemented (in rough priority order):
1. **GAE** — biggest known training gap, would improve credit assignment
2. **Heuristic bootstrap** — behavioral cloning from rule-based bot to skip random phase
3. **Entropy annealing + LR schedule** — low effort standard improvements
4. **Batching game inference** — moderate effort (~1-2 days), ~25-30% speedup
5. **Shallow search at training time** — AlphaZero-lite approach
6. **Auxiliary prediction heads** — opponent modeling, round outcome prediction
7. **Curriculum on player count** — train 3-player first, expand to 4-5

## Files to Read First

- `scout-bot/main.py` — training loop, charts, checkpoint logic
- `scout-bot/training.py` — `ppo_update()` (lines ~293-420), `OpponentPool` (lines ~235-265)
- `scout-bot/third/charts.png` — latest training run results

## How to Verify

```
python main.py --save-dir scout_test --iterations 5 --games-per-iter 5
```
Should print new metrics (clip, kl, ev) in console and generate 4x3 charts.png.
