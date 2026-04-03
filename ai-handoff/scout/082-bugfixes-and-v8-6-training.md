# Bugfixes, Eval Improvements, and v8_6 Training

## Task & State

Fixed three measurement bugs found by audit, improved eval configuration, and launched v8_6 training run. Training is at ~1170 iterations and showing signs of plateauing across all eval opponents.

## What Changed

### training.py — Eval games now use greedy action selection for Q-networks

`_play_turn_v6()` line ~1641 was using `masked_sample` (Gumbel-max, effectively softmax temp=1) for all networks. Added check: `is_q_network = not hasattr(net, 'value')` — Q-networks get argmax, PPO networks keep masked_sample. This was the same bug class as the `eval_scout_quality` fix from session 081.

### main.py — Three fixes

- **Play length circular wrapping**: `_compute_diagnostics()` had `(a % 16) - (a // 16) + 1` (wrong for wrapping plays) → `(a % 16 - a // 16) % 16 + 1`. Two occurrences fixed. Only affects PPO diagnostic charts.
- **Action type chart label**: "Action Type Distribution" → "Action Type Distribution (post-curation)" — clarifies the ratios reflect curated samples, not raw gameplay.
- **`eval_games` config**: Added to both PARAMS and Q_PARAMS (default 40). `_run_eval` now uses `cfg.get("eval_games", 40)` instead of hardcoded `n_eval = 40`. v8_6 is running with `eval_games: 100`.

### main.py — Chart legend positions

Play length distribution and conditional entropies → `center left`. Dormant neurons → `upper left`. Prevents legend from obscuring data on the right side of charts.

### probe.py — eval_scout_quality uses argmax (from session 081)

Already done last session but noting for completeness: `_sample_scout` v6 path uses greedy argmax instead of `masked_sample`. eval_scout_quality bumped from 200 → 2000 samples (~2.9s).

## Decisions

- **Greedy eval for Q-networks**: Q-network outputs are margin predictions, not policy logits. Softmax over raw margins at temp=1 barely differentiates between actions — a 0.4 vs 0.2 gap gives only 1.5x preference across ~40 options. Argmax matches what `_select_action_q_batched` does at temperature=0.

## Next Steps

- **Plateau investigation**: v8_6 eval margins are flattening (v4_2: +8.63 → +9.32 over last ~250 iterations). Possible causes: self-play ceiling, exploration exhaustion (epsilon=0.1, temperature=0.1), signal saturation (rollout_margin_spread flat at ~0.51), or capacity limits.
- **Game visualization**: Q-network path doesn't save game logs. `--match` mode only shows aggregate scores. Need a way to watch the bot play (e.g. `--show` flag that plays one game with GameLog and prints it).
- **Cleanup**: `diag_scout_quality.py` and benchmark scripts from session 081 still exist.

## Watch Out

- **Q_PARAMS still has `probe_reward: "scout_quality"` commented out** (line ~259) and `eval_interval: 30` (changed from 5 for longer runs). These were intentional changes for v8_6.
- **Oracle scout ceiling is 2.83**, current bot achieves ~2.09. Meaningful room for improvement.
- **Session time analysis**: Parsed Claude Code session JSONL files in `~/.claude/projects/`. ~92 active hours on scout-bot across 110 sessions, ~115 hours total in this workspace. Used inter-message gap analysis with 5-10 minute idle threshold.
