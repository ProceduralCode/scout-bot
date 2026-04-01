# Scout Bot: Metrics Fixes, Probe Tooling, and Eval Improvements

## Task
Fix broken metrics under uniform reward distribution, add probe tooling improvements, add scout placement quality eval metric, fix misleading entropy chart, and reorganize charts.

## What Changed

### Modified Files

- **`scout-bot/main.py`**
  - **Reward metric fix** (lines 329-338): replaced `p0_rewards = [r.reward for r in p0_records if r.reward != 0.0]` with per-round grouping by `(game_id, round_num)` and summing. The old filter broke under `reward_distribution: "uniform"` because every step has nonzero reward — `avg_reward` was showing per-step averages instead of per-round totals. Now comparable across terminal vs uniform runs.
  - **Import** (line 13): added `from probe import eval_scout_quality`
  - **Scout play length eval** (lines 409-412): calls `eval_scout_quality()` at every `eval_interval`, tracks `scout_play_len` in metrics_history. Measures avg longest set/run containing the scouted card after insertion (random baseline ~1.5).
  - **Charts reorganized** (lines 92-222): changed from 6×3 to 4×4 grid. Layout:
    - Row 0: Avg Reward, Value Prediction, Score Margin, Scout Play Length
    - Row 1: Policy Loss, Value Loss, Per-Head Entropy, Entropy Floor Penalty
    - Row 2: Clip Fraction, Approx KL, Explained Variance, Advantage Std
    - Row 3: Action Type Dist, Steps Per Game, Avg Play Length, Reward Std
  - Added `_style_eval_ax()` helper for charts using eval x-axis.
  - Chart description for reward updated to "per-round" (line 144).
  - Chart description for entropy updated to note "steps with 2+ options only" (line 189).
  - **Config** (lines 35, 45-50): `reward_distribution` set to `"uniform"`, `save_dir` set to `"v2_6"`, eval_opponents trimmed to v1_4, v2_2, v2_5.
  - `metrics_history` (line 229): added `"scout_play_len"` key.

- **`scout-bot/training.py`**
  - **Filtered entropy metrics** (lines 379-385): added `_filtered_ent_mean()` helper inside `ppo_update`. All four per-head entropy metrics (`at_entropy_mean`, `ps_entropy_mean`, `pe_entropy_mean`, `si_entropy_mean`) now filter to steps with `mask.sum(dim=-1) >= 2` before averaging — same filter the entropy floor penalty uses. Previously `pe_entropy_mean` was ~0.03 (91% forced-choice steps dragging average to zero), now shows true uncertainty on the ~9% of steps with a choice. Chart values are directly comparable to floor thresholds in config.

- **`scout-bot/probe.py`**
  - **`eval_scout_quality()`** (lines 179-204): standalone function for measuring scout placement quality. Generates mid-round states, samples scout insert decisions, builds the new hand, finds longest legal play containing the inserted card. Returns `(avg_length, n_samples)`. Importable by main.py.
  - **`--layers` CLI arg** (line 878): overrides global `LAYER_SIZES` for probe runs. E.g., `--layers 512 256 256 128 128 128`.
  - **`--entropy-floors` CLI flag** (line 880): enables entropy floors using same values as main.py config.
  - **`_train_iteration()`** (lines 148-177): added optional `entropy_floors` and `entropy_floor_coeff` params, defaults to global `ENTROPY_FLOORS`/`ENTROPY_FLOOR_COEFF` so all probes get floors when `--entropy-floors` is passed.
  - **Globals** (lines 35-36): added `ENTROPY_FLOORS = None` and `ENTROPY_FLOOR_COEFF = 1.0`.

## Decisions

- **Per-round reward grouping over filter fix**: instead of tagging terminal steps with a bool, we sum all rewards within a `(game_id, round_num)` group. Simpler, and the total is invariant to distribution mode (terminal puts it all on one step, uniform spreads it, sum is the same).
- **Filtered entropy for chart metric**: switched all per-head entropy metrics to only average over steps with 2+ legal options. The unfiltered version conflated "how often does this head fire" with "how uncertain is it when it does." Only the latter matters for diagnosing entropy collapse.
- **Scout play length over adj_match**: replaced `P(adj_match)` with avg longest play containing the scouted card. More informative — captures degree of quality (1=useless, 2=pairs, 3+=real runs/sets) instead of binary match.

## Key Findings

### Probe 5b with full network
- **[512,256,256,128,128,128] without floors**: PASS, P(adj_match) 0.230→0.295. But entropy collapsed to 0.033.
- **[512,256,256,128,128,128] with floors**: FAIL, P(adj_match) 0.248→0.227. Entropy held at 0.621 but prevented learning.
- **Interpretation**: capacity IS the bottleneck (full network can learn card-value reasoning where [64,32] can't). Entropy floors prevented collapse but also prevented convergence in 150 iters on this isolated task. This is expected — the probe gives 100% scout signal per iteration, unlike real training where scout actions are ~20-30% of steps. The floor exists for real training's sparse-signal regime.

### v2_5 inspection (572 iters, terminal reward)
- Beating all eval opponents by iter ~300
- Explained variance at 0.43
- Action type: 69% play, 22% scout, 9% S&S
- Entropy floor penalty ~0 (floors never triggered with old unfiltered metrics)
- `reward_distribution` was not set (default terminal)

## Next Steps

1. **Run v2_6 training**: `python scout-bot/main.py` — fresh run with uniform reward distribution, filtered entropy metrics, scout play length eval. Config already set (save_dir=v2_6).
2. **Monitor scout_play_len**: watch whether it rises above the ~1.5 random baseline over training. This is the direct test of whether the scout_insert head learns card-value reasoning in real training.
3. **Monitor filtered entropy values**: now directly comparable to floor thresholds. If play_end filtered entropy drops below 0.3, the floor should activate — verify this actually happens.
4. **Compare v2_6 vs v2_5**: v2_6 uses uniform reward distribution; v2_5 used terminal. Reward and eval_margin metrics are now directly comparable due to the per-round grouping fix.

## Watch Out

- **v2_5 may still be running** in a separate terminal. It will compete for CPU with v2_6 (PyTorch uses all cores by default). User was aware; can set `OMP_NUM_THREADS=2` on v2_5 or kill it.
- **Old entropy metrics in saved checkpoints**: v2_5's saved `entropy_play_end` values are the old unfiltered averages (~0.03). When v2_6 resumes from scratch, its entropy values will be the filtered version and not directly comparable to v2_5's saved history. The inspect_run.py script will show the old values for v2_5.
- **`reward_distribution` not in v2_5 config**: v2_5's checkpoint doesn't have this key, so it defaulted to terminal. v2_6 has it set to "uniform" in DEFAULT_CONFIG.

## Files to Read First
- `scout-bot/main.py:329-338` — per-round reward metric
- `scout-bot/main.py:409-412` — scout play length eval
- `scout-bot/main.py:92-222` — 4×4 chart layout
- `scout-bot/training.py:379-385` — filtered entropy metrics
- `scout-bot/probe.py:179-204` — eval_scout_quality function
- `scout-bot/probe.py:875-893` — CLI args (--layers, --entropy-floors)
