# Scout: Charts, Eval, and Display

## Task

Three items from previous next steps: play length breakdown chart, eval random bot toggle, and display improvements. Optimization deferred to a future session.

## What Changed

### Modified Files

- **`scout-bot/main.py`**
  - **Play length distribution chart**: New multi-line chart showing fraction of plays by length (1, 2, 3, 4, 5, 6, 7+). Separate from avg_play_length which remains its own single-line chart. Metrics: `play_len_1_pct` through `play_len_6_pct` + `play_len_7plus_pct`. `avg_play_length` still computed and stored.
  - **Eval random bot as magic word**: `"random": "random"` in `eval_opponents` PARAMS is recognized as a sentinel — stores `RandomBot()` instead of loading a checkpoint. The separate random bot eval block was removed; all opponents go through a single unified eval loop. Old `eval_margin` data migrated to `eval_margin_random` on checkpoint load.
  - **Chart filtering**: `_save_charts` takes `eval_opponent_names` param — only plots `eval_margin_*` keys matching current config. Remove an opponent from PARAMS and its line disappears from the chart.
  - **Chart x-axis alignment fix**: All charts now right-align metrics to the iteration list (`iters[-len(data):]` instead of `iters[:len(data)]`). This fixes metrics added mid-training being plotted against wrong iteration numbers. Trimming also accounts for shorter metrics: `max(trim - start, 0)` where `start = len(all_iters) - len(vals)`.
  - **`scout_play_len` treated as eval metric** in trimming logic (it aligns to `eval_iteration`, not `iteration`, despite not having an `eval_` prefix).
  - **Chart rearrangement**: Row 0 = Score Margin, Steps/Game, Play Length Distribution, Avg Play Length. Row 1 = Scout Play Length, Action Type Distribution, Avg Reward, Reward Std. Row 2 = Policy Loss, Value Loss, Per-Head Entropy, Entropy Floor Penalty. Row 3 = Value Prediction, Clip Fraction, Approx KL, Explained Variance.
  - **Removed**: advantage_std chart and metric append (metric still accepted from old checkpoints, just not displayed or collected).

- **`scout-bot/display.py`**
  - `_val_to_char(10)` returns `"T"` instead of `"0"` (user changed to uppercase T).
  - `_char_to_val` accepts both `"0"` and `"T"` for backward compat.
  - `parse_card` validation updated to accept `"T"`.

- **`scout-bot/game_log.py`**
  - Turn-list hand columns and turn-0 starting hands use `format_showing_values` (top number only) instead of `format_hand` (both sides).
  - `_hand_col_width` formula changed by user to `(45 // num_players + 2) * 2` (was `* 3`).

## Next Steps

- **Further optimization (deferred)** — mask functions returning numpy instead of torch tensors (~0.65s), encoding fill functions in Cython (~0.6s), Play.from_cards in Cython (~0.3s). Each saves 0.1-0.3s. See handoff 016 for profiling details.

## Watch Out

- **`scout_play_len` naming**: It uses the eval x-axis but doesn't have an `eval_` prefix. There's a special case in the trim logic for it. If adding more eval-aligned metrics without `eval_` prefix, add them to the same check.
- **Checkpoint backward compat**: Old checkpoints with `eval_margin` get it migrated to `eval_margin_random`. Old `play_len_4plus_pct` data from very brief mid-session runs will be orphaned (harmless).
