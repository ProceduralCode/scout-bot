# Q-Network Implementation Continued

## Task & State

Continued from session 073. Implemented phases 4-7 of the Q-network training system. All core functionality is in place and syntax-verified. Two small items remain unfinished (see Next Steps).

## What Changed

### training.py — Phases 4-5

- **`rollout_multi_action_v6`** (after `play_games_q_v6`): Selects top-K + random actions per QSample, batches through GPU rollout pipeline in chunks of 512 pairs × N rollouts. Handles games that end immediately after action by computing margin from final scores. Takes `rollout_temperature` param, passes to `rollout_numba`.
- **`prepare_q_batch_v6`**: Builds target/training_mask tensors from rollout margins, applies all 16 rotation augmentations via HAND_SHIFT/FULL_PERM. No forward pass needed (unlike PPO augmentation — just index permutations).
- **`q_update_v6`**: Shuffled mini-batch masked MSE. Loss only at rolled action positions. Returns mse_loss + mean pred/target margins.
- **`rollout_temperature`** added to `rollout_multi_action_v6`, `ReplayBuffer.revalidate`, `ReplayBuffer.check_and_prune` signatures.
- Fixed `state_dict` comment (was "excludes snapshots" but actually includes them).

### numba_engine.py — Rollout temperature

- `rollout_numba` gained `temperature` parameter (default 1.0). temperature=0 → greedy argmax, otherwise divides logits by temperature before `batched_masked_sample`.

### main.py — Phase 6-7

- **`Q_PARAMS`** dict after PARAMS with all agreed config values plus `rollout_temperature: 1.0`.
- **`train_q()`** function: play → rollout → replay buffer → augment → train → log → checkpoint → eval. Auto-resume restores replay buffer. Saves replay_buffer.state_dict() in checkpoints.
- **`_save_q_charts()`**: Currently 3×3 grid. Was being rewritten to 4×4 when interrupted (see Next Steps).
- **`_run_eval`**: Added `chart_fn` parameter so train_q uses `_save_q_charts`.
- **`--mode q|ppo`** argparse flag added. Default still "ppo" (user requested changing to "q" — not done yet).
- Logging computes: conditional entropies (softmax(margins) for play/scout regions), dormant neurons, histogram snapshots (margin predictions, rollout margins, replay buffer age distribution) stored as `_hist_*` keys in metrics_history.

## Decisions

- `action_taken` is always included in the rollout set.
- Augmentation does all 16 rotations (including identity k=0) uniformly, rather than PPO pattern of 1..15 + prepend originals.
- Histograms stored as `_hist_*` keys in metrics_history (latest snapshot, overwritten each iter, not time series). Chart function reads them if present.
- `chart_fn` parameter on `_run_eval` rather than duplicating the function.

## Next Steps

### Immediate: Two unfinished items

1. **Rewrite `_save_q_charts` to 4×4 grid** — current 3×3 version exists but user requested adding: scout play length, conditional entropies, dormant neurons, replay buffer age distribution (histogram as line), margin prediction distribution (histogram as line), rollout margin distribution (histogram as line), pred vs target margin. All histograms should be line charts (not bars) with x-axis = value (not iterations). The existing 3×3 code at ~line 750 needs to be replaced entirely.

2. **Change `--mode` default to "q"** — line 1793, change `default="ppo"` to `default="q"`.

### After that: First training run

Run `python main.py` (once default is q) to verify the full pipeline works end-to-end. Watch for:
- GPU memory issues during rollout batching (540K rollout games per iteration)
- `from_snapshots` performance with large batches
- Whether `replay_margin_max_diff=0.5` is a reasonable starting point

## Watch Out

- The `_hist_*` keys in metrics_history are not time series — they're single snapshots (lists of bin edges + counts). The chart function needs to handle them differently from regular metrics.
- The `import numpy as _np` inside the logging block is redundant (numpy is already imported as `np` at top of training.py, but main.py uses it through the training functions). Should use the top-level `numpy` if available or handle the import cleanly.
- `_save_q_charts` is called from `_run_eval` via `chart_fn` — during eval-triggered calls, histogram data may be stale (from the last logging iteration, not current).
- Old PPO functions in training.py are dead code — they'll crash if called with FlatScoutNetwork.
