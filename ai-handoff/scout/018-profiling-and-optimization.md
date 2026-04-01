# Scout: Profiling and Optimization

## Task

Full profiling of the training loop and optimization of identified bottlenecks. Goal was to speed up `python main.py` iteration time without Cython changes.

## What Changed

### Modified Files

- **`scout-bot/main.py`**
  - Added `--profile N` flag: wraps training loop with pyinstrument for N iterations, saves profile.txt and profile.html to save_dir, then exits. Also dumps profile on Ctrl+C.
  - `train()` gains `profile_iters` parameter.
  - Moved `_save_charts` call out of per-iteration logging block — charts now only generate on eval iterations (every `eval_interval`) and at final save. Was generating every iteration at ~3s each.
  - Import `prepare_ppo_batch` from training.

- **`scout-bot/training.py`**
  - **`prepare_ppo_batch()`** (new): Pre-stacks all StepRecord tensors (states, masks, logits, actions, advantages, returns) into a single batch dict. Called once before the PPO epoch loop. Previously `ppo_update` re-stacked ~16k tensors every epoch (4x).
  - **`ppo_update()`** signature changed: takes `batch: dict` instead of `steps: list[StepRecord]` + `advantages` + `returns`. All tensor access via batch dict keys.
  - **Numpy masks throughout**: All mask functions now return `np.ndarray` instead of `torch.Tensor`. In `play_games_batched`, numpy mask lists are kept alongside the torch batch (`at_masks_np`, `ps_masks_np`, `pe_masks_np`, `si_masks_np`) and used directly for StepRecord storage — avoids torch→numpy round-trip. Single-game paths (`_play_turn`, `_process_turn_from_hidden`) wrap with `torch.from_numpy()` at `masked_sample` call sites.
  - Added `import numpy as np`.

- **`scout-bot/encoding.py`**
  - All 5 mask functions (`get_action_type_mask`, `get_play_start_mask`, `get_play_end_mask`, `get_scout_insert_mask`, `get_sns_insert_mask`) return `np.ndarray` with `np.bool_` dtype instead of `torch.Tensor`.

## Results

Non-eval iteration time: ~5.7s → ~3.9s (32% faster).

| Change | Mechanism | Savings |
|---|---|---|
| Chart throttling | Only on eval iters | ~3s on 9/10 iters |
| Pre-stacked PPO batch | Stack once, reuse 4x | ~0.6s/iter |
| Numpy masks | np.zeros vs torch.zeros | ~1.9s/iter |
| Round-trip elimination | Keep numpy lists for StepRecord | ~0.25s/iter |

## Next Steps

Remaining bottlenecks in `play_games_batched` (~2.6s/iter):
- **Loop overhead** [self] (~1.2s) — Python interpreter cost, needs structural refactor
- **`encode_state`** (~1.1s) — `_fill_metadata` (0.5s), `_fill_hand` (0.25s), `_fill_play` (0.19s). Cython candidate.
- **`Play.from_cards`** (~0.6s) — called from `apply_play` and `apply_scout` in game.py. Cython candidate.
- **Network forward** (~0.45s) — irreducible PyTorch compute

## Watch Out

- **StepRecord mask fields are now numpy arrays**, not torch tensors. Any new code accessing masks from StepRecords must handle numpy. `prepare_ppo_batch` handles the numpy→torch conversion via `torch.from_numpy(np.stack(...))`.
- **`ppo_update` interface changed**: takes a pre-built `batch` dict, not raw StepRecords. Callers must use `prepare_ppo_batch` first. The old `steps`/`advantages`/`returns` parameters are gone.
- **`--profile` saves to save_dir**: profile.txt and profile.html go into v3_4/ (or whatever save_dir is configured).
