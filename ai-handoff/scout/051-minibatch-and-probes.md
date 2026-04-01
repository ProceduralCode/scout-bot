# Mini-batching, Trunk Verification, and Probe Fixes

## Task & State

Investigated and fixed the 11GB memory spike during training (cause: full-batch autograd intermediates in PPO update). Verified LayerNorm trunk fix resolved rank collapse. Fixed probes 10-11 to use rotation augmentation. Probe 11 now passes; probe 10 still fails — investigation in progress when session ended.

Training run v6_7 is active at ~175 iterations. ppo_epochs was reduced from 4 to 2 by the user mid-run (around iter ~140) to address high clip fraction from mini-batching.

## What Changed

- `scout-bot/training.py` — Extracted `_ppo_step_v6()` helper from `ppo_update_v6()`. `ppo_update_v6()` now accepts `mini_batch_size` parameter. When set, shuffles indices with `torch.randperm`, splits into chunks, runs optimizer step per mini-batch, returns weighted-average metrics. Metrics use sum-then-divide pattern for correct weighted averaging (including explained variance from per-batch v_err/v_var).

- `scout-bot/main.py` — Added `"mini_batch_size": 4096` to PARAMS, passed through to `ppo_update_v6()`. Added `SCRIPT_DIR` pattern for cwd-independent execution (resolves `save_dir` and eval opponent paths relative to script directory). Removed all memory profiling instrumentation from session 050.

- `scout-bot/probe_v6.py` — `_train_iteration()` now accepts `augment=False` parameter. When True, overwrites rec.value with GAE return then calls `augment_rotation_v6()` before preparing batch (matching main.py's sequence). Probes 10 and 11 now pass `augment=True`.

## Measurements

### Memory profile (iteration 31, 95K records, 4 PPO epochs)

Full-batch (before):
- After prepare_batch: 972 MB
- After PPO epoch 0: 4,727 MB (+3,755 MB autograd)
- After epoch 3: 7,051 MB (RSS high-water mark)

Mini-batch (4096, after):
- After prepare_batch: 972 MB
- After PPO epoch 0: 959 MB
- After epoch 3: 961 MB

### Trunk analysis (v6_7 iter 102 vs v6_5 iter 224 baseline)

- PC1 variance: 99.96% → 18.65% (collapse resolved)
- Dead ReLU neurons: 0/128
- activation_mean ↔ PC1: r=-1.000 → r=-0.920
- Policy/value gradient ratio flipped: 0.5x → 14.6x (value grads now much smaller than policy)

### Clip fraction impact from mini-batching

Pre-minibatch (4 full-batch epochs = 4 gradient steps): clip ≈ 0.025, KL ≈ 0.003
Post-minibatch (4 epochs × ~23 mini-batches = ~92 steps): clip ≈ 0.31, KL ≈ 0.045
After user reduced to 2 epochs (~46 steps): clip ≈ 0.22-0.28, KL ≈ 0.035-0.045

### Probe results (with rotation augmentation)

- Probe 11 (scout adjacent matching): PASS — P(adj) 0.203 → 0.255
- Probe 10 (scout insertion quality): FAIL — chosen_q 2.19 → 2.19 (zero improvement, even with [512,256,128] network and 200 iters)

Probe 10 reward signal analysis: 37% of samples have zero gap (all positions equivalent), 55% have gap=1 (binary signal), 8% have gap>1. Mean max-min gap: 0.72. Max achievable improvement over random ≈ 0.4 in average chosen_q.

## Decisions

- **Mini-batch size 4096.** Reduces autograd peak from ~3.75GB to ~160MB. Chosen as a round power-of-2 that's small enough for memory savings, large enough for stable gradients.

- **psutil for memory profiling.** `ctypes.windll` approach returned 0 on MSYS Python. Installed psutil as a dependency.

## Next Steps

1. Probe 10 investigation is unfinished. The probe shows zero improvement even with augmentation and large networks. The user challenged the "weak signal" explanation — 63% of samples have nonzero reward, which should be learnable. Debugging was interrupted. Possible angles: check if PPO training dynamics are actually producing gradient on those samples, check if the value function is collapsing advantages to zero, or check if the reward function has a bug.

2. Clip fraction is still elevated (0.22-0.28) after reducing to 2 epochs. Increasing `mini_batch_size` from 4096 to 8192+ would reduce gradient steps per epoch and may help. Not urgent — eval margins are still improving.

3. Charts cleanup was discussed but not implemented. Six low-value panels identified (value prediction, policy loss, value loss, avg reward, reward std, entropy floor penalty). Could add gradient norm (pre-clip, returned by `clip_grad_norm_`). Not urgent.

## Watch Out

- `mini_batch_size` is not in the checkpoint-preserved params list (like `layer_sizes`, `encoding_version`, `attention`). It's overridden from PARAMS on resume, which is the intended behavior.
- `trunk_analysis.py` still segfaults without `-u` flag. Use `python -u scout-bot/trunk_analysis.py`.
- Checkpoint paths in trunk_analysis.py are relative to SCRIPT_DIR: pass `v6_7/latest.pt`, not `scout-bot/v6_7/latest.pt`.
