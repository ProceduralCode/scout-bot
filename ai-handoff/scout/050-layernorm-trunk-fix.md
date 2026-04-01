# LayerNorm Trunk Fix

## Task & State

Investigated root cause of trunk feature collapse (99.96% variance in PC1) from session 049. Identified the cause, added LayerNorm to the FC trunk. Also added `SCRIPT_DIR` pattern to `main.py` so it can run from any working directory. Memory investigation started but not completed — training uses ~11GB on first iteration.

## What Changed

- `scout-bot/network.py` — Added `nn.LayerNorm(size)` between `nn.Linear` and `nn.ReLU` in `FlatScoutNetwork`'s FC trunk construction loop. Trunk is now Linear → LayerNorm → ReLU at each layer. Only affects the non-residual path (residual blocks unchanged, and not used with current `layer_sizes=[512, 256, 128]`).

- `scout-bot/main.py` — Added `SCRIPT_DIR` pattern: resolves `save_dir` and eval opponent checkpoint paths relative to script directory. Also has `tracemalloc` memory profiling instrumentation in the training loop (fires on first iteration only) — **this should be removed** once memory investigation is complete.

- `scout-bot/trunk_analysis.py` — New diagnostic script. Plays games, collects per-decision interpretable features, runs PCA on trunk activations, correlates PC components with game features, and measures gradient magnitudes from each loss component separately. Uses `SCRIPT_DIR` pattern for cwd-independent execution.

- `.claude/hooks/block-cd.sh` — Updated error message to guide toward cwd-robust scripts.

- `scout-bot/test_import.py` — Test file created during debugging, can be deleted.

## Measurements (v6_5 checkpoint, iteration 224, 2705 decisions from 60 games)

### PC1 semantic identification

PC1 is game progress / tempo. Top correlations:
- `hand_size` r=0.815, `collected` r=-0.784, `scout_tokens` r=-0.712, `turn_number` r=-0.720
- `activation_mean` r=-1.000, `activation_norm` r=-1.000 — PC1 *is* activation magnitude
- All features combined: R²=0.895 with PC1
- Early game = large activations, late game = small activations

### Gradient magnitude comparison (policy vs value on trunk)

With loss coefficients applied (value_loss_coeff=0.25, entropy_bonus=0.25):
- Policy trunk grad norm: 3.40
- Value trunk grad norm: 6.52
- Entropy trunk grad norm: 1.87
- Policy/Value ratio: 0.5x — value gradients are *larger*, not smaller

This disproved the hypothesis that policy gradient dominance was causing the collapse.

### Input encoding PCA

Input (309D) has PC1 at 14.6% variance, dominated by hand region (54.5% of loading) and metadata (25.4%). Moderate concentration but not pathological.

## Decisions

- **LayerNorm, not separate trunks or encoding changes.** The collapse is caused by unnormalized Linear+ReLU stacking compounding the dominant input direction (game phase). Gradient analysis showed no head-to-head conflict. The attention block's pre-norm already prevents collapse there; the FC trunk was the only unnormalized part of the network.

- **No config flag for LayerNorm.** It's always-on in the FC trunk. No reason to support the unnormalized variant going forward.

## Next Steps

1. **Memory investigation.** Training uses ~11GB on first iteration (system has 40GB, ~20GB available). `tracemalloc` instrumentation is in `main.py` (first iteration only) but output wasn't captured. Run training and check the `[MEM]` lines to identify where the 11GB goes. Suspected contributors: 72K augmented StepRecordV6 objects (individual tensors + Python overhead), full-batch PPO forward/backward autograd intermediates, replay buffer copies. Mini-batching PPO (`# TODO` in PARAMS) would cut the autograd peak.
2. Start a fresh v6_6 training run with the LayerNorm trunk. Old checkpoints are incompatible (different state dict keys). Re-run `trunk_analysis.py` after ~100 iterations to verify rank collapse is resolved.
3. Probes 10-11 fix and PPO direction probe still pending from session 047.

## Watch Out

- `trunk_analysis.py` segfaults when run as `python scout-bot/trunk_analysis.py` (without `-u`). Works fine with `python -u scout-bot/trunk_analysis.py`. Cause unclear — possibly matplotlib/Cython DLL interaction on Windows. The `-u` flag (unbuffered) is a workaround.
- `main.py` has temporary `tracemalloc` instrumentation (search for `_mem_iter1`). Remove after memory investigation is done.
