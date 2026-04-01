# Adaptive LR & Rollout Forward Pass Chunking

## Task & State

Implemented adaptive learning rate (KL-based) to fix the entropy collapse / KL blowup from handoff 068. Also diagnosed and fixed a severe performance issue with `rollout_fraction=1.0` — the forward pass in `rollout_numba` was OOMing on 388K-sample batches.

Training is running (v7_12, fresh start with `rollout_fraction=1.0`), but the deeper problem remains: training regresses after ~25-30 iterations across multiple runs. Eval margins improve early then degrade. Play lengths shrink back toward singles. This predates the adaptive LR work and is not a hyperparameter issue.

## What Changed

### Modified files
- `scout-bot/main.py`:
  - `PARAMS["learning_rate"]` is now `"adaptive"` (string). New params: `lr_initial` (3e-4), `lr_min` (1e-5), `lr_max` (3e-3). `rollout_fraction` set to 1.0.
  - Optimizer creation resolves `initial_lr` from `lr_initial` when adaptive. `current_lr` tracks mutable state.
  - Per-iteration: when adaptive, uses `current_lr` instead of linear decay.
  - After PPO: nudges LR by 0.9x if KL > target, 1.1x if KL < target (smooth adjustment, no dead zone). Initially used 0.5x/2.0x halve/double with 1.5x/0.5x thresholds — this caused wild oscillation and was replaced.
  - `current_lr` saved in all checkpoints via `extra={"current_lr": current_lr}`, restored on resume.
  - `lr` tracked in `metrics_history`, shown in per-iteration log line, plotted on chart (bottom-right panel, was previously blank).

- `scout-bot/numba_engine.py`:
  - `rollout_numba` forward pass now chunked at 4096 samples. CUDA kernels still run on full batch. Only `network()` + `policy_logits()` are chunked.

### New files
- `scout-bot/bench_batch_size.py` — benchmarks network forward pass throughput at different batch sizes. Finds optimal chunk size for the GPU.

## Key Findings

### Adaptive LR behavior (v7_10, 97 iterations)
- LR oscillated wildly with halve/double: 3e-4 → 1.2e-3 → back to 1e-4. `kl_batch_frac` bounced between 0.29 and 1.0.
- After switching to 0.9x/1.1x, behavior is smoother but training still regresses after initial improvement.

### Forward pass batch size benchmark (RTX 3060 Laptop, 6.4GB VRAM)
- Peak throughput: B=4096 at 551K samples/sec
- Throughput degrades above 128K, collapses at 256K+ (memory-bandwidth-limited)
- B=400K OOMs during attention softmax
- With `rollout_fraction=1.0`, rollout batch was ~388K samples — straight OOM territory. This is why rollout time was 577s instead of ~22s.

### Training regression pattern (persistent across runs)
- Eval margins improve for ~25-30 iterations then degrade
- Play lengths shrink back toward singles over time
- Steps per game increases (weaker play → longer games)
- This pattern appeared in v7_8, v7_9, v7_10 regardless of LR scheme

## Decisions

- Smooth 0.9x/1.1x adjustment chosen over halve/double to prevent oscillation. No dead zone — adjusts every iteration.
- Chunk size of 4096 based on empirical benchmark, not a guess. The benchmark script is kept for re-running on different GPUs.
- Removed the LR change log message (was printing every iteration since there's no dead zone).

## Next Steps

The adaptive LR and rollout chunking are mechanical fixes. The core unsolved problem is training regression — the network learns basics fast then actively gets worse. This is not a hyperparameter issue (it happens across LR schemes). Open question: is self-play converging to a degenerate equilibrium where conservative single-card play is stable against an opponent pool of similarly conservative players?

## Watch Out

- `learning_rate` PARAM is now a string `"adaptive"` not a float. Code that compares it numerically will break. The actual LR is in `current_lr` (mutable variable in the training loop) or `cfg["lr_initial"]`.
- Checkpoint `extra` dict now contains `current_lr`. Old checkpoints without it fall back to `lr_initial`.
- `bench_batch_size.py` requires CUDA. Uses `total_memory` property (not `total_mem`).
- The 16K batch size shows anomalously low throughput on this GPU — avoid that specific size.
