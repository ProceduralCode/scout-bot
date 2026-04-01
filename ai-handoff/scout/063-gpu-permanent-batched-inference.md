# GPU Permanent Placement & Batched GAE Inference

## Task & State

Moved the training network to GPU permanently (was CPU with temporary GPU transfers for rollouts). Also rewrote `play_games_v6` with batched inference across all active games. Changes compile, run, and train — but two fixes are still needed before this is clean.

## What Changed

### Modified files
- `scout-bot/main.py`:
  - `network.cuda()` + optimizer state migration added after setup, before training loop
  - `map_location='cpu'` added to all `torch.load` calls (portability for CUDA-saved checkpoints)
  - Device-aware `.to(dev)` for dormant neuron tracking and diagnostic forward passes
- `scout-bot/training.py`:
  - `play_games_v6()` **rewritten** — now batches inference across all active games per turn step. ~30 batched forward passes of size ~75 instead of ~2,250 batch-1 passes. Flip phase also batched. Opponent path stays unbatched.
  - `play_games_with_rollouts_v6()` — removed `.cuda()/.cpu()` dance, added `.to(dev)` for per-turn inference
  - `_ppo_step_v6()` — moves batch tensors to device at top, `floor_penalty` created on correct device
  - `augment_rotation_v6()` — `.to(dev)` at network forward boundary, bulk `.tolist()` before record loop
  - `_play_round()` — device-aware flip inference (`.to(_dev)` per network)
  - `_play_turn_v6()` — device-aware state/mask transfer, `.cpu().numpy()` for mask storage
  - `rollout_from_states_batched_v6()` — `.to(dev)` for batched states/masks
  - `OpponentPool.add()` — `.cpu()` after deepcopy to keep pool on CPU

## Profiling Results (1 iteration, including eval)

| Component | Before (CPU) | After (GPU) | Notes |
|---|---|---|---|
| `play_games_v6` | 10.7s | **3.1s** | Batched inference working |
| `ppo_update_v6` | 12.1s | **1.2s** | Biggest win — backward on GPU |
| Rollouts | 30.0s | 27.6s | Small gain from removing .cuda()/.cpu() |
| Augmentation | 4.8s | **12.0s** | Regressed — `.item()` GPU syncs |
| Eval | (not in prev profile) | 37.0s | Batch-1 GPU inference + transfer overhead |

Training-only time (excluding eval/startup): ~44.6s vs 58.1s before.

**After the `.tolist()` fix applied to augmentation and play_games_v6** (code already edited, not yet profiled): augmentation should drop from 12s to ~5-6s, play_games_v6 `.item()` time should largely disappear. Expected training-only: ~35-38s.

## Next Steps

1. **Fix eval path.** Two issues:
   - **Performance**: `_play_turn_v6` does batch-1 GPU inference with per-call `.to(dev)` overhead during eval (4.1s on Tensor.to, 3.7s on Tensor.item in just the v6 eval path). Fix: move network to CPU before `_run_eval`, back to CUDA after. This is the simplest fix — eval runs infrequently.
   - **Error**: `WARNING: eval failed at iter 5: indices should be either on cpu or on the same device as the indexed tensor (cpu)` — likely in `eval_scout_quality` or the diagnostic code. A tensor operation is mixing GPU results with CPU indices. Find and fix the device mismatch.

2. **Re-profile** after the eval fix and `.tolist()` changes to get clean numbers.

3. **Verify training correctness.** The batched `play_games_v6` should produce identical training dynamics to the sequential version. Compare a few iterations' metrics (reward, value loss, entropy) against the previous run to sanity check.

## Decisions

- **Opponent pool stays on CPU.** Pool models do batch-1 inference as opponents — GPU would add overhead, not help. `OpponentPool.add()` now calls `.cpu()` after deepcopy.
- **Eval opponents stay on CPU.** Same reasoning. `map_location='cpu'` on checkpoint loads ensures they land on CPU.
- **Records store CPU tensors.** `state` and `mask` in StepRecordV6 stay CPU. GPU transfer happens at batch boundaries (prepare_ppo_batch, augmentation).
- **Permutation tables stay on CPU** in augmentation. The rotation augmentation does `orig_states[:, shift]` etc. on CPU, then `.to(dev)` for the forward pass. Moving tables to GPU would add complexity for marginal gain.

## Watch Out

- **`.item()` on GPU tensors is expensive.** Each call forces a GPU sync. Always bulk-transfer with `.tolist()` or `.cpu()` before Python loops. This bit augmentation hard (6.6s → fixed) and was noticeable in play_games_v6 (1.06s → fixed).
- **`play_games_with_rollouts_v6` still does batch-1 GPU inference** for its source game play (25 games × ~30 turns). The `.to(dev)` per call adds ~2.3s of transfer overhead. Batching this function too would help, but it's more complex due to the per-step snapshot logic.
- **The batched `play_games_v6` only creates records for training seats** (p < training_seats). Non-training seats using the training network get batched inference for actions but no records. `_assign_round_rewards` works correctly with training-seat-only records.
