# V4 Probe Testing

## Task & State

Refactored v4 encoding and network to use a single packed tensor (instead of dual-tensor interface), then wired it into the probe system and ran all probes. V4 is end-to-end functional through the PPO pipeline. Not yet integrated into the main training loop.

## What Changed

- `scout-bot/encoding.py` — Added `CNN_FLAT_SIZE_V4` (165) and `INPUT_SIZE_V4` (279). `encode_state_v4()` now returns a single `(279,)` tensor: `[CNN_flat (165) | flat_scalars (114)]`. CNN portion written via reshaped view, then flattened by the contiguous buffer. `encode_hand_both_orientations_v4()` now returns `(tensor, tensor)` matching v2/v3 interface.

- `scout-bot/network.py` — `CircularCNNScoutNetwork.forward()` now takes a single tensor `(batch, 279)`, splits at `CNN_FLAT_SIZE_V4`, reshapes CNN portion to `(batch, 11, 15)` internally. Imported `CNN_FLAT_SIZE_V4` and `INPUT_SIZE_V4`.

- `scout-bot/probe.py` — Added v4 branches to `_make_network`, `_encode`, `_sample_play`, `_sample_scout`, `_sample_action_type`, `_train_iteration`. Added `--v4` flag. Fixed pre-existing bug: `_sample_scout`'s v1 fallback was calling `_encode()` (which uses the global `ENCODING_VERSION`) instead of encoding directly for v1. This caused crashes when running `--v2/v3/v4` with probe 8 (frozen trunk).

## Decisions

- **Single packed tensor over dual-tensor interface.** The state is one conceptual thing. Packing CNN `(11×15=165)` + flat `(114)` into `(279,)` means the network owns the split/reshape, and everything upstream (StepRecord, prepare_ppo_batch, ppo_update) works unchanged. One `view()` call per forward pass — zero compute cost.

## Next Steps

1. Wire v4 into the main training loop (`training.py`, `main.py`). The single-tensor interface means this is mostly adding encoding version branches alongside v2, not restructuring the pipeline.
2. Run a real training run to verify end-to-end training convergence.

## Watch Out

- Probe 8 (frozen trunk) tests a v1/v2 checkpoint, not v4 — its pass/fail is irrelevant to v4 validation.
- The probe network size `[64, 32]` is small. Probes 2b, 4, 5, 5b, 7 fail on both v4 and v2 with this size. These test harder generalization/multi-step coordination, not basic architecture viability.
