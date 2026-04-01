# V4 Circular CNN Architecture

## Task & State

Designed and implemented the v4 encoding and network architecture. The core idea: replace flat pairwise diffs (v3's failed approach) with a circular CNN that learns hand structure patterns directly. Network and encoding are implemented and shape-tested. Not yet wired into the training loop.

## What Changed

- `scout-bot/encoding.py` — Added v4 constants (`HAND_SLOTS_V4`, `CNN_CHANNELS_V4`, `FLAT_SIZE_V4`, etc.), `_fill_hand_cnn_v4()`, `encode_state_v4()`, `encode_hand_both_orientations_v4()`. V4 encode returns `(cnn_tensor, flat_tensor)` — two tensors, not one.

- `scout-bot/network.py` — Added `CircularCNNScoutNetwork` class (circular CNN hand encoder → FC trunk → action heads). Extracted `build_conditioning()` to standalone function shared by both network classes. 612K total params with default settings (36K conv, rest FC).

- `ai-handoff/scout/context.md` — Added v4 encoding description, design notes on embeddings vs one-hots and circular hand buffer.

## Decisions

- **Circular CNN over pairwise diffs.** The hand is a circular buffer; a circular CNN's translational equivariance matches this symmetry. Kernel size 15 on 15 positions = circulant weight matrix (FC with relative-offset weight sharing). Three conv layers, all kernel 15 — no reason to constrain kernel size when the input is only 15 elements.

- **One-hot top faces for CNN, v3 scalars for flat path.** CNN processes 11-channel one-hots (top face only) for relational pattern detection. Flat path carries v3's scalar encoding (both card faces, play buffer, metadata) for card counting/deduction/scouting. The two paths serve complementary roles — CNN doesn't need bottom values for structural patterns, but the FC trunk can use them for strategic reasoning.

- **No embedding layer.** Embedding lookup is mathematically identical to one-hot × weight matrix. The first conv layer's weights learn the same transformation. One-hots are simpler with no loss of expressiveness.

- **Bottom card values included via flat scalars, not CNN.** Bottom values don't participate in runs/groups (orientation is fixed after flip). But they carry strategic info: card identity for deduction and what opponents can see/scout. Included in the flat path as v3 scalars.

## Next Steps

1. Wire v4 into the training loop. The forward interface change (`network(cnn, flat)` vs `network(state)`) means every call site in `training.py` that calls `network(state_tensor)` needs a v4 branch. Same for eval, flip decision, and checkpoint save/load in `main.py`.
2. Test with a real training run to verify end-to-end correctness.
3. F (num_filters) and num_conv_layers are tunable — defaults are 32 filters, 3 layers. May want to experiment.

## Watch Out

- `encode_state_v4` returns two tensors `(cnn, flat)`, not a single tensor like v2/v3. All callers must handle the tuple.
- `encode_hand_both_orientations_v4` returns `((cnn, flat), (cnn_flip, flat_flip))` — nested tuples of tensors. Different structure from v2/v3 which return `(tensor, tensor)`.
- The batched training path in `play_games_batched()` currently stacks single-tensor states. V4 needs to stack CNN and flat tensors separately before the forward pass.
