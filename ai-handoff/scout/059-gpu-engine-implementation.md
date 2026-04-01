# GPU Engine Implementation (Steps 1–3)

## Task & State

Building the GPU vectorized rollout engine (`scout-bot/gpu_engine.py`) as a drop-in replacement for `rollout_from_states_batched_v6`. Incremental build with tests at each step.

Completed: state representation, encoding, legal play computation.
Not yet done: action masks, apply_actions, rollout_gpu wiring, integration into training.py.

## What Changed

### New files
- `scout-bot/gpu_engine.py` — main engine file. Steps 1–3 complete.
- `scout-bot/test_gpu_state.py` — tests `from_snapshots` conversion
- `scout-bot/test_gpu_encode.py` — tests `encode_states` against `encode_state_v6`
- `scout-bot/test_gpu_legal_plays.py` — tests `compute_legal_plays` against `get_legal_plays`

All three test files pass fully.

## Decisions

- **Fixed MAX_STEPS=100, no sync check** — pure fixed-step execution, one transfer at the end. No `done.all()` check per step.
- **S&S as two separate steps** — SNS_PLAY phase tracked in `state.phase` tensor; the forced play resolves naturally in the next step iteration, same as the Python engine.
- **Left-aligned hand storage** — hands stored at positions 0..hand_len-1, not circular. The circular rotation for encoding augmentation is applied at encode time via `hand_offsets`.
- **Generic player count** — tensors padded to MAX_P=5, num_players tracked per game. Works for 3/4/5 players.
- **`batched_masked_sample` reused as-is** — already GPU-native (Gumbel-max via `torch.rand_like`). No replacement needed.
- **Model must be on CUDA** — the rollout currently runs the model on CPU. `rollout_gpu` will need `.to('cuda')` on the network before inference.

## Next Steps

**Step 4: `compute_action_masks`** — vectorize `get_flat_action_mask`. Takes `state` + `legal_plays` tensor + `hand_offsets`. Returns `[B, 384]` bool.

Structure:
- Play region [0..255]: remap legal_plays from position space to slot space using hand_offsets, then reshape to [B, 256]
- Scout region [256..319]: enabled iff has_play and hand_len < H; all insert positions 0..hand_len valid for each available card choice
- S&S region [320..383]: per-position legality — after inserting the scouted card at each position, must have a legal play against the reduced play. This requires building hypothetical hands and calling something like `compute_legal_plays` on them. Most complex piece.
- SNS_PLAY phase: only play region active (forced play)

Test: compare against `get_flat_action_mask` per game.

**Step 5: `apply_actions`** — update GpuGameState in-place from sampled actions.

**Step 6: `rollout_gpu`** — wire loop + integration into training.py.

## Watch Out

- `compute_legal_plays` returns positions in **hand-position space** (0..hand_len-1). The action mask uses **slot space** (0..H-1 based on hand_offset). The play region of the mask requires remapping via `hand_offsets`.
- For S&S mask: need to construct all hypothetical post-insert hands (4 card choices × up to 17 insert positions per game) and check legal play existence against the *reduced* play (current play minus the scouted card). The reduced play's type/strength/len changes per card_choice, not per insert_pos.
- The `scouts_since_play > 0` test was skipped (games end round before reaching that state in the test generator) — that path is a simple scalar in the metadata and is covered by the encoding match in other cases.
