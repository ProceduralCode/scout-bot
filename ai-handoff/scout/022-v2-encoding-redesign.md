# Scout: V2 Encoding Redesign

## Task

Design and begin implementing a new v2 encoding that fixes the rotation bottleneck and reduces input size. Implementation is partially complete — encoding.py is done, remaining files need updating.

## Design Decisions

### Compact play encoding (replaces 220-dim rotated slot encoding with 67-dim fixed-position encoding)
- 4 × 11 one-hots: left end card (both faces), right end card (both faces)
- For 1-card plays: right card slots are empty (all zeros except value-0 marker)
- For no play: all card slots empty
- 3-dim play type one-hot: no_play / set / run
- 10-dim strength one-hot: values 1-10, game-defined (high card for runs, repeated value for sets), all zeros if no play
- 10-dim length one-hot: lengths 1-10, all zeros if no play
- No rotation — end cards at fixed positions. This is the core fix for the scout insertion learning failure.

### Hand size reduced from 20 to 15 slots
- Max starting hand is 12 (3 players). Scouting blocked at 15 cards.
- SCOUT_INSERT_SIZE = HAND_SLOTS = 15 (the 20/21 mismatch is gone)
- Hand rotation (ho) preserved — it's beneficial for play heads

### Single-round training (no multi-round games)
- Rewards were already per-round margin. Cumulative game scores were noise in the encoding.
- Starting player randomized instead of rotating across rounds.
- Eval games remain multi-round for comparison with v1 opponents.
- Removed from metadata: cumulative scores (5 dims), round_progress (1 dim)
- Collected counts normalized /15 instead of /20

### Total: INPUT_SIZE_V2 = 261 (down from 475, 45% reduction)
- Hand: 165 (15 × 11)
- Play: 67 (compact)
- Metadata: 29

### Output sizes
- play_start: 15 (was 20)
- play_end: 15 (was 20)
- scout_insert: 15 (was 21)
- action_type: 9 (unchanged)

### V1 preserved for eval comparison
- Old networks loaded as eval opponents use v1 encoding
- ScoutNetwork gets encoding_version attribute; dispatch per-network in turn logic
- Save dir: v4_1

## What Changed

- **`scout-bot/encoding.py`** — v2 constants, functions, and parameterized masks all added:
  - V2 constants block (HAND_SLOTS_V2, PLAY_SIZE_V2, INPUT_SIZE_V2, etc.)
  - `_fill_play_v2()` — compact play encoding
  - `_fill_metadata_v2()` — no cumulative scores, no round_progress, collected /15
  - `encode_state_v2()` — no play_offset parameter
  - `encode_hand_both_orientations_v2()` — for flip decisions
  - `_fill_hand()` — parameterized with `num_slots` (default HAND_SLOTS for v1 compat)
  - All mask functions parameterized with `num_slots`/`max_hand` defaults (v1 callers unchanged)
  - `decode_slot_to_hand_index()` — parameterized with `num_slots`

## Next Steps

### Remaining implementation (follow the plan in this session's conversation)

1. **`scout-bot/network.py`** — Parameterize ScoutNetwork:
   - Constructor takes `play_start_size`, `play_end_size`, `scout_insert_size`, `encoding_version`
   - `_build_conditioning` uses `self.play_start_size` instead of module constant
   - RandomBot gets `encoding_version` attribute and parameterized sizes
   - OpponentPool saves/loads `encoding_version`

2. **`scout-bot/training.py`** — V2 encoding dispatch + single-round:
   - `play_games_batched`: add `encoding_version` param, set local aliases for constants/functions, set `total_rounds=1` and random starting player for v2
   - `_play_turn`: dispatch encoding per network via `getattr(net, 'encoding_version', 1)`
   - `_process_turn_from_hidden`: same dispatch
   - `play_game`: single round + random starting player for v2
   - `_build_batch_conditioning`: add `play_start_size` param
   - `ppo_update`: thread `play_start_size` through

3. **`scout-bot/main.py`** — Wire v2:
   - PARAMS: `encoding_version: 2`, `save_dir: "v4_1"`, add `"v3_4"` to eval_opponents
   - Network construction branches on encoding_version (INPUT_SIZE_V2, v2 head sizes)
   - Eval opponent loading detects encoding_version from checkpoint
   - Checkpoint resume: encoding_version from checkpoint (can't change mid-training, like layer_sizes)

4. **Smoke test** — Run a few iterations, verify cross-version eval works

## Watch Out

- `play_games_batched` is ~250 lines of dense batched logic. Many constant references need updating (HAND_SLOTS, PLAY_SLOTS, SCOUT_INSERT_SIZE in ~12+ places). Use local aliases at the top of the function.
- V2 encode functions have different signatures (no play_offset). V2 callers must not pass po.
- `_build_batch_conditioning` in training.py uses `PLAY_START_SIZE` for `F.one_hot` — must be parameterized for v2.
- Probes (probe.py, probe_diagnostic.py) reference v1 constants directly. They'll still work for v1 testing but would need updates if you want to probe v2 encoding.
