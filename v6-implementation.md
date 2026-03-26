# v6 Implementation Plan

Implementation of plan.md. Three passes, each producing runnable code.

Decisions made:
- Rollout mode only (no GAE support for v6)
- No direct_pg_update for v6
- N=10 default (skip small deck idea for now)
- New v6 functions in training.py rather than branching existing ones
- Augmentation deferred to pass 3

## Pass 1 — Core Infrastructure

### game.py

- `create_deck(num_players, num_values=10)` — generate pairs for 1 ≤ i < j ≤ N
  - For fewer players: remove cards containing N (same pattern as current 3p removes 10s)
  - 4p with N=10: remove (9,10) card as before
- `Game.__init__(num_players, num_values=10)` — store `self.num_values`
- `Game.start_round()` — pass `num_values` to `create_deck`
- `get_state_for_player` — add `"num_values": self.num_values`
- No changes to turn logic, scoring, Play class

### encoding.py

Constants (parameterized by N=10, H=16):
```
HAND_SLOTS_V6 = 16
NUM_VALUES_V6 = 10
HAND_DIM_V6 = H * (N+2) + H  # 16*12 + 16 = 208
SCOUT_CARDS_DIM_V6 = 4 * (N+1)  # 44
PLAY_BUFFER_DIM_V6 = 16 + 5  # 21
METADATA_DIM_V6 = 28
INPUT_SIZE_V6 = HAND_DIM_V6 + SCOUT_CARDS_DIM_V6 + PLAY_BUFFER_DIM_V6 + METADATA_DIM_V6  # 301
FLAT_ACTION_SIZE = 384  # 256 play + 64 scout + 64 S&S
```

New functions:
- `encode_state_v6(game, player, hand_offset, forced_play=False)` → tensor [301]
  - Hand: H slots (circular with hand_offset), each N one-hot + empty indicator + top scalar. Separate block of H bottom scalars.
  - Scout cards: 4 options (left normal/flipped, right normal/flipped), each N one-hot showing face + absent flag. NOT offset-rotated.
  - Play buffers: left-aligned and right-aligned 4-card buffers, top+bottom scalars (/N). Plus play metadata (type 3-hot, strength/N, length/10).
  - Metadata: ~28 dims (same as v2 metadata + forced_play flag, player_count as scalar not one-hot)
- `_compute_scout_cards(game)` → list of 4 (value, present) tuples
  - Derives from current play's end cards and their flipped faces
- `get_flat_action_mask(game, player, legal_plays, hand_offset)` → bool tensor [384]
  - Play region [0..255]: True for each (start, end) mapped to offset-rotated slots, index = start_slot*16 + end_slot
  - Scout region [256..319]: True for each available scout card × valid insert positions
  - S&S region [320..383]: True for each S&S variant × positions where inserting yields legal play
  - Forced-play mode: only play region active
- `decode_flat_action(action_idx, hand_offset)` → dict with action type + params
  - [0..255]: play, extract start_slot/end_slot, convert to hand indices
  - [256..319]: scout, extract card choice (0-3) + insert_pos
  - [320..383]: S&S, extract card choice + insert_pos
- `encode_hand_both_orientations_v6(game, player, hand_offset)` → (tensor, tensor_flip)
  - For flip decisions. Same interface as v2/v3.

### network.py

New class `FlatScoutNetwork`:
- Same trunk: configurable layer_sizes with ResidualBlocks (default [512, 256, 256, 128, 128, 128])
- `forward(x)` → hidden state
- `value(hidden)` → scalar
- `policy_logits(hidden)` → [384] raw logits (single nn.Linear)
- Store `encoding_version = 6` as attribute
- No conditioning, no sequential heads

Update `RandomBot`:
- Add `policy_logits(hidden)` → zeros [384]
- Keep old methods for eval compat

Existing `batched_masked_sample` and `masked_sample` work unchanged on [B, 384] logits.

## Pass 2 — Training Pipeline

### training.py — New data structures

`StepRecordV6`:
```python
@dataclass
class StepRecordV6:
    state: torch.Tensor        # [301]
    action: int                # flat index [0..383]
    mask: np.ndarray           # bool [384]
    old_log_prob: float        # log prob at collection time
    value: float
    reward: float
    player: int
    round_num: int
    game_id: int
    hand_offset: int           # needed for augmentation
    # Diagnostics only
    play_length: int | None
    scout_quality: int | None
```

### training.py — Game play

`_play_turn_v6(game, networks, game_log=None)` → list[StepRecordV6]:
- Single forward pass → flat logits → masked sample → decode → apply
- S&S triggers second forward pass with forced_play=True
- Returns 1 record for play/scout, 2 records for S&S (each independent)

`play_games_with_rollouts_v6(network, num_games, num_players, rollouts_per_state, training_seats)`:
- Same snapshot-based flow as current
- S&S produces 2 snapshots: before S&S decision, after insert (before forced play)
- Each S&S record gets its own (before, after) snapshot pair
- Uses `rollout_from_states_batched_v6` for batched rollouts

`rollout_from_states_batched_v6(snapshots, network)`:
- Same structure as current but uses flat action sampling
- Single batched forward pass per step, one masked sample → decode → apply
- S&S in rollouts follows same two-step flow

### training.py — PPO

`prepare_ppo_batch_v6(steps, advantages, returns)` → dict:
- Simple: states, masks, actions, old_log_probs, advantages, v_targets
- No sub-head index tensors

`ppo_update_v6(network, optimizer, batch, ...)`:
- Forward pass → hidden → policy_logits [n, 384] + value [n]
- Log ratio from flat logits directly (no per-head accumulation)
- Single entropy computation over masked flat distribution
- Standard PPO clipped surrogate
- Diagnostics: conditional entropies (play-only, scout-only) computed from the flat distribution

`concatenate_batches_v6(batches)`:
- Simpler: just cat all tensors, re-normalize advantages
- No sub-head index offsetting needed

### training.py — Eval dispatch

`play_eval_game` needs to handle mixed network types:
- v6 networks use flat action interface
- Old networks use multi-head interface
- Dispatch based on `isinstance(net, FlatScoutNetwork)` or `encoding_version`

`_play_turn` already handles this per-network, so `_play_turn_v6` is only called for v6 networks. The eval loop calls the appropriate turn function based on network type.

## Pass 3 — Augmentation + Integration

### encoding.py — Permutation tables

Precompute for each offset k (0-15):
- `PLAY_PERM[k]`: 256-entry mapping for play region. Original (s,e) → ((s+k)%16, (e+k)%16) → new index
- `SCOUT_PERM[k]`: 64-entry mapping for scout region. Position p → (p+k)%16
- `FULL_PERM[k]`: combined 384-dim permutation
- `HAND_SHIFT[k]`: index mapping for shifting hand slots in state tensor

Store as tensors for GPU-side permutation.

### training.py — Augmentation

`augment_rotation(steps, network)` → list[StepRecordV6]:
- For each original sample, create 15 augmented copies
- Shift state tensor's hand portion by k
- Permute action index and mask using precomputed tables
- Batch forward pass on all augmented states to compute correct old_log_probs
- Same advantages and value targets

Called after collecting rollout data, before PPO update.

### main.py

PARAMS updates:
```python
"encoding_version": 6,
"num_values": 10,
"hand_slots": 16,
"use_rollouts": True,
"augment_rotations": 16,
"layer_sizes": [512, 256, 256, 128, 128, 128],
"entropy_bonus": 0.01,  # single global, no per-head floors
```

Training loop:
- Instantiate FlatScoutNetwork when encoding_version == 6
- After collecting rollout data, apply augmentation
- PPO update with ppo_update_v6

Charts:
- Replace per-head entropy panel with single flat entropy + conditional entropies (play-only, scout-only)
- Keep all other panels
- Remove entropy_floor_penalty panel (no floors in v6)

Eval opponent loading:
- Old checkpoints still load as ScoutNetwork
- play_eval_game dispatches based on network type

## Execution Order

Each pass builds on the previous. Within each pass:

Pass 1: game.py → encoding.py → network.py (strict dependency order)
Pass 2: StepRecordV6 + _play_turn_v6 → rollout functions → PPO functions (dependency order)
Pass 3: permutation tables → augmentation function → main.py integration

After each pass, existing code continues to work unchanged (all additions, no modifications to old paths except game.py's backward-compatible parameter additions).
