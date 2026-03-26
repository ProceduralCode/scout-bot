# v6 Plan — Flat Head + New Encoding + Rotation Augmentation

## Motivation

The multi-head action decomposition (action_type → play_start → play_end / scout_insert) has a structural problem: one advantage value trains multiple heads. The scout_insert head can't distinguish "scouting was good timing" from "this insert position was good" because both receive the same advantage. After 5 versions and thousands of iterations, scout_play_len remains ~1.5 (random).

The play heads have a milder version: the start/end decomposition adds credit assignment noise, and the sequential conditioning makes it harder to learn atomic play decisions (especially for rare 3+ card plays).

v6 replaces the multi-head decomposition with a flat action space, redesigns the encoding, and adds rotation augmentation.

## Game Engine — Configurable Card Values

`game.py` currently hardcodes a deck of (i, j) pairs for 1 ≤ i < j ≤ 10. Parameterize to support N card values.

Changes to `game.py`:
- `create_deck(num_players, num_values=10)` — generate pairs for 1 ≤ i < j ≤ N
  - Deck size = C(N, 2) = N*(N-1)/2
  - Player-count adjustments: remove cards containing N for fewer players (same pattern as current 3-player removes 10s). Exact trimming TBD based on divisibility.
  - For N=7: C(7,2) = 21 cards. 4 players → 5 each (remove 1). 3 players → 7 each.
- `Game.__init__` takes `num_values` parameter, stored as `self.num_values`
- `_has_any_legal_play`, `Play.from_cards`, `Play.beats` — unchanged (they don't care about max value)
- `get_state_for_player` — add `num_values` to returned dict

No changes to turn logic, scoring, or phase management.

## New Encoding

Parameterized by `N` (num card values) and `H` (hand buffer size, 16).

### Hand — `H * (N+2) + H` dims

Per hand slot (16 slots, circular with hand_offset):
- N one-hot for card value (1-N)
- 1 empty indicator
- 1 top face scalar (value/N, 0 if empty)

Separate block:
- H bottom face scalars (value/N, 0 if empty), same circular offset

For N=10: 16*12 + 16 = 208
For N=7: 16*9 + 16 = 160

### Scout Cards — `4 * (N+1)` dims

4 possible scout cards (left normal, left flipped, right normal, right flipped).
Per card:
- N one-hot for the face value (which face is showing after the flip choice)
- 1 absent indicator (set when that option doesn't exist, e.g., right-end scout on a single-card play)

For N=10: 44. For N=7: 32.

These are NOT offset-rotated — they describe the play's end cards, not hand positions. They give the network direct visibility of what it would get from each scout option.

### Play Buffers — 16 + 5 = 21 dims

Two 4-card buffers, each position has top and bottom face scalars (/N):
- Left-aligned (right-padded): first 4 cards from left end
- Right-aligned (left-padded): last 4 cards from right end
- If play length > 4, truncate from the non-anchored end
- Empty positions = 0

This gives fixed-position access to both end cards regardless of play length.

Play metadata (5 dims):
- Type: 3 one-hot (none / set / run)
- Strength: scalar /N
- Length: scalar /10

### Metadata — ~28 dims

Same as v2 metadata with one addition:
- Hand size /H: 1
- Opponent hand sizes /H: 4 (zero-padded for < 5 players)
- Collected count /H: 1
- Opponent collected /H: 4
- Scout tokens /5: 1
- Opponent scout tokens /5: 4
- S&S available: 1
- Opponent S&S available: 4
- Player count /5: 1 (scalar, not one-hot)
- Scouts since play: 1 (normalized)
- Play owner relative position: 5 (one-hot)
- **Forced-play flag**: 1 (new — signals S&S step 2)

Total: 28

### Encoding Totals

Formula: `H*(N+2) + H + 4*(N+1) + 21 + 28 = H*N + 3H + 4N + 53`

- N=10, H=16: 301 dims
- N=7, H=16: 241 dims

### Implementation

New functions in `encoding.py`:
- `encode_state_v6(game, player, hand_offset, forced_play=False)` → tensor
- `get_flat_action_mask(game, player, legal_plays, hand_offset)` → bool tensor [384]
- Helper to compute scout card values from current play

All parameterized by `num_values` (from game) and `hand_slots` (constant 16).

Old encoding functions and mask functions remain for eval opponent compatibility.

## Flat Output Head — 384 outputs

Single policy head with 3 regions, one softmax over all legal actions:

```
[0..255]   Play actions — index = start*16 + end (start/end are offset-rotated hand slots)
[256..319] Scout actions — index = 256 + card*16 + insert_pos
             card: 0=left normal, 1=left flipped, 2=right normal, 3=right flipped
             insert_pos: offset-rotated hand position
[320..383] S&S scout actions — same layout as scout, offset by 320
```

Full 256 play outputs including start > end (always masked). Simpler indexing, negligible waste.

### Masking

One boolean mask of shape [384]:
- Play region: True for each (start, end) in `get_legal_plays()`, mapped to offset-rotated indices
- Scout region: True for each available scout card × all valid insert positions (0..hand_len, rotated)
- S&S region: True for each S&S variant × positions where inserting yields a legal play (existing `_sns_variant_legal` logic, but per-position)
- Forced-play mode (S&S step 2): scout and S&S regions fully masked, only play region active

### Network Interface

New network class `FlatScoutNetwork` in `network.py`:
- Same shared trunk: `[512, 256, 256, 128, 128, 128]` with ResidualBlocks
- `forward(x)` → hidden state (unchanged)
- `value(hidden)` → scalar (unchanged)
- `policy_logits(hidden)` → [384] raw logits
- Single linear layer: `nn.Linear(trunk_output_size, 384)`

No conditioning, no sequential heads. One forward pass → logits + value.

Old `ScoutNetwork` and `CircularCNNScoutNetwork` classes remain in `network.py` for loading eval opponent checkpoints.

### RandomBot

Update `RandomBot` to support both interfaces:
- Add `policy_logits(hidden)` → zeros [384]
- Keep old methods for eval compat

## S&S Two-Step Flow

When the flat head selects an S&S action (index 320-383):

**Step 1 — Scout insert:**
- Decode: card choice (left/right, normal/flipped) + insert position
- Execute: insert card into hand, flip S&S token, update play
- Record training sample: (state, action_index, old_log_prob, advantage_step1)
- Advantage from rollout: snapshot before S&S, rollout from after insert (before forced play)

**Step 2 — Forced play:**
- Re-encode state with `forced_play=True` flag
- Forward pass through network → logits with scout/S&S regions masked
- Sample play action from play region
- Execute the play
- Record training sample: (state_with_flag, play_action_index, old_log_prob, advantage_step2)
- Advantage from rollout: snapshot after insert, rollout from after play

Each step is an independent training sample with its own advantage. The value function sees both regular states and forced-play states (distinguished by the flag).

### Rollout Boundaries

The rollout system needs to handle S&S as two snapshot points per action:
- Snapshot A: before the S&S decision (step 1 advantage = V_after_insert - V_before)
- Snapshot B: after insert, before forced play (step 2 advantage = V_after_play - V_after_insert)

This is a change from the current system where S&S produces 2 records sharing one snapshot. Now each record gets its own snapshot and rollout.

## Rotation Augmentation

Each training sample can be augmented by shifting the hand offset. 16 offsets → 16x data (1 original + 15 augmented).

### What shifts

- Hand portion of state tensor: circular shift the slot dimension by k
- Bottom scalars: same shift
- Action index: play (s,e) → ((s+k)%16, (e+k)%16); scout/S&S position p → (p+k)%16
- Mask: same permutation as action index

### What doesn't shift

- Scout card inputs (describe the play, not hand positions)
- Play buffers
- Metadata
- Advantage (same game state, same outcome)
- Value target (same)

### Old log prob issue

The old_log_prob from the original forward pass doesn't apply at augmented offsets (the network isn't rotation-invariant yet — teaching that is the whole point). For each augmented offset, run a forward pass on the shifted state to compute the correct old_log_prob.

Procedure:
1. Collect rollout data at original offsets as normal
2. For each training sample, create 15 augmented copies (shift state, permute action index + mask)
3. Batch all augmented states, run one forward pass to compute old_log_probs
4. Add augmented samples to training batch with same advantages and value targets
5. Train PPO normally

Cost: one batched forward pass over 15 × ~400 = 6,000 states. Negligible.

### Precomputed permutation tables

For each offset k (0-15), precompute:
- `play_perm[k]`: maps original play index to augmented play index
- `scout_perm[k]`: maps original scout/S&S index to augmented index
- `full_perm[k]`: combined permutation for the 384-dim action space
- `hand_shift[k]`: index mapping for shifting hand slots in the state tensor

Store as tensors for GPU-side permutation.

## Training Pipeline

### Game Play

`play_games_with_rollouts()` changes:
- Single forward pass per decision → flat logits → masked sample → action
- S&S triggers a second forward pass for the forced play
- StepRecord stores: state tensor, flat action index, mask, old_log_prob, advantage
- No more separate action_type / start / end / insert fields

### Rollouts

`rollout_from_states_batched()` changes:
- Uses flat action sampling instead of sequential heads
- S&S in rollouts follows the same two-step flow

### PPO Update

`ppo_update()` simplifies:
- One set of logits per sample (no sub-head indexing)
- Policy loss = standard PPO clipped surrogate on the flat log probs
- Entropy = entropy of the masked flat distribution (single number, not per-head)
- No more `play_idx`, `end_idx`, `scout_idx` tensors for sub-head routing
- No `concatenate_batches` index offsetting (flat indices don't need offsetting)

### Replay Buffer

Keep the deque of N iterations. With augmentation, each iteration contributes ~400 × 16 = 6,400 samples. Buffer of 5-10 iterations should be plenty. Tune after initial runs.

### Diagnostics

Since there are no per-head entropies, compute conditional entropies as diagnostics:
- Entropy over play actions only (marginalize out scout/S&S)
- Entropy over scout actions only
- Play length distribution (from chosen play indices: length = end - start + 1)
- Scout insert position distribution
- Clip fraction per buffer age (the original diagnostic from handoff 039)

## Eval Opponents

Old checkpoints (v1_4, v2_5, v3_4, v4_2) keep working as eval opponents:
- Old `ScoutNetwork` / `CircularCNNScoutNetwork` classes remain in network.py
- Eval game logic uses the old multi-head action interface for old checkpoints
- `play_eval_game` dispatches based on network type

The v6 network is identifiable by class (`FlatScoutNetwork` vs `ScoutNetwork`).

## File Changes Summary

### `game.py`
- `create_deck(num_players, num_values=10)` — parameterize deck generation
- `Game.__init__` — accept and store `num_values`
- `get_state_for_player` — include `num_values` in output

### `encoding.py`
- New: `encode_state_v6()`, `get_flat_action_mask()`, scout card helper
- New: constants for v6 encoding (parameterized by N, H)
- New: `decode_flat_action()` — convert flat index to game action
- New: augmentation permutation tables
- Keep: all v1-v4 encoding functions and masks (eval compat)

### `network.py`
- New: `FlatScoutNetwork` class with flat policy head
- New: `batched_masked_sample` for flat logits (or adapt existing)
- Keep: `ScoutNetwork`, `CircularCNNScoutNetwork`, `RandomBot` (eval compat)
- Update `RandomBot` to support flat interface

### `training.py`
- Rewrite game play functions for flat action sampling
- Rewrite rollout functions for flat actions
- Update S&S handling for two-step flow with separate snapshots
- Add rotation augmentation (post-collection, pre-PPO)
- Simplify `ppo_update` (no sub-head routing)
- Simplify `prepare_ppo_batch` (flat indices only)
- Update diagnostics

### `main.py`
- New PARAMS for v6: `num_values`, `hand_slots`, `encoding_version: 6`, `augment_rotations: 16`
- Updated training loop for augmentation step
- Updated chart/logging for flat entropy + conditional entropies

### `fast_game.pyx`
- Keep existing legal play functions (game logic unchanged)
- May need new Cython mask function for flat output format if perf matters

## Open Decisions

- **Replay buffer size**: start with 5, tune based on clip fraction diagnostics
- **Rollout config**: start with 10 games × 50 rollouts, consider adjusting after seeing augmented data volume
- **Entropy bonus / floors**: single flat entropy replaces per-head floors. Start with a global entropy bonus and see if the flat head maintains enough exploration
- **Simplified game player count**: 3 or 4 players with 7 card values (3 players → 7 cards each from 21-card deck, cleanest fit)
