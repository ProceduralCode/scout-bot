# Self-Attention Architecture for Card Comparison

## Task & State

Add self-attention over per-card entities to the v6 architecture. The FC network cannot generalize cross-comparison between card values (4% test accuracy on matching tasks — see handoff 041). Self-attention provides weight-shared pairwise comparison as a built-in primitive.

This is a spec — no code changes have been made yet.

## Background

Supervised generalization tests (041) showed the FC architecture memorizes but cannot generalize "match scouted card value against hand card values." The root cause: FC layers compute `y = Wx + b` with fixed, position-specific weights. Comparing value at position A against value at position B requires multiplicative interaction (`x_a * x_b`), which the FC approximates position-specifically without weight sharing. Each (slot_i, slot_j) comparison must be learned independently.

Self-attention solves this by computing `softmax(QK^T / sqrt(d_k)) V` where Q, K, V projections are shared across all entities. The `QK^T` dot product is an explicit pairwise comparison applied uniformly to all pairs. One learned comparison function generalizes to all positions.

Cross-comparison is important beyond scout insertion — hand-to-hand value comparison across positions is central to Scout strategy (identifying runs, planning plays, evaluating hand quality).

## Design Decisions

- **Self-attention, not input engineering.** Explicit pairwise features (e.g., `|scout_value - slot_value|` per pair) would solve specific known comparisons but not generalize to unanticipated relational reasoning. Architecture fix is better.
- **No embedding layer.** One-hot card values go directly into Q/K/V projections. The projections ARE the embedding — a separate `nn.Embedding` layer is a redundant linear transform. Discussed and confirmed.
- **Keep circular buffer + rotation augmentation.** Self-attention has shared weights across positions (all positions trained equally in attention), but the FC trunk after attention sees flattened per-position features. The circular buffer ensures equal training for those FC weights. Augmentation is cheap (2s training time).
- **Keep value head.** Used for hand flip decisions and PPO value loss. Rollout mode still trains it (value target comes from rollout snapshot values). Negligible compute cost.
- **One-hot positional encoding (20 dims).** Positions 0-15 = hand slots, 16-19 = four scout options. Simpler than sinusoidal/learned, sufficient for 20 positions. Circular adjacency (slot 15 ↔ slot 0) is learned via rotation augmentation training signal.
- **No attention masking for empty/absent entities.** The empty flag gives the network enough signal to learn to down-weight empty slots. Skip `key_padding_mask` unless it becomes a problem.
- **Linear entity projection (no ReLU).** Standard transformer convention — attention provides nonlinear interaction via softmax.
- **Modify v6 in place.** No need for a separate v7 encoding version. Old v6 checkpoints don't need to load (only pre-v6 checkpoints need compat).

## Encoding Changes (encoding.py)

### New constants

```
SCOUT_CARDS_DIM_V6: 44 → 52 (add 4 top scalars + 4 bottom scalars)
INPUT_SIZE_V6: 301 → 309
GLOBAL_START_V6 = HAND_DIM_V6 + SCOUT_CARDS_DIM_V6  # 260
GLOBAL_DIM_V6 = PLAY_BUFFER_DIM_V6 + METADATA_DIM_V6  # 49
NUM_ENTITIES_V6 = HAND_SLOTS_V6 + 4  # 20
```

### New flat layout (309 dims)

```
[0:192]   hand top — 16 slots × 12 (one-hot[10] + empty[1] + top_scalar[1])  [SHIFTED]
[192:208] hand bottom — 16 scalars                                            [SHIFTED]
[208:252] scout one-hots — 4 × 11 (one-hot[10] + absent[1])                  [not shifted]
[252:256] scout top scalars — 4 values (NEW)                                  [not shifted]
[256:260] scout bottom scalars — 4 values (NEW)                               [not shifted]
[260:281] play buffer — 21 dims                                               [not shifted]
[281:309] metadata — 28 dims                                                  [not shifted]
```

### _fill_scout_cards_v6 modifications

After the existing 4 one-hot blocks (44 dims), write 8 additional scalars:

- Scout top scalars: `face_value / N` for each option's showing face
- Scout bottom scalars: `other_face / N` for each option's hidden face

Mapping (cards = current_play.cards, left = cards[0], right = cards[-1]):
- Left normal: top = left[0]/N, bottom = left[1]/N
- Left flipped: top = left[1]/N, bottom = left[0]/N
- Right normal: top = right[0]/N, bottom = right[1]/N
- Right flipped: top = right[1]/N, bottom = right[0]/N

When current_play is None, all scalars = 0.
When len(cards) == 1, right options absent, right scalars = 0.

### encode_state_v6

Replace inline `off += 4 * (N + 1)` with `off += SCOUT_CARDS_DIM_V6`.

### Permutation tables

`_build_permutation_tables` uses `INPUT_SIZE_V6` for `state_size`. The table starts with identity (arange) then overwrites hand positions. New dims 252-259 are in the non-shifted region — identity mapping is correct automatically. Just updating `INPUT_SIZE_V6` is sufficient.

### encode_hand_both_orientations_v6

Only zeros/re-fills the hand region [0:208]. Scout scalars at [252:260] are unaffected by hand flip. No change needed.

## Network Changes (network.py)

### Modify FlatScoutNetwork

Add `attention` parameter to `__init__` (dict with `dim`, `heads`, `layers`):

```python
"attention": {"dim": 32, "heads": 2, "layers": 1}
```

#### New components

- `entity_indices` — registered buffer [20, 13], precomputed indices for gathering each entity's 13 features from the flat input vector:
  - Hand entity i: one-hot[10] from `[i*12 : i*12+10]`, empty from `[i*12+10]`, top_scalar from `[i*12+11]`, bottom_scalar from `[192+i]`
  - Scout entity j: one-hot[10] from `[208+j*11 : 208+j*11+10]`, absent from `[208+j*11+10]`, top_scalar from `[252+j]`, bottom_scalar from `[256+j]`
- `position_onehots` — registered buffer [1, 20, 20] (eye matrix, broadcast over batch)
- `entity_proj` — Linear(33, d_model), projects entity features to attention dimension
- `attention_layers` — ModuleList of pre-norm attention layers, each containing:
  - LayerNorm(d_model)
  - nn.MultiheadAttention(d_model, num_heads, batch_first=True)
  - Residual connection

#### Modified forward()

```
flat input [batch, 309]
  → gather entities [batch, 20, 13]
  → concat position one-hots → [batch, 20, 33]
  → entity_proj → [batch, 20, d_model]
  → attention layers (pre-norm + residual) → [batch, 20, d_model]
  → flatten → [batch, 20 * d_model]
  → concat global features x[:, 260:] → [batch, 20*d_model + 49]
  → FC trunk (layer_sizes) → hidden
```

FC trunk `input_size` = `20 * attention["dim"] + 49`. Computed from attention config, not passed as a separate parameter.

#### policy_logits() and value() — unchanged

### Handle unbatched input

Current code supports both batched and unbatched (single state) inputs. Maintain this — unsqueeze at start, squeeze at end.

## main.py Changes

### PARAMS

Add:
```python
"attention": {"dim": 32, "heads": 2, "layers": 1},
```

### Network construction (line ~347)

Pass attention config:
```python
network = FlatScoutNetwork(INPUT_SIZE_V6, cfg["layer_sizes"],
    encoding_version=6, attention=cfg.get("attention"))
```

## Other Files

### matchup.py (line ~33) and training.py OpponentPool (line ~996)

These construct FlatScoutNetwork from saved checkpoint configs. Need to pass `attention` from the checkpoint's config dict. These construct eval/opponent networks from saved configs, so the attention config must be saved in checkpoints (it already will be — it's in `cfg` which gets saved).

### probe_v6.py

Probes construct FlatScoutNetwork directly. Update `_make_net()` (line ~47) to pass a default attention config. Probe 0 checks INPUT_SIZE_V6 — will automatically use new value.

### test_scout_generalization.py

Scratch file, constructs FlatScoutNetwork. Update if desired, but it's diagnostic and can be left broken.

## What Does NOT Change

- Game logic (game.py)
- Legal play computation (fast_game.pyx, encoding.py mask functions)
- Action space (384 flat actions)
- Action encoding/decoding
- PPO update logic (ppo_update_v6)
- Advantage computation (rollout-based)
- Replay buffer
- Augmentation logic (augment_rotation_v6) — operates on flat states, size change handled by updated constants
- GAE code (non-v6 path)

## Verification

After implementation:
1. Run probe_v6.py — all probes should pass (especially probe 0 for encoding, probe 5 for augmentation)
2. Run a few training iterations — verify loss decreases, no NaNs
3. Check that attention output shapes are correct through the pipeline
4. Optional: re-run test_scout_generalization.py test 4 (adjacent value matching) with the new architecture to confirm the relational reasoning gap is closed

## Watch Out

- The `entity_indices` buffer encodes exact offsets into the flat vector. If the encoding layout changes, these must be updated in sync.
- `nn.MultiheadAttention` requires `d_model` divisible by `num_heads`. Validate in constructor.
- Pre-norm attention (LayerNorm before attention, not after) is important for RL stability per Parisotto et al. 2019.
- The FC trunk's first layer now takes 20*d_model+49 dims instead of 301. With dim=32 that's 689. With dim=64 that's 1329. Large d_model increases the FC trunk's parameter count substantially.
