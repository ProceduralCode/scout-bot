# Scout: V3 Encoding Design — Pairwise Differences

## Task & State

Designed a new v3 encoding to solve the PPO composition failure (handoff 027). The core idea: PPO can learn to *use* comparisons (fixed_val: 0.861) but can't *compute* them through RL gradient. So precompute all pairwise card value differences and provide them as input. This is a relational primitive, not hand-engineered strategy — the network still learns from gameplay reward which comparisons matter.

Design is complete and user-approved. Implementation not started.

## V3 Encoding Layout (286 dims)

```
--- HAND (45 dims) ---
[0:15]      Hand top face values — value/10 per slot (0.0 = empty)
[15:30]     Hand bottom face values — value/10 per slot (0.0 = empty)
[30:45]     Hand occupancy — 1.0 if card present, 0.0 if empty
            Rotated by hand_offset (same as v1/v2)

--- PLAY END CARDS (6 dims, fixed position for pairwise) ---
[45]        Left end top face — value/10 (0.0 = no play)
[46]        Left end bottom face — value/10
[47]        Right end top face — value/10 (0.0 = single card or no play)
[48]        Right end bottom face — value/10
[49]        Left end present — 1.0/0.0
[50]        Right end present — 1.0/0.0

--- PLAY BUFFER (30 dims, rotated context) ---
[51:61]     Play card top face values — value/10 (10 slots, 0.0 = empty)
[61:71]     Play card bottom face values — value/10 (10 slots, 0.0 = empty)
[71:81]     Play card occupancy — 1.0/0.0 (10 slots)
            Rotated by play_offset

--- PLAY META (5 dims) ---
[81:84]     Play type — 3-dim one-hot [no_play, set, run]
[84]        Play strength — strength/10 (0.0 = no play)
[85]        Play length — length/10 (0.0 = no play)

--- PAIRWISE DIFFERENCES (171 dims) ---
[86:257]    Signed diff (value[i] - value[j]) for all pairs i < j
            Zeroed when either card is empty/absent
            19 card values used:
              0–14:  hand top face slots 0–14 (rotated)
              15:    left end top face (fixed)
              16:    left end bottom face (fixed)
              17:    right end top face (fixed)
              18:    right end bottom face (fixed)

--- METADATA (28 dims) ---
[257]       Your hand size / 15
[258:262]   Opponent hand sizes / 15 (4 slots, zero-padded)
[262]       Your collected count / 15
[263:267]   Opponent collected counts / 15 (4 slots, zero-padded)
[267]       Your scout tokens / 5
[268:272]   Opponent scout tokens / 5 (4 slots, zero-padded)
[272]       Your S&S availability
[273:277]   Opponent S&S availability (4 slots, zero-padded)
[277]       Player count / 5
[278]       Scouts since play, normalized
[279:284]   Play owner relative position one-hot (5 slots)
[284]       Turn number / 50

TOTAL: 285 dims
```

Note: metadata is 28 dims (player count changed from 3-dim one-hot to scalar, added turn number). Total is 285, not 286 — the 286 in discussion was before the metadata changes.

## Decisions

- **Scalar card values over one-hots**: value/10 replaces 11-dim one-hot per card. Numerical closeness already captures adjacency (runs) and equality (sets). Network has 6 layers to learn nonlinear value-specific behavior if needed.
- **Hand bottom faces included (15 dims, no pairwise)**: Both faces are visible to human players in physical Scout. When you play cards, opponents can scout them and use the bottom face value. Passive context only — no pairwise comparisons computed for bottom faces.
- **Play buffer with rotation**: Full 10-slot play buffer (both faces + occupancy) rotated by play_offset. Without rotation, slots 5+ get almost no training data since plays are typically 1-4 cards. Separate from the 4 fixed-position end card values used by pairwise.
- **Play end cards duplicated**: The 4 end card values appear both as fixed-position inputs (for pairwise) and within the rotated play buffer (for context). Small redundancy, clean separation.
- **Upper triangle pairwise only (171 pairs)**: Signed difference a−b = −(b−a), so lower triangle is redundant. Linear layers can negate trivially.
- **Empty slot handling**: Pairwise diffs zeroed when either card is empty/absent. Occupancy flags (15 hand + 2 play end) let the network disambiguate "empty-empty" from "same-value match" (both produce diff=0). User considered remapping values to avoid this ambiguity but agreed occupancy flags handle it.
- **Player count as scalar**: Changed from 3-dim one-hot to player_count/5. Consistent with v3 scalar philosophy.
- **Turn number added**: `turn_number / 50` — gives the network a monotonic signal for game progress. Game object doesn't currently track this; needs to be added.
- **Keep v2**: v2 encoding stays for eval against existing best bot. Three parallel encoding tracks.

## What Changed

No code changes — design only.

## Next Steps

1. **Add turn tracking to Game object** — `game.py` doesn't have a turn counter. Need to add one (increment in `_advance_turn()`).
2. **Implement v3 encoding in `encoding.py`** — new constants, `_fill_hand_v3`, `_fill_play_end_v3`, `_fill_play_buffer_v3`, `_fill_pairwise_v3`, `_fill_metadata_v3`, `encode_state_v3(game, player, hand_offset, play_offset)`.
3. **Add v3 support to `network.py`** — `ScoutNetwork` already takes `input_size` as param, so just pass the new size. Head sizes unchanged from v2.
4. **Update probe files for `--v3`** — `probe_diagnostic.py` and `probe_ppo_variants.py` use the real encoding functions. Add v3 dispatch.
5. **Run probe 5b with v3** — the critical test. If adjacent matching (variable target) passes with v3 pairwise diffs, the approach is validated.
6. **If probe passes**: implement v3 in `training.py` game generation paths and train.

## Watch Out

- Total is 285 dims, not 286 — the metadata changes (player count scalar, turn number) net out to −1 from what was discussed mid-conversation.
- `encode_state_v3` takes both `hand_offset` and `play_offset` (like v1, unlike v2 which dropped play_offset).
- The pairwise section uses the 15 rotated hand top values and the 4 fixed play end values. The play buffer's rotated values are NOT used for pairwise — only the fixed end card positions at [45:48].
- Game object needs a turn counter added before v3 metadata can be implemented.
