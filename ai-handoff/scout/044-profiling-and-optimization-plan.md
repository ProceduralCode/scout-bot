# Profiling and Optimization Plan

## Task & State

Investigated why v6 probes 10-11 fail, then profiled the training loop to plan performance work. Two optimization tasks are ready to implement — neither has been started yet.

## Probe 10-11 Investigation Results

Probes 10-11 (scout insertion quality and adjacent value matching) test whether the network can learn relational reasoning about card values. Ran supervised experiments with multiple architecture/data configurations:

**Without rotation augmentation**: Every configuration (FC, attention small/big, bigger trunks, 1k-5k training samples) memorizes perfectly (100% train acc) and gets ~4% test accuracy (random chance). No generalization at all.

**With 16x rotation augmentation** (1000 base samples, test metric = "any valid adjacent position"):

| Config | Test any-valid accuracy |
|---|---|
| Random chance | 17% |
| FC [128,64,64] | 29% |
| Attn32 h2 l1 [128,64,64] | **90%** |
| Attn64 h4 l2 [128,64,64] | 78% (overfits faster) |
| Attn32 h2 l1 [256,128,128] | 89% |

**With augmentation, 3000 base samples**: Attn32 [128,64,64] reaches 95%, Attn32 [256,128,128] reaches 96%.

Key findings:
- Rotation augmentation is essential — without it, networks memorize per-sample instead of learning rotation-equivariant patterns.
- Attention matters: FC+aug tops at 56% (3k data), attention+aug reaches 95%.
- Smaller attention (dim=32, 1 layer) outperforms larger (dim=64, 2 layers) — larger overfits faster.
- The probes fail because they don't use rotation augmentation and use exact-match against a single random target when there are ~5 valid targets per sample (avg 29.4 legal scout actions, avg 5.0 valid adjacent positions). The probes are disconnected from the real training pipeline which does use augmentation.
- Probe fixes needed: add augmentation to training loops, change metric to any-valid. These are probe-level fixes only — the architecture and training pipeline are fine.

## Profiling Results

Ran `python main.py --profile 1` (pyinstrument, 1 iteration). This resumes from iteration 142 in `v6_2/` with attention config `{"dim": 32, "heads": 2, "layers": 1}` and trunk `[512, 256, 128]`.

Total iteration: 111s (108s play, 2.7s train). Breakdown of the 107s inside `rollout_from_states_batched_v6`:

| Component | Time | % | Function |
|---|---|---|---|
| `encode_state_v6` | 30.7s | 29% | `encoding.py:866` |
| Python loop overhead | 16.7s | 16% | `training.py` [self] |
| `copy.deepcopy` | 15.3s | 14% | `training.py:1603` |
| `get_flat_action_mask` | 13.3s | 12% | `encoding.py:899` |
| Network forward | 10.9s | 10% | attention + FC trunk |
| `random.randint` | 6.0s | 6% | `random.py:358` |
| Game mutations | 7.9s | 7% | `apply_play` + `apply_scout` |
| Other | 6.2s | 6% | stack, sample, etc. |

Within `encode_state_v6` (30.7s): `_fill_metadata_v6` is 11.6s (Python attribute access overhead), `_fill_hand_v6` 6.6s, `_fill_play_buffer_v6` 3.4s, `_fill_scout_cards_v6` 2.3s, `_fill_hand_bottom_v6` 1.9s, `torch.from_numpy` 1.4s.

Within `get_flat_action_mask` (13.3s): 9.6s is [self] (the Python loops), 1.7s in `Play.from_cards` (called for S&S reduced play), 1.2s in `torch.from_numpy`.

Per iteration: ~12,500 rollout games, ~13,000 deepcopy calls, ~150,000 encode+mask calls.

## Next Steps — Two Optimizations

### 1. Game.clone() — replace copy.deepcopy (target: 15s → ~1-2s)

Add a `clone()` method to `Game` in `game.py`. The object tree is simple and fully known:
- `Game`: scalars (`num_players`, `num_values`, `total_rounds`, `round_number`, `current_player`, `scouts_since_play`, `turn_number`, `round_ender`, `starting_player`) + `cumulative_scores: list[int]` + `players: list[PlayerState]` + `current_play: Play | None` + `current_play_owner: int | None` + `phase: Phase` + `flips_remaining: set[int]`
- `PlayerState`: `hand: list[Card]`, `collected: list[Card]`, `scout_tokens: int`, `sns_available: bool`
- `Play`: `cards: list[Card]`, `count: int`, `play_type: PlayType`, `strength: int`
- All leaf values are immutable (tuples, ints, enums). Only lists and the set need copying.

Call sites to update in `training.py`:
- Line 1603: `games = [copy.deepcopy(s) for s in snapshots]` → `[s.clone() for s in snapshots]`
- Lines 1697, 1712, 1730, 1749: `copy.deepcopy(game)` → `game.clone()` (4 snapshot sites in `play_games_with_rollouts_v6`)
- Check if there are deepcopy calls in the non-v6 rollout path too (around line 379) — update those as well for consistency.

### 2. Cythonize encode_state_v6 + get_flat_action_mask (target: 44s → ~5-10s)

Create `scout-bot/fast_encoding.pyx` following the pattern of `fast_game.pyx`.

**`encode_state_v6`**: Port all `_fill_*_v6` helpers as C functions. The function takes a Game object, extracts Python data at the boundary (hand list, play cards, player states), then fills a `float[309]` C array with no Python calls. Returns `torch.from_numpy(np.asarray(...))`. Key constants to hardcode or declare as cdef: `H=16`, `N=10`, `SCOUT_CARDS_DIM=52`, `PLAY_BUFFER_DIM=21`, `METADATA_DIM=28`, `INPUT_SIZE=309`, `GLOBAL_START=260`.

The costliest sub-function is `_fill_metadata_v6` (11.6s) which is entirely Python attribute lookups (`game.players[p].hand`, etc.) — these become direct C struct-like access in Cython.

**`get_flat_action_mask`**: Port the mask-building logic. The play region is simple (loop over legal_plays, compute slot indices). The scout region is a loop over card_choices × positions. The S&S region needs `_has_any_legal_play_c` — **duplicate** the ~30-line C function from `fast_game.pyx` into `fast_encoding.pyx` (agreed with user, avoids build coupling). Also inline the `Play.from_cards` logic for the reduced play (same pattern as `_sns_variant_legal` in `fast_game.pyx`).

**Build/import integration**:
- Add `"fast_encoding.pyx"` to `setup.py`'s cythonize call
- Add fallback import at bottom of `encoding.py`: `try: from fast_encoding import encode_state_v6, get_flat_action_mask; except ImportError: pass`
- Build: `pushd scout-bot && python setup.py build_ext --inplace; popd`

**torch.from_numpy overhead** (1.4s + 1.2s = 2.6s combined): The Cython functions should still return torch tensors (the callers expect them). Use `np.asarray` on a C memoryview for the buffer, then `torch.from_numpy()`. This overhead stays but is minor compared to the Python loop elimination.

## Watch Out

- `encode_state_v6` returns a 309-dim float32 tensor. The constants changed in handoff 043 (from 301 to 309) — use the current values.
- `get_flat_action_mask` returns a bool tensor of size 384. The S&S region calls `_has_any_legal_play` which itself is already Cython when `fast_game` is available, but in `fast_encoding.pyx` we need our own C version since we can't cimport across .pyx files without a .pxd.
- The non-v6 paths also use `copy.deepcopy` in `rollout_from_states_batched` (line ~379 in training.py). Worth updating those to `game.clone()` too.
- Run `python main.py --profile 1` after both changes to verify the speedup.
- Run `python probe_v6.py` to verify correctness (probes 0-9 should still pass, especially probe 0 which tests encoding round-trips and probe 5/5b which test rotation).
