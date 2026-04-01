# Scout: Cython & Encoding Optimization

## Task

Profile `play_games_batched` with the production network and optimize the Python bottlenecks identified in the previous session (encoding 22%, masks 19%, get_legal_plays 10%).

## What Changed

### Modified Files

- **`scout-bot/encoding.py`**
  - Added `import numpy as np`.
  - Replaced `_encode_hand`, `_encode_play`, `_encode_metadata` with `_fill_hand`, `_fill_play`, `_fill_metadata` that write directly into pre-allocated numpy arrays instead of building Python lists.
  - `_fill_metadata` reads directly from the `Game` object, skipping the intermediate `get_state_for_player()` dict.
  - `encode_state` and `encode_hand_both_orientations` now allocate `np.zeros(INPUT_SIZE, dtype=np.float32)`, fill it, and return `torch.from_numpy(buf)` (near-free, shared memory).
  - `encode_hand_both_orientations` reuses the play+metadata portion by copying the numpy buffer and overwriting just the hand section for the flipped orientation.
  - Added try/except import at end of file: `from fast_game import get_legal_plays, _has_any_legal_play, _sns_variant_legal` with pure-Python fallback.

### New Files

- **`scout-bot/fast_game.pyx`** — Cython module implementing `get_legal_plays`, `_has_any_legal_play`, `_sns_variant_legal`. Inner loops use C arrays (`int[20]`, `int[21]`) with no Python object manipulation. `_sns_variant_legal` inlines `Play.from_cards` logic and uses incremental array building (O(1) per insert position) instead of constructing new Python lists.
- **`scout-bot/fast_game.cp311-win_amd64.pyd`** — Compiled Cython extension (Windows, Python 3.11).
- **`scout-bot/setup.py`** — Cython build configuration.
- **`scout-bot/profile_batched.py`** — Profiling script for `play_games_batched` using pyinstrument.

## Performance

Baseline → after encoding → after Cython (100 games, production [512,256,256,128,128,128] network):

- **7.58s → 7.05s → 4.04s** (1.88x total speedup)
- Combined with previous 2.2x batching speedup = **~4x faster** than original per-game path.
- Encoding: 2.01s → 0.69s (torch.tensor list conversion eliminated)
- get_legal_plays: 0.55s → invisible (~15x+)
- _sns_variant_legal + _has_any_legal_play: ~0.93s → invisible
- get_action_type_mask: 1.22s → 0.56s (algorithmic cost gone, remaining is torch.zeros + Python conditions)

Remaining bottleneck distribution (4.04s):
- Loop self-time: 0.77s (19%)
- Encoding: 0.61s (15%) — _fill_metadata dominates at 0.25s
- get_action_type_mask self-time: 0.50s (12%)
- Game.apply_play / Play.from_cards: 0.33s (8%)
- Simple mask functions: 0.65s combined (16%) — torch.zeros overhead
- Forward pass: 0.28s (7%) — irreducible

## Build

Rebuild the Cython extension after editing `fast_game.pyx`:
```
python -c "
import os, sys
os.chdir('scout-bot')
sys.argv = ['setup.py', 'build_ext', '--inplace']
from setuptools import setup, Extension
from Cython.Build import cythonize
ext = Extension('fast_game', sources=['fast_game.pyx'])
setup(ext_modules=cythonize([ext], language_level='3'))
"
```
The `scout-bot` directory has a hyphen, which Cython rejects as a module name. The Extension object with explicit name is required — plain `cythonize("fast_game.pyx")` fails even with os.chdir. The `setup.py` in scout-bot/ has the simple form but won't work when invoked from the workspace root.

## Next Steps

- **Play length breakdown** — user wants avg_play_length chart replaced with multi-line showing fraction of plays by length (1-card, 2-card, 3-card, 4+) like the action type distribution chart.
- **Eval random bot toggle** — add `"eval_random_bot": False` to PARAMS so the random bot can be excluded from eval.
- **Further optimization (diminishing returns)** — mask functions returning numpy instead of torch tensors (~0.65s), encoding fill functions in Cython (~0.6s), Play.from_cards in Cython (~0.3s). Each saves 0.1-0.3s.

## Watch Out

- **The .pyd is Windows/Python-3.11-specific.** If Python version changes, rebuild.
- **Three code paths still exist** for turn logic (see context.md). If game mechanics change, all three need updating. The Cython module implements the same algorithms as the Python fallback — changes to legal-play logic need updating in both `encoding.py` AND `fast_game.pyx`.
- **`get_state_for_player()` in game.py is now unused** by encoding but still exists. Don't delete it — may be useful for debugging.
