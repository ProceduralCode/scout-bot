# Scout: V2 Diagnostic Tests — Encoding Cleared, PPO Isolated

## Task & State

Finished v2 support in `probe_diagnostic.py` (left incomplete in handoff 025), then ran v2 diagnostic tests A, F, H, and S. All passed. V2 encoding fully supports adjacent matching with supervised CE at all rotation levels. The problem is now isolated to PPO optimization.

## What Changed

- **`scout-bot/probe_diagnostic.py`** — completed v2 support:
  - Fixed broken `CONDITIONING_SIZE` import (symbol didn't exist in `network.py` — would crash at import time)
  - Tests B, C, D, E: `ScoutNetwork(layer_sizes=...)` → `_make_network(layer_sizes)`
  - Test A: replaced inline v1 encoding with `_encode()` helper, `_ev_insert_size()`, version-aware `_pss` for conditioning
  - Test F + `_eval_adj_rate_standalone`: v2 dispatch (`encode_state_v2` vs `encode_state`), version-aware sizes
  - Test G: `_make_network()`, conditioning size computed as `layer_sizes[-1] + ACTION_TYPE_SIZE + _pss` (was broken `CONDITIONING_SIZE`), `_encode()` for encoding, version-aware sizes
  - Test S: v2 ho-only sweep configs `[(1,1), (2,1), (4,1), (8,1), (15,1)]`, v2-aware MLP dimensions and encoding dispatch
  - Added `--v2` CLI flag to `main()`

## V2 Diagnostic Results

All tests with `--v2`:

- **Test A** (supervised CE, shared trunk [64,32]): 0.259 → 0.864 — PASS
- **Test F** (standalone MLP [256,128], random ho): 0.230 → 0.983 — PASS
- **Test H** (standalone MLP [256,128], fixed ho=0): 0.236 → 0.994 — PASS
- **Test S** (v2 ho-only sweep, [256,128]):

| ho | combos | adj_rate |
|----|--------|----------|
| 1  | 1      | 1.000    |
| 2  | 2      | 0.997    |
| 4  | 4      | 1.000    |
| 8  | 8      | 0.994    |
| 15 | 15     | 0.997    |

Compare v1 sweep (from handoff 021): degraded from 0.97 (1 combo) to 0.25 (200 combos). V2 shows no degradation — flat ~1.0 across all ho counts.

## Key Facts

- V2 encoding supports adjacent matching at every rotation level with supervised CE
- The standalone MLP (no shared trunk) reaches 0.983-0.997 — encoding is not a bottleneck
- The shared trunk [64,32] reaches 0.864 — architecture works too, just slower to converge
- PPO probes 5/5b still FAIL with v2 (from probe.py runs in handoff 025) despite supervised CE working perfectly on the same encoding

## Next Steps

1. **Investigate PPO failure** — the encoding is cleared. The question is now why PPO can't learn what supervised CE learns trivially. Possible angles:
   - Run v2 probe 5b with more iterations/games (current: 50 games × 100 iters, very small)
   - Auxiliary supervised CE loss on scout insertion during full training
   - Reward shaping (graded reward instead of binary +1/-1)
   - Signal dilution from shared trunk competing with other heads
   - Run Tests B-E with `--v2` to see if value loss ablation or fixed ho help PPO specifically

2. **Monitor v4_2 training** — still running (was 8000+ iters as of handoff 025)
