# Scout: Rotation Bottleneck Isolation

## Task

Implement and run the isolation tests proposed in session 020's next steps: standalone MLP (bypassing trunk) and MLP scout head (replacing linear head). Then verify the findings.

## What Changed

- **`scout-bot/probe_diagnostic.py`** — added Tests F, G, H, S and supporting functions:
  - `test_standalone_mlp()` — raw 475-dim state → [256,128] MLP → 21 logits, no trunk. Accepts `fixed_ho`/`fixed_po` params.
  - `test_mlp_scout_head()` — ScoutNetwork with linear scout head replaced by [64,21] MLP. Supervised CE.
  - `test_rotation_sweep()` — trains fresh MLPs at 8 different (ho_count, po_count) configs, prints summary table.
  - `_eval_adj_rate_standalone()` — eval function for standalone MLPs (supports fixed offsets).
  - New imports: `torch.nn as nn`, `INPUT_SIZE`, `CONDITIONING_SIZE`.

## Test Results

### Architecture isolation (supervised CE, 300 iters, 500 games)

| Test | Description | adj_rate | Loss |
|------|-------------|----------|------|
| F | Standalone MLP, random ho/po | 0.250 | 2.47 | FAIL |
| G | MLP scout head on trunk, random ho/po | 0.242 | 2.46 | FAIL |
| H | Standalone MLP, ho=0 po=0 | 0.968 | 1.01 | PASS |

### Single-axis isolation (standalone MLP, 300 iters, 500 games)

| ho | po | adj_rate |
|----|----|----------|
| fixed 0 | fixed 0 | 0.968 |
| random | fixed 0 | 0.826 |
| fixed 0 | random | 0.544 (still climbing at iter 299) |
| random | random | 0.250 |

### Rotation complexity sweep (standalone MLP, 200 iters, 300 games)

| ho_count | po_count | combos | adj_rate | loss |
|----------|----------|--------|----------|------|
| 1 | 1 | 1 | 0.872 | 1.24 |
| 2 | 2 | 4 | 0.584 | 1.52 |
| 4 | 2 | 8 | 0.532 | 1.55 |
| 4 | 4 | 16 | 0.315 | 2.06 |
| 8 | 4 | 32 | 0.270 | 2.29 |
| 10 | 5 | 50 | 0.260 | 2.42 |
| 10 | 10 | 100 | 0.278 | 2.43 |
| 20 | 10 | 200 | 0.251 | 2.46 |

Grouping by po_count shows po dominates: at any ho_count, doubling ho barely changes the rate, but increasing po from 2→4 drops it from ~0.55 to ~0.30.

## Observations

- The adjacent matching task is easily learnable (0.97) when offsets are fixed, even with a simple [256,128] MLP operating on the raw 475-dim encoding.
- Neither trunk compression nor linear head capacity is the bottleneck — the standalone MLP (bypassing both) also fails with random offsets.
- Play offset rotation is the primary difficulty. The MLP can't extract a card value from more than ~2 variable positions in the play encoding.
- Hand offset rotation is secondary — performance with random ho + fixed po is still 0.83. Adjacency structure is preserved under hand rotation, so the network can learn a single pattern that generalizes.
- The combination hits baseline by ~16 ho×po combinations.

## Next Steps

1. **Decide on a fix for the actual training system.** Options include:
   - Adding explicit features (e.g., scouted card value as a scalar or one-hot) so the network doesn't need to extract it from the rotated play encoding
   - Using fixed po for scout decisions while keeping random po for play heads (requires dual encoding or second forward pass)
   - Removing play offset rotation entirely (loses augmentation benefit for play heads)
2. **Mini-batching** — still the highest-impact general training change, independent of scout insertion.
