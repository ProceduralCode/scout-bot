# Scout: Scout Quality Investigation

## Task

Investigate why `eval_scout_quality` is near-random (1.53 vs random ~1.5) and determine whether this is an architecture limitation, training signal problem, or something else.

## Bug Fixes

### probe.py — numpy mask compatibility (from session 018)

`probe.py` wasn't updated when mask functions switched to numpy. Fixed:

- Added `import numpy as np`
- 4 `masked_sample` call sites wrapped with `torch.from_numpy()` (lines ~103, 116, 539, 736)
- 2 force-case masks changed from `torch.zeros(..., dtype=torch.bool)` to `np.zeros(..., dtype=np.bool_)` (lines ~100, 110)
- 2 `forced_at_mask` in probes 5/5b changed to numpy (lines ~619, 702)

### probe.py — ppo_update interface

`_train_iteration` still used the old `ppo_update(network, optimizer, records, advantages, ...)` interface. Updated to call `prepare_ppo_batch()` first, then pass the batch dict. Added `prepare_ppo_batch` to imports.

### probe.py — scout_insert stored decoded position instead of slot

`_make_scout_record` stored `sample["insert_pos"]` (decoded hand position) in `scout_insert`, but `ppo_update` expects the raw slot index (matching training.py's `rec.scout_insert = insert_slot`). This caused NaN policy loss because the slot-space log prob was gathered at a hand-position index. Fixed: `_sample_scout` now returns `insert_slot` in the dict, `_make_scout_record` stores it.

## Probe Results

Wrote two new probes and ran systematic experiments:

### Probe 9 — trivial control ("always insert at position 0")

Tests whether the scout training pipeline works at all.

| Network | Games×Iters | Result |
|---------|-------------|--------|
| [64,32] | 500×300 | **PASS** 0.08→1.00 |
| [512,256,256,128,128,128] | 500×300 | **PASS** 0.07→0.94 |

**Conclusion:** Training mechanics (PPO + GAE + scout head) work. The pipeline can learn scout insertion preferences.

### Probe 5b — adjacent matching (context-dependent insertion)

Tests whether the network can learn "place card next to matching value."

| Network | Games×Iters | Extras | Result |
|---------|-------------|--------|--------|
| [64,32] | 50×100 | — | PASS 0.21→0.29 (marginal) |
| [64,32] | 500×300 | — | FAIL 0.27→0.28 |
| [512,256,...] | 50×100 | — | FAIL 0.29→0.27 |
| [512,256,...] | 500×300 | — | FAIL 0.23→0.22 |
| [512,256,...] | 500×300 | entropy floors | FAIL 0.25→0.19 |

**Conclusion:** Context-dependent insertion consistently fails regardless of network size, data volume, or entropy management.

### Probe 8 — frozen trunk (v3_4 checkpoint, only train scout head)

Tests whether the trained trunk's features support scout insertion.

| Games×Iters | Result |
|-------------|--------|
| 500×300 | FAIL 0.23→0.25 |

**Conclusion:** Even with a pre-trained trunk that knows hand structure from play training, a fresh scout head can't learn adjacent matching via PPO.

### Probe 5 — full hand quality optimization

| Network | Games×Iters | Result |
|---------|-------------|--------|
| [512,256,...] | 500×300 | FAIL 2.25→2.17 |

## Key Observations

The dividing line in the probes is sharp: constant policy learnable, context-dependent policy not.

- The trivial task has one correct answer regardless of hand content — all samples reinforce the same pattern.
- The adjacent matching task requires a conditional mapping that varies per state. Each specific (card value × position × scouted card) combination appears only a few times per batch.
- Play start/end heads succeed in probes, but their masks narrow to 2-5 options. Scout insert has ~11 legal positions.
- The frozen trunk also failed, but this hasn't been compared to supervised learning — unclear if the features are absent or PPO just can't find the mapping.

Investigation is incomplete. The supervised probe (next step) would distinguish between "trunk features are insufficient" and "PPO can't extract the mapping from sufficient features."

## Next Steps

1. **Supervised probe** — compute optimal insertion positions, train the frozen v3_4 trunk's scout head with cross-entropy loss. If supervised learning works, the trunk features ARE sufficient and the problem is purely PPO sample efficiency. If it also fails, the trunk doesn't encode what the scout head needs.

2. **Mini-batching** — still the highest-impact training change overall. With ~250 gradient updates per iteration instead of 4, the policy explores more directions per data collection cycle. This might cross the threshold where PPO can start finding contextual patterns for scout insertion. Agreed on as the next implementation task.

## Modified Files

- **`scout-bot/probe.py`** — numpy mask fixes, ppo_update interface fix, scout_insert slot bug fix, new probes 8 (frozen trunk) and 9 (trivial control), checkpoint path resolution
