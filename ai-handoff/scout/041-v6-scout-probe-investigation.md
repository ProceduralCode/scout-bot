# V6 Scout Probe Investigation

## Task & State

Added v6 scout insertion probes (10 and 11) to `probe_v6.py`, then investigated why they fail. Ran a graduated series of supervised generalization tests to isolate the failure point.

## What Changed

- **Modified `scout-bot/probe_v6.py`** — added probes 10 (scout insert quality) and 11 (scout adjacent matching), plus helpers `_hand_quality`, `_sample_scout_only`. Both probes fail.
- **Created `scout-bot/test_scout_generalization.py`** — four supervised generalization tests at increasing difficulty. Scratch file, can be deleted or kept for reference.

## Key Results

Probe pipeline verified correct — encoding, masks, decoding, position mapping all work. The failures are about learning, not plumbing.

Supervised generalization tests (fixed train/test split, CE loss, [128,64,64] network):

| Test | Requires | Train acc | Test acc |
|---|---|---|---|
| 1. Lowest scout index | State-independent bias | 100% | 100% |
| 2. Insert at middle | Read hand_size from encoding | 100% | 100% |
| 3. Insert next to highest card | Read card values from hand slots | 100% | 70% |
| 4. Adjacent value matching | Match scouted card value against hand values | 100% | 4% |

Test 4 also fails with [512,256,256,128] (production-size network) and 2000 training samples. The network memorizes perfectly but cannot generalize.

Quality landscape statistics (1000 states): 36% of scout situations have completely flat quality (every position gives the same longest-play), 55% have spread=1.

## Decisions

- Probes 10 and 11 are kept even though they fail — they document the boundary of what the architecture can learn.
- Did not modify the encoding or architecture. The investigation was diagnostic only.

## Next Steps

- Run v6 training for 50-100 iterations (original plan from handoff 040). Real training uses rollout advantages that capture full game outcomes, not isolated heuristics.
- Observe whether scout behavior improves from game-level signal even though isolated scout probes fail.
- Diagnostics/visualization during training was discussed but not implemented.

## Watch Out

- The generalization failure is specifically at cross-referencing: matching a value from one part of the encoding (scouted card from current play) against values in another part (hand slots). The network handles global hand properties fine (highest card, hand size).
- This may or may not matter for real training — rollout advantages capture game-level impact, which might be learnable through simpler features than value-matching.
