# V3 Encoding Failure & PPO Trunk Gradient Diagnosis

## Task & State

Implemented v3 encoding (handoff 028 design), tested it, found it doesn't work, diagnosed why. Then pivoted to investigating the deeper question: why can't PPO learn adjacent matching at all? Produced concrete diagnostic evidence about the PPO gradient failure mechanism.

V3 encoding is implemented but proven ineffective. The v3 pairwise diff encoding has a leftover modification: `_fill_pairwise_v3` currently uses `(values[i] - values[j] + 1.0) / 2.0` (shifted similarity) instead of the original `values[i] - values[j]` (signed diff). Neither version helps — the entire scalar approach is a dead end.

## What Changed

- `scout-bot/game.py` — Added `turn_number` tracking (increments in `_advance_turn`)
- `scout-bot/encoding.py` — Full v3 encoding implementation: `encode_state_v3`, `encode_hand_both_orientations_v3`, pairwise diff functions, v3 constants. `_fill_pairwise_v3` was modified to use shifted similarity `(diff + 1) / 2` (didn't help, see below).
- `scout-bot/probe.py` — Added `--v3` flag, v3 dispatch in frozen trunk probe (probe 8)
- `scout-bot/probe_diagnostic.py` — Added `--v3` flag, v3 dispatch
- `scout-bot/probe_ppo_variants.py` — Added `--v3` flag, v3 dispatch. Added three new tests:
  - `test_frozen_trunk` — CE pretrain trunk, freeze it, train scout_insert_head with PPO
  - `test_ce_then_ppo` — CE pretrain full network, switch to PPO with everything unfrozen
  - `test_gradient_compare` — Compute CE and PPO gradients on trunk from same batch, compare cosine similarity and magnitude

## Decisions

- **V3 scalar encoding is a dead end.** V2 one-hots [64,32] supervised CE: 0.775. V3 scalars+pairwise [64,32] CE: 0.312. V3 [256,128] CE: 0.207. Scaling up doesn't help — the failure is representational, not capacity. One-hot structure provides compositional matching that scalars destroy.
- **Shifted similarity (`(diff+1)/2`) didn't help.** Hypothesis was that diff=0 for matching produces zero gradient on connecting weights. Shifting to 0.5 gave nonzero gradient but CE still failed (0.223). The problem with v3 goes beyond the zero-gradient issue.
- **V3 should be dropped or left dormant.** Stay on v2 one-hots for training.

## Probe Results (all v2 unless noted)

| Test | Encoding | Network | Result |
|---|---|---|---|
| Test A (supervised CE) | v2 | [64,32] | **PASS** 0.227 -> 0.775 |
| Test A (supervised CE) | v3 | [64,32] | FAIL 0.223 -> 0.312 |
| Test A (supervised CE) | v3 | [256,128] | FAIL 0.214 -> 0.207 |
| Probe 5b (PPO) | v2 | [64,32] | FAIL ~0.22 -> ~0.23 |
| frozen_trunk (CE trunk + PPO head) | v2 | [64,32] | **PASS** 0.224 -> 0.620 |
| ce_then_ppo (CE pretrain, unfreeze PPO) | v2 | [64,32] | **PASS** 0.837 -> 0.946 |
| grad_compare (cosine sim CE vs PPO) | v2 | [64,32] | cos_sim=0.108, mag_ratio=1.6 |
| fixed_val (static target, PPO) | v2 | [64,32] | **PASS** 0.861 (from prior sessions) |

## Key Findings

- **PPO can learn the scout_insert head if the trunk already has good features** (frozen_trunk: 0.224 -> 0.620).
- **PPO can refine and improve CE-learned features** when starting from a good initialization (ce_then_ppo: 0.837 -> 0.946).
- **PPO's trunk gradient is nearly orthogonal to CE's** (cosine similarity 0.108 across 20 batches). The PPO gradient is not weaker (magnitude ratio 1.6x CE) — it points in a nearly unrelated direction.
- **PPO can learn matching with a static target** (fixed_val: 0.861) but not with a dynamic target (reading scouted card value from the state). The difference: static target produces consistent trunk gradients across samples; dynamic target produces high-variance gradients that average to near-noise.

## Next Steps

The core question is now well-characterized: PPO's policy gradient provides essentially no useful signal to the trunk for learning dynamic matching features. Options to consider:

1. **Supervised pre-training** — CE pretrain the scout matching features, then switch to PPO for full self-play. The ce_then_ppo result (0.946) shows PPO preserves and improves CE features.
2. **Auxiliary CE loss** — Add a supervised matching loss alongside PPO during training. Keeps the trunk learning matching features continuously.
3. **Architecture change** — Give the scout_insert_head direct access to the scouted card's encoding (extracted from the state at known positions), bypassing the trunk bottleneck. The head already receives action_type conditioning; this would add the scouted card identity.
4. **Accept the limitation** — Train with PPO as-is, accept suboptimal scout insertion. Focus on other aspects of play.

## Watch Out

- The `_fill_pairwise_v3` function in encoding.py currently uses the shifted similarity formula `(diff + 1) / 2` instead of the original signed diff. This was an experimental change that didn't help. If v3 is ever revisited, this should be noted.
- `probe_ppo_variants.py` has a `test_hint` that's abandoned mid-implementation (SKIPPED). It tried to append scouted card value directly to the scout head but couldn't work with ppo_update's batched recomputation.
