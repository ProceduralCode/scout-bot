# Cython Optimization and Training Analysis

## Task & State

Implemented both performance optimizations from handoff 044, then analyzed training progress and hyperparameter issues.

## What Changed

- **scout-bot/game.py** — Added `Game.clone()` method (fast manual copy replacing `copy.deepcopy`)
- **scout-bot/training.py** — Replaced 8 `copy.deepcopy(game)` call sites with `game.clone()` (2 network deepcopy calls unchanged)
- **scout-bot/fast_encoding.pyx** — New file. Cython implementations of `encode_state_v6` and `get_flat_action_mask`. Duplicates `_has_any_legal_play_c` from fast_game.pyx to avoid build coupling.
- **scout-bot/setup.py** — Changed to explicit `Extension()` objects (required because "scout-bot" directory name has a hyphen, which Cython rejects as a module name when inferring from path)
- **scout-bot/encoding.py** — Added fallback import for `fast_encoding` (same pattern as `fast_game`)

## Profiling Results

Before → After (1 iteration):

| Component | Before | After |
|---|---|---|
| encode_state_v6 | 30.7s | ~0s (gone from profile) |
| get_flat_action_mask | 13.3s | ~0s (gone from profile) |
| copy.deepcopy → Game.clone | 15.3s | 1.7s |
| **Total iteration** | **111s** | **82s** |

Remaining bottlenecks in the rollout loop: Python loop overhead (30.5s), network forward (15.5s), game mutations (11s), random.randint (9.2s).

Probes 0-9 all pass. Probes 10-11 still fail (expected — they need augmentation fixes, not related to these changes).

## Training Analysis

### v6_2 (iter 365, ppo_epochs=8, entropy_bonus=0.01, rollouts_per_state=50)

Eval margins: v1_4 +1.99, v2_5 -3.04, v3_4 -9.60, v4_2 -17.38. Entropy collapsed to 0.26, clip fraction 0.21, KL 0.04, explained variance declining (0.56→0.51). Play length declining (1.66→1.56), singles increasing.

Diagnosed interaction: low entropy → replay buffer data increasingly off-policy → large importance ratios → high clipping → noisy gradients → value function degradation → noisier advantages (feedback loop).

### v6_3 (iter 75, ppo_epochs=4, entropy_bonus=0.05, rollouts_per_state=25)

Fresh run. Learns pairs rapidly (iter 1-20), then stagnates. At iter 75: entropy stable at 1.5 (good), clip/KL healthy (0.08/0.007), but explained variance only 0.46. Eval: v1_4 -2.27, v2_5 -6.15. Play length oscillating around 1.55.

rollouts_per_state was reduced from 50 to 25 (unclear if intentional). Lower rollout count → noisier advantage estimates → lower explained variance → slower learning of subtle behaviors.

## Open Questions

- **Is rollouts_per_state=25 intentional for v6_3?** At 50 in v6_2, explained variance reached 0.56. At 25 in v6_3, it's stuck at 0.46. This could be a significant factor in the stagnation.
- **Is the rollout advantage signal sufficient for subtle behaviors?** Probes show the network *can* learn play length preference and scout quality with direct signal. But in self-play, the advantage of a longer play is diluted across dozens of subsequent turns and N rollouts. The explained variance (~0.50) means half the return variance is noise from the policy gradient's perspective.
- **Should v6_2 continue?** It's still improving vs most opponents despite the entropy issues. The param changes in v6_3 prevent entropy collapse but may not address the core signal quality issue.
