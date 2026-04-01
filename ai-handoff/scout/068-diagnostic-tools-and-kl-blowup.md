# Diagnostic Tools & KL Blowup Investigation

## Task & State

Built three diagnostic/probe scripts, then investigated why v7_9 is actively regressing. Identified and empirically confirmed the root cause: entropy collapse makes the fixed learning rate produce ~34x too much KL divergence per step, so KL early stopping truncates nearly all learning.

## What Changed

### New files
- `scout-bot/canonical_states.py` — hand-crafted game states with obvious correct actions + 10 random mid-game baselines. Tests whether the network has learned basic strategy.
  - 6 canonical scenarios (pair beats single, no legal plays, opening move, near-empty hand, high single vs scout, can't beat strong play)
  - Reports P(play/scout/sns), top-5 actions, value estimate per state
  - Usage: `python -u scout-bot/canonical_states.py [checkpoint_dir]`

- `scout-bot/attention_probe.py` — extracts self-attention weights without modifying network.py (replays attention path with `average_attn_weights=False`). Reports per-head entropy, cross-block mass (hand↔scout), top edges, head specialization.
  - Usage: `python -u scout-bot/attention_probe.py [checkpoint_dir]`

- `scout-bot/kl_sensitivity_test.py` — loads a checkpoint, generates one training batch, measures KL divergence produced by a single optimizer step at 6 different learning rates (0.01x to 3x base). Directly tests whether the current LR is appropriate for the current entropy level.
  - Usage: `python -u scout-bot/kl_sensitivity_test.py [checkpoint_dir]`

### Modified files
- `scout-bot/training.py`:
  - `_ppo_step_v6`: `explained_variance` now computed only over samples where `v_weight > 0`. When `gae_vloss_weight=0`, EV measures against rollout targets only (previously measured against all targets including GAE, which the value head wasn't trained on). Returns `n_ev` in metrics dict.
  - `ppo_update_v6`: accumulates EV using `total_n_ev` instead of `total_n`.

## Key Findings

### Canonical states (v7_8, iter 181)
- 5/5 correct dominant action type on canonical scenarios. Basic strategy is learned.
- Random states: P(play)=0.65, P(scout)=0.19, P(sns)=0.16 — sensible distribution.

### Attention probe (v7_8, iter 181)
- Attention is 97% of max entropy — nearly uniform. Two heads have 0.957 cosine similarity (no specialization). Attention layer is not doing useful work at this stage.

### KL sensitivity test (v7_9, iter 176)
- At base LR (3e-4), one optimizer step produces KL=0.506 — **33.7x the kl_target of 0.015**.
- At 0.1x LR (3e-5), KL=0.017 — right at target.
- Confirms: the LR is ~10x too high for the current entropy level.

### v7_9 training dynamics (175 iterations)
- Config: same as v7_8 but `gae_vloss_weight: 0.0`
- `entropy_scout`: 2.69 → 0.98 (collapsed below play entropy of 1.05)
- `kl_batch_frac`: 0.94 → 0.29 (only ~3 of ~10 mini-batches run before KL early stopping)
- `approx_kl`: 0.007 → 0.190 (erratic, spiking)
- Eval margins: improved iter 15→55, then regressed through 175
- `explained_variance` was misleadingly low (0.07) because it measured against GAE targets the value head wasn't trained on. Fixed above.

## Decisions

- `explained_variance` fix uses `v_weight > 0` mask rather than checking `gae_vloss_weight` config value, so it generalizes to any weighting scheme.
- Attention probe replays the attention computation manually rather than modifying network.py, keeping the diagnostic self-contained.

## Next Steps

The identified fix has two components:
1. **Adaptive LR** — after each iteration, if observed KL > 1.5× kl_target, halve LR; if KL < 0.5× kl_target, double LR (capped). Standard PPO approach. This prevents the LR from being inappropriate for the current entropy regime.
2. **Possibly entropy floors** — the infrastructure exists (`entropy_floors` param, `entropy_floor_coeff`). Whether it's needed depends on whether adaptive LR alone prevents the scout entropy collapse. Proposal: implement adaptive LR first, run it, and add floors only if scout entropy still collapses.

Open question: does entropy collapse because the LR is too aggressive (in which case adaptive LR alone fixes it), or is there an independent tendency for scout entropy to collapse (in which case floors are also needed)?

## Watch Out

- `augment_rotation_v6` returns a 3-tuple now (from handoff 067). Any new scripts that call it must unpack 3 values.
- `kl_sensitivity_test.py` needs CUDA for rollout games — the net must be on GPU before game generation.
- Windows cp1252 encoding doesn't support unicode arrows — use ASCII in print statements.
