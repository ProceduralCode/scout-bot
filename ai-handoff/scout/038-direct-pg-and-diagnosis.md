# 038 — Direct PG Experiment and Advantage Diagnosis

## Task

Wired in a `use_direct_pg` flag for vanilla policy gradient (no importance sampling, no clipping, forced 1 epoch). Ran v5_2 (direct PG + 50 rollouts, lr=0.001) and v5_3 (PPO + 50 rollouts, lr=0.001, resumed from checkpoint). Diagnosed why both plateau.

## What Changed

- `scout-bot/training.py` — added `direct_pg_update()` function (line ~1245). Same structure as `ppo_update()` but uses `-(log_prob * adv).mean()` instead of clipped surrogate. Also added per-action-type advantage diagnostic logging (ADV-DIAG lines) in `play_games_with_rollouts()` around line 391.
- `scout-bot/main.py` — added `use_direct_pg` param (default False), import of `direct_pg_update`, dispatch logic (~line 459) that overrides ppo_epochs to 1 when active.

## Results

### v5_2 (direct PG, 50 rollouts, lr=0.001, ~600 iters)
- Entropy collapsed to 0.04 by iter ~60, then slowly recovered to 0.1-0.17 by oscillating
- Eval at iter 605: v1_4=+1.16, v2_5=-2.89, v3_4=-11.07, v4_2=-18.73
- Play distribution stuck at ~52/45/3 (1/2/3-card plays)

### v5_3 (PPO, 50 rollouts, lr=0.001, resumed from checkpoint, ~335 iters)
- Smooth entropy decline from 1.19 to 0.07 (no collapse/recovery oscillation)
- Started ahead due to checkpoint but didn't improve much from starting level
- Eval at iter 335: v1_4=-0.53, v2_5=-2.21, v3_4=-12.45, v4_2=-19.38
- Play distribution similar: ~50/46/3-4

### Key finding: both plateau at the same level
Different update mechanics, same ceiling. PPO vs direct PG is not the bottleneck.

## Advantage Diagnostic Results

Ran 3 iterations with per-type advantage logging:
```
PLAY:  n=842  mean=-0.008  std=0.213  range=[-0.66, +0.85]
SCOUT: n=40   mean=+0.029  std=0.209  range=[-0.48, +0.38]
```

Scout and play advantage magnitudes are similar (~0.21 std both). The hypothesis that scout advantages are systematically smaller was wrong.

Two observations:
1. Only ~40 scout steps per iteration vs ~840 play steps (~21x fewer gradient updates for scout head)
2. Each scout step's advantage conflates "was scouting good timing" with "was this insert position good" — the scout_insert head can't distinguish them

## Decisions

- `direct_pg_update` was implemented and tested but doesn't solve the core problem. Confirmed by both runs plateauing at same level. The function remains in training.py as an available option.
- PPO + 50 rollouts is strictly better than direct PG + 50 rollouts (same ceiling, better stability).

## Open Thread

Two candidate fixes discussed but not implemented:

1. **More PPO epochs** (16-32 instead of 4) — v5_3 clip_fraction is 0.007 at iter 335, meaning 99.3% of steps still provide gradient after 4 epochs. Massive headroom to extract more learning per expensive iteration. Addresses the "40 scout steps" data scarcity.

2. **Multi-position advantage for scout_insert** — for each scout decision, roll out all possible insert positions, use `V_after(chosen) - mean(V_after(all positions))` as the scout_insert advantage. Directly isolates position quality from scouting timing. Cost: ~50% more rollouts per iteration (~20k extra on base of ~40k). Concern raised: when multiple positions are equally good, signal is weak — though this is correct behavior (nothing to learn when positions are equivalent).

User also asked about a replay buffer (keeping last N iterations of data). PPO handles slightly stale data via importance sampling — old data where policy hasn't changed much still provides gradient, while heavily drifted steps get clipped to zero automatically. More PPO epochs is the simplest version of this idea.

## Watch Out

- The ADV-DIAG print statements are still in `play_games_with_rollouts()` (~line 391). They print every iteration. Remove or gate behind a flag if noisy.
- `PARAMS["use_direct_pg"]` is currently `True` in main.py. Set to `False` for PPO mode.
