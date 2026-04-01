# Play Length Probe & Legal Play Diagnostics

## Task & State

Investigated why v7_12 (rollout_fraction=1.0, adaptive LR) is improving eval margins but not diversifying play composition. Ran a play_length reward probe and legal play availability analysis to understand the constraints.

v7_12 is still running (1110+ iterations), stable with no regression. KL target was raised from 0.005 to 0.01 mid-run (~iter 500), which accelerated eval improvement. The play_length probe (v7_play_len_probe) ran 290 iterations and is stopped.

## What Changed

### Modified files
- `scout-bot/main.py`: summary.txt now samples 10 evenly-spaced points instead of 5 (line 582, `min(n, 10)`).

### New files
- `scout-bot/legal_play_stats.py` — Plays random games, reports what fraction of turns have legal plays of each length. Key finding: triples legal on 13.8% of turns, quads on 1.1%.
- `scout-bot/triple_probe.py` — Loads a checkpoint, plays random games, reports the policy's probability mass on triples when triples are legal. Uses iter_1104.pt from v7_12.
- `scout-bot/bots/v7_play_len_probe/` — Probe run with `reward_mode="play_length"`, `rollout_fraction=0.0`, `gae_vloss_weight=1.0`, `gamma=0.0`.

## Key Findings

### Legal play availability (legal_play_stats.py, 10K random games)
- Singles legal: 53.5% of turns
- Pairs legal: 72.6%
- Triples legal: 13.8%
- Quads legal: 1.1%
- When triples are legal: avg 1.1 triple options vs 4.4 singles, 3.3 pairs
- Random uniform policy would play triples ~1.7% of the time

### v7_12 triple choice behavior (triple_probe.py, iter 1104)
- When triples are legal, P(triple+) = 0.31, P(single) = 0.40, P(pair) = 0.29
- Argmax is triple: 49.9%, single: 46.5%, pair: 3.6%, scout: 0.0%
- When not choosing triple: 53% had a current play to beat, 47% were openings
- The network does prefer triples over random, but picks singles ~half the time even when triples are available — including opening positions where triples are almost strictly better

### Play length probe (v7_play_len_probe, 290 iterations)
- Reward: play_length/5.0 per step, gamma=0 (no bootstrapping)
- Rapidly shifted singles 68%→35%, pairs 31%→62% in ~20 iterations
- Triples peaked at ~4.8% around iteration 30-40 then declined to ~2.9% by iteration 290
- avg_play_length plateaued at ~1.67 and did not continue increasing
- The network can learn to prefer pairs over singles with direct reward, but triples decline even under direct incentive — this is unexplained

### v7_12 training state at iteration 1110
- eval_margin_v1_4: +3.45, v2_5: -0.16, v3_4: -8.60, v4_2: -16.67
- Value head stuck: EV 0.587, MAE 0.209, correlation 0.763 — no movement across 1100 iterations
- Entropy declining: 1.07 overall, play entropy 0.59
- KL target raised to 0.01 around iter 500, approx_kl now ~0.011

## Decisions

- KL target raised from 0.005 to 0.01 mid-run. Eval improvement accelerated without destabilization.
- summary.txt changed from 5 to 10 sample points for better resolution.
- Play length probe used gamma=0 to eliminate value head dependency. Per-step reward means bootstrapping isn't needed.

## Next Steps

The play_length probe's declining triples under direct per-step reward is the most concerning finding. The network can rapidly learn "pairs > singles" but actively regresses on triples even when rewarded for them. Understanding why is the priority — possible directions:
- Check if the action masking or encoding is correct for triple plays (do masked probabilities match actual legal triples?)
- Check if there's a systematic bias in how triple actions are encoded in the flat 384 space
- Run the triple_probe.py on the play_len_probe checkpoint to see if that network also shows the same 50% triple preference

## Watch Out

- v7_12 PARAMS currently has `kl_target=0.01` (was 0.005 at start of run). The adaptive LR adjusted to this mid-run.
- The play_len_probe PARAMS differ from v7_12: `rollout_fraction=0.0`, `reward_mode="play_length"`, `gae_vloss_weight=1.0`, `gamma=0.0`. These need to be reverted to run normal training.
- `legal_play_stats.py` and `triple_probe.py` use random play to advance games (not network policy), so states analyzed are not identical to states seen during training.
- The existing diagnostics in main.py (`_compute_diagnostics`, line 668) compute play length as `(a % 16) - (a // 16) + 1` which doesn't handle circular slot wrapping. The triple_probe.py uses `(e_slot - s_slot) % H + 1` which does. The main.py diagnostic P(3+) values on charts may be incorrect for wrapped actions.
