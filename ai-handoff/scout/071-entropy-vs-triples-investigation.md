# Entropy vs Triples Investigation

## Task & State

Investigated why the play_length probe (v7_play_len_probe) shows declining triples even under direct per-step reward. Ran a second probe (v7_play_len_probe_2) with mean-centered advantages (no learned value baseline) to test whether the value head's noise was the cause. It wasn't — the second probe was worse.

Both probes are stopped. v7_12 is still running (unchanged, ~455 iters on probe_1's summary window).

## What Changed

### Modified files
- `scout-bot/main.py`: Added `value_baseline` PARAM support. When `value_baseline="mean"`, GAE is bypassed and advantages are computed as `reward - batch_mean(reward)`. Also sets `value_loss_coeff=0` to stop training the value head. Changes are in the temp probe override block (lines 134-135) and the v6 training loop (lines 907-920).

### New files
- `scout-bot/bots/v7_play_len_probe_2/` — Fresh-start probe with `value_baseline="mean"`, `value_loss_coeff=0.0`, otherwise same as probe_1. 385 iterations.

## Key Findings

### Probe_2 results (mean-centered advantages, fresh start)
- Triples initially climbed: 1.7% → 2.5% over ~180 iters (positive advantage signal works)
- Then catastrophic KL spikes (approx_kl = 640, then 5.9 billion) destroyed triple behavior → collapsed to 0.0%
- avg_play_length DECLINED from 1.65 → 1.60 (opposite of intended direction)
- Pairs decreased (61.8% → 58.3%), singles increased (36.6% → 39.3%) in early iters
- Play entropy increased (0.87 → 0.92) — distribution moved toward uniform over legal actions
- Training much less stable without value loss: massive policy_loss spikes (113, then 1 billion)

### Comparison: probe_1 (learned V baseline) vs probe_2 (mean-centered)
- Probe_1 resumed from trained checkpoint, probe_2 started fresh — different starting conditions
- Probe_1: pairs increase (59.7% → 62.4%), triples decline slowly (4.6% → 3.1%), stable training
- Probe_2: pairs decrease, singles increase, triples collapse after KL spikes, unstable training
- Probe_1's play entropy is much lower (0.64 → 0.54) vs probe_2 (0.87 → 0.86) — probe_1's distribution is more peaked

### Entropy bonus analysis
- `entropy_bonus=0.05` is 5-10x the magnitude of the policy loss
- Entropy maximization pushes toward uniform distribution over legal actions
- Since triples have ~1.1 legal actions vs ~4.4 singles and ~3.3 pairs, the entropy-maximizing play distribution is ~56% singles, 42% pairs, 1.7% triples
- Probe_2's distribution was moving toward this entropy-maximizing distribution
- The learned value baseline in probe_1 partially counteracts this: in states where the network plays pairs, V(s) ≈ 0.4, so single advantage = 0.2 - 0.4 = -0.2 (negative, pushes singles down). Mean-centering gives single advantage = 0.2 - 0.177 = +0.023 (always positive, never counteracts entropy push)

### Value head removal findings
- Removing value loss (vloss=0) made training much less stable — the value loss was providing implicit regularization on trunk representations
- The value head (corr=0.53) isn't preventing triple learning — it's providing stabilization and state-dependent negative advantages for suboptimal actions

## Decisions

- Added `value_baseline="mean"` as a PARAM option rather than a permanent change. Easy to revert by removing the probe override lines.
- Probe_2 started fresh rather than resuming from probe_1's checkpoint, to isolate the effect of the advantage computation from the starting checkpoint.

## Next Steps

The leading hypothesis is that `entropy_bonus=0.05` is the primary force eroding triples. The entropy gradient pushes toward uniform-over-legal-actions, which structurally disadvantages longer plays (fewer legal action indices). The direct test: run the play_length probe with `entropy_bonus=0` or very small (0.005). If triples sustain or grow, entropy bonus is confirmed as the cause.

This has implications for the main v7_12 training too: the entropy bonus may be preventing the network from learning to play longer combinations in the game_score reward setting.

## Watch Out

- main.py PARAMS currently have the probe overrides active (lines 129-136): `rollout_fraction=0.0`, `reward_mode="play_length"`, `gae_vloss_weight=1.0`, `gamma=0.0`, `value_baseline="mean"`, `value_loss_coeff=0.0`, `save_dir="bots/v7_play_len_probe"`. These must be reverted/changed to run normal training or a different probe.
- The `value_baseline="mean"` code path only affects the GAE branch (rollout_fraction < 1.0). If rollout_fraction=1.0, advantages come from rollouts and this setting has no effect.
- Probe_2 demonstrated that removing value loss causes severe training instability (KL explosions). Any future probe without value loss should use lower learning rate or fewer PPO epochs.
