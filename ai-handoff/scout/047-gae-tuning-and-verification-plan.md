# GAE Tuning and Verification Plan

## Task & State

Switched v6 from rollouts to GAE to address data starvation (see 046). Ran three GAE experiments with different hyperparameters. v6_5 is the best result — slowly improving but entropy collapsing. Session ended with a plan for bottom-up verification of the training pipeline rather than continued hyperparameter search.

## What Changed

No code changes. PARAMS in `scout-bot/main.py` were edited between runs (LR, entropy_bonus, save_dir). Added "Checkpoints and Summaries" section to context.md.

## Experiments Run

- **v6_4** — LR 0.0003, entropy_bonus 0.05, games_per_iteration 100, use_rollouts False. Fast initial learning (v1 eval -16 → -3.3 by iter 65), then regression to -4.3. KL 0.06-0.14 (very high), entropy collapsed to 0.97 by iter 63. 125 iterations.

- **v6_5** — LR 0.0001, entropy_bonus 0.05. Slower but steadier: v1 eval -29 → -3.1 at iter 220, v3 eval improving to -12.2, v4 improving to -22.3. KL settled to 0.03 (healthier). Entropy still collapsing: 0.85 at iter 220. ~220 iterations, likely still running.

- **v6_6** — LR 0.0001, entropy_bonus 0.25. Entropy stayed at ~2.65 (barely moved), policy couldn't learn. Eval stuck at -38 to -42 vs v1. The bonus completely dominated the policy loss. ~145 iterations.

All three runs: explained_variance stuck at 0.21-0.23 regardless of configuration.

## Decisions

- **Switched from rollouts to GAE** for v6 training. Set `use_rollouts: False`, `games_per_iteration: 100`. GAE uses `play_games_v6()` → `compute_gae()` → `augment_rotation_v6()` → `prepare_ppo_batch_v6()` → `ppo_update_v6()`.

- **Stopped chasing entropy_bonus values.** 0.05 collapses, 0.25 prevents learning. Something between might work but this is blind search. Shifted to verification-first approach.

## Next Steps — Bottom-Up Verification Plan

The policy learns from GAE (v6_5 eval scores improve) but training is fragile (entropy collapse, oscillation). Rather than more hyperparameter search, verify each pipeline component independently:

1. **Value head capacity probe** — Generate games with a fixed policy, compute actual discounted returns, train only the value head on clean supervised targets. If explained_variance stays at ~0.22, the issue is architectural or inherent game noise. If it reaches 0.4+, the issue is training dynamics. This determines whether the stuck explained_variance is a problem or expected for a 4-player hidden-info game with single-game GAE targets.

2. **GAE correctness check** — Take a small batch of completed games, manually compute what advantages should be, compare to `compute_gae()` output. An off-by-one, wrong discount, or normalization bug would be a silent killer that no hyperparameter change can fix. The code looked correct on inspection but hasn't been formally verified for the v6 path.

3. **Fix probes 10-11** — Both need rotation augmentation added to the training loop (currently sample random hand_offset but don't augment). Probe 11 also needs "any valid adjacent position" matching instead of exact-match, since multiple positions can be equally correct.

4. **PPO direction probe** — Given known-correct advantages, verify a PPO step moves the policy toward the reinforced actions. Existing probe 4 partially covers this but could be strengthened.

## Watch Out

- v6_5 may still be running. Check before starting a new run in the same save_dir.
- `compute_gae()` was written for `StepRecord` but works with `StepRecordV6` via duck typing (both have game_id, round_num, player, reward, value). Verified the fields match.
- Explained_variance 0.22 with GAE vs 0.41-0.51 with rollouts may be explained by target noise difference (rollout targets are averages of many games, GAE targets are single-game returns), not value head quality difference.
