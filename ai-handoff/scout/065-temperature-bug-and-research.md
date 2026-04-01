# Temperature Bug Fix & Training Pipeline Research

## Task & State

Investigated rising clip fraction, proxKL, and declining eval margins in early training. Found and fixed a concrete bug in how `sampling_temperature` interacted with `old_log_prob` recording. Conducted extensive research on PPO debugging, pipeline verification, and hyperparameter tuning.

## What Changed

### Modified files
- `scout-bot/training.py`:
  - `play_games_v6` (~line 1743): temperature now applied to a separate `sample_logits` variable for sampling; `old_lps` computed from untempered `logits`. Previously, temperature was applied to `logits` in-place, so `old_log_prob` was recorded at T=temperature while PPO computed `new_log_prob` at T=1.
  - `play_games_with_rollouts_v6` (~line 1899): `old_lp` now computed from `logits` instead of `sample_logits`. Same fix as above.

### The bug mechanism
`augment_rotation_v6` creates 15 augmented copies (shifts 1-15) with fresh `old_log_probs` at T=1 via forward passes. But shift 0 (originals, 1/16 of batch) kept their tempered `old_log_probs`. During PPO, `new_log_prob` is always at T=1. The mismatch created a systematic KL floor:
- At T=1.5: ~0.003 KL contribution (within budget, barely noticeable)
- At T=2.5: ~0.016 KL contribution (right at kl_target=0.015, caused aggressive early stopping)

This explains why v7_6 (T=2.5) had kl_batch_frac as low as 0.148 — the temperature mismatch alone consumed the entire KL budget, causing the training to throw away 85% of each batch.

## Research Findings

Extensive research on PPO debugging, pipeline verification, and hyperparameter tuning. Key sources: Costa Huang's "37 Implementation Details of PPO," Andrychowicz et al.'s "What Matters In On-Policy RL," Andy Jones's "Debugging RL Systems," Schulman's "Nuts and Bolts of Deep RL," the CleanRL project, the Engstrom "Implementation Matters" paper, and a Big 2 card game PPO self-play project (very similar domain).

### Pipeline verification (recommended checks)
1. **Ratio=1.0 assertion** — at first mini-batch of first epoch, before any gradient step, all ratios must be exactly 1.0. This is missing from the v6 path (exists for old path at main.py:984). Would have caught the temperature bug instantly.
2. **Augmentation symmetry check** — for each rotation, verify `inv_perm(pi(T(s))) ≈ pi(s)` on a trained network.
3. **Initial-state value check** — value prediction for fresh game should be ~0 (no player has inherent advantage in self-play).
4. **Probe environments** — graduated complexity: test value function alone → backprop → discounting → policy learning → full system. Each should converge in seconds.

### Hyperparameter findings
- `ppo_epochs=1` is very conservative. A Big 2 card game PPO self-play project (4-player, imperfect info) uses identical params except `ppo_epochs=4`. With KL early stopping as safety valve, increasing to 3-4 should improve sample efficiency.
- Temperature + entropy bonus is double exploration pressure. Research suggests picking one. Entropy bonus is better integrated with the loss function.
- Entropy annealing (start high, decay to ~50%) outperforms fixed entropy.
- GAE lambda should go up with poor value functions. Detached value head argues for lambda=0.97-0.99.
- Schulman: "After fixing a bug, re-evaluate all hyperparameters tuned while the bug was present."

## Next Steps

1. Add ratio=1.0 assertion to `_ppo_step_v6` (prevents this class of bug from recurring)
2. Run the augmentation symmetry check and initial-state value check
3. Reset hyperparameters post-bugfix: reduce temperature to 1.0, increase ppo_epochs to 3, try gae_lambda=0.98, add entropy annealing
4. Run controlled experiments (~200 iterations each) to find good params on the fixed code
5. Long training run once pipeline is verified and params are tuned

## Watch Out

- The context.md comment about temperature ("recorded in old_log_prob so PPO ratios are correct") is now wrong — the fix intentionally breaks that invariant. Should be updated to reflect the new approach.
- All prior training runs (v7_5, v7_6) were affected by this bug. v7_5 at T=1.5 was mildly affected; v7_6 at T=2.5 was severely affected. Hyperparameters tuned under these conditions may need revisiting.
- v7_5 100-iter data showed eval margins still improving (v1_4: -12.9, best yet) despite the bug. The bug was a drag, not a blocker, at T=1.5.
