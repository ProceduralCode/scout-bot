# Scout Bot: NN Debugging & Diagnostic Research

## Context

Training plateaus around ~50 iterations. The agent quickly learns to prefer playing over scouting, but then gets stuck — plays are mostly 2 cards or less, and it doesn't learn to build sets/runs through scouting. Something is preventing further progress.

## Current Diagnostics Infrastructure

Already in place (see `scout-diagnostics-and-tuning.md` for history):

- **charts.png** — 12 metrics across 4 rows (performance, loss/entropy, PPO health, behavior)
- **diagnostic.py** — checkpoint health checks: value analysis, GAE advantages, gradient flow, per-head entropy
- **Per-head entropy tracking** — action_type, play_start, play_end, scout_insert logged separately
- **PPO health** — clip fraction, approx KL, explained variance
- **Behavioral metrics** — play/scout/S&S percentages, steps per game, advantage std

## Research: Diagnostic Techniques Worth Adding

### 1. Probe Environments (Highest ROI for Finding Bugs)

Build minimal test scenarios that isolate one subsystem at a time:

- Single legal action, fixed state, +1 reward → value head learns a constant?
- Single action, random state, state-dependent reward → backprop through shared layers works?
- Two actions, action-dependent reward → policy head learns a preference?
- Small hand (3 cards), trivial play area → multi-step decomposition (type → start → end) chains correctly?

Each probe adds one axis of complexity. Failure in probe N+1 but not N pinpoints the broken component. Especially useful for verifying the conditioned head chain works correctly.

Source: [Andy Jones — Debugging RL Systems](https://andyljones.com/posts/rl-debugging.html)

### 2. Terminal / Penultimate Correlation

- **Terminal correlation**: correlation between the value prediction at the last step of a round and the actual round reward. Should rise from ~0 toward ~1.
- **Penultimate correlation**: same but one step before round end. If terminal is high but penultimate is low, advantages aren't propagating backward through the round.

Directly tests whether the value function is learning the game and whether the "last step only" reward assignment is limiting credit propagation.

### 3. Pre-Clip Gradient Norms

Currently grad clipping is at 0.5 (`clip_grad_norm_`), but the pre-clip norm isn't logged. If clipping fires constantly, it masks a problem. If it never fires, the clipping is irrelevant. Either way, worth knowing.

### 4. Per-Head Gradient Frequency

The conditioned heads only receive gradients from actions that use them:
- play_start/play_end only get gradients from play actions
- scout_insert only from scout/S&S actions

If the action distribution is heavily skewed (which it is — favoring play), some heads get very few gradient updates. Worth tracking the actual count of gradient updates per head per iteration.

### 5. Opponent Pool Diversity

If pool members all play similarly, training sees homogeneous opponents. Track behavioral fingerprints of pool members (play/scout/S&S ratios, average play length) to detect pool homogenization.

### 6. Reward Signal Analysis

- Reward distribution histogram — is it concentrated at a few values?
- Correlation between round length and reward variance — longer rounds = worse credit assignment?
- Per-action-type advantage breakdown — does the advantage signal actually differentiate play vs scout, or is it noise?

### 7. Counterfactual / Rollout Diagnostics

Run the same game state through the policy multiple times. If reward variance is high for identical states, the signal-to-noise ratio is poor and the agent needs more games per iteration to learn.

## Structural Observations From Code Review

These are things noticed while reading the codebase. They may or may not be causing the plateau — runtime data is needed to confirm.

### Per-Head Entropy Control

All heads share a single `entropy_bonus` coefficient (0.25). There's no mechanism to give more exploration pressure to heads that may need it (e.g., play_end or scout_insert) independently of heads that converge appropriately (e.g., action_type).

Potential fix: per-head entropy coefficients, entropy floor/minimum, or separate temperature scaling.

### Multi-Step Decomposition Credit

When the agent makes a play, the advantage signal is shared across the entire (action_type, start, end) decision. There's no mechanism to attribute credit specifically to the play-length choice (the end decision). A 2-card play and a 4-card play in similar states get similar advantages if the round outcome is similar.

### Reward Density

Reward is assigned only to the last step per player per round. All intermediate decisions get reward=0, relying on GAE to propagate signal backward. This is standard for episodic RL but makes it hard to learn that specific mid-round decisions (like a well-placed scout) matter.

### Base Rate of Legal Actions

In a random hand, there are more short legal plays than long ones (more 1-2 card combos than 4-5 card combos). Any head that matches the base rate of training data will naturally skew toward short plays. Overcoming this requires either strong reward signal for longer plays or explicit exploration pressure.

## Possible Intervention Directions

Not prioritized — which to pursue depends on what the diagnostics reveal.

- **Per-head entropy coefficients** — prevent individual head collapse without over-regularizing others
- **Entropy floor** — hard minimum on per-head entropy to prevent any head from going deterministic
- **Play-length reward shaping** — small dense bonus for longer plays (risky if overtuned)
- **Separate advantage streams** — decompose advantage per sub-decision in the multi-step chain
- **Flatten action space** — single head selecting from all legal (start, end) pairs instead of sequential start → end
- **Heuristic bootstrap / behavioral cloning** — skip the random phase by imitating a rule-based bot
- **Curriculum on play complexity** — start with simplified games where long plays are more common/necessary

## Key Sources

- [Andy Jones — Debugging RL Systems](https://andyljones.com/posts/rl-debugging.html) — best single resource, covers probe environments, metric interpretation, common bugs
- [CleanRL PPO Implementation](https://docs.cleanrl.dev/rl-algorithms/ppo/) — reference implementation with standard metric tracking
- [Troubleshooting PPO Instability](https://apxml.com/courses/rlhf-reinforcement-learning-human-feedback/chapter-4-rl-ppo-fine-tuning/troubleshooting-ppo-instability)
- [RLExplorer — Debugging Deep RL](https://arxiv.org/html/2410.04322v1) — academic treatment of systematic RL debugging

## Recommended Next Steps

1. **Look at the actual training data** — load a checkpoint's metrics_history and print real numbers for the key metrics (per-head entropy trajectories, advantage std, explained variance, clip fraction). Don't eyeball charts.
2. **Based on what the data shows**, decide whether to build new diagnostics (probe environments, terminal correlation, etc.) or go straight to an intervention.
3. **If intervening**, per-head entropy control is probably the lowest-risk first experiment — it's a small code change and directly testable.

## Files to Read First

- `scout-bot/main.py` — training loop, charts, checkpoint logic, DEFAULT_CONFIG
- `scout-bot/training.py` — PPO update (~lines 325-468), game play (~43-87), GAE (~257-292)
- `scout-bot/encoding.py` — action masking, legal play enumeration, hand encoding
- `scout-bot/network.py` — model architecture, conditioned heads, masked sampling
- `scout-bot/diagnostic.py` — existing checkpoint health checks
