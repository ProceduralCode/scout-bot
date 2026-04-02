# Q-Network Direction

## Task & State

Investigated probe_3 results (entropy_bonus=0, play_length reward), confirmed entropy bonus was suppressing triples. Then ran additional probes that revealed deeper problems: same hyperparameters, different seeds → completely different outcomes (one learned triples to 9%, another actively learned to suppress them). Discussed the implications and arrived at a decision to move away from PPO toward a Q-network (value-based) approach.

## Key Findings

### Probe_3 confirmed entropy bonus hypothesis
- entropy_bonus=0 with play_length reward: triples climbed from 1% to 9.45% peak (vs declining under entropy_bonus=0.05)
- Entropy collapsed to 0.33 overall, entropy_scout to ~0.006
- Late-stage mild erosion: triples 9.45% → 8.58%, singles climbing back

### Subsequent probes revealed seed sensitivity
- entropy_bonus=0, sampling_temperature=1.5: triples didn't grow, pairs suppressed
- entropy_bonus=0, sampling_temperature=1.0 (same as probe_3): actively learned NOT to play triples — opposite of probe_3
- Same hyperparameters, different random initialization → opposite learning outcomes
- This is under play_length reward (gamma=0, immediate per-step reward) — the clearest possible signal

### Entropy bonus mechanism confirmed
- entropy_bonus=0.05 was 5-10x the magnitude of policy loss
- Entropy maximization pushes toward uniform-over-legal-actions
- Since triples have ~1.1 legal actions vs ~4.4 singles, the entropy-maximizing distribution is ~56% singles, 42% pairs, 1.7% triples
- Probe_2 (mean baseline) distribution moved toward exactly this entropy-maximizing allocation

### KL spikes observed in entropy_bonus=0 runs
- Low-entropy (peaked) policies cause huge probability ratio swings from small weight changes
- approx_kl and policy_loss spiked to 100x+ normal values
- Adaptive LR responds slowly (0.9x per iteration) — can't recover from sudden 100x spikes

## Decisions

### Moving to Q-network approach
PPO's fundamental limitation: the training signal is indirect ("that went well, do it more often") rather than direct ("that action is worth +3.2 margin"). With rollout infrastructure already in place, a Q-network can:
- Train via regression (MSE toward rollout outcomes) — stabler than policy gradient
- Learn values for actions the policy didn't choose (by deliberately rolling out alternative actions)
- Decouple exploration from the loss function entirely (epsilon-greedy at action selection, no entropy bonus)

### Not pursuing entropy fixes
Considered entropy floor, adaptive entropy coefficient, normalized entropy, and sampling temperature. Decided these are patches on a fundamentally noisy optimization method. The seed sensitivity under play_length reward (the easiest possible learning task) suggests the problem isn't just entropy — it's policy gradient noise.

## Next Steps

### Q-network implementation
The main changes needed:
- **Output head**: reinterpret 384 outputs as Q-values (expected margin) instead of action logits
- **Loss function**: MSE regression toward rollout margin outcomes, replacing PPO surrogate
- **Action selection**: argmax Q (or softmax with temperature) during game play, with epsilon for exploration
- **Rollout pipeline**: evaluate multiple actions per state (not just the one taken). Approach: rollout top-K predicted actions + a few random ones per state. Subsample legal actions (10-12 per state, not all 50).
- **Trunk, encoding, game engine, Numba rollout infrastructure**: unchanged

### Exploration strategy for Q-network
- Pick top 5-10 predicted actions + 2-3 random legal actions for rollout per state
- Random actions serve as verification — prevent the network from never discovering it's wrong about an action type
- Total rollout cost: ~3-4x current (currently 1 action per state)

## Watch Out

- main.py PARAMS still have probe overrides active (lines 129-136). Must be reverted before any new training.
- The old ScoutNetwork had "sub-head action decomposition" — worth understanding why it was abandoned before the Q-network redesign, since Q-values per action type is conceptually similar.
- probe_3's value_corr went to -0.43 (anti-correlated). Under play_length reward with gamma=0, the value head was actively anti-predicting. This may indicate trunk representation issues that would also affect a Q-network using the same trunk.
