# Value Head Trunk Analysis

## Task & State

Investigated why the value head produces magnitude-compressed predictions. Started from session 048's finding (OLS EV=0.66 vs value head EV=0.41) and traced the root cause to trunk feature collapse. No changes to the training pipeline or network architecture.

## What Changed

- `scout-bot/value_warmup_test.py` — rewritten with train/test split and multi-variant comparison (Adam+PPO weights, Adam+fresh, SGD+PPO, SGD+fresh). 500 epochs, 40 games, 30% test fraction.

Charts saved to `scout-bot/v6_5/latest_value_warmup.png`.

## Measurements (v6_5, iteration 224)

### OLS train/test validation (40 games, 1753 decisions, 70/30 split)

- OLS train EV: 0.607, test EV: 0.516 (9.5 samples/param). OLS generalizes — the trunk features genuinely contain value signal.
- Adam + PPO weights: train EV 0.40, test EV 0.39 (500 epochs, lr=0.001). Stuck at starting point.
- Adam + fresh weights: train EV 0.03, test EV 0.03. Worse than PPO init.
- SGD (both inits): diverged to NaN at lr=0.01.

### Value head direction analysis (10 games, 408 decisions)

- Cosine similarity between PPO value head weights and OLS optimal weights: **-0.054** (orthogonal).
- Optimal scalar on PPO direction: 1.5x, reaching EV=0.49. Not a scaling problem — it's a direction problem.
- OLS weight norm: 33.9 vs PPO weight norm: 0.62.

### Trunk feature analysis

PCA on final-layer activations (128-dim, 408 samples):

- PC1 explains 99.96% of activation variance. Singular values: [299, 5.5, 1.3, 0.8, ...]
- Effective rank at 90% variance: 1
- 54/128 dimensions always zero (dead ReLU neurons)
- PC1 correlation with value target: -0.26 (EV=0.069). The dominant axis is uninformative for value.
- OLS on full 128D: EV=0.71. Value signal is spread across the residual ~0.04% of variance.

### Per-layer rank progression

| Stage | PC1 var | Eff rank (99%) | Mean act. |
|-------|---------|----------------|-----------|
| Raw encoding (309D) | 14.3% | 68 | 0.14 |
| Attention output (689D) | 35.3% | 46 | 0.03 |
| After Linear(689→512)+ReLU | 95.8% | 25 | 3.81 |
| After Linear(512→256)+ReLU | 99.8% | 1 | 9.91 |
| After Linear(256→128)+ReLU | 99.96% | 1 | 15.33 |

The collapse happens primarily at the first Linear+ReLU layer. The first layer's weight matrix has top singular value 4.7 vs ~1.0 for others — it amplifies the input's dominant direction. The weight matrices themselves have high effective rank (286, 167, 85) and small biases (<0.1). The rank collapse comes from the interaction of weights with data, not from parameter degeneracy.

### Gradient dynamics (20 epochs, Adam lr=0.001)

- Weight norm: 0.617 → 0.618 (barely moved)
- Cosine sim between initial and final weights: 0.99995 (direction unchanged)
- Loss oscillates: 0.12 → 3.64 → 0.34 → 0.89 → 2.13...
- Adam m2 accumulator grows to 661, freezing effective updates

Adam on 1D (PC1 only) converges perfectly to OLS-matching EV=0.069 in 100 epochs. The problem is specific to the high-dimensional ill-conditioned setting.

## Open Questions

- Is the trunk collapse an inevitable consequence of stacking unnormalized Linear+ReLU in a shared actor-critic network, or specific to this training run? The first layer's amplified singular value (4.7) suggests training shaped it, but the basic DC accumulation from ReLU stacking is structural.
- The policy head (128→384, weight norm 11.36) appears to work fine reading these same features — it has enough output neurons to extract diverse projections from tiny residual variations. What's the actual policy head's effective use of the feature space?

## Next Steps

1. Decide on architectural approach to address trunk feature conditioning for the value head. Options discussed: normalization before value head, separate value trunk branching earlier, periodic OLS reset of value head weights.
2. Probes 10-11 fix and PPO direction probe still pending from session 047.
