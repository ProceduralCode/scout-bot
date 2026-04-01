# Dead Neuron Investigation

## Task & State

Investigated why v6_8 performs poorly (-12.47 vs v1_4 at iter 460). Discovered 90/128 neurons dead in the final trunk layer despite LayerNorm. Traced the mechanism to LayerNorm affine parameters stuck at initialization. Added dead neuron tracking to charts.png. No training changes made — this was purely diagnostic.

## What Changed

- `scout-bot/main.py` — Added `_count_dead_neurons()` function that hooks ReLU layers and counts per-layer dead neurons. Added `dead_neurons_total`, `dead_neurons_layer_0/1/2` to metrics_history. Replaced "Value Prediction" chart panel (row 3, col 0) with "Dead Neurons" panel showing per-layer and total counts. Dead neuron count also printed in console log line.

- `ai-handoff/scout/context.md` — Added "Diagnosis Before Treatment" section (empirical testing mandate). Added "V6 Diagnostic History" section summarizing sessions 044-053.

## Measurements

### Dead neuron counts

| Checkpoint | Layer 0 (512) | Layer 1 (256) | Layer 2 (128) | Total |
|---|---|---|---|---|
| v6_7 iter 1158 | 0 | 0 | 0 | 0/896 |
| v6_8 iter 394 | 0 | 53 | 90 | 143/896 |
| v6_8 iter 463 | 0 | 58 | 90 | 148/896 |

Layer 2 (which feeds both policy and value heads) saturated at 90/128 dead by iter 394. Layer 1 is still slowly dying (53→58). Layer 0 is unaffected.

### LayerNorm parameter analysis (v6_8 layer 2)

Dead neurons (90): gamma=1.00, beta=0.00 (initialization values). LN output always negative (max=-0.11).
Alive neurons (38): gamma=1.37, beta=0.37 (learned positive shift). LN output mean=2.45.

Mechanism: Once ReLU output is zero, gradient through that neuron is zero, so its LayerNorm gamma/beta receive no gradient and stay at initialization. Without the positive shift that alive neurons learned, normalized values land negative and stay dead. Chicken-and-egg trap.

### Value head probe (v6_8 iter 463)

- Correlation: 0.858 (up from v6_5's 0.73)
- EV: 0.522 (up from v6_5's 0.35)
- Predicted std: 0.196 vs empirical std: 0.495 (still 2.5x magnitude-compressed)
- Training-reported EV is 0.25 — the gap to 0.52 is likely GAE target noise vs clean rollout targets.

### Trunk analysis (v6_8 iter 463)

- PC1: 59.84% variance (v6_7 was 18.65% at iter 102)
- PC1 dominated by `has_current_play` (r=-0.921)
- Policy/value gradient ratio: 1.9x
- 90/128 dead neurons

### Architecture probes (fresh 512/256/128 network)

Probes 2,3,4,8,11: all PASS. Probe 10: FAIL (known data-starvation issue).

## Next Steps

1. **Determine when neurons die.** No early v6_8 checkpoints exist. Options: (a) start a short fresh training run with dead neuron logging to observe the onset, (b) check fresh random init to see if some neurons are predisposed at initialization.

2. **Determine what hyperparameter difference causes dying in v6_8 but not v6_7.** Key differences: entropy floors (v6_8 has 1.0/1.0, v6_7 none), mini_batch_size (32768 vs 8192), ppo_epochs (1 vs 2→4), entropy_bonus (0.03 vs 0.05), replay (off vs on). Could test by running v6_7's hyperparams on a fresh network and checking dead neurons.

3. **Assess whether dead neurons are the binding constraint on performance.** v6_8 has 38 alive neurons in layer 2 and reached -12.47. v6_7 had 128 alive and reached -1.14. The value head probe shows 0.52 EV despite dead neurons. Need to determine if fixing dead neurons alone would substantially improve performance.

## Watch Out

- `_count_dead_neurons()` calls `network.eval()` but doesn't restore to `network.train()`. This is fine because it's called during the logging block where the network is about to be set to train mode for the next iteration anyway.

- The dead neuron check uses the `training_batch["states"]` tensor (first 2000 samples). If a batch is unusually small or has unusual state distribution, counts could vary. In practice, batches are 64K+ records so this is not a concern.

- The dead neuron chart replaces the old "Value Prediction" panel. The "value" metric is still tracked in metrics_history but no longer charted.
