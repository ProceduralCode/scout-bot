# GELU, KL Early Stopping, and Detached Value Head

## Task & State

Continued dead neuron investigation from session 054. Ran two ablation tests (fresh 100-iteration runs with v6_8 config, changing one variable each). Then implemented three architectural changes based on findings. **Changes are code-complete but not yet tested** — no training run has been started with the new architecture.

## What Changed

### Ablation results (no code changes — diagnostic runs only)

Baseline (all v6_8 params, fresh start): 193/896 dead at iter 100.

| Run | What changed from v6_8 | Dead at iter 100 | Delta |
|---|---|---|---|
| `v6_dead_test_no_zero_scout` | zero_scout_policy_grad=False | 295/896 | +102 |
| `v6_dead_test_no_floors` | entropy_floors=None | 282/896 | +89 |

Both ablations increased death. `zero_scout_policy_grad=True` and entropy floors are protective, not harmful.

### Code changes

- **`scout-bot/network.py`** — Three changes to FlatScoutNetwork:
  1. ReLU → GELU in trunk (line 256)
  2. Value head replaced: was `nn.Linear(128, 1)`, now 2-layer MLP (945→256→1) that takes detached multi-scale features (49-dim global metadata + 512+256+128 trunk layer activations, all `.detach()`ed)
  3. `forward()` returns `(hidden, value_ctx)` tuple instead of bare tensor. `value()` and `policy_logits()` unpack the tuple. No call-site changes needed — callers pass the opaque result through.

- **`scout-bot/training.py`** — KL early stopping in `ppo_update_v6()`: tracks running approx KL across mini-batches, breaks if it exceeds `kl_target` (default 0.015). New param `kl_target`. Returns `kl_batches_used`/`kl_batches_total` in metrics.

- **`scout-bot/main.py`** — Multiple changes:
  - Dead neuron hook updated: detects GELU (not just ReLU), threshold `<= 0` instead of `== 0`
  - KL early stopping also in the epoch loop (breaks across epochs when KL exceeds target)
  - `kl_target: 0.015` added to PARAMS
  - Eval runs at iteration 0 (pre-training) on fresh starts via `_run_eval()` helper
  - Chart layout reorganized: removed Avg Reward and Reward Std panels, moved Policy Loss and Value Loss to row 1, added KL Early Stop (batch fraction) panel, KL chart shows target threshold line
  - PARAMS currently set for entropy floors ablation (`save_dir: "v6_dead_test_no_floors"`, `entropy_floors: None`, `zero_scout_policy_grad: True`). **Needs to be reset for a real training run.**

- **`scout-bot/trunk_analysis.py`** — Line 140: handles tuple return from `forward()`.

## Decisions

- **GELU over LeakyReLU/ELU**: GELU is standard in transformers, smooth everywhere, no hyperparameters. Directly eliminates the hard-zero dead neuron mechanism.
- **Detached value head (not branched trunk)**: Simpler than splitting the trunk into separate policy/value branches. The value head reads all trunk layer activations + raw metadata as detached features — zero gradient interference with the policy trunk. Addresses the gradient interference finding from session 048 (OLS on trunk features got EV=0.66 but the trained value head only achieved EV=0.52).
- **Tuple return from forward()**: Needed because flip-decision code calls forward() twice then value() twice — instance-stored state would use the wrong context. The tuple approach is transparent to callers since they just pass the opaque result through.

## Next Steps

1. **Reset PARAMS for a real training run.** Current PARAMS are from the entropy floors ablation. Need: fresh `save_dir` (e.g. "v6_9"), restore `entropy_floors` to `{play: 1.0, scout: 1.0}`, set `total_iterations` back to 1_000_000, consider whether to keep `zero_scout_policy_grad: True` (it was protective in ablations). The LR annealing disable (`if cfg["total_iterations"] <= 1000`) from session 054 is still in main.py around line 816 — decide whether to keep or remove it.

2. **Run training and verify.** Check: dead neuron counts should stay near zero with GELU. KL early stopping should show `kl_batch_frac < 1.0` when policy is updating aggressively. Value head EV should improve over previous runs given the richer features.

3. **Diagnostic script compatibility.** Only `trunk_analysis.py` was updated for the tuple return. Other scripts (`entropy_diagnostic.py`, `probe_v6.py`, `value_head_probe.py`, `value_warmup_test.py`) pass forward output directly to `value()`/`policy_logits()` — these work unchanged. But any script that does `net(states).numpy()` or treats the forward output as a raw tensor would break. Only `trunk_analysis.py` line 140 had this pattern and was fixed.

## Watch Out

- **No checkpoint compatibility.** GELU + new value head architecture = fresh start required. Cannot resume from any existing v6 checkpoint.
- **Value head parameter count jumped.** Old: 129 params (128→1). New: ~242K params (945→256→1 MLP). Still small relative to the full network but worth monitoring if value loss behaves differently.
- **`kl_batches_used`/`kl_batches_total` averaging across epochs.** These get summed into `ppo_sums` and divided by `actual_epochs` like all other metrics. The ratio `used/total` is preserved, but the raw numbers in `ppo_avg` are fractional. The `kl_batch_frac` metric in `metrics_history` correctly computes the ratio.
