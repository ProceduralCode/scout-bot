# Scout: Supervised Probe & Diagnostic Tests

## Task

Verify the session 019 conclusions about why scout insertion fails. The handoff claimed "PPO sample efficiency" as root cause — this session tested whether that framing is correct by running targeted experiments.

## Diagnostic Tests Run

Created `scout-bot/probe_diagnostic.py` with 5 isolated experiments. All use the probe 5b task (adjacent matching: insert scouted card next to a matching value) unless noted.

### Test A — Supervised learning (cross-entropy, no PPO)

Bypasses PPO entirely. Trains the full network (trunk + scout head) end-to-end with cross-entropy on known-correct positions.

| Config | LR | Epochs/batch | Layers | Result |
|--------|-----|-------------|--------|--------|
| No mask on logits | 3e-4 | 1 | [64,32] | FAIL 0.222→0.250 |
| Masked logits | 3e-4 | 1 | [64,32] | FAIL 0.229→0.231 |
| Masked logits | 3e-4 | 1 | [512,256,256,128,128,128] | FAIL 0.234→0.279 |
| Masked, high LR | 3e-3 | 10 | [64,32] | FAIL 0.228→0.248 |

Loss flat at ~2.48 across all configs — exactly log(12), the uniform-over-legal-slots baseline. The network predicts uniform and cannot learn to discriminate good from bad slots.

Gradient check confirmed non-zero gradients (trunk: 0.002 mean abs, head: 0.0003 mean abs). Signal flows but doesn't improve the loss.

### Test B — No value loss (value_loss_coeff=0)

Tests whether trunk gradient interference from value prediction hurts the policy signal.

| Layers | Result |
|--------|--------|
| [64,32] | FAIL 0.228→0.253 |

No effect. Value loss interference is not the cause.

### Test C — Fixed hand_offset=0

Tests whether the 20-vs-21 modulus misalignment between hand slots and insert slots matters.

| Layers | Result |
|--------|--------|
| [64,32] | FAIL 0.270→0.237 |

No effect. Slot misalignment is not the cause.

### Test D — Simple conditional (PPO)

"Insert at pos 0 if scouted value ≤ 5, else insert at last position." Tests whether PPO can learn ANY conditional scout mapping.

| Layers | Result |
|--------|--------|
| [64,32] | **PASS** 0.096→0.512 |

PPO can learn conditional scout insertion when the decision is a binary split (1 feature → 2 possible positions).

### Test E — Combined B+C

Both fixes together.

| Layers | Result |
|--------|--------|
| [64,32] | FAIL 0.246→0.234 |

## Observations

Supervised cross-entropy with non-zero gradients also fails on adjacent matching. The failure is not specific to PPO — it persists across optimizers, learning rates, network sizes, and number of passes per batch.

Test D (simple conditional) passes, showing the network can learn scout mappings where the decision depends on a global feature (card value) routed to fixed positions (0 or end).

What has NOT been tested: whether the adjacent matching task is learnable with a different head architecture (e.g., MLP instead of single linear layer) or with the raw 475-dim state as input (bypassing trunk compression). Until those tests run, the bottleneck location is still open — it could be the linear scout head, the trunk compression, the encoding, or something else.

## Next Steps

1. **Standalone MLP test** — 2-layer MLP directly from raw 475-dim state → 21 insert logits (no shared trunk). Distinguishes "encoding doesn't support the task" from "trunk + linear head can't extract it."

2. **MLP scout head test** — Replace the single linear scout head with a small MLP, keep the shared trunk. Distinguishes "trunk doesn't provide usable features" from "linear head can't use them."

3. **Mini-batching** — still the highest-impact general training change, independent of scout insertion.

## Modified Files

- **`scout-bot/probe_diagnostic.py`** — new file, 5 diagnostic experiments (A-E) for isolating scout insertion failure causes
