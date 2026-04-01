# Scout: PPO Composition Failure — Isolated and Verified

## Task & State

Investigated why the scout_insert head can't learn context-dependent placement via PPO despite supervised CE working perfectly. Ran systematic experiments to isolate the failure. The root cause is narrowed to a specific compositional gap: PPO can learn each sub-skill independently but cannot compose them.

## What Changed

- **`scout-bot/probe_ppo_variants.py`** — new file with three PPO variant tests:
  - `graded`: distance-based reward instead of binary +1/-1
  - `fixed_val`: "insert adjacent to any card with value=5" (fixed target, no need to read scouted card)
  - `big_net`: adjacent matching with [512,256,256,128,128,128] network
  - (also has a `hint` test that's broken — ppo_update can't propagate per-sample hints through batched recomputation)

- **`ai-handoff/scout/context.md`** — updated Scout Insertion Problem section with composition failure findings

## Experimental Results

All tests with `--v2`:

| Test | Task | Result |
|------|------|--------|
| Probe 9 | Always insert at pos=0 (state-independent) | PASS 0.084→0.202 |
| Test D | If scouted value≤5→pos=0, else→end (simple conditional) | PASS 0.072→0.572 |
| **fixed_val** | **Insert adjacent to any card with value=5 (positional matching, fixed target)** | **PASS 0.249→0.861** |
| Probe 5b (100×50) | Insert adjacent to scouted card's value (variable target) | FAIL 0.254→0.235 |
| Probe 5b (500×200) | Same, 10x data | FAIL 0.233→0.232 |
| Graded reward | Adjacent matching with distance-based reward | FAIL 0.278→0.287 |
| Big network | Adjacent matching with [512,256,256,128,128,128] | FAIL 0.245→0.245 |
| Test B (v2) | Adjacent matching, no value loss | FAIL 0.239→0.249 |

Key finding: fixed_val (0.861) proves PPO can learn positional matching against hand cards when the target value is constant. Test D (0.572) proves PPO can read the scouted card's value and act on it. But PPO cannot compose these: "read scouted value, then find it in hand."

Ruled out: reward sparsity (graded: FAIL), network capacity (big_net: FAIL), trunk gradient interference (test B: FAIL), sample size (10x: FAIL).

## Decisions

- User rejected auxiliary supervised CE loss as a solution. Reasoning: the network needs to develop general relational reasoning from gameplay reward for advanced strategies (look-ahead, proactive scouting). Teaching it specific heuristics via supervised targets doesn't build that capability.
- Discussed attention as an architectural solution. User pushed back on complexity — questioned why learned embeddings beat simple value/10 scalars, and whether self-attention's weighted-sum aggregation loses pairwise information.

## Next Steps

1. **Determine the right architectural change** to give the network relational primitives without hand-engineering specific comparisons. Options discussed but none decided:
   - Self-attention over hand card positions (concerns: weighted-sum lossy, learned embeddings may be unnecessary complexity)
   - Pairwise relational network (explicit all-pairs comparisons, addresses user's "15×15" instinct, heavier)
   - FiLM conditioning (scouted card modulates per-position features, lighter)
   - Processing hand as structured sequence rather than flat vector (benefits all heads)

2. **Research question**: what's the simplest architectural change that gives the trunk relational capability across all heads, not just scout_insert?

## Watch Out

- `probe_ppo_variants.py` has a broken `hint` test (HintedScoutNetwork) — `ppo_update` recomputes logits from stored states and can't access per-sample hints. The `fixed_val` test covers the same hypothesis more cleanly.
- The composition failure isn't specific to scout insertion — it implies play heads also can't learn relational card reasoning from PPO. This hasn't been tested but is likely true given the same architecture.
