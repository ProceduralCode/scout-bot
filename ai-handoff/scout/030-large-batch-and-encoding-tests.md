# Large Batch PPO & Encoding Isolation Tests

## Task & State

Investigated two questions: (1) Does PPO have a correct but dilute signal for learning dynamic matching? (2) Is the "one-hots required for matching" conclusion from v3 actually verified?

Both questions partially answered with new evidence. Probe 5b long run is ready for a 500-iter follow-up. V3 failure mechanism remains unexplained.

## What Changed

- `scout-bot/probe.py` — Probe 5b logging: changed from verbose on first 3 + last iter to every `n_iters // 10` iterations. Logs one line per checkpoint: `iter, adj_rate, entropy, ploss`. Removed intermediate verbose output from `_train_iteration`.
- `scout-bot/probe_diagnostic.py` — Added Test I (`test_scalar_hand`): v2 encoding with hand card one-hots collapsed to scalar values, everything else unchanged. Includes `_collapse_hand_to_scalar()` and `_eval_adj_rate_scalar()`. Dispatch: `--test I --v2`.
- `ai-handoff/scout/context.md` — Updated PPO trunk gradient section: reframed orthogonal gradient (independent, not necessarily wrong), added current options list, added rejection reasoning for auxiliary CE.

## Key Discussion Points

### PPO signal: dilute but present
- User challenged the claim that PPO's orthogonal gradient means "wrong direction." Orthogonal means independent — could be two valid paths to useful features.
- User proposed: "why not just use CE with the played action as label on wins, inverse on losses?" — this is essentially REINFORCE, the algorithm PPO is built on.
- User argued the expected gradient should be weak but correct over many games. This is theoretically true (policy gradient theorem). The question is practical: does it converge at feasible batch sizes?

### Probe 5b at 10k games/iter (100 iters)
Results show the signal IS present:
```
iter   0  adj=0.181
iter  20  adj=0.258
iter  40  adj=0.310
iter  60  adj=0.346
iter  80  adj=0.230
iter  99  adj=0.318
```
Movement from 0.181 to 0.346 is real. But unstable — drops to 0.230 at iter 80 before recovering. The test formally FAILed (0.269 → 0.317, threshold is +0.05 from init but init measured at 0.269 not 0.181).

### V3 encoding failure: not well understood
- Test I (scalar hands, no diffs, supervised CE): 0.302. Confirms scalars alone can't support matching.
- But v3 (scalars + 171 pairwise diffs) also failed (0.312). The pairwise diffs directly encode matching (diff=0 = same value). The shifted similarity variant (0.5 = match) also failed (0.223).
- User pointed out: neural networks don't have "needle in haystack" problems — a fully connected layer sees all 171 features equally and learns which matter. The "15 useful diffs buried in 156 irrelevant ones" framing is wrong.
- V3's failure remains genuinely unexplained. The diffs encode matching, the network can attend to any input features, yet supervised CE can't learn it. This contradicts the "one-hot structure is required" conclusion stated in context.md.

### Auxiliary CE rejection reasoning (from handoff 027, now in context.md)
User wants the network to develop general relational reasoning from gameplay reward alone. Supervised targets teach specific heuristics that don't build toward advanced strategies (look-ahead, proactive scouting). This is a design philosophy, not a technical limitation.

### Web research: relational deep RL
Research agent found Zambaldi et al. 2018 (Relational Deep RL) solved a structurally identical matching task (Box-World key-lock matching) using self-attention from pure RL reward. Also found: pointer networks for action selection, PCGrad for multi-head gradient conflicts, PPG for phasic policy optimization, plasticity loss literature. Full research output in agent transcript.

User noted the attention approach is conceptually similar to what v3 already tried (building comparisons into the architecture), though the mechanism differs (learned dot-product attention on embeddings vs precomputed scalar diffs).

## Decisions

- Probe 5b logging changed to 10 evenly-spaced checkpoints with one-line output.
- Auxiliary CE rejection reasoning now explicit in context.md with "do not remove" marker.
- V3 "one-hots required" conclusion should be treated as unverified — the mechanism is unexplained.

## Next Steps

1. Run probe 5b with 500 iters × 10k games to see if adj_rate stabilizes or keeps climbing.
2. Consider testing: (a) scalars + only scouted-vs-hand diffs (15 instead of 171), (b) one-hots + appended diffs, (c) one-hots + only scouted-vs-hand diffs. These would further isolate what's wrong with v3.
3. Decide on path forward based on results: larger batches in real training, architecture change (attention/pointer), or CE pretrain → PPO.

## Watch Out

- `_eval_adj_rate` in probe_diagnostic.py uses `_sample_scout` with normal encoding. Test I has its own `_eval_adj_rate_scalar` that applies the scalar transform. Other tests using scalar/modified encodings would need similar custom eval functions.
- The 500-iter probe 5b run will take a long time at 10k games/iter.
