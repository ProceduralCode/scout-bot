# V3 Failure Mechanism Identified

## Task & State

Investigated two questions from handoff 030: (1) Does PPO converge on matching with enough iterations? (2) Why does v3 encoding fail at supervised CE?

Both answered definitively.

## Key Results

### PPO long run (probe 5b, 500 iters × 10k games, v2)
```
iter    0  adj=0.259
iter   50  adj=0.246
iter  100  adj=0.243
iter  150  adj=0.244
iter  200  adj=0.398
iter  250  adj=0.469
iter  300  adj=0.485
iter  350  adj=0.597
iter  400  adj=0.674
iter  450  adj=0.681
iter  499  adj=0.806
PASS  P(adj_match) 0.213 -> 0.699
```
~150 iters of plateau, then steady climb. PPO can learn dynamic matching from pure RL — it's a convergence speed issue, not a fundamental limitation.

### V3 failure: scalar diffs vs binary diffs

Test F with v3 (`--v3 --layers 512 256`): 0.233 → 0.375 after 300 iters. Loss barely moved (2.25→2.11). Rules out capacity — a 512-wide first layer has plenty of room. The problem is optimization.

Test J — v3 encoding with binary match indicators (1.0 if values equal, 0.0 otherwise) instead of scalar diffs:
```
iter   0  adj=0.305
iter  50  adj=0.997
iter 100  adj=1.000
PASS  0.235 -> 1.000
```
Perfect matching by iter 50 with [256, 128]. Same v3 structure, same hand scalars, same everything — only the pairwise diff encoding changed from continuous scalars to binary indicators.

## What Changed

- `scout-bot/probe_diagnostic.py` — Added Test J (`test_binary_diffs`): v3 encoding with binary pairwise match indicators. Includes `_fill_pairwise_binary()`, `_encode_v3_binary()`, `_eval_adj_rate_binary()`. Added `PAIRWISE_CARDS_V3` to imports. Dispatch: `--test J` (always uses v3 input size, ignores `--v3` flag). Good slots computed inline with `SCOUT_INSERT_SIZE_V3` to avoid global `ENCODING_VERSION` dependency.

## Decisions

- V3's scalar pairwise diffs are the confirmed bottleneck for matching. The scalar hand encoding itself is not independently sufficient (Test I: 0.302), but the diffs were supposed to compensate — and they do, but only when binary.
- PPO matching signal: confirmed present but slow. ~200 iters × 10k games to break through plateau. CE pretrain → PPO remains the pragmatic path for real training.

## Next Steps

1. Decide what to do with v3: abandon it, switch to binary diffs, or pivot to one-hots + supplementary binary diffs as a hybrid.
2. The binary diff result suggests a possible v2 enhancement: append 15 binary hand-vs-scouted-card match indicators to v2's existing encoding. This gives the network explicit match signals alongside the one-hots it already uses well.
3. If sticking with v2, the CE pretrain → PPO path (0.837 → 0.946 from handoff 027) is ready to integrate into real training.

## Watch Out

- Test J bypasses the global `ENCODING_VERSION` — it always uses v3 input size internally. Running it with `--v2` or `--v3` flags has no effect on the test itself.
- The 500-iter probe 5b result (0.806 at iter 499) is from the isolated probe environment with 10k games/iter. Real training uses 400 games/iter and a shared trunk with multiple competing heads — convergence will be much slower.
