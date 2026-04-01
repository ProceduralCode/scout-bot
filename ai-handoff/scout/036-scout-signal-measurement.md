# Scout Signal Measurement

## Task & State

Ran v4 probes with larger network, then investigated why the scout head doesn't learn from gameplay despite the signal objectively existing. Created a diagnostic tool that directly measures rollout value differences across all insert positions. The finding: the signal is large and statistically significant, but the training pipeline fails to extract it.

## What Changed

- `scout-bot/probe_scout_signal.py` — New diagnostic. Loads a checkpoint, generates scoutable states, tries every insert position with rollouts, reports signal strength. Usage: `python scout-bot/probe_scout_signal.py --checkpoint PATH [--states N] [--rollouts N]`. Has a unicode encoding error on Windows in the summary print (approx symbol) — cosmetic only, all data prints before the crash.

## Probe Results (v4 and v2 with [256, 128, 128], 300 iters, 100 games)

Previously failing probes at [64, 32]:

| Probe | v4 | v2 |
|---|---|---|
| 2b start best-ends | PASS (0.12→0.26) | PASS (0.08→0.25) |
| 4 full chain | PASS (1.14→1.27) | PASS (1.15→1.31) |
| 5 scout insert | FAIL (2.23→2.23) | FAIL (2.23→2.24) |
| 5b scout adjacent | PASS (0.17→0.31) | PASS (0.23→0.30) |
| 7 GAE multi-step | FAIL (0.31→0.32) | FAIL (0.28→0.32) |

Probes 2b, 4, 5b were capacity-limited — pass with bigger network. Probes 5 and 7 fail on both encodings at any size.

## Scout Signal Diagnostic (v4_2 checkpoint, 100 states, 20 rollouts/position)

- Best - Worst position: 0.514 margin (5.1 game-score points)
- Best - Random position: 0.252 margin (2.5 game-score points)
- Adjacent-match vs non-adjacent: 0.147 margin (1.5 points), t-stat=9.05
- 61/100 states had cards with matching values in hand

The signal is large. The v4_2 play policy exploits pairs (60% pairs, 32% singles at 19k iters), and placing next to a match produces measurably better rollout outcomes. The training pipeline fails to transmit this to the scout head.

## v5_1 Training Run

v5_1 is v2 encoding + rollout mode (not v4 encoding as initially assumed). At 720 iters:
- Eval margins still improving vs v1_4 (+7.95) and v2_5 (+5.01)
- vs v3_4: -4.83, vs v4_2: -12.15
- Learning curve oscillates rather than steady monotonic improvement seen in v4_2 (GAE)
- play=7.2s train=4.5s per iteration (vs ~1.5s/1.0s for GAE)
- Rollouts_per_state=5 in PARAMS, noise per advantage estimate ~0.35 margin

## Next Steps

1. Fix the unicode error in probe_scout_signal.py (replace ≈ with ~=)
2. Investigate why PPO can't extract the scout signal despite it being large. Candidates: advantage normalization mixing scout/play records, shared trunk gradient dilution, scout head seeing too few samples per iteration, rollout noise floor.
3. Decide whether to continue with rollout mode or revert to GAE for further training.

## Watch Out

- v5_1 PARAMS has `"encoding_version": 4` currently set but the checkpoint is v2. Resume logic reads encoding_version from checkpoint, but fresh start would create wrong network. The commented-out `"encoding_version": 2` line is the one that matches v5_1.
- Probe 7 (GAE multi-step) uses independent random states per "episode step" — no real temporal structure for GAE to exploit. May be a probe design issue rather than a training failure.
- Probe 5 (scout insert quality) shows zero improvement even with direct per-step reward and large network. The reward signal ("did hand quality improve?") may be too diffuse for the probe's setup, despite the signal being clearly present in rollout-based measurement.
