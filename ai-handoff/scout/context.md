# Scout Bot Training

Training a neural network bot for the card game Scout (3-5 players) via self-play. Q-network (value-based) approach, replacing PPO. Pipeline verified end-to-end through iteration 5.

## Diagnosis Before Treatment

Do not propose fixes based on theories. This project has repeatedly encountered situations where plausible-sounding hypotheses turned out to be wrong once tested empirically. The pattern: Claude suggests X is the problem, we build a fix for X, it doesn't help, then empirical investigation reveals the actual cause was Y.

The rule: **understand first, fix second.** When something isn't working:

1. **Build empirical tests** — write scripts, collect data, examine actual values. Not "look at the chart and reason about it." Generate numbers, distributions, per-sample breakdowns. Look at the raw data.
2. **Design for understanding, not fixing.** Every code change during investigation should produce diagnostic output, not alter training behavior. Ablations and instrumentation, not patches.
3. **Verify the mechanism, not just the correlation.** If you think A causes B, construct a test that isolates A. Check counterexamples. Confirm the causal direction.
4. **Exhaust diagnosis before proposing architecture changes.** Structural changes are expensive (new runs, lost checkpoints, days of training). They should be the last resort after the problem is empirically pinned down.
5. **Don't theorize past your evidence.** If you've identified 3 possible causes, say so and propose tests to distinguish them. Don't pick the most narratively satisfying one.

Do not edit this section without asking the user.

## Self-Play Purity Principle

**The goal is a bot that learns strategy entirely through self-play from terminal outcomes (win/loss margin). No dense rewards, no heuristic opponents, no supervised bootstrapping.**

Do not edit this section without asking the user.

## Architecture

- **FlatScoutNetwork** — Self-attention over 20 per-card entities → FC trunk (Linear → LayerNorm → GELU) → flat head (384 predicted margins). No value head — state value = max predicted margin over legal actions.
- **`forward()` returns hidden tensor directly.** `policy_logits()` maps hidden → 384 outputs. `state_value(logits, mask)` is a static method for max-Q.
- **ScoutNetwork** — FC-only with sub-head action decomposition and value head. Used by older eval checkpoints (v1-v4).

## Q-Network Training Pipeline

`train_q()` in main.py, invoked via `python main.py` (default mode). Config in `Q_PARAMS` dict.

Per iteration: `play_games_q_v6` → `rollout_multi_action_v6` → `ReplayBuffer.add_cohort` → `prepare_q_batch_v6` (16x augmentation) → `q_update_v6` (masked MSE).

- **Multi-action rollouts**: top-K + random extra actions per decision point. `action_taken` always included. Batched through GPU in chunks of 512 pairs × N rollouts. Immediate-end games (round over after action) compute margin directly.
- **Rollout temperature**: `rollout_numba` accepts `temperature` param. 0.0 = greedy, 1.0 = softmax sampling from margin predictions.
- **Action selection** (game play): softmax(temperature) + epsilon-greedy.
- **Replay buffer**: cohort-based. Periodic revalidation re-rollouts a subset; weight = `max(0, 1 - mae / margin_max_diff)`. Dead cohorts removed.
- **Augmentation**: on-the-fly via `FULL_PERM`/`HAND_SHIFT` permutation tables. All 16 rotations (including identity). No forward pass needed — just index permutations on state, target, and training mask tensors.
- **Flip decisions**: max predicted margin over play actions [0..255] for each hand orientation.

## Encoding

- **V6** (active): 16 circular hand slots, flat 384-action space, 309 dims (260 entity + 49 global metadata). Rotation augmentation (16x) via permutation tables.
- **Flat action space**: Play [0..255] = start_slot × 16 + end_slot. Scout [256..319]. S&S [320..383]. Play length = `(end_slot - start_slot) % 16 + 1` (must use modular arithmetic for circular wrapping).

## GPU Engine & Rollouts

`rollout_numba` in `numba_engine.py`: 4 Numba CUDA kernels + PyTorch network forward. Active-game compaction: forward pass only runs on non-done games each step (gather active indices → forward → scatter logits back). Numba kernels still process all B games (only 6% of step time). Forward pass chunked at 1024 samples (RTX 3060).

Pipeline: `from_snapshots` (gpu_engine.py) → `repeat_state` → `rollout_numba` → margin computation. `gpu_engine.py` is the reference/test oracle.

Key design facts: hands are left-aligned (not circular), S&S is two steps via `state.phase`, `compute_legal_plays` returns hand-position space (not slot space).

## Rollout Performance Profile

Per-step cost at B=15,300 (one chunk): forward pass 79%, sampling 8%, Numba kernels 10%, overhead 3%. Rollouts hit MAX_STEPS=100 every chunk; at step 100, ~11% of games still active (long scouting rounds with temperature=1.0 early in training). `chunk_pairs` parameter has negligible effect (benchmarked 256–16384+).

## Cython Acceleration

Two Cython modules with pure-Python fallbacks (import pattern at bottom of `encoding.py`):
- `fast_game.pyx` — legal-play computation
- `fast_encoding.pyx` — v6 state encoding and action masking

Changes to encoding/masking logic must be updated in both `.py` and `.pyx`. Build: `pushd scout-bot && python setup.py build_ext --inplace; popd`.

## Training Setup

**Device model**: Training on CUDA, eval on CPU. Never call `.item()` in a loop on GPU tensors.

Scripts use `SCRIPT_DIR` pattern. `trunk_analysis.py` segfaults without `-u` flag on Windows. Use `python -u` for unbuffered output when running training in background.

## Key File Locations

- `scout-bot/main.py` — `Q_PARAMS`, `train_q()`, `_save_q_charts()` (4×4 grid), `_run_eval()` (has `chart_fn` param), PPO `train()` preserved
- `scout-bot/training.py` — `QSample`, `ReplayBuffer`, `play_games_q_v6`, `rollout_multi_action_v6`, `prepare_q_batch_v6`, `q_update_v6`. Old PPO functions preserved as dead code.
- `scout-bot/network.py` — FlatScoutNetwork, ScoutNetwork (v1-v4)
- `scout-bot/encoding.py` — state encoding (v1-v6), mask functions, action decoding, permutation tables
- `scout-bot/game.py` — game engine; cards are `(showing_value, hidden_value)` tuples
- `scout-bot/numba_engine.py` — Numba CUDA rollout engine (active-game compaction, `temperature` param)
