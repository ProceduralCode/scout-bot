# Scout Bot Training

Training a neural network bot for the card game Scout (3-5 players) via self-play. Q-network (value-based) approach, replacing PPO.

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
- **Attention**: entities are 16 hand slots + 4 scout card options. Each has 13 raw features + 20-dim positional one-hot → projected to `dim` → self-attention layers with pre-norm residuals. Config in `Q_PARAMS["attention"]`.
- **ScoutNetwork** — FC-only with sub-head action decomposition and value head. Used by older eval checkpoints (v1-v4).

## Q-Network Training Pipeline

`train_q()` in main.py, invoked via `python main.py` (default mode). Config in `Q_PARAMS` dict.

Per iteration: `play_games_q_v6` → [`curate_samples`] → `attach_snapshots` → `rollout_multi_action_v6` → `ReplayBuffer.add_cohort` → `prepare_q_batch_v6` (16x augmentation) → `q_update_v6` (masked MSE).

- **Deferred snapshots**: `play_games_q_v6` returns `(samples, game_replays)`. Samples have `snapshot=None`; game_replays stores initial state + flip decisions + action list per game. `attach_snapshots()` replays only games with surviving samples to create Game clones on demand.
- **Sample curation** (`curation_multiplier` config): plays N× more games, scores each sample by inverse frequency of its legal actions (rotation-aware), subsamples to equalize per-output-neuron training signal.
- **Multi-action rollouts**: top-K + random extra actions per decision point. `action_taken` always included. Batched through GPU in chunks of 512 pairs × N rollouts.
- **Rollout temperature**: `rollout_numba` accepts `temperature` param. 0.0 = greedy, 1.0 = softmax sampling.
- **Action selection** (game play): softmax(temperature) + epsilon-greedy.
- **Replay buffer**: cohort-based. Periodic revalidation re-rollouts a subset (batched in chunks of 512); weight = `max(0, 1 - mae / margin_max_diff)`. Dead cohorts removed. Known issue: revalidation MAE is dominated by rollout noise, not policy drift — cohorts settle to ~0.5 weight and never get pruned.
- **Augmentation**: on-the-fly via `FULL_PERM`/`HAND_SHIFT` permutation tables. All 16 rotations (including identity).
- **Flip decisions**: max predicted margin over play actions [0..255] for each hand orientation.
- **Charts**: 4×4 grid in `_save_q_charts()`. Row 0: eval margins, MSE, pred vs target, steps/game. Row 1: play length distribution, avg play length, scout play length, action type distribution. Row 2: entropies, dormant neurons, rollout margins histogram, rollout signal vs noise. Row 3: margin predictions histogram, replay cohorts by age, (hidden), (hidden).
- **Progress bars**: tqdm on outer training loop (hidden during inner bars), inner bars for games and rollouts.

## Encoding

- **V6** (active): 16 circular hand slots, flat 384-action space, 309 dims (260 entity + 49 global metadata). Rotation augmentation (16x) via permutation tables.
- **Flat action space**: Play [0..255] = start_slot × 16 + end_slot. Scout [256..319]. S&S [320..383]. Play length = `(end_slot - start_slot) % 16 + 1` (must use modular arithmetic for circular wrapping).

## GPU Engine & Rollouts

`rollout_numba` in `numba_engine.py`: 4 Numba CUDA kernels + PyTorch network forward. Active-game compaction: forward pass only runs on non-done games each step. Forward pass chunked at 1024 samples (RTX 3060).

Pipeline: `from_snapshots` (gpu_engine.py) → `repeat_state` → `rollout_numba` → margin computation. `gpu_engine.py` is the reference/test oracle.

Key design facts: hands are left-aligned (not circular), S&S is two steps via `state.phase`, `compute_legal_plays` returns hand-position space (not slot space).

## Cython Acceleration

Two Cython modules with pure-Python fallbacks (import pattern at bottom of `encoding.py`):
- `fast_game.pyx` — legal-play computation
- `fast_encoding.pyx` — v6 state encoding and action masking

Changes to encoding/masking logic must be updated in both `.py` and `.pyx`. Build: `pushd scout-bot && python setup.py build_ext --inplace; popd`.

## Training Setup

**Device model**: Training on CUDA, eval on CPU. Never call `.item()` in a loop on GPU tensors.

Scripts use `SCRIPT_DIR` pattern. Use `python -u` for unbuffered output when running training in background.

## Key File Locations

- `scout-bot/main.py` — `Q_PARAMS`, `train_q()`, `_save_q_charts()` (4×4 grid + summary.txt), `_run_eval()`
- `scout-bot/training.py` — `QSample`, `ReplayBuffer`, `play_games_q_v6`, `attach_snapshots`, `curate_samples`, `rollout_multi_action_v6`, `prepare_q_batch_v6`, `q_update_v6`
- `scout-bot/network.py` — FlatScoutNetwork, ScoutNetwork (v1-v4)
- `scout-bot/encoding.py` — state encoding (v1-v6), mask functions, action decoding, permutation tables
- `scout-bot/game.py` — game engine; cards are `(showing_value, hidden_value)` tuples
- `scout-bot/numba_engine.py` — Numba CUDA rollout engine
