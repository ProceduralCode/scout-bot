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

- **FlatScoutNetwork** — Self-attention over 20 per-card entities (single-head d=64, 2 layers, SDPA fused kernel) → FC trunk `[512, 256, 256, 128, 128]` (Linear → LayerNorm → GELU) → flat head (384 predicted margins). No value head — state value = max predicted margin over legal actions.
- **`forward()` returns hidden tensor directly.** `policy_logits()` maps hidden → 384 outputs. `state_value(logits, mask)` is a static method for max-Q.
- **Attention**: entities are 16 hand slots + 4 scout card options. Each has 13 raw features + 20-dim positional one-hot → projected to `dim` → `F.scaled_dot_product_attention` → out Linear, with pre-norm residuals.
- **ScoutNetwork** — FC-only with sub-head action decomposition and value head. Used by older eval checkpoints (v1-v4).

## Q-Network Training Pipeline

`train_q()` in main.py, invoked via `python main.py` (default mode). Config in `Q_PARAMS` dict.

Per iteration: `play_games_q_v6` → [`curate_samples`] → `attach_snapshots` → `rollout_multi_action_v6` → `ReplayBuffer.add_cohort` → `prepare_q_batch_v6` (16x augmentation) → `q_update_v6` (masked MSE).

- **Deferred snapshots**: `play_games_q_v6` returns `(samples, game_replays)`. Samples have `snapshot=None`; game_replays stores initial state + flip decisions + action list per game. `attach_snapshots()` replays only games with surviving samples to create Game clones on demand.
- **Sample curation** (`curation_multiplier` config): plays N× more games, scores each sample by inverse frequency of its legal actions (rotation-aware), subsamples to equalize per-output-neuron training signal.
- **Multi-action rollouts**: top-K + random extra actions per decision point. `action_taken` always included. Batched through GPU in chunks of 512 pairs × N rollouts. Supports `probe_reward` param to skip rollouts and assign deterministic targets for architecture validation.
- **Action selection**: gameplay uses `_select_action_q_batched` — softmax(logits/temperature) + epsilon-greedy. Eval uses `_play_turn_v6` which detects Q-networks via `not hasattr(net, 'value')` and uses argmax (greedy).
- **Replay buffer**: sliding-window cohort buffer. `replay_window` config controls how many iterations of data to keep. `replay_window: N` means each sample is trained on ~N times, plus 16x augmentation.
- **Flip decisions**: max predicted margin over play actions [0..255] for each hand orientation.
- **Charts**: 4×4 grid in `_save_q_charts()`. Row 0: eval margins, MSE, pred vs target, steps/game. Row 1: play length dist, avg play length, scout play length, action type dist (post-curation). Row 2: margin hist, rollout hist, signal vs noise, (hidden). Row 3: entropies, dormant neurons, (hidden), (hidden).
- **Signal vs noise chart**: `mean_rollout_std` in metrics_history actually stores SE (not raw std).
- **Eval config**: `eval_games` controls games per opponent (default 40). `eval_scout_quality` uses 2000 samples with greedy (argmax) action selection.

## Encoding

- **V6** (active): 16 circular hand slots, flat 384-action space, 309 dims (260 entity + 49 global metadata). Rotation augmentation (16x) via permutation tables.
- **Flat action space**: Play [0..255] = start_slot × 16 + end_slot. Scout [256..319]. S&S [320..383]. Play length = `(end_slot - start_slot) % 16 + 1` (must use modular arithmetic for circular wrapping).

## GPU Engine & Rollouts

`rollout_numba` in `numba_engine.py`: 4 Numba CUDA kernels + PyTorch network forward. Active-game compaction at two levels: forward pass only runs on non-done games each step; batch-level compaction (`compact_threshold`, default 0.5) gathers only active games into smaller tensors when active fraction drops below threshold. Forward pass chunked at 1024 samples (RTX 3060).

Key design facts: hands are left-aligned (not circular), S&S is two steps via `state.phase`, `compute_legal_plays` returns hand-position space (not slot space).

## Cython Acceleration

Two Cython modules with pure-Python fallbacks (import pattern at bottom of `encoding.py`):
- `fast_game.pyx` — legal-play computation
- `fast_encoding.pyx` — v6 state encoding and action masking

Changes to encoding/masking logic must be updated in both `.py` and `.pyx`. Build: `pushd scout-bot && python setup.py build_ext --inplace; popd`.

## Training Setup

**Device model**: Training on CUDA, eval on CPU. Never call `.item()` in a loop on GPU tensors.

Scripts use `SCRIPT_DIR` pattern. Use `python -u` for unbuffered output when running training in background.

## Game Visualization

Q-network path does not save game logs. PPO path saves them at `save_interval`. To replay a saved game: `python scout-bot/main.py --replay <path.json>`. To run a match (scores only): `python scout-bot/main.py --match <agent1> <agent2> ...`. No way to watch a Q-network game live yet.

## Key File Locations

- `scout-bot/main.py` — `Q_PARAMS`, `train_q()`, `_save_q_charts()`, `_run_eval()`
- `scout-bot/training.py` — `QSample`, `ReplayBuffer`, `play_games_q_v6`, `attach_snapshots`, `curate_samples`, `rollout_multi_action_v6`, `prepare_q_batch_v6`, `q_update_v6`
- `scout-bot/network.py` — FlatScoutNetwork, ScoutNetwork (v1-v4)
- `scout-bot/encoding.py` — state encoding (v1-v6), mask functions, action decoding, permutation tables, `get_legal_plays`
- `scout-bot/game.py` — game engine; cards are `(showing_value, hidden_value)` tuples
- `scout-bot/numba_engine.py` — Numba CUDA rollout engine
- `scout-bot/probe.py` — `eval_scout_quality` (2000 samples, greedy) and other probe functions
- `scout-bot/matchup.py` — `--match` mode agent loading and matchup runner
- `scout-bot/game_log.py` — `GameLog`, `print_replay` for game visualization
