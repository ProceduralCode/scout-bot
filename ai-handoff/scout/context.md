# Scout Bot Training

Training a neural network bot for the card game Scout (3-5 players) using PPO self-play.

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

The reasoning: if the network can't learn basic strategy (e.g., pairs beat singles) from self-play signal alone, adding scaffolding masks the problem rather than solving it. And if scaffolding is required to learn basics, there's no reason to believe the network will discover advanced strategies through self-play once that scaffolding is removed. The whole point is self-play working end-to-end.

Do not edit this section without asking the user.

**The legitimate question to ask instead:** Is this "the algorithm fundamentally can't learn" or "compute is insufficient at current scale"? These require different responses. Empirically check training curves before concluding the approach is broken — the signal may exist but be slow to accumulate.

## Architecture

- **FlatScoutNetwork** (active) — Self-attention over 20 per-card entities → FC trunk (Linear → LayerNorm → GELU) → flat policy head (384 actions). Value head is a detached 2-layer MLP reading raw global metadata + all trunk layer activations (`.detach()`ed). Config via `PARAMS["attention"]` dict, preserved from checkpoint on resume (like `layer_sizes`).
- **`forward()` returns a `(hidden, value_ctx)` tuple.** `value()` and `policy_logits()` unpack it.
- **ScoutNetwork** — FC-only with sub-head action decomposition. Used by older eval checkpoints (v1-v4).

## Encoding

- **V6** (active): 16 circular hand slots, flat 384-action space, 309 dims (260 entity + 49 global metadata). Rotation augmentation (16x) via permutation tables (`FULL_PERM`, `HAND_SHIFT`).
- **V2**: 15 hand slots, 261 dims. Used by eval checkpoints v1_4 through v4_2.
- **Flat action space**: Play [0..255] = start_slot × 16 + end_slot. Scout [256..319]. S&S [320..383]. Play length = `(end_slot - start_slot) % 16 + 1` (must use modular arithmetic for circular wrapping).

## Training Pipeline

Hybrid GAE + rollout mode via `rollout_fraction` PARAM. Rollout games provide ground-truth value targets; GAE games are cheaper. `gae_vloss_weight` controls GAE contribution to value loss (0.0 = value head trains on rollout data only). Setting `rollout_fraction=0.0` requires `gae_vloss_weight > 0` or the value head gets no gradient.

**value_baseline PARAM**: `"learned"` (default) uses GAE with the value head. `"mean"` bypasses V(s) and uses `reward - batch_mean(reward)` as advantages (lines 907-920 of main.py). Only affects the GAE path.

**Adaptive LR**: `PARAMS["learning_rate"]` is the string `"adaptive"`. Each iteration, LR is nudged by 0.9x if KL > target, 1.1x if KL < target, clamped to `[lr_min, lr_max]`. `current_lr` is a mutable variable in the training loop, saved in checkpoints via `extra={"current_lr": ...}`. Set `learning_rate` to a float to disable adaptive mode and use fixed/linear-decay LR.

**Temperature scaling**: `sampling_temperature` divides logits during data collection. `old_log_prob` recorded at T=1 (the actual policy). NOT applied in rollout completions.

**KL early stopping**: `ppo_update_v6` breaks out of mini-batch loop when `approx_kl` exceeds `kl_target`. `kl_batch_frac` tracks fraction of mini-batches that actually run.

**Reward modes**: `reward_mode` PARAM controls reward signal. `"game_score"` (default) uses margin. `"play_length"` gives `play_length/5.0` per step. `"play_and_scout"` blends both. Only affects GAE path — rollout advantages are always margin-based.

**Explained variance** computed only over samples where `v_weight > 0`.

**Ratio=1.0 invariant**: v6 path checks `first_batch_ratio` from first mini-batch of epoch 0.

Rejected (do not remove): auxiliary CE loss / supervised hand-holding during PPO training.

## Legal Play Statistics

From `legal_play_stats.py` (10K random 3p games): singles legal 53.5% of turns, pairs 72.6%, triples 13.8%, quads 1.1%. When triples are legal: avg 1.1 triple options vs 4.4 singles, 3.3 pairs. Random uniform policy plays triples ~1.7% of the time.

## Device Model

**Training on CUDA, eval on CPU.** `_run_eval` moves network to CPU, restores in `finally`. Opponent pool and eval opponents stay on CPU. Never call `.item()` in a loop on GPU tensors.

## Cython Acceleration

Two Cython modules with pure-Python fallbacks (import pattern at bottom of `encoding.py`):
- `fast_game.pyx` — legal-play computation
- `fast_encoding.pyx` — v6 state encoding and action masking

Changes to encoding/masking logic must be updated in both `.py` and `.pyx`. Build: `pushd scout-bot && python setup.py build_ext --inplace; popd`.

## Game Generation

Four versions of turn logic — keep in sync when changing game mechanics:
- `_play_turn()` / `_play_turn_v6()` — single-game (eval, rollout source)
- `_process_turn_from_hidden()` — opponent path in `play_games_batched()`
- Inline batched logic in `play_games_v6()` / `play_games_batched()` — training path
- Inline batched logic in `rollout_from_states_batched_v6()` — rollout path

## GPU Engine

`rollout_numba` in `numba_engine.py`: 4 Numba CUDA kernels + PyTorch network forward. Forward pass chunked at 4096 samples (empirically optimal on RTX 3060 6.4GB — see `bench_batch_size.py`). Rollout pipeline: `from_snapshots` → `repeat_state` → `rollout_numba` → vectorized margin computation. `gpu_engine.py` is the reference/test oracle.

Key design facts: hands are left-aligned (not circular), S&S is two steps via `state.phase`, `compute_legal_plays` returns hand-position space (not slot space).

## Training Setup

Hyperparameters in `PARAMS` dict at top of `scout-bot/main.py` — edit directly, no CLI args. On resume, PARAMS overrides saved config except `layer_sizes`, `encoding_version`, and `attention`.

Scripts use `SCRIPT_DIR` pattern. `trunk_analysis.py` segfaults without `-u` flag on Windows.

## Known Bug

`_compute_diagnostics` in main.py (line 668) computes play length as `(a % 16) - (a // 16) + 1` — doesn't handle circular slot wrapping. Should use `(a % 16 - a // 16) % 16 + 1`. Affects chart diagnostic values `diag_policy_p_single/p_pair/p_3plus` only; does not affect training.

## Key File Locations

- `scout-bot/main.py` — training loop, PARAMS, charts, adaptive LR, `_run_eval()`
- `scout-bot/training.py` — game play, PPO update (`ppo_update_v6`), rollouts, GAE
- `scout-bot/network.py` — FlatScoutNetwork (tuple-return forward, detached value head)
- `scout-bot/encoding.py` — state encoding (v1-v6), mask functions, action decoding, permutation tables
- `scout-bot/game.py` — game engine; cards are `(showing_value, hidden_value)` tuples
- `scout-bot/numba_engine.py` — Numba CUDA rollout engine (chunked forward pass)
- `scout-bot/bench_batch_size.py` — GPU forward pass throughput benchmark
- `scout-bot/canonical_states.py` — hand-crafted game states, tests basic strategy
- `scout-bot/kl_sensitivity_test.py` — KL divergence per optimizer step at various LRs
- `scout-bot/legal_play_stats.py` — legal play availability by length (random games)
- `scout-bot/triple_probe.py` — policy probability analysis for triples (loads checkpoint)
