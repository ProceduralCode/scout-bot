# Q-Network Implementation (In Progress)

## Task & State

Started implementing the Q-network training system. Phases 1-3 of 7 are complete, phase 4 was in progress when interrupted.

## What Changed

### network.py — FlatScoutNetwork simplified
- Removed value_head, `value()` method, `_run_trunk()` intermediate tracking
- `forward()` now returns hidden tensor directly (not a `(hidden, value_ctx)` tuple)
- `policy_logits(h)` takes hidden directly instead of unpacking a tuple
- Added `state_value(logits, mask)` static method — max over masked outputs
- Attention, trunk, entity encoding, policy_head all unchanged

### training.py — New data structures and game play
- Added `QSample` dataclass (state, action_mask, network_outputs, snapshot, rolled_actions, rollout_margins, etc.)
- Added `ReplayBuffer` class with cohort management: add_cohort, sample_training_data, revalidate, check_and_prune, state_dict/load_state_dict
- Added `_apply_action_to_game()` helper (factored out action application logic)
- Added `_select_action_q()` and `_select_action_q_batched()` — softmax(temperature) + epsilon-greedy
- Added `play_games_q_v6()` — plays games collecting QSamples with game snapshots at each decision point
- Fixed `_play_round` flip decisions for v6: uses max-Q over play actions instead of removed value head
- Fixed `_play_turn_v6` to use `hasattr(net, 'value')` check for backward compat with eval

### main.py — Not yet modified

## Decisions

- **No value head** — state value = max(predicted margin over legal actions). The value head was redundant since the network already predicts per-action margins.
- **Flip decisions via max-Q** — at round start (no play on table), compare max predicted margin over play actions [0..255] for each hand orientation. Eval opponents using old ScoutNetwork/CircularCNN still use their value heads.
- **Cohort-based replay buffer** — each iteration's samples are a cohort. Periodic revalidation re-rollouts a subset to measure staleness. Linear weight fade: `max(0, 1 - mae / margin_max_diff)`. Dead cohorts (weight below min_replay_perc) are removed entirely.
- **Store full 384-output vector per sample** for debugging and analysis
- **Store game snapshots in cohorts** for revalidation rollouts (~7-8MB per cohort, manageable)
- **Augment on the fly** during training, not pre-stored — just permutation table lookups per sample
- **Training mask** — only the 10-12 rolled-out actions per state get loss gradients, rest are masked

## Next Steps

### Immediate: Continue from Phase 4
The phases and what remains:
1. ~~Network changes~~ — done
2. ~~Data structures~~ — done
3. ~~Game play function~~ — done
4. **Multi-action rollout** (`rollout_multi_action_v6`) — NOT STARTED. Takes QSamples, selects top-K + random actions, batches through GPU rollout pipeline, fills in rollout_margins/rollout_stds.
5. **Training step** (`prepare_q_batch_v6`, `q_update_v6`) — prepare_q_batch builds tensors with on-the-fly augmentation, q_update does masked MSE loss
6. **Main loop** — Q_PARAMS dict (overlaid on existing PARAMS for reference), new train() iteration flow, checkpoint saving with replay buffer
7. **Charts & metrics** — drop PPO-specific, add: MSE loss, Q-value accuracy, replay buffer health

### Config parameters agreed on
```
game_count: 100, temperature: 0.0, epsilon: 0.05
rollout_actions_per_sample: 10, rollout_actions_random_extra: 2, rollouts_per_action: 30
augment_rotations: 16 (on the fly)
training_epochs: 3, mini_batch_size: 2**14, learning_rate: 0.0003
cohort_check_interval: 5, replay_check_perc: 0.1
replay_margin_max_diff: TBD, min_replay_perc: 0.3
```

## Watch Out

- `play_games_q_v6` records the snapshot BEFORE applying the action (the snapshot is the decision point state). The multi-action rollout needs to clone that snapshot and apply each candidate action before rolling out.
- Old PPO functions (`play_games_v6`, `play_games_with_rollouts_v6`, `ppo_update_v6`) are still in training.py as dead code — not deleted, not updated for the new network interface. They'll crash if called with the new FlatScoutNetwork.
- The `_select_action_q_batched` epsilon path uses a per-element loop for random legal action selection. May need vectorization if it becomes a bottleneck.
- Game log saving in main.py (~line 1278) calls `play_game()` which goes through `_play_round` → `_play_turn` → `_play_turn_v6`. The `_play_turn_v6` fix (hasattr check) makes this work but it records value=0.0 for all steps.
- Rollout cost estimate: 100 games × ~15 states × 12 actions × 30 rollouts = ~540K rollout games per iteration. Should be feasible on the Numba GPU engine but worth monitoring.
