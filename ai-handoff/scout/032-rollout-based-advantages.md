# Rollout-Based Advantage Estimation

## Task & State

Designed and implemented a rollout-based advantage estimation system as an alternative to GAE. The motivation: PPO's trunk gradient for dynamic matching is nearly orthogonal to CE's (cos_sim ~0.1), likely because the learned value function is poor early in training, making GAE advantages noisy. Rollout-based V estimates bypass the value function bootstrap entirely.

Implementation is complete and tested end-to-end. Not yet run on real training — needs a training run to evaluate whether it improves learning signal.

## What Changed

- `scout-bot/training.py` — Added two functions:
  - `rollout_from_state()` — deepcopy a game snapshot, play to round end with network, return scores
  - `play_games_with_rollouts()` — plays games sequentially, snapshots game state before every `_play_turn` call, runs N rollouts from each snapshot after the game finishes, computes advantages as V_after - V_before
- `scout-bot/main.py` — Added PARAMS: `use_rollouts` (bool), `rollout_games` (int), `rollouts_per_state` (int). Conditional in training loop uses new path when `use_rollouts` is true.

## Decisions

- **Advantage = V_after - V_before** — empirical temporal difference. Snapshot before the action, snapshot after. Run N rollouts from each. The difference in average margins isolates the action's effect on expected outcome. This avoids needing GAE, per-step rewards, or discount factor math.
- **Store full round scores per rollout, not per-player outcomes** — enables reusing rollouts across players. V_after for player A's action is the same game state as V_before for player B's next action; same rollout data, different player perspective. Halves rollout cost.
- **Keep PPO** — user wants to save games and learn from them multiple times. PPO's importance sampling ratio enables multiple epochs over the same batch, amortizing expensive rollout cost.
- **Sequential rollouts for now** — batching N rollouts from the same state would be a significant speedup but adds complexity. Left for later once we know the approach works.
- **S&S records share snapshot pair** — S&S produces 2 records (scout + forced play) from one `_play_turn` call. Both get the same (before, after) snapshot pair.

## Next Steps

1. Run a training session with `use_rollouts: True` to see if it improves learning (especially scout insertion quality). Start with small network or few iterations to validate signal quality.
2. If the approach works, batch the rollouts for speed — all N rollouts from the same snapshot can be run with `play_games_batched`-style logic.
3. Consider whether the value function target (currently `record.value = V_rollout_before`) is the right thing to train against, or whether to skip value function training entirely in rollout mode.

## Watch Out

- **Cost**: with [64,32] network, ~5s per game at 3 rollouts/state. Full-size network with 40 games × 10 rollouts/state will be ~10-30 min per iteration. The cost is O(games × decisions_per_game × rollouts_per_state).
- **Snapshot-record alignment**: the mapping between snapshots and records handles skip-turns (0 records, no snapshot added) and S&S (2 records, 1 snapshot pair). If `_play_turn` logic changes, the cursor-based mapping in `play_games_with_rollouts` may need updating.
- **No opponent pool in rollout mode**: all seats use the training network. The opponent pool still gets populated (snapshot_interval) but isn't used during rollout game generation. This is intentional — rollouts estimate value under self-play.
