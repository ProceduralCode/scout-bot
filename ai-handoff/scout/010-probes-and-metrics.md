# Scout Bot: New Probes, Metrics, and Distributed Reward

## Task
Extend the probe framework to test more subsystems, add training metrics/charts, and add a distributed reward option for better credit assignment.

## What Changed

### Modified Files

- **`scout-bot/training.py`**
  - `StepRecord` (line 42): added `play_length: int | None = None` field
  - `_play_turn` (line 180): sets `rec.play_length = end_idx - start_idx + 1` for play actions
  - `play_game` (line 44-48): added `reward_distribution: str = "terminal"` parameter
  - Reward assignment block (lines 67-85): refactored to support `"terminal"` (only last step gets reward, original behavior) and `"uniform"` (spread reward equally across all player steps in the round). Groups records by player, computes per-step reward as `round_reward / num_steps`.

- **`scout-bot/main.py`**
  - `DEFAULT_CONFIG` (line 34): added `"reward_distribution": "terminal"` — **user changed this to `"uniform"` to try it**
  - `metrics_history` (line 218): added `"avg_play_length"` and `"reward_std"` tracking
  - Logging block (lines 322, 334-335, 350-351): computes `reward_std` (std of per-round rewards) and `avg_play_length` (mean cards per play action)
  - Charts (line 92): expanded from 4×3 to 5×3 grid
  - Row 4 charts (lines 197-203): `avg_play_length`, `reward_std`, `entropy_floor_penalty`
  - `play_game` call (line 283): passes `reward_distribution` from config

- **`scout-bot/probe.py`** — extensive additions:
  - Updated imports (lines 18-24): added `get_scout_insert_mask`, `decode_action_type`, `PLAY_SLOTS`, `SCOUT_INSERT_SIZE`
  - `_encode` (line 45): now randomizes `play_offset` when `game.current_play` exists (was hardcoded to 0)
  - `_mid_round_state()` (line 449): helper that creates a game state where scouting is legal (player 0 plays, then it's player 1's turn)
  - `_hand_quality()` (line 465): scores a hand by its longest available play
  - `_sample_scout()` (line 472): samples a scout insert decision from the network (always left-end, no flip)
  - `_make_scout_record()` (line 505): builds a StepRecord for scout actions
  - **Probe 5** `probe_scout_insert` (line 522): tests if insert head can learn to maximize hand quality. Uses continuous reward, locked at_mask. FAILS at [64,32] — task requires state-dependent card reasoning the small network can't handle.
  - **Probe 5b** `probe_scout_adjacent` (line 603): simpler test — place card next to matching value. Also FAILS at [64,32]. Head works mechanically (trivial "always pick pos 0" passes easily) but can't learn value-matching from the encoding at this capacity.
  - `_sample_action_type()` (line 679): samples an action type decision from mid-round states
  - `_make_action_type_record()` (line 703): builds a StepRecord for action type decisions
  - **Probe 6** `probe_action_type` (line 717): tests if action_type head can learn play-over-scout preference. PASSES easily (0.15 → 0.91).
  - **Probe 7** `probe_gae_multistep` (line 778): tests if value head learns to predict returns from early steps in a 3-step episode with terminal-only reward. FAILS — V can't bootstrap backward across unrelated states. Demonstrates the real credit assignment problem.
  - `ALL_PROBES` dict (line 853): maps probe numbers to functions. Probe 5b uses key `55`.
  - `main()` (line 864): added `--probe` argument for selecting specific probes by number.

## Decisions

- **Distributed reward over reward shaping**: user was concerned about GAE not propagating signal to early-round actions (which we confirmed with probe 7). Instead of adding intermediate heuristic rewards (reward shaping, which biases learning), we spread the terminal reward equally across all steps. Crude but unbiased — every step gets signal proportional to the round outcome.
- **Continuous reward for scout probe**: binary +1/-1 was too sparse for 12+ positions. Switched to scaled quality difference. Didn't help at [64,32] — the issue is network capacity, not reward sparsity.
- **Locked at_mask in scout probes**: forcing the action_type mask to only allow the scout action eliminates gradient noise from the action_type head, isolating the insert head's learning.

## Key Findings

### Probe Results Summary

| Probe | Result | What it shows |
|---|---|---|
| 5 (scout insert quality) | FAIL | Hand quality optimization too hard for [64,32] |
| 5b (scout adjacent match) | FAIL | Even value-matching fails at [64,32] |
| 6 (action type) | PASS (0.15→0.91) | Action type head learns simple preferences easily |
| 7 (GAE multi-step) | FAIL | V can't bootstrap backward across unrelated states |
| trivial (always pos 0) | PASS (0.12→1.0) | Insert head works mechanically, problem is capacity |

### GAE Understanding
- With terminal-only reward, non-terminal steps get advantage ≈ 0 when V is flat (early training). Only the last step per round gets real gradient signal.
- The value function needs to learn state discrimination before credit can propagate backward. This is slow but correct — the concern is whether heads collapse before V gets good enough.
- Distributed reward (`"uniform"`) provides a signal floor: every step gets directional signal even with flat V.

### Scout Insert Head
- Architecturally sound (trivial preference test passes immediately)
- Can't learn state-dependent card-value reasoning with [64,32] network
- With [128,64], briefly reached 0.415 P(adj_match) (2.4× baseline) but then entropy-collapsed
- The real network [512,256,256,128,128,128] has 128-dim hidden state (4× probe) — may have enough capacity
- Entropy floors should help prevent the collapse seen with [128,64]

## Next Steps

1. **Run training with `reward_distribution: "uniform"`** — user already changed the config. Use `python scout-bot/main.py` (save_dir is `v2_5`). Compare reward/eval trajectories to terminal-only baseline.
2. **Try probe 5b with larger network** — `[256, 128]` or `[512, 256, 256, 128, 128, 128]` to see if capacity is the bottleneck for scout insert learning.
3. **Try probe 5b with entropy floor** — the [128,64] run showed learning before collapse; an entropy floor might sustain it.
4. **Alternative objective training** — train the bot to maximize play length as a direct reward (simpler signal than game outcome). Tests whether the play heads can learn in real games with clear reward.

## Watch Out
- `reward_distribution: "uniform"` changes the reward semantics. With `"uniform"`, ALL steps have nonzero reward, so the `p0_rewards` filter (`if r.reward != 0.0`) in the logging block now includes every step, not just terminal ones. The `avg_reward` and `reward_std` metrics will have different meaning — they're now per-step averages rather than per-round terminal rewards. This is fine but means the metrics aren't directly comparable across `terminal` vs `uniform` runs.
- The probe `LAYER_SIZES` constant is `[64, 32]` (line 30). Probes that test state-dependent reasoning (5, 5b) may need this bumped up to be meaningful, or a `--layers` CLI argument added.
- Probe 7 uses independent random states per step, which is a worst case for GAE. Real training has correlated sequential states within a round, so the probe may be too pessimistic.

## Files to Read First
- `scout-bot/probe.py:449-677` — new helpers and probes 5/5b
- `scout-bot/probe.py:679-849` — probes 6 and 7
- `scout-bot/training.py:44-85` — distributed reward implementation
- `scout-bot/main.py:197-203` — new chart row
- `scout-bot/main.py:34` — reward_distribution config (user set to "uniform")
