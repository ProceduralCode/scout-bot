# Training Improvements Handoff

## Task

The Scout bot plateaus after ~200 iterations — reward flattens at ~0.84, bot rarely plays 3+ card combos. We diagnosed the issues and planned four changes to break through the ceiling. Ready to implement.

## What Was Completed

### Minor changes already applied to `main.py`:
- Added `import textwrap` (line 5)
- Chart descriptions now use `textwrap.fill(desc, 50)` for multi-line wrapping, font size 7, positioned at y=-0.15
- Figure size reduced from (16,22) to (14,20)
- Tighter margins: `left=0.06, right=0.98, top=0.96, bottom=0.03, hspace=0.40, wspace=0.25`
- `bbox_inches='tight', pad_inches=0.15` on savefig
- Chart descriptions updated for accuracy:
  - Avg Reward: "Positive = winning more than losing"
  - Clip Fraction: "Values <0.01 are typical with masked multi-head actions"
  - Approx KL: ">0.05 = aggressive updates"
  - Advantage Std: "<0.01 = all actions look equally good"

### No implementation changes made yet for the main improvements.

## Decisions Made

### Root Cause Analysis
The bot's plateau is caused by multiple reinforcing issues:
1. **No GAE** — every step in a round gets the same reward (round_score/10). The network can't learn which specific actions were good. This is the primary bottleneck.
2. **Network too shallow** — 475→256→128 (2 hidden layers) is insufficient for the strategic depth needed.
3. **LR too high** — 1e-3 is 3-4x standard PPO (3e-4). Too coarse for refinement past the easy wins.
4. **Entropy bonus static at 0.04** — high, but user prefers manual tuning via config+resume over automatic annealing.

### Architecture Decisions
- **Flexible `layer_sizes` parameter** instead of hardcoded `hidden_size`/`first_hidden_size`. Old bots = `[256, 128]`, new = `[512, 256, 256, 128, 128, 128]`. This lets different bot architectures coexist.
- **ResidualBlock** for same-width consecutive layers (the 256→256 and 128→128→128 segments).
- **No entropy annealing** — user prefers manual control. Run, observe charts, adjust config, resume.
- **LR annealing** — linear decay from 3e-4 to 0 over total iterations (standard PPO practice).
- **Multi-seat training** — assign training network to 3 of 4 seats instead of just P0. 3x training data per game. Config param `training_seats: 3`.
- **Eval against old bots** — load previous best checkpoints as eval opponents, track separate margin lines alongside random baseline.

## Implementation Plan

### network.py
- `ScoutNetwork.__init__` takes `layer_sizes` list instead of `hidden_size`/`first_hidden_size`
- Build shared layers dynamically. Regular `Linear+ReLU` for width changes, `ResidualBlock` (linear + ReLU + skip) for same-width consecutive layers
- Store `layer_sizes` as attribute for checkpoint serialization
- Backward compat: detect old config format and convert `hidden_size`/`first_hidden_size` → `layer_sizes=[first_hidden, hidden]`
- Heads use `layer_sizes[-1]` as input width (unchanged externally)
- `RandomBot` needs no changes

### training.py
- **Reward restructuring in `play_game()`** (line 64-68): only last step per player per round gets `round_score/10`. All intermediate steps get reward=0.
- **New `compute_gae(records, gamma, lam)`** replacing `compute_advantages()` (line 262). Groups records by `(round_num, player)`. For each group, walks backward computing TD errors: `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)` where V(s_{t+1}) comes from next record's `.value` field (0 at round end). Accumulates GAE advantages. Returns `(advantages, returns)` where returns = advantages + values (used as value targets).
- **`ppo_update()`** (line 303): new `returns` parameter. Use `returns` for value targets instead of `s.reward` (line 324).
- **Multi-seat training**: `play_game()` assigns training network to first `training_seats` players, rest from opponent pool.

### main.py
- **Config changes**:
  - `layer_sizes: [512, 256, 256, 128, 128, 128]` (replaces `hidden_size`/`first_hidden_size`)
  - `learning_rate: 3e-4` (was 1e-3)
  - `training_seats: 3` (new)
  - `gamma: 0.99` (new, for GAE)
  - `gae_lambda: 0.95` (new, for GAE)
  - `eval_opponents: {}` (new, dict of name→checkpoint_path for eval)
  - Remove `replay_buffer_size` (unused)
  - Keep commented-out old values for reference
- **LR annealing**: `lr = initial_lr * (1 - iteration / total_iterations)` before each PPO update
- **ScoutNetwork construction**: pass `layer_sizes` from config
- **Eval expansion**: load eval opponent checkpoints at startup (detecting architecture from saved config — old format uses `[first_hidden_size, hidden_size]`, new format uses `layer_sizes`). Play eval games against each. Track separate metrics per opponent. Chart shows multiple eval lines.
- **Opponent pool serialization**: save `(layer_sizes, state_dict)` per member. Load by constructing right architecture per member. Fall back to current architecture for old-format pools.

## Files to Read First
- `scout-bot/network.py` — current network architecture (small file, ~116 lines)
- `scout-bot/training.py` — play_game, compute_advantages, ppo_update
- `scout-bot/main.py` — config, training loop, chart generation

## Watch Out
- **Start fresh run with new save_dir** — don't resume from old checkpoint. Architecture mismatch would fail.
- **S&S actions** produce 2 records for same player in same turn (scout + forced play via recursive `_play_turn` call at line 213). GAE handles these as consecutive steps — no special case needed.
- **`networks[r.player] is network`** identity check (training.py:75) already correctly filters to training-network seats. Multi-seat training works with this filter as-is.
- The opponent pool's `load_state_dicts()` (line 253) deep-copies a template and loads state dicts — needs to handle per-member architecture for mixed pools.
- `advantage_std` metric in main.py (line 276) computed from raw `reward - value` — needs updating to use GAE advantages instead.
