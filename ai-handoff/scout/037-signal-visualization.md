# Signal Visualization

## Task & State

Built a visualization tool (`scout-bot/visualize_signal.py`) to see the actual rollout values flowing through the training pipeline. The goal was to get a concrete feel for the numbers rather than theorize. The result reframed the scout head problem from a learning problem to a signal problem.

## What Changed

- `scout-bot/visualize_signal.py` — New diagnostic tool. Two sections:
  - **Section 1**: For N scoutable states, runs rollouts at every insert position at 5/10/20/50 rollout counts. Produces bar charts showing value by position with error bars. Charts saved as `signal_positions_stateN.png`.
  - **Section 2**: Plays a full game with the checkpoint, computes rollout-based V_before and V_after for every action, shows raw and normalized advantages. Chart saved as `signal_game_advantages.png`.
  - Usage: `python scout-bot/visualize_signal.py --checkpoint PATH [--seed N] [--states N] [--section 1 2]`
  - Has a try/except for S&S actions that fail the legal-play-after-scout assertion (falls back to regular scout).

## Key Observations

### Position-level signal at different rollout counts (3 states tested, v4_2 checkpoint)

- **5 rollouts** (what training uses): position rankings are essentially random. In State 1, the "best" position at 5 rollouts became one of the worst at 50. Error bars overlap completely across all positions.
- **50 rollouts**: real signal emerges. State 3 showed a 0.5+ margin gap between best and worst positions (scouting a 4 next to an existing 4). State 2 showed a 9-T run being the best position — the signal captures runs, not just pairs.
- At 5 rollouts, the advantage for a scout action contains no usable information about insert position quality.

### Full game advantage distribution (50 actions, 20 rollouts/snapshot)

- Scout and play advantages are completely intermixed in magnitude and distribution. No systematic difference.
- Play advantages: mean +0.010, std 0.087, range [-0.16, +0.25]
- Scout advantages: mean -0.024, std 0.136, range [-0.35, +0.25]
- Normalization doesn't preferentially squash scout advantages — they look the same as play advantages.

## Decisions

- Dropped the "good position vs bad position" framing from the first version of the script. The user correctly pointed out this was projecting assumptions and also missing runs (only considering same-value adjacency). Rewrote to show all positions without labels and let the data speak.

## Next Steps

1. The reframing: the scout head training problem is a signal-to-noise problem, not an architecture/encoding/normalization problem. At 5 rollouts (or with the value head at explained_variance=0.28), the advantage contains no position-quality information.
2. The open question is how to get position-quality information into the training loop. Candidates discussed:
   - Increase rollout count (expensive, still mixes "was scouting right?" with "was position right?")
   - Compare multiple positions per scout decision (try K positions, rollout each, train toward better ones — bypasses advantage pipeline)
   - Use raw V_after instead of advantage (removes V_before subtraction noise, still needs sufficient rollouts)
3. The user expressed interest in bypassing PPO/advantage entirely for the scout head. No code changes made toward this yet.

## Watch Out

- `visualize_signal.py` generates states by playing 1-8 random turns (mix of plays and random scouts). States are more varied than `probe_scout_signal.py` which only did 1 play from a fresh round.
- The S&S fallback uses a bare `except` on `AssertionError` (Python typo for `AssertionError` — but it was modified by a linter to work, so it catches the right thing now). Check if this needs fixing.
