# Scout Bot: Shaped Rewards, Offset Fix, and Reward Distribution

## Task
Diagnose why the bot plateaus after learning basic "don't be random" behavior, experiment with shaped rewards, fix scout_insert coordinate mismatch, and configure reward system for v3_4 overnight run.

## What Changed

### Modified Files

- **`scout-bot/training.py`**
  - **`StepRecord`** (line 43): added `scout_quality: int | None = None` field — longest legal play containing an inserted scout card.
  - **`play_game()` signature** (lines 45-51): added `reward_mode`, `shaped_bonus_scale` params. `reward_distribution` now accepts float 0-1 for hybrid terminal/uniform split.
  - **Reward assignment** (lines 75-107): three reward modes:
    - `"play_length"`: per-play reward = `play_length / 5.0`, scouts get 0.
    - `"play_and_scout"`: plays get `play_length / 5.0`, scouts get `(scout_quality - 1) / 8.0`.
    - `"game_score"`: original score-delta logic, now with hybrid distribution support — `reward_distribution` can be `"terminal"`, `"uniform"`, or a float (uniform fraction, e.g. `0.7` = 70% uniform / 30% terminal).
  - **Shaped bonus** (lines 102-107): after main reward, optionally adds `play_length / 5.0 * shaped_bonus_scale` per play and `(scout_quality - 1) / 8.0 * shaped_bonus_scale` per scout. Applied regardless of reward_mode if scale > 0.
  - **Scout_insert offset fix** (lines 210-213, 235-238): scout and S&S paths now use `get_scout_insert_mask(game, hand_offset)`, sample in slot space, decode to game coordinates via `(insert_slot - hand_offset) % SCOUT_INSERT_SIZE`. `rec.scout_insert` stores the slot value (for PPO ratio computation).
  - **Scout quality computation** (lines 222-227, 246-251): after scouting, constructs post-insertion hand and finds longest legal play containing the inserted card position. Computed for both scout and S&S paths.
  - **Import** (line 12): added `SCOUT_INSERT_SIZE`.

- **`scout-bot/encoding.py`**
  - **`get_scout_insert_mask()`** (lines 364-371): now takes `hand_offset` param, applies offset to mask positions via `(hand_offset + pos) % SCOUT_INSERT_SIZE`.
  - **`get_sns_insert_mask()`** (lines 374-389): same offset fix.

- **`scout-bot/interactive.py`**
  - (lines 201-204): updated to pass `hand_offset` to mask functions and decode insert_slot back to game coordinates.
  - (line 12): added `SCOUT_INSERT_SIZE` import.

- **`scout-bot/probe.py`**
  - (line 534): updated `_sample_scout` to pass `ho` to `get_scout_insert_mask` and decode slot to game coordinates.

- **`scout-bot/main.py`**
  - **Config** (lines 35-37, 47): `reward_mode: "game_score"`, `reward_distribution: 0.7` (70% uniform, 30% terminal), `shaped_bonus_scale: 0.05`, `save_dir: "v3_4"`.
  - **`play_game` call** (lines 306-310): passes `reward_mode`, `shaped_bonus_scale` from config.

## Decisions

- **Shaped rewards alone failed**: `play_length` mode (v3_1) got 0.73 explained variance but avg_play_length stuck at 1.57. `play_and_scout` mode (v3_3) caused the agent to over-scout — it learned that scouting = more cards = more future plays = more total reward. Steps/game climbed from 160 to 200+, eval margins went negative. Zero-basing scout reward to `(quality-1)/8` wasn't enough.
- **Offset fix was a real bug**: scout_insert masks were in game coordinates while play_start/play_end used slot space (with hand_offset). The scout_insert head was operating in a different coordinate system from the state encoding. All previous runs had this bug.
- **Hybrid game_score + tiny shaped bonus**: game_score provides the "did you win?" signal that keeps the agent honest. Small per-action bonus (0.05 scale) nudges toward longer plays and better scouting without creating perverse accumulation incentives, because game_score dominates.
- **70/30 uniform/terminal split**: uniform spreads win/loss signal to all steps; terminal adds emphasis on end-of-round states. Configurable as a float.
- **Entropy floors lowered to 0.05**: play_end floor at 0.3 was actively constraining the head that controls play length. All floors set to 0.05 as safety net.

## Key Findings

### Training Run Results
- **v2_6** (game_score, uniform): 241 iters, EV 0.17, losing to v2_5. Uniform reward was worse than terminal.
- **v3_1** (play_length): 417 iters, EV 0.73, avg_play_length 1.57, scout_play_len 1.54. Value function learned but policy didn't improve play quality.
- **v3_3** (play_and_scout): 504 iters, EV 0.85, but agent gamed the reward — scout_pct climbed to 26%, steps/game to 202, eval margins went strongly negative. Getting worse at winning while getting more reward.

### Core Diagnosis
The network has capacity (probes proved it) but faces two bottlenecks:
1. **Credit assignment**: end-of-round score across ~40 decisions with hidden information caps value function quality. Shaped rewards fixed the signal but created perverse incentives.
2. **Representation**: cross-position card matching in FC layers is hard. The scout_insert offset bug made it impossible; with the fix, it's merely difficult. 91% of play_end steps have only 1 legal option, so play length is mostly determined by hand structure (which depends on scout insertions).

## Next Steps

1. **Run v3_4 overnight**: `python scout-bot/main.py` — game_score + offset fix + shaped bonus + hybrid distribution + lower entropy floors. This is the cleanest test of whether the offset fix + reward improvements enable learning.
2. **Monitor**: scout_play_len (does it rise above ~1.5?), explained_variance (how high?), eval_margin trends (does it beat v2_5?), steps_per_game (should stay ~160, not climb).
3. **If v3_4 plateaus like v2_5**: the representation bottleneck is likely real. Consider architectural help — match hints in state encoding, attention over hand positions, or CNN for adjacency patterns.

## Watch Out

- **Old bot compatibility**: the offset fix changes scout_insert coordinate space. Old bots (v1_x, v2_x) will have randomized scout decisions when played against, since their weights were trained in game coordinates. Their play_start/play_end and action_type still work. Eval margins against old bots may shift slightly but old bots never learned meaningful scouting anyway.
- **`reward_distribution` type changed**: now accepts float in addition to strings. Config saved in checkpoints will have `0.7` instead of `"terminal"` or `"uniform"`. Old checkpoints still work (string values handled).
- **Shaped bonus applies to all reward modes**: if `shaped_bonus_scale > 0`, the bonus is added on top regardless of `reward_mode`. Currently only intended for use with `game_score` mode.

## Files to Read First
- `scout-bot/training.py:75-107` — reward assignment logic (all three modes + shaped bonus)
- `scout-bot/training.py:207-255` — scout/S&S paths with offset fix and scout_quality computation
- `scout-bot/encoding.py:364-389` — offset-aware mask functions
- `scout-bot/main.py:35-37` — current config (v3_4)
