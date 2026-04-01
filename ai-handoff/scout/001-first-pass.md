# Scout Neural Network Bot — First Pass Implementation

## Task

Implement all five files for a Scout card game neural network bot from `Scout/design.md`. Actor-critic architecture trained via self-play (PPO) to play the card game Scout (3-5 players).

## Current State

All five files written and reviewed. Game engine stress-tested (600+ random games, all player counts). PyTorch code is syntactically valid but untested at runtime — torch is not installed in this environment.

## Files Created

- `Scout/__init__.py` — empty, makes Scout a package
- `Scout/game.py` — Game engine: card/deck management, play validation, scout/S&S mechanics, round/game scoring, reward calculation
- `Scout/encoding.py` — State-to-tensor (463 floats), mask generation for all output heads, hand offset randomization
- `Scout/network.py` — ScoutNetwork (nn.Module): shared backbone (256→128 ReLU), conditioned policy heads, value head, masked sampling helpers
- `Scout/training.py` — Self-play game runner, step recording, PPO with per-step gradient accumulation, replay buffer, opponent pool
- `Scout/main.py` — Training loop orchestration, CLI args, periodic logging/saving

## Key Decisions

- **S&S edge case**: Scout & Show can result in no legal plays after scouting (~15% of the time under random play). Fixed by degrading to a regular scout — S&S token consumed, scout happens, but no play made. See `game.py:226-244`.

- **State rotation**: `get_state_for_player` rotates all per-player arrays so index 0 = requesting player, opponents in clockwise order. This was a bug fix — the encoding assumed index 0 = "you" but the original code returned absolute indices. See `game.py:307-327`.

- **Hand offset sentinel**: `encode_state` uses `hand_offset=None` for random offset (training augmentation) and explicit int for fixed offset. Originally used 0 as sentinel which caused a 1/20 misalignment bug between encoding and masks.

- **Opponent record filtering**: `play_game` only returns records from players using the training network (`networks[r.player] is network`). Without this, PPO ratios would be computed between different networks' logits.

- **Gradient accumulation**: `ppo_update` does per-step `.backward()` scaled by `1/n` instead of building one graph across all steps. Gradient clipping at `max_norm=0.5`.

## What's Deferred (per design doc's open questions)

- Hidden layer sizes: set to 256/128, configurable
- PPO hyperparameters: standard defaults, all configurable
- Replay buffer size: 500 games default
- Opponent pool management: simple FIFO, snapshot every 10 iterations
- Auxiliary prediction heads: not implemented
- GAE / per-round discounting: TODO in `training.py:187`, using simple reward−value

## Next Steps

1. Install PyTorch: `pip install torch`
2. Run a short training test: `python -m Scout.main --iterations 10 --games-per-iter 5`
3. Debug any runtime issues in the encoding/network/training pipeline
4. Once training runs, observe training curves and tune hyperparameters
5. Consider adding GAE for better credit assignment

## Watch Out

- `get_legal_plays` is called multiple times per turn (by `get_legal_action_types`, `get_legal_play_starts`, `get_legal_play_ends`). No caching — O(n²) per call with n≤20, so fine for now but could be optimized if profiling shows it's hot.
- The `_do_scout` method mutates `self.current_play.cards` in place via `.pop()`. Works because the Play object is immediately replaced, but would break if anyone held a reference to the old Play.
- Single-card plays on the table: scout right options are masked (only left offered) to avoid duplicate actions. The network sees 1-2 fewer options for single-card plays.

## How to Verify

```bash
# Game engine (no torch needed)
python -c "from Scout.game import Game, Phase; g = Game(4); g.start_round(); print('OK')"

# Full pipeline (needs torch)
python -m Scout.main --iterations 10 --games-per-iter 5
```
