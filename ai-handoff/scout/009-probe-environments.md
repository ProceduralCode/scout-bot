# Scout Bot: Probe Environments & Entropy Floor

## Task
Diagnose why training plateaus at ~50 iterations. The agent learns to prefer playing over scouting quickly but plays mostly 1-2 card plays and never learns longer sequences.

## What We Found

### Diagnostic Data (v2_4/latest.pt, 63 iterations)
- **play_end head collapsed**: normalized entropy 0.016, gradient norm 0.002 (50x smaller than other heads). The head was nearly deterministic from iteration 1 (raw entropy 0.138) and only got worse.
- **91% of play_end steps have exactly 1 legal option** (confirmed by sampling 50 games with the trained model). Only 8.9% of steps have 2+ options.
- **Filtered entropy** (2+ option steps only): fresh network = 0.709, trained model = 0.057. The head collapsed on the rare steps where it actually had a choice.
- Reward drop at ~iteration 35 is expected — that's when `snapshot_interval` adds a new opponent to the pool.
- PPO barely updating by end: clip_fraction 0.005, policy_loss -0.0005.

### Probe Environments Confirmed Architecture is Sound
Built `scout-bot/probe.py` with 4 probes testing subsystems in isolation:

| Probe | Result (200 iters, 100 games) | What it tests |
|---|---|---|
| 1. Value head | PASS (0.00 → 1.03) | Can the value head learn a constant return? |
| 2a. play_start (simple) | PASS (0.10 → 0.79) | Can a conditioned head learn a simple preference? |
| 2b. play_start (complex) | PASS (0.10 → 0.18) | Can it learn a complex state-dependent preference? |
| 3. play_end (longest) | PASS (0.34 → 0.76) | Can play_end learn to prefer longer plays? |
| 4. Full chain | PASS (1.19 → 1.34) | Can start+end jointly learn to maximize play length? |

Key finding: all probes pass with sufficient budget (200 iters, 100 games). With insufficient budget (50 iters, 30 games), only value head passes — policy heads need more signal. The play_end head is not architecturally broken; it just collapses before accumulating enough reward signal in real training.

Why entropy floors (not higher LR): once the head goes deterministic, it only samples one action, so advantages are ~0 after normalization. The gradient vanishes regardless of LR. Higher LR would actually accelerate collapse during early training by making the initial push toward one action stronger.

## What Changed

### New Files
- `scout-bot/probe.py` — Probe environments for isolating NN subsystems. Usage: `python probe.py [--iters N] [--games N]`. Uses real `Game` objects for valid state encodings but assigns synthetic rewards to test specific heads.
- `scout-bot/inspect_metrics.py` — Quick script to dump metrics_history trajectories from a checkpoint. Usage: `python inspect_metrics.py <checkpoint_path>`.

### Modified Files
- `scout-bot/training.py` — `ppo_update()` (line 329) gains `entropy_floors: dict[str, float] | None` and `entropy_floor_coeff: float` params. Computes quadratic penalty `coeff * max(0, floor - mean_ent)²` per head when filtered mean entropy (steps with 2+ legal options) drops below the floor. Returns `entropy_floor_penalty` in metrics dict. Fully backward compatible (floors=None by default).
- `scout-bot/main.py` — `DEFAULT_CONFIG` (line 27) gets `entropy_floors` dict and `entropy_floor_coeff: 1.0`. Floor values: action_type=0.1, play_start=0.1, play_end=0.3, scout_insert=0.1. Passed through to `ppo_update` in the training loop. `metrics_history` tracks `entropy_floor_penalty`. Save dir changed to `v2_5`.

## Decisions
- **Per-head entropy floors over higher LR**: The root cause is exploration collapse, not slow optimization. Once a head goes deterministic, gradients vanish regardless of LR.
- **Quadratic penalty (separate from entropy bonus)**: Soft floor with smooth gradients. Only fires when entropy drops below floor. The existing entropy bonus continues to reward exploration generally.
- **Filter to steps with 2+ legal options**: The penalty only applies to steps where the head actually has a choice. With 1 legal option, entropy is 0 by definition and unimprovable.
- **play_end floor = 0.3**: For 2-option steps (the majority of multi-option cases), this forces roughly a 65/35 split minimum. Aggressive but necessary — the head needs to explore alternatives to get differential reward signal.

## Next Steps

1. **Run training with entropy floors** — `python main.py` (save_dir already set to v2_5). Watch whether play_end entropy stabilizes above the floor and whether the agent learns longer plays.
2. **Monitor floor penalty** — If `entropy_floor_penalty` stays high for many iterations, the floor might be too aggressive and fighting the policy gradient. If it drops to 0 quickly, the head found a way to maintain entropy naturally.
3. **Tune floor values** — 0.3 for play_end is a first guess. May need adjustment based on how training behaves.

## Watch Out
- The 91% single-option stat is from the trained model playing short. A model that learns longer plays should see more multi-option play_end steps (positive feedback loop).
- `probe.py` doesn't use entropy floors — it tests the raw architecture. To test floors in isolation, you'd need to pass `entropy_floors` to `_train_iteration`'s `ppo_update` call.
- The floor penalty computation uses `torch.clamp` and goes through the compute graph, so it properly contributes gradients to the play_end head weights.

## Files to Read First
- `scout-bot/training.py:329-473` — ppo_update with entropy floor implementation
- `scout-bot/main.py:15-51` — DEFAULT_CONFIG with floor settings
- `scout-bot/probe.py` — probe environments (if modifying or extending probes)
- `ai-handoff/scout-nn-debugging-research.md` — the research doc that motivated this work
