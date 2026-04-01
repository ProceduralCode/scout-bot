# Bridging Probe, Play-Length Signal, and Compute Discovery

## Task & State

Built two new diagnostic probes and discovered that CUDA has never been enabled — all training has been CPU-only. GPU (RTX 3060 6GB) is present and CUDA drivers are installed, but PyTorch was installed as CPU-only build. Reinstalled with `cu124` but benchmarking showed GPU is **slower** for this workload due to the small model size and per-turn CPU↔GPU transfer overhead.

## What Changed

### New files
- `scout-bot/bridging_probe.py` — Tests whether GAE can propagate credit through multi-step episodes. Finds mid-game states with divergent play/scout rollout values, forces good/bad first actions, plays episodes to completion, runs `compute_gae()`, checks if decision-1 advantage has the right sign. Results on v7_3 (iter 217): 70% sign accuracy, 0.28 GAE-rollout correlation.
- `scout-bot/play_length_probe.py` — Measures rollout-based advantage (v_after - v_before) by play length. Pure ground truth, no value head involved. Results on v7_3: singles +0.04 margin advantage, pairs +0.08, correlation with length 0.17. Signal exists but is small relative to noise (std ~0.18).

### PyTorch reinstall
- Reinstalled torch with CUDA: `pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu124`
- Now `torch 2.5.1+cu124`, `torch.cuda.is_available() == True`
- GPU benchmark was slower (172s vs 113s per iteration) due to small model + transfer overhead
- No code changes to use GPU were kept — model stays on CPU

### network.py
- No net changes (device-handling lines were added then reverted)

## Decisions

- **GPU not used**: RTX 3060 is available but doesn't help — model is ~500K params, forward passes are tiny, game sim + encoding is the real bottleneck. CPU-GPU transfer overhead dominates.
- **Bridging probe design**: Forces play vs scout pairs specifically (not arbitrary action pairs). Uses rollout ground truth for screening, GAE for measurement. Single episode per (state, action) pair — noisy but sufficient for aggregate statistics.
- **Play-length probe**: v_after - v_before via rollouts, grouped by play length. Confirmed playing is a positive-value action on average, pairs slightly better than singles, but effect size is small.

## Next Steps

The session ended mid-discussion about a fundamental tension:

1. **Rollouts give reliable signal but are slow** — 25 rollout games take ~106s (94% of iteration time). At ~2min/iter, 10,000 iterations = ~14 days.
2. **GAE is fast but unreliable** — at iter 217, only 70% sign accuracy on credit assignment. The value head isn't good enough for the bootstrapping chain to work.
3. **The per-decision signal is genuinely small** — pairs are +0.08 margin better than singles, with 0.18 std. SNR ~0.4.
4. **Random-policy rollouts** were considered as a cheaper alternative but rejected — they'd only teach obvious strategies, and the same bootstrapping wall would appear at the next skill level.

The open question: is the training approach fundamentally viable at this compute budget, or does something structural need to change? Options discussed but not decided:
- Batch `play_games_v6` across games (currently sequential, one game at a time)
- Reduce rollouts_per_state (20 → 5-10)
- Accept the pace and let it train
- Rethink reward structure (more information per game than just terminal margin)

## Watch Out

- **PyTorch version downgraded**: Was `2.10.0+cpu`, now `2.5.1+cu124`. Some dependency versions also changed (typing-extensions 4.15→4.9, sympy 1.14→1.13.1, jinja2 3.1.6→3.1.3). Watch for compatibility issues.
- **Rollout timing data**: GAE 75 games = 6.7s, rollout 25 games = 105.9s (CPU). The rollout cost scales with decisions_per_game × rollouts_per_state.
- **v7_3 is the only checkpoint compatible with current FlatScoutNetwork** (GELU + LayerNorm + detached value head). v6_x checkpoints fail to load due to old value head structure.
