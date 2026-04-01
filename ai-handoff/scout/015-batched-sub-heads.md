# Scout: Batched Sub-Head Forward Passes

## Task

Batch the sub-head linear layer calls (action_type, play_start, play_end, scout_insert) during game generation, which were the bottleneck at 87% of batched game gen time due to per-game PyTorch dispatch overhead.

## What Changed

### Modified Files

- **`scout-bot/network.py`**
  - Added `batched_masked_sample(logits, mask)` — Gumbel-max sampling for [B, C] batches, returns [B] LongTensor. Placed next to existing `masked_sample`.

- **`scout-bot/training.py`**
  - Import updated to include `batched_masked_sample`.
  - `play_games_batched()` turn loop rewritten for training-network games: instead of calling `_process_turn_from_hidden` per-game, the turn loop now batches all sub-head calls. Structure: batch action_type → partition into play/scout groups → batch play_start+play_end for play group → batch scout_insert for scout+sns group → per-game mutations and StepRecord construction.
  - Opponent-network games still use `_process_turn_from_hidden` (different networks, can't batch).
  - Sub-head logits called directly (`network.action_type_head(cond)`) bypassing the per-game wrapper methods, using existing `_build_batch_conditioning` for per-element one-hot conditioning.

## Performance

- **2.2x speedup** over per-game `play_game()` path (was 1.3x with shared-only batching).
- Neural network calls dropped from dominant cost to **4.8%** of game gen time.
- New bottleneck is Python game logic: encoding 21.6%, mask computation 19%, `get_legal_plays` 9.6%, self-time (loops/list building) 17.3%.

## Decisions

- **Direct head calls instead of wrapper methods** — the `network.action_type_logits()` etc. methods use `_build_conditioning` with scalar conditioning, which doesn't support per-element batching. Calling `network.action_type_head()` directly with `_build_batch_conditioning` matches the pattern already used in `ppo_update`.
- **StepRecord logits are views into batch tensors** — not standalone tensors like the per-game path. Functionally identical (same values, same shapes when indexed). The batch tensor stays alive via Python reference counting as long as any view exists.
- **Kept `_process_turn_from_hidden`** for the opponent path rather than deleting it.

## LR Exploration

User tested learning rates during this session:
- 0.003 — actively worse (too high)
- 0.001 — still worse
- 0.0003 — stable baseline
- 0.0001 — same learning rate as 0.0003, no benefit
- 0.0006 — next to try (binary search midpoint between 0.0003 and 0.001)

## Next Steps

- **Try LR 0.0006** — if stable, doubles effective learning rate over 0.0003. Watch clip_fraction (>0.1 = warning, >0.2 = unstable) and approx_kl (>0.02 = unstable).
- **Profile with production network** — the 2.2x measurement used [256, 128, 128]. The production [512, 256, 256, 128, 128, 128] network may show different ratios since shared forward is heavier.
- **Further optimization** — remaining bottleneck is Python (encoding, masks, get_legal_plays). These are per-game game-state-dependent computations, harder to batch. Possible approaches: Cython/C for hot loops, or restructuring mask computation.

## Watch Out

- **`_process_turn_from_hidden` still exists** and is used by the opponent path. If it ever needs changes, the batched turn loop needs matching changes (they implement the same game logic differently).
- **pyinstrument** is installed and useful for profiling — `PYTHONPATH=scout-bot python -c "import pyinstrument; ..."` works.
