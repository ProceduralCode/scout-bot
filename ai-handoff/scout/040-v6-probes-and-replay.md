# V6 Probes and Replay Buffer Fix

## Task & State

Built a comprehensive probe suite for the v6 flat-action-space network (`probe_v6.py`), and changed `replay_past` from absolute sample counts to fractions of current batch size.

All probes pass. Training is runnable with the v6 PARAMS config (save_dir `v6_1`).

## What Changed

- **Created `scout-bot/probe_v6.py`** — 10 probes validating the v6 pipeline end-to-end:
  - 0: encoding dimensions, mask validity, encode→decode round-trip, offset invariance
  - 1: forward pass shapes, no NaNs, masked softmax
  - 2: value head learns constant return (-0.17 → 1.01)
  - 3: flat head learns play-over-scout preference (11% → 86%)
  - 4: PPO batch pipeline produces gradients, loss decreases
  - 5: rotation augmentation mask counts and decoded actions match
  - 6: full game loop (`play_games_with_rollouts_v6`) produces valid records + PPO runs
  - 7: S&S mask entries round-trip through decode→legality check (both true and false entries)
  - 8: flat head learns to prefer longer plays within play region (1.22 → 1.95)
  - 9: rotation hand slot content verifiably shifts to correct positions

- **Modified `scout-bot/main.py`** — `replay_past` changed from `[200, 100, 50, 50, 50]` to `[0.4, 0.2, 0.1, 0.1, 0.1]`. Fractions are relative to `batch["n"]` (post-augmentation). Both v6 and non-v6 paths updated.

- **PARAMS consolidated** — the two PARAMS blocks were merged into one with v6 as active config (encoding_version=6, use_rollouts=True, augment_rotations=16, save_dir=v6_1).

## Decisions

- Separate `probe_v6.py` rather than extending `probe.py` — the sub-head infrastructure is too different to share.
- `replay_past` fractions are relative to post-augmentation batch size (user's explicit preference). With ~8000 augmented records, 0.4 = 3200 from previous iteration.

## Next Steps

- Run v6 training for 50-100 iterations and observe value loss, entropy, clip fraction, eval scores.
- Watch ADV-DIAG output for scout advantage signal quality.

## Watch Out

- First iteration takes 2+ minutes (expected with rollouts + 16x augmentation).
- The old v2 PARAMS block no longer exists as a separate block — it was merged into one active config.
