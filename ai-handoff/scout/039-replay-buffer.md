# 039 — Replay Buffer

## Task

Added a replay buffer to the PPO training loop. Keeps the last N iterations of batch data and concatenates them for PPO updates. PPO's importance sampling naturally handles stale data — old samples where the policy has drifted get clipped to zero.

## What Changed

- `scout-bot/training.py` — added `concatenate_batches()` function (line ~1100). Takes a list of batch dicts from `prepare_ppo_batch`, `torch.cat`s all tensors, offsets sub-head index tensors (`play_idx`, `end_idx`, `scout_idx`) by cumulative batch sizes, and re-normalizes advantages across the combined data. Handles batches with different sub-head compositions (e.g., one batch has no scout actions).

- `scout-bot/main.py` — added `replay_buffer_size` param (default 5, user set to 20), `from collections import deque` import, `concatenate_batches` import. Before training loop: creates `deque(maxlen=replay_buffer_size)`. After `prepare_ppo_batch`: appends current batch, concatenates all buffered batches for PPO. Skipped for direct PG mode. Log line shows `steps=400(7800)` when buffer active. Epoch-0 ratio warning suppressed when buffer > 1 (stale data shifts mean ratio by design).

## Decisions

- Re-normalize advantages when concatenating — each iteration's advantages are independently normalized to zero mean/unit std, so concatenation preserves this approximately, but re-normalizing is more correct.
- Single-batch passthrough (`len(batches) == 1` returns original dict) avoids unnecessary tensor copies.
- No explicit `.detach()` needed — `_play_turn` wraps all forward passes in `torch.no_grad()`, so logits in StepRecords are already leaf tensors.
- Replay buffer not used with direct PG (no importance sampling to handle staleness).

## Current Config (v5_5)

User adjusted alongside the buffer: `replay_buffer_size=20`, `ppo_epochs=8` (down from 32), `learning_rate=0.0003`.

## Open Thread

User wants per-age clip fraction diagnostics — how much of the older buffer data is actually contributing gradient vs being fully clipped. Two approaches discussed:

1. **Per-age breakdown** — tag each batch with iteration number, compute clip fraction per age band after ppo_update. More granular, shows where the cliff is.
2. **New-vs-old split** — just report clip fraction for the newest batch vs the rest. Simpler, answers "is old data helping?"

Not implemented yet. User invoked handoff before choosing.
