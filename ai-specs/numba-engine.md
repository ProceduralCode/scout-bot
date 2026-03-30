# Numba CUDA Rollout Engine

## Goal

Replace the PyTorch tensor-op GPU engine (`gpu_engine.py`) with Numba CUDA kernels for game logic. Target: 10,000–50,000 games/s (vs current 270 games/s CPU Cython, 960 games/s PyTorch GPU compiled).

## Why

The current rollout pipeline is bottlenecked by Python being in the loop — not by compute. A game of Scout takes ~3μs of actual work in C. The current CPU path achieves ~4ms per game (1,300x overhead from Python ↔ Cython transitions). The PyTorch GPU path has ~100 tensor ops per step, each with 10–50μs of Python dispatch overhead, capping throughput at ~960 games/s regardless of batch size.

### Benchmarks that led here

| Approach | B=5,000 | Throughput | Bottleneck |
|:---------|:--------|:-----------|:-----------|
| CPU Cython (single process) | 19.9s | 270 games/s | Sequential, Python loop |
| GPU PyTorch (torch.compile) | 6.1s | 820 games/s | Python dispatch per tensor op |
| GPU PyTorch (B=50,000) | 52.1s | 960 games/s | Same — dispatch overhead plateaus |
| CPU multiprocessing (8 workers) | 7.3s | 682 games/s | Pickle serialization, thermal |

None of these approaches break 1,000 games/s. The fundamental problem is that Python orchestrates every operation. Numba CUDA kernels eliminate Python from the game logic entirely — one kernel launch replaces 40–60 tensor ops.

## Architecture

```
Python loop (100 steps):
  ├─ compute_legal_plays_kernel()    ← Numba: 1 launch replaces ~40 tensor ops
  ├─ compute_action_masks_kernel()   ← Numba: 1 launch replaces ~60 tensor ops
  ├─ encode_states_kernel()          ← Numba: 1 launch replaces ~50 tensor ops
  ├─ network.forward() + policy_logits()  ← PyTorch (stays as-is)
  ├─ batched_masked_sample()         ← PyTorch (stays as-is)
  └─ apply_actions_kernel()          ← Numba: 1 launch replaces ~50 tensor ops
```

~6 dispatches per step instead of ~200+. At 100 steps: 600 dispatches × ~50μs = 30ms overhead. The rest is actual compute.

### Data flow

All game state lives as PyTorch CUDA tensors in `GpuGameState` (reused from `gpu_engine.py`). Numba reads/writes them via `numba.cuda.as_cuda_array()` — zero-copy, wraps the pointer. The encoded output is a pre-allocated PyTorch tensor that the Numba kernel writes into, then goes directly to `network()`. No copies between Numba and PyTorch.

```
GpuGameState (PyTorch CUDA tensors)
       │
       ▼ as_cuda_array() — zero-copy wrap
  Numba kernels read/write state
       │
       ▼ encode output: PyTorch tensor written by Numba
  network(encoded) → logits
       │
       ▼ batched_masked_sample → actions (PyTorch tensor)
       │
       ▼ as_cuda_array()
  apply_actions_kernel writes state
```

### Thread model

One CUDA thread per game. With B=5,000: 5,000 threads. Each thread handles all logic for its game — iterating over hand cards, checking plays, building masks, encoding, applying actions. Inner loops are small (hand size ≤ 16, at most 136 start/end pairs for legal plays).

## What stays, what changes

### Reused from gpu_engine.py (not in hot loop)
- `GpuGameState` dataclass — unchanged
- `from_snapshots()` — runs once at start, converts Python Games to GPU state
- `compute_scores()` — runs once at end, extracts scores

### Rewritten as Numba kernels (new file: `numba_engine.py`)
- `compute_legal_plays` → `compute_legal_plays_kernel`
- `encode_states` → `encode_states_kernel`
- `compute_action_masks` → `compute_action_masks_kernel`
- `apply_actions` → `apply_actions_kernel`
- `rollout_gpu` → `rollout_numba`

### Unchanged (PyTorch)
- `FlatScoutNetwork.forward()` / `policy_logits()`
- `batched_masked_sample()`
- Hand offset generation (`torch.randint`)

## Kernel Specifications

### Constants

```
H = 16         # hand slots
MAX_P = 5      # max players
MAX_PLAY = 16  # max play length
N = 10         # card values (1-10)
FLAT_ACTION_SIZE = 384
```

### compute_legal_plays_kernel

**Signature:** `(hands_show, hand_len, current_player, play_len, play_type, play_strength, done, num_players, out_legal, B)`

**Thread mapping:** `b = cuda.grid(1)`, one thread per game.

**Algorithm per thread:**
1. If `done[b]`: zero the output row, return
2. Get current player's hand: `cp = current_player[b]`, iterate `vals[0..hand_len-1]`
3. For each `(start, end)` pair where `0 <= start <= end < hand_len`:
   - Check if contiguous subarray is a set (all equal), ascending run (+1), or descending run (-1)
   - If valid type, check if it beats the current play (longer, or equal length with higher strength / set-beats-run)
   - Write `out_legal[b, start, end] = True/False`

**Output:** `[B, H, H]` bool

### encode_states_kernel

**Signature:** `(state fields..., hand_offsets, out_encoded, B)`

**Thread mapping:** one thread per game.

**Algorithm per thread:** Compute all 309 output dimensions directly:
1. **Hand top face (192 dims):** For each slot 0..15: determine which hand position maps here (via `hand_offsets`), write one-hot of showing value + empty flag + scalar. Uses a local array `slot_map[H]` to track position → slot mapping.
2. **Hand bottom (16 dims):** Hiding value / N for each slot.
3. **Scout cards (52 dims):** 4 card choices × (N+1 one-hot + 2 scalars). Read play endpoints, compute top/bottom for normal/flipped variants.
4. **Play buffer (21 dims):** Left-aligned and right-aligned views of play cards (4 each × 2 values), plus play type one-hot, strength, length.
5. **Metadata (28 dims):** Hand lengths, collected cards, scout tokens, sns availability for all players (relative to current); num_players, scouts_since_play, owner relative position one-hot, forced flag.

**Output:** `[B, 309]` float32

### compute_action_masks_kernel

**Signature:** `(state fields..., legal_plays, hand_offsets, out_mask, B)`

**Thread mapping:** one thread per game.

**Algorithm per thread:**
1. **Play region [0..255]:** For each `(s, e)` in `legal_plays[b]`, compute slot indices `s_slot = (hand_offset + s) % H`, `e_slot = (hand_offset + e) % H`, set `out_mask[b, s_slot * H + e_slot] = True`.
2. **Scout region [256..319]:** If game has a play and hand has room and not in SNS_PLAY phase: for each of 4 card choices, for each valid insert position `p` (0..hand_len), compute `ins_slot = (hand_offset + p) % H`, set `out_mask[b, 256 + c * H + ins_slot] = True`.
3. **S&S region [320..383]:** Same gating as scout, plus `sns_available`. For each `(card_choice, insert_pos)`: build hypothetical hand in a local array (insert scouted card at position), compute reduced play (remove scouted card from current play), check if hypothetical hand has any legal play against reduced play (nested loop over start/end pairs — reuses legal play logic as a device function). If yes, set mask bit.

**The S&S inner loop** is the most expensive part: 4 card choices × up to 17 insert positions × up to 136 start/end pairs = ~9,248 iterations per game. With B=5,000 threads, the GPU handles this fine.

**Output:** `[B, 384]` bool

### apply_actions_kernel

**Signature:** `(state fields..., actions, hand_offsets, active, B)`

**Thread mapping:** one thread per game.

**Algorithm per thread:**
1. If `!active[b]`: return
2. Decode action: `< 256` = play, `256–319` = scout, `320–383` = S&S
3. **Play:** Extract cards from hand at `[start..end]`, determine play type/strength, remove from hand (shift left), collect old play's cards, check round end (empty hand)
4. **Scout:** Determine scouted card (left/right end, normal/flipped), insert into hand at position (shift right), remove from play, award scout token, check round end (scouts_since_play)
5. **S&S:** Same as scout for the card insertion, but set phase to SNS_PLAY, mark sns_available = false for current player
6. Update current_player (advance turn for play/scout, stay for S&S), update phase, update done flag

All writes go directly to state tensors. No intermediate allocations.

## Testing Strategy

Each kernel is tested against the corresponding `gpu_engine.py` PyTorch function:
1. Generate diverse game states via `from_snapshots` (various player counts, game stages, with/without current play, SNS_PLAY phase, etc.)
2. Run both the PyTorch function and the Numba kernel on the same input
3. Assert outputs match exactly (bool) or within tolerance (float32 encoding)

The existing `test_gpu_*.py` files already generate good test fixtures — reuse the same snapshot generation.

## Expected Performance

At B=5,000 and 100 steps:

| Component | Per-step estimate | Total (100 steps) |
|:----------|:------------------|:-------------------|
| 4 Numba kernel launches | ~200μs | 20ms |
| Game logic compute (5K threads) | ~100μs | 10ms |
| Network inference | ~2ms | 200ms |
| Sampling | ~100μs | 10ms |
| **Total** | **~2.5ms** | **~250ms** |

5,000 games in ~250ms = **20,000 games/s** (~75x over CPU Cython).

At B=50,000: network inference dominates, ~2.5s total = **20,000 games/s** (throughput plateaus at network inference speed, not game logic).

## Implementation Order

1. **Spike** — verify Numba CUDA ↔ PyTorch tensor interchange works, measure kernel launch overhead
2. **compute_legal_plays_kernel** + test
3. **encode_states_kernel** + test
4. **compute_action_masks_kernel** + test (hardest — S&S logic)
5. **apply_actions_kernel** + test
6. **rollout_numba** — wire loop, benchmark
7. **Integration** — hook into training.py as rollout backend

Each step is independently testable against the existing reference implementation.

## Files

- `scout-bot/numba_engine.py` — all Numba kernels + `rollout_numba` entry point
- `scout-bot/test_numba_*.py` — per-kernel test files
- `scout-bot/bench_numba.py` — benchmark script
- `scout-bot/gpu_engine.py` — unchanged, serves as reference implementation and test oracle
