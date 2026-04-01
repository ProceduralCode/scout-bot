# Scout: V2 Encoding Implementation Complete

## Task & State

Implemented the v2 encoding redesign planned in session 022. All code changes are complete and smoke-tested. No training run started yet — ready to launch v4_1.

## What Changed

- **`scout-bot/network.py`** — `ScoutNetwork` constructor takes `play_start_size`, `play_end_size`, `scout_insert_size`, `encoding_version` (defaults preserve v1 behavior). `_build_conditioning` uses `self.play_start_size`. `RandomBot` takes the same params.

- **`scout-bot/training.py`** — Full v2 dispatch:
  - `play_game`: single-round + random starting player for v2 networks
  - `_play_round`: per-network encoding dispatch in flip phase
  - `_play_turn` / `_process_turn_from_hidden`: per-network dispatch via `getattr(net, 'encoding_version', 1)`, all mask/decode calls pass v2 sizes via `_hs`/`_sis` aliases
  - `play_games_batched`: local aliases `_hs/_sis/_pss`, v2 game setup (`total_rounds=1`, random `starting_player`), per-network encoding in flip+turn phases
  - `OpponentPool.state_dicts/load_state_dicts`: saves/loads `encoding_version`
  - `_build_batch_conditioning`: takes `play_start_size` param
  - `ppo_update`: takes `play_start_size` param, threads through all conditioning calls

- **`scout-bot/main.py`** — PARAMS: `encoding_version: 2`, `save_dir: "v4_1"`, added `v3_4` to eval opponents. Network construction branches on encoding_version. Eval opponent loading detects encoding_version from checkpoint. Resume preserves encoding_version from checkpoint (like layer_sizes). `ppo_update` called with `play_start_size=network.play_start_size`.

- **`scout-bot/probe.py`** — `_sample_scout` dispatches on encoding version so `eval_scout_quality` works with v2 networks. Other probes remain v1-only.

- **`scout-bot/matchup.py`** — `load_agent` detects encoding_version from checkpoint config.

## Decisions

- **`play_start_size` threaded explicitly** rather than inferred from network inside `ppo_update` — keeps the function's contract clear (it takes a batch dict, not a network to inspect).

- **Per-network dispatch in `_play_turn`** rather than a global encoding_version parameter — each network in a mixed-version eval game encodes using its own version. The `play_games_batched` training path uses the training network's version for the batched path, and `_process_turn_from_hidden` dispatches per-opponent.

- **`play_offset` kept in `_process_turn_from_hidden` signature** even though it's unused — avoids changing the call site tuple structure. v2 callers pass 0.

- **`_encode` and `_encode_flip` aliases** set at top of `play_games_batched` but only `_encode_flip` is actually unused — v1 encoding needs `po` passed inline so the alias doesn't help. Could clean up later.

## Next Steps

1. Launch v4_1 training run: `python scout-bot/main.py`
2. Watch first few iterations for crashes or NaN
3. After ~100 iterations, check eval margins against v3_4 — v4_1 starts from scratch so it should be losing initially
4. After ~1000 iterations, check if scout_play_len improves (the whole point of v2)

## Watch Out

- The unused `_encode` alias in `play_games_batched` (line 394) is dead code. Harmless but could confuse a reader.
- Probes (except `eval_scout_quality`) still hardcode v1. Running `python probe.py --probe 5` with a v2 network would produce wrong-sized tensors.
- `ev` variable in `train()` is set before resume. If PARAMS encoding_version differs from checkpoint, state dict load fails (shape mismatch) — correct behavior, but the error message won't say "encoding version mismatch."
