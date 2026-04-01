# Scout: V4 Training Launch, Robustness, and Probe V2 Support

## Task & State

Launched first v2 training (v4_1), monitored progress, made robustness improvements to the training loop, then re-ran scout diagnostic probes with v2 encoding. Partway through adding v2 support to `probe_diagnostic.py` when session ended — the remaining `ScoutNetwork(layer_sizes=...)` replacements and `--v2` CLI flag in that file are not yet done.

## What Changed

- **`scout-bot/main.py`** — Robustness improvements:
  - `_save_checkpoint`: writes to `.tmp` file first, then `os.replace` with 5 retries on OSError (Windows file locking fix — error code 32 crashed v4_1 at iter 8189)
  - `fig.savefig`, `summary.txt` write, `log.save`: wrapped in try/except OSError with warning
  - Eval block: wrapped in try/except Exception, cleans up partial metrics on failure
  - NaN guard: checks `math.isnan(r.value)` on iteration records before PPO update, skips iteration if NaN detected
  - Added `import math`
  - User set `games_per_iteration` to 400 (from 100) to restore v1-equivalent batch sizes for single-round v2 training

- **`scout-bot/probe.py`** — Full v2 support:
  - `--v2` CLI flag sets `ENCODING_VERSION = 2`
  - `_make_network()` helper creates ScoutNetwork with correct input_size, head sizes, encoding_version
  - `_encode()` dispatches to v1/v2 encoding based on `ENCODING_VERSION`
  - `_sample_play()` uses v2 mask sizes and slot counts
  - `_sample_action_type()` passes `max_hand` for v2
  - `_train_iteration()` passes `play_start_size` for v2
  - Probe 8 (frozen trunk): loads encoding_version from checkpoint config

- **`scout-bot/probe_diagnostic.py`** — Partial v2 support (IN PROGRESS):
  - Added v2 imports, `ENCODING_VERSION` global, `_ev_hand_slots()`/`_ev_insert_size()`/`_ev_input_size()` helpers
  - Updated `_encode()`, `_sample_scout()`, `_find_best_adjacent_slots()`, `_eval_adj_rate()`, `_make_network()`, `_train_iteration()`
  - Test A (`test_supervised`) updated to use `_make_network`
  - **NOT YET DONE**: remaining test functions (B, C, D, E) still use `ScoutNetwork(layer_sizes=...)` directly. Tests F, G, S use v1 constants inline (INPUT_SIZE, SCOUT_INSERT_SIZE, encode_state) and need v2 paths. `--v2` CLI flag not added to `main()`.

## Decisions

- **400 games/iter for v2** — v2 single-round produces ~1/4 the steps of v1 multi-round (4 players × 4 rounds → 1 round). Bumping from 100→400 games restores the original batch size. The LR was tuned for that batch size, so this avoids retuning.

- **Retry-based checkpoint save vs crash** — Windows file locking (antivirus/indexer) is unavoidable. Atomic write (tmp + rename) with retry is the standard fix. Non-critical files (charts, summary, game log) just warn and continue.

## Next Steps

1. **Finish v2 support in `probe_diagnostic.py`** — replace remaining `ScoutNetwork(layer_sizes=...)` with `_make_network(layer_sizes)` in tests B-E. Add v2 encoding paths to tests F, G, S (standalone MLP, MLP scout head, rotation sweep). Add `--v2` flag to CLI.

2. **Run diagnostic tests with v2** — specifically Test A (supervised CE), Test F (standalone MLP), Test H (standalone MLP fixed ho=0), and a v2 ho-only sweep. The key question: does v2 (no po rotation, only ho rotation over 15 slots) actually improve supervised learning of adjacent matching? The old sweep showed random-ho + fixed-po = 0.826 with v1 encoding, but v2 is a different layout.

3. **Probe results so far** — `probe.py` probes 5, 5b, 9 ran with `--v2`: same pattern as v1 (trivial PASS, quality/adjacent FAIL). But these use PPO with small batches (50 games × 100 iters). The diagnostic tests (supervised CE, standalone MLP) are the higher-signal experiments.

## Watch Out

- v4_2 is actively training (8000+ iters, beating v3_4). Don't modify files it depends on while it's running.
- `probe_diagnostic.py` tests F, G, S construct MLPs and encode states inline — they don't use `_sample_scout` or `_make_network`. Each needs its own v2 path with `INPUT_SIZE_V2`, `SCOUT_INSERT_SIZE_V2`, `encode_state_v2`.
