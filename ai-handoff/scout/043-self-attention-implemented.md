# Self-Attention Architecture Implemented

## Task & State

Implemented the self-attention architecture specified in handoff 042. All code changes are complete and verified — probes 0-9 pass.

## What Changed

- `encoding.py` — `SCOUT_CARDS_DIM_V6` 44→52, `INPUT_SIZE_V6` 301→309. Added `GLOBAL_START_V6` (260), `GLOBAL_DIM_V6` (49), `NUM_ENTITIES_V6` (20). `_fill_scout_cards_v6` writes 8 new scalars (top/bottom per scout option). `encode_state_v6` uses `SCOUT_CARDS_DIM_V6` constant for offset.
- `network.py` — `FlatScoutNetwork.__init__` takes optional `attention` dict. When present: registers `entity_indices` buffer [20,13], `position_onehots` [1,20,20], builds `entity_proj` Linear(33→d_model), pre-norm `MultiheadAttention` layers. Forward gathers entities, concatenates position one-hots, projects, runs attention, flattens, appends global features (x[:,260:]), then FC trunk. Stores `attention_cfg` attribute for serialization.
- `main.py` — `PARAMS["attention"]` added. Passed to `FlatScoutNetwork` constructor. Preserved from checkpoint on resume (like `layer_sizes`). Passed for eval opponent loading.
- `matchup.py` — Passes `attention` from checkpoint config for v6 networks.
- `training.py` — `OpponentPool.state_dicts()` saves `attention_cfg`. `load_state_dicts()` passes it when constructing v6 networks.
- `probe_v6.py` — `_make_network()` passes `{"dim": 32, "heads": 2, "layers": 1}`.

## Next Steps

1. Start fresh training with the new architecture (old v6 checkpoints are incompatible due to encoding size change 301→309 and different FC trunk input size).
2. After some training iterations, re-run `test_scout_generalization.py` test 4 (adjacent value matching) to check if the relational reasoning gap is closed.
3. Monitor whether scout insertion quality (probe metric, scout_play_len chart) improves compared to previous FC-only training runs.

## Watch Out

- Old v6 checkpoints cannot load — `INPUT_SIZE_V6` changed and FC trunk input size changed. Pre-v6 checkpoints (v1-v4) are unaffected.
- `test_scout_generalization.py` still constructs `FlatScoutNetwork` without attention. It's a scratch file and was intentionally left unmodified per the spec.
