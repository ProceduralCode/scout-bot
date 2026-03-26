# V6 Code Review — Parallel Agents

Run this review by spawning one agent per file group, all in parallel. Each agent should be a `general-purpose` agent.

Before spawning, run `git log --oneline -20` in `scout-bot/` to find the commit range for v6 changes. Use that range in the diff commands below (replace `BEFORE_V6` with the last pre-v6 commit hash).

Spawn 5 agents in parallel with these prompts. Each prompt is self-contained — paste it exactly.

---

## Agent 1: game.py

```
You are reviewing code changes to scout-bot/game.py for a v6 implementation of a Scout card game bot.

STEP 1 — Read these files for review guidelines:
- ai-context/shared/workflows/code/review.md
- ai-context/ProceduralCode/library/code/commenting-style.md
- ai-context/shared/library/code/preserve-comments.md
- ai-context/shared/library/code/code-changes.md
- ai-context/shared/library/code/code-documentation.md

STEP 2 — Read the design spec and implementation plan:
- scout-bot/plan.md (section: "Game Engine — Configurable Card Values")
- scout-bot/v6-implementation.md (section: "Pass 1 — game.py")

STEP 3 — Read the full current file:
- scout-bot/game.py

STEP 4 — Run: git -C scout-bot diff BEFORE_V6 -- game.py
(This shows exactly what changed. Replace BEFORE_V6 with the commit hash.)

STEP 5 — Review against these questions:
- Do the changes match the spec? Anything missing or extra?
- Were any existing comments removed or made stale?
- Were any existing behaviors broken (backward compat)?
- Is the code as clean and simple as it could be?
- Edge cases: does create_deck work correctly for all player counts with parameterized num_values?
- Does get_state_for_player include everything v6 encoding needs?

Report all findings. Do not make any changes.
```

---

## Agent 2: encoding.py

```
You are reviewing code changes to scout-bot/encoding.py for a v6 implementation of a Scout card game bot.

STEP 1 — Read these files for review guidelines:
- ai-context/shared/workflows/code/review.md
- ai-context/ProceduralCode/library/code/commenting-style.md
- ai-context/shared/library/code/preserve-comments.md
- ai-context/shared/library/code/code-changes.md
- ai-context/shared/library/code/code-documentation.md

STEP 2 — Read the design spec and implementation plan:
- scout-bot/plan.md (sections: "New Encoding", "Flat Output Head", "Rotation Augmentation — Precomputed permutation tables")
- scout-bot/v6-implementation.md (sections: "Pass 1 — encoding.py", "Pass 3 — encoding.py")

STEP 3 — Read the full current file:
- scout-bot/encoding.py

STEP 4 — Run: git -C scout-bot diff BEFORE_V6 -- encoding.py
(Replace BEFORE_V6 with the commit hash.)

STEP 5 — Review against these questions:
- Do constants (INPUT_SIZE_V6=301, FLAT_ACTION_SIZE=384, etc.) match the spec's formulas?
- Hand encoding: N one-hot + empty + top scalar per slot, plus H bottom scalars. Correct dims?
- Scout cards: 4 options × (N+1) dims, correct face values for normal/flipped, absent flags?
- Play buffers: left-aligned and right-aligned 4-card buffers, correct indexing for plays > 4 cards?
- Metadata: 28 dims, forced_play flag present, player_count as scalar (not one-hot)?
- get_flat_action_mask: play region [0-255], scout [256-319], S&S [320-383] with per-position legality?
- decode_flat_action: correct inverse of the mask encoding?
- Permutation tables: PLAY_PERM, SCOUT_PERM, FULL_PERM, HAND_SHIFT — correct shift logic? Is inverse via FULL_PERM[(16-k)%16] valid?
- Were any existing comments removed or made stale by the additions?
- Were any existing functions or constants accidentally modified?
- Does the Cython override at the bottom still work correctly with the new code above it?

Report all findings. Do not make any changes.
```

---

## Agent 3: network.py

```
You are reviewing code changes to scout-bot/network.py for a v6 implementation of a Scout card game bot.

STEP 1 — Read these files for review guidelines:
- ai-context/shared/workflows/code/review.md
- ai-context/ProceduralCode/library/code/commenting-style.md
- ai-context/shared/library/code/preserve-comments.md
- ai-context/shared/library/code/code-changes.md
- ai-context/shared/library/code/code-documentation.md

STEP 2 — Read the design spec and implementation plan:
- scout-bot/plan.md (sections: "Flat Output Head — Network Interface", "RandomBot")
- scout-bot/v6-implementation.md (sections: "Pass 1 — network.py")

STEP 3 — Read the full current file:
- scout-bot/network.py

STEP 4 — Run: git -C scout-bot diff BEFORE_V6 -- network.py
(Replace BEFORE_V6 with the commit hash.)

STEP 5 — Review against these questions:
- FlatScoutNetwork: same trunk builder pattern as ScoutNetwork? Correct policy_head size (384)?
- Does it store encoding_version=6 as attribute?
- Value head: Linear(output_size, 1) — same as other networks?
- RandomBot: policy_logits returns zeros [384]? Old methods preserved?
- Were any existing comments removed or made stale?
- Were any existing classes (ScoutNetwork, CircularCNNScoutNetwork) accidentally modified?
- Is FlatScoutNetwork placed in a sensible location in the file (near related code)?
- Import at the top: FLAT_ACTION_SIZE added correctly?

Report all findings. Do not make any changes.
```

---

## Agent 4: training.py

```
You are reviewing code changes to scout-bot/training.py for a v6 implementation of a Scout card game bot. This is the largest and most critical file.

STEP 1 — Read these files for review guidelines:
- ai-context/shared/workflows/code/review.md
- ai-context/ProceduralCode/library/code/commenting-style.md
- ai-context/shared/library/code/preserve-comments.md
- ai-context/shared/library/code/code-changes.md
- ai-context/shared/library/code/code-documentation.md

STEP 2 — Read the design spec and implementation plan:
- scout-bot/plan.md (sections: "S&S Two-Step Flow", "Training Pipeline")
- scout-bot/v6-implementation.md (sections: "Pass 2", "Pass 3 — training.py")

STEP 3 — Read the full current file (it's large, read in chunks):
- scout-bot/training.py

STEP 4 — Run: git -C scout-bot diff BEFORE_V6 -- training.py
(Replace BEFORE_V6 with the commit hash.)

STEP 5 — Review against these questions:

Existing code preservation:
- Were any existing comments removed or made stale?
- Were any existing functions modified beyond the minimal dispatch additions (v6 branch in _play_round, _play_turn)?
- Do the dispatch additions (elif ev == 6) follow the existing pattern?

StepRecordV6:
- Has all needed fields? Matches spec?

_play_turn_v6:
- Single forward pass → flat logits → masked sample → decode → apply?
- S&S: recursive call for forced play? forced_play flag set correctly?
- old_log_prob computed via masked_log_prob (not just from masked_sample)?
- Edge case: no legal actions → _advance_turn?

play_games_with_rollouts_v6:
- S&S produces 2 separate snapshot pairs (not shared like old code)?
- reuse_before_snap correctly avoids duplicate snapshot between S&S steps?
- Rollout aggregation: per-snapshot, per-player margins, same formula as old version?
- Advantage = V_after - V_before for each record's player?
- record.value = rollout estimate from before_snap?

rollout_from_states_batched_v6:
- Batched forward pass, handles S&S inline (not recursive)?
- No-action games advance turn correctly?

PPO functions (prepare, subsample, concatenate, update):
- prepare_ppo_batch_v6: v_target uses record.value when returns is None?
- ppo_update_v6: single log_ratio computation, no sub-head routing?
- Conditional entropies computed correctly (mask to region, compute entropy)?
- No entropy floors (unlike old ppo_update)?

augment_rotation_v6:
- 15 augmented copies per sample (shifts 1..15)?
- State shifted via HAND_SHIFT gather?
- Action permuted via FULL_PERM forward?
- Mask permuted via FULL_PERM inverse?
- Batched forward pass for correct old_log_probs?
- Same advantages and value targets for augmented copies?

OpponentPool.load_state_dicts:
- v6 branch instantiates FlatScoutNetwork correctly?

Report all findings. Do not make any changes.
```

---

## Agent 5: main.py + probe.py + matchup.py

```
You are reviewing code changes to scout-bot/main.py, scout-bot/probe.py, and scout-bot/matchup.py for a v6 implementation of a Scout card game bot.

STEP 1 — Read these files for review guidelines:
- ai-context/shared/workflows/code/review.md
- ai-context/ProceduralCode/library/code/commenting-style.md
- ai-context/shared/library/code/preserve-comments.md
- ai-context/shared/library/code/code-changes.md
- ai-context/shared/library/code/code-documentation.md

STEP 2 — Read the design spec and implementation plan:
- scout-bot/plan.md (sections: "Training Pipeline", "Eval Opponents", "Diagnostics")
- scout-bot/v6-implementation.md (section: "Pass 3 — main.py")

STEP 3 — Read the full current files:
- scout-bot/main.py
- scout-bot/probe.py
- scout-bot/matchup.py

STEP 4 — Run these diffs (replace BEFORE_V6 with the commit hash):
- git -C scout-bot diff BEFORE_V6 -- main.py
- git -C scout-bot diff BEFORE_V6 -- probe.py
- git -C scout-bot diff BEFORE_V6 -- matchup.py

STEP 5 — Review against these questions:

main.py PARAMS:
- Were any existing PARAMS values changed (not just new ones added)?
- Are v6 defaults correct per spec (encoding_version:6, use_rollouts:True, augment_rotations:16, entropy_floors:None, save_dir:"v6")?
- Were any comments in PARAMS removed or made stale?

main.py training loop:
- v6 branch: play_games_with_rollouts_v6 → augment_rotation_v6 → prepare/subsample/concatenate/ppo_update all v6 versions?
- original_records saved before augmentation for behavioral metrics?
- Metrics logging: v6 uses entropy_play/entropy_scout instead of per-head entropies?
- Action type classification uses flat index ranges (0-255, 256-319, 320+)?

main.py charts:
- Auto-detects v6 (checks entropy_play existence) vs old (per-head entropies)?
- entropy_floor_penalty panel still present (shows nothing for v6, that's fine)?

main.py eval:
- Eval opponent loading: v6 branch instantiates FlatScoutNetwork?
- play_eval_game works with mixed network types (v6 training + old eval opponents)?

main.py general:
- Were any existing comments removed or made stale?
- Were any non-v6 code paths accidentally modified?
- Any imports added that aren't used, or missing imports?

probe.py:
- v6 branch in _sample_scout: samples from scout region only (256-319)?
- Behavioral change: old code always scouts left-normal, v6 samples any scout action. Is this intentional/acceptable?
- Were any existing comments or code modified?

matchup.py:
- v6 branch in load_agent: FlatScoutNetwork with correct input_size and encoding_version?
- Were any existing comments or code modified?

Report all findings. Do not make any changes.
```
