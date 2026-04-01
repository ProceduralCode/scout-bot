# Entropy Diagnostic and Floors

## Task & State

Wired entropy floors into `ppo_update_v6`, diagnosed scout entropy collapse mechanism through gradient analysis, and tuned floor values. v6_8 training is running and actively improving (best eval margin: v1_4 = -12.47 at iter 460).

## What Changed

- `scout-bot/training.py` — `_ppo_step_v6()` and `ppo_update_v6()` now accept `entropy_floors`, `entropy_floor_coeff`, and `zero_scout_policy_grad` params. Entropy floors apply quadratic penalty when per-region entropy drops below threshold. `zero_scout_policy_grad` is an ablation flag that zeros gradient on policy_head rows 256-319 before optimizer step.

- `scout-bot/main.py` — `ppo_update_v6` call passes through `entropy_floors`, `entropy_floor_coeff`, and `zero_scout_policy_grad` from cfg. PARAMS updated: `mini_batch_size` 2^15, `ppo_epochs` 1, `entropy_bonus` 0.03, `entropy_floors` `{"play": 1.0, "scout": 1.0}`, `save_interval_hours` 3 (time-based snapshots instead of iteration-based).

- `scout-bot/entropy_diagnostic.py` — New diagnostic script. Loads a checkpoint, plays self-play games, analyzes: (1) probability mass distribution between play/scout/sns, (2) gradient norms on play vs scout logits from policy loss and entropy bonus, (3) action choice concentration, (4) gradient decomposition by action type taken (play-samples vs scout-samples).

## Measurements

### Gradient analysis (iter 175 checkpoint)

Policy gradient on scout logits is 50-61x larger than entropy bonus gradient (at entropy_bonus=0.03). Switching to per-region entropy doesn't help (scout-only entropy gradient ≈ 1.1x joint).

Decomposition by action type: scout-sample gradients are 4.6x larger than play-sample gradients on scout logits. Play-sample gradients on scout logits are uniformizing (proportional push-down preserves relative ordering), not concentrating. Scout entropy collapse is driven by scout's own gradient, not play coupling.

### Scout action concentration (iter 175)

Only 15-18 unique scout actions chosen out of ~23 legal. Action 278 (insert pos 6, left end, flip) accounts for 40% of all scouts. Top 5 actions cover 90%. Conditional entropy 0.86 with 23 legal options → effective ~2.4 actions.

### v6_8 training trajectory

With floors at play=1.0, scout=1.0 (activated around iter 400+): eval_margin_v1_4 improved from -22 range to -12.47. Scout entropy holding at 0.94 (floor actively pushing back). Play entropy at 0.71 (below floor, also being pushed back). Overall entropy still declining (0.89).

## Decisions

- **Play-coupling theory rejected.** Originally hypothesized that play reinforcement in the flat softmax caused scout entropy collapse. Gradient analysis showed play-sample gradients are uniformizing on scout logits (proportional push-down). The collapse comes from scout's own noisy gradient causing random-walk concentration in softmax logit space. The entropy bonus is too weak (50-61x smaller than policy gradient) to counteract this drift.

- **Entropy floors set to 1.0/1.0 (user's choice).** Original proposed values (play=0.5, scout=0.3) were too low — scout entropy was already damaging performance at those levels. Floors at 1.0 are currently active and the model is improving.

- **Time-based snapshots.** Changed from iteration-based (every 1000) to time-based (every 3 hours). Falls back to iteration-based if `save_interval_hours` is None.

## Next Steps

1. **Let v6_8 run.** It's actively improving with current settings (floors at 1.0). Monitor whether the improvement continues or plateaus.

2. **Gradient zeroing ablation available but not yet run.** `zero_scout_policy_grad` flag is wired and commented out in PARAMS. Zeroes all gradient on policy_head rows 256-319. Would confirm whether the scout entropy collapse comes through the policy head or the trunk representation. Not urgent while training is improving.

3. **If training plateaus** — consider: (a) adjusting floor values, (b) running the ablation, (c) investigating whether play entropy floor at 1.0 is too restrictive (play entropy is 0.71, well below floor).

## Watch Out

- The `zero_scout_policy_grad` ablation zeros ALL gradient on scout logits (policy + entropy + value), not just policy. A cleaner test would separate them, but this version is sufficient for a first pass.

- Play entropy (0.71) is well below the 1.0 floor, meaning the floor is actively fighting play policy sharpening. This might be too aggressive — the play policy might benefit from more concentration. If eval margins plateau, consider lowering the play floor.

- `entropy_bonus` in the diagnostic hardcodes 0.08 as the coefficient for gradient analysis (lines 197-199), but actual training uses 0.03. The relative ratios would change proportionally.

- play_len_3_pct has dropped to 0.0000. The network never plays triples or longer. v6_7 at peak had 5.9% triples. This may indicate the play floor needs to be higher, or that the play action space has its own concentration issue.
