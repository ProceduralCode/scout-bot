"""Dump key metrics trajectories from a checkpoint's metrics_history."""
import sys, torch

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
mh = ckpt["metrics_history"]
iters = mh.get("iteration", [])

if not iters:
    print("No metrics_history recorded.")
    sys.exit(0)

print(f"Iterations recorded: {len(iters)} (range {iters[0]}–{iters[-1]})")
print()

def summary(name, vals):
    if not vals:
        print(f"  {name}: (empty)")
        return
    n = len(vals)
    early = vals[:5]
    late = vals[-5:]
    print(f"  {name} ({n} pts)")
    print(f"    early: {['%.4f' % v for v in early]}")
    print(f"    late:  {['%.4f' % v for v in late]}")
    if n >= 10:
        mid_start = n // 2 - 2
        mid = vals[mid_start:mid_start + 5]
        print(f"    mid:   {['%.4f' % v for v in mid]}")

# Per-head entropy
print("=== PER-HEAD ENTROPY ===")
for key in ["entropy_action_type", "entropy_play_start", "entropy_play_end", "entropy_scout_insert"]:
    summary(key, mh.get(key, []))
print()

# PPO health
print("=== PPO HEALTH ===")
for key in ["clip_fraction", "approx_kl", "explained_variance"]:
    summary(key, mh.get(key, []))
print()

# Behavioral
print("=== BEHAVIOR ===")
for key in ["play_pct", "scout_pct", "sns_pct", "steps_per_game", "advantage_std"]:
    summary(key, mh.get(key, []))
print()

# Loss
print("=== LOSS ===")
for key in ["policy_loss", "value_loss", "entropy", "reward", "value"]:
    summary(key, mh.get(key, []))
print()

# Eval
print("=== EVAL ===")
eval_iters = mh.get("eval_iteration", [])
eval_margin = mh.get("eval_margin", [])
if eval_iters:
    print(f"  eval points: {len(eval_iters)}")
    for k in sorted(mh.keys()):
        if k.startswith("eval_margin"):
            vals = mh[k]
            if vals:
                summary(k, vals)
