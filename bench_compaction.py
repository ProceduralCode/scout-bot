"""Benchmark rollout with active-game compaction vs baseline timing."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from training import play_games_q_v6, rollout_multi_action_v6, attach_snapshots
from network import FlatScoutNetwork
from encoding import INPUT_SIZE_V6

network = FlatScoutNetwork(INPUT_SIZE_V6, [256, 128],
	encoding_version=6, attention={"dim": 32, "heads": 2, "layers": 1})
ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots", "q_v1", "latest.pt")
if os.path.exists(ckpt_path):
	ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
	network.load_state_dict(ckpt["model_state"])
network.cuda()
network.eval()

# Warmup
print("Warmup...")
warmup, warmup_replays = play_games_q_v6(network, 5, 4, training_seats=4, temperature=0.0, epsilon=0.05)
attach_snapshots(warmup, warmup_replays)
rollout_multi_action_v6(warmup, network, 4,
	rollout_actions_per_sample=3, rollout_actions_random_extra=1,
	rollouts_per_action=5, rollout_temperature=1.0, chunk_pairs=64)
torch.cuda.synchronize()

# Collect samples
print("Playing 100 games...")
samples, game_replays = play_games_q_v6(network, 100, 4, training_seats=4,
	temperature=0.0, epsilon=0.05)
attach_snapshots(samples, game_replays)
print(f"  {len(samples)} samples")

# Run rollout with compaction
print("\nRollout with active-game compaction...")
for s in samples:
	s.rolled_actions = None
	s.rollout_margins = None
	s.rollout_stds = None
torch.cuda.synchronize()
t0 = time.time()
rollout_multi_action_v6(
	samples, network, 4,
	rollout_actions_per_sample=10,
	rollout_actions_random_extra=2,
	rollouts_per_action=30,
	rollout_temperature=1.0,
)
torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"  Time: {elapsed:.1f}s")

# Verify results are valid
n_valid = sum(1 for s in samples if s.rollout_margins is not None)
margins = [m for s in samples if s.rollout_margins for m in s.rollout_margins]
print(f"  Samples with rollouts: {n_valid}/{len(samples)}")
print(f"  Margin range: [{min(margins):.3f}, {max(margins):.3f}]")
print(f"  Margin mean: {sum(margins)/len(margins):.3f}")
