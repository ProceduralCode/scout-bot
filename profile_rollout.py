"""Profile rollout_multi_action_v6 to find the bottleneck."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from pyinstrument import Profiler
from training import play_games_q_v6, rollout_multi_action_v6
from network import FlatScoutNetwork
from encoding import INPUT_SIZE_V6

cfg = {
	"num_players": 4,
	"training_seats": 4,
	"temperature": 0.0,
	"epsilon": 0.05,
	"rollout_actions_per_sample": 10,
	"rollout_actions_random_extra": 2,
	"rollouts_per_action": 30,
	"rollout_temperature": 1.0,
}

network = FlatScoutNetwork(INPUT_SIZE_V6, [256, 128],
	encoding_version=6, attention={"dim": 32, "heads": 2, "layers": 1})

# Load trained weights if available
ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots", "q_v1", "latest.pt")
if os.path.exists(ckpt_path):
	ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
	network.load_state_dict(ckpt["model_state"])
	print("Loaded checkpoint weights")

network.cuda()
network.eval()

# Warmup: play + rollout one small batch to JIT-compile Numba kernels
print("Warmup...")
warmup = play_games_q_v6(network, 10, 4, training_seats=4, temperature=0.0, epsilon=0.05)
rollout_multi_action_v6(warmup, network, 4,
	rollout_actions_per_sample=5, rollout_actions_random_extra=1,
	rollouts_per_action=5, rollout_temperature=1.0)
torch.cuda.synchronize()

# Collect real samples
print("Playing 100 games...")
samples = play_games_q_v6(network, 100, 4, training_seats=4,
	temperature=0.0, epsilon=0.05)
print(f"  {len(samples)} samples collected")

pairs_count = sum(
	min(cfg["rollout_actions_per_sample"], sum(s.action_mask)) + cfg["rollout_actions_random_extra"]
	for s in samples)
total_games = pairs_count * cfg["rollouts_per_action"]
print(f"  ~{pairs_count} action pairs, ~{total_games} rollout games expected")

# Profile the rollout
print("Profiling rollout...")
torch.cuda.synchronize()
profiler = Profiler()
profiler.start()
t0 = time.time()

rollout_multi_action_v6(
	samples, network, 4,
	rollout_actions_per_sample=cfg["rollout_actions_per_sample"],
	rollout_actions_random_extra=cfg["rollout_actions_random_extra"],
	rollouts_per_action=cfg["rollouts_per_action"],
	rollout_temperature=cfg["rollout_temperature"],
)

torch.cuda.synchronize()
elapsed = time.time() - t0
profiler.stop()

print(f"\nRollout took {elapsed:.1f}s")
print(f"Throughput: {total_games / elapsed:.0f} rollout games/sec")
print()
print(profiler.output_text(unicode=False, color=False))

# Save HTML profile
out_dir = os.path.dirname(os.path.abspath(__file__))
html_path = os.path.join(out_dir, "rollout_profile.html")
with open(html_path, "w") as f:
	f.write(profiler.output_html())
print(f"HTML profile saved to {html_path}")
