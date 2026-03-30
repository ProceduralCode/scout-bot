"""Smoke test: verify play_games_with_rollouts_v6 works with gpu_rollout=True."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import time
from encoding import INPUT_SIZE_V6
from network import FlatScoutNetwork
from training import play_games_with_rollouts_v6

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_integration():
	net = FlatScoutNetwork(input_size=INPUT_SIZE_V6, layer_sizes=[128, 64], attention=None)

	# GPU path
	if DEVICE == 'cuda':
		print("Running GPU rollout path...")
		t0 = time.time()
		records_gpu, advantages_gpu, std_gpu = play_games_with_rollouts_v6(
			net, num_games=3, num_players=4,
			rollouts_per_state=5, training_seats=4,
			gpu_rollout=True)
		t_gpu = time.time() - t0
		print(f"  GPU: {len(records_gpu)} records, {len(advantages_gpu)} advantages, "
			  f"margin_std={std_gpu:.4f}, time={t_gpu:.2f}s")
		# Verify network is back on CPU
		param = next(net.parameters())
		assert not param.is_cuda, "Network should be back on CPU"
		print("  Network confirmed back on CPU after GPU rollout")
	else:
		print("SKIP: no CUDA device")
		return True

	# CPU path for comparison
	print("Running CPU rollout path...")
	t0 = time.time()
	records_cpu, advantages_cpu, std_cpu = play_games_with_rollouts_v6(
		net, num_games=3, num_players=4,
		rollouts_per_state=5, training_seats=4,
		gpu_rollout=False)
	t_cpu = time.time() - t0
	print(f"  CPU: {len(records_cpu)} records, {len(advantages_cpu)} advantages, "
		  f"margin_std={std_cpu:.4f}, time={t_cpu:.2f}s")

	# Both paths should produce the same structure
	errors = []
	if len(records_gpu) != len(records_cpu):
		# Different random paths → different record counts are OK
		print(f"  Note: record counts differ (GPU={len(records_gpu)}, CPU={len(records_cpu)}) — "
			  "expected with different random paths")
	if len(advantages_gpu) != len(records_gpu):
		errors.append(f"GPU advantages count {len(advantages_gpu)} != records {len(records_gpu)}")
	if len(advantages_cpu) != len(records_cpu):
		errors.append(f"CPU advantages count {len(advantages_cpu)} != records {len(records_cpu)}")

	# Verify records have all required fields
	for r in records_gpu[:3]:
		if r.state is None:
			errors.append("GPU record missing state")
		if r.mask is None:
			errors.append("GPU record missing mask")

	if errors:
		print(f"FAIL: {len(errors)} errors")
		for e in errors:
			print(f"  {e}")
		return False

	print(f"\nPASS: integration test (GPU {t_gpu:.2f}s vs CPU {t_cpu:.2f}s)")
	return True


if __name__ == '__main__':
	passed = test_integration()
	print(f"\n{'PASSED' if passed else 'FAILED'}")
