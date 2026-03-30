"""Benchmark GPU rollout scaling at large batch sizes."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import random
import time
import torch
from game import Game, Phase
from encoding import get_legal_plays, INPUT_SIZE_V6, HAND_SLOTS_V6
from network import FlatScoutNetwork
from training import rollout_from_states_batched_v6
from gpu_engine import from_snapshots, rollout_gpu, MAX_STEPS

H = HAND_SLOTS_V6

def make_snapshots(num_players: int, count: int, seed_start: int = 0) -> list[Game]:
	"""Generate game snapshots ready for rollout."""
	games = []
	for i in range(seed_start, seed_start + count * 20):
		random.seed(i)
		g = Game(num_players)
		g.start_round()
		for p in range(num_players):
			g.submit_flip_decision(p, do_flip=False)
		for _ in range(random.randint(1, 3)):
			if g.phase != Phase.TURN:
				break
			p = g.current_player
			legal = get_legal_plays(g.players[p].hand, g.current_play)
			if not legal:
				g._advance_turn()
				continue
			g.apply_play(*random.choice(legal))
		if g.phase in (Phase.TURN, Phase.SNS_PLAY):
			games.append(g)
		if len(games) >= count:
			break
	return games


def bench_cpu(snapshots, network):
	network.cpu()
	network.eval()
	t0 = time.time()
	with torch.no_grad():
		rollout_from_states_batched_v6(snapshots, network)
	return time.time() - t0


def bench_gpu(snapshots, network, warmup=True):
	network.cuda()
	network.eval()
	if warmup:
		with torch.no_grad():
			ws = from_snapshots(snapshots[:2], device='cuda')
			rollout_gpu(ws, network, max_steps=5)
		torch.cuda.synchronize()
	torch.cuda.synchronize()
	t0 = time.time()
	with torch.no_grad():
		state = from_snapshots(snapshots, device='cuda')
		rollout_gpu(state, network)
	torch.cuda.synchronize()
	elapsed = time.time() - t0
	network.cpu()
	return elapsed


if __name__ == '__main__':
	print(f"CUDA: {torch.cuda.is_available()}")
	if torch.cuda.is_available():
		print(f"GPU: {torch.cuda.get_device_name()}")
		mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
		print(f"VRAM: {mem:.1f} GB")
	print()

	net = FlatScoutNetwork(input_size=INPUT_SIZE_V6, layer_sizes=[512, 256, 128], attention=None)
	print(f"Network params: {sum(p.numel() for p in net.parameters()):,}\n")

	# Generate a pool of snapshots, duplicate to reach target batch sizes
	print("Generating snapshot pool...")
	pool = make_snapshots(4, 500, seed_start=42000)
	print(f"Pool: {len(pool)} snapshots\n")

	batch_sizes = [500, 1000, 2000, 5000, 10000, 20000, 50000]

	# CPU baseline at small sizes
	print("=== CPU (Cython) ===")
	cpu_times = {}
	for B in [500, 1000, 2000]:
		snaps = [pool[i % len(pool)].clone() for i in range(B)]
		t = bench_cpu(snaps, net)
		cpu_times[B] = t
		rate = B / t
		print(f"  B={B:>6d}: {t:>6.2f}s  ({rate:>7.0f} games/s)")
	print()

	# GPU at all sizes
	print("=== GPU (torch.compile) ===")
	gpu_times = {}
	for B in batch_sizes:
		snaps = [pool[i % len(pool)].clone() for i in range(B)]
		try:
			t = bench_gpu(snaps, net, warmup=(B == batch_sizes[0]))
			gpu_times[B] = t
			rate = B / t
			speedup_cpu = ""
			if B in cpu_times:
				speedup_cpu = f"  vs CPU: {cpu_times[B] / t:.2f}x"
			print(f"  B={B:>6d}: {t:>6.2f}s  ({rate:>7.0f} games/s){speedup_cpu}")
		except torch.cuda.OutOfMemoryError:
			print(f"  B={B:>6d}: OOM")
			break
		except Exception as e:
			print(f"  B={B:>6d}: ERROR: {e}")
			break
	print()

	# Summary
	print("=== Scaling summary ===")
	if gpu_times:
		base_rate = list(gpu_times.values())[0] / list(gpu_times.keys())[0]
		base_B = list(gpu_times.keys())[0]
		for B, t in gpu_times.items():
			rate = B / t
			efficiency = (rate / (base_rate * B / base_B)) if base_rate > 0 else 0
			print(f"  B={B:>6d}: {rate:>7.0f} games/s  (scaling efficiency: {efficiency:.0%})")
