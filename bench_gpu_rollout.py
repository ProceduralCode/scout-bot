"""Benchmark GPU vs CPU rollout at various batch sizes."""
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
		# Play 1-2 turns to get a more interesting state
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


def bench_cpu(snapshots, network, label=""):
	network.cpu()
	network.eval()
	t0 = time.time()
	with torch.no_grad():
		scores = rollout_from_states_batched_v6(snapshots, network)
	elapsed = time.time() - t0
	done = sum(1 for s in scores if any(v != 0 for v in s))
	print(f"  CPU [{label}]: {len(snapshots)} games, {elapsed:.2f}s, {done} finished")
	return elapsed


def bench_gpu(snapshots, network, label=""):
	network.cuda()
	network.eval()
	# Warmup
	with torch.no_grad():
		warm_state = from_snapshots(snapshots[:2], device='cuda')
		rollout_gpu(warm_state, network, max_steps=5)
	torch.cuda.synchronize()
	t0 = time.time()
	with torch.no_grad():
		state = from_snapshots(snapshots, device='cuda')
		scores = rollout_gpu(state, network)
	torch.cuda.synchronize()
	elapsed = time.time() - t0
	done = state.done.sum().item()
	network.cpu()
	print(f"  GPU [{label}]: {len(snapshots)} games, {elapsed:.2f}s, {done} finished")
	return elapsed


if __name__ == '__main__':
	print(f"CUDA available: {torch.cuda.is_available()}")
	if torch.cuda.is_available():
		print(f"GPU: {torch.cuda.get_device_name()}\n")

	# Realistic network size (matches training)
	net = FlatScoutNetwork(input_size=INPUT_SIZE_V6, layer_sizes=[512, 256, 128], attention=None)
	print(f"Network params: {sum(p.numel() for p in net.parameters()):,}\n")

	for batch_size in [25, 50, 100, 200, 500]:
		print(f"--- Batch size: {batch_size} ---")
		snaps = make_snapshots(4, batch_size, seed_start=batch_size * 100)
		if len(snaps) < batch_size:
			print(f"  Only got {len(snaps)} snapshots, needed {batch_size}")
			continue
		# Clone for each path
		cpu_snaps = [s.clone() for s in snaps]
		gpu_snaps = [s.clone() for s in snaps]
		t_cpu = bench_cpu(cpu_snaps, net, f"B={batch_size}")
		if torch.cuda.is_available():
			t_gpu = bench_gpu(gpu_snaps, net, f"B={batch_size}")
			print(f"  Speedup: {t_cpu / t_gpu:.2f}x")
		print()
