"""Benchmark Numba CUDA rollout engine at various batch sizes."""
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import random
import time
import glob
import torch
from game import Game, Phase
from encoding import get_legal_plays
from gpu_engine import from_snapshots
from numba_engine import rollout_numba

DEVICE = 'cuda'

def make_games(num_players, count, seed_start=0, turns=1):
	games = []
	for i in range(seed_start, seed_start + count * 30):
		random.seed(i)
		g = Game(num_players)
		g.start_round()
		for p in range(num_players):
			g.submit_flip_decision(p, do_flip=False)
		for _ in range(turns):
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

def load_network():
	from network import FlatScoutNetwork
	for pattern in [
		os.path.join(SCRIPT_DIR, 'checkpoints', 'checkpoint_*.pt'),
		os.path.join(SCRIPT_DIR, 'bots', 'v7_*', 'latest.pt'),
	]:
		files = glob.glob(pattern)
		if files:
			break
	if not files:
		return None
	latest = max(files, key=os.path.getmtime)
	data = torch.load(latest, map_location='cpu', weights_only=False)
	config = data.get('config', {})
	net = FlatScoutNetwork(
		input_size=309,
		layer_sizes=config.get('layer_sizes', [512, 256, 128]),
		attention=config.get('attention', None),
	)
	net.load_state_dict(data['model_state'])
	net = net.to(DEVICE)
	net.eval()
	print(f"Loaded: {os.path.basename(latest)}")
	return net

def bench(net, batch_size, warmup_runs=1, timed_runs=3):
	# Generate games
	games = make_games(4, batch_size, seed_start=batch_size * 7, turns=1)
	if len(games) < batch_size:
		# Duplicate to fill
		while len(games) < batch_size:
			games.extend(games[:batch_size - len(games)])
		games = games[:batch_size]

	# Warmup
	for _ in range(warmup_runs):
		state = from_snapshots(games, device=DEVICE)
		torch.manual_seed(0)
		rollout_numba(state, net, max_steps=100)
		torch.cuda.synchronize()

	# Timed runs
	times = []
	done_counts = []
	for r in range(timed_runs):
		state = from_snapshots(games, device=DEVICE)
		torch.cuda.synchronize()
		t0 = time.perf_counter()
		torch.manual_seed(r + 1)
		rollout_numba(state, net, max_steps=100)
		torch.cuda.synchronize()
		elapsed = time.perf_counter() - t0
		times.append(elapsed)
		done_counts.append(state.done.sum().item())

	avg_time = sum(times) / len(times)
	avg_done = sum(done_counts) / len(done_counts)
	throughput = batch_size / avg_time
	return avg_time, throughput, avg_done / batch_size

def main():
	net = load_network()
	if net is None:
		print("No checkpoint found.")
		return

	batch_sizes = [100, 500, 1000, 2000, 5000]

	print(f"\n{'B':>8} | {'Time':>8} | {'Games/s':>10} | {'Done%':>6}")
	print("-" * 45)
	for B in batch_sizes:
		avg_time, throughput, done_pct = bench(net, B)
		print(f"{B:>8} | {avg_time:>7.3f}s | {throughput:>10.0f} | {done_pct:>5.1%}")

	print("\nFor reference:")
	print("  CPU Cython: ~270 games/s")
	print("  GPU PyTorch (torch.compile): ~960 games/s peak")

if __name__ == '__main__':
	main()
