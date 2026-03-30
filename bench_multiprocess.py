"""Benchmark multiprocessing rollout scaling.

Uses persistent worker pool to measure steady-state throughput,
not process spawn overhead.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import random
import time
import pickle
import torch
import multiprocessing as mp
from game import Game, Phase
from encoding import get_legal_plays, INPUT_SIZE_V6
from network import FlatScoutNetwork

# Globals set in each worker by initializer
_worker_net = None

def _init_worker(model_state, layer_sizes, attention):
	"""Called once per worker process — loads model into process memory."""
	global _worker_net
	import sys, os
	sys.path.insert(0, os.path.dirname(__file__))
	from network import FlatScoutNetwork
	from encoding import INPUT_SIZE_V6
	_worker_net = FlatScoutNetwork(input_size=INPUT_SIZE_V6, layer_sizes=layer_sizes, attention=attention)
	_worker_net.load_state_dict(model_state)
	_worker_net.eval()


def _worker_rollout(snapshot_bytes):
	"""Run rollouts on a chunk. Model already loaded."""
	from training import rollout_from_states_batched_v6
	snapshots = pickle.loads(snapshot_bytes)
	with torch.no_grad():
		scores = rollout_from_states_batched_v6(snapshots, _worker_net)
	return len(scores)


def make_snapshots(num_players: int, count: int, seed_start: int = 0) -> list[Game]:
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


def bench_single(snapshots, network):
	from training import rollout_from_states_batched_v6
	network.cpu()
	network.eval()
	t0 = time.time()
	with torch.no_grad():
		rollout_from_states_batched_v6(snapshots, network)
	return time.time() - t0


def bench_mp(snapshots, pool, n_workers):
	"""Send work to an already-running pool. Measures only rollout time."""
	chunk_size = (len(snapshots) + n_workers - 1) // n_workers
	chunks = []
	for i in range(n_workers):
		start = i * chunk_size
		end = min(start + chunk_size, len(snapshots))
		if start >= len(snapshots):
			break
		chunks.append(pickle.dumps(snapshots[start:end]))

	t0 = time.time()
	results = pool.map(_worker_rollout, chunks)
	elapsed = time.time() - t0
	return elapsed, sum(results)


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)

	print(f"CPU cores: {mp.cpu_count()}")
	print()

	net = FlatScoutNetwork(input_size=INPUT_SIZE_V6, layer_sizes=[512, 256, 128], attention=None)
	print(f"Network params: {sum(p.numel() for p in net.parameters()):,}\n")

	model_state = net.state_dict()
	layer_sizes = net.layer_sizes
	attention = getattr(net, 'attention_cfg', None)

	print("Generating snapshots...")
	snap_pool = make_snapshots(4, 2000, seed_start=42000)
	print(f"Generated: {len(snap_pool)} snapshots\n")

	worker_counts = [2, 4, 6, 8, 12]

	# Pre-create all pools (pay spawn cost upfront)
	print("Spawning worker pools...")
	pools = {}
	for n in worker_counts:
		t0 = time.time()
		pools[n] = mp.Pool(n, initializer=_init_worker, initargs=(model_state, layer_sizes, attention))
		print(f"  {n} workers: spawned in {time.time() - t0:.2f}s")
	print()

	for total_games in [500, 1000, 2000, 5000]:
		snaps = [snap_pool[i % len(snap_pool)].clone() for i in range(total_games)]
		print(f"=== {total_games} games ===")

		# Single process baseline
		t_single = bench_single([s.clone() for s in snaps], net)
		rate_single = total_games / t_single
		print(f"  1 worker:   {t_single:>6.2f}s  ({rate_single:>6.0f} games/s)")

		# Persistent pool runs
		for n in worker_counts:
			# Warmup run (first call may have import overhead)
			if total_games == 500:
				warm_snaps = [snap_pool[0].clone() for _ in range(n)]
				bench_mp(warm_snaps, pools[n], n)

			t_mp, done = bench_mp([s.clone() for s in snaps], pools[n], n)
			rate = total_games / t_mp
			speedup = t_single / t_mp
			print(f"  {n:>2d} workers: {t_mp:>6.2f}s  ({rate:>6.0f} games/s)  {speedup:.2f}x")
		print()

	# Cleanup
	for p in pools.values():
		p.terminate()
		p.join()
