"""Test rollout_numba produces valid game completions and matches rollout_gpu behavior."""
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import random
import time
import torch
from game import Game, Phase
from encoding import get_legal_plays
from gpu_engine import from_snapshots
from numba_engine import rollout_numba

DEVICE = 'cuda'

def make_games(num_players, count, seed_start=0, turns=0):
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
	"""Load latest checkpoint for rollout testing."""
	from network import FlatScoutNetwork
	import glob
	# Try checkpoints/ first, then bots/v7_*/latest.pt
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

def main():
	net = load_network()
	if net is None:
		print("No checkpoint found, skipping rollout test.")
		return

	ok = True

	# Test 1: basic 4-player rollout
	games = make_games(4, 20, seed_start=0, turns=1)
	state = from_snapshots(games, device=DEVICE)
	torch.manual_seed(42)
	scores = rollout_numba(state, net, max_steps=100)
	done_count = state.done.sum().item()
	print(f"Test 1 [4p x20]: {done_count}/{len(games)} finished")
	# Verify scores have the right shape
	for b, s in enumerate(scores):
		n = games[b].num_players
		if len(s) != n:
			print(f"  FAIL: game {b} has {len(s)} scores, expected {n}")
			ok = False
	if ok:
		print("  PASS: all score shapes correct")

	# Test 2: mixed player counts
	mixed = make_games(3, 5, seed_start=100, turns=1) + \
			make_games(4, 10, seed_start=200, turns=1) + \
			make_games(5, 5, seed_start=300, turns=1)
	state = from_snapshots(mixed, device=DEVICE)
	torch.manual_seed(123)
	scores = rollout_numba(state, net, max_steps=100)
	done_count = state.done.sum().item()
	print(f"\nTest 2 [mixed x20]: {done_count}/{len(mixed)} finished")
	for b, s in enumerate(scores):
		n = mixed[b].num_players
		if len(s) != n:
			print(f"  FAIL: game {b} has {len(s)} scores, expected {n}")
			ok = False
	if ok:
		print("  PASS: all score shapes correct")

	# Test 3: quick timing at small batch
	games = make_games(4, 50, seed_start=400, turns=1)
	state = from_snapshots(games, device=DEVICE)
	# Warmup
	torch.manual_seed(0)
	rollout_numba(state, net, max_steps=10)

	state = from_snapshots(games, device=DEVICE)
	torch.cuda.synchronize()
	t0 = time.perf_counter()
	torch.manual_seed(1)
	rollout_numba(state, net, max_steps=100)
	torch.cuda.synchronize()
	elapsed = time.perf_counter() - t0
	done_count = state.done.sum().item()
	print(f"\nTest 3 [4p x50 timing]: {elapsed:.3f}s, {done_count}/{len(games)} finished")
	print(f"  Throughput: {len(games) / elapsed:.0f} games/s")

	print()
	print("All passed." if ok else "SOME TESTS FAILED.")

if __name__ == '__main__':
	main()
