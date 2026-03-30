"""Test compute_legal_plays_kernel against gpu_engine.compute_legal_plays (PyTorch reference)."""
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import random
import torch
from numba import cuda as ncuda
from game import Game, Phase
from encoding import get_legal_plays
from gpu_engine import from_snapshots, compute_legal_plays as ref_legal_plays
from numba_engine import compute_legal_plays_kernel, H, TPB, _grid

DEVICE = 'cuda'

def make_games(num_players, count, seed_start=0, turns=0):
	games = []
	for i in range(seed_start, seed_start + count * 20):
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

def run_numba_legal(state):
	B = state.done.shape[0]
	out = torch.zeros(B, H, H, dtype=torch.bool, device=DEVICE)
	compute_legal_plays_kernel[_grid(B), TPB](
		ncuda.as_cuda_array(state.hands_show),
		ncuda.as_cuda_array(state.hand_len),
		ncuda.as_cuda_array(state.current_player),
		ncuda.as_cuda_array(state.play_len),
		ncuda.as_cuda_array(state.play_type),
		ncuda.as_cuda_array(state.play_strength),
		ncuda.as_cuda_array(state.done),
		ncuda.as_cuda_array(out),
		B,
	)
	ncuda.synchronize()
	return out

def check(games, label):
	state = from_snapshots(games, device=DEVICE)
	ref = ref_legal_plays(state).cpu()
	numba_out = run_numba_legal(state).cpu()
	mismatches = (ref != numba_out).sum().item()
	if mismatches:
		# Find first mismatch for debugging
		for b in range(len(games)):
			if (ref[b] != numba_out[b]).any():
				diff = (ref[b] != numba_out[b])
				coords = diff.nonzero()[:5]
				print(f"FAIL [{label}]: {mismatches} mismatched cells")
				print(f"  First diff at game {b}:")
				for c in coords:
					s, e = c[0].item(), c[1].item()
					print(f"    ({s},{e}): ref={ref[b,s,e].item()} numba={numba_out[b,s,e].item()}")
				return False
	print(f"PASS [{label}]: {len(games)} games, all cells match")
	return True

def main():
	ok = True
	ok &= check(make_games(4, 20, seed_start=0), "4p fresh")
	ok &= check(make_games(3, 10, seed_start=100), "3p fresh")
	ok &= check(make_games(5, 10, seed_start=200), "5p fresh")
	ok &= check(make_games(4, 20, seed_start=300, turns=1), "4p 1 play")
	ok &= check(make_games(4, 20, seed_start=400, turns=3), "4p 3 plays")
	ok &= check(make_games(5, 15, seed_start=500, turns=2), "5p 2 plays")
	mixed = (make_games(3, 5, seed_start=600, turns=2) +
			 make_games(4, 5, seed_start=700, turns=2) +
			 make_games(5, 5, seed_start=800, turns=2))
	ok &= check(mixed, "mixed")
	print()
	print("All passed." if ok else "SOME TESTS FAILED.")

if __name__ == '__main__':
	main()
