"""Test compute_action_masks_kernel against gpu_engine.compute_action_masks."""
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import random
import torch
from numba import cuda as ncuda
from game import Game, Phase, Play
from encoding import get_legal_plays, get_flat_action_mask, HAND_SLOTS_V6, FLAT_ACTION_SIZE
from gpu_engine import from_snapshots, compute_legal_plays as ref_legal_plays, compute_action_masks as ref_masks
from numba_engine import (
	compute_legal_plays_kernel, compute_action_masks_kernel,
	H, TPB, _grid, FLAT_ACTION_SIZE as FA,
)

DEVICE = 'cuda'

def make_games(num_players, count, seed_start=0, turns=0, need_sns_available=False):
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
		if g.phase not in (Phase.TURN, Phase.SNS_PLAY):
			continue
		if need_sns_available and not g.players[g.current_player].sns_available:
			continue
		games.append(g)
		if len(games) >= count:
			break
	return games

def make_sns_play_games(count, seed_start=0):
	games = []
	for i in range(seed_start, seed_start + count * 50):
		random.seed(i)
		g = Game(4)
		g.start_round()
		for p in range(4):
			g.submit_flip_decision(p, do_flip=False)
		p = g.current_player
		legal = get_legal_plays(g.players[p].hand, g.current_play)
		if not legal:
			continue
		g.apply_play(*legal[0])
		if g.phase != Phase.TURN or g.current_play is None:
			continue
		p2 = g.current_player
		ps2 = g.players[p2]
		if not ps2.sns_available or len(ps2.hand) >= H:
			continue
		cp2 = g.current_play
		left_card = cp2.cards[0]
		new_hand = [left_card] + list(ps2.hand)
		remaining = cp2.cards[1:]
		reduced = Play.from_cards(remaining) if remaining else None
		has_legal = any(True for _ in get_legal_plays(new_hand, reduced))
		if not has_legal:
			continue
		g.apply_sns_scout(left_end=True, flip=False, insert_pos=0)
		if g.phase == Phase.SNS_PLAY:
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

def run_numba_masks(state, legal, hand_offsets):
	B = state.done.shape[0]
	out = torch.zeros(B, FA, dtype=torch.bool, device=DEVICE)
	compute_action_masks_kernel[_grid(B), TPB](
		ncuda.as_cuda_array(state.hands_show),
		ncuda.as_cuda_array(state.hand_len),
		ncuda.as_cuda_array(state.current_player),
		ncuda.as_cuda_array(state.play_show),
		ncuda.as_cuda_array(state.play_hide),
		ncuda.as_cuda_array(state.play_len),
		ncuda.as_cuda_array(state.play_type),
		ncuda.as_cuda_array(state.phase),
		ncuda.as_cuda_array(state.sns_available),
		ncuda.as_cuda_array(state.num_players),
		ncuda.as_cuda_array(legal),
		ncuda.as_cuda_array(hand_offsets),
		ncuda.as_cuda_array(out),
		B,
	)
	ncuda.synchronize()
	return out

def check(games, label):
	if not games:
		print(f"SKIP [{label}]: no games")
		return True
	B = len(games)
	state = from_snapshots(games, device=DEVICE)
	random.seed(999)
	ho = torch.tensor([random.randint(0, H-1) for _ in range(B)], dtype=torch.long, device=DEVICE)
	ref_leg = ref_legal_plays(state)
	ref_m = ref_masks(state, ref_leg, ho).cpu()
	numba_leg = run_numba_legal(state)
	numba_m = run_numba_masks(state, numba_leg, ho).cpu()
	mismatches = (ref_m != numba_m).sum().item()
	if mismatches:
		for b in range(B):
			diff = (ref_m[b] != numba_m[b])
			if diff.any():
				diff_idx = diff.nonzero(as_tuple=True)[0].tolist()
				extra = [i for i in diff_idx if numba_m[b, i] and not ref_m[b, i]]
				missing = [i for i in diff_idx if not numba_m[b, i] and ref_m[b, i]]
				g = games[b]
				print(f"FAIL [{label}]: {mismatches} mismatched cells")
				print(f"  game {b} (n={g.num_players}, hl={len(g.players[g.current_player].hand)}, "
					  f"phase={g.phase.name}, "
					  f"play={'len'+str(len(g.current_play.cards)) if g.current_play else 'no'}, "
					  f"sns={g.players[g.current_player].sns_available})")
				if extra:
					regions = ['play' if i < 256 else 'scout' if i < 320 else 'sns' for i in extra[:5]]
					print(f"  extra (numba has, ref doesn't): {extra[:5]} regions={regions}")
				if missing:
					regions = ['play' if i < 256 else 'scout' if i < 320 else 'sns' for i in missing[:5]]
					print(f"  missing (ref has, numba doesn't): {missing[:5]} regions={regions}")
				return False
	print(f"PASS [{label}]: {B} games, all {FA} actions match")
	return True

def main():
	ok = True
	ok &= check(make_games(4, 20, seed_start=0), "4p fresh")
	ok &= check(make_games(3, 10, seed_start=100), "3p fresh")
	ok &= check(make_games(5, 10, seed_start=200), "5p fresh")
	ok &= check(make_games(4, 30, seed_start=300, turns=1), "4p 1 play")
	ok &= check(make_games(4, 30, seed_start=400, turns=2), "4p 2 plays")
	ok &= check(make_games(5, 20, seed_start=500, turns=2), "5p 2 plays")
	ok &= check(make_games(3, 20, seed_start=600, turns=2), "3p 2 plays")
	ok &= check(make_sns_play_games(15, seed_start=700), "SNS_PLAY")
	ok &= check(make_games(4, 30, seed_start=800, turns=1, need_sns_available=True), "sns_available")
	mixed = (make_games(3, 5, seed_start=900, turns=2) +
			 make_games(4, 5, seed_start=1000, turns=2) +
			 make_games(5, 5, seed_start=1100, turns=2))
	ok &= check(mixed, "mixed")
	# Single-card plays (no right card choices)
	one_card = []
	for i in range(500):
		random.seed(i + 2000)
		g = Game(4)
		g.start_round()
		for p in range(4):
			g.submit_flip_decision(p, do_flip=False)
		p = g.current_player
		hand = g.players[p].hand
		singles = [(s, e) for s, e in get_legal_plays(hand, g.current_play) if s == e]
		if not singles:
			continue
		g.apply_play(*singles[0])
		if g.phase == Phase.TURN and g.current_play and len(g.current_play.cards) == 1:
			one_card.append(g)
		if len(one_card) >= 15:
			break
	ok &= check(one_card, f"play_len==1 x{len(one_card)}")
	print()
	print("All passed." if ok else "SOME TESTS FAILED.")

if __name__ == '__main__':
	main()
