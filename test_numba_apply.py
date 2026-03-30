"""Test apply_actions_kernel against gpu_engine.apply_actions (PyTorch reference)."""
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import copy
import random
import torch
from numba import cuda as ncuda
from game import Game, Phase, Play
from encoding import get_legal_plays, HAND_SLOTS_V6, FLAT_ACTION_SIZE
from gpu_engine import (
	from_snapshots, compute_legal_plays as ref_legal,
	compute_action_masks as ref_masks, apply_actions as ref_apply,
	compute_scores, GpuGameState, H, MAX_P, MAX_PLAY,
)
from numba_engine import (
	compute_legal_plays_kernel, compute_action_masks_kernel,
	apply_actions_kernel, TPB, _grid, FLAT_ACTION_SIZE as FA,
)

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

def clone_gpu_state(state):
	"""Deep copy a GpuGameState (all tensors cloned)."""
	return GpuGameState(**{f.name: getattr(state, f.name).clone() for f in state.__dataclass_fields__.values()})

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

def run_numba_apply(state, actions, hand_offsets, active):
	B = state.done.shape[0]
	apply_actions_kernel[_grid(B), TPB](
		ncuda.as_cuda_array(state.hands_show),
		ncuda.as_cuda_array(state.hands_hide),
		ncuda.as_cuda_array(state.hand_len),
		ncuda.as_cuda_array(state.play_show),
		ncuda.as_cuda_array(state.play_hide),
		ncuda.as_cuda_array(state.play_len),
		ncuda.as_cuda_array(state.play_owner),
		ncuda.as_cuda_array(state.play_type),
		ncuda.as_cuda_array(state.play_strength),
		ncuda.as_cuda_array(state.current_player),
		ncuda.as_cuda_array(state.phase),
		ncuda.as_cuda_array(state.scouts_since_play),
		ncuda.as_cuda_array(state.sns_available),
		ncuda.as_cuda_array(state.num_players),
		ncuda.as_cuda_array(state.collected),
		ncuda.as_cuda_array(state.scout_tokens),
		ncuda.as_cuda_array(state.round_ender),
		ncuda.as_cuda_array(state.done),
		ncuda.as_cuda_array(actions),
		ncuda.as_cuda_array(hand_offsets),
		ncuda.as_cuda_array(active),
		B,
	)
	ncuda.synchronize()

def compare_states(ref_state, numba_state, B, label):
	"""Compare all fields of two GpuGameStates."""
	errors = []
	fields = [
		'hands_show', 'hands_hide', 'hand_len',
		'play_show', 'play_hide', 'play_len', 'play_owner', 'play_type', 'play_strength',
		'current_player', 'phase', 'scouts_since_play', 'sns_available',
		'num_players', 'collected', 'scout_tokens', 'round_ender', 'done',
	]
	for fname in fields:
		r = getattr(ref_state, fname)
		n = getattr(numba_state, fname)
		if not torch.equal(r, n):
			diff_mask = (r != n)
			count = diff_mask.sum().item()
			# Find first difference
			idx = diff_mask.nonzero()[0].tolist()
			rv = r[tuple(idx)].item()
			nv = n[tuple(idx)].item()
			errors.append(f"  {fname}: {count} diffs, first at {idx}: ref={rv} numba={nv}")
	if errors:
		print(f"FAIL [{label}]:")
		for e in errors:
			print(e)
		return False
	return True

def test_single_action(games, action_filter, label):
	"""Apply one action per game, compare ref vs numba state."""
	if not games:
		print(f"SKIP [{label}]")
		return True
	B = len(games)
	state_ref = from_snapshots(games, device=DEVICE)
	state_numba = clone_gpu_state(state_ref)

	random.seed(42)
	ho = torch.tensor([random.randint(0, H-1) for _ in range(B)], dtype=torch.long, device=DEVICE)

	legal = ref_legal(state_ref)
	masks = ref_masks(state_ref, legal, ho)

	chosen = torch.zeros(B, device=DEVICE, dtype=torch.long)
	valid = torch.ones(B, device=DEVICE, dtype=torch.bool)
	for b in range(B):
		m = masks[b]
		if action_filter == 'play':
			region = m[:256]
		elif action_filter == 'scout':
			region = m[256:320]
		elif action_filter == 'sns':
			region = m[320:]
		else:
			region = m
		indices = region.nonzero(as_tuple=True)[0]
		if len(indices) == 0:
			valid[b] = False
			continue
		random.seed(b * 1000 + 7)
		idx = indices[random.randint(0, len(indices) - 1)].item()
		if action_filter == 'scout':
			idx += 256
		elif action_filter == 'sns':
			idx += 320
		chosen[b] = idx

	ref_apply(state_ref, chosen, ho, valid)
	run_numba_apply(state_numba, chosen, ho, valid)

	ok = compare_states(state_ref, state_numba, B, label)
	if ok:
		valid_count = valid.sum().item()
		print(f"PASS [{label}]: {valid_count}/{B} games")
	return ok

def test_multi_step(games, num_steps, label):
	"""Apply multiple steps, comparing after each."""
	if not games:
		print(f"SKIP [{label}]")
		return True
	B = len(games)
	state_ref = from_snapshots(games, device=DEVICE)
	state_numba = clone_gpu_state(state_ref)

	for step in range(num_steps):
		active = ~state_ref.done
		if not active.any():
			break
		random.seed(step * 100)
		ho = torch.tensor([random.randint(0, H-1) for _ in range(B)], dtype=torch.long, device=DEVICE)

		legal = ref_legal(state_ref)
		masks = ref_masks(state_ref, legal, ho)
		has_action = masks.any(dim=1)

		# Advance turn for no-action games
		no_action = active & ~has_action
		if no_action.any():
			adv_cp = ((state_ref.current_player.long() + 1) %
				state_ref.num_players.long()).to(torch.int8)
			state_ref.current_player = torch.where(no_action, adv_cp, state_ref.current_player)
			state_numba.current_player = torch.where(no_action, adv_cp, state_numba.current_player)

		chosen = torch.zeros(B, device=DEVICE, dtype=torch.long)
		apply_mask = active & has_action
		for b in range(B):
			if not apply_mask[b].item():
				continue
			indices = masks[b].nonzero(as_tuple=True)[0]
			if len(indices) == 0:
				apply_mask[b] = False
				continue
			random.seed(step * 10000 + b)
			chosen[b] = indices[random.randint(0, len(indices) - 1)].item()

		ref_apply(state_ref, chosen, ho, apply_mask)
		run_numba_apply(state_numba, chosen, ho, apply_mask)

		ok = compare_states(state_ref, state_numba, B, f"{label} step={step}")
		if not ok:
			return False

	done_count = state_ref.done.sum().item()
	print(f"PASS [{label}]: {B} games x {step + 1} steps ({done_count} finished)")
	return True

def main():
	ok = True
	fresh = make_games(4, 20, seed_start=0)
	played = make_games(4, 20, seed_start=100, turns=1)
	played2 = make_games(4, 20, seed_start=200, turns=2)
	mixed = make_games(3, 10, seed_start=300, turns=1) + make_games(5, 10, seed_start=400, turns=1)

	ok &= test_single_action(fresh, 'play', "play-fresh")
	ok &= test_single_action(played, 'play', "play-after1")
	ok &= test_single_action(played, 'scout', "scout-after1")
	ok &= test_single_action(played2, 'play', "play-after2")
	ok &= test_single_action(played2, 'scout', "scout-after2")
	ok &= test_single_action(played2, 'sns', "sns-after2")
	ok &= test_single_action(mixed, None, "any-mixed")

	ok &= test_multi_step(make_games(4, 20, seed_start=500, turns=1), 5, "multi-5-4p")
	ok &= test_multi_step(make_games(4, 15, seed_start=600, turns=1), 10, "multi-10-4p")
	ok &= test_multi_step(make_games(3, 10, seed_start=700, turns=1), 5, "multi-5-3p")
	ok &= test_multi_step(make_games(5, 10, seed_start=800, turns=1), 5, "multi-5-5p")

	# Full game to completion
	ok &= test_multi_step(make_games(4, 10, seed_start=900, turns=2), 80, "full-4p")

	print()
	print("All passed." if ok else "SOME TESTS FAILED.")

if __name__ == '__main__':
	main()
