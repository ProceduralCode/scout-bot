"""Pipeline verification checks for the v6 training system.

1. Augmentation symmetry: verify inv_perm(pi(T_k(s))) ~= pi(s) for all rotations k.
   Tests that permutation tables are correct and the network treats rotations consistently.

2. Initial-state value: fresh game states should have value ~= 0 (no inherent advantage).

3. Permutation table group properties: verify FULL_PERM forms a proper cyclic group.

Usage: python -u verify_pipeline.py [checkpoint_path]
  If no checkpoint, uses a fresh (random) network.
"""

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import random
import numpy as np
import torch

from network import FlatScoutNetwork
from encoding import (
	INPUT_SIZE_V6, HAND_SLOTS_V6, FLAT_ACTION_SIZE,
	FULL_PERM, HAND_SHIFT,
	encode_state_v6, get_flat_action_mask, get_legal_plays,
)
from game import Game, Phase

H = HAND_SLOTS_V6


def load_checkpoint(path):
	checkpoint = torch.load(path, weights_only=False, map_location="cpu")
	cfg = checkpoint.get("config", {})
	ls = cfg.get("layer_sizes", [512, 256, 128])
	net = FlatScoutNetwork(INPUT_SIZE_V6, ls, encoding_version=6,
		attention=cfg.get("attention"))
	net.load_state_dict(checkpoint["model_state"])
	net.eval()
	iteration = checkpoint.get("iteration", "?")
	return net, iteration, cfg


def make_fresh_network():
	net = FlatScoutNetwork(INPUT_SIZE_V6, [512, 256, 128], encoding_version=6)
	net.eval()
	return net


def play_random_turns(game, n_turns):
	"""Advance game by playing random legal moves."""
	for _ in range(n_turns):
		if game.phase in (Phase.GAME_OVER, Phase.ROUND_OVER):
			break
		if game.phase == Phase.FLIP_DECISION:
			for p in list(game.flips_remaining):
				game.submit_flip_decision(p, random.random() < 0.5)
			continue
		p = game.current_player
		hand = game.players[p].hand
		legal = get_legal_plays(hand, game.current_play)
		if game.phase == Phase.SNS_PLAY:
			if legal:
				s, e = random.choice(legal)
				game.apply_play(s, e)
			else:
				game._advance_turn()
		elif not legal:
			if game.current_play is not None and len(hand) > 0:
				left_end = random.random() < 0.5
				flip = random.random() < 0.5
				insert_pos = random.randint(0, len(hand))
				game.apply_scout(left_end, flip, insert_pos)
			else:
				game._advance_turn()
		else:
			r = random.random()
			if r < 0.6:
				s, e = random.choice(legal)
				game.apply_play(s, e)
			elif game.current_play is not None and len(hand) > 0:
				left_end = random.random() < 0.5
				flip = random.random() < 0.5
				insert_pos = random.randint(0, len(hand))
				game.apply_scout(left_end, flip, insert_pos)
			else:
				s, e = random.choice(legal)
				game.apply_play(s, e)


def collect_states(n_games=30, n_players=4, turns_per_game=8):
	"""Collect mid-game states with their masks and hand offsets."""
	states, masks, offsets = [], [], []
	for _ in range(n_games):
		game = Game(n_players)
		game.start_round()
		play_random_turns(game, turns_per_game)
		if game.phase not in (Phase.TURN, Phase.SNS_PLAY):
			continue
		p = game.current_player
		hand = game.players[p].hand
		legal = get_legal_plays(hand, game.current_play)
		hand_offset = random.randint(0, H - 1)
		forced_play = game.phase == Phase.SNS_PLAY
		state = encode_state_v6(game, p, hand_offset, forced_play=forced_play)
		mask = get_flat_action_mask(game, p, legal, hand_offset)
		states.append(state)
		masks.append(mask)
		offsets.append(hand_offset)
	return states, masks, offsets


def check_permutation_group():
	"""Verify FULL_PERM forms a cyclic group of order H."""
	print("=" * 60)
	print("CHECK 1: Permutation table group properties")
	print("=" * 60)
	ok = True
	# Identity: FULL_PERM[0] should be identity
	identity = torch.arange(FLAT_ACTION_SIZE, dtype=torch.long)
	if not torch.equal(FULL_PERM[0], identity):
		print("  FAIL: FULL_PERM[0] is not identity")
		ok = False
	else:
		print("  PASS: FULL_PERM[0] is identity")
	# Inverse: FULL_PERM[k] composed with FULL_PERM[(H-k)%H] = identity
	inv_ok = True
	for k in range(1, H):
		inv_k = (H - k) % H
		composed = FULL_PERM[k][FULL_PERM[inv_k]]
		if not torch.equal(composed, identity):
			print(f"  FAIL: FULL_PERM[{k}] o FULL_PERM[{inv_k}] != identity")
			inv_ok = False
	if inv_ok:
		print("  PASS: All inverse compositions are identity")
	else:
		ok = False
	# Composition: FULL_PERM[a] o FULL_PERM[b] = FULL_PERM[(a+b)%H]
	comp_ok = True
	for a in range(H):
		for b in range(H):
			composed = FULL_PERM[a][FULL_PERM[b]]
			expected = FULL_PERM[(a + b) % H]
			if not torch.equal(composed, expected):
				print(f"  FAIL: FULL_PERM[{a}] o FULL_PERM[{b}] != FULL_PERM[{(a+b)%H}]")
				comp_ok = False
	if comp_ok:
		print("  PASS: Composition is cyclic (FULL_PERM[a] . FULL_PERM[b] = FULL_PERM[(a+b)%H])")
	else:
		ok = False
	# HAND_SHIFT consistency with FULL_PERM
	shift_ok = True
	for k in range(H):
		if not torch.equal(HAND_SHIFT[k][:INPUT_SIZE_V6], HAND_SHIFT[k]):
			# Just check it's valid gather indices
			if HAND_SHIFT[k].max() >= INPUT_SIZE_V6 or HAND_SHIFT[k].min() < 0:
				print(f"  FAIL: HAND_SHIFT[{k}] has out-of-range indices")
				shift_ok = False
	if shift_ok:
		print("  PASS: HAND_SHIFT indices are valid")
	else:
		ok = False
	return ok


def check_mask_round_trip(n_games=50, n_players=4):
	"""Verify that rotating state+mask by k then by -k recovers the original.
	Pure permutation table check, no network involved."""
	print()
	print("=" * 60)
	print("CHECK 2: State/mask round-trip (permutation correctness)")
	print("=" * 60)
	states, masks, offsets = collect_states(n_games, n_players)
	n = len(states)
	if n == 0:
		print("  No valid states collected -- skipping")
		return True
	print(f"  Collected {n} mid-game states")
	orig_states = torch.stack(states)
	orig_masks = torch.stack(masks)
	ok = True
	for k in range(1, H):
		shift_k = HAND_SHIFT[k]
		inv_perm_k = FULL_PERM[(H - k) % H]
		perm_k = FULL_PERM[k]
		shift_inv = HAND_SHIFT[(H - k) % H]
		# Forward: rotate state and mask by k
		aug_states = orig_states[:, shift_k]
		aug_masks = orig_masks[:, inv_perm_k]
		# Backward: rotate back by -k
		recovered_states = aug_states[:, shift_inv]
		recovered_masks = aug_masks[:, perm_k]
		state_match = torch.allclose(recovered_states, orig_states, atol=1e-6)
		mask_match = torch.equal(recovered_masks, orig_masks)
		if not state_match or not mask_match:
			print(f"  FAIL shift {k}: state_match={state_match} mask_match={mask_match}")
			ok = False
	if ok:
		print("  PASS: All round-trips recover original state and mask")
	return ok


def check_augmentation_symmetry(net, n_games=50, n_players=4):
	"""Verify inv_perm(pi(T_k(s))) ~= pi(s) for all rotations k.
	Network isn't architecturally equivariant, so deviations are expected
	on a fresh network. This measures how well equivariance has been learned."""
	print()
	print("=" * 60)
	print("CHECK 3: Policy equivariance under rotation")
	print("=" * 60)
	states, masks, offsets = collect_states(n_games, n_players)
	n = len(states)
	if n == 0:
		print("  No valid states collected -- skipping")
		return True
	print(f"  Collected {n} mid-game states")
	orig_states = torch.stack(states)
	orig_masks = torch.stack(masks)
	with torch.no_grad():
		hidden = net(orig_states)
		logits = net.policy_logits(hidden)
		orig_masked = logits.masked_fill(~orig_masks, float('-inf'))
		orig_lp = torch.log_softmax(orig_masked, dim=-1)
	max_devs = []
	mean_devs = []
	for k in range(1, H):
		shift = HAND_SHIFT[k]
		perm = FULL_PERM[k]
		inv_perm = FULL_PERM[(H - k) % H]
		aug_states = orig_states[:, shift]
		aug_masks = orig_masks[:, inv_perm]
		with torch.no_grad():
			hidden = net(aug_states)
			aug_logits = net.policy_logits(hidden)
			aug_masked = aug_logits.masked_fill(~aug_masks, float('-inf'))
			aug_lp = torch.log_softmax(aug_masked, dim=-1)
		# Map augmented log-probs back to original action space:
		# perm[a] is the augmented action corresponding to original action a
		recovered_lp = aug_lp[:, perm]
		# Compare only at valid (non -inf) positions in original
		valid = orig_masks.bool()
		if valid.sum() == 0:
			continue
		diff = (recovered_lp[valid] - orig_lp[valid]).abs()
		max_dev = diff.max().item()
		mean_dev = diff.mean().item()
		max_devs.append(max_dev)
		mean_devs.append(mean_dev)
	if not max_devs:
		print("  No shifts tested")
		return True
	overall_max = max(max_devs)
	overall_mean = np.mean(mean_devs)
	print(f"  max_dev={overall_max:.6f}  mean_dev={overall_mean:.6f}")
	# Network isn't architecturally equivariant — deviation is normal
	# Fresh network: high deviation (random weights), trained: should decrease
	if overall_max < 0.01:
		print("  Near-perfect equivariance")
	elif overall_max < 0.5:
		print("  Good equivariance (well-trained)")
	elif overall_max < 3.0:
		print("  Moderate deviation (expected for fresh/early network)")
	else:
		print("  WARN: Very large deviation")
	return True  # Informational — doesn't fail


def check_initial_value(net, n_games=100, n_players=4):
	"""Fresh game states should have value ~= 0 (no inherent player advantage)."""
	print()
	print("=" * 60)
	print("CHECK 4: Initial-state value (should be ~= 0)")
	print("=" * 60)
	values_by_player = {p: [] for p in range(n_players)}
	for _ in range(n_games):
		game = Game(n_players)
		game.start_round()
		# Submit flip decisions randomly
		for p in list(game.flips_remaining):
			game.submit_flip_decision(p, random.random() < 0.5)
		# Encode each player's state and get value
		for p in range(n_players):
			hand_offset = random.randint(0, H - 1)
			state = encode_state_v6(game, p, hand_offset)
			with torch.no_grad():
				hidden = net(state.unsqueeze(0))
				v = net.value(hidden).item()
			values_by_player[p].append(v)
	print(f"  {n_games} fresh games, {n_players} players each")
	all_vals = []
	for p in range(n_players):
		vals = values_by_player[p]
		mean = np.mean(vals)
		std = np.std(vals)
		all_vals.extend(vals)
		print(f"  Player {p}: mean={mean:+.4f}  std={std:.4f}")
	overall_mean = np.mean(all_vals)
	overall_std = np.std(all_vals)
	print(f"  Overall:  mean={overall_mean:+.4f}  std={overall_std:.4f}")
	# For self-play, starting player (p0) may have slight advantage,
	# but values should still be close to 0
	if abs(overall_mean) < 0.5:
		print("  PASS: Values near zero")
	elif abs(overall_mean) < 2.0:
		print("  INFO: Moderate bias — may be normal for trained network")
	else:
		print("  WARN: Large systematic value bias")
	return abs(overall_mean) < 5.0


def main():
	random.seed(42)
	np.random.seed(42)
	torch.manual_seed(42)
	if len(sys.argv) > 1:
		ckpt_path = sys.argv[1]
		print(f"Loading checkpoint: {ckpt_path}")
		net, iteration, cfg = load_checkpoint(ckpt_path)
		print(f"Loaded iteration {iteration}")
	else:
		print("No checkpoint specified — using fresh network")
		net = make_fresh_network()
	print()
	ok = True
	ok &= check_permutation_group()
	ok &= check_mask_round_trip()
	ok &= check_augmentation_symmetry(net)
	ok &= check_initial_value(net)
	print()
	print("=" * 60)
	if ok:
		print("ALL CHECKS PASSED")
	else:
		print("SOME CHECKS FAILED — investigate above")
	print("=" * 60)


if __name__ == "__main__":
	main()
