"""Canonical state policy tracking: hand-crafted game states with obvious
correct actions, plus random mid-game baselines.

Tests whether the network has learned basic strategy by checking action
distributions on states where the right move is clear.

Usage:
	python -u canonical_states.py [checkpoint_dir]
	python -u canonical_states.py bots/v7_8
"""

import sys
import os
import random
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from network import FlatScoutNetwork
from encoding import (
	INPUT_SIZE_V6, HAND_SLOTS_V6, FLAT_ACTION_SIZE,
	encode_state_v6, get_flat_action_mask, decode_flat_action, get_legal_plays,
)
from game import Game, Play, PlayType, Phase, PlayerState


def load_checkpoint(path):
	checkpoint = torch.load(path, weights_only=False, map_location='cpu')
	cfg = checkpoint.get("config", {})
	ls = cfg.get("layer_sizes", [512, 256, 128])
	net = FlatScoutNetwork(INPUT_SIZE_V6, ls, encoding_version=6,
		attention=cfg.get("attention"))
	net.load_state_dict(checkpoint["model_state"])
	net.eval()
	iteration = checkpoint.get("iteration", "?")
	return net, cfg, iteration


def make_game_shell(num_players=4):
	"""Create a minimal Game object in TURN phase, ready for hand/play injection."""
	g = Game.__new__(Game)
	g.num_players = num_players
	g.num_values = 10
	g.total_rounds = num_players
	g.round_number = 0
	g.current_player = 0
	g.scouts_since_play = 0
	g.turn_number = 3
	g.round_ender = None
	g.starting_player = 0
	g.phase = Phase.TURN
	g.current_play = None
	g.current_play_owner = None
	g.cumulative_scores = [0] * num_players
	g.flips_remaining = set()
	# Default players with filler hands
	g.players = [
		PlayerState(hand=[(i + 1, i + 2) for i in range(8)], scout_tokens=0, sns_available=True)
		for _ in range(num_players)
	]
	return g


def build_canonical_scenarios():
	"""Return list of (name, description, expected_type, game, player, hand_offset) tuples."""
	scenarios = []

	# 1. Pair beats single — should play
	g = make_game_shell()
	g.players[0].hand = [(5, 2), (5, 8), (3, 6), (7, 1), (2, 9)]
	g.current_play = Play(cards=[(3, 7)], count=1, play_type=PlayType.SET, strength=3)
	g.current_play_owner = 1
	scenarios.append((
		"pair_beats_single",
		"Table: single 3. Hand has pair of 5s. Should play the pair.",
		"play", g, 0, 0,
	))

	# 2. No legal plays — must scout
	g = make_game_shell()
	g.players[0].hand = [(2, 5), (4, 7), (1, 6), (3, 9)]
	g.current_play = Play(
		cards=[(8, 1), (8, 2), (8, 3)], count=3,
		play_type=PlayType.SET, strength=8)
	g.current_play_owner = 2
	scenarios.append((
		"no_legal_plays",
		"Table: triple 8s. Hand is scattered low cards. Must scout.",
		"scout", g, 0, 0,
	))

	# 3. Opening move — should play something
	g = make_game_shell()
	g.players[0].hand = [(6, 2), (6, 3), (4, 8), (7, 1), (3, 9)]
	g.current_play = None
	g.current_play_owner = None
	g.turn_number = 0
	scenarios.append((
		"opening_move",
		"No current play. Hand has pair of 6s + singles. Should play.",
		"play", g, 0, 0,
	))

	# 4. Near-empty hand — play to finish the round
	g = make_game_shell()
	g.players[0].hand = [(5, 3), (5, 7)]
	g.current_play = Play(cards=[(2, 8)], count=1, play_type=PlayType.SET, strength=2)
	g.current_play_owner = 3
	g.turn_number = 20
	scenarios.append((
		"near_empty_hand",
		"Table: single 2. Hand: pair of 5s only. Play empties hand, wins round.",
		"play", g, 0, 0,
	))

	# 5. High single vs scout — ambiguous
	g = make_game_shell()
	g.players[0].hand = [(5, 2), (1, 8), (2, 7), (3, 6)]
	g.current_play = Play(cards=[(4, 9)], count=1, play_type=PlayType.SET, strength=4)
	g.current_play_owner = 1
	scenarios.append((
		"high_single_vs_scout",
		"Table: single 4. Hand has single 5 (beats it) + low cards. Ambiguous.",
		None, g, 0, 0,  # None = no expected type
	))

	# 6. Can't beat strong play — should scout
	g = make_game_shell()
	g.players[0].hand = [(2, 5), (4, 7), (6, 1), (3, 8), (1, 9)]
	g.current_play = Play(
		cards=[(9, 1), (9, 2)], count=2,
		play_type=PlayType.SET, strength=9)
	g.current_play_owner = 2
	scenarios.append((
		"cant_beat_strong_play",
		"Table: pair of 9s. Hand is scattered. Should scout.",
		"scout", g, 0, 0,
	))

	return scenarios


def collect_random_states(n=10, num_players=4, min_turns=3, max_turns=15):
	"""Collect random mid-game states for baseline comparison."""
	states = []
	for _ in range(n * 3):  # oversample to account for invalid states
		game = Game(num_players)
		game.start_round()
		for p in list(game.flips_remaining):
			game.submit_flip_decision(p, random.random() < 0.5)
		turns = random.randint(min_turns, max_turns)
		for _ in range(turns):
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
		if game.phase not in (Phase.TURN, Phase.SNS_PLAY):
			continue
		p = game.current_player
		hand = game.players[p].hand
		if not hand:
			continue
		hand_offset = random.randint(0, HAND_SLOTS_V6 - 1)
		legal = get_legal_plays(hand, game.current_play)
		has_play = len(legal) > 0
		has_scout = game.current_play is not None and len(hand) < HAND_SLOTS_V6
		desc = (f"turn {game.turn_number}, hand_len={len(hand)}, "
			f"play={'yes' if game.current_play else 'none'}, "
			f"legal_plays={len(legal)}, can_scout={'yes' if has_scout else 'no'}")
		states.append((f"random_{len(states)}", desc, None, game, p, hand_offset))
		if len(states) >= n:
			break
	return states


def classify_action(action_idx):
	if action_idx < 256:
		return "play"
	elif action_idx < 320:
		return "scout"
	else:
		return "sns"


def describe_action(action_idx, hand_offset):
	"""Human-readable description of a flat action."""
	d = decode_flat_action(action_idx, hand_offset)
	if d["type"] == "play":
		return f"play [{d['start']}:{d['end']}]"
	side = "left" if d["left_end"] else "right"
	flip = "+flip" if d["flip"] else ""
	return f"{d['type']} {side}{flip} @{d['insert_pos']}"


def analyze_state(net, game, player, hand_offset):
	"""Run policy on a single state, return analysis dict."""
	hand = game.players[player].hand
	legal = get_legal_plays(hand, game.current_play)
	forced_play = game.phase == Phase.SNS_PLAY
	state = encode_state_v6(game, player, hand_offset, forced_play=forced_play)
	mask = get_flat_action_mask(game, player, legal, hand_offset)
	with torch.no_grad():
		state_t = state.unsqueeze(0)
		hidden = net(state_t)
		logits = net.policy_logits(hidden).squeeze(0)
		value = net.value(hidden).item()
	# Masked softmax
	masked_logits = logits.clone()
	masked_logits[~mask] = float('-inf')
	probs = torch.softmax(masked_logits, dim=0)
	# Action type masses
	play_mass = probs[:256].sum().item()
	scout_mass = probs[256:320].sum().item()
	sns_mass = probs[320:].sum().item()
	# Top-k actions
	top_k = 5
	top_vals, top_idxs = probs.topk(top_k)
	top_actions = []
	for prob, idx in zip(top_vals.tolist(), top_idxs.tolist()):
		if prob < 1e-6:
			continue
		top_actions.append((idx, prob, describe_action(idx, hand_offset)))
	return {
		"value": value,
		"play_mass": play_mass,
		"scout_mass": scout_mass,
		"sns_mass": sns_mass,
		"top_actions": top_actions,
		"n_legal": mask.sum().item(),
	}


def print_section(title):
	print(f"\n{'=' * 60}")
	print(f"  {title}")
	print(f"{'=' * 60}")


def run(checkpoint_dir):
	checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")
	if not os.path.exists(checkpoint_path):
		print(f"No checkpoint at {checkpoint_path}")
		sys.exit(1)

	print(f"Loading checkpoint: {checkpoint_path}")
	net, cfg, iteration = load_checkpoint(checkpoint_path)
	print(f"Iteration: {iteration}")
	print(f"Architecture: {cfg.get('layer_sizes')}, attention={cfg.get('attention')}")

	# Canonical scenarios
	scenarios = build_canonical_scenarios()
	print_section("Canonical States")
	n_correct = 0
	n_expected = 0
	for name, desc, expected_type, game, player, hand_offset in scenarios:
		result = analyze_state(net, game, player, hand_offset)
		# Determine dominant type
		masses = {"play": result["play_mass"], "scout": result["scout_mass"], "sns": result["sns_mass"]}
		dominant = max(masses, key=masses.get)
		correct = ""
		if expected_type is not None:
			n_expected += 1
			if dominant == expected_type:
				n_correct += 1
				correct = " [CORRECT]"
			else:
				correct = f" [WRONG — expected {expected_type}]"
		print(f"\n  {name}: {desc}")
		print(f"    V={result['value']:+.4f}  legal={result['n_legal']:.0f}")
		print(f"    P(play)={result['play_mass']:.3f}  P(scout)={result['scout_mass']:.3f}  P(sns)={result['sns_mass']:.3f}{correct}")
		for idx, prob, action_desc in result["top_actions"]:
			print(f"      {prob:.3f}  {action_desc}")

	if n_expected > 0:
		print(f"\n  Canonical score: {n_correct}/{n_expected} correct dominant action type")

	# Random baseline states
	random.seed(42)
	random_states = collect_random_states(10)
	print_section("Random Mid-Game States")
	all_play = []
	all_scout = []
	all_sns = []
	for name, desc, _, game, player, hand_offset in random_states:
		result = analyze_state(net, game, player, hand_offset)
		all_play.append(result["play_mass"])
		all_scout.append(result["scout_mass"])
		all_sns.append(result["sns_mass"])
		masses = {"play": result["play_mass"], "scout": result["scout_mass"], "sns": result["sns_mass"]}
		dominant = max(masses, key=masses.get)
		print(f"\n  {name}: {desc}")
		print(f"    V={result['value']:+.4f}  legal={result['n_legal']:.0f}")
		print(f"    P(play)={result['play_mass']:.3f}  P(scout)={result['scout_mass']:.3f}  P(sns)={result['sns_mass']:.3f}  [{dominant}]")
		for idx, prob, action_desc in result["top_actions"][:3]:
			print(f"      {prob:.3f}  {action_desc}")

	# Summary
	print_section("Summary")
	if n_expected > 0:
		print(f"Canonical score: {n_correct}/{n_expected}")
	print(f"Random states — mean type masses:")
	print(f"  P(play):  {np.mean(all_play):.3f} +/- {np.std(all_play):.3f}")
	print(f"  P(scout): {np.mean(all_scout):.3f} +/- {np.std(all_scout):.3f}")
	print(f"  P(sns):   {np.mean(all_sns):.3f} +/- {np.std(all_sns):.3f}")
	print()


def main():
	if len(sys.argv) < 2:
		checkpoint_dir = os.path.join(SCRIPT_DIR, "bots", "v7_8")
	else:
		arg = sys.argv[1]
		if os.path.isabs(arg):
			checkpoint_dir = arg
		else:
			checkpoint_dir = os.path.join(SCRIPT_DIR, arg)
	run(checkpoint_dir)


if __name__ == "__main__":
	main()
