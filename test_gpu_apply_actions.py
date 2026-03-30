"""Step 5 test: verify apply_actions matches CPU game engine action application."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import random
import torch
from game import Game, Phase, Play, PlayType
from encoding import (
	get_legal_plays, get_flat_action_mask, decode_flat_action,
	HAND_SLOTS_V6, FLAT_ACTION_SIZE,
)
from gpu_engine import (
	from_snapshots, compute_legal_plays, compute_action_masks,
	apply_actions, compute_scores,
	PHASE_TURN, PHASE_SNS_PLAY, H, MAX_P, PLAY_SET, PLAY_RUN, PLAY_NONE,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_games(num_players: int, count: int, seed_start: int = 0,
			   turns: int = 0) -> list[Game]:
	"""Generate games in TURN or SNS_PLAY phase."""
	games = []
	for i in range(seed_start, seed_start + count * 30):
		random.seed(i)
		g = Game(num_players)
		g.start_round()
		for p in range(num_players):
			g.submit_flip_decision(p, do_flip=False)
		for _ in range(turns):
			if g.phase not in (Phase.TURN,):
				break
			p = g.current_player
			legal = get_legal_plays(g.players[p].hand, g.current_play)
			if not legal:
				g._advance_turn()
				continue
			g.apply_play(*random.choice(legal))
		if g.phase not in (Phase.TURN, Phase.SNS_PLAY):
			continue
		games.append(g)
		if len(games) >= count:
			break
	return games


def compare_state(gpu_state, cpu_game, b, label):
	"""Compare a single game slot in gpu_state against a CPU Game object."""
	g = cpu_game
	n = g.num_players
	errors = []
	def err(msg):
		errors.append(f"  [{label}] game {b}: {msg}")

	# Current player
	gpu_cp = gpu_state.current_player[b].item()
	if g.phase in (Phase.TURN, Phase.SNS_PLAY):
		if gpu_cp != g.current_player:
			err(f"current_player: GPU={gpu_cp} CPU={g.current_player}")

	# Phase
	gpu_phase = gpu_state.phase[b].item()
	expected_phase = PHASE_SNS_PLAY if g.phase == Phase.SNS_PLAY else PHASE_TURN
	if g.phase in (Phase.TURN, Phase.SNS_PLAY):
		if gpu_phase != expected_phase:
			err(f"phase: GPU={gpu_phase} CPU={expected_phase}")

	# Done / round_ender
	gpu_done = gpu_state.done[b].item()
	cpu_done = g.phase in (Phase.ROUND_OVER, Phase.GAME_OVER)
	if gpu_done != cpu_done:
		err(f"done: GPU={gpu_done} CPU={cpu_done}")
	if cpu_done:
		gpu_ender = gpu_state.round_ender[b].item()
		if gpu_ender != g.round_ender:
			err(f"round_ender: GPU={gpu_ender} CPU={g.round_ender}")

	# Hands
	for p in range(n):
		cpu_hand = g.players[p].hand
		gpu_hl = gpu_state.hand_len[b, p].item()
		if gpu_hl != len(cpu_hand):
			err(f"hand_len[{p}]: GPU={gpu_hl} CPU={len(cpu_hand)}")
			continue
		for i in range(gpu_hl):
			gs = gpu_state.hands_show[b, p, i].item()
			gh = gpu_state.hands_hide[b, p, i].item()
			cs, ch = cpu_hand[i]
			if gs != cs or gh != ch:
				err(f"hand[{p}][{i}]: GPU=({gs},{gh}) CPU=({cs},{ch})")
		# Verify zeroed beyond hand_len
		for i in range(gpu_hl, H):
			if gpu_state.hands_show[b, p, i].item() != 0:
				err(f"hand_show[{p}][{i}] not zero (beyond hand_len)")
			if gpu_state.hands_hide[b, p, i].item() != 0:
				err(f"hand_hide[{p}][{i}] not zero (beyond hand_len)")

	# Play state
	if g.current_play is None:
		if gpu_state.play_len[b].item() != 0:
			err(f"play_len: GPU={gpu_state.play_len[b].item()} but CPU has no play")
	else:
		cp_play = g.current_play
		gpu_plen = gpu_state.play_len[b].item()
		if gpu_plen != cp_play.count:
			err(f"play_len: GPU={gpu_plen} CPU={cp_play.count}")
		else:
			for i in range(gpu_plen):
				gs = gpu_state.play_show[b, i].item()
				gh = gpu_state.play_hide[b, i].item()
				cs, ch = cp_play.cards[i]
				if gs != cs or gh != ch:
					err(f"play[{i}]: GPU=({gs},{gh}) CPU=({cs},{ch})")
		gpu_pt = gpu_state.play_type[b].item()
		expected_pt = PLAY_SET if cp_play.play_type == PlayType.SET else PLAY_RUN
		if gpu_pt != expected_pt:
			err(f"play_type: GPU={gpu_pt} CPU={expected_pt}")
		gpu_ps = gpu_state.play_strength[b].item()
		if gpu_ps != cp_play.strength:
			err(f"play_strength: GPU={gpu_ps} CPU={cp_play.strength}")
		gpu_po = gpu_state.play_owner[b].item()
		if gpu_po != g.current_play_owner:
			err(f"play_owner: GPU={gpu_po} CPU={g.current_play_owner}")

	# Play owner should be -1 when no play
	if g.current_play is None and g.current_play_owner is None:
		if gpu_state.play_owner[b].item() != -1:
			err(f"play_owner should be -1 when no play, got {gpu_state.play_owner[b].item()}")

	# Collected, scout_tokens
	for p in range(n):
		gc = gpu_state.collected[b, p].item()
		cc = len(g.players[p].collected)
		if gc != cc:
			err(f"collected[{p}]: GPU={gc} CPU={cc}")
		gt = gpu_state.scout_tokens[b, p].item()
		ct = g.players[p].scout_tokens
		if gt != ct:
			err(f"scout_tokens[{p}]: GPU={gt} CPU={ct}")

	# scouts_since_play
	if g.phase in (Phase.TURN, Phase.SNS_PLAY):
		gpu_ssp = gpu_state.scouts_since_play[b].item()
		if gpu_ssp != g.scouts_since_play:
			err(f"scouts_since_play: GPU={gpu_ssp} CPU={g.scouts_since_play}")

	# sns_available
	for p in range(n):
		ga = gpu_state.sns_available[b, p].item()
		ca = g.players[p].sns_available
		if ga != ca:
			err(f"sns_available[{p}]: GPU={ga} CPU={ca}")

	return errors


def test_apply_single_action(games, action_type_filter=None, label=""):
	"""Apply one action to each game via both GPU and CPU, compare results.
	action_type_filter: 'play', 'scout', 'sns', or None (any)."""
	B = len(games)
	if B == 0:
		print(f"SKIP [{label}]: no games")
		return True

	# Clone games for CPU path
	cpu_games = [g.clone() for g in games]
	state = from_snapshots(games, device=DEVICE)

	# Compute masks
	random.seed(42)
	ho_list = [random.randint(0, H - 1) for _ in range(B)]
	ho = torch.tensor(ho_list, device=DEVICE, dtype=torch.long)

	legal = compute_legal_plays(state)
	masks = compute_action_masks(state, legal, ho)

	# For each game, pick an action matching the filter
	chosen_actions = torch.zeros(B, device=DEVICE, dtype=torch.long)
	valid_games = torch.ones(B, device=DEVICE, dtype=torch.bool)

	for b in range(B):
		mask_b = masks[b]
		if action_type_filter == 'play':
			region = mask_b[:256]
		elif action_type_filter == 'scout':
			region = mask_b[256:320]
		elif action_type_filter == 'sns':
			region = mask_b[320:384]
		else:
			region = mask_b

		indices = region.nonzero(as_tuple=True)[0]
		if len(indices) == 0:
			valid_games[b] = False
			continue

		# Pick a random valid action
		random.seed(b * 1000 + 7)
		idx = indices[random.randint(0, len(indices) - 1)].item()
		if action_type_filter == 'scout':
			idx += 256
		elif action_type_filter == 'sns':
			idx += 320
		chosen_actions[b] = idx

	# Apply on GPU
	apply_actions(state, chosen_actions, ho, valid_games)

	# Apply on CPU
	for b in range(B):
		if not valid_games[b]:
			continue
		action = decode_flat_action(chosen_actions[b].item(), ho_list[b])
		g = cpu_games[b]
		if action['type'] == 'play':
			g.apply_play(action['start'], action['end'])
		elif action['type'] == 'scout':
			g.apply_scout(action['left_end'], action['flip'], action['insert_pos'])
		elif action['type'] == 'sns':
			g.apply_sns_scout(action['left_end'], action['flip'], action['insert_pos'])

	# Compare
	all_errors = []
	for b in range(B):
		if not valid_games[b]:
			continue
		errs = compare_state(state, cpu_games[b], b, label)
		all_errors.extend(errs)

	if all_errors:
		print(f"FAIL [{label}]: {len(all_errors)} errors")
		for e in all_errors[:20]:
			print(e)
		return False
	valid_count = valid_games.sum().item()
	print(f"PASS [{label}]: {valid_count}/{B} games checked")
	return True


def test_multi_step(num_steps=5, num_games=20, num_players=4, label="multi-step"):
	"""Apply multiple actions sequentially, comparing after each step."""
	games = make_games(num_players, num_games, seed_start=500, turns=1)
	B = len(games)
	if B == 0:
		print(f"SKIP [{label}]: no games")
		return True

	cpu_games = [g.clone() for g in games]
	state = from_snapshots(games, device=DEVICE)
	all_errors = []

	for step in range(num_steps):
		active = ~state.done
		if not active.any():
			break
		random.seed(step * 100)
		ho_list = [random.randint(0, H - 1) for _ in range(B)]
		ho = torch.tensor(ho_list, device=DEVICE, dtype=torch.long)

		legal = compute_legal_plays(state)
		masks = compute_action_masks(state, legal, ho)
		has_action = masks.any(dim=1)

		# Handle no-action: advance turn on both sides
		no_action = active & ~has_action
		if no_action.any():
			adv_cp = ((state.current_player.long() + 1) %
				state.num_players.long()).to(torch.int8)
			state.current_player = torch.where(
				no_action, adv_cp, state.current_player)
			for b in range(B):
				if no_action[b].item():
					cpu_games[b]._advance_turn()

		# Pick random valid actions
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

		# Apply GPU
		apply_actions(state, chosen, ho, apply_mask)

		# Apply CPU
		for b in range(B):
			if not apply_mask[b].item():
				continue
			action = decode_flat_action(chosen[b].item(), ho_list[b])
			g = cpu_games[b]
			if action['type'] == 'play':
				g.apply_play(action['start'], action['end'])
			elif action['type'] == 'scout':
				g.apply_scout(action['left_end'], action['flip'], action['insert_pos'])
			elif action['type'] == 'sns':
				g.apply_sns_scout(action['left_end'], action['flip'], action['insert_pos'])

		# Compare after each step
		for b in range(B):
			errs = compare_state(state, cpu_games[b], b, f"{label} step={step}")
			all_errors.extend(errs)

		if all_errors:
			break

	if all_errors:
		print(f"FAIL [{label}]: {len(all_errors)} errors after {step + 1} steps")
		for e in all_errors[:20]:
			print(e)
		return False
	print(f"PASS [{label}]: {B} games × {step + 1} steps")
	return True


def test_scores(label="scores"):
	"""Run games to completion, compare GPU scores against CPU scores."""
	games = make_games(4, 10, seed_start=800, turns=2)
	B = len(games)
	if B == 0:
		print(f"SKIP [{label}]: no games")
		return True

	cpu_games = [g.clone() for g in games]
	state = from_snapshots(games, device=DEVICE)
	max_steps = 80

	for step in range(max_steps):
		active = ~state.done
		if not active.any():
			break
		random.seed(step * 77)
		ho_list = [random.randint(0, H - 1) for _ in range(B)]
		ho = torch.tensor(ho_list, device=DEVICE, dtype=torch.long)

		legal = compute_legal_plays(state)
		masks = compute_action_masks(state, legal, ho)
		has_action = masks.any(dim=1)

		no_action = active & ~has_action
		if no_action.any():
			adv_cp = ((state.current_player.long() + 1) %
				state.num_players.long()).to(torch.int8)
			state.current_player = torch.where(
				no_action, adv_cp, state.current_player)
			for b in range(B):
				if no_action[b].item():
					cpu_games[b]._advance_turn()

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

		apply_actions(state, chosen, ho, apply_mask)

		for b in range(B):
			if not apply_mask[b].item():
				continue
			action = decode_flat_action(chosen[b].item(), ho_list[b])
			g = cpu_games[b]
			if action['type'] == 'play':
				g.apply_play(action['start'], action['end'])
			elif action['type'] == 'scout':
				g.apply_scout(action['left_end'], action['flip'], action['insert_pos'])
			elif action['type'] == 'sns':
				g.apply_sns_scout(action['left_end'], action['flip'], action['insert_pos'])

	# Compare scores
	gpu_scores = compute_scores(state)
	errors = []
	for b in range(B):
		if not state.done[b].item():
			# Game didn't finish — compare state instead
			continue
		cpu_scores = cpu_games[b].get_round_scores()
		if gpu_scores[b] != cpu_scores:
			errors.append(f"  game {b}: GPU={gpu_scores[b]} CPU={cpu_scores}")

	if errors:
		print(f"FAIL [{label}]: {len(errors)} score mismatches")
		for e in errors[:10]:
			print(e)
		return False
	done_count = state.done.sum().item()
	print(f"PASS [{label}]: {done_count}/{B} games finished, scores match")
	return True


if __name__ == '__main__':
	print(f"Device: {DEVICE}\n")
	passed = True

	# Single-action tests by type
	fresh = make_games(4, 20, seed_start=0, turns=0)
	played = make_games(4, 20, seed_start=100, turns=1)
	played2 = make_games(4, 20, seed_start=200, turns=2)
	mixed = make_games(3, 10, seed_start=300, turns=1) + make_games(5, 10, seed_start=400, turns=1)

	passed &= test_apply_single_action(fresh, 'play', "play-fresh-4p")
	passed &= test_apply_single_action(played, 'play', "play-after1-4p")
	passed &= test_apply_single_action(played, 'scout', "scout-after1-4p")
	passed &= test_apply_single_action(played2, 'play', "play-after2-4p")
	passed &= test_apply_single_action(played2, 'scout', "scout-after2-4p")
	passed &= test_apply_single_action(played2, 'sns', "sns-after2-4p")
	passed &= test_apply_single_action(mixed, None, "any-mixed-players")

	# Multi-step tests
	passed &= test_multi_step(5, 20, 4, "multi-5step-4p")
	passed &= test_multi_step(10, 15, 4, "multi-10step-4p")
	passed &= test_multi_step(5, 10, 3, "multi-5step-3p")
	passed &= test_multi_step(5, 10, 5, "multi-5step-5p")

	# Score test
	passed &= test_scores("scores-4p")

	print(f"\n{'ALL PASSED' if passed else 'SOME TESTS FAILED'}")
