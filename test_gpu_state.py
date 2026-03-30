"""Step 1 test: verify from_snapshots produces correct GPU state for real game objects."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import random
import torch
from game import Game, Phase, PlayType
from encoding import get_legal_plays, HAND_SLOTS_V6
from gpu_engine import from_snapshots, PHASE_TURN, PHASE_SNS_PLAY, PLAY_SET, PLAY_RUN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_game_in_turn(num_players: int, seed: int | None = None) -> Game:
	"""Create a game and advance past flip decisions so it's in TURN phase."""
	if seed is not None:
		random.seed(seed)
	game = Game(num_players)
	game.start_round()
	# Submit flip decisions for all players (all choose not to flip for simplicity)
	for p in range(num_players):
		game.submit_flip_decision(p, do_flip=False)
	assert game.phase == Phase.TURN
	return game

def make_game_in_sns_play(num_players: int, seed: int | None = None) -> Game | None:
	"""Try to create a game where a S&S is the first available action, leaving it in SNS_PLAY."""
	if seed is not None:
		random.seed(seed)
	for attempt in range(200):
		game = Game(num_players)
		game.start_round()
		for p in range(num_players):
			game.submit_flip_decision(p, do_flip=False)
		# Make a play so there's a current play to scout from
		p = game.current_player
		hand = game.players[p].hand
		legal = get_legal_plays(hand, game.current_play)
		if not legal:
			continue
		s, e = legal[0]
		game.apply_play(s, e)
		# Next player: try a S&S if available
		p = game.current_player
		ps = game.players[p]
		if not ps.sns_available or game.current_play is None:
			continue
		hand = ps.hand
		if len(hand) >= HAND_SLOTS_V6:
			continue
		# Attempt S&S: scout left card, no flip, insert at position 0
		cp = game.current_play
		left_card = cp.cards[0]
		new_hand = [left_card] + list(hand)
		reduced = cp.cards[1:] if len(cp.cards) > 1 else []
		# Check if a legal play exists after scouting (simplified check)
		from game import Play
		reduced_play = Play.from_cards(reduced) if reduced else None
		has_play = any(True for s, e in get_legal_plays(new_hand, reduced_play))
		if not has_play:
			continue
		game.apply_sns_scout(left_end=True, flip=False, insert_pos=0)
		if game.phase == Phase.SNS_PLAY:
			return game
	return None

def check_state(state, games: list[Game], label: str = ""):
	"""Compare GPU state fields against the original Python game objects."""
	errors = []
	B = len(games)
	# Move to CPU for comparison
	hs  = state.hands_show.cpu()
	hh  = state.hands_hide.cpu()
	hl  = state.hand_len.cpu()
	ps_ = state.play_show.cpu()
	ph  = state.play_hide.cpu()
	pl  = state.play_len.cpu()
	po  = state.play_owner.cpu()
	pt  = state.play_type.cpu()
	pst = state.play_strength.cpu()
	cp_ = state.current_player.cpu()
	ph_ = state.phase.cpu()
	sp  = state.scouts_since_play.cpu()
	sa  = state.sns_available.cpu()
	np_ = state.num_players.cpu()
	col = state.collected.cpu()
	tok = state.scout_tokens.cpu()
	done = state.done.cpu()

	for b, game in enumerate(games):
		n = game.num_players
		if int(np_[b]) != n:
			errors.append(f"[{b}] num_players: got {int(np_[b])}, expected {n}")

		expected_cp = game.current_player
		if int(cp_[b]) != expected_cp:
			errors.append(f"[{b}] current_player: got {int(cp_[b])}, expected {expected_cp}")

		expected_sp = game.scouts_since_play
		if int(sp[b]) != expected_sp:
			errors.append(f"[{b}] scouts_since_play: got {int(sp[b])}, expected {expected_sp}")

		expected_phase = PHASE_SNS_PLAY if game.phase == Phase.SNS_PLAY else PHASE_TURN
		if int(ph_[b]) != expected_phase:
			errors.append(f"[{b}] phase: got {int(ph_[b])}, expected {expected_phase}")

		# Hands
		for p in range(n):
			ps = game.players[p]
			expected_hl = len(ps.hand)
			if int(hl[b, p]) != expected_hl:
				errors.append(f"[{b}] player {p} hand_len: got {int(hl[b,p])}, expected {expected_hl}")
				continue
			for i, (sv, hv) in enumerate(ps.hand):
				if int(hs[b, p, i]) != sv:
					errors.append(f"[{b}] p{p} hand[{i}] show: got {int(hs[b,p,i])}, expected {sv}")
				if int(hh[b, p, i]) != hv:
					errors.append(f"[{b}] p{p} hand[{i}] hide: got {int(hh[b,p,i])}, expected {hv}")
			# Slots beyond hand_len should be 0
			for i in range(expected_hl, HAND_SLOTS_V6):
				if int(hs[b, p, i]) != 0:
					errors.append(f"[{b}] p{p} hand[{i}] show beyond hand_len: got {int(hs[b,p,i])}, expected 0")

			if int(col[b, p]) != len(ps.collected):
				errors.append(f"[{b}] p{p} collected: got {int(col[b,p])}, expected {len(ps.collected)}")
			if int(tok[b, p]) != ps.scout_tokens:
				errors.append(f"[{b}] p{p} scout_tokens: got {int(tok[b,p])}, expected {ps.scout_tokens}")
			if bool(sa[b, p]) != ps.sns_available:
				errors.append(f"[{b}] p{p} sns_available: got {bool(sa[b,p])}, expected {ps.sns_available}")

		# Play
		gcp = game.current_play
		if gcp is None:
			if int(pl[b]) != 0:
				errors.append(f"[{b}] play_len: got {int(pl[b])}, expected 0 (no play)")
		else:
			expected_pl = len(gcp.cards)
			if int(pl[b]) != expected_pl:
				errors.append(f"[{b}] play_len: got {int(pl[b])}, expected {expected_pl}")
			else:
				for i, (sv, hv) in enumerate(gcp.cards):
					if int(ps_[b, i]) != sv:
						errors.append(f"[{b}] play[{i}] show: got {int(ps_[b,i])}, expected {sv}")
					if int(ph[b, i]) != hv:
						errors.append(f"[{b}] play[{i}] hide: got {int(ph[b,i])}, expected {hv}")
			expected_pt = PLAY_SET if gcp.play_type == PlayType.SET else PLAY_RUN
			if int(pt[b]) != expected_pt:
				errors.append(f"[{b}] play_type: got {int(pt[b])}, expected {expected_pt}")
			if int(pst[b]) != gcp.strength:
				errors.append(f"[{b}] play_strength: got {int(pst[b])}, expected {gcp.strength}")
			if int(po[b]) != game.current_play_owner:
				errors.append(f"[{b}] play_owner: got {int(po[b])}, expected {game.current_play_owner}")

	if errors:
		print(f"FAIL [{label}]: {len(errors)} error(s)")
		for e in errors[:10]:
			print(f"  {e}")
		if len(errors) > 10:
			print(f"  ... and {len(errors) - 10} more")
		return False
	print(f"PASS [{label}]: all {B} games correct")
	return True

def main():
	all_passed = True

	# Test 1: 4-player games in TURN phase (standard case)
	games = [make_game_in_turn(4, seed=i) for i in range(20)]
	state = from_snapshots(games, device=DEVICE)
	all_passed &= check_state(state, games, "4-player TURN x20")

	# Test 2: 3-player and 5-player games
	games3 = [make_game_in_turn(3, seed=i) for i in range(10)]
	games5 = [make_game_in_turn(5, seed=i+100) for i in range(10)]
	state3 = from_snapshots(games3, device=DEVICE)
	state5 = from_snapshots(games5, device=DEVICE)
	all_passed &= check_state(state3, games3, "3-player TURN x10")
	all_passed &= check_state(state5, games5, "5-player TURN x10")

	# Test 3: mixed batch of player counts
	mixed = [make_game_in_turn(random.choice([3, 4, 5]), seed=i+200) for i in range(15)]
	state_mixed = from_snapshots(mixed, device=DEVICE)
	all_passed &= check_state(state_mixed, mixed, "mixed player counts x15")

	# Test 4: games that have had some turns played (non-trivial state)
	advanced_games = []
	for i in range(10):
		random.seed(i + 300)
		game = make_game_in_turn(4)
		# Play 3-5 random legal actions to create interesting state
		for _ in range(random.randint(3, 5)):
			if game.phase not in (Phase.TURN, Phase.SNS_PLAY):
				break
			p = game.current_player
			hand = game.players[p].hand
			legal = get_legal_plays(hand, game.current_play)
			if not legal:
				game._advance_turn()
				continue
			s, e = random.choice(legal)
			game.apply_play(s, e)
		if game.phase in (Phase.TURN, Phase.SNS_PLAY):
			advanced_games.append(game)
	if advanced_games:
		state_adv = from_snapshots(advanced_games, device=DEVICE)
		all_passed &= check_state(state_adv, advanced_games, f"advanced state x{len(advanced_games)}")

	# Test 5: SNS_PLAY phase
	sns_games = []
	for i in range(50):
		g = make_game_in_sns_play(4, seed=i + 400)
		if g is not None:
			sns_games.append(g)
		if len(sns_games) >= 5:
			break
	if sns_games:
		state_sns = from_snapshots(sns_games, device=DEVICE)
		all_passed &= check_state(state_sns, sns_games, f"SNS_PLAY phase x{len(sns_games)}")
	else:
		print("SKIP [SNS_PLAY]: could not construct test cases")

	print()
	print("All tests passed." if all_passed else "SOME TESTS FAILED.")

if __name__ == '__main__':
	main()
