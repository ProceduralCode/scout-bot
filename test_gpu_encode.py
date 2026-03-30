"""Step 2 test: verify encode_states matches encode_state_v6 exactly."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import random
import torch
from game import Game, Phase, PlayType
from encoding import encode_state_v6, get_legal_plays, HAND_SLOTS_V6
from gpu_engine import from_snapshots, encode_states, PHASE_TURN, PHASE_SNS_PLAY

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
H = HAND_SLOTS_V6

def make_games_with_play(num_players: int, count: int, seed_start: int = 0) -> list[Game]:
	"""Create games that have had several turns played (so current_play is set)."""
	games = []
	for i in range(seed_start, seed_start + count * 10):
		random.seed(i)
		game = Game(num_players)
		game.start_round()
		for p in range(num_players):
			game.submit_flip_decision(p, do_flip=False)
		# Play 1-4 turns to get interesting state
		for _ in range(random.randint(1, 4)):
			if game.phase not in (Phase.TURN, Phase.SNS_PLAY):
				break
			p = game.current_player
			legal = get_legal_plays(game.players[p].hand, game.current_play)
			if not legal:
				game._advance_turn()
				continue
			s, e = random.choice(legal)
			game.apply_play(s, e)
		if game.phase in (Phase.TURN, Phase.SNS_PLAY):
			games.append(game)
		if len(games) >= count:
			break
	return games

def check_encoding(games: list[Game], label: str = "") -> bool:
	"""Compare encode_states output against per-game encode_state_v6."""
	B = len(games)
	state = from_snapshots(games, device=DEVICE)

	# Random hand offsets (same as training would use)
	random.seed(42)
	hand_offsets_list = [random.randint(0, H - 1) for _ in range(B)]
	hand_offsets = torch.tensor(hand_offsets_list, dtype=torch.long, device=DEVICE)

	# GPU batch encoding
	gpu_out = encode_states(state, hand_offsets).cpu()  # [B, 309]

	errors = []
	for b, game in enumerate(games):
		p = game.current_player
		ho = hand_offsets_list[b]
		forced = game.phase == Phase.SNS_PLAY
		ref = encode_state_v6(game, p, ho, forced_play=forced)  # [309]

		diff = (gpu_out[b] - ref).abs()
		max_diff = diff.max().item()
		if max_diff > 1e-5:
			worst_dim = diff.argmax().item()
			errors.append(
				f"  game {b} (n={game.num_players}, phase={game.phase.name}): "
				f"max diff {max_diff:.2e} at dim {worst_dim} "
				f"(gpu={gpu_out[b, worst_dim]:.4f}, ref={ref[worst_dim]:.4f})"
			)

	if errors:
		print(f"FAIL [{label}]: {len(errors)}/{B} games mismatched")
		for e in errors[:5]:
			print(e)
		return False
	print(f"PASS [{label}]: all {B} encodings match")
	return True

def main():
	all_passed = True

	# Test 1: games immediately after flip decisions (no current play)
	games_fresh = []
	for i in range(20):
		random.seed(i)
		g = Game(4)
		g.start_round()
		for p in range(4):
			g.submit_flip_decision(p, do_flip=False)
		games_fresh.append(g)
	all_passed &= check_encoding(games_fresh, "no current play (fresh)")

	# Test 2: games with current_play set, various player counts
	for n in [3, 4, 5]:
		games = make_games_with_play(n, count=15, seed_start=n*100)
		all_passed &= check_encoding(games, f"{n}-player with play")

	# Test 3: mixed player counts in one batch
	mixed = []
	for n in [3, 4, 5]:
		mixed += make_games_with_play(n, count=5, seed_start=n*200)
	all_passed &= check_encoding(mixed, "mixed player counts")

	# Test 4: games with scouts_since_play > 0
	games_scouted = []
	for i in range(100):
		random.seed(i + 500)
		g = Game(4)
		g.start_round()
		for p in range(4):
			g.submit_flip_decision(p, do_flip=False)
		# Make a play first
		p = g.current_player
		legal = get_legal_plays(g.players[p].hand, g.current_play)
		if not legal:
			continue
		g.apply_play(*legal[0])
		# Scout if possible
		if g.phase == Phase.TURN and g.current_play is not None:
			p2 = g.current_player
			hand2 = g.players[p2].hand
			if len(hand2) < H:
				g.apply_scout(left_end=True, flip=False, insert_pos=0)
		if g.phase in (Phase.TURN, Phase.SNS_PLAY) and g.scouts_since_play > 0:
			games_scouted.append(g)
		if len(games_scouted) >= 10:
			break
	if games_scouted:
		all_passed &= check_encoding(games_scouted, "scouts_since_play > 0")
	else:
		print("SKIP [scouts_since_play > 0]: no suitable games found")

	# Test 5: SNS_PLAY phase (forced play)
	sns_games = []
	for i in range(200):
		random.seed(i + 700)
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
		# Try S&S
		cp2 = g.current_play
		left_card = cp2.cards[0]
		new_hand = [left_card] + list(ps2.hand)
		remaining = cp2.cards[1:]
		from game import Play
		reduced = Play.from_cards(remaining) if remaining else None
		has_legal = any(True for _ in get_legal_plays(new_hand, reduced))
		if not has_legal:
			continue
		g.apply_sns_scout(left_end=True, flip=False, insert_pos=0)
		if g.phase == Phase.SNS_PLAY:
			sns_games.append(g)
		if len(sns_games) >= 10:
			break
	if sns_games:
		all_passed &= check_encoding(sns_games, "SNS_PLAY forced play")
	else:
		print("SKIP [SNS_PLAY]: no suitable games found")

	print()
	print("All tests passed." if all_passed else "SOME TESTS FAILED.")

if __name__ == '__main__':
	main()
