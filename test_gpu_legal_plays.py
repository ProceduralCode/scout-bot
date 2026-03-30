"""Step 3 test: verify compute_legal_plays matches get_legal_plays exactly."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import random
import torch
from game import Game, Phase
from encoding import get_legal_plays, HAND_SLOTS_V6
from gpu_engine import from_snapshots, compute_legal_plays

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_games(num_players: int, count: int, seed_start: int = 0,
               turns: int = 0) -> list[Game]:
	games = []
	for i in range(seed_start, seed_start + count * 20):
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
		if g.phase in (Phase.TURN, Phase.SNS_PLAY):
			games.append(g)
		if len(games) >= count:
			break
	return games

def check_legal_plays(games: list[Game], label: str = "") -> bool:
	B = len(games)
	state = from_snapshots(games, device=DEVICE)
	gpu_legal = compute_legal_plays(state).cpu()  # [B, H, H]

	errors = []
	for b, game in enumerate(games):
		p = game.current_player
		hand = game.players[p].hand
		hl = len(hand)

		# Reference: get_legal_plays returns list of (start, end)
		ref_plays = set(get_legal_plays(hand, game.current_play))

		# GPU result: collect all True entries in [0..hl-1, 0..hl-1]
		gpu_plays = set()
		for s in range(hl):
			for e in range(s, hl):
				if gpu_legal[b, s, e].item():
					gpu_plays.add((s, e))

		# Also check that no out-of-range positions are True
		for s in range(H := HAND_SLOTS_V6):
			for e in range(H):
				if (s >= hl or e >= hl or e < s) and gpu_legal[b, s, e].item():
					errors.append(
						f"  game {b}: out-of-range position ({s},{e}) is True "
						f"(hl={hl}, phase={game.phase.name})"
					)

		if gpu_plays != ref_plays:
			missing = ref_plays - gpu_plays
			extra   = gpu_plays - ref_plays
			errors.append(
				f"  game {b} (n={game.num_players}, hl={hl}, "
				f"play={'yes' if game.current_play else 'no'}): "
				f"missing={missing}, extra={extra}"
			)

	if errors:
		print(f"FAIL [{label}]: {len(errors)}/{B} games mismatched")
		for e in errors[:5]:
			print(e)
		return False
	print(f"PASS [{label}]: all {B} games correct")
	return True

def main():
	all_passed = True

	# Test 1: fresh games (no current play) — any single card or run is legal
	all_passed &= check_legal_plays(make_games(4, 20, seed_start=0),  "4p fresh (no play)")
	all_passed &= check_legal_plays(make_games(3, 10, seed_start=100), "3p fresh")
	all_passed &= check_legal_plays(make_games(5, 10, seed_start=200), "5p fresh")

	# Test 2: after plays (current play exists, beats check matters)
	all_passed &= check_legal_plays(make_games(4, 20, seed_start=300, turns=1), "4p after 1 play")
	all_passed &= check_legal_plays(make_games(4, 20, seed_start=400, turns=3), "4p after 3 plays")
	all_passed &= check_legal_plays(make_games(5, 15, seed_start=500, turns=2), "5p after 2 plays")

	# Test 3: mixed player counts
	mixed = (make_games(3, 5, seed_start=600, turns=2) +
			 make_games(4, 5, seed_start=700, turns=2) +
			 make_games(5, 5, seed_start=800, turns=2))
	all_passed &= check_legal_plays(mixed, "mixed player counts")

	# Test 4: exhausted hands (very few cards left)
	sparse = []
	for i in range(300):
		random.seed(i + 900)
		g = Game(4)
		g.start_round()
		for p in range(4):
			g.submit_flip_decision(p, do_flip=False)
		for _ in range(8):  # play many turns
			if g.phase != Phase.TURN:
				break
			p = g.current_player
			legal = get_legal_plays(g.players[p].hand, g.current_play)
			if not legal:
				g._advance_turn()
				continue
			g.apply_play(*random.choice(legal))
		if g.phase == Phase.TURN and len(g.players[g.current_player].hand) <= 3:
			sparse.append(g)
		if len(sparse) >= 10:
			break
	if sparse:
		all_passed &= check_legal_plays(sparse, f"sparse hands x{len(sparse)}")
	else:
		print("SKIP [sparse]: no suitable games")

	print()
	print("All tests passed." if all_passed else "SOME TESTS FAILED.")

if __name__ == '__main__':
	main()
