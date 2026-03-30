"""Step 4 test: verify compute_action_masks matches get_flat_action_mask exactly."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import random
import torch
from game import Game, Phase, Play
from encoding import get_legal_plays, get_flat_action_mask, HAND_SLOTS_V6, FLAT_ACTION_SIZE
from gpu_engine import (
	from_snapshots, compute_legal_plays, compute_action_masks,
	PHASE_TURN, PHASE_SNS_PLAY,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
H = HAND_SLOTS_V6


def make_games(num_players: int, count: int, seed_start: int = 0,
			   turns: int = 0, need_sns_available: bool = False) -> list[Game]:
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
		if need_sns_available and not g.players[g.current_player].sns_available:
			continue
		games.append(g)
		if len(games) >= count:
			break
	return games


def make_sns_play_games(count: int, seed_start: int = 0) -> list[Game]:
	"""Generate games in SNS_PLAY phase (player just did S&S scout, now must play)."""
	games = []
	for i in range(seed_start, seed_start + count * 50):
		random.seed(i)
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
		if g.phase != Phase.TURN or g.current_play is None:
			continue
		p2 = g.current_player
		ps2 = g.players[p2]
		if not ps2.sns_available or len(ps2.hand) >= H:
			continue
		# Try to do an S&S scout: verify at least one variant is legal
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


def check_masks(games: list[Game], label: str = "") -> bool:
	"""Compare compute_action_masks output against per-game get_flat_action_mask."""
	B = len(games)
	if B == 0:
		print(f"SKIP [{label}]: no games")
		return True

	state = from_snapshots(games, device=DEVICE)

	# Compute legal plays first (needed for both GPU and reference)
	gpu_legal = compute_legal_plays(state)  # [B, H, H] on DEVICE

	# Random hand offsets
	random.seed(999)
	ho_list = [random.randint(0, H - 1) for _ in range(B)]
	hand_offsets = torch.tensor(ho_list, dtype=torch.long, device=DEVICE)

	# GPU batch mask
	gpu_masks = compute_action_masks(state, gpu_legal, hand_offsets).cpu()  # [B, 384]

	errors = []
	for b, game in enumerate(games):
		p = game.current_player
		ho = ho_list[b]
		hand = game.players[p].hand
		legal_plays = get_legal_plays(hand, game.current_play)
		ref = get_flat_action_mask(game, p, legal_plays, ho).bool()  # [384]

		gpu = gpu_masks[b]  # [384]

		if not torch.equal(gpu, ref):
			diff_idx = (gpu != ref).nonzero(as_tuple=True)[0].tolist()
			gpu_extra  = [i for i in diff_idx if gpu[i] and not ref[i]]
			gpu_missing = [i for i in diff_idx if not gpu[i] and ref[i]]
			errors.append(
				f"  game {b} (n={game.num_players}, hl={len(hand)}, "
				f"phase={game.phase.name}, "
				f"play={'len' + str(len(game.current_play.cards)) if game.current_play else 'no'}, "
				f"sns={game.players[p].sns_available}): "
				f"extra={gpu_extra[:5]}, missing={gpu_missing[:5]}"
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

	# Test 1: fresh games (no current play) — only play region active
	all_passed &= check_masks(make_games(4, 20, seed_start=0),   "4p fresh (no play)")
	all_passed &= check_masks(make_games(3, 10, seed_start=100), "3p fresh")
	all_passed &= check_masks(make_games(5, 10, seed_start=200), "5p fresh")

	# Test 2: games with current play — play + scout + S&S regions
	all_passed &= check_masks(make_games(4, 30, seed_start=300, turns=1), "4p after 1 play")
	all_passed &= check_masks(make_games(4, 30, seed_start=400, turns=2), "4p after 2 plays")
	all_passed &= check_masks(make_games(5, 20, seed_start=500, turns=2), "5p after 2 plays")
	all_passed &= check_masks(make_games(3, 20, seed_start=600, turns=2), "3p after 2 plays")

	# Test 3: SNS_PLAY phase (forced play, scout/S&S masked out)
	sns_games = make_sns_play_games(count=15, seed_start=700)
	all_passed &= check_masks(sns_games, f"SNS_PLAY phase x{len(sns_games)}")

	# Test 4: games where S&S is available (sns_available=True, has play, hand not full)
	sns_avail_games = make_games(4, 30, seed_start=800, turns=1, need_sns_available=True)
	all_passed &= check_masks(sns_avail_games, f"sns_available games x{len(sns_avail_games)}")

	# Test 5: mixed player counts
	mixed = (
		make_games(3, 5, seed_start=900, turns=2) +
		make_games(4, 5, seed_start=1000, turns=2) +
		make_games(5, 5, seed_start=1100, turns=2)
	)
	all_passed &= check_masks(mixed, "mixed player counts")

	# Test 6: play_len == 1 (no right-card choices)
	one_card_plays = []
	for i in range(500):
		random.seed(i + 2000)
		g = Game(4)
		g.start_round()
		for p in range(4):
			g.submit_flip_decision(p, do_flip=False)
		p = g.current_player
		hand = g.players[p].hand
		# Play a single card (start == end)
		singles = [(s, e) for s, e in get_legal_plays(hand, g.current_play) if s == e]
		if not singles:
			continue
		g.apply_play(*singles[0])
		if g.phase == Phase.TURN and g.current_play and len(g.current_play.cards) == 1:
			one_card_plays.append(g)
		if len(one_card_plays) >= 15:
			break
	if one_card_plays:
		all_passed &= check_masks(one_card_plays, f"play_len==1 x{len(one_card_plays)}")
	else:
		print("SKIP [play_len==1]: no suitable games")

	print()
	print("All tests passed." if all_passed else "SOME TESTS FAILED.")


if __name__ == '__main__':
	main()
