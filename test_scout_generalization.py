"""Three graduated difficulty tests for scout position learning.
Tests whether the FlatScoutNetwork can generalize different types of
scout insertion patterns from supervised data."""
import sys, random, torch, torch.nn.functional as F
from game import Game, Phase
from encoding import (
	encode_state_v6, get_flat_action_mask, decode_flat_action,
	get_legal_plays, HAND_SLOTS_V6, INPUT_SIZE_V6, FLAT_ACTION_SIZE,
)
from network import FlatScoutNetwork, masked_sample

NUM_PLAYERS = 4
H = HAND_SLOTS_V6

def fresh_round():
	game = Game(NUM_PLAYERS)
	game.start_round()
	for p in range(NUM_PLAYERS):
		game.submit_flip_decision(p, do_flip=random.random() < 0.5)
	return game

def mid_round_state():
	for _ in range(100):
		game = fresh_round()
		for _ in range(10):
			if game.phase != Phase.TURN:
				break
			p = game.current_player
			hand = game.players[p].hand
			legal_plays = get_legal_plays(hand, game.current_play)
			if legal_plays:
				s, e = random.choice(legal_plays)
				game.apply_play(s, e)
			elif game.current_play is not None:
				game.apply_scout(True, False, 0)
			else:
				break
			if game.current_play is not None and game.phase == Phase.TURN:
				return game, game.current_player
	return None

def scout_state_and_mask(game, player):
	hand = game.players[player].hand
	legal_plays = get_legal_plays(hand, game.current_play)
	ho = random.randint(0, H - 1)
	state = encode_state_v6(game, player, ho)
	mask = get_flat_action_mask(game, player, legal_plays, ho)
	scout_mask = mask.clone()
	scout_mask[:256] = False
	scout_mask[320:] = False
	if not scout_mask.any():
		return None
	return state, scout_mask, ho, hand, list(game.current_play.cards)

def train_and_eval(name, collect_fn, layers=[128, 64, 64], n_train=1000, n_test=300, epochs=50):
	print(f"\n=== {name} ===")
	random.seed(42)
	train_s, train_t, train_m = collect_fn(n_train)
	test_s, test_t, test_m = collect_fn(n_test)
	print(f"  Train: {len(train_s)}, Test: {len(test_s)}")

	network = FlatScoutNetwork(INPUT_SIZE_V6, layers, encoding_version=6)
	optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
	bs = 128

	for epoch in range(epochs):
		perm = torch.randperm(len(train_s))
		ts, tt, tm = train_s[perm], train_t[perm], train_m[perm]
		network.train()
		for b in range(len(ts) // bs):
			s = ts[b*bs:(b+1)*bs]
			t = tt[b*bs:(b+1)*bs]
			m = tm[b*bs:(b+1)*bs]
			h = network(s)
			l = network.policy_logits(h).masked_fill(~m, float('-inf'))
			loss = F.cross_entropy(l, t)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		if epoch % 10 == 0 or epoch == epochs - 1:
			network.eval()
			with torch.no_grad():
				l = network.policy_logits(network(train_s)).masked_fill(~train_m, float('-inf'))
				train_acc = (l.argmax(1) == train_t).float().mean().item()
				l = network.policy_logits(network(test_s)).masked_fill(~test_m, float('-inf'))
				test_acc = (l.argmax(1) == test_t).float().mean().item()
			print(f"  Epoch {epoch:2d}: train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

# --- Test 1: State-independent (prefer lowest legal scout action index) ---
def collect_lowest(n):
	states, targets, masks = [], [], []
	while len(states) < n:
		result = mid_round_state()
		if result is None:
			continue
		game, player = result
		r = scout_state_and_mask(game, player)
		if r is None:
			continue
		state, scout_mask, ho, hand, play_cards = r
		target = scout_mask.nonzero()[0].item()
		states.append(state)
		targets.append(target)
		masks.append(scout_mask)
	return torch.stack(states), torch.tensor(targets, dtype=torch.long), torch.stack(masks)

# --- Test 2: Hand-size dependent (prefer position len(hand)//2) ---
def collect_midpos(n):
	states, targets, masks = [], [], []
	while len(states) < n:
		result = mid_round_state()
		if result is None:
			continue
		game, player = result
		r = scout_state_and_mask(game, player)
		if r is None:
			continue
		state, scout_mask, ho, hand, play_cards = r
		target_pos = len(hand) // 2
		found = None
		for idx in range(64):
			if not scout_mask[256 + idx]:
				continue
			decoded = decode_flat_action(256 + idx, ho)
			if decoded["insert_pos"] == target_pos:
				found = 256 + idx
				break
		if found is None:
			continue
		states.append(state)
		targets.append(found)
		masks.append(scout_mask)
	return torch.stack(states), torch.tensor(targets, dtype=torch.long), torch.stack(masks)

# --- Test 3: Card-value dependent (insert next to highest-valued card in hand) ---
def collect_highval(n):
	states, targets, masks = [], [], []
	while len(states) < n:
		result = mid_round_state()
		if result is None:
			continue
		game, player = result
		r = scout_state_and_mask(game, player)
		if r is None:
			continue
		state, scout_mask, ho, hand, play_cards = r
		best_idx = None
		best_neighbor_val = -1
		for idx in range(64):
			if not scout_mask[256 + idx]:
				continue
			decoded = decode_flat_action(256 + idx, ho)
			pos = decoded["insert_pos"]
			for npos in [pos - 1, pos]:
				if 0 <= npos < len(hand):
					val = hand[npos][0]
					if val > best_neighbor_val:
						best_neighbor_val = val
						best_idx = 256 + idx
		if best_idx is None:
			continue
		states.append(state)
		targets.append(best_idx)
		masks.append(scout_mask)
	return torch.stack(states), torch.tensor(targets, dtype=torch.long), torch.stack(masks)

# --- Test 4: Adjacent matching (same as probe 11) ---
def collect_adjacent(n):
	states, targets, masks = [], [], []
	while len(states) < n:
		result = mid_round_state()
		if result is None:
			continue
		game, player = result
		r = scout_state_and_mask(game, player)
		if r is None:
			continue
		state, scout_mask, ho, hand, play_cards = r
		best_actions = []
		has_match = False
		for idx in range(64):
			if not scout_mask[256 + idx]:
				continue
			action_idx = 256 + idx
			decoded = decode_flat_action(action_idx, ho)
			card = play_cards[0] if decoded["left_end"] else play_cards[-1]
			if decoded["flip"]:
				card = (card[1], card[0])
			if not any(c[0] == card[0] for c in hand):
				continue
			has_match = True
			pos = decoded["insert_pos"]
			val = card[0]
			if (pos > 0 and hand[pos - 1][0] == val) or (pos < len(hand) and hand[pos][0] == val):
				best_actions.append(action_idx)
		if not has_match or not best_actions:
			continue
		states.append(state)
		targets.append(random.choice(best_actions))
		masks.append(scout_mask)
	return torch.stack(states), torch.tensor(targets, dtype=torch.long), torch.stack(masks)

if __name__ == "__main__":
	train_and_eval("Test 1: State-independent (lowest scout index)", collect_lowest)
	train_and_eval("Test 2: Hand-size dependent (insert at middle)", collect_midpos)
	train_and_eval("Test 3: Neighbor-value dependent (insert next to highest card)", collect_highval)
	train_and_eval("Test 4: Adjacent matching (insert next to same value)", collect_adjacent)
