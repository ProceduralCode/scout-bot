import torch
from game import Game, Play, Phase

HAND_SLOTS = 20
PLAY_SLOTS = 10
CARD_VALUES = 11 # 0=empty, 1-10=card values
HAND_SIZE = HAND_SLOTS * CARD_VALUES
PLAY_SIZE = PLAY_SLOTS * CARD_VALUES * 2
# Metadata breakdown:
#   1  your score (/20)
#   4  opponent scores (/20), zero-padded
#   1  your hand size (/15)
#   4  opponent hand sizes (/15), zero-padded
#   1  your collected count (/20)
#   4  opponent collected counts (/20), zero-padded
#   1  your scout tokens (/5)
#   4  opponent scout tokens (/5), zero-padded
#   1  your S&S availability
#   4  opponent S&S availability, zero-padded
#   3  player count one-hot (3/4/5)
#   1  scouts since play, normalized
#   4  play owner relative position one-hot (1-4 seats away)
#   1  round progress (round/total_rounds)
# Total: 34
METADATA_SIZE = 34
INPUT_SIZE = HAND_SIZE + PLAY_SIZE + METADATA_SIZE
ACTION_TYPE_SIZE = 9
PLAY_START_SIZE = 20
PLAY_END_SIZE = 20
SCOUT_INSERT_SIZE = 21

# Action type index layout:
# 0: Play
# 1: Scout left, normal    2: Scout left, flipped
# 3: Scout right, normal   4: Scout right, flipped
# 5: S&S left, normal      6: S&S left, flipped
# 7: S&S right, normal     8: S&S right, flipped
_ACTION_TYPE_TABLE = [
	{"type": "play",  "left_end": False, "flip": False},
	{"type": "scout", "left_end": True,  "flip": False},
	{"type": "scout", "left_end": True,  "flip": True},
	{"type": "scout", "left_end": False, "flip": False},
	{"type": "scout", "left_end": False, "flip": True},
	{"type": "sns",   "left_end": True,  "flip": False},
	{"type": "sns",   "left_end": True,  "flip": True},
	{"type": "sns",   "left_end": False, "flip": False},
	{"type": "sns",   "left_end": False, "flip": True},
]

def decode_action_type(index: int) -> dict:
	return _ACTION_TYPE_TABLE[index].copy()

# --- Legal move helpers ---

def get_legal_plays(hand: list, current_play: Play | None) -> list[tuple[int, int]]:
	"""Return (start, end) pairs for all legal plays from this hand."""
	plays = []
	for start in range(len(hand)):
		for end in range(start, len(hand)):
			candidate = Play.from_cards(hand[start:end + 1])
			if candidate is None:
				continue
			if current_play is None or candidate.beats(current_play):
				plays.append((start, end))
	return plays

def _has_any_legal_play(hand: list, current_play: Play | None) -> bool:
	"""Check if any legal play exists (short-circuits on first match)."""
	if current_play is None:
		return len(hand) > 0
	for start in range(len(hand)):
		for end in range(start, len(hand)):
			candidate = Play.from_cards(hand[start:end + 1])
			if candidate is not None and candidate.beats(current_play):
				return True
	return False

def _sns_variant_legal(hand: list, play_cards: list, left_end: bool, flip: bool) -> bool:
	"""Check if an S&S variant is legal: after scouting, does any insert
	position yield a hand with a legal play against the reduced play?"""
	remaining = list(play_cards)
	card = remaining.pop(0) if left_end else remaining.pop()
	if flip:
		card = (card[1], card[0])
	reduced_play = Play.from_cards(remaining) if remaining else None
	# Check if inserting the scouted card at any position creates a legal play
	for insert_pos in range(len(hand) + 1):
		new_hand = hand[:insert_pos] + [card] + hand[insert_pos:]
		if _has_any_legal_play(new_hand, reduced_play):
			return True
	return False

# --- Encoding functions ---

def _encode_hand(hand: list, hand_offset: int) -> list[float]:
	"""Encode hand into HAND_SLOTS one-hot slots with given offset."""
	buf = [0.0] * HAND_SIZE
	# Fill empty slots with the "empty" one-hot (index 0 = 1.0)
	for i in range(HAND_SLOTS):
		buf[i * CARD_VALUES] = 1.0
	# Place hand cards at offset
	for i, card in enumerate(hand):
		slot = (hand_offset + i) % HAND_SLOTS
		pos = slot * CARD_VALUES
		buf[pos] = 0.0
		buf[pos + card[0]] = 1.0
	return buf

def _encode_play(current_play: list | None, play_offset: int) -> list[float]:
	"""Encode the current play (both sides of each card) into PLAY_SLOTS slots with given offset."""
	buf = [0.0] * PLAY_SIZE
	# Fill empty slots with the "empty" one-hot (index 0 = 1.0 for each side)
	for i in range(PLAY_SLOTS):
		pos = i * CARD_VALUES * 2
		buf[pos] = 1.0
		buf[pos + CARD_VALUES] = 1.0
	if current_play is None:
		return buf
	for i, (side_a, side_b) in enumerate(current_play):
		if i >= PLAY_SLOTS:
			break
		slot = (play_offset + i) % PLAY_SLOTS
		pos = slot * CARD_VALUES * 2
		buf[pos] = 0.0
		buf[pos + side_a] = 1.0
		buf[pos + CARD_VALUES] = 0.0
		buf[pos + CARD_VALUES + side_b] = 1.0
	return buf

def _encode_metadata(state: dict) -> list[float]:
	"""Encode metadata into a flat list of floats."""
	num_players = state["num_players"]
	scores = state["scores"]
	hand_sizes = state["hand_sizes"]
	collected_counts = state["collected_counts"]
	scout_tokens = state["scout_tokens"]
	sns_available = state["sns_available"]
	max_opponents = 4
	buf = []
	# Your score (you are always index 0 in the rotated state)
	buf.append(scores[0] / 20.0)
	# Opponent scores, zero-padded to 4
	for i in range(max_opponents):
		if i + 1 < num_players:
			buf.append(scores[i + 1] / 20.0)
		else:
			buf.append(0.0)
	# Your hand size
	buf.append(hand_sizes[0] / 15.0)
	# Opponent hand sizes, zero-padded to 4
	for i in range(max_opponents):
		if i + 1 < num_players:
			buf.append(hand_sizes[i + 1] / 15.0)
		else:
			buf.append(0.0)
	# Your collected count
	buf.append(collected_counts[0] / 20.0)
	# Opponent collected counts, zero-padded to 4
	for i in range(max_opponents):
		if i + 1 < num_players:
			buf.append(collected_counts[i + 1] / 20.0)
		else:
			buf.append(0.0)
	# Your scout tokens
	buf.append(scout_tokens[0] / 5.0)
	# Opponent scout tokens, zero-padded to 4
	for i in range(max_opponents):
		if i + 1 < num_players:
			buf.append(scout_tokens[i + 1] / 5.0)
		else:
			buf.append(0.0)
	# Your S&S availability
	buf.append(1.0 if sns_available[0] else 0.0)
	# Opponent S&S availability, zero-padded to 4
	for i in range(max_opponents):
		if i + 1 < num_players:
			buf.append(1.0 if sns_available[i + 1] else 0.0)
		else:
			buf.append(0.0)
	# Player count one-hot (3/4/5)
	for n in (3, 4, 5):
		buf.append(1.0 if num_players == n else 0.0)
	# Scouts since current play, normalized by (num_players - 1)
	denom = max(num_players - 1, 1)
	buf.append(state["scouts_since_play"] / denom)
	# Play owner relative position one-hot (4 slots: 1-4 seats away)
	owner_pos = state["play_owner_relative_pos"]
	for dist in range(1, 5):
		buf.append(1.0 if owner_pos == dist else 0.0)
	# Round progress
	total = state["total_rounds"]
	buf.append(state["round_number"] / total if total > 0 else 0.0)
	return buf

def encode_state(game: Game, player: int, hand_offset: int, play_offset: int) -> torch.Tensor:
	"""Convert game state to a 1D float input tensor."""
	state = game.get_state_for_player(player)
	buf = []
	buf.extend(_encode_hand(state["hand"], hand_offset))
	buf.extend(_encode_play(state["current_play"], play_offset))
	buf.extend(_encode_metadata(state))
	return torch.tensor(buf, dtype=torch.float32)

def encode_hand_both_orientations(game: Game, player: int, hand_offset: int, play_offset: int) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return state tensors for both hand orientations (for flip decision)."""
	state = game.get_state_for_player(player)
	hand = state["hand"]
	# Normal orientation
	buf_normal = []
	buf_normal.extend(_encode_hand(hand, hand_offset))
	buf_normal.extend(_encode_play(state["current_play"], play_offset))
	buf_normal.extend(_encode_metadata(state))
	# Flipped orientation: swap showing/hidden values
	flipped_hand = [(hidden, showing) for showing, hidden in hand]
	buf_flipped = []
	buf_flipped.extend(_encode_hand(flipped_hand, hand_offset))
	buf_flipped.extend(_encode_play(state["current_play"], play_offset))
	buf_flipped.extend(_encode_metadata(state))
	return (
		torch.tensor(buf_normal, dtype=torch.float32),
		torch.tensor(buf_flipped, dtype=torch.float32),
	)

# --- Mask functions ---

def get_action_type_mask(game: Game, legal_plays: list[tuple[int, int]]) -> torch.Tensor:
	"""Return a boolean mask of shape [9] where True = legal action type."""
	mask = torch.zeros(ACTION_TYPE_SIZE, dtype=torch.bool)
	player = game.players[game.current_player]
	hand = player.hand
	has_play = game.current_play is not None

	if game.phase == Phase.SNS_PLAY:
		mask[0] = True
		return mask

	# Play: legal if no current play or at least one legal play exists
	if legal_plays:
		mask[0] = True

	if has_play:
		play_cards = game.current_play.cards
		# Scout: always legal if there's a play to scout from
		mask[1] = True  # left normal
		mask[2] = True  # left flipped
		if len(play_cards) > 1:
			mask[3] = True  # right normal
			mask[4] = True  # right flipped

		# S&S: need the token + must have a legal play after scouting
		if player.sns_available:
			if _sns_variant_legal(hand, play_cards, left_end=True, flip=False):
				mask[5] = True
			if _sns_variant_legal(hand, play_cards, left_end=True, flip=True):
				mask[6] = True
			if len(play_cards) > 1:
				if _sns_variant_legal(hand, play_cards, left_end=False, flip=False):
					mask[7] = True
				if _sns_variant_legal(hand, play_cards, left_end=False, flip=True):
					mask[8] = True

	return mask

def get_play_start_mask(legal_plays: list[tuple[int, int]], hand_offset: int) -> torch.Tensor:
	"""Return a boolean mask of shape [20] for legal play start positions.
	Positions correspond to encoded hand slots (with hand_offset applied)."""
	mask = torch.zeros(PLAY_START_SIZE, dtype=torch.bool)
	for start, end in legal_plays:
		slot = (hand_offset + start) % HAND_SLOTS
		mask[slot] = True
	return mask

def get_play_end_mask(legal_plays: list[tuple[int, int]], start: int, hand_offset: int) -> torch.Tensor:
	"""Return a boolean mask of shape [20] for legal play end positions.
	Args:
		start: The raw hand index (before offset) of the play start.
	"""
	mask = torch.zeros(PLAY_END_SIZE, dtype=torch.bool)
	for s, end in legal_plays:
		if s == start:
			slot = (hand_offset + end) % HAND_SLOTS
			mask[slot] = True
	return mask

def get_scout_insert_mask(game: Game) -> torch.Tensor:
	"""Return a boolean mask of shape [21] for legal scout insert positions.
	All positions 0 to len(hand) are legal."""
	hand_len = len(game.players[game.current_player].hand)
	mask = torch.zeros(SCOUT_INSERT_SIZE, dtype=torch.bool)
	for pos in range(hand_len + 1):
		mask[pos] = True
	return mask

def decode_slot_to_hand_index(slot: int, hand_offset: int) -> int:
	"""Convert an encoded slot position back to a raw hand index."""
	return (slot - hand_offset) % HAND_SLOTS
