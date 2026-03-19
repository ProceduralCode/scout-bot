import torch
import random
from Scout.game import Game

HAND_SLOTS = 20
PLAY_SLOTS = 10
CARD_VALUES = 11 # 0=empty, 1-10=card values
HAND_SIZE = HAND_SLOTS * CARD_VALUES # 220
PLAY_SIZE = PLAY_SLOTS * CARD_VALUES * 2 # 220
# Metadata breakdown:
#   1  your score (/20)
#   4  opponent scores (/20), zero-padded
#   4  opponent hand sizes (/15), zero-padded
#   1  your S&S availability
#   4  opponent S&S availability, zero-padded
#   3  player count one-hot (3/4/5)
#   1  scouts since play, normalized
#   4  play owner relative position one-hot (1-4 seats away)
#   1  round progress (round/total_rounds)
# Total: 23
METADATA_SIZE = 23
INPUT_SIZE = HAND_SIZE + PLAY_SIZE + METADATA_SIZE # 463
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
		# Clear the empty one-hot
		buf[pos] = 0.0
		# Set the card value one-hot (showing_value is card[0])
		buf[pos + card[0]] = 1.0
	return buf

def _encode_play(current_play: list | None) -> list[float]:
	"""Encode the current play (both sides of each card) into PLAY_SLOTS slots."""
	buf = [0.0] * PLAY_SIZE
	if current_play is None:
		return buf
	for i, (side_a, side_b) in enumerate(current_play):
		if i >= PLAY_SLOTS:
			break
		pos = i * CARD_VALUES * 2
		# First side one-hot
		buf[pos + side_a] = 1.0
		# Second side one-hot
		buf[pos + CARD_VALUES + side_b] = 1.0
	return buf

def _encode_metadata(state: dict) -> list[float]:
	"""Encode metadata into a flat list of floats."""
	num_players = state["num_players"]
	scores = state["scores"]
	hand_sizes = state["hand_sizes"]
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
	# Opponent hand sizes, zero-padded to 4
	for i in range(max_opponents):
		if i + 1 < num_players:
			buf.append(hand_sizes[i + 1] / 15.0)
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

def encode_state(game: Game, player: int, hand_offset: int | None = None) -> torch.Tensor:
	"""Convert game state to a 1D float input tensor.
	Args:
		game: The game instance.
		player: The player index to encode for.
		hand_offset: Slot offset for hand placement. None = random offset
			for training augmentation. Explicit int = use that offset.
	Returns:
		1D float tensor of size INPUT_SIZE.
	"""
	if hand_offset is None:
		hand_offset = random.randint(0, HAND_SLOTS - 1)
	state = game.get_state_for_player(player)
	buf = []
	buf.extend(_encode_hand(state["hand"], hand_offset))
	buf.extend(_encode_play(state["current_play"]))
	buf.extend(_encode_metadata(state))
	return torch.tensor(buf, dtype=torch.float32)

def encode_hand_both_orientations(game: Game, player: int, hand_offset: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return state tensors for both hand orientations (for flip decision).
	The first tensor uses the hand as-is; the second flips each card
	(swaps showing and hidden values) and reverses hand order.
	"""
	if hand_offset is None:
		hand_offset = random.randint(0, HAND_SLOTS - 1)
	state = game.get_state_for_player(player)
	hand = state["hand"]
	# Normal orientation
	buf_normal = []
	buf_normal.extend(_encode_hand(hand, hand_offset))
	buf_normal.extend(_encode_play(state["current_play"]))
	buf_normal.extend(_encode_metadata(state))
	# Flipped orientation: swap showing/hidden and reverse order
	flipped_hand = [(hidden, showing) for showing, hidden in reversed(hand)]
	buf_flipped = []
	buf_flipped.extend(_encode_hand(flipped_hand, hand_offset))
	buf_flipped.extend(_encode_play(state["current_play"]))
	buf_flipped.extend(_encode_metadata(state))
	return (
		torch.tensor(buf_normal, dtype=torch.float32),
		torch.tensor(buf_flipped, dtype=torch.float32),
	)

def get_action_type_mask(game: Game) -> torch.Tensor:
	"""Return a boolean mask of shape [9] where True = legal action type."""
	legal = game.get_legal_action_types()
	mask = torch.zeros(ACTION_TYPE_SIZE, dtype=torch.bool)
	for a in legal:
		mask[a] = True
	return mask

def get_play_start_mask(game: Game, hand_offset: int = 0) -> torch.Tensor:
	"""Return a boolean mask of shape [20] for legal play start positions.
	Positions correspond to encoded hand slots (with hand_offset applied).
	"""
	legal_starts = game.get_legal_play_starts()
	mask = torch.zeros(PLAY_START_SIZE, dtype=torch.bool)
	for idx in legal_starts:
		slot = (hand_offset + idx) % HAND_SLOTS
		mask[slot] = True
	return mask

def get_play_end_mask(game: Game, start: int, hand_offset: int = 0) -> torch.Tensor:
	"""Return a boolean mask of shape [20] for legal play end positions.
	Args:
		game: The game instance.
		start: The raw hand index (before offset) of the play start.
		hand_offset: Must match the offset used during encoding.
	"""
	legal_ends = game.get_legal_play_ends(start)
	mask = torch.zeros(PLAY_END_SIZE, dtype=torch.bool)
	for idx in legal_ends:
		slot = (hand_offset + idx) % HAND_SLOTS
		mask[slot] = True
	return mask

def get_scout_insert_mask(game: Game) -> torch.Tensor:
	"""Return a boolean mask of shape [21] for legal scout insert positions."""
	legal_positions = game.get_legal_insert_positions()
	mask = torch.zeros(SCOUT_INSERT_SIZE, dtype=torch.bool)
	for pos in legal_positions:
		mask[pos] = True
	return mask

def decode_slot_to_hand_index(slot: int, hand_offset: int) -> int:
	"""Convert an encoded slot position back to a raw hand index."""
	return (slot - hand_offset) % HAND_SLOTS
