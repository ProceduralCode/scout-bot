import numpy as np
import torch
from game import Game, Play, PlayType, Phase

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
#   5  play owner relative position one-hot (0=self, 1-4 seats away)
#   1  round progress (round/total_rounds)
# Total: 35
METADATA_SIZE = 35
INPUT_SIZE = HAND_SIZE + PLAY_SIZE + METADATA_SIZE
ACTION_TYPE_SIZE = 9
PLAY_START_SIZE = 20
PLAY_END_SIZE = 20
SCOUT_INSERT_SIZE = 21

# V2 encoding: smaller hand, compact play, trimmed metadata
HAND_SLOTS_V2 = 15
HAND_SIZE_V2 = HAND_SLOTS_V2 * CARD_VALUES  # 165
# Compact play: end cards (4 one-hots) + type + strength + length
PLAY_CARDS_V2 = 4 * CARD_VALUES  # 44
PLAY_TYPE_V2 = 3   # no_play / set / run
PLAY_STRENGTH_V2 = 10  # values 1-10
PLAY_LENGTH_V2 = 10    # lengths 1-10
PLAY_SIZE_V2 = PLAY_CARDS_V2 + PLAY_TYPE_V2 + PLAY_STRENGTH_V2 + PLAY_LENGTH_V2  # 67
# Metadata v2 breakdown:
#   1  your hand size (/15)
#   4  opponent hand sizes (/15), zero-padded
#   1  your collected count (/15)
#   4  opponent collected counts (/15), zero-padded
#   1  your scout tokens (/5)
#   4  opponent scout tokens (/5), zero-padded
#   1  your S&S availability
#   4  opponent S&S availability, zero-padded
#   3  player count one-hot (3/4/5)
#   1  scouts since play, normalized
#   5  play owner relative position one-hot (0=self, 1-4 seats away)
# Total: 29
METADATA_SIZE_V2 = 29
INPUT_SIZE_V2 = HAND_SIZE_V2 + PLAY_SIZE_V2 + METADATA_SIZE_V2  # 261
PLAY_START_SIZE_V2 = 15
PLAY_END_SIZE_V2 = 15
SCOUT_INSERT_SIZE_V2 = 15  # == HAND_SLOTS_V2, no more 20/21 mismatch

# V3 encoding: scalar values + precomputed pairwise signed differences
HAND_SLOTS_V3 = 15
HAND_SIZE_V3 = HAND_SLOTS_V3 * 3  # top + bottom + occupancy = 45
PLAY_END_CARDS_V3 = 6  # 4 face values + 2 presence flags
PLAY_BUFFER_SLOTS_V3 = 10
PLAY_BUFFER_SIZE_V3 = PLAY_BUFFER_SLOTS_V3 * 3  # top + bottom + occupancy = 30
PLAY_META_V3 = 5  # type(3) + strength + length
PAIRWISE_CARDS_V3 = 19  # 15 hand tops + 4 play end faces
PAIRWISE_SIZE_V3 = PAIRWISE_CARDS_V3 * (PAIRWISE_CARDS_V3 - 1) // 2  # 171
# Metadata v3 breakdown:
#   1  your hand size (/15)
#   4  opponent hand sizes (/15), zero-padded
#   1  your collected count (/15)
#   4  opponent collected counts (/15), zero-padded
#   1  your scout tokens (/5)
#   4  opponent scout tokens (/5), zero-padded
#   1  your S&S availability
#   4  opponent S&S availability, zero-padded
#   1  player count (/5)
#   1  scouts since play, normalized
#   5  play owner relative position one-hot (0=self, 1-4 seats away)
#   1  turn number (/50)
# Total: 28
METADATA_SIZE_V3 = 28
INPUT_SIZE_V3 = (HAND_SIZE_V3 + PLAY_END_CARDS_V3 + PLAY_BUFFER_SIZE_V3 +
	PLAY_META_V3 + PAIRWISE_SIZE_V3 + METADATA_SIZE_V3)  # 285
PLAY_START_SIZE_V3 = 15
PLAY_END_SIZE_V3 = 15
SCOUT_INSERT_SIZE_V3 = 15

# V4 encoding: circular CNN hand (one-hot tops) + v3 flat scalars
HAND_SLOTS_V4 = 15
CNN_CHANNELS_V4 = CARD_VALUES  # 11 one-hot channels (top face)
# Flat path reuses v3 scalar layout (no pairwise diffs)
FLAT_HAND_SIZE_V4 = HAND_SLOTS_V4 * 3  # top + bottom + occupancy = 45
FLAT_PLAY_END_V4 = 6   # 4 face values + 2 presence flags
FLAT_PLAY_BUFFER_SLOTS_V4 = 10
FLAT_PLAY_BUFFER_V4 = FLAT_PLAY_BUFFER_SLOTS_V4 * 3  # 30
FLAT_PLAY_META_V4 = 5  # type(3) + strength + length
FLAT_METADATA_V4 = METADATA_SIZE_V3  # 28 (same layout as v3)
FLAT_SIZE_V4 = (FLAT_HAND_SIZE_V4 + FLAT_PLAY_END_V4 + FLAT_PLAY_BUFFER_V4 +
	FLAT_PLAY_META_V4 + FLAT_METADATA_V4)  # 114
CNN_FLAT_SIZE_V4 = CNN_CHANNELS_V4 * HAND_SLOTS_V4  # 165 (flattened CNN input)
INPUT_SIZE_V4 = CNN_FLAT_SIZE_V4 + FLAT_SIZE_V4  # 279 (packed single tensor)
PLAY_START_SIZE_V4 = 15
PLAY_END_SIZE_V4 = 15
SCOUT_INSERT_SIZE_V4 = 15

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
	"""Return (start, end) pairs for all legal plays from this hand.
	Uses precomputed run lengths to avoid constructing Play objects."""
	n = len(hand)
	if n == 0:
		return []
	values = [c[0] for c in hand]
	# Precompute max contiguous run length starting at each position
	set_len = [1] * n
	asc_len = [1] * n
	desc_len = [1] * n
	for i in range(n - 2, -1, -1):
		if values[i] == values[i + 1]:
			set_len[i] = set_len[i + 1] + 1
		if values[i + 1] == values[i] + 1:
			asc_len[i] = asc_len[i + 1] + 1
		if values[i + 1] == values[i] - 1:
			desc_len[i] = desc_len[i + 1] + 1
	# Extract beats-check info once
	if current_play is not None:
		cp_count = current_play.count
		cp_is_set = current_play.play_type == PlayType.SET
		cp_strength = current_play.strength
	plays = []
	for start in range(n):
		max_run = max(set_len[start], asc_len[start], desc_len[start])
		for end in range(start, min(start + max_run, n)):
			length = end - start + 1
			# Classify: set > ascending run > descending run (precedence)
			if length <= set_len[start]:
				is_set = True
				strength = values[start]
			elif length <= asc_len[start]:
				is_set = False
				strength = values[end]
			elif length <= desc_len[start]:
				is_set = False
				strength = values[start]
			else:
				continue
			# Inline beats check
			if current_play is not None:
				if length < cp_count:
					continue
				if length == cp_count:
					if is_set != cp_is_set:
						if not is_set:
							continue
					elif strength <= cp_strength:
						continue
			plays.append((start, end))
	return plays

def _has_any_legal_play(hand: list, current_play: Play | None) -> bool:
	"""Check if any legal play exists (short-circuits on first match).
	Uses precomputed run lengths to avoid constructing Play objects."""
	if current_play is None:
		return len(hand) > 0
	n = len(hand)
	if n == 0:
		return False
	values = [c[0] for c in hand]
	set_len = [1] * n
	asc_len = [1] * n
	desc_len = [1] * n
	for i in range(n - 2, -1, -1):
		if values[i] == values[i + 1]:
			set_len[i] = set_len[i + 1] + 1
		if values[i + 1] == values[i] + 1:
			asc_len[i] = asc_len[i + 1] + 1
		if values[i + 1] == values[i] - 1:
			desc_len[i] = desc_len[i + 1] + 1
	cp_count = current_play.count
	cp_is_set = current_play.play_type == PlayType.SET
	cp_strength = current_play.strength
	for start in range(n):
		max_run = max(set_len[start], asc_len[start], desc_len[start])
		for end in range(start, min(start + max_run, n)):
			length = end - start + 1
			if length <= set_len[start]:
				is_set = True
				strength = values[start]
			elif length <= asc_len[start]:
				is_set = False
				strength = values[end]
			elif length <= desc_len[start]:
				is_set = False
				strength = values[start]
			else:
				continue
			if length < cp_count:
				continue
			if length == cp_count:
				if is_set != cp_is_set:
					if not is_set:
						continue
				elif strength <= cp_strength:
					continue
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
# Write directly into numpy buffers to avoid Python list building and
# torch.tensor(list) overhead. torch.from_numpy() is near-free (shared memory).

def _fill_hand(buf, offset, hand, hand_offset, num_slots=HAND_SLOTS):
	"""Write hand encoding into buf[offset:offset+num_slots*CARD_VALUES]."""
	for i in range(num_slots):
		buf[offset + i * CARD_VALUES] = 1.0
	for i, card in enumerate(hand):
		slot = (hand_offset + i) % num_slots
		pos = offset + slot * CARD_VALUES
		buf[pos] = 0.0
		buf[pos + card[0]] = 1.0

def _fill_play(buf, offset, current_play, play_offset):
	"""Write play encoding into buf[offset:offset+PLAY_SIZE]."""
	for i in range(PLAY_SLOTS):
		pos = offset + i * CARD_VALUES * 2
		buf[pos] = 1.0
		buf[pos + CARD_VALUES] = 1.0
	if current_play is None:
		return
	for i, (side_a, side_b) in enumerate(current_play.cards):
		if i >= PLAY_SLOTS:
			break
		slot = (play_offset + i) % PLAY_SLOTS
		pos = offset + slot * CARD_VALUES * 2
		buf[pos] = 0.0
		buf[pos + side_a] = 1.0
		buf[pos + CARD_VALUES] = 0.0
		buf[pos + CARD_VALUES + side_b] = 1.0

def _fill_metadata(buf, offset, game, player):
	"""Write metadata into buf[offset:offset+METADATA_SIZE].
	Reads directly from game state, skipping intermediate dict creation."""
	n = game.num_players
	players = game.players
	i = offset
	buf[i] = game.cumulative_scores[player] / 20.0; i += 1
	for j in range(4):
		if j < n - 1:
			buf[i] = game.cumulative_scores[(player + 1 + j) % n] / 20.0
		i += 1
	buf[i] = len(players[player].hand) / 15.0; i += 1
	for j in range(4):
		if j < n - 1:
			buf[i] = len(players[(player + 1 + j) % n].hand) / 15.0
		i += 1
	buf[i] = len(players[player].collected) / 20.0; i += 1
	for j in range(4):
		if j < n - 1:
			buf[i] = len(players[(player + 1 + j) % n].collected) / 20.0
		i += 1
	buf[i] = players[player].scout_tokens / 5.0; i += 1
	for j in range(4):
		if j < n - 1:
			buf[i] = players[(player + 1 + j) % n].scout_tokens / 5.0
		i += 1
	buf[i] = 1.0 if players[player].sns_available else 0.0; i += 1
	for j in range(4):
		if j < n - 1:
			buf[i] = 1.0 if players[(player + 1 + j) % n].sns_available else 0.0
		i += 1
	for pc in (3, 4, 5):
		buf[i] = 1.0 if n == pc else 0.0; i += 1
	# Scouts since play, normalized
	buf[i] = game.scouts_since_play / max(n - 1, 1); i += 1
	# Play owner relative position one-hot (5 slots: 0=self, 1-4 seats away)
	owner_rel = (game.current_play_owner - player) % n if game.current_play_owner is not None else None
	for dist in range(5):
		buf[i] = 1.0 if owner_rel == dist else 0.0; i += 1
	# Round progress
	buf[i] = game.round_number / game.total_rounds if game.total_rounds > 0 else 0.0

def _fill_play_v2(buf, offset, current_play):
	"""Write compact play encoding: 4 card one-hots + type + strength + length.
	No rotation — end cards at fixed positions so the network can read them."""
	# Initialize all 4 card slots to empty (value 0)
	for i in range(4):
		buf[offset + i * CARD_VALUES] = 1.0
	type_off = offset + PLAY_CARDS_V2
	if current_play is None:
		buf[type_off] = 1.0  # no_play
		return
	cards = current_play.cards
	# Left end card (both faces)
	left = cards[0]
	buf[offset] = 0.0
	buf[offset + left[0]] = 1.0
	buf[offset + CARD_VALUES] = 0.0
	buf[offset + CARD_VALUES + left[1]] = 1.0
	# Right end card (both faces), empty if single-card play
	if len(cards) > 1:
		right = cards[-1]
		buf[offset + 2 * CARD_VALUES] = 0.0
		buf[offset + 2 * CARD_VALUES + right[0]] = 1.0
		buf[offset + 3 * CARD_VALUES] = 0.0
		buf[offset + 3 * CARD_VALUES + right[1]] = 1.0
	# Play type: 0=no_play, 1=set, 2=run
	if current_play.play_type == PlayType.SET:
		buf[type_off + 1] = 1.0
	else:
		buf[type_off + 2] = 1.0
	# Strength: high card for runs, repeated value for sets (index 0-9 for values 1-10)
	buf[type_off + PLAY_TYPE_V2 + current_play.strength - 1] = 1.0
	# Length (index 0-9 for lengths 1-10)
	buf[type_off + PLAY_TYPE_V2 + PLAY_STRENGTH_V2 + current_play.count - 1] = 1.0

def _fill_metadata_v2(buf, offset, game, player):
	"""Write v2 metadata: no cumulative scores or round progress.
	Collected counts normalized /15 instead of /20."""
	n = game.num_players
	players = game.players
	i = offset
	buf[i] = len(players[player].hand) / 15.0; i += 1
	for j in range(4):
		if j < n - 1:
			buf[i] = len(players[(player + 1 + j) % n].hand) / 15.0
		i += 1
	buf[i] = len(players[player].collected) / 15.0; i += 1
	for j in range(4):
		if j < n - 1:
			buf[i] = len(players[(player + 1 + j) % n].collected) / 15.0
		i += 1
	buf[i] = players[player].scout_tokens / 5.0; i += 1
	for j in range(4):
		if j < n - 1:
			buf[i] = players[(player + 1 + j) % n].scout_tokens / 5.0
		i += 1
	buf[i] = 1.0 if players[player].sns_available else 0.0; i += 1
	for j in range(4):
		if j < n - 1:
			buf[i] = 1.0 if players[(player + 1 + j) % n].sns_available else 0.0
		i += 1
	for pc in (3, 4, 5):
		buf[i] = 1.0 if n == pc else 0.0; i += 1
	buf[i] = game.scouts_since_play / max(n - 1, 1); i += 1
	owner_rel = (game.current_play_owner - player) % n if game.current_play_owner is not None else None
	for dist in range(5):
		buf[i] = 1.0 if owner_rel == dist else 0.0; i += 1

def encode_state_v2(game, player, hand_offset):
	"""V2 state encoding: compact play (no rotation), smaller hand."""
	buf = np.zeros(INPUT_SIZE_V2, dtype=np.float32)
	_fill_hand(buf, 0, game.players[player].hand, hand_offset, HAND_SLOTS_V2)
	_fill_play_v2(buf, HAND_SIZE_V2, game.current_play)
	_fill_metadata_v2(buf, HAND_SIZE_V2 + PLAY_SIZE_V2, game, player)
	return torch.from_numpy(buf)

def encode_hand_both_orientations_v2(game, player, hand_offset):
	"""V2 flip decision encoding: both hand orientations."""
	buf = np.zeros(INPUT_SIZE_V2, dtype=np.float32)
	hand = game.players[player].hand
	_fill_hand(buf, 0, hand, hand_offset, HAND_SLOTS_V2)
	_fill_play_v2(buf, HAND_SIZE_V2, game.current_play)
	_fill_metadata_v2(buf, HAND_SIZE_V2 + PLAY_SIZE_V2, game, player)
	buf_flip = buf.copy()
	buf_flip[:HAND_SIZE_V2] = 0.0
	_fill_hand(buf_flip, 0, [(b, a) for a, b in hand], hand_offset, HAND_SLOTS_V2)
	return torch.from_numpy(buf), torch.from_numpy(buf_flip)

def encode_state(game, player, hand_offset, play_offset):
	"""Convert game state to a 1D float input tensor."""
	buf = np.zeros(INPUT_SIZE, dtype=np.float32)
	_fill_hand(buf, 0, game.players[player].hand, hand_offset)
	_fill_play(buf, HAND_SIZE, game.current_play, play_offset)
	_fill_metadata(buf, HAND_SIZE + PLAY_SIZE, game, player)
	return torch.from_numpy(buf)

def encode_hand_both_orientations(game, player, hand_offset, play_offset):
	"""Return state tensors for both hand orientations (for flip decision)."""
	buf = np.zeros(INPUT_SIZE, dtype=np.float32)
	hand = game.players[player].hand
	_fill_hand(buf, 0, hand, hand_offset)
	_fill_play(buf, HAND_SIZE, game.current_play, play_offset)
	_fill_metadata(buf, HAND_SIZE + PLAY_SIZE, game, player)
	# Copy and overwrite hand portion for flipped orientation
	buf_flip = buf.copy()
	buf_flip[:HAND_SIZE] = 0.0
	_fill_hand(buf_flip, 0, [(b, a) for a, b in hand], hand_offset)
	return torch.from_numpy(buf), torch.from_numpy(buf_flip)

# --- V3 encoding functions ---

def _fill_hand_v3(buf, offset, hand, hand_offset):
	"""Write v3 hand: 15 top face scalars + 15 bottom face scalars + 15 occupancy.
	Empty slots stay 0.0 (buf is zero-initialized)."""
	for i, card in enumerate(hand):
		slot = (hand_offset + i) % HAND_SLOTS_V3
		buf[offset + slot] = card[0] / 10.0
		buf[offset + HAND_SLOTS_V3 + slot] = card[1] / 10.0
		buf[offset + 2 * HAND_SLOTS_V3 + slot] = 1.0

def _fill_play_end_v3(buf, offset, current_play):
	"""Write 4 end card face values (value/10) + 2 presence flags. Fixed position."""
	if current_play is None:
		return
	cards = current_play.cards
	left = cards[0]
	buf[offset] = left[0] / 10.0
	buf[offset + 1] = left[1] / 10.0
	buf[offset + 4] = 1.0
	if len(cards) > 1:
		right = cards[-1]
		buf[offset + 2] = right[0] / 10.0
		buf[offset + 3] = right[1] / 10.0
		buf[offset + 5] = 1.0

def _fill_play_buffer_v3(buf, offset, current_play, play_offset):
	"""Write 10-slot rotated play buffer: top + bottom + occupancy scalars."""
	if current_play is None:
		return
	for i, card in enumerate(current_play.cards):
		if i >= PLAY_BUFFER_SLOTS_V3:
			break
		slot = (play_offset + i) % PLAY_BUFFER_SLOTS_V3
		buf[offset + slot] = card[0] / 10.0
		buf[offset + PLAY_BUFFER_SLOTS_V3 + slot] = card[1] / 10.0
		buf[offset + 2 * PLAY_BUFFER_SLOTS_V3 + slot] = 1.0

def _fill_play_meta_v3(buf, offset, current_play):
	"""Write play type (3 one-hot) + strength/10 + length/10."""
	if current_play is None:
		buf[offset] = 1.0  # no_play
		return
	if current_play.play_type == PlayType.SET:
		buf[offset + 1] = 1.0
	else:
		buf[offset + 2] = 1.0
	buf[offset + 3] = current_play.strength / 10.0
	buf[offset + 4] = current_play.count / 10.0

def _build_pairwise_arrays_v3(hand, hand_offset, current_play):
	"""Build 19-element value/present arrays for pairwise diffs.
	Indices 0-14: hand top face slots (rotated), 15-18: play end faces."""
	values = np.zeros(PAIRWISE_CARDS_V3, dtype=np.float32)
	present = np.zeros(PAIRWISE_CARDS_V3, dtype=np.bool_)
	for i, card in enumerate(hand):
		slot = (hand_offset + i) % HAND_SLOTS_V3
		values[slot] = card[0] / 10.0
		present[slot] = True
	if current_play is not None:
		cards = current_play.cards
		left = cards[0]
		values[15] = left[0] / 10.0
		values[16] = left[1] / 10.0
		present[15] = True
		present[16] = True
		if len(cards) > 1:
			right = cards[-1]
			values[17] = right[0] / 10.0
			values[18] = right[1] / 10.0
			present[17] = True
			present[18] = True
	return values, present

def _fill_pairwise_v3(buf, offset, values, present):
	"""Write 171 pairwise similarities for 19 card values.
	Encodes (raw_diff + 10) / 20: equal=0.5, absent=0.0, range [0.05, 0.95]."""
	idx = offset
	for i in range(PAIRWISE_CARDS_V3):
		for j in range(i + 1, PAIRWISE_CARDS_V3):
			if present[i] and present[j]:
				buf[idx] = (values[i] - values[j] + 1.0) / 2.0
			idx += 1

def _fill_metadata_v3(buf, offset, game, player):
	"""Write v3 metadata: scalar player count, turn number, no scores."""
	n = game.num_players
	players = game.players
	i = offset
	buf[i] = len(players[player].hand) / 15.0; i += 1
	for j in range(4):
		if j < n - 1:
			buf[i] = len(players[(player + 1 + j) % n].hand) / 15.0
		i += 1
	buf[i] = len(players[player].collected) / 15.0; i += 1
	for j in range(4):
		if j < n - 1:
			buf[i] = len(players[(player + 1 + j) % n].collected) / 15.0
		i += 1
	buf[i] = players[player].scout_tokens / 5.0; i += 1
	for j in range(4):
		if j < n - 1:
			buf[i] = players[(player + 1 + j) % n].scout_tokens / 5.0
		i += 1
	buf[i] = 1.0 if players[player].sns_available else 0.0; i += 1
	for j in range(4):
		if j < n - 1:
			buf[i] = 1.0 if players[(player + 1 + j) % n].sns_available else 0.0
		i += 1
	buf[i] = n / 5.0; i += 1
	buf[i] = game.scouts_since_play / max(n - 1, 1); i += 1
	owner_rel = (game.current_play_owner - player) % n if game.current_play_owner is not None else None
	for dist in range(5):
		buf[i] = 1.0 if owner_rel == dist else 0.0; i += 1
	buf[i] = game.turn_number / 50.0

def encode_state_v3(game, player, hand_offset, play_offset):
	"""V3 state encoding: scalar values + pairwise differences."""
	buf = np.zeros(INPUT_SIZE_V3, dtype=np.float32)
	hand = game.players[player].hand
	off = 0
	_fill_hand_v3(buf, off, hand, hand_offset)
	off += HAND_SIZE_V3
	_fill_play_end_v3(buf, off, game.current_play)
	off += PLAY_END_CARDS_V3
	_fill_play_buffer_v3(buf, off, game.current_play, play_offset)
	off += PLAY_BUFFER_SIZE_V3
	_fill_play_meta_v3(buf, off, game.current_play)
	off += PLAY_META_V3
	values, present = _build_pairwise_arrays_v3(hand, hand_offset, game.current_play)
	_fill_pairwise_v3(buf, off, values, present)
	off += PAIRWISE_SIZE_V3
	_fill_metadata_v3(buf, off, game, player)
	return torch.from_numpy(buf)

def encode_hand_both_orientations_v3(game, player, hand_offset, play_offset):
	"""V3 flip decision encoding: both hand orientations."""
	buf = np.zeros(INPUT_SIZE_V3, dtype=np.float32)
	hand = game.players[player].hand
	_fill_hand_v3(buf, 0, hand, hand_offset)
	play_off = HAND_SIZE_V3
	_fill_play_end_v3(buf, play_off, game.current_play)
	buf_off = play_off + PLAY_END_CARDS_V3
	_fill_play_buffer_v3(buf, buf_off, game.current_play, play_offset)
	meta_off = buf_off + PLAY_BUFFER_SIZE_V3
	_fill_play_meta_v3(buf, meta_off, game.current_play)
	pw_off = meta_off + PLAY_META_V3
	values, present = _build_pairwise_arrays_v3(hand, hand_offset, game.current_play)
	_fill_pairwise_v3(buf, pw_off, values, present)
	md_off = pw_off + PAIRWISE_SIZE_V3
	_fill_metadata_v3(buf, md_off, game, player)
	# Flipped copy: hand top/bottom swap changes hand section + pairwise
	buf_flip = buf.copy()
	flipped = [(b, a) for a, b in hand]
	buf_flip[:HAND_SIZE_V3] = 0.0
	_fill_hand_v3(buf_flip, 0, flipped, hand_offset)
	buf_flip[pw_off:pw_off + PAIRWISE_SIZE_V3] = 0.0
	values_f, present_f = _build_pairwise_arrays_v3(flipped, hand_offset, game.current_play)
	_fill_pairwise_v3(buf_flip, pw_off, values_f, present_f)
	return torch.from_numpy(buf), torch.from_numpy(buf_flip)

# --- V4 encoding functions ---

def _fill_hand_cnn_v4(buf, hand, hand_offset):
	"""Write 11-channel one-hot hand for CNN input.
	buf shape: (11, 15). Channel 0 = empty, channels 1-10 = card values."""
	# Initialize all slots to empty (channel 0)
	buf[0, :] = 1.0
	for i, card in enumerate(hand):
		slot = (hand_offset + i) % HAND_SLOTS_V4
		buf[0, slot] = 0.0
		buf[card[0], slot] = 1.0

def encode_state_v4(game, player, hand_offset, play_offset):
	"""V4 state encoding: packed single tensor (279,).
	Layout: [CNN_flat (165) | flat_scalars (114)].
	CNN portion is 11×15 one-hot hand flattened row-major;
	the network reshapes it back to (11, 15) internally."""
	buf = np.zeros(INPUT_SIZE_V4, dtype=np.float32)
	hand = game.players[player].hand
	# CNN hand: one-hot top faces, written into (11, 15) view then flattened
	cnn_view = buf[:CNN_FLAT_SIZE_V4].reshape(CNN_CHANNELS_V4, HAND_SLOTS_V4)
	_fill_hand_cnn_v4(cnn_view, hand, hand_offset)
	# Flat path: reuse v3 scalar functions (hand scalars, play, metadata)
	flat_off = CNN_FLAT_SIZE_V4
	_fill_hand_v3(buf, flat_off, hand, hand_offset)
	flat_off += FLAT_HAND_SIZE_V4
	_fill_play_end_v3(buf, flat_off, game.current_play)
	flat_off += FLAT_PLAY_END_V4
	_fill_play_buffer_v3(buf, flat_off, game.current_play, play_offset)
	flat_off += FLAT_PLAY_BUFFER_V4
	_fill_play_meta_v3(buf, flat_off, game.current_play)
	flat_off += FLAT_PLAY_META_V4
	_fill_metadata_v3(buf, flat_off, game, player)
	return torch.from_numpy(buf)

def encode_hand_both_orientations_v4(game, player, hand_offset, play_offset):
	"""V4 flip decision encoding: both hand orientations.
	Returns (tensor, tensor_flip) — same interface as v2/v3."""
	hand = game.players[player].hand
	flipped = [(b, a) for a, b in hand]
	# Normal orientation
	state = encode_state_v4(game, player, hand_offset, play_offset)
	# Flipped: copy and overwrite hand-dependent portions
	buf_flip = state.numpy().copy()
	# Overwrite CNN hand portion
	cnn_view = buf_flip[:CNN_FLAT_SIZE_V4].reshape(CNN_CHANNELS_V4, HAND_SLOTS_V4)
	cnn_view[:] = 0.0
	_fill_hand_cnn_v4(cnn_view, flipped, hand_offset)
	# Overwrite flat hand scalars (top + bottom + occupancy)
	flat_hand_start = CNN_FLAT_SIZE_V4
	buf_flip[flat_hand_start:flat_hand_start + FLAT_HAND_SIZE_V4] = 0.0
	_fill_hand_v3(buf_flip, flat_hand_start, flipped, hand_offset)
	return state, torch.from_numpy(buf_flip)

# --- Mask functions ---

def get_action_type_mask(game: Game, legal_plays: list[tuple[int, int]], max_hand: int = HAND_SLOTS) -> np.ndarray:
	"""Return a boolean mask of shape [9] where True = legal action type."""
	mask = np.zeros(ACTION_TYPE_SIZE, dtype=np.bool_)
	player = game.players[game.current_player]
	hand = player.hand
	has_play = game.current_play is not None

	if game.phase == Phase.SNS_PLAY:
		mask[0] = True
		return mask

	# Play: legal if no current play or at least one legal play exists
	if legal_plays:
		mask[0] = True

	# Scout/S&S only if there's a table play and hand has room
	if has_play and len(hand) < max_hand:
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

def get_play_start_mask(legal_plays: list[tuple[int, int]], hand_offset: int, num_slots: int = HAND_SLOTS) -> np.ndarray:
	"""Return a boolean mask for legal play start positions.
	Positions correspond to encoded hand slots (with hand_offset applied)."""
	mask = np.zeros(num_slots, dtype=np.bool_)
	for start, end in legal_plays:
		slot = (hand_offset + start) % num_slots
		mask[slot] = True
	return mask

def get_play_end_mask(legal_plays: list[tuple[int, int]], start: int, hand_offset: int, num_slots: int = HAND_SLOTS) -> np.ndarray:
	"""Return a boolean mask for legal play end positions.
	Args:
		start: The raw hand index (before offset) of the play start.
	"""
	mask = np.zeros(num_slots, dtype=np.bool_)
	for s, end in legal_plays:
		if s == start:
			slot = (hand_offset + end) % num_slots
			mask[slot] = True
	return mask

def get_scout_insert_mask(game: Game, hand_offset: int, num_slots: int = SCOUT_INSERT_SIZE) -> np.ndarray:
	"""Return a boolean mask for legal scout insert positions.
	Positions correspond to encoded hand slots (with hand_offset applied)."""
	hand_len = len(game.players[game.current_player].hand)
	mask = np.zeros(num_slots, dtype=np.bool_)
	for pos in range(min(hand_len + 1, num_slots)):
		slot = (hand_offset + pos) % num_slots
		mask[slot] = True
	return mask

def get_sns_insert_mask(game: Game, left_end: bool, flip: bool, hand_offset: int, num_slots: int = SCOUT_INSERT_SIZE) -> np.ndarray:
	"""Return a boolean mask for S&S insert positions.
	Only positions where inserting the scouted card yields a hand with
	at least one legal play against the reduced play are legal.
	Positions correspond to encoded hand slots (with hand_offset applied)."""
	hand = game.players[game.current_player].hand
	play_cards = list(game.current_play.cards)
	remaining = list(play_cards)
	card = remaining.pop(0) if left_end else remaining.pop()
	if flip:
		card = (card[1], card[0])
	reduced_play = Play.from_cards(remaining) if remaining else None
	mask = np.zeros(num_slots, dtype=np.bool_)
	for pos in range(min(len(hand) + 1, num_slots)):
		new_hand = hand[:pos] + [card] + hand[pos:]
		if _has_any_legal_play(new_hand, reduced_play):
			slot = (hand_offset + pos) % num_slots
			mask[slot] = True
	return mask

def decode_slot_to_hand_index(slot: int, hand_offset: int, num_slots: int = HAND_SLOTS) -> int:
	"""Convert an encoded slot position back to a raw hand index."""
	return (slot - hand_offset) % num_slots

# Override hot functions with Cython if compiled extension is available
try:
	from fast_game import get_legal_plays, _has_any_legal_play, _sns_variant_legal
except ImportError:
	pass
