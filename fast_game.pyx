# cython: boundscheck=False, wraparound=False, cdivision=True
"""Fast C implementations of legal move computation for Scout bot training.

Replaces the pure-Python get_legal_plays, _has_any_legal_play, and
_sns_variant_legal from encoding.py. The inner loops run as C with no
Python object manipulation, giving 10-30x speedup on these functions."""

from game import PlayType
_PLAY_TYPE_SET = PlayType.SET


cdef bint _has_any_legal_play_c(int* values, int n,
								int cp_count, bint cp_is_set, int cp_strength):
	"""Pure C legal-play check. Returns True on first legal play found."""
	cdef int set_len[21], asc_len[21], desc_len[21]
	cdef int i, start, end, length, max_run, strength, end_limit
	cdef bint is_set

	for i in range(n):
		set_len[i] = 1
		asc_len[i] = 1
		desc_len[i] = 1
	for i in range(n - 2, -1, -1):
		if values[i] == values[i + 1]:
			set_len[i] = set_len[i + 1] + 1
		if values[i + 1] == values[i] + 1:
			asc_len[i] = asc_len[i + 1] + 1
		if values[i + 1] == values[i] - 1:
			desc_len[i] = desc_len[i + 1] + 1

	for start in range(n):
		max_run = set_len[start]
		if asc_len[start] > max_run:
			max_run = asc_len[start]
		if desc_len[start] > max_run:
			max_run = desc_len[start]
		end_limit = start + max_run
		if end_limit > n:
			end_limit = n
		for end in range(start, end_limit):
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
			if length > cp_count:
				return True
			# length == cp_count
			if is_set != cp_is_set:
				if is_set:
					return True
				continue
			if strength > cp_strength:
				return True
	return False


cpdef list get_legal_plays(list hand, object current_play):
	"""Return (start, end) pairs for all legal plays from this hand.
	Uses precomputed run lengths to avoid constructing Play objects."""
	cdef int n = len(hand)
	if n == 0:
		return []

	cdef int values[20]
	cdef int set_len[20], asc_len[20], desc_len[20]
	cdef int i, start, end, length, max_run, strength, end_limit
	cdef bint is_set, has_cp
	cdef int cp_count, cp_strength
	cdef bint cp_is_set

	for i in range(n):
		values[i] = hand[i][0]

	for i in range(n):
		set_len[i] = 1
		asc_len[i] = 1
		desc_len[i] = 1
	for i in range(n - 2, -1, -1):
		if values[i] == values[i + 1]:
			set_len[i] = set_len[i + 1] + 1
		if values[i + 1] == values[i] + 1:
			asc_len[i] = asc_len[i + 1] + 1
		if values[i + 1] == values[i] - 1:
			desc_len[i] = desc_len[i + 1] + 1

	has_cp = current_play is not None
	if has_cp:
		cp_count = current_play.count
		cp_is_set = current_play.play_type is _PLAY_TYPE_SET
		cp_strength = current_play.strength

	cdef list plays = []
	for start in range(n):
		max_run = set_len[start]
		if asc_len[start] > max_run:
			max_run = asc_len[start]
		if desc_len[start] > max_run:
			max_run = desc_len[start]
		end_limit = start + max_run
		if end_limit > n:
			end_limit = n
		for end in range(start, end_limit):
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
			if has_cp:
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


cpdef bint _has_any_legal_play(list hand, object current_play):
	"""Check if any legal play exists (short-circuits on first match)."""
	if current_play is None:
		return len(hand) > 0
	cdef int n = len(hand)
	if n == 0:
		return False
	cdef int values[20]
	cdef int i
	for i in range(n):
		values[i] = hand[i][0]
	return _has_any_legal_play_c(
		values, n,
		current_play.count,
		current_play.play_type is _PLAY_TYPE_SET,
		current_play.strength)


cpdef bint _sns_variant_legal(list hand, list play_cards, bint left_end, bint flip):
	"""Check if an S&S variant is legal: after scouting, does any insert
	position yield a hand with a legal play against the reduced play?
	Inlines Play.from_cards logic to avoid Python object creation."""
	cdef int n = len(hand)
	cdef int pc_len = len(play_cards)
	cdef tuple card
	cdef int card_val

	# Pop card from appropriate end
	if left_end:
		card = play_cards[0]
	else:
		card = play_cards[pc_len - 1]
	if flip:
		card_val = card[1]
	else:
		card_val = card[0]

	# Compute reduced play properties (inline Play.from_cards)
	cdef int rp_count = pc_len - 1
	cdef bint has_rp = rp_count > 0
	cdef int rp_values[10]
	cdef bint rp_is_set = True
	cdef int rp_strength = 0
	cdef int i, rp_start

	if has_rp:
		rp_start = 1 if left_end else 0
		for i in range(rp_count):
			rp_values[i] = play_cards[rp_start + i][0]
		# Check if set (all same value)
		for i in range(1, rp_count):
			if rp_values[i] != rp_values[0]:
				rp_is_set = False
				break
		if rp_is_set:
			rp_strength = rp_values[0]
		else:
			# Run — strength is max value
			rp_strength = rp_values[0]
			for i in range(1, rp_count):
				if rp_values[i] > rp_strength:
					rp_strength = rp_values[i]

	# Extract hand values into C array
	cdef int hand_values[20]
	for i in range(n):
		hand_values[i] = hand[i][0]

	# Try each insert position using incremental array building
	cdef int new_values[21]
	cdef int new_n = n + 1

	# Initial: [card_val, hand_values[0], hand_values[1], ...]
	new_values[0] = card_val
	for i in range(n):
		new_values[i + 1] = hand_values[i]

	for insert_pos in range(new_n):
		if insert_pos > 0:
			# Shift card one position right
			new_values[insert_pos - 1] = new_values[insert_pos]
			new_values[insert_pos] = card_val
		if not has_rp:
			return True  # No play to beat, any non-empty hand can play
		if _has_any_legal_play_c(new_values, new_n, rp_count, rp_is_set, rp_strength):
			return True
	return False
