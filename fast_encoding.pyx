# cython: boundscheck=False, wraparound=False, cdivision=True
"""Fast C implementations of v6 state encoding and action masking.

Replaces encode_state_v6 and get_flat_action_mask from encoding.py.
The inner loops run as C with minimal Python object manipulation."""

import numpy as np
import torch
from game import PlayType, Phase

_PLAY_TYPE_SET = PlayType.SET
_PHASE_SNS_PLAY = Phase.SNS_PLAY

DEF H = 16
DEF FLAT_ACTION_SIZE = 384
DEF MAX_PLAYERS = 5
DEF MAX_PLAY_LEN = 10


# Duplicated from fast_game.pyx to avoid build coupling between .pyx files
cdef bint _has_any_legal_play_c(int* values, int n,
								int cp_count, bint cp_is_set, int cp_strength):
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
			if is_set != cp_is_set:
				if is_set:
					return True
				continue
			if strength > cp_strength:
				return True
	return False


cpdef object encode_state_v6(object game, int player, int hand_offset, bint forced_play=False):
	cdef int N = game.num_values
	cdef int n_players = game.num_players
	cdef int slot_size = N + 2
	cdef int hand_oh = H * slot_size
	cdef int hand_total = hand_oh + H
	cdef int card_size = N + 1
	cdef int scout_dim = 4 * card_size + 8
	cdef int play_dim = 21
	cdef int meta_dim = 28
	cdef int total = hand_total + scout_dim + play_dim + meta_dim
	cdef int hand_len, i, j, s, pos, off, p_idx, dist
	cdef int h_top[H], h_bot[H]
	cdef bint has_play, cp_is_set
	cdef int pc_len, pc_top[MAX_PLAY_LEN], pc_bot[MAX_PLAY_LEN]
	cdef int cp_strength, cp_count
	cdef int p_hand_len[MAX_PLAYERS], p_coll_len[MAX_PLAYERS], p_scout_tok[MAX_PLAYERS]
	cdef bint p_sns[MAX_PLAYERS]
	cdef int owner_rel, scouts_since, max_ssp
	cdef int limit, card_idx, buf_idx

	buf = np.zeros(total, dtype=np.float32)
	cdef float[:] b = buf

	# Extract hand card values at Python boundary
	hand = game.players[player].hand
	hand_len = len(hand)
	for i in range(hand_len):
		card = hand[i]
		h_top[i] = card[0]
		h_bot[i] = card[1]

	# Extract play card values
	current_play = game.current_play
	has_play = current_play is not None
	pc_len = 0
	cp_is_set = False
	cp_strength = 0
	cp_count = 0
	if has_play:
		play_cards = current_play.cards
		pc_len = len(play_cards)
		for i in range(pc_len):
			pc_top[i] = play_cards[i][0]
			pc_bot[i] = play_cards[i][1]
		cp_is_set = current_play.play_type is _PLAY_TYPE_SET
		cp_strength = current_play.strength
		cp_count = current_play.count

	# Extract per-player data for metadata
	players = game.players
	for j in range(n_players):
		ps = players[j]
		p_hand_len[j] = len(ps.hand)
		p_coll_len[j] = len(ps.collected)
		p_scout_tok[j] = ps.scout_tokens
		p_sns[j] = ps.sns_available

	owner_rel = -1
	owner = game.current_play_owner
	if owner is not None:
		owner_rel = (owner - player) % n_players
	scouts_since = game.scouts_since_play

	# --- Pure C from here ---

	# Hand one-hot + scalar (H slots x (N+2) dims)
	off = 0
	for s in range(H):
		b[off + s * slot_size + N] = 1.0  # empty flag
	for i in range(hand_len):
		s = (hand_offset + i) % H
		pos = off + s * slot_size
		b[pos + N] = 0.0
		b[pos + h_top[i] - 1] = 1.0
		b[pos + N + 1] = <float>h_top[i] / N

	# Hand bottom scalars (H dims)
	off = hand_oh
	for i in range(hand_len):
		s = (hand_offset + i) % H
		b[off + s] = <float>h_bot[i] / N

	# Scout cards (4x(N+1) one-hot + 8 scalars)
	off = hand_total
	if not has_play:
		for i in range(4):
			b[off + i * card_size + N] = 1.0
	else:
		b[off + pc_top[0] - 1] = 1.0
		b[off + card_size + pc_bot[0] - 1] = 1.0
		if pc_len > 1:
			b[off + 2 * card_size + pc_top[pc_len - 1] - 1] = 1.0
			b[off + 3 * card_size + pc_bot[pc_len - 1] - 1] = 1.0
		else:
			b[off + 2 * card_size + N] = 1.0
			b[off + 3 * card_size + N] = 1.0
		s = off + 4 * card_size
		b[s] = <float>pc_top[0] / N
		b[s + 1] = <float>pc_bot[0] / N
		b[s + 4] = <float>pc_bot[0] / N
		b[s + 5] = <float>pc_top[0] / N
		if pc_len > 1:
			b[s + 2] = <float>pc_top[pc_len - 1] / N
			b[s + 3] = <float>pc_bot[pc_len - 1] / N
			b[s + 6] = <float>pc_bot[pc_len - 1] / N
			b[s + 7] = <float>pc_top[pc_len - 1] / N

	# Play buffer (2x4 card scalars + 5 metadata = 21 dims)
	off = hand_total + scout_dim
	if not has_play:
		b[off + 16] = 1.0
	else:
		limit = pc_len if pc_len < 4 else 4
		for i in range(limit):
			b[off + i * 2] = <float>pc_top[i] / N
			b[off + i * 2 + 1] = <float>pc_bot[i] / N
		for i in range(limit):
			card_idx = pc_len - 1 - i
			buf_idx = 3 - i
			b[off + 8 + buf_idx * 2] = <float>pc_top[card_idx] / N
			b[off + 8 + buf_idx * 2 + 1] = <float>pc_bot[card_idx] / N
		pos = off + 16
		if cp_is_set:
			b[pos + 1] = 1.0
		else:
			b[pos + 2] = 1.0
		b[pos + 3] = <float>cp_strength / N
		b[pos + 4] = <float>cp_count / 10.0

	# Metadata (28 dims)
	off = hand_total + scout_dim + play_dim
	max_ssp = n_players - 1
	if max_ssp < 1:
		max_ssp = 1
	i = off
	b[i] = <float>p_hand_len[player] / H; i += 1
	for j in range(4):
		if j < n_players - 1:
			p_idx = (player + 1 + j) % n_players
			b[i] = <float>p_hand_len[p_idx] / H
		i += 1
	b[i] = <float>p_coll_len[player] / H; i += 1
	for j in range(4):
		if j < n_players - 1:
			p_idx = (player + 1 + j) % n_players
			b[i] = <float>p_coll_len[p_idx] / H
		i += 1
	b[i] = <float>p_scout_tok[player] / 5.0; i += 1
	for j in range(4):
		if j < n_players - 1:
			p_idx = (player + 1 + j) % n_players
			b[i] = <float>p_scout_tok[p_idx] / 5.0
		i += 1
	b[i] = 1.0 if p_sns[player] else 0.0; i += 1
	for j in range(4):
		if j < n_players - 1:
			p_idx = (player + 1 + j) % n_players
			b[i] = 1.0 if p_sns[p_idx] else 0.0
		i += 1
	b[i] = <float>n_players / 5.0; i += 1
	b[i] = <float>scouts_since / max_ssp; i += 1
	for dist in range(5):
		b[i] = 1.0 if owner_rel == dist else 0.0; i += 1
	b[i] = 1.0 if forced_play else 0.0

	return torch.from_numpy(buf)


cpdef object get_flat_action_mask(object game, int player, list legal_plays, int hand_offset):
	cdef int hand_len, i, j, s_slot, e_slot, start, end
	cdef int hand_values[H]
	cdef int play_len, left_top, left_bot, right_top, right_bot
	cdef int card_vals[4]
	cdef bint card_avail[4]
	cdef int cc, base, pos
	cdef int rp_count
	cdef bint has_rp
	cdef int rp_vals_left[MAX_PLAY_LEN], rp_vals_right[MAX_PLAY_LEN]
	cdef bint rp_is_set_left, rp_is_set_right
	cdef int rp_str_left, rp_str_right
	cdef int new_values[21]
	cdef int new_n, card_val, rp_str
	cdef bint left_end, rp_is_set

	mask_np = np.zeros(FLAT_ACTION_SIZE, dtype=np.uint8)
	cdef unsigned char[:] mask = mask_np

	hand = game.players[player].hand
	hand_len = len(hand)
	for i in range(hand_len):
		hand_values[i] = hand[i][0]

	# Play region [0..255]: index = start_slot * H + end_slot
	for play in legal_plays:
		start = play[0]
		end = play[1]
		s_slot = (hand_offset + start) % H
		e_slot = (hand_offset + end) % H
		mask[s_slot * H + e_slot] = 1

	if game.phase is _PHASE_SNS_PLAY:
		return torch.from_numpy(mask_np.view(np.bool_))

	# Scout and S&S regions
	current_play = game.current_play
	if current_play is not None and hand_len < H:
		play_cards = current_play.cards
		play_len = len(play_cards)
		left_top = play_cards[0][0]
		left_bot = play_cards[0][1]
		right_top = 0
		right_bot = 0
		if play_len > 1:
			right_top = play_cards[play_len - 1][0]
			right_bot = play_cards[play_len - 1][1]

		# Scouted card showing value: 0=left normal, 1=left flip, 2=right normal, 3=right flip
		card_vals[0] = left_top
		card_vals[1] = left_bot
		card_vals[2] = right_top
		card_vals[3] = right_bot
		card_avail[0] = True
		card_avail[1] = True
		card_avail[2] = play_len > 1
		card_avail[3] = play_len > 1

		# Scout region [256..319]
		for cc in range(4):
			if not card_avail[cc]:
				continue
			base = 256 + cc * H
			for pos in range(hand_len + 1):
				mask[base + (hand_offset + pos) % H] = 1

		# S&S region [320..383]
		if game.players[player].sns_available:
			# Precompute reduced play properties for left-end and right-end scouting
			rp_count = play_len - 1
			has_rp = rp_count > 0
			rp_is_set_left = True
			rp_str_left = 0
			rp_is_set_right = True
			rp_str_right = 0

			if has_rp:
				# Left-end: remaining = play_cards[1:]
				for j in range(rp_count):
					rp_vals_left[j] = play_cards[1 + j][0]
				for j in range(1, rp_count):
					if rp_vals_left[j] != rp_vals_left[0]:
						rp_is_set_left = False
						break
				if rp_is_set_left:
					rp_str_left = rp_vals_left[0]
				else:
					rp_str_left = rp_vals_left[0]
					for j in range(1, rp_count):
						if rp_vals_left[j] > rp_str_left:
							rp_str_left = rp_vals_left[j]
				# Right-end: remaining = play_cards[:-1]
				for j in range(rp_count):
					rp_vals_right[j] = play_cards[j][0]
				for j in range(1, rp_count):
					if rp_vals_right[j] != rp_vals_right[0]:
						rp_is_set_right = False
						break
				if rp_is_set_right:
					rp_str_right = rp_vals_right[0]
				else:
					rp_str_right = rp_vals_right[0]
					for j in range(1, rp_count):
						if rp_vals_right[j] > rp_str_right:
							rp_str_right = rp_vals_right[j]

			new_n = hand_len + 1
			for cc in range(4):
				if not card_avail[cc]:
					continue
				left_end = cc < 2
				card_val = card_vals[cc]
				if left_end:
					rp_is_set = rp_is_set_left
					rp_str = rp_str_left
				else:
					rp_is_set = rp_is_set_right
					rp_str = rp_str_right

				base = 320 + cc * H
				# Incremental new_values: start with card at position 0
				new_values[0] = card_val
				for j in range(hand_len):
					new_values[j + 1] = hand_values[j]
				for pos in range(new_n):
					if pos > 0:
						new_values[pos - 1] = new_values[pos]
						new_values[pos] = card_val
					if not has_rp:
						mask[base + (hand_offset + pos) % H] = 1
					elif _has_any_legal_play_c(new_values, new_n, rp_count, rp_is_set, rp_str):
						mask[base + (hand_offset + pos) % H] = 1

	return torch.from_numpy(mask_np.view(np.bool_))
