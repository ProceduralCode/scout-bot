"""Numba CUDA rollout engine — game logic as CUDA kernels, one thread per game.

Replaces PyTorch tensor-op dispatch in gpu_engine.py with Numba @cuda.jit kernels.
Network inference + sampling stay in PyTorch. Data interchange is zero-copy via
numba.cuda.as_cuda_array() on PyTorch CUDA tensors.

Reference implementation: gpu_engine.py (used as test oracle for each kernel).
"""
from __future__ import annotations
import warnings
import torch
from torch import Tensor
from numba import cuda
import numba.types as nbt

from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", "Grid size", category=NumbaPerformanceWarning)

from gpu_engine import GpuGameState, from_snapshots, compute_scores, compute_scores_tensor

# ── Constants ────────────────────────────────────────────────────────────────

H = 16         # hand slots
MAX_P = 5      # max players
MAX_PLAY = 16  # max play length
N = 10         # card values (1-10)
FLAT_ACTION_SIZE = 384

PHASE_TURN = 0
PHASE_SNS_PLAY = 1
PLAY_NONE = 0
PLAY_SET = 1
PLAY_RUN = 2

MAX_STEPS = 100

TPB = 256  # threads per block


# ── Helpers ──────────────────────────────────────────────────────────────────

def _grid(B: int) -> int:
	"""Grid size for B games."""
	return (B + TPB - 1) // TPB


# ── compute_legal_plays_kernel ───────────────────────────────────────────────

@cuda.jit
def compute_legal_plays_kernel(
	hands_show,    # [B, MAX_P, H] int8
	hand_len,      # [B, MAX_P] int8
	current_player,# [B] int8
	play_len,      # [B] int8
	play_type,     # [B] int8
	play_strength, # [B] int8
	done,          # [B] bool
	out_legal,     # [B, H, H] bool — output
	B,             # scalar
):
	b = cuda.grid(1)
	if b >= B:
		return

	# Zero output
	for s in range(H):
		for e in range(H):
			out_legal[b, s, e] = False

	if done[b]:
		return

	cp = current_player[b]
	hl = hand_len[b, cp]
	p_len = play_len[b]
	p_type = play_type[b]
	p_str = play_strength[b]
	no_play = (p_len == 0)

	# Read current player's hand showing values into local array
	vals = cuda.local.array(H, nbt.int8)
	for i in range(H):
		vals[i] = hands_show[b, cp, i]

	# Check each (start, end) pair
	for s in range(hl):
		for e in range(s, hl):
			span = e - s  # number of diffs = e - s
			length = span + 1

			# Check type: set, ascending run, or descending run
			is_set = True
			is_asc = True
			is_desc = True
			for k in range(s, e):
				d = vals[k + 1] - vals[k]
				if d != 0:
					is_set = False
				if d != 1:
					is_asc = False
				if d != -1:
					is_desc = False

			if not (is_set or is_asc or is_desc):
				continue

			# Strength: ascending uses end value, set/descending use start value
			if is_asc:
				strength = vals[e]
			else:
				strength = vals[s]

			# Beats check
			if no_play:
				out_legal[b, s, e] = True
			elif length > p_len:
				out_legal[b, s, e] = True
			elif length == p_len:
				# is_set_range beats run (p_type == PLAY_RUN)
				cp_is_set = (p_type == PLAY_SET)
				if is_set and not cp_is_set:
					out_legal[b, s, e] = True
				elif (is_set == cp_is_set) and (strength > p_str):
					# Same type category, strictly higher strength
					out_legal[b, s, e] = True


# ── Device function: check if any legal play exists ─────────────────────────

@cuda.jit(device=True)
def _any_legal_play(hand_show, hand_len, no_play, play_is_set, play_strength, play_len):
	"""Check if any contiguous subarray of hand[0..hand_len-1] beats the given play.
	hand_show: local array of int8 showing values, left-aligned."""
	for s in range(hand_len):
		for e in range(s, hand_len):
			length = e - s + 1
			is_set = True
			is_asc = True
			is_desc = True
			for k in range(s, e):
				d = hand_show[k + 1] - hand_show[k]
				if d != 0:
					is_set = False
				if d != 1:
					is_asc = False
				if d != -1:
					is_desc = False
			if not (is_set or is_asc or is_desc):
				continue
			if is_asc:
				strength = hand_show[e]
			else:
				strength = hand_show[s]
			if no_play:
				return True
			if length > play_len:
				return True
			if length == play_len:
				if is_set and not play_is_set:
					return True
				if (is_set == play_is_set) and (strength > play_strength):
					return True
	return False


# ── compute_action_masks_kernel ──────────────────────────────────────────────

@cuda.jit
def compute_action_masks_kernel(
	hands_show,       # [B, MAX_P, H] int8
	hand_len,         # [B, MAX_P] int8
	current_player,   # [B] int8
	play_show,        # [B, MAX_PLAY] int8
	play_hide,        # [B, MAX_PLAY] int8
	play_len,         # [B] int8
	play_type,        # [B] int8
	phase,            # [B] int8
	sns_available,    # [B, MAX_P] bool
	num_players,      # [B] int8
	legal_plays,      # [B, H, H] bool (from compute_legal_plays_kernel)
	hand_offsets,     # [B] int64
	out_mask,         # [B, FLAT_ACTION_SIZE] bool — output
	B,
):
	b = cuda.grid(1)
	if b >= B:
		return

	# Zero output
	for i in range(FLAT_ACTION_SIZE):
		out_mask[b, i] = False

	cp = current_player[b]
	hl = hand_len[b, cp]
	ho = hand_offsets[b]
	p_len = play_len[b]

	# ── Play region [0..255]: remap legal plays to slot space ────────────
	for s in range(H):
		for e in range(H):
			if legal_plays[b, s, e]:
				s_slot = (ho + s) % H
				e_slot = (ho + e) % H
				out_mask[b, s_slot * H + e_slot] = True

	# ── Common gating for scout / S&S ────────────────────────────────────
	sns_phase = (phase[b] == PHASE_SNS_PLAY)
	has_play = (p_len > 0)
	has_room = (hl < H)
	can_scout_sns = has_play and has_room and (not sns_phase)
	if not can_scout_sns:
		return

	has_right = (p_len > 1)
	right_idx = p_len - 1 if p_len > 0 else 0

	# Card availability per choice
	card_avail = cuda.local.array(4, nbt.boolean)
	card_avail[0] = True       # left normal
	card_avail[1] = True       # left flipped
	card_avail[2] = has_right  # right normal
	card_avail[3] = has_right  # right flipped

	# ── Scout region [256..319] ──────────────────────────────────────────
	for c in range(4):
		if not card_avail[c]:
			continue
		for p in range(hl + 1):
			ins_slot = (ho + p) % H
			out_mask[b, 256 + c * H + ins_slot] = True

	# ── S&S region [320..383] ────────────────────────────────────────────
	can_sns = can_scout_sns and sns_available[b, cp]
	if not can_sns:
		return

	# Scouted card showing values per choice
	scouted_show = cuda.local.array(4, nbt.int8)
	scouted_show[0] = play_show[b, 0]          # left normal
	scouted_show[1] = play_hide[b, 0]          # left flipped
	scouted_show[2] = play_show[b, right_idx]  # right normal
	scouted_show[3] = play_hide[b, right_idx]  # right flipped

	# Reduced play properties
	# Type: same as original (removing an end card preserves set/run)
	is_set_type = (play_type[b] == PLAY_SET)
	# Check ascending: run where second card > first card
	is_asc = (not is_set_type) and (p_len >= 2) and (play_show[b, 1] > play_show[b, 0])

	red_len = p_len - 1
	red_no_play = (red_len == 0)
	# Reduced type: if only 1 card left it's a SET, else preserve type
	red_is_set = is_set_type if red_len > 1 else True

	# Reduced strength for left vs right removal
	if is_asc:
		red_str_left = play_show[b, right_idx]       # ascending: last unchanged
		sec_last = p_len - 2 if p_len >= 2 else 0
		red_str_right = play_show[b, sec_last]        # ascending: second-to-last
	else:
		red_str_left = play_show[b, 1]                # desc/set: new first = old[1]
		red_str_right = play_show[b, 0]               # desc/set: first unchanged

	# Read current player's hand
	orig = cuda.local.array(H, nbt.int8)
	for i in range(H):
		orig[i] = hands_show[b, cp, i]

	hyp = cuda.local.array(H, nbt.int8)  # hypothetical hand after insert

	for c in range(4):
		if not card_avail[c]:
			continue
		left_end = (c < 2)
		red_str = red_str_left if left_end else red_str_right

		for p in range(hl + 1):
			# Build hypothetical hand: insert scouted_show[c] at position p
			for i in range(p):
				hyp[i] = orig[i]
			hyp[p] = scouted_show[c]
			for i in range(p, hl):
				hyp[i + 1] = orig[i]

			new_hl = hl + 1
			if _any_legal_play(hyp, new_hl, red_no_play, red_is_set, red_str, red_len):
				ins_slot = (ho + p) % H
				out_mask[b, 320 + c * H + ins_slot] = True


# ── encode_states_kernel ─────────────────────────────────────────────────────
# Output: [B, 309] float32
# Layout: hand_top(192) + hand_bottom(16) + scout_cards(52) + play_buffer(21) + metadata(28)

@cuda.jit
def encode_states_kernel(
	hands_show,       # [B, MAX_P, H] int8
	hands_hide,       # [B, MAX_P, H] int8
	hand_len,         # [B, MAX_P] int8
	current_player,   # [B] int8
	play_show,        # [B, MAX_PLAY] int8
	play_hide,        # [B, MAX_PLAY] int8
	play_len,         # [B] int8
	play_owner,       # [B] int8
	play_type,        # [B] int8
	play_strength,    # [B] int8
	phase,            # [B] int8
	scouts_since_play,# [B] int8
	sns_available,    # [B, MAX_P] bool
	num_players,      # [B] int8
	collected,        # [B, MAX_P] int8
	scout_tokens,     # [B, MAX_P] int8
	hand_offsets,     # [B] int64
	out_encoded,      # [B, 309] float32 — output
	B,
):
	b = cuda.grid(1)
	if b >= B:
		return

	cp = current_player[b]
	n_p = num_players[b]
	hl = hand_len[b, cp]
	ho = hand_offsets[b]

	# Zero output
	for d in range(309):
		out_encoded[b, d] = 0.0

	# ── Segment 1: Hand top face — H slots x (N+2) = 192 dims ───────────
	# Position i -> slot (ho + i) % H. Slot s is occupied if (s - ho) % H < hl.
	# Per slot: [one-hot(N) for showing_value, empty_flag, scalar]
	for slot in range(H):
		pos = (slot - ho) % H  # which hand position maps to this slot
		base = slot * (N + 2)
		if pos < hl:
			sv = hands_show[b, cp, pos]  # 1-10
			out_encoded[b, base + sv - 1] = 1.0        # one-hot
			out_encoded[b, base + N + 1] = float(sv) / N  # scalar
		else:
			out_encoded[b, base + N] = 1.0              # empty flag

	# ── Segment 2: Hand bottom face — 16 dims at offset 192 ─────────────
	for slot in range(H):
		pos = (slot - ho) % H
		if pos < hl:
			hv = hands_hide[b, cp, pos]
			out_encoded[b, 192 + slot] = float(hv) / N

	# ── Segment 3: Scout cards — 52 dims at offset 208 ──────────────────
	# 4 options: [left-normal, left-flipped, right-normal, right-flipped]
	# Each: (N+1) one-hot + top_scalar + bot_scalar
	p_len = play_len[b]
	has_play = (p_len > 0)
	has_right = (p_len > 1)
	right_idx = p_len - 1 if p_len > 0 else 0

	left_show = play_show[b, 0]
	left_hide = play_hide[b, 0]
	right_show = play_show[b, right_idx]
	right_hide = play_hide[b, right_idx]

	# opt_top/bot for each of 4 choices
	# c=0: left normal  (top=left_show, bot=left_hide)
	# c=1: left flipped (top=left_hide, bot=left_show)
	# c=2: right normal (top=right_show, bot=right_hide)
	# c=3: right flipped(top=right_hide, bot=right_show)
	opt_top = cuda.local.array(4, nbt.int8)
	opt_bot = cuda.local.array(4, nbt.int8)
	opt_avail = cuda.local.array(4, nbt.boolean)
	opt_top[0] = left_show;  opt_bot[0] = left_hide;  opt_avail[0] = has_play
	opt_top[1] = left_hide;  opt_bot[1] = left_show;  opt_avail[1] = has_play
	opt_top[2] = right_show; opt_bot[2] = right_hide; opt_avail[2] = has_right
	opt_top[3] = right_hide; opt_bot[3] = right_show; opt_avail[3] = has_right

	for c in range(4):
		oh_base = 208 + c * (N + 1)
		if opt_avail[c]:
			tv = opt_top[c] - 1  # 0-indexed
			out_encoded[b, oh_base + tv] = 1.0
		else:
			out_encoded[b, oh_base + N] = 1.0  # absent flag

	# Scalars at offset 208 + 44 = 252: 4 top + 4 bot
	for c in range(4):
		if opt_avail[c]:
			out_encoded[b, 252 + c] = float(opt_top[c]) / N
			out_encoded[b, 256 + c] = float(opt_bot[c]) / N

	# ── Segment 4: Play buffer — 21 dims at offset 260 ──────────────────
	# Left-aligned (first 4) and right-aligned (last 4) views + metadata
	for i in range(4):
		if p_len > i:
			out_encoded[b, 260 + i * 2] = float(play_show[b, i]) / N
			out_encoded[b, 260 + i * 2 + 1] = float(play_hide[b, i]) / N
			# Right-aligned: position (3-i) from the right
			ri = p_len - 1 - i
			buf_pos = 3 - i
			out_encoded[b, 260 + 8 + buf_pos * 2] = float(play_show[b, ri]) / N
			out_encoded[b, 260 + 8 + buf_pos * 2 + 1] = float(play_hide[b, ri]) / N

	# Play type one-hot + strength + length at offset 260+16 = 276
	if p_len == 0:
		out_encoded[b, 276] = 1.0  # no_play
	elif play_type[b] == PLAY_SET:
		out_encoded[b, 277] = 1.0  # is_set
	else:
		out_encoded[b, 278] = 1.0  # is_run
	out_encoded[b, 279] = float(play_strength[b]) / N
	out_encoded[b, 280] = float(p_len) / 10.0

	# ── Segment 5: Metadata — 28 dims at offset 281 ─────────────────────
	for j in range(5):
		actual = (cp + j) % n_p
		if j < n_p:
			out_encoded[b, 281 + j] = float(hand_len[b, actual]) / H
			out_encoded[b, 286 + j] = float(collected[b, actual]) / H
			out_encoded[b, 291 + j] = float(scout_tokens[b, actual]) / 5.0
			out_encoded[b, 296 + j] = 1.0 if sns_available[b, actual] else 0.0

	out_encoded[b, 301] = float(n_p) / 5.0
	denom = float(n_p - 1) if n_p > 1 else 1.0
	out_encoded[b, 302] = float(scouts_since_play[b]) / denom

	# Play owner relative position one-hot
	if play_owner[b] >= 0:
		owner_rel = (play_owner[b] - cp) % n_p
		out_encoded[b, 303 + owner_rel] = 1.0

	# Forced play flag
	if phase[b] == PHASE_SNS_PLAY:
		out_encoded[b, 308] = 1.0


# ── apply_actions_kernel ─────────────────────────────────────────────────────

@cuda.jit
def apply_actions_kernel(
	hands_show,        # [B, MAX_P, H] int8
	hands_hide,        # [B, MAX_P, H] int8
	hand_len,          # [B, MAX_P] int8
	play_show,         # [B, MAX_PLAY] int8
	play_hide,         # [B, MAX_PLAY] int8
	play_len,          # [B] int8
	play_owner,        # [B] int8
	play_type,         # [B] int8
	play_strength,     # [B] int8
	current_player,    # [B] int8
	phase,             # [B] int8
	scouts_since_play, # [B] int8
	sns_available,     # [B, MAX_P] bool
	num_players,       # [B] int8
	collected,         # [B, MAX_P] int8
	scout_tokens,      # [B, MAX_P] int8
	round_ender,       # [B] int8
	done,              # [B] bool
	actions,           # [B] int64
	hand_offsets,      # [B] int64
	active,            # [B] bool
	B,
):
	b = cuda.grid(1)
	if b >= B:
		return
	if not active[b]:
		return

	cp = current_player[b]
	hl = hand_len[b, cp]
	n_p = num_players[b]
	ho = hand_offsets[b]
	act = actions[b]
	old_plen = play_len[b]

	is_play = (act < 256)
	is_scout = (act >= 256) and (act < 320)
	is_sns = (act >= 320)

	# Local arrays for hand manipulation
	new_show = cuda.local.array(H, nbt.int8)
	new_hide = cuda.local.array(H, nbt.int8)
	for i in range(H):
		new_show[i] = hands_show[b, cp, i]
		new_hide[i] = hands_hide[b, cp, i]

	if is_play:
		# ── Play action ──────────────────────────────────────────────
		s_slot = (act % 256) // H
		e_slot = (act % 256) % H
		start = (s_slot - ho) % H
		end = (e_slot - ho) % H
		play_count = end - start + 1

		# Write new play: hand[start..end]
		for i in range(H):
			if i < play_count:
				play_show[b, i] = new_show[start + i]
				play_hide[b, i] = new_hide[start + i]
			else:
				play_show[b, i] = 0
				play_hide[b, i] = 0

		# Determine play type and strength
		first_sv = new_show[start]
		last_sv = new_show[end]
		if play_count == 1:
			play_type[b] = PLAY_SET
		elif first_sv == new_show[start + 1]:
			play_type[b] = PLAY_SET
		else:
			play_type[b] = PLAY_RUN
		play_strength[b] = first_sv if first_sv > last_sv else last_sv

		# Current player collects old play's cards
		if old_plen > 0:
			collected[b, cp] = collected[b, cp] + old_plen

		# Remove played cards from hand: shift left
		new_hl = hl - play_count
		for i in range(H):
			if i < start:
				pass  # stays
			elif i < new_hl:
				new_show[i] = new_show[i + play_count]
				new_hide[i] = new_hide[i + play_count]
			else:
				new_show[i] = 0
				new_hide[i] = 0

		# Write back hand
		for i in range(H):
			hands_show[b, cp, i] = new_show[i]
			hands_hide[b, cp, i] = new_hide[i]
		hand_len[b, cp] = new_hl

		play_len[b] = play_count
		play_owner[b] = cp
		scouts_since_play[b] = 0
		phase[b] = PHASE_TURN

		# Round end: empty hand
		if new_hl == 0:
			done[b] = True
			round_ender[b] = cp
		else:
			# Advance turn
			current_player[b] = (cp + 1) % n_p

	else:
		# ── Scout or S&S ─────────────────────────────────────────────
		raw_idx = act - 320 if is_sns else act - 256
		card_choice = raw_idx // H
		ins_slot = raw_idx % H
		ins_pos = (ins_slot - ho) % H
		left_end = (card_choice < 2)
		flip = (card_choice % 2) == 1

		right_idx = old_plen - 1 if old_plen > 0 else 0

		# Save original play_owner before any modifications
		orig_po = play_owner[b]
		orig_po_safe = orig_po if orig_po >= 0 else 0

		# Determine scouted card (read BEFORE modifying play)
		if left_end:
			raw_sv = play_show[b, 0]
			raw_hv = play_hide[b, 0]
		else:
			raw_sv = play_show[b, right_idx]
			raw_hv = play_hide[b, right_idx]

		if flip:
			scouted_sv = raw_hv
			scouted_hv = raw_sv
		else:
			scouted_sv = raw_sv
			scouted_hv = raw_hv

		# Insert scouted card into hand at ins_pos (shift right)
		for i in range(hl, ins_pos, -1):
			new_show[i] = new_show[i - 1]
			new_hide[i] = new_hide[i - 1]
		new_show[ins_pos] = scouted_sv
		new_hide[ins_pos] = scouted_hv
		new_hl = hl + 1
		for i in range(new_hl, H):
			new_show[i] = 0
			new_hide[i] = 0

		# Write back hand
		for i in range(H):
			hands_show[b, cp, i] = new_show[i]
			hands_hide[b, cp, i] = new_hide[i]
		hand_len[b, cp] = new_hl

		# Update play: remove scouted card
		new_plen = old_plen - 1
		if left_end:
			for i in range(new_plen):
				play_show[b, i] = play_show[b, i + 1]
				play_hide[b, i] = play_hide[b, i + 1]
		for i in range(new_plen, H):
			play_show[b, i] = 0
			play_hide[b, i] = 0
		play_len[b] = new_plen

		# Play type/strength of reduced play
		# Reference: new_plen <= 1 -> PLAY_SET unconditionally
		if new_plen == 0:
			play_type[b] = PLAY_SET
			play_strength[b] = 0
			play_owner[b] = -1
		elif new_plen == 1:
			play_type[b] = PLAY_SET
			play_strength[b] = play_show[b, 0]
		else:
			# Type preserved. Strength = max(first, last) — works for set/asc/desc.
			first_sv = play_show[b, 0]
			last_sv = play_show[b, new_plen - 1]
			play_strength[b] = first_sv if first_sv > last_sv else last_sv

		# Award scout token to ORIGINAL play owner
		if orig_po >= 0:
			scout_tokens[b, orig_po] = scout_tokens[b, orig_po] + 1

		# scouts_since_play and round end
		new_ssp = scouts_since_play[b] + 1

		if is_scout:
			scout_round_end = (new_ssp >= n_p - 1)
			if scout_round_end:
				done[b] = True
				round_ender[b] = orig_po_safe
				scouts_since_play[b] = new_ssp
			elif new_plen == 0:
				# Play emptied without round end
				scouts_since_play[b] = 0
			else:
				scouts_since_play[b] = new_ssp
			phase[b] = PHASE_TURN
			if not done[b]:
				current_player[b] = (cp + 1) % n_p
		else:
			# S&S
			if new_plen == 0:
				scouts_since_play[b] = 0
			else:
				scouts_since_play[b] = new_ssp
			phase[b] = PHASE_SNS_PLAY
			sns_available[b, cp] = False


# ── Rollout entry point ──────────────────────────────────────────────────────

def rollout_numba(
	state: GpuGameState,
	network,
	max_steps: int = MAX_STEPS,
) -> Tensor:
	"""Run all games for max_steps (done games masked), return [B, MAX_P] score tensor."""
	from network import batched_masked_sample
	B = state.done.shape[0]
	dev = state.done.device

	# Pre-allocate output tensors (reused each step)
	legal_buf = torch.zeros(B, H, H, dtype=torch.bool, device=dev)
	mask_buf = torch.zeros(B, FLAT_ACTION_SIZE, dtype=torch.bool, device=dev)
	encode_buf = torch.zeros(B, 309, dtype=torch.float32, device=dev)

	# Wrap state tensors for Numba (zero-copy)
	d_hands_show = cuda.as_cuda_array(state.hands_show)
	d_hands_hide = cuda.as_cuda_array(state.hands_hide)
	d_hand_len = cuda.as_cuda_array(state.hand_len)
	d_play_show = cuda.as_cuda_array(state.play_show)
	d_play_hide = cuda.as_cuda_array(state.play_hide)
	d_play_len = cuda.as_cuda_array(state.play_len)
	d_play_owner = cuda.as_cuda_array(state.play_owner)
	d_play_type = cuda.as_cuda_array(state.play_type)
	d_play_strength = cuda.as_cuda_array(state.play_strength)
	d_current_player = cuda.as_cuda_array(state.current_player)
	d_phase = cuda.as_cuda_array(state.phase)
	d_scouts_since_play = cuda.as_cuda_array(state.scouts_since_play)
	d_sns_available = cuda.as_cuda_array(state.sns_available)
	d_num_players = cuda.as_cuda_array(state.num_players)
	d_collected = cuda.as_cuda_array(state.collected)
	d_scout_tokens = cuda.as_cuda_array(state.scout_tokens)
	d_round_ender = cuda.as_cuda_array(state.round_ender)
	d_done = cuda.as_cuda_array(state.done)
	d_legal = cuda.as_cuda_array(legal_buf)
	d_mask = cuda.as_cuda_array(mask_buf)
	d_encode = cuda.as_cuda_array(encode_buf)

	grid = _grid(B)

	network.eval()
	with torch.no_grad():
		for step in range(max_steps):
			active = ~state.done
			if not active.any():
				break

			hand_offsets = torch.randint(0, H, (B,), device=dev, dtype=torch.long)
			d_offsets = cuda.as_cuda_array(hand_offsets)

			# 1. Legal plays
			compute_legal_plays_kernel[grid, TPB](
				d_hands_show, d_hand_len, d_current_player,
				d_play_len, d_play_type, d_play_strength,
				d_done, d_legal, B,
			)

			# 2. Action masks
			compute_action_masks_kernel[grid, TPB](
				d_hands_show, d_hand_len, d_current_player,
				d_play_show, d_play_hide, d_play_len, d_play_type,
				d_phase, d_sns_available, d_num_players,
				d_legal, d_offsets, d_mask, B,
			)

			# 3. Encode states
			encode_states_kernel[grid, TPB](
				d_hands_show, d_hands_hide, d_hand_len, d_current_player,
				d_play_show, d_play_hide, d_play_len, d_play_owner,
				d_play_type, d_play_strength, d_phase,
				d_scouts_since_play, d_sns_available, d_num_players,
				d_collected, d_scout_tokens, d_offsets, d_encode, B,
			)

			# 4. Network forward + sampling (PyTorch, stays on GPU)
			h = network(encode_buf)
			logits = network.policy_logits(h)

			# Advance turn for active games with no legal actions
			has_action = mask_buf.any(dim=1)
			no_action = active & ~has_action
			if no_action.any():
				adv_cp = ((state.current_player.long() + 1) %
					state.num_players.long()).to(torch.int8)
				state.current_player = torch.where(
					no_action, adv_cp, state.current_player)
				# Re-wrap since torch.where creates a new tensor
				d_current_player = cuda.as_cuda_array(state.current_player)

			actions = batched_masked_sample(logits, mask_buf)
			d_actions = cuda.as_cuda_array(actions)
			apply_active = active & has_action
			d_apply_active = cuda.as_cuda_array(apply_active)

			# 5. Apply actions
			apply_actions_kernel[grid, TPB](
				d_hands_show, d_hands_hide, d_hand_len,
				d_play_show, d_play_hide, d_play_len, d_play_owner,
				d_play_type, d_play_strength, d_current_player,
				d_phase, d_scouts_since_play, d_sns_available,
				d_num_players, d_collected, d_scout_tokens,
				d_round_ender, d_done, d_actions, d_offsets,
				d_apply_active, B,
			)

	return compute_scores_tensor(state)
