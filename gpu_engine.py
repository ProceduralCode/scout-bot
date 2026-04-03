from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor
from game import Game, Phase, PlayType
from encoding import HAND_SLOTS_V6, FLAT_ACTION_SIZE, NUM_VALUES_V6, INPUT_SIZE_V6

# ── Constants ─────────────────────────────────────────────────────────────────

N        = NUM_VALUES_V6   # 10 card values
H        = HAND_SLOTS_V6  # 16 circular hand slots (matches V6 encoding)
MAX_P    = 5              # max players; tensors are padded to this size
MAX_PLAY = H              # max cards in a play (bounded by max hand size)
MAX_STEPS = 100           # fixed rollout horizon; done games are masked each step

# Phase encodings
PHASE_TURN     = 0
PHASE_SNS_PLAY = 1

# Play type encodings
PLAY_NONE = 0
PLAY_SET  = 1
PLAY_RUN  = 2


# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class GpuGameState:
	"""Batched game state for B concurrent rollouts, all tensors on the same device."""

	# Hands: left-aligned — positions 0..hand_len[b,p]-1 are valid cards.
	# Values are 1-10; 0 means the slot is empty (beyond hand_len).
	hands_show:    Tensor  # [B, MAX_P, H]  int8
	hands_hide:    Tensor  # [B, MAX_P, H]  int8
	hand_len:      Tensor  # [B, MAX_P]     int8

	# Current play on the table: left-aligned, 0..play_len[b]-1 are valid.
	play_show:     Tensor  # [B, MAX_PLAY]  int8
	play_hide:     Tensor  # [B, MAX_PLAY]  int8
	play_len:      Tensor  # [B]            int8  — 0 means no play
	play_owner:    Tensor  # [B]            int8  — -1 means no play
	play_type:     Tensor  # [B]            int8  — PLAY_NONE / PLAY_SET / PLAY_RUN
	play_strength: Tensor  # [B]            int8

	# Turn state
	current_player:    Tensor  # [B]        int8
	phase:             Tensor  # [B]        int8  — PHASE_TURN or PHASE_SNS_PLAY
	scouts_since_play: Tensor  # [B]        int8
	sns_available:     Tensor  # [B, MAX_P] bool
	num_players:       Tensor  # [B]        int8

	# Round scoring accumulators (used to compute round scores at completion)
	collected:    Tensor  # [B, MAX_P]  int8  — card count collected this round
	scout_tokens: Tensor  # [B, MAX_P]  int8

	# Terminal state
	round_ender:  Tensor  # [B]  int8  — player index who ended round, -1 while active
	done:         Tensor  # [B]  bool


# ── Conversion ────────────────────────────────────────────────────────────────

def from_snapshots(snapshots: list[Game], device: str = 'cuda') -> GpuGameState:
	"""Convert a list of Python Game snapshots to a batched GPU game state.
	Snapshots should be in TURN or SNS_PLAY phase (standard rollout starting points)."""
	import numpy as np
	B = len(snapshots)

	# Use numpy arrays for fast bulk filling, convert to torch at the end
	hs = np.zeros((B, MAX_P, H), dtype=np.int8)
	hh = np.zeros((B, MAX_P, H), dtype=np.int8)
	hl = np.zeros((B, MAX_P), dtype=np.int8)
	ps_buf = np.zeros((B, MAX_PLAY), dtype=np.int8)
	ph_buf = np.zeros((B, MAX_PLAY), dtype=np.int8)
	pl_buf = np.zeros(B, dtype=np.int8)
	po_buf = np.full(B, -1, dtype=np.int8)
	pt_buf = np.zeros(B, dtype=np.int8)
	pstr_buf = np.zeros(B, dtype=np.int8)
	cp_buf = np.zeros(B, dtype=np.int8)
	phase_buf = np.zeros(B, dtype=np.int8)
	ssp_buf = np.zeros(B, dtype=np.int8)
	sns_buf = np.ones((B, MAX_P), dtype=np.bool_)
	np_buf = np.zeros(B, dtype=np.int8)
	col_buf = np.zeros((B, MAX_P), dtype=np.int8)
	st_buf = np.zeros((B, MAX_P), dtype=np.int8)
	re_buf = np.full(B, -1, dtype=np.int8)
	done_buf = np.zeros(B, dtype=np.bool_)

	for b, game in enumerate(snapshots):
		n = game.num_players
		np_buf[b] = n
		cp_buf[b] = game.current_player
		ssp_buf[b] = game.scouts_since_play
		phase_buf[b] = PHASE_SNS_PLAY if game.phase == Phase.SNS_PLAY else PHASE_TURN
		if game.phase not in (Phase.TURN, Phase.SNS_PLAY):
			done_buf[b] = True

		for p in range(n):
			pstate = game.players[p]
			hand = pstate.hand
			hand_len = len(hand)
			hl[b, p] = hand_len
			if hand_len > 0:
				# Batch-assign hand cards via numpy
				arr = np.array(hand, dtype=np.int8)
				hs[b, p, :hand_len] = arr[:, 0]
				hh[b, p, :hand_len] = arr[:, 1]
			col_buf[b, p] = len(pstate.collected)
			st_buf[b, p] = pstate.scout_tokens
			sns_buf[b, p] = pstate.sns_available

		if game.current_play is not None:
			cp = game.current_play
			cards = cp.cards
			plen = len(cards)
			pl_buf[b] = plen
			po_buf[b] = game.current_play_owner
			pt_buf[b] = PLAY_SET if cp.play_type == PlayType.SET else PLAY_RUN
			pstr_buf[b] = cp.strength
			if plen > 0:
				arr = np.array(cards, dtype=np.int8)
				ps_buf[b, :plen] = arr[:, 0]
				ph_buf[b, :plen] = arr[:, 1]

	return GpuGameState(
		hands_show=torch.from_numpy(hs).to(device),
		hands_hide=torch.from_numpy(hh).to(device),
		hand_len=torch.from_numpy(hl).to(device),
		play_show=torch.from_numpy(ps_buf).to(device),
		play_hide=torch.from_numpy(ph_buf).to(device),
		play_len=torch.from_numpy(pl_buf).to(device),
		play_owner=torch.from_numpy(po_buf).to(device),
		play_type=torch.from_numpy(pt_buf).to(device),
		play_strength=torch.from_numpy(pstr_buf).to(device),
		current_player=torch.from_numpy(cp_buf).to(device),
		phase=torch.from_numpy(phase_buf).to(device),
		scouts_since_play=torch.from_numpy(ssp_buf).to(device),
		sns_available=torch.from_numpy(sns_buf).to(device),
		num_players=torch.from_numpy(np_buf).to(device),
		collected=torch.from_numpy(col_buf).to(device),
		scout_tokens=torch.from_numpy(st_buf).to(device),
		round_ender=torch.from_numpy(re_buf).to(device),
		done=torch.from_numpy(done_buf).to(device),
	)


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_states(state: GpuGameState, hand_offsets: Tensor) -> Tensor:
	"""Vectorized encode_state_v6 for all B games simultaneously.

	hand_offsets: [B] int64 — circular rotation offset for data augmentation.
	Returns [B, 309] float32 on the same device as state.

	Mirrors _fill_hand_v6, _fill_hand_bottom_v6, _fill_scout_cards_v6,
	_fill_play_buffer_v6, _fill_metadata_v6 from encoding.py exactly."""
	B   = state.done.shape[0]
	dev = state.done.device
	brange = torch.arange(B, device=dev)

	cp  = state.current_player.long()   # [B]
	n_p = state.num_players.long()      # [B]

	# Current player's hand (left-aligned)
	hand_show = state.hands_show[brange, cp]  # [B, H] int8
	hand_hide = state.hands_hide[brange, cp]  # [B, H] int8
	hl        = state.hand_len[brange, cp].long()  # [B]

	positions = torch.arange(H, device=dev)  # [H]

	# Slot assignment: hand position i → slot (hand_offset + i) % H
	slots = (hand_offsets.unsqueeze(1) + positions.unsqueeze(0)) % H  # [B, H]

	# Which positions hold real cards
	valid = positions.unsqueeze(0) < hl.unsqueeze(1)  # [B, H] bool

	# Which slots are occupied (a real card lives there)
	occupied = torch.zeros(B, H, device=dev, dtype=torch.bool)
	occupied.scatter_(1, slots, valid)

	# Showing/hiding values indexed by slot (0 for empty slots)
	show_at_slot = torch.zeros(B, H, device=dev, dtype=torch.long)
	hide_at_slot = torch.zeros(B, H, device=dev, dtype=torch.float32)
	show_at_slot.scatter_(1, slots, hand_show.long() * valid.long())
	hide_at_slot.scatter_(1, slots, hand_hide.float() * valid.float())

	# ── Segment 1: Hand top face — H slots × (N+2) dims = 192 ────────────────
	# Per slot: N-dim one-hot for showing value + empty flag + showing scalar
	slot_size = N + 2
	hand_top = torch.zeros(B, H, slot_size, device=dev)
	hand_top[:, :, N] = (~occupied).float()                           # empty flag
	oh = F.one_hot((show_at_slot - 1).clamp(0), num_classes=N).float()  # [B, H, N]
	oh *= occupied.unsqueeze(-1)                                       # zero empty slots
	hand_top[:, :, :N]  = oh
	hand_top[:, :, N+1] = show_at_slot.float() / N * occupied.float() # top scalar

	# ── Segment 2: Hand bottom face — H dims = 16 ────────────────────────────
	hand_bottom = hide_at_slot / N  # [B, H]

	# ── Segment 3: Scout cards — 4×(N+1) one-hot + 8 scalars = 52 ───────────
	# 4 options: left-normal, left-flipped, right-normal, right-flipped.
	# "top face" of the scouted card is what matters for the one-hot.
	has_play  = state.play_len > 0  # [B]
	has_right = state.play_len > 1  # [B]

	left_show  = state.play_show[:, 0].long()  # [B]
	left_hide  = state.play_hide[:, 0].long()
	right_idx  = (state.play_len.long() - 1).clamp(0)
	right_show = state.play_show[brange, right_idx].long()  # [B]
	right_hide = state.play_hide[brange, right_idx].long()

	# opt_top[b, c] = top face of option c; opt_bot = bottom face
	opt_top   = torch.stack([left_show, left_hide, right_show, right_hide], dim=1)  # [B, 4]
	opt_bot   = torch.stack([left_hide, left_show, right_hide, right_show], dim=1)
	opt_avail = torch.stack([has_play, has_play, has_right, has_right],     dim=1)  # [B, 4]

	# One-hot blocks [B, 4, N+1]: dim N is the absent flag
	scout_oh = torch.zeros(B, 4, N+1, device=dev)
	scout_oh[:, :, N] = 1.0  # all absent by default
	for c in range(4):
		avail_c = opt_avail[:, c]
		top_c   = (opt_top[:, c] - 1).clamp(0)  # 0-indexed value
		oh_c    = F.one_hot(top_c, num_classes=N+1).float()  # [B, N+1], dim N = 0 for values 0-9
		scout_oh[avail_c, c] = oh_c[avail_c]

	scout_blocks  = scout_oh.view(B, 4 * (N+1))                            # [B, 44]
	tops_f        = opt_top.float() * opt_avail.float() / N                # [B, 4]
	bots_f        = opt_bot.float() * opt_avail.float() / N                # [B, 4]
	scout_section = torch.cat([scout_blocks, tops_f, bots_f], dim=1)       # [B, 52]

	# ── Segment 4: Play buffer — 2×8 card scalars + 5 metadata = 21 ─────────
	# Left-aligned (first 4 cards) and right-aligned (last 4 cards) views.
	play_buf = torch.zeros(B, 21, device=dev)
	for i in range(4):
		has_card    = state.play_len > i  # [B]
		right_i_idx = (state.play_len.long() - 1 - i).clamp(0)
		buf_pos     = 3 - i

		play_buf[has_card, i*2]             = state.play_show[has_card, i].float() / N
		play_buf[has_card, i*2+1]           = state.play_hide[has_card, i].float() / N
		play_buf[has_card, 8 + buf_pos*2]   = state.play_show[brange, right_i_idx][has_card].float() / N
		play_buf[has_card, 8 + buf_pos*2+1] = state.play_hide[brange, right_i_idx][has_card].float() / N

	# Play type: one-hot [no_play, is_set, is_run] at offset 16
	play_buf[:, 16] = (state.play_len == 0).float()
	play_buf[:, 17] = (state.play_type == PLAY_SET).float()
	play_buf[:, 18] = (state.play_type == PLAY_RUN).float()
	play_buf[:, 19] = state.play_strength.float() / N
	play_buf[:, 20] = state.play_len.float() / 10.0

	# ── Segment 5: Metadata — 28 dims ────────────────────────────────────────
	# Layout: 5 hand_len + 5 collected + 5 scout_tokens + 5 sns_available
	#         + num_players + scouts_since_play + 5 owner_rel one-hot + forced_flag
	meta = torch.zeros(B, 28, device=dev)
	for j in range(5):
		# j=0 is self (current player), j=1..4 are opponents in seat order.
		# Slot is zero-padded if j >= num_players.
		actual  = (cp + j) % n_p                  # [B] actual player index
		valid_j = (j < n_p).float()               # [B] 1.0 if this seat exists
		meta[:, j]      = state.hand_len[brange, actual].float()     / H * valid_j
		meta[:, 5 + j]  = state.collected[brange, actual].float()    / H * valid_j
		meta[:, 10 + j] = state.scout_tokens[brange, actual].float() / 5.0 * valid_j
		meta[:, 15 + j] = state.sns_available[brange, actual].float()      * valid_j

	meta[:, 20] = state.num_players.float() / 5.0
	meta[:, 21] = state.scouts_since_play.float() / (n_p - 1).float().clamp(min=1)

	# Play owner relative position (0 = self owns play, 1-4 = seats away)
	has_owner = state.play_owner >= 0                          # [B]
	owner_abs = state.play_owner.long().clamp(min=0)
	owner_rel = (owner_abs - cp) % n_p                        # [B]
	owner_oh  = F.one_hot(owner_rel, num_classes=5).float() * has_owner.unsqueeze(1).float()
	meta[:, 22:27] = owner_oh

	meta[:, 27] = (state.phase == PHASE_SNS_PLAY).float()     # forced play flag

	# ── Concatenate ───────────────────────────────────────────────────────────
	return torch.cat([
		hand_top.view(B, H * slot_size),  # 192
		hand_bottom,                       #  16
		scout_section,                     #  52
		play_buf,                          #  21
		meta,                              #  28
	], dim=1)  # [B, 309]


# ── Legal play computation ────────────────────────────────────────────────────

def compute_legal_plays(state: GpuGameState) -> Tensor:
	"""Compute legal play validity for all B games simultaneously.

	Returns [B, H, H] bool where result[b, start, end] = True iff cards
	hand[start..end] of the current player in game b form a legal play.
	Coordinates are hand-position indices (0..hand_len-1); out-of-range
	positions are False.

	A play is legal if it forms a valid set or run AND beats the current play
	on the table (or there is no current play)."""
	B   = state.done.shape[0]
	dev = state.done.device
	brange = torch.arange(B, device=dev)

	cp = state.current_player.long()
	hl = state.hand_len[brange, cp].long()  # [B] actual hand sizes

	# Showing values of current player's hand (0 beyond hand_len)
	vals = state.hands_show[brange, cp].long()  # [B, H]

	# Consecutive differences — basis of the set/run check
	diff = vals[:, 1:] - vals[:, :-1]  # [B, H-1]

	# Prefix sums with a leading zero: cs[b, k] = count of qualifying diffs in [0..k-1].
	# cs[b, e] - cs[b, s] = count of qualifying diffs in [s..e-1] = over range [s, e].
	zero    = torch.zeros(B, 1, device=dev, dtype=torch.long)
	cs_set  = torch.cat([zero, torch.cumsum((diff == 0).long(),  dim=1)], dim=1)  # [B, H]
	cs_asc  = torch.cat([zero, torch.cumsum((diff == 1).long(),  dim=1)], dim=1)
	cs_desc = torch.cat([zero, torch.cumsum((diff == -1).long(), dim=1)], dim=1)

	# Range counts: count[b, start, end] = cs[b, end] - cs[b, start]
	cs_set_r  = cs_set.unsqueeze(1)  - cs_set.unsqueeze(2)   # [B, H, H]
	cs_asc_r  = cs_asc.unsqueeze(1)  - cs_asc.unsqueeze(2)
	cs_desc_r = cs_desc.unsqueeze(1) - cs_desc.unsqueeze(2)

	# span[start, end] = end - start = number of diffs in the range [start..end-1]
	s_idx = torch.arange(H, device=dev)
	e_idx = torch.arange(H, device=dev)
	span  = e_idx.view(1, H) - s_idx.view(H, 1)  # [H, H], span[s, e] = e - s

	# A range [start, end] is a valid set/run iff all diffs in [start..end-1] qualify.
	# span==0 (single card) trivially satisfies all three (0 diffs needed, 0 present).
	is_set_range  = cs_set_r  == span.unsqueeze(0)  # [B, H, H]
	is_asc_range  = cs_asc_r  == span.unsqueeze(0)
	is_desc_range = cs_desc_r == span.unsqueeze(0)
	is_valid_type = is_set_range | is_asc_range | is_desc_range

	# Valid positions: start <= end, both strictly within hand_len[b]
	hl_3d   = hl.view(B, 1, 1)
	s_valid = s_idx.view(1, H, 1) < hl_3d  # [B, H, 1]
	e_valid = e_idx.view(1, 1, H) < hl_3d  # [B, 1, H]
	pos_valid = s_valid & e_valid & (span.unsqueeze(0) >= 0)

	# ── Beats check ───────────────────────────────────────────────────────────
	# Strength = vals[end] for ascending runs, vals[start] for sets and descending runs.
	val_start = vals.unsqueeze(2).expand(B, H, H)  # vals at start position
	val_end   = vals.unsqueeze(1).expand(B, H, H)  # vals at end position
	strength  = torch.where(is_asc_range, val_end, val_start)

	cp_len   = state.play_len.long().view(B, 1, 1)
	cp_isset = (state.play_type == PLAY_SET).view(B, 1, 1)
	cp_str   = state.play_strength.long().view(B, 1, 1)
	no_play  = (state.play_len == 0).view(B, 1, 1)

	length       = span.unsqueeze(0) + 1             # [1, H, H]
	beats_longer = length > cp_len
	equal_length = length == cp_len

	# For equal length: set beats run; same type needs strictly higher strength
	beats_eq_len = (
		(is_set_range & ~cp_isset) |
		(~(is_set_range ^ cp_isset) & (strength > cp_str))
	)
	beats = no_play | beats_longer | (equal_length & beats_eq_len)

	return is_valid_type & pos_valid & beats


# ── Action mask computation ───────────────────────────────────────────────────

def _any_legal_play_batched(
	hand_show: Tensor,     # [B', H] long — left-aligned, 0 beyond hand_len
	hand_len: Tensor,      # [B'] long
	no_play: Tensor,       # [B'] bool — True if no reduced play (any move beats it)
	play_isset: Tensor,    # [B'] bool
	play_strength: Tensor, # [B'] long
	play_len: Tensor,      # [B'] long
) -> Tensor:               # [B'] bool
	"""Batched check: does any contiguous subarray of each hand beat the given play?
	Mirrors _has_any_legal_play from encoding.py."""
	Bp  = hand_show.shape[0]
	dev = hand_show.device

	diff = hand_show[:, 1:] - hand_show[:, :-1]  # [B', H-1]

	zero    = torch.zeros(Bp, 1, device=dev, dtype=torch.long)
	cs_set  = torch.cat([zero, torch.cumsum((diff == 0).long(), dim=1)], dim=1)  # [B', H]
	cs_asc  = torch.cat([zero, torch.cumsum((diff == 1).long(), dim=1)], dim=1)
	cs_desc = torch.cat([zero, torch.cumsum((diff == -1).long(), dim=1)], dim=1)

	# Range counts: cs[e] - cs[s] = number of qualifying diffs in positions [s, e-1]
	cs_set_r  = cs_set.unsqueeze(1)  - cs_set.unsqueeze(2)   # [B', H, H]
	cs_asc_r  = cs_asc.unsqueeze(1)  - cs_asc.unsqueeze(2)
	cs_desc_r = cs_desc.unsqueeze(1) - cs_desc.unsqueeze(2)

	s_idx = torch.arange(H, device=dev)
	e_idx = torch.arange(H, device=dev)
	span  = e_idx.view(1, H) - s_idx.view(H, 1)  # [H, H] — e - s

	is_set_r  = cs_set_r  == span.unsqueeze(0)   # [B', H, H]
	is_asc_r  = cs_asc_r  == span.unsqueeze(0)
	is_valid  = is_set_r | is_asc_r | (cs_desc_r == span.unsqueeze(0))

	hl3    = hand_len.view(Bp, 1, 1)
	pos_ok = (s_idx.view(1, H, 1) < hl3) & (e_idx.view(1, 1, H) < hl3) & (span.unsqueeze(0) >= 0)

	# Strength: ascending uses end value; set/descending use start value
	val_start = hand_show.unsqueeze(2).expand(Bp, H, H)  # [B', H, H]
	val_end   = hand_show.unsqueeze(1).expand(Bp, H, H)
	strength  = torch.where(is_asc_r, val_end, val_start)

	cp_len   = play_len.view(Bp, 1, 1)
	cp_str   = play_strength.view(Bp, 1, 1)
	cp_isset = play_isset.view(Bp, 1, 1)
	length   = span.unsqueeze(0) + 1

	beats_eq = (is_set_r & ~cp_isset) | (~(is_set_r ^ cp_isset) & (strength > cp_str))
	beats    = no_play.view(Bp, 1, 1) | (length > cp_len) | ((length == cp_len) & beats_eq)

	return (is_valid & pos_ok & beats).view(Bp, H * H).any(dim=1)


def compute_action_masks(
	state: GpuGameState,
	legal_plays: Tensor,   # [B, H, H] bool from compute_legal_plays (hand-position space)
	hand_offsets: Tensor,  # [B] int64
) -> Tensor:               # [B, FLAT_ACTION_SIZE] bool
	"""Vectorized get_flat_action_mask for all B games simultaneously.

	Play region [0..255]:    start_slot * H + end_slot
	Scout region [256..319]: 256 + card_choice * H + insert_slot
	S&S region [320..383]:   320 + card_choice * H + insert_slot
	"""
	B   = state.done.shape[0]
	dev = state.done.device
	brange = torch.arange(B, device=dev)

	cp = state.current_player.long()        # [B]
	hl = state.hand_len[brange, cp].long()  # [B]

	mask = torch.zeros(B, FLAT_ACTION_SIZE, device=dev, dtype=torch.bool)

	# ── Play region [0..255] ──────────────────────────────────────────────────
	# Remap (s, e) from hand-position space to slot space via hand_offsets.
	# s_slots[b, s] = (hand_offsets[b] + s) % H — a cyclic permutation of 0..H-1.
	h_idx   = torch.arange(H, device=dev)
	s_slots = (hand_offsets.unsqueeze(1) + h_idx.unsqueeze(0)) % H  # [B, H]
	# play_act[b, s, e] = s_slots[b,s] * H + s_slots[b,e]
	play_act = s_slots.unsqueeze(2) * H + s_slots.unsqueeze(1)      # [B, H, H]
	mask.scatter_(1, play_act.view(B, H * H), legal_plays.view(B, H * H))

	# ── Common gating for scout / S&S ─────────────────────────────────────────
	sns_phase    = state.phase == PHASE_SNS_PLAY               # [B]
	has_play     = state.play_len > 0                          # [B]
	has_room     = hl < H                                      # [B]
	can_scout_sns = has_play & has_room & ~sns_phase           # [B]

	has_right  = state.play_len > 1                            # [B]
	# card_avail[b, c]: card choice c is available in game b
	card_avail = torch.stack(
		[has_play, has_play, has_right, has_right], dim=1
	) & can_scout_sns.unsqueeze(1)                             # [B, 4]

	p_idx     = torch.arange(H, device=dev)
	ins_slots = (hand_offsets.unsqueeze(1) + p_idx.unsqueeze(0)) % H  # [B, H]
	# Insert position p is valid iff p <= hand_len (positions 0..hand_len inclusive)
	p_valid   = p_idx.unsqueeze(0) <= hl.unsqueeze(1)                  # [B, H]

	# ── Scout region [256..319] ───────────────────────────────────────────────
	for c in range(4):
		act   = 256 + c * H + ins_slots                       # [B, H]
		valid = p_valid & card_avail[:, c].unsqueeze(1)       # [B, H]
		mask.scatter_(1, act, valid)

	# ── S&S region [320..383] ─────────────────────────────────────────────────
	sns_avail = state.sns_available[brange, cp]               # [B]
	can_sns   = can_scout_sns & sns_avail                     # [B]

	# S&S region: always compute (no .any() sync); can_sns mask handles correctness
	# Scouted card showing value for each card choice [B, 4]
	right_idx    = (state.play_len.long() - 1).clamp(0)   # [B] index of rightmost play card
	scouted_show = torch.stack([
		state.play_show[:, 0].long(),                      # c=0: left normal
		state.play_hide[:, 0].long(),                      # c=1: left flipped
		state.play_show[brange, right_idx].long(),         # c=2: right normal
		state.play_hide[brange, right_idx].long(),         # c=3: right flipped
	], dim=1)  # [B, 4]

	# Reduced play type: same as original (removing an end card preserves set/run)
	is_set = (state.play_type == PLAY_SET)                 # [B]
	# Ascending iff it's a run and the second card is larger than the first
	is_asc = (~is_set) & (state.play_len >= 2) & (
		state.play_show[:, 1].long() > state.play_show[:, 0].long()
	)  # [B]

	# Indices for indexing into play_show
	second_idx    = (state.play_len.long() - 2).clamp(0)  # [B]
	last_show     = state.play_show[brange, right_idx].long()       # play_show[play_len-1]
	sec_show      = state.play_show[:, 1].long()                    # play_show[1]
	sec_last_show = state.play_show[brange, second_idx].long()      # play_show[play_len-2]
	first_show    = state.play_show[:, 0].long()                    # play_show[0]

	# Strength of the reduced play after removing the scouted card:
	#   remove left (c=0,1): ascending → max unchanged (last_show); else → sec_show
	#   remove right (c=2,3): ascending → sec_last_show; else → first_show (max unchanged)
	red_str_l = torch.where(is_asc, last_show, sec_show)           # [B]
	red_str_r = torch.where(is_asc, sec_last_show, first_show)     # [B]
	red_str   = torch.stack([red_str_l, red_str_l, red_str_r, red_str_r], dim=1)  # [B, 4]

	red_len     = (state.play_len.long() - 1).clamp(0)    # [B]
	red_no_play = (state.play_len == 1)                    # [B] — reduced play is None
	# Expand [B] → [B, 4]:
	red_len4    = red_len.unsqueeze(1).expand(B, 4)
	red_np4     = red_no_play.unsqueeze(1).expand(B, 4)
	red_isset4  = is_set.unsqueeze(1).expand(B, 4)

	# Build hypothetical hands [B, 4, H, H]:
	# hyp_show[b, c, p, i] = new hand after inserting scouted_show[b,c] at position p
	orig_show = state.hands_show[brange, cp].long()        # [B, H]
	# orig_im1[b, i] = orig[b, i-1] for i >= 1, 0 for i = 0
	orig_im1  = torch.cat([
		torch.zeros(B, 1, device=dev, dtype=torch.long),
		orig_show[:, :H - 1],
	], dim=1)                                              # [B, H]

	p_bcast = p_idx.view(1, 1, H, 1)                      # insert position dim
	i_bcast = p_idx.view(1, 1, 1, H)                      # new hand slot dim

	orig_at_i   = orig_show.view(B, 1, 1, H).expand(B, 4, H, H)
	orig_at_im1 = orig_im1.view(B, 1, 1, H).expand(B, 4, H, H)
	scouted_exp = scouted_show.view(B, 4, 1, 1).expand(B, 4, H, H)

	hyp_show = torch.where(i_bcast < p_bcast, orig_at_i,
			   torch.where(i_bcast == p_bcast, scouted_exp,
			   orig_at_im1))                               # [B, 4, H, H]

	# Flatten to [flat_B, H] for batched legal-play check
	flat_B          = B * 4 * H
	hyp_show_flat   = hyp_show.reshape(flat_B, H)
	hyp_len_flat    = (hl.view(B, 1, 1) + 1).expand(B, 4, H).reshape(flat_B)
	red_str_flat    = red_str.unsqueeze(2).expand(B, 4, H).reshape(flat_B)
	red_len_flat    = red_len4.unsqueeze(2).expand(B, 4, H).reshape(flat_B)
	red_np_flat     = red_np4.unsqueeze(2).expand(B, 4, H).reshape(flat_B)
	red_isset_flat  = red_isset4.unsqueeze(2).expand(B, 4, H).reshape(flat_B)

	any_legal = _any_legal_play_batched(
		hyp_show_flat, hyp_len_flat,
		red_np_flat, red_isset_flat, red_str_flat, red_len_flat,
	).view(B, 4, H)  # [B, 4, H]

	# Gate by position validity, card availability, and can_sns
	sns_ok = (
		any_legal
		& card_avail.unsqueeze(2)   # [B, 4, H]
		& p_valid.unsqueeze(1)      # [B, 4, H]
		& can_sns.view(B, 1, 1)
	)

	for c in range(4):
		act = 320 + c * H + ins_slots  # [B, H]
		mask.scatter_(1, act, sns_ok[:, c, :])

	return mask


# ── Action application ───────────────────────────────────────────────────────

def apply_actions(
	state: GpuGameState,
	actions: Tensor,      # [B] int64 — sampled flat action indices
	hand_offsets: Tensor,  # [B] int64
	active: Tensor,       # [B] bool — which games to process
) -> None:
	"""Update GpuGameState in-place from sampled actions.
	Games where active=False are left unchanged. No CPU sync points."""
	B = state.done.shape[0]
	dev = state.done.device
	brange = torch.arange(B, device=dev)
	pos = torch.arange(H, device=dev)
	cp = state.current_player.long()
	hl = state.hand_len[brange, cp].long()
	n_p = state.num_players.long()

	# ── Decode actions ────────────────────────────────────────────────────
	is_play = active & (actions < 256)
	is_scout = active & (actions >= 256) & (actions < 320)
	is_sns = active & (actions >= 320)
	is_scout_or_sns = is_scout | is_sns

	# Play: start/end in hand-position space
	play_bits = actions % 256
	s_slot = play_bits // H
	e_slot = play_bits % H
	start = (s_slot - hand_offsets) % H
	end = (e_slot - hand_offsets) % H
	play_len_p = (end - start + 1).clamp(min=1, max=H)

	# Scout/S&S: card choice and insert position
	raw_idx = torch.where(is_sns, actions - 320, (actions - 256).clamp(min=0))
	card_choice = raw_idx // H
	ins_slot = raw_idx % H
	ins_pos = (ins_slot - hand_offsets) % H
	left_end = card_choice < 2
	flip = (card_choice % 2) == 1

	# Current player's hand
	old_show = state.hands_show[brange, cp].long()  # [B, H]
	old_hide = state.hands_hide[brange, cp].long()

	has_old_play = state.play_len > 0
	old_play_len = state.play_len.long()
	plen_idx = (old_play_len - 1).clamp(0)       # rightmost play card
	sec_last_idx = (old_play_len - 2).clamp(0)

	# ── Play action: extract new play, remove from hand ───────────────────
	# Gather the played cards: hand[start..end]
	play_gather = (start.unsqueeze(1) + pos.unsqueeze(0)).clamp(max=H - 1)
	new_play_show_p = old_show.gather(1, play_gather)
	new_play_hide_p = old_hide.gather(1, play_gather)
	pvalid = pos.unsqueeze(0) < play_len_p.unsqueeze(1)
	new_play_show_p = new_play_show_p * pvalid.long()
	new_play_hide_p = new_play_hide_p * pvalid.long()

	# Play type: single or all-same → SET, otherwise RUN
	first_val = new_play_show_p[:, 0]
	second_val = new_play_show_p[:, 1]
	last_play_idx = (play_len_p - 1).clamp(0)
	last_val = new_play_show_p[brange, last_play_idx]
	is_set_new = (play_len_p == 1) | (first_val == second_val)
	new_type_p = torch.where(is_set_new,
		torch.full((B,), PLAY_SET, device=dev, dtype=torch.int8),
		torch.full((B,), PLAY_RUN, device=dev, dtype=torch.int8))
	# Strength = max(first, last) — works for sets (equal), asc (last), desc (first)
	new_strength_p = torch.max(first_val, last_val).to(torch.int8)

	# Remove played cards: shift hand left by play_len_p starting at `start`
	shift = play_len_p.unsqueeze(1) * (pos.unsqueeze(0) >= start.unsqueeze(1)).long()
	rm_idx = (pos.unsqueeze(0) + shift).clamp(max=H - 1)
	hand_play_show = old_show.gather(1, rm_idx)
	hand_play_hide = old_hide.gather(1, rm_idx)
	new_hl_play = (hl - play_len_p).clamp(min=0)
	beyond_play = pos.unsqueeze(0) >= new_hl_play.unsqueeze(1)
	hand_play_show = hand_play_show.masked_fill(beyond_play, 0)
	hand_play_hide = hand_play_hide.masked_fill(beyond_play, 0)

	# Current player collects old play's cards
	collect_amt = torch.where(is_play & has_old_play,
		old_play_len, torch.zeros_like(old_play_len))

	play_round_end = is_play & (new_hl_play == 0)

	# ── Scout/S&S: determine scouted card, insert into hand ──────────────
	raw_show_l = state.play_show[:, 0].long()
	raw_hide_l = state.play_hide[:, 0].long()
	raw_show_r = state.play_show[brange, plen_idx].long()
	raw_hide_r = state.play_hide[brange, plen_idx].long()
	raw_show = torch.where(left_end, raw_show_l, raw_show_r)
	raw_hide = torch.where(left_end, raw_hide_l, raw_hide_r)
	scouted_show = torch.where(flip, raw_hide, raw_show)
	scouted_hide = torch.where(flip, raw_show, raw_hide)

	# Insert scouted card: shift positions after ins_pos right by 1
	src = torch.where(pos.unsqueeze(0) > ins_pos.unsqueeze(1),
		(pos - 1).clamp(0).unsqueeze(0).expand(B, H),
		pos.unsqueeze(0).expand(B, H))
	hand_scout_show = old_show.gather(1, src)
	hand_scout_hide = old_hide.gather(1, src)
	hand_scout_show.scatter_(1, ins_pos.unsqueeze(1), scouted_show.unsqueeze(1))
	hand_scout_hide.scatter_(1, ins_pos.unsqueeze(1), scouted_hide.unsqueeze(1))
	new_hl_scout = hl + 1
	beyond_scout = pos.unsqueeze(0) >= new_hl_scout.unsqueeze(1)
	hand_scout_show = hand_scout_show.masked_fill(beyond_scout, 0)
	hand_scout_hide = hand_scout_hide.masked_fill(beyond_scout, 0)

	# Update play after scouting: left removal shifts, right just shortens
	shifted_pshow = torch.cat([state.play_show[:, 1:].long(),
		torch.zeros(B, 1, device=dev, dtype=torch.long)], dim=1)
	shifted_phide = torch.cat([state.play_hide[:, 1:].long(),
		torch.zeros(B, 1, device=dev, dtype=torch.long)], dim=1)
	play_scout_show = torch.where(left_end.view(B, 1),
		shifted_pshow, state.play_show.long())
	play_scout_hide = torch.where(left_end.view(B, 1),
		shifted_phide, state.play_hide.long())
	new_plen_scout = (old_play_len - 1).clamp(min=0)
	beyond_ps = pos.unsqueeze(0) >= new_plen_scout.unsqueeze(1)
	play_scout_show = play_scout_show.masked_fill(beyond_ps, 0)
	play_scout_hide = play_scout_hide.masked_fill(beyond_ps, 0)

	# Reduced play strength (same approach as compute_action_masks)
	is_set_type = (state.play_type == PLAY_SET)
	is_asc = (~is_set_type) & (old_play_len >= 2) & (
		state.play_show[:, 1].long() > state.play_show[:, 0].long())
	red_str_l = torch.where(is_asc,
		state.play_show[brange, plen_idx].long(),   # ascending: last unchanged
		state.play_show[:, 1].long())                # desc/set: new first = old[1]
	red_str_r = torch.where(is_asc,
		state.play_show[brange, sec_last_idx].long(), # ascending: second-to-last
		state.play_show[:, 0].long())                 # desc/set: first unchanged
	red_str = torch.where(left_end, red_str_l, red_str_r)

	# Reduced to 1 card → SET; else preserve type
	new_ptype_scout = torch.where(new_plen_scout <= 1,
		torch.full((B,), PLAY_SET, device=dev, dtype=torch.int8),
		state.play_type)

	play_owner_safe = state.play_owner.long().clamp(0)

	# scouts_since_play tracking
	new_ssp = state.scouts_since_play.long() + 1
	scout_round_end = is_scout & (new_ssp >= n_p - 1)
	play_emptied = is_scout_or_sns & (new_plen_scout == 0)
	# Clear play state when scouted empty without ending the round
	scout_clears = is_scout & (new_plen_scout == 0) & ~scout_round_end
	sns_clears = is_sns & (new_plen_scout == 0)

	# ── Apply all updates ─────────────────────────────────────────────────
	# Hands
	new_show = torch.where(is_play.view(B, 1), hand_play_show,
		torch.where(is_scout_or_sns.view(B, 1), hand_scout_show, old_show))
	new_hide = torch.where(is_play.view(B, 1), hand_play_hide,
		torch.where(is_scout_or_sns.view(B, 1), hand_scout_hide, old_hide))
	new_hl = torch.where(is_play, new_hl_play,
		torch.where(is_scout_or_sns, new_hl_scout, hl))
	cp_3d = cp.view(B, 1, 1).expand(B, 1, H)
	state.hands_show.scatter_(1, cp_3d, new_show.unsqueeze(1).to(torch.int8))
	state.hands_hide.scatter_(1, cp_3d, new_hide.unsqueeze(1).to(torch.int8))
	state.hand_len.scatter_(1, cp.view(B, 1), new_hl.unsqueeze(1).to(torch.int8))

	# Collected: current player collects old play's cards on play action
	collect_delta = torch.zeros(B, MAX_P, device=dev, dtype=torch.long)
	collect_delta.scatter_add_(1, cp.unsqueeze(1), collect_amt.unsqueeze(1))
	state.collected = (state.collected.long() + collect_delta).to(torch.int8)

	# Scout tokens: award to play_owner on scout/sns
	token_delta = torch.zeros(B, MAX_P, device=dev, dtype=torch.long)
	token_delta.scatter_add_(1, play_owner_safe.unsqueeze(1),
		is_scout_or_sns.long().unsqueeze(1))
	state.scout_tokens = (state.scout_tokens.long() + token_delta).to(torch.int8)

	# Play state
	final_pshow = torch.where(is_play.view(B, 1), new_play_show_p,
		torch.where(is_scout_or_sns.view(B, 1), play_scout_show,
		state.play_show.long()))
	final_phide = torch.where(is_play.view(B, 1), new_play_hide_p,
		torch.where(is_scout_or_sns.view(B, 1), play_scout_hide,
		state.play_hide.long()))
	state.play_show = final_pshow.to(torch.int8)
	state.play_hide = final_phide.to(torch.int8)

	state.play_len = torch.where(is_play, play_len_p,
		torch.where(is_scout_or_sns, new_plen_scout,
		old_play_len)).to(torch.int8)

	state.play_type = torch.where(is_play, new_type_p,
		torch.where(is_scout_or_sns, new_ptype_scout, state.play_type))

	state.play_strength = torch.where(is_play, new_strength_p,
		torch.where(is_scout_or_sns, red_str.to(torch.int8),
		state.play_strength))

	# play_owner: play → current player; emptied → -1; else unchanged
	state.play_owner = torch.where(is_play, cp.to(torch.int8),
		torch.where(play_emptied,
			torch.full((B,), -1, device=dev, dtype=torch.int8),
			state.play_owner))

	# scouts_since_play: play resets; scout/sns increments; cleared on empty
	final_ssp = torch.where(is_play, torch.zeros(B, device=dev, dtype=torch.long),
		torch.where(scout_clears | sns_clears,
			torch.zeros(B, device=dev, dtype=torch.long),
			torch.where(is_scout_or_sns, new_ssp,
				state.scouts_since_play.long())))
	state.scouts_since_play = final_ssp.to(torch.int8)

	# S&S availability: disable for current player on S&S action
	disable = torch.zeros(B, MAX_P, device=dev, dtype=torch.bool)
	disable.scatter_(1, cp.unsqueeze(1), is_sns.unsqueeze(1))
	state.sns_available = state.sns_available & ~disable

	# Round end
	round_end = play_round_end | scout_round_end
	state.round_ender = torch.where(play_round_end, cp.to(torch.int8),
		torch.where(scout_round_end, play_owner_safe.to(torch.int8),
		state.round_ender))
	state.done = state.done | round_end

	# Phase: play/scout → TURN; S&S → SNS_PLAY
	state.phase = torch.where(is_play | is_scout,
		torch.zeros(B, device=dev, dtype=torch.int8),
		torch.where(is_sns,
			torch.ones(B, device=dev, dtype=torch.int8),
			state.phase))

	# Advance turn: play and scout advance (unless round ended); S&S stays
	advance = (is_play | is_scout) & ~round_end
	new_cp = ((cp + 1) % n_p).to(torch.int8)
	state.current_player = torch.where(advance, new_cp, state.current_player)


# ── State utilities ──────────────────────────────────────────────────────────

def repeat_state(state: GpuGameState, n: int) -> GpuGameState:
	"""Repeat each game n times along the batch dim (for rollout expansion)."""
	from dataclasses import fields
	return GpuGameState(**{
		f.name: getattr(state, f.name).repeat_interleave(n, dim=0)
		for f in fields(state)
	})


# ── Scoring ──────────────────────────────────────────────────────────────────

def compute_scores(state: GpuGameState) -> list[list[int]]:
	"""Per-player round scores: collected + scout_tokens - hand_len (ender exempt)."""
	B = state.done.shape[0]
	dev = state.done.device
	p_idx = torch.arange(MAX_P, device=dev)
	n_p = state.num_players.long()
	is_ender = (p_idx.unsqueeze(0) == state.round_ender.long().unsqueeze(1))
	scores = (state.collected.long() + state.scout_tokens.long()
		- state.hand_len.long() * (~is_ender).long())
	result = []
	for b in range(B):
		n = n_p[b].item()
		result.append(scores[b, :n].tolist())
	return result


def compute_scores_tensor(state: GpuGameState) -> Tensor:
	"""Per-player round scores as [B, MAX_P] long tensor (stays on device).
	Slots beyond num_players are 0."""
	p_idx = torch.arange(MAX_P, device=state.done.device)
	is_ender = (p_idx.unsqueeze(0) == state.round_ender.long().unsqueeze(1))
	return (state.collected.long() + state.scout_tokens.long()
		- state.hand_len.long() * (~is_ender).long())


# ── GPU rollout ──────────────────────────────────────────────────────────────

# torch.compile wrappers — compiled once on first call, cached thereafter.
# fullgraph=False allows graph breaks from dataclass field access.
_compiled_fns: dict = {}

def _get_compiled_fns():
	"""Lazily compile GPU engine functions. Returns dict of compiled callables."""
	if not _compiled_fns:
		try:
			_compiled_fns['legal'] = torch.compile(compute_legal_plays, fullgraph=False)
			_compiled_fns['masks'] = torch.compile(compute_action_masks, fullgraph=False)
			_compiled_fns['encode'] = torch.compile(encode_states, fullgraph=False)
			_compiled_fns['apply'] = torch.compile(apply_actions, fullgraph=False)
			_compiled_fns['available'] = True
		except Exception:
			_compiled_fns['available'] = False
	return _compiled_fns


def rollout_gpu(
	state: GpuGameState,
	network,
	max_steps: int = MAX_STEPS,
	use_compile: bool = True,
) -> list[list[int]]:
	"""GPU-native drop-in replacement for rollout_from_states_batched_v6.
	Runs all games for max_steps (done games are masked), returns round scores.
	No CPU sync points during the loop — one transfer in, one out."""
	from network import batched_masked_sample
	B = state.done.shape[0]
	dev = state.done.device

	fns = _get_compiled_fns() if use_compile else {}
	do_legal = fns.get('legal', compute_legal_plays)  if fns.get('available') else compute_legal_plays
	do_masks = fns.get('masks', compute_action_masks)  if fns.get('available') else compute_action_masks
	do_encode = fns.get('encode', encode_states)       if fns.get('available') else encode_states
	do_apply = fns.get('apply', apply_actions)         if fns.get('available') else apply_actions

	network.eval()
	with torch.no_grad():
		for step in range(max_steps):
			active = ~state.done
			hand_offsets = torch.randint(0, H, (B,), device=dev, dtype=torch.long)
			legal = do_legal(state)
			masks = do_masks(state, legal, hand_offsets)
			encoded = do_encode(state, hand_offsets)
			h = network(encoded)
			logits = network.policy_logits(h)
			# Advance turn for active games with no legal actions (all via masks, no sync)
			has_action = masks.any(dim=1)  # [B] bool, stays on GPU
			no_action = active & ~has_action
			adv_cp = ((state.current_player.long() + 1) %
				state.num_players.long()).to(torch.int8)
			state.current_player = torch.where(
				no_action, adv_cp, state.current_player)
			actions = batched_masked_sample(logits, masks)
			do_apply(state, actions, hand_offsets, active & has_action)
	return compute_scores(state)
