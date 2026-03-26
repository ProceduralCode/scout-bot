import numpy as np
import torch
import torch.nn.functional as F
import random
import copy
from dataclasses import dataclass
from game import Game, Phase
from encoding import (
	encode_state, encode_hand_both_orientations, get_action_type_mask,
	get_play_start_mask, get_play_end_mask, get_scout_insert_mask,
	get_sns_insert_mask,
	get_legal_plays, decode_action_type, decode_slot_to_hand_index,
	HAND_SLOTS, PLAY_SLOTS, ACTION_TYPE_SIZE, PLAY_START_SIZE, SCOUT_INSERT_SIZE,
	# V2
	encode_state_v2, encode_hand_both_orientations_v2,
	HAND_SLOTS_V2, PLAY_START_SIZE_V2, SCOUT_INSERT_SIZE_V2,
	# V6
	encode_state_v6, encode_hand_both_orientations_v6,
	get_flat_action_mask, decode_flat_action,
	HAND_SLOTS_V6, FLAT_ACTION_SIZE,
)
from network import ScoutNetwork, FlatScoutNetwork, masked_sample, batched_masked_sample, masked_log_prob
from game_log import GameLog

@dataclass
class StepRecord:
	"""One decision point in a game, for training."""
	state: torch.Tensor
	# Which head(s) were used, and their logits/masks/choices
	action_type: int
	action_type_logits: torch.Tensor
	action_type_mask: torch.Tensor
	# Play-specific (None if not a play action)
	play_start: int | None = None
	play_start_logits: torch.Tensor | None = None
	play_start_mask: torch.Tensor | None = None
	play_end: int | None = None
	play_end_logits: torch.Tensor | None = None
	play_end_mask: torch.Tensor | None = None
	# Scout/S&S-specific (None if not a scout action)
	scout_insert: int | None = None
	scout_insert_logits: torch.Tensor | None = None
	scout_insert_mask: torch.Tensor | None = None
	# Filled in after the round
	value: float = 0.0
	reward: float = 0.0
	player: int = 0
	round_num: int = 0
	game_id: int = 0
	play_length: int | None = None
	scout_quality: int | None = None

@dataclass
class StepRecordV6:
	"""One decision point for v6 flat action space."""
	state: torch.Tensor         # [INPUT_SIZE_V6]
	action: int                 # flat index [0..383]
	mask: np.ndarray            # bool [384]
	old_log_prob: float         # log prob at collection time
	value: float
	reward: float
	player: int
	round_num: int
	game_id: int
	hand_offset: int            # needed for augmentation
	play_length: int | None
	scout_quality: int | None

def _assign_round_rewards(round_records: list[StepRecord], game: Game,
						  round_idx: int, reward_mode: str = "game_score",
						  reward_distribution: str = "terminal",
						  shaped_bonus_scale: float = 0.0):
	"""Assign rewards to step records for one round. Mutates records in place."""
	for rec in round_records:
		rec.round_num = round_idx
		rec.reward = 0.0
	if reward_mode == "play_length":
		for rec in round_records:
			rec.reward = rec.play_length / 5.0 if rec.play_length is not None else 0.0
	elif reward_mode == "play_and_scout":
		for rec in round_records:
			if rec.play_length is not None:
				rec.reward = rec.play_length / 5.0
			elif rec.scout_quality is not None:
				rec.reward = (rec.scout_quality - 1) / 8.0
	else:  # "game_score"
		round_scores = game.get_round_scores()
		player_indices: dict[int, list[int]] = {}
		for i, rec in enumerate(round_records):
			player_indices.setdefault(rec.player, []).append(i)
		# Parse distribution: "terminal", "uniform", or float 0-1 (uniform fraction)
		if reward_distribution == "uniform":
			uniform_frac = 1.0
		elif reward_distribution == "terminal":
			uniform_frac = 0.0
		else:
			uniform_frac = float(reward_distribution)
		for player_id, indices in player_indices.items():
			opponent_scores = [round_scores[j] for j in range(len(round_scores)) if j != player_id]
			mean_opponent = sum(opponent_scores) / len(opponent_scores)
			round_reward = (round_scores[player_id] - mean_opponent) / 10.0
			if uniform_frac > 0:
				per_step = round_reward * uniform_frac / len(indices)
				for i in indices:
					round_records[i].reward = per_step
			if uniform_frac < 1:
				round_records[indices[-1]].reward += round_reward * (1 - uniform_frac)
	if shaped_bonus_scale > 0:
		for rec in round_records:
			if rec.play_length is not None:
				rec.reward += rec.play_length / 5.0 * shaped_bonus_scale
			elif rec.scout_quality is not None:
				rec.reward += (rec.scout_quality - 1) / 8.0 * shaped_bonus_scale

def play_game(network: ScoutNetwork, num_players: int,
			  opponent_pool: list[ScoutNetwork] | None = None,
			  game_log: GameLog | None = None,
			  training_seats: int = 1,
			  reward_distribution: str = "terminal",  # "terminal", "uniform", or float 0-1 (uniform fraction)
			  reward_mode: str = "game_score",
			  shaped_bonus_scale: float = 0.0) -> list[StepRecord]:
	"""Play a complete game using the network, recording all decisions.
	First training_seats players use the training network, rest from opponent_pool.
	Only returns records from players using the training network — opponent
	records are discarded since their old logits would corrupt PPO ratios.
	If game_log is provided, human-readable events are recorded into it."""
	game = Game(num_players)
	ev = getattr(network, 'encoding_version', 1)
	if ev == 2:
		game.starting_player = random.randint(0, num_players - 1)
		game.total_rounds = 1
	networks = []
	for i in range(num_players):
		if i < training_seats:
			networks.append(network)
		elif opponent_pool:
			networks.append(random.choice(opponent_pool))
		else:
			networks.append(network)
	all_records: list[StepRecord] = []
	for round_idx in range(game.total_rounds):
		game.start_round()
		if game_log:
			game_log.record_round_start(game)
		round_records = _play_round(game, networks, game_log)
		_assign_round_rewards(round_records, game, round_idx, reward_mode,
							  reward_distribution, shaped_bonus_scale)
		all_records.extend(round_records)
		if game_log:
			game_log.record_round_end(game)
	if game_log:
		game_log.record_game_end(game.cumulative_scores)
	# Filter to only records from the training network
	records = [r for r in all_records if networks[r.player] is network]
	return records

def play_eval_game(networks: list, num_players: int,
				   game_log: GameLog | None = None) -> list[int]:
	"""Play a game with specific networks assigned to seats.
	Returns cumulative scores for all players."""
	game = Game(num_players)
	for _ in range(game.total_rounds):
		game.start_round()
		if game_log:
			game_log.record_round_start(game)
		_play_round(game, networks, game_log)
		if game_log:
			game_log.record_round_end(game)
	if game_log:
		game_log.record_game_end(game.cumulative_scores)
	return game.cumulative_scores

def rollout_from_state(game_snapshot: Game, network: ScoutNetwork) -> list[int]:
	"""Play a game snapshot to round completion using network for all seats.
	Returns round scores for all players."""
	game = copy.deepcopy(game_snapshot)
	num_players = game.num_players
	networks = [network] * num_players
	# Play remaining turns until round ends
	while game.phase in (Phase.TURN, Phase.SNS_PLAY):
		_play_turn(game, networks)
	return game.get_round_scores()

def rollout_from_states_batched(snapshots: list[Game], network: ScoutNetwork) -> list[list[int]]:
	"""Play multiple game snapshots to round completion with batched forward passes.
	Each snapshot is deepcopied and played independently using network for all seats.
	Returns list of round scores (one per snapshot)."""
	if not snapshots:
		return []
	ev = getattr(network, 'encoding_version', 1)
	v2 = ev == 2
	_hs = HAND_SLOTS_V2 if v2 else HAND_SLOTS
	_sis = SCOUT_INSERT_SIZE_V2 if v2 else SCOUT_INSERT_SIZE
	_pss = PLAY_START_SIZE_V2 if v2 else PLAY_START_SIZE
	games = [copy.deepcopy(s) for s in snapshots]
	with torch.no_grad():
		while True:
			# Collect games that still need turns played
			pending = []  # (game_idx, player, hand_offset, state_tensor)
			for g_idx, g in enumerate(games):
				if g.phase in (Phase.TURN, Phase.SNS_PLAY):
					p = g.current_player
					if v2:
						ho = random.randint(0, HAND_SLOTS_V2 - 1)
						state = encode_state_v2(g, p, ho)
					else:
						ho = random.randint(0, HAND_SLOTS - 1)
						po = random.randint(0, PLAY_SLOTS - 1)
						state = encode_state(g, p, ho, po)
					pending.append((g_idx, p, ho, 0 if v2 else po, state))
			if not pending:
				break
			B = len(pending)
			states = torch.stack([p[4] for p in pending])
			hidden_batch = network(states)
			# --- Action type ---
			tp_games = []
			for g_idx, p, ho, po, state in pending:
				g = games[g_idx]
				hand = g.players[p].hand
				legal_plays = get_legal_plays(hand, g.current_play)
				tp_games.append((g, g_idx, p, ho, po, hand, legal_plays))
			at_cond = _build_batch_conditioning(hidden_batch, None, None, play_start_size=_pss)
			at_logits_batch = network.action_type_head(at_cond)
			at_masks_np = [get_action_type_mask(tp[0], tp[6], max_hand=_hs) for tp in tp_games]
			at_masks = torch.from_numpy(np.stack(at_masks_np))
			has_action = at_masks.any(dim=1)
			for bi in range(B):
				if not has_action[bi]:
					tp_games[bi][0]._advance_turn()
			action_types = batched_masked_sample(at_logits_batch, at_masks)
			action_infos = [decode_action_type(action_types[bi].item()) for bi in range(B)]
			play_bi = [bi for bi in range(B) if has_action[bi] and action_infos[bi]["type"] == "play"]
			scout_all = [bi for bi in range(B) if has_action[bi] and action_infos[bi]["type"] in ("scout", "sns")]
			# --- Play start + end ---
			play_starts = {}
			play_ends = {}
			if play_bi:
				p_idx = torch.tensor(play_bi, dtype=torch.long)
				ps_cond = _build_batch_conditioning(
					hidden_batch[p_idx], action_types[p_idx], None, play_start_size=_pss)
				ps_logits = network.play_start_head(ps_cond)
				ps_masks = torch.from_numpy(np.stack(
					[get_play_start_mask(tp_games[bi][6], tp_games[bi][3], num_slots=_hs) for bi in play_bi]))
				ps_samples = batched_masked_sample(ps_logits, ps_masks)
				for i, bi in enumerate(play_bi):
					play_starts[bi] = ps_samples[i].item()
				pe_cond = _build_batch_conditioning(
					hidden_batch[p_idx], action_types[p_idx], ps_samples, play_start_size=_pss)
				pe_logits = network.play_end_head(pe_cond)
				pe_masks = torch.from_numpy(np.stack([
					get_play_end_mask(tp_games[bi][6],
						decode_slot_to_hand_index(play_starts[bi], tp_games[bi][3], num_slots=_hs),
						tp_games[bi][3], num_slots=_hs)
					for bi in play_bi]))
				pe_samples = batched_masked_sample(pe_logits, pe_masks)
				for i, bi in enumerate(play_bi):
					play_ends[bi] = pe_samples[i].item()
			# --- Scout / S&S insert ---
			scout_inserts = {}
			if scout_all:
				s_idx = torch.tensor(scout_all, dtype=torch.long)
				si_cond = _build_batch_conditioning(
					hidden_batch[s_idx], action_types[s_idx], None, play_start_size=_pss)
				si_logits = network.scout_insert_head(si_cond)
				si_masks_list = []
				for bi in scout_all:
					if action_infos[bi]["type"] == "scout":
						si_masks_list.append(get_scout_insert_mask(tp_games[bi][0], tp_games[bi][3], num_slots=_sis))
					else:
						si_masks_list.append(get_sns_insert_mask(
							tp_games[bi][0], action_infos[bi]["left_end"],
							action_infos[bi]["flip"], tp_games[bi][3], num_slots=_sis))
				si_masks = torch.from_numpy(np.stack(si_masks_list))
				si_samples = batched_masked_sample(si_logits, si_masks)
				for i, bi in enumerate(scout_all):
					scout_inserts[bi] = si_samples[i].item()
			# --- Apply game mutations ---
			for bi in range(B):
				if not has_action[bi]:
					continue
				g, g_idx, p, ho, po, hand, legal_plays = tp_games[bi]
				info = action_infos[bi]
				if info["type"] == "play":
					start_idx = decode_slot_to_hand_index(play_starts[bi], ho, num_slots=_hs)
					end_idx = decode_slot_to_hand_index(play_ends[bi], ho, num_slots=_hs)
					g.apply_play(start_idx, end_idx)
				elif info["type"] == "scout":
					insert_pos = (scout_inserts[bi] - ho) % _sis
					g.apply_scout(info["left_end"], info["flip"], insert_pos)
				elif info["type"] == "sns":
					insert_pos = (scout_inserts[bi] - ho) % _sis
					g.apply_sns_scout(info["left_end"], info["flip"], insert_pos)
	return [g.get_round_scores() for g in games]

def play_games_with_rollouts(network: ScoutNetwork, num_games: int,
							 num_players: int, rollouts_per_state: int = 10,
							 training_seats: int = 4) -> tuple[list[StepRecord], list[float]]:
	"""Play games with rollout-based advantage estimation.
	At each decision point, snapshots the game state. After the game,
	runs rollouts from each snapshot to estimate state values.
	Returns (records, normalized_advantages)."""
	ev = getattr(network, 'encoding_version', 1)
	all_records: list[StepRecord] = []
	all_advantages: list[float] = []
	network.eval()
	with torch.no_grad():
		for game_idx in range(num_games):
			game = Game(num_players)
			if ev == 2:
				game.starting_player = random.randint(0, num_players - 1)
				game.total_rounds = 1
			networks = [network] * num_players
			game.start_round()
			# Flip decisions
			for p in range(num_players):
				net = networks[p]
				ev_p = getattr(net, 'encoding_version', 1)
				if ev_p == 2:
					ho = random.randint(0, HAND_SLOTS_V2 - 1)
					t_normal, t_flipped = encode_hand_both_orientations_v2(game, p, ho)
				else:
					ho = random.randint(0, HAND_SLOTS - 1)
					po = random.randint(0, PLAY_SLOTS - 1)
					t_normal, t_flipped = encode_hand_both_orientations(game, p, ho, po)
				h_normal = net(t_normal)
				h_flipped = net(t_flipped)
				v_normal = net.value(h_normal).item()
				v_flipped = net.value(h_flipped).item()
				game.submit_flip_decision(p, do_flip=v_flipped > v_normal)
			# Play turns, snapshotting before each decision
			snapshots = []  # (game_snapshot, record_indices)
			records = []
			snapshots.append(copy.deepcopy(game))
			while game.phase in (Phase.TURN, Phase.SNS_PLAY):
				pre_len = len(records)
				step_records = _play_turn(game, networks)
				records.extend(step_records)
				# Snapshot after action (= before next action)
				if step_records:
					snapshots.append(copy.deepcopy(game))
			# Map each record to its (before_snapshot_idx, after_snapshot_idx)
			# Records are created in order, one snapshot between each action
			# snapshot[0] = before first action, snapshot[1] = after first action, etc.
			# Some _play_turn calls produce 0 records (skip) or 2 records (S&S).
			# Walk through and assign snapshot indices to records.
			record_snapshot_pairs = []  # (before_idx, after_idx) per record
			snap_idx = 0
			rec_cursor = 0
			while rec_cursor < len(records):
				# Find how many records came from this turn
				# Each _play_turn call produced records between snapshots[snap_idx] and snapshots[snap_idx+1]
				# We need to replay the turn structure. A turn can produce:
				# - 0 records (skip, no snapshot added)
				# - 1 record (play or scout)
				# - 2 records (S&S: scout + forced play)
				# Snapshots were added after each _play_turn that produced records.
				# So snap_idx corresponds to before, snap_idx+1 to after.
				# For S&S with 2 records, both share the same snapshot pair.
				# Find the next batch: records from the same turn share the same player
				# and were added in one _play_turn call
				turn_start = rec_cursor
				# S&S produces 2 records: scout then play. The play record has
				# a different action_type but was from the same _play_turn call.
				# We know the turn boundary: after the first record, if the next
				# record is an S&S forced play (action_type == 0 following an sns),
				# they belong together.
				rec_cursor += 1
				if rec_cursor < len(records) and records[turn_start].action_type >= 5:
					# S&S scout — next record is the forced play
					if rec_cursor < len(records):
						rec_cursor += 1
				for i in range(turn_start, rec_cursor):
					record_snapshot_pairs.append((snap_idx, snap_idx + 1))
				snap_idx += 1
			# Run rollouts from all snapshots in one batched call
			# Expand: each snapshot × rollouts_per_state copies
			expanded = [snap for snap in snapshots for _ in range(rollouts_per_state)]
			all_scores = rollout_from_states_batched(expanded, network)
			# Aggregate back to per-snapshot, per-player margins
			num_snapshots = len(snapshots)
			snapshot_values = []  # [snap_idx][player] = avg_margin
			for snap_idx in range(num_snapshots):
				player_margins = [0.0] * num_players
				base = snap_idx * rollouts_per_state
				for r in range(rollouts_per_state):
					scores = all_scores[base + r]
					for p in range(num_players):
						opp_scores = [scores[j] for j in range(num_players) if j != p]
						margin = (scores[p] - sum(opp_scores) / len(opp_scores)) / 10.0
						player_margins[p] += margin
				snapshot_values.append([m / rollouts_per_state for m in player_margins])
			# Compute advantages: V_after - V_before for each record's player
			game_advantages = []
			for rec_idx, (before_snap, after_snap) in enumerate(record_snapshot_pairs):
				p = records[rec_idx].player
				v_before = snapshot_values[before_snap][p]
				v_after = snapshot_values[after_snap][p]
				game_advantages.append(v_after - v_before)
			# Store the rollout value as record.value (for value function training)
			for rec_idx, (before_snap, _) in enumerate(record_snapshot_pairs):
				p = records[rec_idx].player
				records[rec_idx].value = snapshot_values[before_snap][p]
			# Assign game_id and filter to training seats
			for rec in records:
				rec.game_id = game_idx
			# Keep all records since training_seats=4 means all players use training network
			filtered_indices = [i for i, r in enumerate(records) if r.player < training_seats]
			all_records.extend(records[i] for i in filtered_indices)
			all_advantages.extend(game_advantages[i] for i in filtered_indices)
	# Diagnostic: per-action-type advantage statistics (before normalization)
	if all_advantages:
		play_advs = [a for a, r in zip(all_advantages, all_records) if r.action_type < 5]
		scout_advs = [a for a, r in zip(all_advantages, all_records) if r.action_type >= 5 and r.scout_insert is not None]
		at_only_advs = [a for a, r in zip(all_advantages, all_records) if r.action_type >= 5 and r.scout_insert is None]
		def _stats(vals, label):
			if not vals:
				return
			m = sum(vals) / len(vals)
			s = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
			print(f"  ADV-DIAG {label}: n={len(vals)}  mean={m:+.5f}  std={s:.5f}  "
				  f"range=[{min(vals):+.5f}, {max(vals):+.5f}]")
		_stats(all_advantages, "ALL")
		_stats(play_advs, "PLAY")
		_stats(scout_advs, "SCOUT")
	# Normalize advantages
	if all_advantages:
		mean = sum(all_advantages) / len(all_advantages)
		std = (sum((a - mean) ** 2 for a in all_advantages) / len(all_advantages)) ** 0.5
		all_advantages = [(a - mean) / (std + 1e-8) for a in all_advantages]
	return all_records, all_advantages

def _play_round(game: Game, networks: list[ScoutNetwork],
				game_log: GameLog | None = None) -> list[StepRecord]:
	"""Play one round, returning step records for all players."""
	records = []
	# Flip decisions
	for p in range(game.num_players):
		net = networks[p]
		with torch.no_grad():
			ev = getattr(net, 'encoding_version', 1)
			if ev == 6:
				ho = random.randint(0, HAND_SLOTS_V6 - 1)
				t_normal, t_flipped = encode_hand_both_orientations_v6(game, p, ho)
			elif ev == 2:
				ho = random.randint(0, HAND_SLOTS_V2 - 1)
				t_normal, t_flipped = encode_hand_both_orientations_v2(game, p, ho)
			else:
				ho = random.randint(0, HAND_SLOTS - 1)
				po = random.randint(0, PLAY_SLOTS - 1)
				t_normal, t_flipped = encode_hand_both_orientations(game, p, ho, po)
			h_normal = net(t_normal)
			h_flipped = net(t_flipped)
			v_normal = net.value(h_normal).item()
			v_flipped = net.value(h_flipped).item()
		did_flip = v_flipped > v_normal
		game.submit_flip_decision(p, do_flip=did_flip)
		if game_log:
			game_log.record_flip(game.round_number, p, did_flip, game.players[p].hand)
	# Play turns
	while game.phase == Phase.TURN:
		step_records = _play_turn(game, networks, game_log)
		records.extend(step_records)
	return records

def _play_turn(game: Game, networks: list[ScoutNetwork],
			   game_log: GameLog | None = None) -> list[StepRecord]:
	"""Execute one turn, returning step records."""
	p = game.current_player
	net = networks[p]
	ev = getattr(net, 'encoding_version', 1)
	if ev == 6:
		return _play_turn_v6(game, networks, game_log)
	_hs = HAND_SLOTS_V2 if ev == 2 else HAND_SLOTS
	_sis = SCOUT_INSERT_SIZE_V2 if ev == 2 else SCOUT_INSERT_SIZE
	hand_offset = random.randint(0, _hs - 1)
	round_num = game.round_number  # capture before apply_* may increment it
	records = []
	with torch.no_grad():
		if ev == 2:
			state_tensor = encode_state_v2(game, p, hand_offset)
		else:
			play_offset = random.randint(0, PLAY_SLOTS - 1)
			state_tensor = encode_state(game, p, hand_offset, play_offset)
		hidden = net(state_tensor)
		value = net.value(hidden).item()
		# Compute legal plays once for all mask functions
		hand = game.players[p].hand
		legal_plays = get_legal_plays(hand, game.current_play)
		# Step 1: action type
		at_logits = net.action_type_logits(hidden)
		at_mask_np = get_action_type_mask(game, legal_plays, max_hand=_hs)
		# Edge case: hand full and no legal plays — skip turn
		if not at_mask_np.any():
			game._advance_turn()
			return records
		at_mask = torch.from_numpy(at_mask_np)
		action_type, at_log_prob = masked_sample(at_logits, at_mask)
		rec = StepRecord(
			state=state_tensor,
			action_type=action_type,
			action_type_logits=at_logits,
			action_type_mask=at_mask_np,
			value=value,
			player=p,
		)
		action_info = decode_action_type(action_type)
		if action_info["type"] == "play":
			# Step 2: play start
			ps_logits = net.play_start_logits(hidden, action_type)
			ps_mask_np = get_play_start_mask(legal_plays, hand_offset, num_slots=_hs)
			ps_mask = torch.from_numpy(ps_mask_np)
			start_slot, _ = masked_sample(ps_logits, ps_mask)
			start_idx = decode_slot_to_hand_index(start_slot, hand_offset, num_slots=_hs)
			rec.play_start = start_slot
			rec.play_start_logits = ps_logits
			rec.play_start_mask = ps_mask_np
			# Step 3: play end
			pe_logits = net.play_end_logits(hidden, action_type, start_slot)
			pe_mask_np = get_play_end_mask(legal_plays, start_idx, hand_offset, num_slots=_hs)
			pe_mask = torch.from_numpy(pe_mask_np)
			end_slot, _ = masked_sample(pe_logits, pe_mask)
			end_idx = decode_slot_to_hand_index(end_slot, hand_offset, num_slots=_hs)
			rec.play_end = end_slot
			rec.play_end_logits = pe_logits
			rec.play_end_mask = pe_mask_np
			rec.play_length = end_idx - start_idx + 1
			records.append(rec)
			played_cards = hand[start_idx:end_idx + 1]
			game.apply_play(start_idx, end_idx)
			if game_log:
				game_log.record_play(game, p, played_cards, round_num=round_num)
		elif action_info["type"] == "scout":
			# Step 2: insert position (slot space, like play_start/play_end)
			si_logits = net.scout_insert_logits(hidden, action_type)
			si_mask_np = get_scout_insert_mask(game, hand_offset, num_slots=_sis)
			si_mask = torch.from_numpy(si_mask_np)
			insert_slot, _ = masked_sample(si_logits, si_mask)
			insert_pos = (insert_slot - hand_offset) % _sis
			rec.scout_insert = insert_slot
			rec.scout_insert_logits = si_logits
			rec.scout_insert_mask = si_mask_np
			records.append(rec)
			# Capture scouted card before applying
			left_end = action_info["left_end"]
			play_cards = game.current_play.cards
			scouted = play_cards[0] if left_end else play_cards[-1]
			if action_info["flip"]:
				scouted = (scouted[1], scouted[0])
			new_hand = list(hand[:insert_pos]) + [scouted] + list(hand[insert_pos:])
			max_len = 1
			for s, e in get_legal_plays(new_hand, None):
				if s <= insert_pos <= e:
					max_len = max(max_len, e - s + 1)
			rec.scout_quality = max_len
			game.apply_scout(left_end, action_info["flip"], insert_pos)
			if game_log:
				game_log.record_scout(game, p, scouted, left_end, insert_pos, round_num=round_num)
		elif action_info["type"] == "sns":
			# Scout portion — use restricted mask so insert guarantees a legal play
			si_logits = net.scout_insert_logits(hidden, action_type)
			si_mask_np = get_sns_insert_mask(game, action_info["left_end"], action_info["flip"], hand_offset, num_slots=_sis)
			si_mask = torch.from_numpy(si_mask_np)
			insert_slot, _ = masked_sample(si_logits, si_mask)
			insert_pos = (insert_slot - hand_offset) % _sis
			rec.scout_insert = insert_slot
			rec.scout_insert_logits = si_logits
			rec.scout_insert_mask = si_mask_np
			records.append(rec)
			# Capture scouted card before applying
			left_end = action_info["left_end"]
			play_cards = game.current_play.cards
			scouted = play_cards[0] if left_end else play_cards[-1]
			if action_info["flip"]:
				scouted = (scouted[1], scouted[0])
			new_hand = list(hand[:insert_pos]) + [scouted] + list(hand[insert_pos:])
			max_len = 1
			for s, e in get_legal_plays(new_hand, None):
				if s <= insert_pos <= e:
					max_len = max(max_len, e - s + 1)
			rec.scout_quality = max_len
			game.apply_sns_scout(left_end, action_info["flip"], insert_pos)
			if game_log:
				game_log.record_scout(game, p, scouted, left_end, insert_pos, round_num=round_num)
			# Play portion (forced, separate turn record)
			# The recursive call logs the play event itself
			if game.phase == Phase.SNS_PLAY:
				sns_records = _play_turn(game, networks, game_log)
				records.extend(sns_records)
	return records

def _process_turn_from_hidden(game: Game, network: ScoutNetwork,
							  hidden: torch.Tensor, value: float,
							  state_tensor: torch.Tensor,
							  hand_offset: int, play_offset: int) -> list[StepRecord]:
	"""Execute one turn using a pre-computed hidden state.
	Like _play_turn but without the forward pass or S&S recursion —
	S&S leaves the game in SNS_PLAY phase for the batch loop to handle."""
	p = game.current_player
	ev = getattr(network, 'encoding_version', 1)
	_hs = HAND_SLOTS_V2 if ev == 2 else HAND_SLOTS
	_sis = SCOUT_INSERT_SIZE_V2 if ev == 2 else SCOUT_INSERT_SIZE
	round_num = game.round_number
	records = []
	hand = game.players[p].hand
	legal_plays = get_legal_plays(hand, game.current_play)
	# Step 1: action type
	at_logits = network.action_type_logits(hidden)
	at_mask_np = get_action_type_mask(game, legal_plays, max_hand=_hs)
	# Edge case: hand full and no legal plays — skip turn
	if not at_mask_np.any():
		game._advance_turn()
		return records
	at_mask = torch.from_numpy(at_mask_np)
	action_type, _ = masked_sample(at_logits, at_mask)
	rec = StepRecord(
		state=state_tensor,
		action_type=action_type,
		action_type_logits=at_logits,
		action_type_mask=at_mask_np,
		value=value,
		player=p,
	)
	action_info = decode_action_type(action_type)
	if action_info["type"] == "play":
		# Step 2: play start
		ps_logits = network.play_start_logits(hidden, action_type)
		ps_mask = torch.from_numpy(get_play_start_mask(legal_plays, hand_offset, num_slots=_hs))
		start_slot, _ = masked_sample(ps_logits, ps_mask)
		start_idx = decode_slot_to_hand_index(start_slot, hand_offset, num_slots=_hs)
		rec.play_start = start_slot
		rec.play_start_logits = ps_logits
		rec.play_start_mask = ps_mask.numpy()
		# Step 3: play end
		pe_logits = network.play_end_logits(hidden, action_type, start_slot)
		pe_mask = torch.from_numpy(get_play_end_mask(legal_plays, start_idx, hand_offset, num_slots=_hs))
		end_slot, _ = masked_sample(pe_logits, pe_mask)
		end_idx = decode_slot_to_hand_index(end_slot, hand_offset, num_slots=_hs)
		rec.play_end = end_slot
		rec.play_end_logits = pe_logits
		rec.play_end_mask = pe_mask.numpy()
		rec.play_length = end_idx - start_idx + 1
		records.append(rec)
		game.apply_play(start_idx, end_idx)
	elif action_info["type"] == "scout":
		si_logits = network.scout_insert_logits(hidden, action_type)
		si_mask = torch.from_numpy(get_scout_insert_mask(game, hand_offset, num_slots=_sis))
		insert_slot, _ = masked_sample(si_logits, si_mask)
		insert_pos = (insert_slot - hand_offset) % _sis
		rec.scout_insert = insert_slot
		rec.scout_insert_logits = si_logits
		rec.scout_insert_mask = si_mask.numpy()
		records.append(rec)
		left_end = action_info["left_end"]
		play_cards = game.current_play.cards
		scouted = play_cards[0] if left_end else play_cards[-1]
		if action_info["flip"]:
			scouted = (scouted[1], scouted[0])
		new_hand = list(hand[:insert_pos]) + [scouted] + list(hand[insert_pos:])
		max_len = 1
		for s, e in get_legal_plays(new_hand, None):
			if s <= insert_pos <= e:
				max_len = max(max_len, e - s + 1)
		rec.scout_quality = max_len
		game.apply_scout(left_end, action_info["flip"], insert_pos)
	elif action_info["type"] == "sns":
		si_logits = network.scout_insert_logits(hidden, action_type)
		si_mask = torch.from_numpy(get_sns_insert_mask(game, action_info["left_end"], action_info["flip"], hand_offset, num_slots=_sis))
		insert_slot, _ = masked_sample(si_logits, si_mask)
		insert_pos = (insert_slot - hand_offset) % _sis
		rec.scout_insert = insert_slot
		rec.scout_insert_logits = si_logits
		rec.scout_insert_mask = si_mask.numpy()
		records.append(rec)
		left_end = action_info["left_end"]
		play_cards = game.current_play.cards
		scouted = play_cards[0] if left_end else play_cards[-1]
		if action_info["flip"]:
			scouted = (scouted[1], scouted[0])
		new_hand = list(hand[:insert_pos]) + [scouted] + list(hand[insert_pos:])
		max_len = 1
		for s, e in get_legal_plays(new_hand, None):
			if s <= insert_pos <= e:
				max_len = max(max_len, e - s + 1)
		rec.scout_quality = max_len
		game.apply_sns_scout(left_end, action_info["flip"], insert_pos)
		# No recursive call — game enters SNS_PLAY, batch loop handles it
	return records

def play_games_batched(network: ScoutNetwork, num_games: int, num_players: int,
					   training_seats: int = 1,
					   opponent_pool: list[ScoutNetwork] | None = None,
					   reward_distribution: str = "terminal",
					   reward_mode: str = "game_score",
					   shaped_bonus_scale: float = 0.0) -> list[StepRecord]:
	"""Play multiple games simultaneously with batched forward passes.
	Same semantics as calling play_game() num_games times, but batches
	the shared-layer forward passes across all active games."""
	ev = getattr(network, 'encoding_version', 1)
	v2 = ev == 2
	_hs = HAND_SLOTS_V2 if v2 else HAND_SLOTS
	_sis = SCOUT_INSERT_SIZE_V2 if v2 else SCOUT_INSERT_SIZE
	_pss = PLAY_START_SIZE_V2 if v2 else PLAY_START_SIZE
	games = [Game(num_players) for _ in range(num_games)]
	if v2:
		for g in games:
			g.starting_player = random.randint(0, num_players - 1)
			g.total_rounds = 1
	# Set up per-game network assignments
	game_networks = []
	for _ in range(num_games):
		nets = []
		for seat in range(num_players):
			if seat < training_seats:
				nets.append(network)
			elif opponent_pool:
				nets.append(random.choice(opponent_pool))
			else:
				nets.append(network)
		game_networks.append(nets)
	all_records = [[] for _ in range(num_games)]
	total_rounds = games[0].total_rounds
	with torch.no_grad():
		for round_idx in range(total_rounds):
			round_records = [[] for _ in range(num_games)]
			for g in games:
				g.start_round()
			# === Flip phase ===
			# Collect all flip encodings grouped by network
			flip_data = []  # (game_idx, player)
			flip_normals = []
			flip_flipped = []
			for g_idx, g in enumerate(games):
				for p in range(num_players):
					net = game_networks[g_idx][p]
					ev_p = getattr(net, 'encoding_version', 1)
					if ev_p == 2:
						ho = random.randint(0, HAND_SLOTS_V2 - 1)
						t_normal, t_flipped = encode_hand_both_orientations_v2(g, p, ho)
					else:
						ho = random.randint(0, HAND_SLOTS - 1)
						po = random.randint(0, PLAY_SLOTS - 1)
						t_normal, t_flipped = encode_hand_both_orientations(g, p, ho, po)
					flip_data.append((g_idx, p))
					flip_normals.append(t_normal)
					flip_flipped.append(t_flipped)
			# Batch: all normals then all flipped through training network
			# Separate training vs opponent forward passes
			train_flip_idx = [i for i, (g_idx, p) in enumerate(flip_data)
							  if game_networks[g_idx][p] is network]
			if train_flip_idx:
				normals = torch.stack([flip_normals[i] for i in train_flip_idx])
				flipped = torch.stack([flip_flipped[i] for i in train_flip_idx])
				h_normals = network(normals)
				h_flipped = network(flipped)
				v_normals = network.value(h_normals).squeeze(-1)
				v_flipped = network.value(h_flipped).squeeze(-1)
				for batch_i, fi in enumerate(train_flip_idx):
					g_idx, p = flip_data[fi]
					did_flip = v_flipped[batch_i].item() > v_normals[batch_i].item()
					games[g_idx].submit_flip_decision(p, do_flip=did_flip)
			# Opponent flips (unbatched, different networks)
			opp_flip_idx = [i for i in range(len(flip_data)) if i not in set(train_flip_idx)]
			for fi in opp_flip_idx:
				g_idx, p = flip_data[fi]
				net = game_networks[g_idx][p]
				h_n = net(flip_normals[fi])
				h_f = net(flip_flipped[fi])
				did_flip = net.value(h_f).item() > net.value(h_n).item()
				games[g_idx].submit_flip_decision(p, do_flip=did_flip)
			# === Turn phase ===
			while any(g.phase in (Phase.TURN, Phase.SNS_PLAY) for g in games):
				# Collect pending decisions
				pending = []  # (game_idx, player, hand_offset, play_offset, state_tensor)
				for g_idx, g in enumerate(games):
					if g.phase in (Phase.TURN, Phase.SNS_PLAY):
						p = g.current_player
						net = game_networks[g_idx][p]
						ev_p = getattr(net, 'encoding_version', 1)
						if ev_p == 2:
							ho = random.randint(0, HAND_SLOTS_V2 - 1)
							state = encode_state_v2(g, p, ho)
						else:
							ho = random.randint(0, HAND_SLOTS - 1)
							po = random.randint(0, PLAY_SLOTS - 1)
							state = encode_state(g, p, ho, po)
						pending.append((g_idx, p, ho, 0 if ev_p == 2 else po, state))
				if not pending:
					break
				# Split into training network vs opponent network
				train_pend = [(i, p) for i, p in enumerate(pending)
							  if game_networks[p[0]][p[1]] is network]
				opp_pend = [(i, p) for i, p in enumerate(pending)
							if game_networks[p[0]][p[1]] is not network]
				# Batched forward pass + sub-heads for training network
				if train_pend:
					B = len(train_pend)
					states = torch.stack([pending[i][4] for i, _ in train_pend])
					hidden_batch = network(states)
					values = network.value(hidden_batch).squeeze(-1)
					# Per-game data needed for masks and game mutations
					tp_games = []  # (game, g_idx, p, ho, po, state, hand, legal_plays)
					for pend_i, _ in train_pend:
						g_idx, p, ho, po, state = pending[pend_i]
						g = games[g_idx]
						hand = g.players[p].hand
						legal_plays = get_legal_plays(hand, g.current_play)
						tp_games.append((g, g_idx, p, ho, po, state, hand, legal_plays))
					# --- Action type (all games) ---
					at_cond = _build_batch_conditioning(hidden_batch, None, None, play_start_size=_pss)
					at_logits_batch = network.action_type_head(at_cond)
					at_masks_np = [get_action_type_mask(tp[0], tp[7], max_hand=_hs) for tp in tp_games]
					at_masks = torch.from_numpy(np.stack(at_masks_np))
					# Handle skip-turn: games with no legal actions
					has_action = at_masks.any(dim=1)
					for bi in range(B):
						if not has_action[bi]:
							tp_games[bi][0]._advance_turn()
					action_types = batched_masked_sample(at_logits_batch, at_masks)
					# Decode action types and partition into groups
					action_infos = [decode_action_type(action_types[bi].item()) for bi in range(B)]
					play_bi = [bi for bi in range(B) if has_action[bi] and action_infos[bi]["type"] == "play"]
					scout_all = [bi for bi in range(B) if has_action[bi] and action_infos[bi]["type"] in ("scout", "sns")]
					# --- Play start + end ---
					play_starts = {}  # bi → start_slot
					play_ends = {}  # bi → end_slot
					play_sub = {bi: i for i, bi in enumerate(play_bi)}
					if play_bi:
						p_idx = torch.tensor(play_bi, dtype=torch.long)
						ps_cond = _build_batch_conditioning(
							hidden_batch[p_idx], action_types[p_idx], None, play_start_size=_pss)
						ps_logits = network.play_start_head(ps_cond)
						ps_masks_np = [get_play_start_mask(tp_games[bi][7], tp_games[bi][3], num_slots=_hs) for bi in play_bi]
						ps_masks = torch.from_numpy(np.stack(ps_masks_np))
						ps_samples = batched_masked_sample(ps_logits, ps_masks)
						for i, bi in enumerate(play_bi):
							play_starts[bi] = ps_samples[i].item()
						pe_cond = _build_batch_conditioning(
							hidden_batch[p_idx], action_types[p_idx], ps_samples, play_start_size=_pss)
						pe_logits = network.play_end_head(pe_cond)
						pe_masks_list = []
						for i, bi in enumerate(play_bi):
							start_idx = decode_slot_to_hand_index(play_starts[bi], tp_games[bi][3], num_slots=_hs)
							pe_masks_list.append(get_play_end_mask(tp_games[bi][7], start_idx, tp_games[bi][3], num_slots=_hs))
						pe_masks_np = pe_masks_list
						pe_masks = torch.from_numpy(np.stack(pe_masks_np))
						pe_samples = batched_masked_sample(pe_logits, pe_masks)
						for i, bi in enumerate(play_bi):
							play_ends[bi] = pe_samples[i].item()
					# --- Scout / S&S insert ---
					scout_inserts = {}  # bi → insert_slot
					scout_sub = {bi: i for i, bi in enumerate(scout_all)}
					if scout_all:
						s_idx = torch.tensor(scout_all, dtype=torch.long)
						si_cond = _build_batch_conditioning(
							hidden_batch[s_idx], action_types[s_idx], None, play_start_size=_pss)
						si_logits = network.scout_insert_head(si_cond)
						si_masks_list = []
						for bi in scout_all:
							if action_infos[bi]["type"] == "scout":
								si_masks_list.append(get_scout_insert_mask(tp_games[bi][0], tp_games[bi][3], num_slots=_sis))
							else:
								si_masks_list.append(get_sns_insert_mask(
									tp_games[bi][0], action_infos[bi]["left_end"],
									action_infos[bi]["flip"], tp_games[bi][3], num_slots=_sis))
						si_masks_np = si_masks_list
						si_masks = torch.from_numpy(np.stack(si_masks_np))
						si_samples = batched_masked_sample(si_logits, si_masks)
						for i, bi in enumerate(scout_all):
							scout_inserts[bi] = si_samples[i].item()
					# --- Apply game mutations, build StepRecords ---
					for bi in range(B):
						if not has_action[bi]:
							continue
						g, g_idx, p, ho, po, state_tensor, hand, legal_plays = tp_games[bi]
						at = action_types[bi].item()
						info = action_infos[bi]
						rec = StepRecord(
							state=state_tensor,
							action_type=at,
							action_type_logits=at_logits_batch[bi],
							action_type_mask=at_masks_np[bi],
							value=values[bi].item(),
							player=p,
						)
						if info["type"] == "play":
							start_slot = play_starts[bi]
							end_slot = play_ends[bi]
							start_idx = decode_slot_to_hand_index(start_slot, ho, num_slots=_hs)
							end_idx = decode_slot_to_hand_index(end_slot, ho, num_slots=_hs)
							rec.play_start = start_slot
							rec.play_start_logits = ps_logits[play_sub[bi]]
							rec.play_start_mask = ps_masks_np[play_sub[bi]]
							rec.play_end = end_slot
							rec.play_end_logits = pe_logits[play_sub[bi]]
							rec.play_end_mask = pe_masks_np[play_sub[bi]]
							rec.play_length = end_idx - start_idx + 1
							round_records[g_idx].append(rec)
							g.apply_play(start_idx, end_idx)
						elif info["type"] == "scout":
							insert_slot = scout_inserts[bi]
							insert_pos = (insert_slot - ho) % _sis
							si_sub_idx = scout_sub[bi]
							rec.scout_insert = insert_slot
							rec.scout_insert_logits = si_logits[si_sub_idx]
							rec.scout_insert_mask = si_masks_np[si_sub_idx]
							round_records[g_idx].append(rec)
							left_end = info["left_end"]
							play_cards = g.current_play.cards
							scouted = play_cards[0] if left_end else play_cards[-1]
							if info["flip"]:
								scouted = (scouted[1], scouted[0])
							new_hand = list(hand[:insert_pos]) + [scouted] + list(hand[insert_pos:])
							max_len = 1
							for s, e in get_legal_plays(new_hand, None):
								if s <= insert_pos <= e:
									max_len = max(max_len, e - s + 1)
							rec.scout_quality = max_len
							g.apply_scout(left_end, info["flip"], insert_pos)
						elif info["type"] == "sns":
							insert_slot = scout_inserts[bi]
							insert_pos = (insert_slot - ho) % _sis
							si_sub_idx = scout_sub[bi]
							rec.scout_insert = insert_slot
							rec.scout_insert_logits = si_logits[si_sub_idx]
							rec.scout_insert_mask = si_masks_np[si_sub_idx]
							round_records[g_idx].append(rec)
							left_end = info["left_end"]
							play_cards = g.current_play.cards
							scouted = play_cards[0] if left_end else play_cards[-1]
							if info["flip"]:
								scouted = (scouted[1], scouted[0])
							new_hand = list(hand[:insert_pos]) + [scouted] + list(hand[insert_pos:])
							max_len = 1
							for s, e in get_legal_plays(new_hand, None):
								if s <= insert_pos <= e:
									max_len = max(max_len, e - s + 1)
							rec.scout_quality = max_len
							g.apply_sns_scout(left_end, info["flip"], insert_pos)
							# Game enters SNS_PLAY, batch loop handles forced play next iteration
				# Opponent turns (unbatched)
				for pend_i, pend_data in opp_pend:
					g_idx, p, ho, po, state = pending[pend_i]
					net = game_networks[g_idx][p]
					hidden = net(state)
					value = net.value(hidden).item()
					recs = _process_turn_from_hidden(
						games[g_idx], net, hidden, value, state, ho, po)
					round_records[g_idx].extend(recs)
			# Assign rewards for this round
			for g_idx, g in enumerate(games):
				_assign_round_rewards(round_records[g_idx], g, round_idx,
									  reward_mode, reward_distribution, shaped_bonus_scale)
				all_records[g_idx].extend(round_records[g_idx])
	# Flatten and filter to training network records
	records = []
	for g_idx in range(num_games):
		for r in all_records[g_idx]:
			r.game_id = g_idx
		records.extend(r for r in all_records[g_idx]
					   if game_networks[g_idx][r.player] is network)
	return records

class OpponentPool:
	"""Pool of past network versions for diverse self-play."""
	def __init__(self, max_size: int = 10):
		self.max_size = max_size
		self.versions: list[ScoutNetwork] = []
	def add(self, network: ScoutNetwork):
		snapshot = copy.deepcopy(network)
		snapshot.eval()
		self.versions.append(snapshot)
		if len(self.versions) > self.max_size:
			self.versions.pop(0)
	def sample(self, count: int) -> list[ScoutNetwork]:
		"""Sample networks from the pool. Returns empty list if pool is empty."""
		if not self.versions:
			return []
		return [random.choice(self.versions) for _ in range(count)]
	def state_dicts(self) -> list[dict]:
		return [{"layer_sizes": v.layer_sizes,
				 "encoding_version": getattr(v, 'encoding_version', 1),
				 "state_dict": v.state_dict()}
				for v in self.versions]
	def load_state_dicts(self, states: list[dict], template: ScoutNetwork):
		"""Restore pool from saved state dicts. Handles per-member architecture
		(new format) and bare state dicts (old format, uses template)."""
		from encoding import (INPUT_SIZE_V2, PLAY_START_SIZE_V2,
							  PLAY_END_SIZE_V2, SCOUT_INSERT_SIZE_V2,
							  INPUT_SIZE_V6)
		self.versions = []
		for entry in states:
			if isinstance(entry, dict) and "layer_sizes" in entry:
				ev = entry.get("encoding_version", 1)
				if ev == 6:
					net = FlatScoutNetwork(INPUT_SIZE_V6, entry["layer_sizes"],
						encoding_version=6)
				elif ev == 2:
					net = ScoutNetwork(INPUT_SIZE_V2, entry["layer_sizes"],
						play_start_size=PLAY_START_SIZE_V2,
						play_end_size=PLAY_END_SIZE_V2,
						scout_insert_size=SCOUT_INSERT_SIZE_V2,
						encoding_version=2)
				else:
					net = ScoutNetwork(layer_sizes=entry["layer_sizes"])
				net.load_state_dict(entry["state_dict"])
			else:
				net = copy.deepcopy(template)
				net.load_state_dict(entry)
			net.eval()
			self.versions.append(net)

def compute_gae(records: list[StepRecord], gamma: float = 0.99,
				lam: float = 0.95) -> tuple[list[float], list[float], float]:
	"""Compute GAE advantages and value targets.
	Groups records by (game_id, round_num, player) and walks backward within each group.
	Returns (normalized_advantages, returns, raw_advantage_std)."""
	if not records:
		return [], [], 0.0
	# Group record indices by (round_num, player)
	groups: dict[tuple[int, int], list[int]] = {}
	for i, rec in enumerate(records):
		groups.setdefault((rec.game_id, rec.round_num, rec.player), []).append(i)
	# Sanity check: no group should exceed max decisions per round per player
	MAX_GROUP_SIZE = 60
	for key, indices in groups.items():
		if len(indices) > MAX_GROUP_SIZE:
			print(f"  WARNING: GAE group {key} has {len(indices)} records (max expected {MAX_GROUP_SIZE})")
	advantages = [0.0] * len(records)
	returns = [0.0] * len(records)
	for indices in groups.values():
		gae = 0.0
		for t in reversed(range(len(indices))):
			idx = indices[t]
			# V(s_{t+1}) from next record's value, 0 at round end
			if t < len(indices) - 1:
				next_value = records[indices[t + 1]].value
			else:
				next_value = 0.0
			delta = records[idx].reward + gamma * next_value - records[idx].value
			gae = delta + gamma * lam * gae
			advantages[idx] = gae
			returns[idx] = gae + records[idx].value
	# Compute std before normalizing
	mean = sum(advantages) / len(advantages)
	std = (sum((a - mean) ** 2 for a in advantages) / len(advantages)) ** 0.5
	normalized = [(a - mean) / (std + 1e-8) for a in advantages]
	return normalized, returns, std

def _batched_masked_log_prob(logits: torch.Tensor, masks: torch.Tensor,
							actions: torch.Tensor) -> torch.Tensor:
	"""Batched masked log prob. logits/masks: [B, C], actions: [B] → [B]."""
	masked = logits.masked_fill(~masks, float('-inf'))
	log_probs = torch.log_softmax(masked, dim=-1)
	return log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

def _batched_masked_entropy(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
	"""Batched masked entropy. logits/masks: [B, C] → [B]."""
	masked = logits.masked_fill(~masks, float('-inf'))
	probs = torch.softmax(masked, dim=-1)
	return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

def _build_batch_conditioning(hidden: torch.Tensor,
							  action_types: torch.Tensor | None,
							  starts: torch.Tensor | None,
							  play_start_size: int = PLAY_START_SIZE) -> torch.Tensor:
	"""Build batched conditioning vectors for sub-heads.
	hidden: [B, H], action_types/starts: [B] LongTensor or None.
	Returns [B, H + ACTION_TYPE_SIZE + play_start_size]."""
	B = hidden.shape[0]
	device = hidden.device
	if action_types is not None:
		at_oh = F.one_hot(action_types.long(), ACTION_TYPE_SIZE).float().to(device)
	else:
		at_oh = torch.zeros(B, ACTION_TYPE_SIZE, device=device)
	if starts is not None:
		st_oh = F.one_hot(starts.long(), play_start_size).float().to(device)
	else:
		st_oh = torch.zeros(B, play_start_size, device=device)
	return torch.cat([hidden, at_oh, st_oh], dim=1)

def prepare_ppo_batch(steps: list[StepRecord], advantages: list[float],
					  returns: list[float] | None = None) -> dict:
	"""Pre-stack all StepRecord tensors into batched tensors. Call once, reuse across PPO epochs."""
	n = len(steps)
	if n == 0:
		return None
	batch = {
		"n": n,
		"states": torch.stack([s.state for s in steps]),
		"at_masks": torch.from_numpy(np.stack([s.action_type_mask for s in steps])),
		"at_actions": torch.tensor([s.action_type for s in steps], dtype=torch.long),
		"old_at_logits": torch.stack([s.action_type_logits for s in steps]),
		"adv": torch.tensor(advantages, dtype=torch.float32),
	}
	if returns is not None:
		batch["v_target"] = torch.tensor(returns, dtype=torch.float32)
	else:
		batch["v_target"] = torch.tensor([s.reward for s in steps], dtype=torch.float32)
	# Sub-head indices and tensors
	play_idx = [i for i, s in enumerate(steps) if s.play_start is not None]
	end_idx = [i for i, s in enumerate(steps) if s.play_end is not None]
	scout_idx = [i for i, s in enumerate(steps) if s.scout_insert is not None]
	if play_idx:
		batch["play_idx"] = torch.tensor(play_idx, dtype=torch.long)
		batch["play_at"] = torch.tensor([steps[i].action_type for i in play_idx], dtype=torch.long)
		batch["play_masks"] = torch.from_numpy(np.stack([steps[i].play_start_mask for i in play_idx]))
		batch["play_actions"] = torch.tensor([steps[i].play_start for i in play_idx], dtype=torch.long)
		batch["play_old_logits"] = torch.stack([steps[i].play_start_logits for i in play_idx])
	if end_idx:
		batch["end_idx"] = torch.tensor(end_idx, dtype=torch.long)
		batch["end_at"] = torch.tensor([steps[i].action_type for i in end_idx], dtype=torch.long)
		batch["end_starts"] = torch.tensor([steps[i].play_start for i in end_idx], dtype=torch.long)
		batch["end_masks"] = torch.from_numpy(np.stack([steps[i].play_end_mask for i in end_idx]))
		batch["end_actions"] = torch.tensor([steps[i].play_end for i in end_idx], dtype=torch.long)
		batch["end_old_logits"] = torch.stack([steps[i].play_end_logits for i in end_idx])
	if scout_idx:
		batch["scout_idx"] = torch.tensor(scout_idx, dtype=torch.long)
		batch["scout_at"] = torch.tensor([steps[i].action_type for i in scout_idx], dtype=torch.long)
		batch["scout_masks"] = torch.from_numpy(np.stack([steps[i].scout_insert_mask for i in scout_idx]))
		batch["scout_actions"] = torch.tensor([steps[i].scout_insert for i in scout_idx], dtype=torch.long)
		batch["scout_old_logits"] = torch.stack([steps[i].scout_insert_logits for i in scout_idx])
	return batch

def subsample_batch(batch: dict, keep_n: int) -> dict:
	"""Randomly subsample a PPO batch to keep_n samples, remapping sub-head indices."""
	n = batch["n"]
	if keep_n >= n:
		return batch
	keep_n = max(1, keep_n)
	keep = torch.randperm(n)[:keep_n].sort().values
	# Map old state indices → new positions (-1 = dropped)
	idx_map = torch.full((n,), -1, dtype=torch.long)
	idx_map[keep] = torch.arange(keep_n)
	result = {
		"n": keep_n,
		"states": batch["states"][keep],
		"at_masks": batch["at_masks"][keep],
		"at_actions": batch["at_actions"][keep],
		"old_at_logits": batch["old_at_logits"][keep],
		"adv": batch["adv"][keep],
		"v_target": batch["v_target"][keep],
	}
	sub_heads = [
		("play_idx", ["play_at", "play_masks", "play_actions", "play_old_logits"]),
		("end_idx", ["end_at", "end_starts", "end_masks", "end_actions", "end_old_logits"]),
		("scout_idx", ["scout_at", "scout_masks", "scout_actions", "scout_old_logits"]),
	]
	for idx_key, data_keys in sub_heads:
		if idx_key not in batch:
			continue
		old_idx = batch[idx_key]
		mask = idx_map[old_idx] >= 0
		result[idx_key] = idx_map[old_idx[mask]]
		for k in data_keys:
			result[k] = batch[k][mask]
	return result

def concatenate_batches(batches: list[dict]) -> dict:
	"""Concatenate multiple PPO batches, offsetting sub-head indices.
	Used for replay buffer: combine current + previous iterations' data."""
	if len(batches) == 1:
		return batches[0]

	combined = {
		"states": torch.cat([b["states"] for b in batches]),
		"at_masks": torch.cat([b["at_masks"] for b in batches]),
		"at_actions": torch.cat([b["at_actions"] for b in batches]),
		"old_at_logits": torch.cat([b["old_at_logits"] for b in batches]),
		"adv": torch.cat([b["adv"] for b in batches]),
		"v_target": torch.cat([b["v_target"] for b in batches]),
	}
	combined["n"] = combined["states"].shape[0]

	# Re-normalize advantages across combined batch
	adv = combined["adv"]
	combined["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)

	# Sub-heads have idx tensors (indices into states) that need offset
	sub_heads = [
		("play_idx", ["play_at", "play_masks", "play_actions", "play_old_logits"]),
		("end_idx", ["end_at", "end_starts", "end_masks", "end_actions", "end_old_logits"]),
		("scout_idx", ["scout_at", "scout_masks", "scout_actions", "scout_old_logits"]),
	]
	for idx_key, data_keys in sub_heads:
		idx_parts = []
		data_parts = {k: [] for k in data_keys}
		offset = 0
		for b in batches:
			if idx_key in b:
				idx_parts.append(b[idx_key] + offset)
				for k in data_keys:
					data_parts[k].append(b[k])
			offset += b["n"]
		if idx_parts:
			combined[idx_key] = torch.cat(idx_parts)
			for k in data_keys:
				combined[k] = torch.cat(data_parts[k])

	return combined

def ppo_update(network: ScoutNetwork, optimizer: torch.optim.Optimizer,
			   batch: dict, clip_epsilon: float = 0.2, entropy_bonus: float = 0.01,
			   value_loss_coeff: float = 0.25, max_grad_norm: float = 0.5,
			   entropy_floors: dict[str, float] | None = None,
			   entropy_floor_coeff: float = 1.0,
			   play_start_size: int = PLAY_START_SIZE):
	"""One PPO update step. Takes pre-stacked batch from prepare_ppo_batch."""
	empty_metrics = {
		"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
		"clip_fraction": 0.0, "approx_kl": 0.0, "explained_variance": 0.0,
		"entropy_action_type": 0.0, "entropy_play_start": 0.0,
		"entropy_play_end": 0.0, "entropy_scout_insert": 0.0,
		"entropy_floor_penalty": 0.0,
	}
	if batch is None:
		return empty_metrics

	n = batch["n"]

	# Batched forward pass through shared layers
	hidden_all = network(batch["states"])  # [n, hidden_size]

	# Value loss (all steps)
	v_pred = network.value(hidden_all).squeeze(-1)  # [n]
	v_target = batch["v_target"]
	value_loss = F.mse_loss(v_pred, v_target)

	# Action type (all steps, no conditioning)
	at_logits = network.action_type_logits(hidden_all)  # [n, AT_SIZE]
	at_masks = batch["at_masks"]
	at_actions = batch["at_actions"]
	old_at_logits = batch["old_at_logits"]

	log_ratio = (_batched_masked_log_prob(at_logits, at_masks, at_actions)
				 - _batched_masked_log_prob(old_at_logits, at_masks, at_actions))
	at_ent = _batched_masked_entropy(at_logits, at_masks)
	entropy = at_ent.clone()
	def _filtered_ent_mean(ent, masks):
		"""Mean entropy over steps with 2+ legal options (matches floor logic)."""
		has_choice = masks.sum(dim=-1) >= 2
		if has_choice.any():
			return ent[has_choice].mean().item()
		return 0.0
	at_entropy_mean = _filtered_ent_mean(at_ent, at_masks)
	ps_entropy_mean = 0.0
	pe_entropy_mean = 0.0
	si_entropy_mean = 0.0
	# Track per-head entropy tensors and masks for floor penalty
	_head_ent = {"action_type": (at_ent, at_masks)}
	_head_ent["play_start"] = None
	_head_ent["play_end"] = None
	_head_ent["scout_insert"] = None

	# Play start — build conditioning manually, call linear head directly
	if "play_idx" in batch:
		idx_t = batch["play_idx"]
		cond = _build_batch_conditioning(hidden_all[idx_t], batch["play_at"], None, play_start_size=play_start_size)
		logits = network.play_start_head(cond)
		masks = batch["play_masks"]
		actions = batch["play_actions"]
		old_logits = batch["play_old_logits"]
		delta = (_batched_masked_log_prob(logits, masks, actions)
				 - _batched_masked_log_prob(old_logits, masks, actions))
		# Accumulate via scatter to avoid in-place ops on graph tensors
		ps_ent = _batched_masked_entropy(logits, masks)
		log_ratio = log_ratio + torch.zeros(n).scatter(0, idx_t, delta)
		entropy = entropy + torch.zeros(n).scatter(0, idx_t, ps_ent)
		ps_entropy_mean = _filtered_ent_mean(ps_ent, masks)
		_head_ent["play_start"] = (ps_ent, masks)

	# Play end
	if "end_idx" in batch:
		idx_t = batch["end_idx"]
		cond = _build_batch_conditioning(hidden_all[idx_t], batch["end_at"], batch["end_starts"], play_start_size=play_start_size)
		logits = network.play_end_head(cond)
		masks = batch["end_masks"]
		actions = batch["end_actions"]
		old_logits = batch["end_old_logits"]
		delta = (_batched_masked_log_prob(logits, masks, actions)
				 - _batched_masked_log_prob(old_logits, masks, actions))
		pe_ent = _batched_masked_entropy(logits, masks)
		log_ratio = log_ratio + torch.zeros(n).scatter(0, idx_t, delta)
		entropy = entropy + torch.zeros(n).scatter(0, idx_t, pe_ent)
		pe_entropy_mean = _filtered_ent_mean(pe_ent, masks)
		_head_ent["play_end"] = (pe_ent, masks)

	# Scout insert
	if "scout_idx" in batch:
		idx_t = batch["scout_idx"]
		cond = _build_batch_conditioning(hidden_all[idx_t], batch["scout_at"], None, play_start_size=play_start_size)
		logits = network.scout_insert_head(cond)
		masks = batch["scout_masks"]
		actions = batch["scout_actions"]
		old_logits = batch["scout_old_logits"]
		delta = (_batched_masked_log_prob(logits, masks, actions)
				 - _batched_masked_log_prob(old_logits, masks, actions))
		si_ent = _batched_masked_entropy(logits, masks)
		log_ratio = log_ratio + torch.zeros(n).scatter(0, idx_t, delta)
		entropy = entropy + torch.zeros(n).scatter(0, idx_t, si_ent)
		si_entropy_mean = _filtered_ent_mean(si_ent, masks)
		_head_ent["scout_insert"] = (si_ent, masks)

	# PPO clipped objective
	adv = batch["adv"]
	ratio = torch.exp(log_ratio)
	surr1 = ratio * adv
	surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
	policy_loss = -torch.min(surr1, surr2).mean()

	loss = policy_loss + value_loss_coeff * value_loss - entropy_bonus * entropy.mean()

	# Per-head entropy floor penalty: quadratic penalty when mean entropy
	# drops below the floor, only for steps with 2+ legal options
	floor_penalty_val = 0.0
	if entropy_floors:
		floor_penalty = torch.tensor(0.0)
		for key, pair in _head_ent.items():
			floor = entropy_floors.get(key, 0.0)
			if floor <= 0 or pair is None:
				continue
			ent_tensor, mask_tensor = pair
			has_choice = mask_tensor.sum(dim=-1) >= 2
			if not has_choice.any():
				continue
			mean_ent = ent_tensor[has_choice].mean()
			violation = torch.clamp(floor - mean_ent, min=0.0)
			floor_penalty = floor_penalty + violation ** 2
		loss = loss + entropy_floor_coeff * floor_penalty
		floor_penalty_val = floor_penalty.item()

	if torch.isnan(loss):
		print(f"  WARNING: NaN loss detected (policy={policy_loss.item()}, value={value_loss.item()}, entropy={entropy.mean().item()})")
		return empty_metrics
	optimizer.zero_grad()
	loss.backward()
	torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=max_grad_norm)
	optimizer.step()

	# Diagnostics (detached, no grad)
	with torch.no_grad():
		mean_ratio = ratio.mean().item()
		clip_fraction = (torch.abs(ratio - 1.0) > clip_epsilon).float().mean().item()
		approx_kl = ((ratio - 1) - log_ratio).mean().item()
		var_returns = v_target.var()
		if var_returns < 1e-8:
			explained_var = 0.0
		else:
			explained_var = (1 - (v_target - v_pred.detach()).var() / var_returns).item()

	return {
		"policy_loss": policy_loss.item(),
		"value_loss": value_loss.item(),
		"entropy": entropy.mean().item(),
		"mean_ratio": mean_ratio,
		"clip_fraction": clip_fraction,
		"approx_kl": approx_kl,
		"explained_variance": explained_var,
		"entropy_action_type": at_entropy_mean,
		"entropy_play_start": ps_entropy_mean,
		"entropy_play_end": pe_entropy_mean,
		"entropy_scout_insert": si_entropy_mean,
		"entropy_floor_penalty": floor_penalty_val,
	}

def direct_pg_update(network: ScoutNetwork, optimizer: torch.optim.Optimizer,
					 batch: dict, entropy_bonus: float = 0.01,
					 value_loss_coeff: float = 0.25, max_grad_norm: float = 0.5,
					 entropy_floors: dict[str, float] | None = None,
					 entropy_floor_coeff: float = 1.0,
					 play_start_size: int = PLAY_START_SIZE):
	"""Vanilla policy gradient update: loss = -log_prob(action) * advantage.
	No importance sampling, no clipping. Should be called for 1 epoch only.
	Takes the same batch format as ppo_update."""
	empty_metrics = {
		"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
		"clip_fraction": 0.0, "approx_kl": 0.0, "explained_variance": 0.0,
		"entropy_action_type": 0.0, "entropy_play_start": 0.0,
		"entropy_play_end": 0.0, "entropy_scout_insert": 0.0,
		"entropy_floor_penalty": 0.0,
	}
	if batch is None:
		return empty_metrics

	n = batch["n"]

	# Batched forward pass through shared layers
	hidden_all = network(batch["states"])  # [n, hidden_size]

	# Value loss (all steps)
	v_pred = network.value(hidden_all).squeeze(-1)  # [n]
	v_target = batch["v_target"]
	value_loss = F.mse_loss(v_pred, v_target)

	# Action type log probs
	at_logits = network.action_type_logits(hidden_all)  # [n, AT_SIZE]
	at_masks = batch["at_masks"]
	at_actions = batch["at_actions"]

	log_prob = _batched_masked_log_prob(at_logits, at_masks, at_actions)
	at_ent = _batched_masked_entropy(at_logits, at_masks)
	entropy = at_ent.clone()

	def _filtered_ent_mean(ent, masks):
		has_choice = masks.sum(dim=-1) >= 2
		if has_choice.any():
			return ent[has_choice].mean().item()
		return 0.0

	at_entropy_mean = _filtered_ent_mean(at_ent, at_masks)
	ps_entropy_mean = 0.0
	pe_entropy_mean = 0.0
	si_entropy_mean = 0.0
	_head_ent = {"action_type": (at_ent, at_masks)}
	_head_ent["play_start"] = None
	_head_ent["play_end"] = None
	_head_ent["scout_insert"] = None

	# Play start
	if "play_idx" in batch:
		idx_t = batch["play_idx"]
		cond = _build_batch_conditioning(hidden_all[idx_t], batch["play_at"], None, play_start_size=play_start_size)
		logits = network.play_start_head(cond)
		masks = batch["play_masks"]
		actions = batch["play_actions"]
		lp = _batched_masked_log_prob(logits, masks, actions)
		ps_ent = _batched_masked_entropy(logits, masks)
		log_prob = log_prob + torch.zeros(n).scatter(0, idx_t, lp)
		entropy = entropy + torch.zeros(n).scatter(0, idx_t, ps_ent)
		ps_entropy_mean = _filtered_ent_mean(ps_ent, masks)
		_head_ent["play_start"] = (ps_ent, masks)

	# Play end
	if "end_idx" in batch:
		idx_t = batch["end_idx"]
		cond = _build_batch_conditioning(hidden_all[idx_t], batch["end_at"], batch["end_starts"], play_start_size=play_start_size)
		logits = network.play_end_head(cond)
		masks = batch["end_masks"]
		actions = batch["end_actions"]
		lp = _batched_masked_log_prob(logits, masks, actions)
		pe_ent = _batched_masked_entropy(logits, masks)
		log_prob = log_prob + torch.zeros(n).scatter(0, idx_t, lp)
		entropy = entropy + torch.zeros(n).scatter(0, idx_t, pe_ent)
		pe_entropy_mean = _filtered_ent_mean(pe_ent, masks)
		_head_ent["play_end"] = (pe_ent, masks)

	# Scout insert
	if "scout_idx" in batch:
		idx_t = batch["scout_idx"]
		cond = _build_batch_conditioning(hidden_all[idx_t], batch["scout_at"], None, play_start_size=play_start_size)
		logits = network.scout_insert_head(cond)
		masks = batch["scout_masks"]
		actions = batch["scout_actions"]
		lp = _batched_masked_log_prob(logits, masks, actions)
		si_ent = _batched_masked_entropy(logits, masks)
		log_prob = log_prob + torch.zeros(n).scatter(0, idx_t, lp)
		entropy = entropy + torch.zeros(n).scatter(0, idx_t, si_ent)
		si_entropy_mean = _filtered_ent_mean(si_ent, masks)
		_head_ent["scout_insert"] = (si_ent, masks)

	# Vanilla policy gradient: -log_prob * advantage
	adv = batch["adv"]
	policy_loss = -(log_prob * adv).mean()

	loss = policy_loss + value_loss_coeff * value_loss - entropy_bonus * entropy.mean()

	# Entropy floor penalty (same as PPO)
	floor_penalty_val = 0.0
	if entropy_floors:
		floor_penalty = torch.tensor(0.0)
		for key, pair in _head_ent.items():
			floor = entropy_floors.get(key, 0.0)
			if floor <= 0 or pair is None:
				continue
			ent_tensor, mask_tensor = pair
			has_choice = mask_tensor.sum(dim=-1) >= 2
			if not has_choice.any():
				continue
			mean_ent = ent_tensor[has_choice].mean()
			violation = torch.clamp(floor - mean_ent, min=0.0)
			floor_penalty = floor_penalty + violation ** 2
		loss = loss + entropy_floor_coeff * floor_penalty
		floor_penalty_val = floor_penalty.item()

	if torch.isnan(loss):
		print(f"  WARNING: NaN loss detected (policy={policy_loss.item()}, value={value_loss.item()}, entropy={entropy.mean().item()})")
		return empty_metrics

	optimizer.zero_grad()
	loss.backward()
	torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=max_grad_norm)
	optimizer.step()

	# Diagnostics
	with torch.no_grad():
		var_returns = v_target.var()
		if var_returns < 1e-8:
			explained_var = 0.0
		else:
			explained_var = (1 - (v_target - v_pred.detach()).var() / var_returns).item()

	return {
		"policy_loss": policy_loss.item(),
		"value_loss": value_loss.item(),
		"entropy": entropy.mean().item(),
		"mean_ratio": 1.0,  # no importance sampling
		"clip_fraction": 0.0,  # no clipping
		"approx_kl": 0.0,  # no old policy comparison
		"explained_variance": explained_var,
		"entropy_action_type": at_entropy_mean,
		"entropy_play_start": ps_entropy_mean,
		"entropy_play_end": pe_entropy_mean,
		"entropy_scout_insert": si_entropy_mean,
		"entropy_floor_penalty": floor_penalty_val,
	}

# --- V6 training functions ---

def _play_turn_v6(game: Game, networks: list, game_log=None) -> list[StepRecordV6]:
	"""Execute one turn using v6 flat action space.
	Returns 1 record for play/scout, 2 for S&S (recursive forced play)."""
	p = game.current_player
	net = networks[p]
	H = HAND_SLOTS_V6
	hand_offset = random.randint(0, H - 1)
	round_num = game.round_number
	records = []
	with torch.no_grad():
		hand = game.players[p].hand
		legal_plays = get_legal_plays(hand, game.current_play)
		forced_play = game.phase == Phase.SNS_PLAY
		state = encode_state_v6(game, p, hand_offset, forced_play=forced_play)
		hidden = net(state)
		value = net.value(hidden).item()
		logits = net.policy_logits(hidden)
		mask_t = get_flat_action_mask(game, p, legal_plays, hand_offset)
		if not mask_t.any():
			game._advance_turn()
			return records
		action_idx, _ = masked_sample(logits, mask_t)
		old_lp = masked_log_prob(logits, mask_t, action_idx).item()
		action = decode_flat_action(action_idx, hand_offset)
		rec = StepRecordV6(
			state=state, action=action_idx, mask=mask_t.numpy(),
			old_log_prob=old_lp, value=value, reward=0.0,
			player=p, round_num=round_num, game_id=0,
			hand_offset=hand_offset, play_length=None, scout_quality=None,
		)
		if action['type'] == 'play':
			rec.play_length = action['end'] - action['start'] + 1
			records.append(rec)
			played_cards = hand[action['start']:action['end'] + 1]
			game.apply_play(action['start'], action['end'])
			if game_log:
				game_log.record_play(game, p, played_cards, round_num=round_num)
		elif action['type'] == 'scout':
			left_end, flip = action['left_end'], action['flip']
			insert_pos = action['insert_pos']
			play_cards = game.current_play.cards
			scouted = play_cards[0] if left_end else play_cards[-1]
			if flip:
				scouted = (scouted[1], scouted[0])
			new_hand = list(hand[:insert_pos]) + [scouted] + list(hand[insert_pos:])
			max_len = 1
			for s, e in get_legal_plays(new_hand, None):
				if s <= insert_pos <= e:
					max_len = max(max_len, e - s + 1)
			rec.scout_quality = max_len
			records.append(rec)
			game.apply_scout(left_end, flip, insert_pos)
			if game_log:
				game_log.record_scout(game, p, scouted, left_end, insert_pos, round_num=round_num)
		elif action['type'] == 'sns':
			left_end, flip = action['left_end'], action['flip']
			insert_pos = action['insert_pos']
			play_cards = game.current_play.cards
			scouted = play_cards[0] if left_end else play_cards[-1]
			if flip:
				scouted = (scouted[1], scouted[0])
			new_hand = list(hand[:insert_pos]) + [scouted] + list(hand[insert_pos:])
			max_len = 1
			for s, e in get_legal_plays(new_hand, None):
				if s <= insert_pos <= e:
					max_len = max(max_len, e - s + 1)
			rec.scout_quality = max_len
			records.append(rec)
			game.apply_sns_scout(left_end, flip, insert_pos)
			if game_log:
				game_log.record_scout(game, p, scouted, left_end, insert_pos, round_num=round_num)
			# S&S step 2: forced play (recursive)
			if game.phase == Phase.SNS_PLAY:
				sns_records = _play_turn_v6(game, networks, game_log)
				records.extend(sns_records)
	return records

def rollout_from_states_batched_v6(snapshots: list[Game], network) -> list[list[int]]:
	"""Play game snapshots to round completion with v6 flat actions.
	Returns list of round scores (one per snapshot)."""
	if not snapshots:
		return []
	H = HAND_SLOTS_V6
	games = [copy.deepcopy(s) for s in snapshots]
	with torch.no_grad():
		while True:
			pending = []  # (game_idx, player, hand_offset)
			states = []
			masks = []
			for g_idx, g in enumerate(games):
				if g.phase not in (Phase.TURN, Phase.SNS_PLAY):
					continue
				p = g.current_player
				ho = random.randint(0, H - 1)
				hand = g.players[p].hand
				legal_plays = get_legal_plays(hand, g.current_play)
				forced_play = g.phase == Phase.SNS_PLAY
				state = encode_state_v6(g, p, ho, forced_play=forced_play)
				mask = get_flat_action_mask(g, p, legal_plays, ho)
				pending.append((g_idx, p, ho))
				states.append(state)
				masks.append(mask)
			if not pending:
				break
			state_batch = torch.stack(states)
			mask_batch = torch.stack(masks)
			hidden = network(state_batch)
			logits = network.policy_logits(hidden)
			has_action = mask_batch.any(dim=1)
			for bi in range(len(pending)):
				if not has_action[bi]:
					games[pending[bi][0]]._advance_turn()
			actions = batched_masked_sample(logits, mask_batch)
			for bi in range(len(pending)):
				if not has_action[bi]:
					continue
				g_idx, p, ho = pending[bi]
				action = decode_flat_action(actions[bi].item(), ho)
				g = games[g_idx]
				if action['type'] == 'play':
					g.apply_play(action['start'], action['end'])
				elif action['type'] == 'scout':
					g.apply_scout(action['left_end'], action['flip'], action['insert_pos'])
				elif action['type'] == 'sns':
					g.apply_sns_scout(action['left_end'], action['flip'], action['insert_pos'])
	return [g.get_round_scores() for g in games]

def play_games_with_rollouts_v6(network, num_games: int, num_players: int,
								rollouts_per_state: int = 10,
								training_seats: int = 4) -> tuple[list[StepRecordV6], list[float]]:
	"""Play games with rollout-based advantage estimation using v6 flat actions.
	S&S produces 2 separate snapshot pairs (one per step).
	Returns (records, normalized_advantages)."""
	H = HAND_SLOTS_V6
	all_records: list[StepRecordV6] = []
	all_advantages: list[float] = []
	network.eval()
	with torch.no_grad():
		for game_idx in range(num_games):
			game = Game(num_players)
			networks = [network] * num_players
			game.start_round()
			# Flip decisions
			for p in range(num_players):
				ho = random.randint(0, H - 1)
				t_normal, t_flipped = encode_hand_both_orientations_v6(game, p, ho)
				h_normal = network(t_normal)
				h_flipped = network(t_flipped)
				v_normal = network.value(h_normal).item()
				v_flipped = network.value(h_flipped).item()
				game.submit_flip_decision(p, do_flip=v_flipped > v_normal)
			# Play turns with per-step snapshots
			snapshots = []
			records = []
			record_snap_pairs = []  # (before_idx, after_idx) per record
			reuse_before_snap = None  # set after S&S to avoid duplicate snapshot
			while game.phase in (Phase.TURN, Phase.SNS_PLAY):
				p = game.current_player
				hand = game.players[p].hand
				legal_plays = get_legal_plays(hand, game.current_play)
				hand_offset = random.randint(0, H - 1)
				forced_play = game.phase == Phase.SNS_PLAY
				state = encode_state_v6(game, p, hand_offset, forced_play=forced_play)
				hidden = network(state)
				value = network.value(hidden).item()
				logits = network.policy_logits(hidden)
				mask_t = get_flat_action_mask(game, p, legal_plays, hand_offset)
				if not mask_t.any():
					game._advance_turn()
					reuse_before_snap = None
					continue
				# Snapshot before action (reuse post-insert snapshot for S&S forced play)
				if reuse_before_snap is not None:
					before_snap = reuse_before_snap
					reuse_before_snap = None
				else:
					before_snap = len(snapshots)
					snapshots.append(copy.deepcopy(game))
				action_idx, _ = masked_sample(logits, mask_t)
				old_lp = masked_log_prob(logits, mask_t, action_idx).item()
				action = decode_flat_action(action_idx, hand_offset)
				rec = StepRecordV6(
					state=state, action=action_idx, mask=mask_t.numpy(),
					old_log_prob=old_lp, value=value, reward=0.0,
					player=p, round_num=game.round_number, game_id=game_idx,
					hand_offset=hand_offset, play_length=None, scout_quality=None,
				)
				if action['type'] == 'play':
					rec.play_length = action['end'] - action['start'] + 1
					records.append(rec)
					game.apply_play(action['start'], action['end'])
					after_snap = len(snapshots)
					snapshots.append(copy.deepcopy(game))
					record_snap_pairs.append((before_snap, after_snap))
				elif action['type'] == 'scout':
					left_end, flip = action['left_end'], action['flip']
					insert_pos = action['insert_pos']
					play_cards = game.current_play.cards
					scouted = play_cards[0] if left_end else play_cards[-1]
					if flip:
						scouted = (scouted[1], scouted[0])
					new_hand = list(hand[:insert_pos]) + [scouted] + list(hand[insert_pos:])
					max_len = 1
					for s, e in get_legal_plays(new_hand, None):
						if s <= insert_pos <= e:
							max_len = max(max_len, e - s + 1)
					rec.scout_quality = max_len
					records.append(rec)
					game.apply_scout(left_end, flip, insert_pos)
					after_snap = len(snapshots)
					snapshots.append(copy.deepcopy(game))
					record_snap_pairs.append((before_snap, after_snap))
				elif action['type'] == 'sns':
					left_end, flip = action['left_end'], action['flip']
					insert_pos = action['insert_pos']
					play_cards = game.current_play.cards
					scouted = play_cards[0] if left_end else play_cards[-1]
					if flip:
						scouted = (scouted[1], scouted[0])
					new_hand = list(hand[:insert_pos]) + [scouted] + list(hand[insert_pos:])
					max_len = 1
					for s, e in get_legal_plays(new_hand, None):
						if s <= insert_pos <= e:
							max_len = max(max_len, e - s + 1)
					rec.scout_quality = max_len
					records.append(rec)
					game.apply_sns_scout(left_end, flip, insert_pos)
					# Snapshot after insert (before forced play)
					after_insert_snap = len(snapshots)
					snapshots.append(copy.deepcopy(game))
					record_snap_pairs.append((before_snap, after_insert_snap))
					# Forced play uses this snapshot as its "before"
					reuse_before_snap = after_insert_snap
			# Run rollouts from all snapshots
			expanded = [snap for snap in snapshots for _ in range(rollouts_per_state)]
			all_scores = rollout_from_states_batched_v6(expanded, network)
			# Aggregate to per-snapshot, per-player margins
			num_snapshots = len(snapshots)
			snapshot_values = []
			for snap_idx in range(num_snapshots):
				player_margins = [0.0] * num_players
				base = snap_idx * rollouts_per_state
				for r in range(rollouts_per_state):
					scores = all_scores[base + r]
					for pi in range(num_players):
						opp = [scores[j] for j in range(num_players) if j != pi]
						margin = (scores[pi] - sum(opp) / len(opp)) / 10.0
						player_margins[pi] += margin
				snapshot_values.append([m / rollouts_per_state for m in player_margins])
			# Compute advantages and value targets
			game_advantages = []
			for rec_idx, (before_snap, after_snap) in enumerate(record_snap_pairs):
				pi = records[rec_idx].player
				v_before = snapshot_values[before_snap][pi]
				v_after = snapshot_values[after_snap][pi]
				game_advantages.append(v_after - v_before)
			for rec_idx, (before_snap, _) in enumerate(record_snap_pairs):
				records[rec_idx].value = snapshot_values[before_snap][records[rec_idx].player]
			for rec in records:
				rec.game_id = game_idx
			filtered = [i for i, r in enumerate(records) if r.player < training_seats]
			all_records.extend(records[i] for i in filtered)
			all_advantages.extend(game_advantages[i] for i in filtered)
	# Normalize advantages
	if all_advantages:
		mean = sum(all_advantages) / len(all_advantages)
		std = (sum((a - mean) ** 2 for a in all_advantages) / len(all_advantages)) ** 0.5
		all_advantages = [(a - mean) / (std + 1e-8) for a in all_advantages]
	return all_records, all_advantages

def prepare_ppo_batch_v6(steps: list[StepRecordV6], advantages: list[float],
						 returns: list[float] | None = None) -> dict | None:
	"""Pre-stack v6 StepRecords into a PPO batch. Simple: no sub-head indexing."""
	n = len(steps)
	if n == 0:
		return None
	batch = {
		"n": n,
		"states": torch.stack([s.state for s in steps]),
		"masks": torch.from_numpy(np.stack([s.mask for s in steps])),
		"actions": torch.tensor([s.action for s in steps], dtype=torch.long),
		"old_log_probs": torch.tensor([s.old_log_prob for s in steps], dtype=torch.float32),
		"adv": torch.tensor(advantages, dtype=torch.float32),
	}
	if returns is not None:
		batch["v_target"] = torch.tensor(returns, dtype=torch.float32)
	else:
		batch["v_target"] = torch.tensor([s.value for s in steps], dtype=torch.float32)
	return batch

def subsample_batch_v6(batch: dict, keep_n: int) -> dict:
	"""Randomly subsample a v6 PPO batch."""
	n = batch["n"]
	if keep_n >= n:
		return batch
	keep = torch.randperm(n)[:max(1, keep_n)].sort().values
	return {
		"n": len(keep),
		"states": batch["states"][keep],
		"masks": batch["masks"][keep],
		"actions": batch["actions"][keep],
		"old_log_probs": batch["old_log_probs"][keep],
		"adv": batch["adv"][keep],
		"v_target": batch["v_target"][keep],
	}

def concatenate_batches_v6(batches: list[dict]) -> dict:
	"""Concatenate v6 PPO batches, re-normalizing advantages."""
	if len(batches) == 1:
		return batches[0]
	combined = {
		"states": torch.cat([b["states"] for b in batches]),
		"masks": torch.cat([b["masks"] for b in batches]),
		"actions": torch.cat([b["actions"] for b in batches]),
		"old_log_probs": torch.cat([b["old_log_probs"] for b in batches]),
		"adv": torch.cat([b["adv"] for b in batches]),
		"v_target": torch.cat([b["v_target"] for b in batches]),
	}
	combined["n"] = combined["states"].shape[0]
	adv = combined["adv"]
	combined["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)
	return combined

def ppo_update_v6(network, optimizer: torch.optim.Optimizer,
				  batch: dict, clip_epsilon: float = 0.2, entropy_bonus: float = 0.01,
				  value_loss_coeff: float = 0.25, max_grad_norm: float = 0.5):
	"""PPO update for v6 flat action space. Single forward pass, no sub-head routing."""
	empty_metrics = {
		"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
		"clip_fraction": 0.0, "approx_kl": 0.0, "explained_variance": 0.0,
		"entropy_play": 0.0, "entropy_scout": 0.0,
	}
	if batch is None:
		return empty_metrics
	n = batch["n"]
	hidden = network(batch["states"])
	# Value loss
	v_pred = network.value(hidden).squeeze(-1)
	v_target = batch["v_target"]
	value_loss = F.mse_loss(v_pred, v_target)
	# Policy: flat logits [n, 384]
	logits = network.policy_logits(hidden)
	masks = batch["masks"]
	actions = batch["actions"]
	old_log_probs = batch["old_log_probs"]
	new_log_probs = _batched_masked_log_prob(logits, masks, actions)
	log_ratio = new_log_probs - old_log_probs
	entropy = _batched_masked_entropy(logits, masks)
	# PPO clipped surrogate
	adv = batch["adv"]
	ratio = torch.exp(log_ratio)
	surr1 = ratio * adv
	surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
	policy_loss = -torch.min(surr1, surr2).mean()
	loss = policy_loss + value_loss_coeff * value_loss - entropy_bonus * entropy.mean()
	if torch.isnan(loss):
		print(f"  WARNING: NaN loss (policy={policy_loss.item()}, value={value_loss.item()}, ent={entropy.mean().item()})")
		return empty_metrics
	optimizer.zero_grad()
	loss.backward()
	torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=max_grad_norm)
	optimizer.step()
	# Diagnostics
	with torch.no_grad():
		clip_fraction = (torch.abs(ratio - 1.0) > clip_epsilon).float().mean().item()
		approx_kl = ((ratio - 1) - log_ratio).mean().item()
		var_returns = v_target.var()
		explained_var = 0.0
		if var_returns >= 1e-8:
			explained_var = (1 - (v_target - v_pred.detach()).var() / var_returns).item()
		# Conditional entropies
		play_only = torch.zeros_like(masks)
		play_only[:, :256] = masks[:, :256]
		has_play = play_only.any(dim=1)
		play_ent = _batched_masked_entropy(logits[has_play], play_only[has_play]).mean().item() if has_play.any() else 0.0
		scout_only = torch.zeros_like(masks)
		scout_only[:, 256:320] = masks[:, 256:320]
		has_scout = scout_only.any(dim=1)
		scout_ent = _batched_masked_entropy(logits[has_scout], scout_only[has_scout]).mean().item() if has_scout.any() else 0.0
	return {
		"policy_loss": policy_loss.item(),
		"value_loss": value_loss.item(),
		"entropy": entropy.mean().item(),
		"mean_ratio": ratio.mean().item(),
		"clip_fraction": clip_fraction,
		"approx_kl": approx_kl,
		"explained_variance": explained_var,
		"entropy_play": play_ent,
		"entropy_scout": scout_ent,
	}

def augment_rotation_v6(steps: list[StepRecordV6], advantages: list[float],
						network) -> tuple[list[StepRecordV6], list[float]]:
	"""Create 15 rotation-augmented copies of each training sample.
	Shifts hand portion of state, permutes action index and mask.
	Runs one batched forward pass per shift to compute correct old_log_probs.
	Returns (original + augmented steps, original + augmented advantages)."""
	from encoding import FULL_PERM, HAND_SHIFT, HAND_SLOTS_V6
	H = HAND_SLOTS_V6
	n = len(steps)
	if n == 0:
		return steps, advantages

	all_steps = list(steps)
	all_advs = list(advantages)

	# Stack originals for vectorized permutation
	orig_states = torch.stack([s.state for s in steps])             # [n, 301]
	orig_actions = torch.tensor([s.action for s in steps], dtype=torch.long)  # [n]
	orig_masks = torch.from_numpy(np.stack([s.mask for s in steps]))  # [n, 384]

	network.eval()
	with torch.no_grad():
		for k in range(1, H):
			shift = HAND_SHIFT[k]           # [301] gather index
			perm = FULL_PERM[k]             # [384] forward: orig → aug
			inv_perm = FULL_PERM[(H - k) % H]  # [384] inverse

			aug_states = orig_states[:, shift]       # [n, 301]
			aug_actions = perm[orig_actions]          # [n]
			aug_masks = orig_masks[:, inv_perm]       # [n, 384]

			# Forward pass to compute correct old_log_probs at shifted states
			hidden = network(aug_states)
			logits = network.policy_logits(hidden)
			masked_logits = logits.masked_fill(~aug_masks, float('-inf'))
			log_probs = torch.log_softmax(masked_logits, dim=-1)
			old_lps = log_probs.gather(1, aug_actions.unsqueeze(1)).squeeze(1)

			aug_masks_np = aug_masks.numpy()
			for i, (step, adv) in enumerate(zip(steps, advantages)):
				all_steps.append(StepRecordV6(
					state=aug_states[i],
					action=aug_actions[i].item(),
					mask=aug_masks_np[i].copy(),
					old_log_prob=old_lps[i].item(),
					value=step.value,
					reward=step.reward,
					player=step.player,
					round_num=step.round_num,
					game_id=step.game_id,
					hand_offset=(step.hand_offset + k) % H,
					play_length=step.play_length,
					scout_quality=step.scout_quality,
				))
				all_advs.append(adv)

	return all_steps, all_advs
