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
	HAND_SLOTS, PLAY_SLOTS, ACTION_TYPE_SIZE, PLAY_START_SIZE,
)
from network import ScoutNetwork, masked_sample, masked_log_prob
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

def play_game(network: ScoutNetwork, num_players: int,
			  opponent_pool: list[ScoutNetwork] | None = None,
			  game_log: GameLog | None = None,
			  training_seats: int = 1) -> list[StepRecord]:
	"""Play a complete game using the network, recording all decisions.
	First training_seats players use the training network, rest from opponent_pool.
	Only returns records from players using the training network — opponent
	records are discarded since their old logits would corrupt PPO ratios.
	If game_log is provided, human-readable events are recorded into it."""
	game = Game(num_players)
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
		# Only last step per player per round gets the reward (for GAE)
		# Reward is margin vs mean opponent score, not absolute score
		round_scores = game.get_round_scores()
		last_by_player: dict[int, int] = {}
		for i, rec in enumerate(round_records):
			rec.round_num = round_idx
			rec.reward = 0.0
			last_by_player[rec.player] = i
		for i in last_by_player.values():
			p = round_records[i].player
			opponent_scores = [round_scores[j] for j in range(len(round_scores)) if j != p]
			mean_opponent = sum(opponent_scores) / len(opponent_scores)
			round_records[i].reward = (round_scores[p] - mean_opponent) / 10.0
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

def _play_round(game: Game, networks: list[ScoutNetwork],
				game_log: GameLog | None = None) -> list[StepRecord]:
	"""Play one round, returning step records for all players."""
	records = []
	# Flip decisions
	for p in range(game.num_players):
		net = networks[p]
		with torch.no_grad():
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
	hand_offset = random.randint(0, HAND_SLOTS - 1)
	play_offset = random.randint(0, PLAY_SLOTS - 1)
	round_num = game.round_number  # capture before apply_* may increment it
	records = []
	with torch.no_grad():
		state_tensor = encode_state(game, p, hand_offset, play_offset)
		hidden = net(state_tensor)
		value = net.value(hidden).item()
		# Compute legal plays once for all mask functions
		hand = game.players[p].hand
		legal_plays = get_legal_plays(hand, game.current_play)
		# Step 1: action type
		at_logits = net.action_type_logits(hidden)
		at_mask = get_action_type_mask(game, legal_plays)
		# Edge case: hand full (20 cards) and no legal plays — skip turn
		if not at_mask.any():
			game._advance_turn()
			return records
		action_type, at_log_prob = masked_sample(at_logits, at_mask)
		rec = StepRecord(
			state=state_tensor,
			action_type=action_type,
			action_type_logits=at_logits,
			action_type_mask=at_mask,
			value=value,
			player=p,
		)
		action_info = decode_action_type(action_type)
		if action_info["type"] == "play":
			# Step 2: play start
			ps_logits = net.play_start_logits(hidden, action_type)
			ps_mask = get_play_start_mask(legal_plays, hand_offset)
			start_slot, _ = masked_sample(ps_logits, ps_mask)
			start_idx = decode_slot_to_hand_index(start_slot, hand_offset)
			rec.play_start = start_slot
			rec.play_start_logits = ps_logits
			rec.play_start_mask = ps_mask
			# Step 3: play end
			pe_logits = net.play_end_logits(hidden, action_type, start_slot)
			pe_mask = get_play_end_mask(legal_plays, start_idx, hand_offset)
			end_slot, _ = masked_sample(pe_logits, pe_mask)
			end_idx = decode_slot_to_hand_index(end_slot, hand_offset)
			rec.play_end = end_slot
			rec.play_end_logits = pe_logits
			rec.play_end_mask = pe_mask
			records.append(rec)
			played_cards = hand[start_idx:end_idx + 1]
			game.apply_play(start_idx, end_idx)
			if game_log:
				game_log.record_play(game, p, played_cards, round_num=round_num)
		elif action_info["type"] == "scout":
			# Step 2: insert position
			si_logits = net.scout_insert_logits(hidden, action_type)
			si_mask = get_scout_insert_mask(game)
			insert_pos, _ = masked_sample(si_logits, si_mask)
			rec.scout_insert = insert_pos
			rec.scout_insert_logits = si_logits
			rec.scout_insert_mask = si_mask
			records.append(rec)
			# Capture scouted card before applying
			left_end = action_info["left_end"]
			play_cards = game.current_play.cards
			scouted = play_cards[0] if left_end else play_cards[-1]
			if action_info["flip"]:
				scouted = (scouted[1], scouted[0])
			game.apply_scout(left_end, action_info["flip"], insert_pos)
			if game_log:
				game_log.record_scout(game, p, scouted, left_end, insert_pos, round_num=round_num)
		elif action_info["type"] == "sns":
			# Scout portion — use restricted mask so insert guarantees a legal play
			si_logits = net.scout_insert_logits(hidden, action_type)
			si_mask = get_sns_insert_mask(game, action_info["left_end"], action_info["flip"])
			insert_pos, _ = masked_sample(si_logits, si_mask)
			rec.scout_insert = insert_pos
			rec.scout_insert_logits = si_logits
			rec.scout_insert_mask = si_mask
			records.append(rec)
			# Capture scouted card before applying
			left_end = action_info["left_end"]
			play_cards = game.current_play.cards
			scouted = play_cards[0] if left_end else play_cards[-1]
			if action_info["flip"]:
				scouted = (scouted[1], scouted[0])
			game.apply_sns_scout(left_end, action_info["flip"], insert_pos)
			if game_log:
				game_log.record_scout(game, p, scouted, left_end, insert_pos, round_num=round_num)
			# Play portion (forced, separate turn record)
			# The recursive call logs the play event itself
			if game.phase == Phase.SNS_PLAY:
				sns_records = _play_turn(game, networks, game_log)
				records.extend(sns_records)
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
		return [{"layer_sizes": v.layer_sizes, "state_dict": v.state_dict()}
				for v in self.versions]
	def load_state_dicts(self, states: list[dict], template: ScoutNetwork):
		"""Restore pool from saved state dicts. Handles per-member architecture
		(new format) and bare state dicts (old format, uses template)."""
		self.versions = []
		for entry in states:
			if isinstance(entry, dict) and "layer_sizes" in entry:
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
							  starts: torch.Tensor | None) -> torch.Tensor:
	"""Build batched conditioning vectors for sub-heads.
	hidden: [B, H], action_types/starts: [B] LongTensor or None.
	Returns [B, H + ACTION_TYPE_SIZE + PLAY_START_SIZE]."""
	B = hidden.shape[0]
	device = hidden.device
	if action_types is not None:
		at_oh = F.one_hot(action_types.long(), ACTION_TYPE_SIZE).float().to(device)
	else:
		at_oh = torch.zeros(B, ACTION_TYPE_SIZE, device=device)
	if starts is not None:
		st_oh = F.one_hot(starts.long(), PLAY_START_SIZE).float().to(device)
	else:
		st_oh = torch.zeros(B, PLAY_START_SIZE, device=device)
	return torch.cat([hidden, at_oh, st_oh], dim=1)

def ppo_update(network: ScoutNetwork, optimizer: torch.optim.Optimizer,
			   steps: list[StepRecord], advantages: list[float],
			   clip_epsilon: float = 0.2, entropy_bonus: float = 0.01,
			   value_loss_coeff: float = 0.25, max_grad_norm: float = 0.5,
			   returns: list[float] | None = None):
	"""One PPO update step. Batched forward pass for efficiency."""
	n = len(steps)
	empty_metrics = {
		"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
		"clip_fraction": 0.0, "approx_kl": 0.0, "explained_variance": 0.0,
		"entropy_action_type": 0.0, "entropy_play_start": 0.0,
		"entropy_play_end": 0.0, "entropy_scout_insert": 0.0,
	}
	if n == 0:
		return empty_metrics

	# Batched forward pass through shared layers
	states = torch.stack([s.state for s in steps])
	hidden_all = network(states)  # [n, hidden_size]

	# Value loss (all steps)
	v_pred = network.value(hidden_all).squeeze(-1)  # [n]
	if returns is not None:
		v_target = torch.tensor(returns, dtype=torch.float32)
	else:
		v_target = torch.tensor([s.reward for s in steps], dtype=torch.float32)
	value_loss = F.mse_loss(v_pred, v_target)

	# Action type (all steps, no conditioning)
	at_logits = network.action_type_logits(hidden_all)  # [n, AT_SIZE]
	at_masks = torch.stack([s.action_type_mask for s in steps])
	at_actions = torch.tensor([s.action_type for s in steps], dtype=torch.long)
	old_at_logits = torch.stack([s.action_type_logits for s in steps])

	log_ratio = (_batched_masked_log_prob(at_logits, at_masks, at_actions)
				 - _batched_masked_log_prob(old_at_logits, at_masks, at_actions))
	at_ent = _batched_masked_entropy(at_logits, at_masks)
	entropy = at_ent.clone()
	at_entropy_mean = at_ent.mean().item()
	ps_entropy_mean = 0.0
	pe_entropy_mean = 0.0
	si_entropy_mean = 0.0

	# Sub-head indices
	play_idx = [i for i, s in enumerate(steps) if s.play_start is not None]
	end_idx = [i for i, s in enumerate(steps) if s.play_end is not None]
	scout_idx = [i for i, s in enumerate(steps) if s.scout_insert is not None]

	# Play start — build conditioning manually, call linear head directly
	if play_idx:
		idx_t = torch.tensor(play_idx, dtype=torch.long)
		cond = _build_batch_conditioning(
			hidden_all[idx_t],
			torch.tensor([steps[i].action_type for i in play_idx], dtype=torch.long),
			None)
		logits = network.play_start_head(cond)
		masks = torch.stack([steps[i].play_start_mask for i in play_idx])
		actions = torch.tensor([steps[i].play_start for i in play_idx], dtype=torch.long)
		old_logits = torch.stack([steps[i].play_start_logits for i in play_idx])
		delta = (_batched_masked_log_prob(logits, masks, actions)
				 - _batched_masked_log_prob(old_logits, masks, actions))
		# Accumulate via scatter to avoid in-place ops on graph tensors
		ps_ent = _batched_masked_entropy(logits, masks)
		log_ratio = log_ratio + torch.zeros(n).scatter(0, idx_t, delta)
		entropy = entropy + torch.zeros(n).scatter(0, idx_t, ps_ent)
		ps_entropy_mean = ps_ent.mean().item()

	# Play end
	if end_idx:
		idx_t = torch.tensor(end_idx, dtype=torch.long)
		cond = _build_batch_conditioning(
			hidden_all[idx_t],
			torch.tensor([steps[i].action_type for i in end_idx], dtype=torch.long),
			torch.tensor([steps[i].play_start for i in end_idx], dtype=torch.long))
		logits = network.play_end_head(cond)
		masks = torch.stack([steps[i].play_end_mask for i in end_idx])
		actions = torch.tensor([steps[i].play_end for i in end_idx], dtype=torch.long)
		old_logits = torch.stack([steps[i].play_end_logits for i in end_idx])
		delta = (_batched_masked_log_prob(logits, masks, actions)
				 - _batched_masked_log_prob(old_logits, masks, actions))
		pe_ent = _batched_masked_entropy(logits, masks)
		log_ratio = log_ratio + torch.zeros(n).scatter(0, idx_t, delta)
		entropy = entropy + torch.zeros(n).scatter(0, idx_t, pe_ent)
		pe_entropy_mean = pe_ent.mean().item()

	# Scout insert
	if scout_idx:
		idx_t = torch.tensor(scout_idx, dtype=torch.long)
		cond = _build_batch_conditioning(
			hidden_all[idx_t],
			torch.tensor([steps[i].action_type for i in scout_idx], dtype=torch.long),
			None)
		logits = network.scout_insert_head(cond)
		masks = torch.stack([steps[i].scout_insert_mask for i in scout_idx])
		actions = torch.tensor([steps[i].scout_insert for i in scout_idx], dtype=torch.long)
		old_logits = torch.stack([steps[i].scout_insert_logits for i in scout_idx])
		delta = (_batched_masked_log_prob(logits, masks, actions)
				 - _batched_masked_log_prob(old_logits, masks, actions))
		si_ent = _batched_masked_entropy(logits, masks)
		log_ratio = log_ratio + torch.zeros(n).scatter(0, idx_t, delta)
		entropy = entropy + torch.zeros(n).scatter(0, idx_t, si_ent)
		si_entropy_mean = si_ent.mean().item()

	# PPO clipped objective
	adv = torch.tensor(advantages, dtype=torch.float32)
	ratio = torch.exp(log_ratio)
	surr1 = ratio * adv
	surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
	policy_loss = -torch.min(surr1, surr2).mean()

	loss = policy_loss + value_loss_coeff * value_loss - entropy_bonus * entropy.mean()

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
	}
