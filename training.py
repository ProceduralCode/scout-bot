import torch
import torch.nn.functional as F
import random
import copy
from dataclasses import dataclass
from game import Game, Phase
from encoding import (
	encode_state, encode_hand_both_orientations, get_action_type_mask,
	get_play_start_mask, get_play_end_mask, get_scout_insert_mask,
	get_legal_plays, decode_action_type, decode_slot_to_hand_index,
	HAND_SLOTS, PLAY_SLOTS,
)
from network import ScoutNetwork, masked_sample, masked_log_prob

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
	# Filled in after the game
	value: float = 0.0
	reward: float = 0.0
	player: int = 0

def play_game(network: ScoutNetwork, num_players: int,
			  opponent_pool: list[ScoutNetwork] | None = None) -> list[StepRecord]:
	"""Play a complete game using the network, recording all decisions.
	If opponent_pool is provided, randomly assign opponents from the pool.
	Only returns records from players using the training network — opponent
	records are discarded since their old logits would corrupt PPO ratios."""
	game = Game(num_players)
	# Assign networks: player 0 = main network, others from pool or same
	networks = [network]
	for i in range(1, num_players):
		if opponent_pool:
			networks.append(random.choice(opponent_pool))
		else:
			networks.append(network)
	all_records: list[StepRecord] = []
	for _ in range(game.total_rounds):
		game.start_round()
		round_records = _play_round(game, networks)
		all_records.extend(round_records)
	rewards = game.get_rewards()
	# Filter to only records from the training network
	records = [r for r in all_records if networks[r.player] is network]
	for rec in records:
		rec.reward = rewards[rec.player]
	return records

def _play_round(game: Game, networks: list[ScoutNetwork]) -> list[StepRecord]:
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
		game.submit_flip_decision(p, do_flip=(v_flipped > v_normal))
	# Play turns
	while game.phase == Phase.TURN:
		step_records = _play_turn(game, networks)
		records.extend(step_records)
	return records

def _play_turn(game: Game, networks: list[ScoutNetwork]) -> list[StepRecord]:
	"""Execute one turn, returning step records."""
	p = game.current_player
	net = networks[p]
	hand_offset = random.randint(0, HAND_SLOTS - 1)
	play_offset = random.randint(0, PLAY_SLOTS - 1)
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
			game.apply_play(start_idx, end_idx)
		elif action_info["type"] == "scout":
			# Step 2: insert position
			si_logits = net.scout_insert_logits(hidden, action_type)
			si_mask = get_scout_insert_mask(game)
			insert_pos, _ = masked_sample(si_logits, si_mask)
			rec.scout_insert = insert_pos
			rec.scout_insert_logits = si_logits
			rec.scout_insert_mask = si_mask
			records.append(rec)
			game.apply_scout(action_info["left_end"], action_info["flip"], insert_pos)
		elif action_info["type"] == "sns":
			# Scout portion
			si_logits = net.scout_insert_logits(hidden, action_type)
			si_mask = get_scout_insert_mask(game)
			insert_pos, _ = masked_sample(si_logits, si_mask)
			rec.scout_insert = insert_pos
			rec.scout_insert_logits = si_logits
			rec.scout_insert_mask = si_mask
			records.append(rec)
			game.apply_sns_scout(action_info["left_end"], action_info["flip"], insert_pos)
			# Play portion (forced, separate turn record)
			if game.phase == Phase.SNS_PLAY:
				sns_records = _play_turn(game, networks)
				records.extend(sns_records)
	return records

class ReplayBuffer:
	"""Rolling buffer of game step records for experience replay."""
	def __init__(self, max_games: int = 1000):
		self.max_games = max_games
		self.games: list[list[StepRecord]] = []
	def add_game(self, records: list[StepRecord]):
		self.games.append(records)
		if len(self.games) > self.max_games:
			self.games.pop(0)
	def sample_steps(self, batch_size: int) -> list[StepRecord]:
		"""Sample individual steps uniformly across all stored games."""
		all_steps = [step for game in self.games for step in game]
		if len(all_steps) <= batch_size:
			return all_steps
		return random.sample(all_steps, batch_size)
	def total_steps(self) -> int:
		return sum(len(g) for g in self.games)

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

def compute_advantages(records: list[StepRecord], gamma: float = 0.99) -> list[float]:
	"""Compute advantage = reward - value for each step.
	Since reward is a single game-end signal (not per-step), all steps for a
	player share the same reward. Advantage is simply reward - value estimate."""
	# TODO: Consider GAE or per-round discounting as a later refinement
	return [rec.reward - rec.value for rec in records]

def ppo_update(network: ScoutNetwork, optimizer: torch.optim.Optimizer,
			   steps: list[StepRecord], advantages: list[float],
			   clip_epsilon: float = 0.2, entropy_bonus: float = 0.01,
			   value_loss_coeff: float = 0.5, max_grad_norm: float = 0.5):
	"""One PPO update step on a batch of experience.
	Uses per-step gradient accumulation to avoid building one giant graph."""
	n = len(steps)
	if n == 0:
		return 0.0, 0.0, 0.0
	optimizer.zero_grad()
	total_policy = 0.0
	total_value = 0.0
	total_entropy = 0.0
	for step, advantage in zip(steps, advantages):
		adv = torch.tensor(advantage, dtype=torch.float32)
		hidden = network(step.state)
		# Value loss
		v_pred = network.value(hidden).squeeze()
		v_target = torch.tensor(step.reward, dtype=torch.float32)
		value_loss = F.mse_loss(v_pred, v_target)
		# Policy loss: compute current log probs and ratio against old
		at_logits = network.action_type_logits(hidden)
		new_at_log_prob = masked_log_prob(at_logits, step.action_type_mask, step.action_type)
		old_at_log_prob = masked_log_prob(step.action_type_logits, step.action_type_mask, step.action_type)
		log_ratio = new_at_log_prob - old_at_log_prob
		if step.play_start is not None:
			ps_logits = network.play_start_logits(hidden, step.action_type)
			new_ps_log_prob = masked_log_prob(ps_logits, step.play_start_mask, step.play_start)
			old_ps_log_prob = masked_log_prob(step.play_start_logits, step.play_start_mask, step.play_start)
			log_ratio = log_ratio + (new_ps_log_prob - old_ps_log_prob)
		if step.play_end is not None:
			pe_logits = network.play_end_logits(hidden, step.action_type, step.play_start)
			new_pe_log_prob = masked_log_prob(pe_logits, step.play_end_mask, step.play_end)
			old_pe_log_prob = masked_log_prob(step.play_end_logits, step.play_end_mask, step.play_end)
			log_ratio = log_ratio + (new_pe_log_prob - old_pe_log_prob)
		if step.scout_insert is not None:
			si_logits = network.scout_insert_logits(hidden, step.action_type)
			new_si_log_prob = masked_log_prob(si_logits, step.scout_insert_mask, step.scout_insert)
			old_si_log_prob = masked_log_prob(step.scout_insert_logits, step.scout_insert_mask, step.scout_insert)
			log_ratio = log_ratio + (new_si_log_prob - old_si_log_prob)
		ratio = torch.exp(log_ratio)
		surr1 = ratio * adv
		surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
		policy_loss = -torch.min(surr1, surr2)
		# Entropy bonus (from action type head only, for simplicity)
		at_probs = torch.softmax(at_logits.clone().masked_fill(~step.action_type_mask, float('-inf')), dim=-1)
		entropy = -(at_probs * torch.log(at_probs + 1e-8)).sum()
		# Per-step backward with scaling for correct averaging
		step_loss = (policy_loss + value_loss_coeff * value_loss - entropy_bonus * entropy) / n
		step_loss.backward()
		total_policy += policy_loss.item()
		total_value += value_loss.item()
		total_entropy += entropy.item()
	torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=max_grad_norm)
	optimizer.step()
	return total_policy / n, total_value / n, total_entropy / n
