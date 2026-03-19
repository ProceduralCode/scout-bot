from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
import random

# (showing_value, hidden_value)
Card = tuple[int, int]

class PlayType(Enum):
	SINGLE = auto()
	SET = auto()
	RUN = auto()

class Phase(Enum):
	FLIP_DECISION = auto()
	PLAY = auto()
	SNS_PLAY = auto() # must play after S&S scout portion
	ROUND_OVER = auto()
	GAME_OVER = auto()

@dataclass
class Play:
	cards: list[Card]
	play_type: PlayType
	strength: int # value for sets/singles, max value for runs

@dataclass
class PlayerState:
	hand: list[Card]
	collected: list[Card] = field(default_factory=list)
	scout_tokens: int = 0
	sns_available: bool = True

def classify_play(cards: list[Card]) -> tuple[PlayType, int] | None:
	"""Classify cards as a play type. Returns (type, strength) or None if invalid."""
	values = [c[0] for c in cards]
	if len(values) == 1:
		return (PlayType.SINGLE, values[0])
	if all(v == values[0] for v in values):
		return (PlayType.SET, values[0])
	ascending = all(values[i + 1] == values[i] + 1 for i in range(len(values) - 1))
	descending = all(values[i + 1] == values[i] - 1 for i in range(len(values) - 1))
	if ascending or descending:
		return (PlayType.RUN, max(values))
	return None

def play_beats(new_type: PlayType, new_strength: int, new_count: int,
			   cur_type: PlayType, cur_strength: int, cur_count: int) -> bool:
	"""Check if a new play beats the current play on the table."""
	if new_count > cur_count:
		return True
	if new_count < cur_count:
		return False
	if new_count == 1:
		return new_strength > cur_strength
	# 2+ cards, same count: set beats run
	if new_type != cur_type:
		return new_type == PlayType.SET
	# Same type: higher strength wins, equal can't be played
	return new_strength > cur_strength

def flip_hand(hand: list[Card]) -> list[Card]:
	"""Flip entire hand: reverse order, swap each card's showing/hidden values."""
	return [(b, a) for a, b in reversed(hand)]

def create_deck(num_players: int) -> list[Card]:
	"""Create the deck for the given player count."""
	cards = []
	for i in range(1, 11):
		for j in range(i + 1, 11):
			cards.append((i, j))
	if num_players == 3:
		# Remove all cards with 10 on either side (9 cards -> 36 remain)
		cards = [(a, b) for a, b in cards if a != 10 and b != 10]
	elif num_players == 4:
		# Remove only the 9/10 card (1 card -> 44 remain)
		cards = [(a, b) for a, b in cards if not (a == 9 and b == 10)]
	return cards

class Game:
	def __init__(self, num_players: int):
		assert 3 <= num_players <= 5
		self.num_players = num_players
		self.total_rounds = num_players
		self.round_number = 0
		self.cumulative_scores = [0] * num_players
		self.starting_player = 0
		self.players: list[PlayerState] = []
		self.current_play: Play | None = None
		self.current_play_owner: int | None = None
		self.current_player = 0
		self.scouts_since_play = 0
		self.phase = Phase.GAME_OVER
		self.round_ender: int | None = None
		self.flips_remaining: set[int] = set()

	def start_round(self):
		"""Deal cards and enter flip decision phase."""
		deck = create_deck(self.num_players)
		random.shuffle(deck)
		# Randomize each card's initial orientation
		for i in range(len(deck)):
			if random.random() < 0.5:
				a, b = deck[i]
				deck[i] = (b, a)
		cards_per_player = len(deck) // self.num_players
		self.players = []
		for p in range(self.num_players):
			start = p * cards_per_player
			self.players.append(PlayerState(hand=deck[start:start + cards_per_player]))
		self.current_play = None
		self.current_play_owner = None
		self.scouts_since_play = 0
		self.current_player = self.starting_player
		self.round_ender = None
		self.phase = Phase.FLIP_DECISION
		self.flips_remaining = set(range(self.num_players))

	def submit_flip_decision(self, player: int, do_flip: bool):
		assert self.phase == Phase.FLIP_DECISION
		assert player in self.flips_remaining
		if do_flip:
			self.players[player].hand = flip_hand(self.players[player].hand)
		self.flips_remaining.discard(player)
		if not self.flips_remaining:
			self.phase = Phase.PLAY

	def get_legal_action_types(self) -> list[int]:
		"""Return legal action type indices (0-8) for the current player.
		0=Play, 1-4=Scout variants, 5-8=S&S variants.
		Scout/S&S indices encode end (left/right) and orientation (normal/flipped):
		  1/5=left normal, 2/6=left flipped, 3/7=right normal, 4/8=right flipped."""
		assert self.phase in (Phase.PLAY, Phase.SNS_PLAY)
		player = self.players[self.current_player]
		has_play = self.current_play is not None
		if self.phase == Phase.SNS_PLAY:
			# Must play after S&S scout
			if self._has_any_legal_play():
				return [0]
			return [] # shouldn't happen in practice
		legal = []
		if not has_play or self._has_any_legal_play():
			legal.append(0) # Play
		if has_play:
			# Scout: always legal if there's a play to scout from
			legal.extend([1, 2]) # left normal, left flipped
			if len(self.current_play.cards) > 1:
				# Right end is distinct from left only with 2+ cards
				legal.extend([3, 4])
			# S&S: also need the token
			if player.sns_available:
				legal.extend([5, 6])
				if len(self.current_play.cards) > 1:
					legal.extend([7, 8])
		return legal

	def get_legal_plays(self) -> list[tuple[int, int]]:
		"""Return (start, end) index pairs for all legal plays by the current player."""
		hand = self.players[self.current_player].hand
		legal = []
		for start in range(len(hand)):
			for end in range(start, len(hand)):
				cards = hand[start:end + 1]
				result = classify_play(cards)
				if result is None:
					continue
				play_type, strength = result
				if self.current_play is None:
					legal.append((start, end))
				elif play_beats(play_type, strength, len(cards),
								self.current_play.play_type, self.current_play.strength,
								len(self.current_play.cards)):
					legal.append((start, end))
		return legal

	def get_legal_play_starts(self) -> list[int]:
		return sorted(set(s for s, e in self.get_legal_plays()))

	def get_legal_play_ends(self, start: int) -> list[int]:
		return sorted(e for s, e in self.get_legal_plays() if s == start)

	def get_legal_insert_positions(self) -> list[int]:
		"""Valid positions for inserting a scouted card: 0 to len(hand)."""
		return list(range(len(self.players[self.current_player].hand) + 1))

	def apply_play(self, start: int, end: int):
		"""Play consecutive cards hand[start:end+1] onto the table."""
		assert self.phase in (Phase.PLAY, Phase.SNS_PLAY)
		player = self.players[self.current_player]
		cards = player.hand[start:end + 1]
		result = classify_play(cards)
		assert result is not None, f"Invalid play: {cards}"
		play_type, strength = result
		if self.current_play is not None:
			assert play_beats(play_type, strength, len(cards),
							  self.current_play.play_type, self.current_play.strength,
							  len(self.current_play.cards)), "Play doesn't beat current"
		# Collect previous play's cards
		if self.current_play is not None:
			player.collected.extend(self.current_play.cards)
		# Remove played cards from hand
		player.hand = player.hand[:start] + player.hand[end + 1:]
		# New play on table
		self.current_play = Play(cards=cards, play_type=play_type, strength=strength)
		self.current_play_owner = self.current_player
		self.scouts_since_play = 0
		# Hand empty -> round ends
		if not player.hand:
			self.round_ender = self.current_player
			self._end_round()
			return
		self.phase = Phase.PLAY
		self._advance_turn()

	def apply_scout(self, left_end: bool, flip: bool, insert_pos: int):
		"""Scout a card from the current play (regular scout, not S&S)."""
		assert self.phase == Phase.PLAY
		assert self.current_play is not None
		self._do_scout(left_end, flip, insert_pos)
		if self.scouts_since_play >= self.num_players - 1:
			self.round_ender = self.current_play_owner
			self._end_round()
			return
		self._advance_turn()

	def apply_sns_scout(self, left_end: bool, flip: bool, insert_pos: int):
		"""Scout portion of Scout & Show. After this, phase becomes SNS_PLAY
		if legal plays exist, otherwise degrades to a regular scout."""
		assert self.phase == Phase.PLAY
		assert self.current_play is not None
		player = self.players[self.current_player]
		assert player.sns_available
		player.sns_available = False
		self._do_scout(left_end, flip, insert_pos)
		if self._has_any_legal_play():
			# S&S is atomic — don't check round end between scout and play
			self.phase = Phase.SNS_PLAY
		else:
			# Can't play after scouting: degrade to regular scout
			if self.scouts_since_play >= self.num_players - 1:
				self.round_ender = self.current_play_owner
				self._end_round()
				return
			self._advance_turn()

	def _do_scout(self, left_end: bool, flip: bool, insert_pos: int):
		play_cards = self.current_play.cards
		card = play_cards.pop(0) if left_end else play_cards.pop()
		if flip:
			card = (card[1], card[0])
		self.players[self.current_player].hand.insert(insert_pos, card)
		if self.current_play_owner is not None:
			self.players[self.current_play_owner].scout_tokens += 1
		self.scouts_since_play += 1
		# Update or clear the play
		if play_cards:
			result = classify_play(play_cards)
			# Removing an end from a valid play always yields a valid play
			self.current_play = Play(cards=play_cards, play_type=result[0], strength=result[1])
		else:
			self.current_play = None
			self.current_play_owner = None
			self.scouts_since_play = 0

	def _has_any_legal_play(self) -> bool:
		return len(self.get_legal_plays()) > 0

	def _advance_turn(self):
		self.current_player = (self.current_player + 1) % self.num_players

	def _end_round(self):
		self.phase = Phase.ROUND_OVER
		for i, player in enumerate(self.players):
			score = len(player.collected) + player.scout_tokens
			if i != self.round_ender:
				score -= len(player.hand)
			self.cumulative_scores[i] += score
		self.round_number += 1
		self.starting_player = (self.starting_player + 1) % self.num_players
		if self.round_number >= self.total_rounds:
			self.phase = Phase.GAME_OVER

	def get_round_scores(self) -> list[int]:
		"""Per-player scores for the most recent round. Call before start_round()."""
		scores = []
		for i, player in enumerate(self.players):
			score = len(player.collected) + player.scout_tokens
			if i != self.round_ender:
				score -= len(player.hand)
			scores.append(score)
		return scores

	def get_rewards(self) -> list[float]:
		"""Normalized rewards for each player at game end.
		Reward = (score margin vs next-highest + 5 if winner) / 20."""
		assert self.phase == Phase.GAME_OVER
		max_score = max(self.cumulative_scores)
		rewards = []
		for i in range(self.num_players):
			my_score = self.cumulative_scores[i]
			next_highest = max(s for j, s in enumerate(self.cumulative_scores) if j != i)
			margin = my_score - next_highest
			bonus = 5 if my_score >= max_score else 0
			rewards.append((margin + bonus) / 20.0)
		return rewards

	def get_state_for_player(self, player: int) -> dict:
		"""Visible game state from a player's perspective, for encoding.
		All per-player arrays are rotated so index 0 = requesting player,
		and subsequent indices are opponents in clockwise seat order."""
		def rotate(lst):
			return lst[player:] + lst[:player]
		return {
			"hand": self.players[player].hand,
			"current_play": [(c[0], c[1]) for c in self.current_play.cards] if self.current_play else None,
			"scores": rotate(self.cumulative_scores[:]),
			"hand_sizes": rotate([len(p.hand) for p in self.players]),
			"sns_available": rotate([p.sns_available for p in self.players]),
			"scouts_since_play": self.scouts_since_play,
			"play_owner_relative_pos": (
				(self.current_play_owner - player) % self.num_players
				if self.current_play_owner is not None else None
			),
			"num_players": self.num_players,
			"round_number": self.round_number,
			"total_rounds": self.total_rounds,
		}

	def get_relative_position(self, from_player: int, to_player: int) -> int:
		"""Seat distance from from_player to to_player (1 to num_players-1)."""
		return (to_player - from_player) % self.num_players
