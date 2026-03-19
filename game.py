from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
import random

# (showing_value, hidden_value)
Card = tuple[int, int]

class PlayType(Enum):
	SET = auto()
	RUN = auto()

class Phase(Enum):
	FLIP_DECISION = auto()
	TURN = auto()
	SNS_PLAY = auto() # must play cards after S&S scout
	ROUND_OVER = auto()
	GAME_OVER = auto()

@dataclass
class Play:
	cards: list[Card]
	count: int
	play_type: PlayType
	strength: int

	@classmethod
	def from_cards(cls, cards: list[Card]) -> Play | None:
		"""Create a Play from cards, or None if not a valid play."""
		values = [c[0] for c in cards]
		n = len(values)
		if all(v == values[0] for v in values):
			return cls(cards=cards, count=n, play_type=PlayType.SET, strength=values[0])
		ascending = all(values[i + 1] == values[i] + 1 for i in range(n - 1))
		descending = all(values[i + 1] == values[i] - 1 for i in range(n - 1))
		if ascending or descending:
			return cls(cards=cards, count=n, play_type=PlayType.RUN, strength=max(values))
		return None

	def beats(self, other: Play) -> bool:
		"""Check if this play beats another."""
		if self.count > other.count:
			return True
		if self.count < other.count:
			return False
		# Same count: set beats run, then higher strength wins
		if self.play_type != other.play_type:
			return self.play_type == PlayType.SET
		return self.strength > other.strength

@dataclass
class PlayerState:
	hand: list[Card]
	collected: list[Card] = field(default_factory=list)
	scout_tokens: int = 0
	sns_available: bool = True

def flip_hand(hand: list[Card]) -> list[Card]:
	"""Flip entire hand: swap each card's showing/hidden values."""
	return [(b, a) for a, b in hand]

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
			self.phase = Phase.TURN

	def apply_play(self, start: int, end: int):
		"""Play consecutive cards hand[start:end+1] onto the table."""
		assert self.phase in (Phase.TURN, Phase.SNS_PLAY)
		player = self.players[self.current_player]
		cards = player.hand[start:end + 1]
		new_play = Play.from_cards(cards)
		assert new_play is not None, f"Invalid play: {cards}"
		if self.current_play is not None:
			assert new_play.beats(self.current_play), "Play doesn't beat current"
			player.collected.extend(self.current_play.cards)
		player.hand = player.hand[:start] + player.hand[end + 1:]
		self.current_play = new_play
		self.current_play_owner = self.current_player
		self.scouts_since_play = 0
		# Hand empty -> round ends
		if not player.hand:
			self.round_ender = self.current_player
			self._end_round()
			return
		self.phase = Phase.TURN
		self._advance_turn()

	def apply_scout(self, left_end: bool, flip: bool, insert_pos: int):
		"""Scout a card from the current play (regular scout, not S&S)."""
		assert self.phase == Phase.TURN
		assert self.current_play is not None
		self._do_scout(left_end, flip, insert_pos)
		if self.scouts_since_play >= self.num_players - 1:
			self.round_ender = self.current_play_owner
			self._end_round()
			return
		self._advance_turn()

	def apply_sns_scout(self, left_end: bool, flip: bool, insert_pos: int):
		"""Scout portion of Scout & Show. Phase becomes SNS_PLAY;
		caller must ensure a legal play exists after scouting."""
		assert self.phase == Phase.TURN
		assert self.current_play is not None
		player = self.players[self.current_player]
		assert player.sns_available
		player.sns_available = False
		self._do_scout(left_end, flip, insert_pos)
		assert self._has_any_legal_play(), "S&S requires a legal play after scouting"
		self.phase = Phase.SNS_PLAY

	def _do_scout(self, left_end: bool, flip: bool, insert_pos: int):
		play_cards = list(self.current_play.cards)
		card = play_cards.pop(0) if left_end else play_cards.pop()
		if flip:
			card = (card[1], card[0])
		self.players[self.current_player].hand.insert(insert_pos, card)
		self.players[self.current_play_owner].scout_tokens += 1
		self.scouts_since_play += 1
		# Update or clear the play
		if play_cards:
			# Removing an end from a valid play always yields a valid play
			self.current_play = Play.from_cards(play_cards)
		else:
			self.current_play = None
			self.current_play_owner = None
			self.scouts_since_play = 0

	def _has_any_legal_play(self) -> bool:
		hand = self.players[self.current_player].hand
		if not self.current_play:
			return len(hand) > 0
		cur = self.current_play
		values = [c[0] for c in hand]
		n = len(values)
		for start in range(n):
			is_set = is_asc = is_desc = True
			for end in range(start, n):
				if end > start:
					v, pv = values[end], values[end - 1]
					is_set = is_set and v == values[start]
					is_asc = is_asc and v == pv + 1
					is_desc = is_desc and v == pv - 1
					if not is_set and not is_asc and not is_desc:
						break
				count = end - start + 1
				if count < cur.count:
					continue
				if count > cur.count:
					return True
				# Same count: check type and strength
				if is_set:
					if cur.play_type == PlayType.RUN or values[start] > cur.strength:
						return True
				elif is_asc:
					if cur.play_type == PlayType.RUN and values[end] > cur.strength:
						return True
				elif is_desc:
					if cur.play_type == PlayType.RUN and values[start] > cur.strength:
						return True
		return False

	def _advance_turn(self):
		self.current_player = (self.current_player + 1) % self.num_players

	def _end_round(self):
		self.phase = Phase.ROUND_OVER
		for i, score in enumerate(self.get_round_scores()):
			self.cumulative_scores[i] += score
		self.round_number += 1
		self.starting_player = (self.starting_player + 1) % self.num_players
		if self.round_number >= self.total_rounds:
			self.phase = Phase.GAME_OVER

	def get_round_scores(self) -> list[int]:
		"""Per-player scores for the most recent round."""
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
			"current_play": list(self.current_play.cards) if self.current_play else None,
			"scores": rotate(self.cumulative_scores[:]),
			"hand_sizes": rotate([len(p.hand) for p in self.players]),
			"collected_counts": rotate([len(p.collected) for p in self.players]),
			"scout_tokens": rotate([p.scout_tokens for p in self.players]),
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
