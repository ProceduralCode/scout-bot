from __future__ import annotations
import argparse
import random
import torch
from dataclasses import dataclass, field
from game import Card, Play, PlayType, Phase, flip_hand
from encoding import (
	encode_state, encode_hand_both_orientations, get_action_type_mask,
	get_play_start_mask, get_play_end_mask, get_scout_insert_mask,
	get_sns_insert_mask,
	get_legal_plays, decode_action_type, decode_slot_to_hand_index,
	INPUT_SIZE, HAND_SLOTS, PLAY_SLOTS,
)
from network import ScoutNetwork, masked_sample
from display import parse_cards, format_hand, format_play, format_card

@dataclass
class PlayerInfo:
	"""Tracks per-player state. Bot has real hand; opponents just have size."""
	hand: list[Card] = field(default_factory=list) # only meaningful for bot
	hand_size: int = 0
	collected_count: int = 0
	scout_tokens: int = 0
	sns_available: bool = True

class LiveGame:
	"""Partial-information game state for interactive play.
	Duck-types the Game attributes that encode_state and mask functions need."""
	def __init__(self, num_players: int, bot_seat: int):
		self.num_players = num_players
		self.bot_seat = bot_seat
		self.total_rounds = num_players
		self.round_number = 0
		self.cumulative_scores = [0] * num_players
		self.starting_player = 0
		self.players: list[PlayerInfo] = [PlayerInfo() for _ in range(num_players)]
		self.current_play: Play | None = None
		self.current_play_owner: int | None = None
		self.current_player = 0
		self.scouts_since_play = 0
		self.phase = Phase.GAME_OVER
		self.round_ender: int | None = None
		self.bot_alive = True

	def start_round(self, bot_hand: list[Card], hand_sizes: list[int]):
		"""Begin a new round. Bot hand is real cards; others are just sizes."""
		self.players = [PlayerInfo() for _ in range(self.num_players)]
		self.players[self.bot_seat].hand = list(bot_hand)
		self.players[self.bot_seat].hand_size = len(bot_hand)
		for i in range(self.num_players):
			if i != self.bot_seat:
				self.players[i].hand_size = hand_sizes[i]
		self.current_play = None
		self.current_play_owner = None
		self.scouts_since_play = 0
		self.current_player = self.starting_player
		self.round_ender = None
		self.phase = Phase.TURN
		self.bot_alive = True

	def get_state_for_player(self, player: int) -> dict:
		"""Build state dict matching Game.get_state_for_player format."""
		def rotate(lst):
			return lst[player:] + lst[:player]
		return {
			"hand": self.players[player].hand,
			"current_play": list(self.current_play.cards) if self.current_play else None,
			"scores": rotate(self.cumulative_scores[:]),
			"hand_sizes": rotate([p.hand_size for p in self.players]),
			"collected_counts": rotate([p.collected_count for p in self.players]),
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

	def apply_opponent_play(self, cards: list[Card]):
		"""Opponent plays cards onto the table."""
		p = self.current_player
		new_play = Play.from_cards(cards)
		if new_play is None:
			print(f"  Warning: {_card_list(cards)} is not a valid play")
			return False
		if self.current_play is not None:
			if not new_play.beats(self.current_play):
				print(f"  Warning: {_card_list(cards)} doesn't beat current play")
				return False
			self.players[p].collected_count += len(self.current_play.cards)
		self.players[p].hand_size -= len(cards)
		self.current_play = new_play
		self.current_play_owner = p
		self.scouts_since_play = 0
		if self.players[p].hand_size <= 0:
			self.round_ender = p
			self._end_round()
			return True
		self._advance_turn()
		return True

	def apply_opponent_scout(self, left_end: bool):
		"""Opponent scouts a card from the current play."""
		if self.current_play is None:
			print("  Warning: no play to scout from")
			return False
		p = self.current_player
		play_cards = list(self.current_play.cards)
		play_cards.pop(0) if left_end else play_cards.pop()
		self.players[p].hand_size += 1
		if self.current_play_owner is not None:
			self.players[self.current_play_owner].scout_tokens += 1
		self.scouts_since_play += 1
		if play_cards:
			self.current_play = Play.from_cards(play_cards)
		else:
			self.current_play = None
			self.current_play_owner = None
			self.scouts_since_play = 0
		if self.scouts_since_play >= self.num_players - 1:
			self.round_ender = self.current_play_owner
			self._end_round()
			return True
		self._advance_turn()
		return True

	def apply_opponent_sns(self, left_end: bool, played_cards: list[Card]):
		"""Opponent does S&S: scout then play."""
		p = self.current_player
		self.players[p].sns_available = False
		# Scout portion
		play_cards = list(self.current_play.cards)
		play_cards.pop(0) if left_end else play_cards.pop()
		self.players[p].hand_size += 1
		if self.current_play_owner is not None:
			self.players[self.current_play_owner].scout_tokens += 1
		self.scouts_since_play += 1
		if play_cards:
			self.current_play = Play.from_cards(play_cards)
		else:
			self.current_play = None
			self.current_play_owner = None
			self.scouts_since_play = 0
		# Play portion
		new_play = Play.from_cards(played_cards)
		if new_play is None:
			print(f"  Warning: {_card_list(played_cards)} is not a valid play")
			return False
		if self.current_play is not None:
			if not new_play.beats(self.current_play):
				print(f"  Warning: {_card_list(played_cards)} doesn't beat current play after scout")
				return False
			self.players[p].collected_count += len(self.current_play.cards)
		self.players[p].hand_size -= len(played_cards)
		self.current_play = new_play
		self.current_play_owner = p
		self.scouts_since_play = 0
		if self.players[p].hand_size <= 0:
			self.round_ender = p
			self._end_round()
			return True
		self._advance_turn()
		return True

	def get_bot_action(self, network: ScoutNetwork) -> dict | None:
		"""Run the network on current state and return the chosen action."""
		if not self.bot_alive:
			return None
		p = self.bot_seat
		hand = self.players[p].hand
		hand_offset = random.randint(0, HAND_SLOTS - 1)
		play_offset = random.randint(0, PLAY_SLOTS - 1)
		with torch.no_grad():
			state_tensor = encode_state(self, p, hand_offset, play_offset)
			hidden = network(state_tensor)
			legal_plays = get_legal_plays(hand, self.current_play)
			at_logits = network.action_type_logits(hidden)
			at_mask = get_action_type_mask(self, legal_plays)
			action_type, _ = masked_sample(at_logits, at_mask)
			action_info = decode_action_type(action_type)
			result = {"type": action_info["type"]}
			if action_info["type"] == "play":
				ps_logits = network.play_start_logits(hidden, action_type)
				ps_mask = get_play_start_mask(legal_plays, hand_offset)
				start_slot, _ = masked_sample(ps_logits, ps_mask)
				start_idx = decode_slot_to_hand_index(start_slot, hand_offset)
				pe_logits = network.play_end_logits(hidden, action_type, start_slot)
				pe_mask = get_play_end_mask(legal_plays, start_idx, hand_offset)
				end_slot, _ = masked_sample(pe_logits, pe_mask)
				end_idx = decode_slot_to_hand_index(end_slot, hand_offset)
				result["start"] = start_idx
				result["end"] = end_idx
				result["cards"] = hand[start_idx:end_idx + 1]
			elif action_info["type"] in ("scout", "sns"):
				si_logits = network.scout_insert_logits(hidden, action_type)
				if action_info["type"] == "sns":
					si_mask = get_sns_insert_mask(self, action_info["left_end"], action_info["flip"])
				else:
					si_mask = get_scout_insert_mask(self)
				insert_pos, _ = masked_sample(si_logits, si_mask)
				result["left_end"] = action_info["left_end"]
				result["flip"] = action_info["flip"]
				result["insert_pos"] = insert_pos
				# Which card will be scouted
				play_cards = self.current_play.cards
				card = play_cards[0] if action_info["left_end"] else play_cards[-1]
				if action_info["flip"]:
					card = (card[1], card[0])
				result["scouted_card"] = card
				if action_info["type"] == "sns":
					# After scouting, compute the forced play
					new_hand = list(hand)
					new_hand.insert(insert_pos, card)
					# Remove scouted card from play
					remaining = list(play_cards)
					remaining.pop(0) if action_info["left_end"] else remaining.pop()
					reduced_play = Play.from_cards(remaining) if remaining else None
					# Re-run network for the play decision
					# Update hand temporarily for encoding
					old_hand = self.players[p].hand
					self.players[p].hand = new_hand
					old_play = self.current_play
					self.current_play = reduced_play
					self.phase = Phase.SNS_PLAY
					state2 = encode_state(self, p, hand_offset, play_offset)
					hidden2 = network(state2)
					sns_legal = get_legal_plays(new_hand, reduced_play)
					ps_logits = network.play_start_logits(hidden2, 0) # action_type 0 = play
					ps_mask = get_play_start_mask(sns_legal, hand_offset)
					start_slot, _ = masked_sample(ps_logits, ps_mask)
					start_idx = decode_slot_to_hand_index(start_slot, hand_offset)
					pe_logits = network.play_end_logits(hidden2, 0, start_slot)
					pe_mask = get_play_end_mask(sns_legal, start_idx, hand_offset)
					end_slot, _ = masked_sample(pe_logits, pe_mask)
					end_idx = decode_slot_to_hand_index(end_slot, hand_offset)
					result["play_start"] = start_idx
					result["play_end"] = end_idx
					result["play_cards"] = new_hand[start_idx:end_idx + 1]
					# Restore state
					self.players[p].hand = old_hand
					self.current_play = old_play
					self.phase = Phase.TURN
		return result

	def apply_bot_play(self, start: int, end: int):
		"""Apply the bot's play action."""
		p = self.bot_seat
		hand = self.players[p].hand
		cards = hand[start:end + 1]
		new_play = Play.from_cards(cards)
		if new_play is None or (self.current_play and not new_play.beats(self.current_play)):
			return False
		if self.current_play is not None:
			self.players[p].collected_count += len(self.current_play.cards)
		self.players[p].hand = hand[:start] + hand[end + 1:]
		self.players[p].hand_size = len(self.players[p].hand)
		self.current_play = new_play
		self.current_play_owner = p
		self.scouts_since_play = 0
		if not self.players[p].hand:
			self.round_ender = p
			self._end_round()
			return True
		self._advance_turn()
		return True

	def apply_bot_scout(self, left_end: bool, flip: bool, insert_pos: int):
		"""Apply the bot's scout action."""
		p = self.bot_seat
		play_cards = list(self.current_play.cards)
		card = play_cards.pop(0) if left_end else play_cards.pop()
		if flip:
			card = (card[1], card[0])
		self.players[p].hand.insert(insert_pos, card)
		self.players[p].hand_size = len(self.players[p].hand)
		if self.current_play_owner is not None:
			self.players[self.current_play_owner].scout_tokens += 1
		self.scouts_since_play += 1
		if play_cards:
			self.current_play = Play.from_cards(play_cards)
		else:
			self.current_play = None
			self.current_play_owner = None
			self.scouts_since_play = 0
		if self.scouts_since_play >= self.num_players - 1:
			self.round_ender = self.current_play_owner
			self._end_round()
			return True
		self._advance_turn()
		return True

	def apply_bot_sns(self, left_end: bool, flip: bool, insert_pos: int,
					  play_start: int, play_end: int):
		"""Apply the bot's S&S: scout then play."""
		p = self.bot_seat
		self.players[p].sns_available = False
		# Scout portion
		play_cards = list(self.current_play.cards)
		card = play_cards.pop(0) if left_end else play_cards.pop()
		if flip:
			card = (card[1], card[0])
		self.players[p].hand.insert(insert_pos, card)
		if self.current_play_owner is not None:
			self.players[self.current_play_owner].scout_tokens += 1
		self.scouts_since_play += 1
		if play_cards:
			self.current_play = Play.from_cards(play_cards)
		else:
			self.current_play = None
			self.current_play_owner = None
			self.scouts_since_play = 0
		# Play portion
		hand = self.players[p].hand
		cards = hand[play_start:play_end + 1]
		new_play = Play.from_cards(cards)
		if new_play is None or (self.current_play and not new_play.beats(self.current_play)):
			return False
		if self.current_play is not None:
			self.players[p].collected_count += len(self.current_play.cards)
		self.players[p].hand = hand[:play_start] + hand[play_end + 1:]
		self.players[p].hand_size = len(self.players[p].hand)
		self.current_play = new_play
		self.current_play_owner = p
		self.scouts_since_play = 0
		if not self.players[p].hand:
			self.round_ender = p
			self._end_round()
			return True
		self._advance_turn()
		return True

	def get_bot_flip_decision(self, network: ScoutNetwork) -> bool:
		"""Use the network to decide whether to flip the bot's hand."""
		p = self.bot_seat
		ho = random.randint(0, HAND_SLOTS - 1)
		po = random.randint(0, PLAY_SLOTS - 1)
		with torch.no_grad():
			t_normal, t_flipped = encode_hand_both_orientations(self, p, ho, po)
			h_normal = network(t_normal)
			h_flipped = network(t_flipped)
			v_normal = network.value(h_normal).item()
			v_flipped = network.value(h_flipped).item()
		return v_flipped > v_normal

	def apply_flip(self, do_flip: bool):
		"""Apply flip decision for the bot."""
		if do_flip:
			self.players[self.bot_seat].hand = flip_hand(self.players[self.bot_seat].hand)

	def get_round_scores(self) -> list[int]:
		"""Compute round scores from tracked state."""
		scores = []
		for i, p in enumerate(self.players):
			score = p.collected_count + p.scout_tokens
			if i != self.round_ender:
				score -= p.hand_size
			scores.append(score)
		return scores

	def _advance_turn(self):
		self.current_player = (self.current_player + 1) % self.num_players
		# Skip bot if it's been marked out
		if self.current_player == self.bot_seat and not self.bot_alive:
			self._advance_turn()

	def _end_round(self):
		self.phase = Phase.ROUND_OVER
		scores = self.get_round_scores()
		for i, s in enumerate(scores):
			self.cumulative_scores[i] += s
		self.round_number += 1
		self.starting_player = (self.starting_player + 1) % self.num_players

def _card_list(cards: list[Card]) -> str:
	return " ".join(format_card(c) for c in cards)

# --- CLI ---

def _format_bot_action(action: dict, game: LiveGame) -> str:
	"""Format the bot's chosen action as a human-readable instruction."""
	if action["type"] == "play":
		# 1-indexed for physical card counting
		start = action["start"] + 1
		end = action["end"] + 1
		cards = action["cards"]
		cards_str = format_hand(cards)
		if start == end:
			return f"Play card {start}: {cards_str}"
		return f"Play cards {start}-{end}: {cards_str}"
	elif action["type"] == "scout":
		side = "left" if action["left_end"] else "right"
		pos = action["insert_pos"] + 1 # 1-indexed
		card_str = format_card(action["scouted_card"])
		return f"Scout {side} ({card_str}), insert at position {pos}"
	elif action["type"] == "sns":
		side = "left" if action["left_end"] else "right"
		scout_pos = action["insert_pos"] + 1
		card_str = format_card(action["scouted_card"])
		play_start = action["play_start"] + 1
		play_end = action["play_end"] + 1
		play_cards_str = format_hand(action["play_cards"])
		if play_start == play_end:
			return (f"S&S: scout {side} ({card_str}), insert at {scout_pos}, "
					f"play card {play_start}: {play_cards_str}")
		return (f"S&S: scout {side} ({card_str}), insert at {scout_pos}, "
				f"play cards {play_start}-{play_end}: {play_cards_str}")
	return "???"

def _show_table(game: LiveGame):
	"""Print current table state."""
	if game.current_play:
		print(f"  Table: {format_play(game.current_play)} by P{game.current_play_owner}")
	else:
		print("  Table: empty")

def _parse_opponent_input(line: str, game: LiveGame) -> bool:
	"""Parse and apply opponent input. Returns True if action was applied."""
	parts = line.strip().split()
	if not parts:
		return False
	cmd = parts[0].lower()
	if cmd == "p":
		# Play: p 37 43 51
		if len(parts) < 2:
			print("  Usage: p <card> [card ...]")
			return False
		try:
			cards = parse_cards(" ".join(parts[1:]))
		except ValueError as e:
			print(f"  {e}")
			return False
		return game.apply_opponent_play(cards)
	elif cmd == "s":
		# Scout: s l / s r
		if len(parts) < 2 or parts[1].lower() not in ("l", "r"):
			print("  Usage: s l|r")
			return False
		left_end = parts[1].lower() == "l"
		return game.apply_opponent_scout(left_end)
	elif cmd == "ss":
		# S&S: ss l 53 68 73
		if len(parts) < 3 or parts[1].lower() not in ("l", "r"):
			print("  Usage: ss l|r <card> [card ...]")
			return False
		left_end = parts[1].lower() == "l"
		try:
			played_cards = parse_cards(" ".join(parts[2:]))
		except ValueError as e:
			print(f"  {e}")
			return False
		return game.apply_opponent_sns(left_end, played_cards)
	else:
		print(f"  Unknown command: {cmd}")
		print("  Commands: p <cards>, s l|r, ss l|r <cards>")
		return False

def run_interactive(model_path: str, num_players: int, bot_seat: int):
	"""Run an interactive game session."""
	# Load model
	checkpoint = torch.load(model_path, weights_only=False)
	cfg = checkpoint.get("config", {})
	network = ScoutNetwork(
		input_size=INPUT_SIZE,
		hidden_size=cfg.get("hidden_size", 128),
		first_hidden_size=cfg.get("first_hidden_size", 256),
	)
	network.load_state_dict(checkpoint["model_state"])
	network.eval()
	print(f"Loaded model from {model_path}")
	print(f"  {num_players} players, bot is P{bot_seat}")
	print()
	game = LiveGame(num_players, bot_seat)
	# Determine cards per player from deck size
	from game import create_deck
	deck = create_deck(num_players)
	cards_per_player = len(deck) // num_players
	for round_num in range(num_players):
		print(f"=== Round {round_num + 1} (P{game.starting_player} starts) ===")
		# Enter bot's hand
		while True:
			try:
				raw = input(f"Bot (P{bot_seat}) hand: ").strip()
				if not raw:
					continue
				bot_hand = parse_cards(raw)
				if len(bot_hand) != cards_per_player:
					print(f"  Expected {cards_per_player} cards, got {len(bot_hand)}")
					continue
				break
			except ValueError as e:
				print(f"  {e}")
			except (EOFError, KeyboardInterrupt):
				print("\nBye!")
				return
		# All opponents get the same hand size
		hand_sizes = [cards_per_player] * num_players
		game.start_round(bot_hand, hand_sizes)
		# Flip decision
		do_flip = game.get_bot_flip_decision(network)
		game.apply_flip(do_flip)
		if do_flip:
			print(f"Bot: FLIP")
			print(f"  Hand now: {format_hand(game.players[bot_seat].hand)}")
		else:
			print(f"Bot: KEEP")
		print()
		# Turn loop
		while game.phase == Phase.TURN:
			cp = game.current_player
			_show_table(game)
			if cp == bot_seat:
				if not game.bot_alive:
					print(f"P{cp} (bot)> [out]")
					game._advance_turn()
					continue
				action = game.get_bot_action(network)
				if action is None:
					print(f"P{cp} (bot)> [out]")
					game._advance_turn()
					continue
				instruction = _format_bot_action(action, game)
				print(f"BOT> {instruction}")
				# Wait for confirmation or 'x' to mark invalid
				try:
					confirm = input("  (enter to confirm, x to mark invalid) ").strip().lower()
				except (EOFError, KeyboardInterrupt):
					print("\nBye!")
					return
				if confirm == "x":
					game.bot_alive = False
					print("  Bot is out for this round")
					game._advance_turn()
					continue
				# Apply the action
				if action["type"] == "play":
					ok = game.apply_bot_play(action["start"], action["end"])
				elif action["type"] == "scout":
					ok = game.apply_bot_scout(
						action["left_end"], action["flip"], action["insert_pos"])
				elif action["type"] == "sns":
					ok = game.apply_bot_sns(
						action["left_end"], action["flip"], action["insert_pos"],
						action["play_start"], action["play_end"])
				else:
					ok = False
				if not ok:
					print("  Action failed — bot is out for this round")
					game.bot_alive = False
					game._advance_turn()
			else:
				# Opponent's turn
				while True:
					try:
						line = input(f"P{cp}> ").strip()
					except (EOFError, KeyboardInterrupt):
						print("\nBye!")
						return
					if not line:
						continue
					if _parse_opponent_input(line, game):
						break
		# Round over
		scores = game.get_round_scores()
		scores_str = "  ".join(f"P{i}={s:+d}" for i, s in enumerate(scores))
		ender_str = f" (P{game.round_ender} went out)" if game.round_ender is not None else ""
		print(f"\nRound scores: {scores_str}{ender_str}")
		cumulative_str = "  ".join(f"P{i}={s}" for i, s in enumerate(game.cumulative_scores))
		print(f"Cumulative:   {cumulative_str}\n")
	# Game over
	print("=== Game Over ===")
	final_str = "  ".join(f"P{i}={s}" for i, s in enumerate(game.cumulative_scores))
	winner = game.cumulative_scores.index(max(game.cumulative_scores))
	print(f"Final scores: {final_str}")
	print(f"Winner: P{winner}")

def main():
	parser = argparse.ArgumentParser(description="Play Scout with the bot")
	parser.add_argument("model", help="Path to model checkpoint (.pt)")
	parser.add_argument("--players", type=int, default=4, choices=[3, 4, 5])
	parser.add_argument("--bot-seat", type=int, default=0,
		help="Which player seat the bot occupies (0-indexed)")
	args = parser.parse_args()
	if args.bot_seat >= args.players:
		print(f"Error: bot seat {args.bot_seat} >= number of players {args.players}")
		return
	run_interactive(args.model, args.players, args.bot_seat)

if __name__ == "__main__":
	main()
