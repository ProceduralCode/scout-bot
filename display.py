from __future__ import annotations
from game import Card, Play, PlayType

# 0 represents 10 in the compact notation
def _val_to_char(v: int) -> str:
	return "T" if v == 10 else str(v)

def _char_to_val(c: str) -> int:
	return 10 if c in ("0", "T") else int(c)

def parse_card(s: str) -> Card:
	"""Parse compact notation to a Card. "37" -> (3, 7), "05" -> (10, 5)."""
	s = s.strip()
	if len(s) != 2 or not all(c.isdigit() or c == "T" for c in s):
		raise ValueError(f"Invalid card: {s!r} (expected 2 chars, t=10)")
	return (_char_to_val(s[0]), _char_to_val(s[1]))

def format_card(card: Card) -> str:
	"""Format a Card to compact notation. (3, 7) -> "37", (10, 5) -> "05"."""
	return _val_to_char(card[0]) + _val_to_char(card[1])

def parse_cards(s: str) -> list[Card]:
	"""Parse space-separated compact cards. "37 43 51" -> [(3,7), (4,3), (5,1)]."""
	return [parse_card(tok) for tok in s.split()]

def format_hand(hand: list[Card]) -> str:
	"""Format a hand as space-separated compact cards."""
	return " ".join(format_card(c) for c in hand)

def format_play(play: Play) -> str:
	"""Format a play: cards + type + strength."""
	cards = " ".join(format_card(c) for c in play.cards)
	kind = "set" if play.play_type == PlayType.SET else "run"
	return f"{cards} ({kind} of {play.count}, str {play.strength})"

def format_play_type(play: Play) -> str:
	"""Format just the type info of a play, no cards."""
	kind = "set" if play.play_type == PlayType.SET else "run"
	return f"{kind} of {play.count}, str {play.strength}"

def format_showing_values(cards: list[Card]) -> str:
	"""Format only the showing values of cards, space-separated."""
	return " ".join(_val_to_char(c[0]) for c in cards)
