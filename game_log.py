from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from game import Card, Play, Game, Phase
from display import format_card, format_hand, format_play, format_play_type, format_showing_values

@dataclass
class GameEvent:
	round_num: int
	turn: int
	player: int
	phase: str # "flip", "play", "scout", "sns", "sns_play"
	cards_involved: list[Card]
	hand_after: list[Card] | None = None # acting player's hand after action
	table_after: list[Card] | None = None # current play after action
	table_play: Play | None = None # Play object for table (not serialized directly)
	insert_pos: int | None = None # for scout/sns
	scout_end: str | None = None # "left" or "right"
	flip_decision: bool | None = None # for flip events

@dataclass
class RoundStart:
	round_num: int
	starting_player: int
	hands: dict[int, list[Card]] # player -> initial hand

@dataclass
class RoundEnd:
	round_num: int
	scores: list[int]
	ender: int | None

@dataclass
class GameLog:
	num_players: int
	round_starts: list[RoundStart] = field(default_factory=list)
	round_ends: list[RoundEnd] = field(default_factory=list)
	events: list[GameEvent] = field(default_factory=list)
	final_scores: list[int] | None = None
	_turn_counter: int = field(default=0, repr=False)

	def record_round_start(self, game: Game):
		hands = {}
		for i in range(game.num_players):
			hands[i] = list(game.players[i].hand)
		self.round_starts.append(RoundStart(
			round_num=game.round_number,
			starting_player=game.current_player,
			hands=hands,
		))
		self._turn_counter = 0

	def record_flip(self, round_num: int, player: int, did_flip: bool, hand_after: list[Card]):
		self.events.append(GameEvent(
			round_num=round_num,
			turn=0,
			player=player,
			phase="flip",
			cards_involved=[],
			hand_after=list(hand_after),
			flip_decision=did_flip,
		))

	def record_play(self, game: Game, player: int, cards: list[Card], round_num: int | None = None):
		self._turn_counter += 1
		table = list(game.current_play.cards) if game.current_play else None
		self.events.append(GameEvent(
			round_num=round_num if round_num is not None else game.round_number,
			turn=self._turn_counter,
			player=player,
			phase="play",
			cards_involved=list(cards),
			hand_after=list(game.players[player].hand),
			table_after=table,
			table_play=game.current_play,
		))

	def record_scout(self, game: Game, player: int, card: Card, left_end: bool, insert_pos: int, round_num: int | None = None):
		self._turn_counter += 1
		table = list(game.current_play.cards) if game.current_play else None
		self.events.append(GameEvent(
			round_num=round_num if round_num is not None else game.round_number,
			turn=self._turn_counter,
			player=player,
			phase="scout",
			cards_involved=[card],
			hand_after=list(game.players[player].hand),
			table_after=table,
			scout_end="left" if left_end else "right",
			insert_pos=insert_pos,
		))

	def record_sns(self, game: Game, player: int, scout_card: Card, left_end: bool,
				   insert_pos: int, played_cards: list[Card]):
		self._turn_counter += 1
		table = list(game.current_play.cards) if game.current_play else None
		self.events.append(GameEvent(
			round_num=game.round_number,
			turn=self._turn_counter,
			player=player,
			phase="sns",
			cards_involved=[scout_card] + list(played_cards),
			hand_after=list(game.players[player].hand),
			table_after=table,
			table_play=game.current_play,
			scout_end="left" if left_end else "right",
			insert_pos=insert_pos,
		))

	def record_round_end(self, game: Game):
		scores = game.get_round_scores()
		self.round_ends.append(RoundEnd(
			round_num=game.round_number - 1, # round_number already incremented
			scores=scores,
			ender=game.round_ender,
		))

	def record_game_end(self, scores: list[int]):
		self.final_scores = list(scores)

	# --- Serialization ---

	def to_dict(self) -> dict:
		data = {
			"num_players": self.num_players,
			"round_starts": [],
			"round_ends": [],
			"events": [],
			"final_scores": self.final_scores,
		}
		for rs in self.round_starts:
			data["round_starts"].append({
				"round_num": rs.round_num,
				"starting_player": rs.starting_player,
				"hands": {str(k): v for k, v in rs.hands.items()},
			})
		for re in self.round_ends:
			data["round_ends"].append({
				"round_num": re.round_num,
				"scores": re.scores,
				"ender": re.ender,
			})
		for ev in self.events:
			d = {
				"round_num": ev.round_num,
				"turn": ev.turn,
				"player": ev.player,
				"phase": ev.phase,
				"cards_involved": ev.cards_involved,
				"hand_after": ev.hand_after,
				"table_after": ev.table_after,
				"insert_pos": ev.insert_pos,
				"scout_end": ev.scout_end,
				"flip_decision": ev.flip_decision,
			}
			data["events"].append(d)
		return data

	def save(self, path: str):
		with open(path, "w") as f:
			json.dump(self.to_dict(), f, indent=2)

	@classmethod
	def load(cls, path: str) -> GameLog:
		with open(path) as f:
			data = json.load(f)
		log = cls(num_players=data["num_players"])
		log.final_scores = data.get("final_scores")
		for rs in data["round_starts"]:
			hands = {int(k): [tuple(c) for c in v] for k, v in rs["hands"].items()}
			log.round_starts.append(RoundStart(
				round_num=rs["round_num"],
				starting_player=rs["starting_player"],
				hands=hands,
			))
		for re in data["round_ends"]:
			log.round_ends.append(RoundEnd(
				round_num=re["round_num"],
				scores=re["scores"],
				ender=re["ender"],
			))
		for ev in data["events"]:
			cards = [tuple(c) for c in ev["cards_involved"]]
			hand = [tuple(c) for c in ev["hand_after"]] if ev["hand_after"] else None
			table = [tuple(c) for c in ev["table_after"]] if ev["table_after"] else None
			log.events.append(GameEvent(
				round_num=ev["round_num"],
				turn=ev["turn"],
				player=ev["player"],
				phase=ev["phase"],
				cards_involved=cards,
				hand_after=hand,
				table_after=table,
				insert_pos=ev.get("insert_pos"),
				scout_end=ev.get("scout_end"),
				flip_decision=ev.get("flip_decision"),
			))
		return log

# --- Replay viewer ---

# Column widths for turn-by-turn display
_COL_T = 4   # turn number
_COL_P = 2   # player
_COL_A = 30  # action
_COL_TBL = 20 # table
_SEP = "  "

def _hand_col_width(num_players: int) -> int:
	return (45 // num_players + 2) * 2

def _replay_header(num_players: int):
	hw = _hand_col_width(num_players)
	t = "Turn".rjust(_COL_T)
	a = "Action".ljust(_COL_A)
	tbl = "Table".ljust(_COL_TBL)
	hand_headers = _SEP.join(f"P{i} Hand".ljust(hw) for i in range(num_players))
	print(f"  {t}{_SEP}Pl{_SEP}{a}{_SEP}{tbl}{_SEP}{hand_headers}")
	hand_seps = _SEP.join('-' * hw for _ in range(num_players))
	print(f"  {'-'*_COL_T}{_SEP}{'-'*_COL_P}{_SEP}{'-'*_COL_A}{_SEP}{'-'*_COL_TBL}{_SEP}{hand_seps}")

def _print_turn_event(ev: GameEvent, num_players: int):
	if ev.phase == "play":
		action = f"PLAY  {format_hand(ev.cards_involved)}"
	elif ev.phase == "scout":
		action = f"SCOUT {ev.scout_end} {format_hand(ev.cards_involved)} -> pos {ev.insert_pos}"
	elif ev.phase == "sns":
		scout_card = format_hand(ev.cards_involved[:1])
		played_cards = format_hand(ev.cards_involved[1:])
		action = f"S&S   scout {ev.scout_end} {scout_card} -> pos {ev.insert_pos}, play {played_cards}"
	elif ev.phase == "sns_play":
		action = f"S&S   play {format_hand(ev.cards_involved)}"
	else:
		action = ev.phase

	if ev.hand_after:
		hand_str = format_showing_values(ev.hand_after)
	elif ev.hand_after is not None:
		hand_str = "(out)"
	else:
		hand_str = ""

	table_str = format_hand(ev.table_after) if ev.table_after else ""

	hw = _hand_col_width(num_players)
	hand_cols = []
	for i in range(num_players):
		if i == ev.player:
			hand_cols.append(hand_str.ljust(hw))
		else:
			hand_cols.append(" " * hw)

	t = str(ev.turn).rjust(_COL_T)
	pl = f"P{ev.player}"
	a = action.ljust(_COL_A)
	tbl = table_str.ljust(_COL_TBL)
	print(f"  {t}{_SEP}{pl}{_SEP}{a}{_SEP}{tbl}{_SEP}{_SEP.join(hand_cols)}")

def print_replay(log: GameLog):
	"""Print a formatted game replay."""
	print(f"\n{'=' * 130}")
	print(f"  Scout Game - {log.num_players} players")
	print(f"{'=' * 130}\n")

	round_starts = {rs.round_num: rs for rs in log.round_starts}
	round_ends = {re.round_num: re for re in log.round_ends}
	events_by_round: dict[int, list[GameEvent]] = {}
	for ev in log.events:
		events_by_round.setdefault(ev.round_num, []).append(ev)

	for round_num in sorted(set(
		list(events_by_round.keys()) +
		list(round_starts.keys()) +
		list(round_ends.keys())
	)):
		rs = round_starts.get(round_num)
		round_events = events_by_round.get(round_num, [])
		flips = {e.player: e for e in round_events if e.phase == "flip"}

		if rs:
			print(f"  -- Round {round_num + 1} (P{rs.starting_player} starts) {'-' * 100}\n")
			hand_width = max(len(format_hand(h)) for h in rs.hands.values())
			for p in sorted(rs.hands):
				hand_str = format_hand(rs.hands[p]).ljust(hand_width)
				flip_ev = flips.get(p)
				if flip_ev:
					if flip_ev.flip_decision and flip_ev.hand_after:
						flip_str = f"  -> FLIP ->  {format_hand(flip_ev.hand_after)}"
					elif flip_ev.flip_decision:
						flip_str = "  -> FLIP"
					else:
						flip_str = "  KEEP"
				else:
					flip_str = ""
				print(f"    P{p}  {hand_str}{flip_str}")
			print()

		turns = [e for e in round_events if e.phase != "flip"]
		if turns:
			_replay_header(log.num_players)
			# Turn 0: show starting hands (post-flip)
			if rs:
				hw = _hand_col_width(log.num_players)
				hand_cols = []
				for i in range(log.num_players):
					flip_ev = flips.get(i)
					if flip_ev and flip_ev.flip_decision and flip_ev.hand_after:
						h = format_showing_values(flip_ev.hand_after)
					elif i in rs.hands:
						h = format_showing_values(rs.hands[i])
					else:
						h = ""
					hand_cols.append(h.ljust(hw))
				t = "0".rjust(_COL_T)
				a = "".ljust(_COL_A)
				tbl = "".ljust(_COL_TBL)
				print(f"  {t}{_SEP}  {_SEP}{a}{_SEP}{tbl}{_SEP}{_SEP.join(hand_cols)}")
			for ev in turns:
				_print_turn_event(ev, log.num_players)
			print()

		re_ev = round_ends.get(round_num)
		if re_ev:
			scores = "  ".join(f"P{i}={s:+d}" for i, s in enumerate(re_ev.scores))
			ender = f" (P{re_ev.ender} went out)" if re_ev.ender is not None else ""
			print(f"  Scores: {scores}{ender}\n")

	if log.final_scores:
		print(f"{'=' * 130}")
		scores = "  ".join(f"P{i}={s}" for i, s in enumerate(log.final_scores))
		winner = log.final_scores.index(max(log.final_scores))
		print(f"  Final: {scores}  - P{winner} wins")
		print(f"{'=' * 130}")
