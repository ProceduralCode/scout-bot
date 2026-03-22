"""Probe environments for isolating Scout NN subsystems.
Each probe tests one axis of learning capability:
  1. Value head: can it learn a constant return?
  2. Play start: can a conditioned sub-head learn a preference? (control for #3)
  3. Play end: can the play_end head learn to prefer longer plays?
  4. Full chain: can start+end jointly learn to maximize play length?
  5. Scout insert: can the insert head learn to place cards for better plays?
  5b. Scout adjacent: can the insert head learn to place next to matching value?
  6. Action type: can the action_type head learn play vs scout preference?
  7. GAE multi-step: does credit assignment work across multiple decisions?

Usage: python probe.py [--iters N] [--games N] [--probe N]
"""
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from game import Game
from encoding import (
	encode_state, get_legal_plays, get_action_type_mask,
	get_play_start_mask, get_play_end_mask, get_scout_insert_mask,
	decode_slot_to_hand_index, decode_action_type,
	HAND_SLOTS, PLAY_SLOTS, PLAY_START_SIZE, PLAY_END_SIZE,
	ACTION_TYPE_SIZE, SCOUT_INSERT_SIZE,
	# V2
	encode_state_v2, HAND_SLOTS_V2, SCOUT_INSERT_SIZE_V2,
)
from network import ScoutNetwork, masked_sample
from training import StepRecord, compute_gae, prepare_ppo_batch, ppo_update

NUM_PLAYERS = 4
LAYER_SIZES = [64, 32]
LR = 3e-4
PPO_EPOCHS = 4
ENTROPY_BONUS = 0.01
CLIP_EPSILON = 0.2
VALUE_LOSS_COEFF = 0.25
ENTROPY_FLOORS = None
ENTROPY_FLOOR_COEFF = 1.0

# --- Helpers ---

def _fresh_round():
	"""Create a game at the start of a round (TURN phase, no current play)."""
	game = Game(NUM_PLAYERS)
	game.start_round()
	for p in range(NUM_PLAYERS):
		game.submit_flip_decision(p, do_flip=random.random() < 0.5)
	return game

def _encode(game, player=0):
	"""Encode game state with random offsets. Returns (state_tensor, hand_offset, play_offset)."""
	ho = random.randint(0, HAND_SLOTS - 1)
	po = random.randint(0, PLAY_SLOTS - 1) if game.current_play else 0
	return encode_state(game, player, ho, po), ho, po

def _group_plays_by_start(legal_plays):
	"""Group legal plays by start index. Returns {start: [end1, end2, ...]}."""
	groups = {}
	for s, e in legal_plays:
		groups.setdefault(s, []).append(e)
	return groups

def _find_multi_end_start(hand, legal_plays, min_ends=3):
	"""Find a start index with at least min_ends valid end positions.
	Returns (start_idx, [end_indices]) or None."""
	groups = _group_plays_by_start(legal_plays)
	for start, ends in groups.items():
		if len(ends) >= min_ends:
			return start, sorted(ends)
	return None

def _masked_probs(logits, mask):
	"""Compute masked softmax probabilities."""
	masked = logits.masked_fill(~mask, float('-inf'))
	return torch.softmax(masked, dim=-1)

def _sample_play(network, game, player=0, force_start_idx=None, force_end_to_start=False):
	"""Run network forward on a game state and sample a play action.
	Returns dict with all StepRecord fields (except reward/game_id) plus metadata,
	or None if the state doesn't meet constraints (e.g., force_start has no multi-end)."""
	hand = game.players[player].hand
	legal_plays = get_legal_plays(hand, game.current_play)
	if not legal_plays:
		return None

	state, ho, po = _encode(game, player)
	with torch.no_grad():
		hidden = network(state)
		value = network.value(hidden).item()

		# Action type — always play at round start (only legal option)
		at_logits = network.action_type_logits(hidden)
		at_mask = get_action_type_mask(game, legal_plays)
		action_type = 0  # play

		# Play start
		ps_logits = network.play_start_logits(hidden, action_type)
		if force_start_idx is not None:
			# Override mask to only allow the forced start
			ps_mask = np.zeros(PLAY_START_SIZE, dtype=np.bool_)
			ps_mask[(ho + force_start_idx) % HAND_SLOTS] = True
		else:
			ps_mask = get_play_start_mask(legal_plays, ho)
		start_slot, _ = masked_sample(ps_logits, torch.from_numpy(ps_mask))
		start_idx = decode_slot_to_hand_index(start_slot, ho)

		# Play end
		pe_logits = network.play_end_logits(hidden, action_type, start_slot)
		if force_end_to_start:
			pe_mask = np.zeros(PLAY_END_SIZE, dtype=np.bool_)
			pe_mask[start_slot] = True
		else:
			pe_mask = get_play_end_mask(legal_plays, start_idx, ho)
		if not pe_mask.any():
			return None
		end_slot, _ = masked_sample(pe_logits, torch.from_numpy(pe_mask))
		end_idx = decode_slot_to_hand_index(end_slot, ho)

	length = end_idx - start_idx + 1
	return {
		"state": state, "value": value,
		"at_logits": at_logits, "at_mask": at_mask, "action_type": action_type,
		"ps_logits": ps_logits, "ps_mask": ps_mask, "start_slot": start_slot,
		"pe_logits": pe_logits, "pe_mask": pe_mask, "end_slot": end_slot,
		"start_idx": start_idx, "end_idx": end_idx, "length": length,
		"ho": ho, "legal_plays": legal_plays, "hand": hand,
	}

def _make_record(sample, reward, game_id):
	"""Convert a _sample_play result into a StepRecord."""
	return StepRecord(
		state=sample["state"],
		action_type=sample["action_type"],
		action_type_logits=sample["at_logits"],
		action_type_mask=sample["at_mask"],
		play_start=sample["start_slot"],
		play_start_logits=sample["ps_logits"],
		play_start_mask=sample["ps_mask"],
		play_end=sample["end_slot"],
		play_end_logits=sample["pe_logits"],
		play_end_mask=sample["pe_mask"],
		value=sample["value"],
		reward=reward,
		player=0,
		round_num=0,
		game_id=game_id,
	)

def _train_iteration(network, optimizer, records, verbose=False,
					 entropy_floors="USE_GLOBAL", entropy_floor_coeff="USE_GLOBAL"):
	"""Run one training iteration: GAE + PPO epochs. Returns avg metrics."""
	if entropy_floors == "USE_GLOBAL":
		entropy_floors = ENTROPY_FLOORS
	if entropy_floor_coeff == "USE_GLOBAL":
		entropy_floor_coeff = ENTROPY_FLOOR_COEFF
	advantages, returns, adv_std = compute_gae(records, gamma=0.99, lam=0.95)
	if verbose:
		rewards = [r.reward for r in records]
		mean_r = sum(rewards) / len(rewards)
		print(f"    records={len(records)}  mean_reward={mean_r:+.3f}  adv_std={adv_std:.4f}  "
			  f"returns_range=[{min(returns):.3f}, {max(returns):.3f}]")
	batch = prepare_ppo_batch(records, advantages, returns=returns)
	ppo_sums = {}
	for _ in range(PPO_EPOCHS):
		network.train()
		m = ppo_update(
			network, optimizer, batch,
			clip_epsilon=CLIP_EPSILON, entropy_bonus=ENTROPY_BONUS,
			value_loss_coeff=VALUE_LOSS_COEFF,
			entropy_floors=entropy_floors,
			entropy_floor_coeff=entropy_floor_coeff,
		)
		for k, v in m.items():
			ppo_sums[k] = ppo_sums.get(k, 0.0) + v
	avg = {k: v / PPO_EPOCHS for k, v in ppo_sums.items()}
	if verbose:
		print(f"    ploss={avg['policy_loss']:.5f}  vloss={avg['value_loss']:.4f}  "
			  f"ent={avg['entropy']:.3f}  clip={avg['clip_fraction']:.3f}  kl={avg['approx_kl']:.5f}")
	return avg

# --- Eval utilities (importable by main.py) ---

def eval_scout_quality(network, n_samples=200):
	"""Measure scout placement quality: average length of the longest legal play
	(set or run) containing the inserted card. 1.0 = card never extends a play,
	2.0 = always creates pairs, etc. Returns (avg_length, n_samples_used)."""
	network.eval()
	lengths = []
	for _ in range(n_samples):
		game = _mid_round_state()
		if game is None:
			continue
		sample = _sample_scout(network, game)
		if sample is None:
			continue
		hand = list(sample["hand"])
		card = sample["card"]
		pos = sample["insert_pos"]
		new_hand = hand[:pos] + [card] + hand[pos:]
		plays = get_legal_plays(new_hand, None)
		max_len = 1
		for s, e in plays:
			if s <= pos <= e:
				max_len = max(max_len, e - s + 1)
		lengths.append(max_len)
	return sum(lengths) / max(len(lengths), 1), len(lengths)

# --- Probes ---

def probe_value(n_iters=100, n_games=50):
	"""Probe 1: Can the value head learn a constant return?
	Every state gets reward=+1.0, so the value head should converge to ~1.0."""
	network = ScoutNetwork(layer_sizes=LAYER_SIZES)
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)

	# Measure initial
	network.eval()
	init_vals = []
	for _ in range(n_games):
		game = _fresh_round()
		sample = _sample_play(network, game)
		if sample:
			init_vals.append(sample["value"])
	init_mean = sum(init_vals) / len(init_vals) if init_vals else 0.0

	# Train
	for it in range(n_iters):
		network.eval()
		records = []
		for g in range(n_games):
			game = _fresh_round()
			sample = _sample_play(network, game)
			if sample:
				records.append(_make_record(sample, reward=1.0, game_id=g))
		if records:
			_train_iteration(network, optimizer, records)

	# Measure final
	network.eval()
	final_vals = []
	for _ in range(n_games):
		game = _fresh_round()
		sample = _sample_play(network, game)
		if sample:
			final_vals.append(sample["value"])
	final_mean = sum(final_vals) / len(final_vals) if final_vals else 0.0

	passed = final_mean > 0.7
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 1 (value learning):    {status}  value {init_mean:.3f} -> {final_mean:.3f}  (target: >0.7)")
	return passed

def probe_play_start(n_iters=100, n_games=50):
	"""Probe 2: Can the play_start head learn a preference?
	Control for probe 3 — uses same conditioned-head mechanism.
	Two sub-tests:
	  2a: Reward a specific slot (pure memorization). If this fails, the head is broken.
	  2b: Reward the start with the most end options (generalization).
	Forces end=start (1-card play) so play_end has nothing to learn."""
	passed_a = _probe_play_start_fixed(n_iters, n_games)
	passed_b = _probe_play_start_best(n_iters, n_games)
	return passed_a

def _probe_play_start_fixed(n_iters, n_games):
	"""2a: Reward whichever legal start slot has the lowest index.
	The network just needs to learn 'pick the lowest slot' — trivial pattern."""
	network = ScoutNetwork(layer_sizes=LAYER_SIZES)
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)

	def target_start(legal_plays):
		return min(s for s, e in legal_plays)

	network.eval()
	n_hit = 0
	n_total = 0
	for _ in range(200):
		game = _fresh_round()
		sample = _sample_play(network, game, force_end_to_start=True)
		if sample:
			n_total += 1
			if sample["start_idx"] == target_start(sample["legal_plays"]):
				n_hit += 1
	init_rate = n_hit / max(n_total, 1)

	for it in range(n_iters):
		network.eval()
		records = []
		for g in range(n_games):
			game = _fresh_round()
			sample = _sample_play(network, game, force_end_to_start=True)
			if not sample:
				continue
			target = target_start(sample["legal_plays"])
			reward = 1.0 if sample["start_idx"] == target else -1.0
			records.append(_make_record(sample, reward=reward, game_id=g))
		if records:
			verbose = (it < 3 or it == n_iters - 1)
			if verbose:
				print(f"  [2a iter {it}]")
			_train_iteration(network, optimizer, records, verbose=verbose)

	network.eval()
	n_hit = 0
	n_total = 0
	for _ in range(200):
		game = _fresh_round()
		sample = _sample_play(network, game, force_end_to_start=True)
		if sample:
			n_total += 1
			if sample["start_idx"] == target_start(sample["legal_plays"]):
				n_hit += 1
	final_rate = n_hit / max(n_total, 1)

	passed = final_rate > init_rate + 0.05
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 2a (start fixed):      {status}  P(lowest_start) {init_rate:.3f} -> {final_rate:.3f}  (target: improvement)")
	return passed

def _probe_play_start_best(n_iters, n_games):
	"""2b: Reward the start with the most end options (harder generalization)."""
	network = ScoutNetwork(layer_sizes=LAYER_SIZES)
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)

	def best_start(legal_plays):
		groups = _group_plays_by_start(legal_plays)
		return max(groups.keys(), key=lambda s: len(groups[s]))

	network.eval()
	n_best = 0
	n_total = 0
	for _ in range(200):
		game = _fresh_round()
		sample = _sample_play(network, game, force_end_to_start=True)
		if sample:
			n_total += 1
			if sample["start_idx"] == best_start(sample["legal_plays"]):
				n_best += 1
	init_rate = n_best / max(n_total, 1)

	for it in range(n_iters):
		network.eval()
		records = []
		for g in range(n_games):
			game = _fresh_round()
			sample = _sample_play(network, game, force_end_to_start=True)
			if not sample:
				continue
			target = best_start(sample["legal_plays"])
			reward = 1.0 if sample["start_idx"] == target else -1.0
			records.append(_make_record(sample, reward=reward, game_id=g))
		if records:
			_train_iteration(network, optimizer, records)

	network.eval()
	n_best = 0
	n_total = 0
	for _ in range(200):
		game = _fresh_round()
		sample = _sample_play(network, game, force_end_to_start=True)
		if sample:
			n_total += 1
			if sample["start_idx"] == best_start(sample["legal_plays"]):
				n_best += 1
	final_rate = n_best / max(n_total, 1)

	passed = final_rate > init_rate + 0.05
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 2b (start best-ends):  {status}  P(best_start) {init_rate:.3f} -> {final_rate:.3f}  (target: improvement)")
	return passed

def probe_play_end(n_iters=100, n_games=50):
	"""Probe 3: Can the play_end head learn to prefer longer plays?
	Forces start to a position with 3+ valid ends.
	Rewards the longest available play."""
	network = ScoutNetwork(layer_sizes=LAYER_SIZES)
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)

	def find_state_and_start():
		"""Keep generating games until we find one with a multi-end start."""
		for _ in range(100):
			game = _fresh_round()
			hand = game.players[0].hand
			legal_plays = get_legal_plays(hand, game.current_play)
			result = _find_multi_end_start(hand, legal_plays, min_ends=3)
			if result:
				return game, result[0], result[1]
		return None, None, None

	# Measure initial: P(longest_end)
	network.eval()
	n_longest = 0
	n_total = 0
	for _ in range(200):
		game, forced_start, ends = find_state_and_start()
		if game is None:
			continue
		longest_end = max(ends)
		sample = _sample_play(network, game, force_start_idx=forced_start)
		if sample:
			n_total += 1
			if sample["end_idx"] == longest_end:
				n_longest += 1
	init_rate = n_longest / max(n_total, 1)

	# Train
	for it in range(n_iters):
		network.eval()
		records = []
		for g in range(n_games):
			game, forced_start, ends = find_state_and_start()
			if game is None:
				continue
			sample = _sample_play(network, game, force_start_idx=forced_start)
			if not sample:
				continue
			longest_end = max(ends)
			reward = 1.0 if sample["end_idx"] == longest_end else -1.0
			records.append(_make_record(sample, reward=reward, game_id=g))
		if records:
			verbose = (it < 3 or it == n_iters - 1)
			if verbose:
				print(f"  [iter {it}]")
			_train_iteration(network, optimizer, records, verbose=verbose)

	# Measure final
	network.eval()
	n_longest = 0
	n_total = 0
	for _ in range(200):
		game, forced_start, ends = find_state_and_start()
		if game is None:
			continue
		longest_end = max(ends)
		sample = _sample_play(network, game, force_start_idx=forced_start)
		if sample:
			n_total += 1
			if sample["end_idx"] == longest_end:
				n_longest += 1
	final_rate = n_longest / max(n_total, 1)

	passed = final_rate > init_rate + 0.05
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 3 (play_end longest):  {status}  P(longest_end) {init_rate:.3f} -> {final_rate:.3f}  (target: improvement)")
	return passed

def probe_full_chain(n_iters=100, n_games=50):
	"""Probe 4: Can start+end jointly learn to maximize play length?
	No forced actions — both heads free. Reward proportional to play length."""
	network = ScoutNetwork(layer_sizes=LAYER_SIZES)
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)

	# Measure initial mean play length
	network.eval()
	init_lengths = []
	for _ in range(200):
		game = _fresh_round()
		sample = _sample_play(network, game)
		if sample:
			init_lengths.append(sample["length"])
	init_mean = sum(init_lengths) / len(init_lengths) if init_lengths else 1.0

	# Train
	for it in range(n_iters):
		network.eval()
		records = []
		for g in range(n_games):
			game = _fresh_round()
			sample = _sample_play(network, game)
			if not sample:
				continue
			# Reward scales with length: 1-card=−1, longer=better
			reward = (sample["length"] - 1) * 0.5 - 0.5
			records.append(_make_record(sample, reward=reward, game_id=g))
		if records:
			_train_iteration(network, optimizer, records)

	# Measure final
	network.eval()
	final_lengths = []
	for _ in range(200):
		game = _fresh_round()
		sample = _sample_play(network, game)
		if sample:
			final_lengths.append(sample["length"])
	final_mean = sum(final_lengths) / len(final_lengths) if final_lengths else 1.0

	passed = final_mean > init_mean + 0.1
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 4 (full chain length): {status}  mean_length {init_mean:.3f} -> {final_mean:.3f}  (target: improvement)")
	return passed

def _mid_round_state():
	"""Create a game state where player 1 can scout (current play exists on the table)."""
	for _ in range(100):
		game = Game(NUM_PLAYERS)
		game.start_round()
		for p in range(NUM_PLAYERS):
			game.submit_flip_decision(p, do_flip=random.random() < 0.5)
		# Player 0 makes a play to establish current_play
		hand = game.players[0].hand
		legal_plays = get_legal_plays(hand, game.current_play)
		if legal_plays:
			start, end = random.choice(legal_plays)
			game.apply_play(start, end)
			return game
	return None

def _hand_quality(hand):
	"""Score a hand by its longest available play (no opponent play to beat)."""
	plays = get_legal_plays(hand, None)
	if not plays:
		return 0
	return max(e - s + 1 for s, e in plays)

def _sample_scout(network, game, player=1):
	"""Run network on a scoutable state and sample a scout insert position.
	Always scouts from left end, no flip (action_type=1).
	Returns dict with fields for building a StepRecord, or None."""
	hand = game.players[player].hand
	play_cards = game.current_play.cards
	if len(play_cards) == 0:
		return None
	card = play_cards[0]  # left end, no flip
	action_type = 1  # scout-left-normal
	ev = getattr(network, 'encoding_version', 1)
	if ev == 2:
		_hs = HAND_SLOTS_V2
		_sis = SCOUT_INSERT_SIZE_V2
		ho = random.randint(0, _hs - 1)
		state = encode_state_v2(game, player, ho)
	else:
		_hs = HAND_SLOTS
		_sis = SCOUT_INSERT_SIZE
		state, ho, po = _encode(game, player)
	with torch.no_grad():
		hidden = network(state)
		value = network.value(hidden).item()
		at_logits = network.action_type_logits(hidden)
		legal_plays = get_legal_plays(hand, game.current_play)
		at_mask = get_action_type_mask(game, legal_plays, max_hand=_hs)
		si_logits = network.scout_insert_logits(hidden, action_type)
		si_mask = get_scout_insert_mask(game, ho, num_slots=_sis)
		if not si_mask.any():
			return None
		insert_slot, _ = masked_sample(si_logits, torch.from_numpy(si_mask))
		insert_pos = (insert_slot - ho) % _sis
	return {
		"state": state, "value": value,
		"at_logits": at_logits, "at_mask": at_mask, "action_type": action_type,
		"si_logits": si_logits, "si_mask": si_mask,
		"insert_slot": insert_slot, "insert_pos": insert_pos,
		"card": card, "hand": list(hand),
	}

def _make_scout_record(sample, reward, game_id):
	"""Convert a _sample_scout result into a StepRecord."""
	return StepRecord(
		state=sample["state"],
		action_type=sample["action_type"],
		action_type_logits=sample["at_logits"],
		action_type_mask=sample["at_mask"],
		scout_insert=sample["insert_slot"],
		scout_insert_logits=sample["si_logits"],
		scout_insert_mask=sample["si_mask"],
		value=sample["value"],
		reward=reward,
		player=1,
		round_num=0,
		game_id=game_id,
	)

def probe_scout_insert(n_iters=100, n_games=50):
	"""Probe 5: Can the scout_insert head learn to place cards where they create better plays?
	Forces a scout action with a locked action_type mask (no gradient noise from at head).
	Continuous reward based on hand quality at chosen position vs best/worst possible."""
	network = ScoutNetwork(layer_sizes=LAYER_SIZES)
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)

	def insert_qualities(hand, card):
		"""Returns list of (pos, quality) and (min_q, max_q)."""
		results = []
		for pos in range(len(hand) + 1):
			new_hand = hand[:pos] + [card] + hand[pos:]
			q = _hand_quality(new_hand)
			results.append((pos, q))
		qs = [q for _, q in results]
		return results, min(qs), max(qs)

	def _eval_quality(network, n_samples=200):
		"""Measure mean quality of chosen insert positions vs max possible."""
		total_q, max_q_sum, n = 0, 0, 0
		for _ in range(n_samples):
			game = _mid_round_state()
			if game is None:
				continue
			sample = _sample_scout(network, game)
			if sample is None:
				continue
			chosen_hand = sample["hand"][:sample["insert_pos"]] + [sample["card"]] + sample["hand"][sample["insert_pos"]:]
			chosen_q = _hand_quality(chosen_hand)
			_, _, max_q = insert_qualities(sample["hand"], sample["card"])
			total_q += chosen_q
			max_q_sum += max_q
			n += 1
		if n == 0:
			return 0.0, 0.0
		return total_q / n, max_q_sum / n

	# Measure initial quality
	network.eval()
	init_q, init_max = _eval_quality(network)

	# Train
	for it in range(n_iters):
		network.eval()
		records = []
		for g in range(n_games):
			game = _mid_round_state()
			if game is None:
				continue
			sample = _sample_scout(network, game)
			if sample is None:
				continue
			# Lock action_type mask to only allow scout — no gradient noise from at head
			forced_at_mask = np.zeros(ACTION_TYPE_SIZE, dtype=np.bool_)
			forced_at_mask[sample["action_type"]] = True
			sample["at_mask"] = forced_at_mask
			# Continuous reward based on quality relative to best/worst
			_, min_q, max_q = insert_qualities(sample["hand"], sample["card"])
			chosen_hand = sample["hand"][:sample["insert_pos"]] + [sample["card"]] + sample["hand"][sample["insert_pos"]:]
			chosen_q = _hand_quality(chosen_hand)
			if max_q == min_q:
				reward = 0.0  # all positions equal
			else:
				reward = (chosen_q - min_q) / (max_q - min_q) * 2.0 - 1.0
			records.append(_make_scout_record(sample, reward=reward, game_id=g))
		if records:
			verbose = (it < 3 or it == n_iters - 1)
			if verbose:
				print(f"  [5 iter {it}]")
			_train_iteration(network, optimizer, records, verbose=verbose)

	# Measure final quality
	network.eval()
	final_q, final_max = _eval_quality(network)

	passed = final_q > init_q + 0.1
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 5 (scout insert):      {status}  chosen_quality {init_q:.2f} -> {final_q:.2f}  "
		  f"(max_possible={final_max:.2f}, target: improvement)")
	return passed

def probe_scout_adjacent(n_iters=100, n_games=50):
	"""Probe 5b: Can the scout_insert head learn to place a card next to a matching value?
	Simpler than probe 5 — tests whether the head can extract card values from the
	encoding and match them, rather than optimizing full hand quality.
	Only uses states where the hand contains at least one card matching the scouted card's value."""
	network = ScoutNetwork(layer_sizes=LAYER_SIZES)
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)

	def _is_adjacent_match(hand, card, insert_pos):
		"""Is the inserted card next to a card with the same showing value?"""
		val = card[0]
		if insert_pos > 0 and hand[insert_pos - 1][0] == val:
			return True
		if insert_pos < len(hand) and hand[insert_pos][0] == val:
			return True
		return False

	def _has_any_match(hand, card):
		return any(c[0] == card[0] for c in hand)

	def _eval_rate(network, n_samples=200):
		"""Measure P(adjacent_match) over scoutable states with a match available."""
		n_adj, n_total = 0, 0
		for _ in range(n_samples):
			game = _mid_round_state()
			if game is None:
				continue
			sample = _sample_scout(network, game)
			if sample is None:
				continue
			if not _has_any_match(sample["hand"], sample["card"]):
				continue
			n_total += 1
			if _is_adjacent_match(sample["hand"], sample["card"], sample["insert_pos"]):
				n_adj += 1
		return n_adj / max(n_total, 1)

	# Measure initial
	network.eval()
	init_rate = _eval_rate(network)

	# Train
	for it in range(n_iters):
		network.eval()
		records = []
		for g in range(n_games):
			game = _mid_round_state()
			if game is None:
				continue
			sample = _sample_scout(network, game)
			if sample is None:
				continue
			if not _has_any_match(sample["hand"], sample["card"]):
				continue
			# Lock action_type mask
			forced_at_mask = np.zeros(ACTION_TYPE_SIZE, dtype=np.bool_)
			forced_at_mask[sample["action_type"]] = True
			sample["at_mask"] = forced_at_mask
			adj = _is_adjacent_match(sample["hand"], sample["card"], sample["insert_pos"])
			reward = 1.0 if adj else -1.0
			records.append(_make_scout_record(sample, reward=reward, game_id=g))
		if records:
			verbose = (it < 3 or it == n_iters - 1)
			if verbose:
				print(f"  [5b iter {it}]")
			_train_iteration(network, optimizer, records, verbose=verbose)

	# Measure final
	network.eval()
	final_rate = _eval_rate(network)

	passed = final_rate > init_rate + 0.05
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 5b (scout adjacent):   {status}  P(adj_match) {init_rate:.3f} -> {final_rate:.3f}  (target: improvement)")
	return passed

def _sample_action_type(network, game, player=1):
	"""Run network on a mid-round state and sample an action type.
	Returns dict with action type info, or None."""
	hand = game.players[player].hand
	legal_plays = get_legal_plays(hand, game.current_play)

	state, ho, po = _encode(game, player)
	with torch.no_grad():
		hidden = network(state)
		value = network.value(hidden).item()

		at_logits = network.action_type_logits(hidden)
		at_mask = get_action_type_mask(game, legal_plays)
		if not at_mask.any():
			return None
		action_type, _ = masked_sample(at_logits, torch.from_numpy(at_mask))

	action_info = decode_action_type(action_type)
	return {
		"state": state, "value": value,
		"at_logits": at_logits, "at_mask": at_mask, "action_type": action_type,
		"action_info": action_info, "has_legal_play": len(legal_plays) > 0,
	}

def _make_action_type_record(sample, reward, game_id):
	"""Convert a _sample_action_type result into a StepRecord."""
	return StepRecord(
		state=sample["state"],
		action_type=sample["action_type"],
		action_type_logits=sample["at_logits"],
		action_type_mask=sample["at_mask"],
		value=sample["value"],
		reward=reward,
		player=1,
		round_num=0,
		game_id=game_id,
	)

def probe_action_type(n_iters=100, n_games=50):
	"""Probe 6: Can the action_type head learn to prefer play over scout?
	Mid-round states where both play and scout are legal.
	Reward +1 for playing, -1 for scouting. Simple preference test."""
	network = ScoutNetwork(layer_sizes=LAYER_SIZES)
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)

	# Measure initial play rate
	network.eval()
	n_play, n_total = 0, 0
	for _ in range(200):
		game = _mid_round_state()
		if game is None:
			continue
		sample = _sample_action_type(network, game)
		if sample is None:
			continue
		n_total += 1
		if sample["action_info"]["type"] == "play":
			n_play += 1
	init_rate = n_play / max(n_total, 1)

	# Train
	for it in range(n_iters):
		network.eval()
		records = []
		for g in range(n_games):
			game = _mid_round_state()
			if game is None:
				continue
			sample = _sample_action_type(network, game)
			if sample is None:
				continue
			reward = 1.0 if sample["action_info"]["type"] == "play" else -1.0
			records.append(_make_action_type_record(sample, reward=reward, game_id=g))
		if records:
			verbose = (it < 3 or it == n_iters - 1)
			if verbose:
				print(f"  [6 iter {it}]")
			_train_iteration(network, optimizer, records, verbose=verbose)

	# Measure final play rate
	network.eval()
	n_play, n_total = 0, 0
	for _ in range(200):
		game = _mid_round_state()
		if game is None:
			continue
		sample = _sample_action_type(network, game)
		if sample is None:
			continue
		n_total += 1
		if sample["action_info"]["type"] == "play":
			n_play += 1
	final_rate = n_play / max(n_total, 1)

	passed = final_rate > init_rate + 0.05
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 6 (action type play):  {status}  P(play) {init_rate:.3f} -> {final_rate:.3f}  (target: improvement)")
	return passed

def probe_gae_multistep(n_iters=100, n_games=50):
	"""Probe 7: Does credit assignment work across multiple sequential decisions?
	Simulates a 3-step episode per game: three play decisions from fresh states,
	grouped as one sequence. Only the last step gets reward (terminal).
	If GAE works, the value head should learn to predict the eventual outcome
	from earlier states, and policy improvement should propagate backward.

	Measures whether the value head learns to predict returns from step 1 and 2
	(not just step 3), which requires GAE bootstrapping to work."""
	network = ScoutNetwork(layer_sizes=LAYER_SIZES)
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)
	STEPS_PER_EP = 3

	def _collect_episode(network, game_id):
		"""Collect a 3-step episode with terminal-only reward."""
		records = []
		total_length = 0
		for step in range(STEPS_PER_EP):
			game = _fresh_round()
			sample = _sample_play(network, game)
			if sample is None:
				return None
			total_length += sample["length"]
			rec = _make_record(sample, reward=0.0, game_id=game_id)
			rec.round_num = 0  # same "round" so GAE groups them
			records.append(rec)
		# Terminal reward on last step only: reward = total play length
		records[-1].reward = (total_length - STEPS_PER_EP) * 0.5
		return records

	# Measure initial: how well does V predict step-1 returns?
	network.eval()
	init_v1_errors = []
	for _ in range(200):
		eps = _collect_episode(network, 0)
		if eps:
			# True return for step 1 ≈ terminal reward (undiscounted for simplicity)
			init_v1_errors.append(abs(eps[0].value - eps[-1].reward))
	init_v1_err = sum(init_v1_errors) / max(len(init_v1_errors), 1)

	# Train
	for it in range(n_iters):
		network.eval()
		all_records = []
		for g in range(n_games):
			eps = _collect_episode(network, g)
			if eps:
				all_records.extend(eps)
		if all_records:
			verbose = (it < 3 or it == n_iters - 1)
			if verbose:
				print(f"  [7 iter {it}]")
			_train_iteration(network, optimizer, all_records, verbose=verbose)

	# Measure final: V prediction error at step 1
	network.eval()
	final_v1_errors = []
	terminal_rewards = []
	for _ in range(200):
		eps = _collect_episode(network, 0)
		if eps:
			final_v1_errors.append(abs(eps[0].value - eps[-1].reward))
			terminal_rewards.append(eps[-1].reward)
	final_v1_err = sum(final_v1_errors) / max(len(final_v1_errors), 1)
	avg_terminal = sum(terminal_rewards) / max(len(terminal_rewards), 1)

	# Pass if V's step-1 prediction error decreased meaningfully
	passed = final_v1_err < init_v1_err - 0.1
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 7 (GAE multi-step):    {status}  V_step1_error {init_v1_err:.3f} -> {final_v1_err:.3f}  "
		  f"avg_terminal={avg_terminal:.2f}  (target: error decreases)")
	return passed

def probe_frozen_trunk_scout(n_iters=100, n_games=50):
	"""Probe 8: Load trained v3_4 checkpoint, freeze shared trunk, train only scout_insert_head.
	Tests whether the trained trunk's features support scout insertion learning.
	If this passes: the trunk has useful features but the scout signal is too weak in full training.
	If this fails: the trunk doesn't encode what the scout head needs."""
	import os
	# Try both relative paths (from scout-bot/ and from workspace root)
	ckpt_path = os.path.join("v3_4", "latest.pt")
	if not os.path.exists(ckpt_path):
		ckpt_path = os.path.join("scout-bot", "v3_4", "latest.pt")
	if not os.path.exists(ckpt_path):
		print(f"  Probe 8 (frozen trunk):      SKIP  (no checkpoint at {ckpt_path})")
		return False

	checkpoint = torch.load(ckpt_path, weights_only=False)
	ckpt_cfg = checkpoint.get("config", {})
	layer_sizes = ckpt_cfg.get("layer_sizes", [256, 128])

	network = ScoutNetwork(layer_sizes=layer_sizes)
	network.load_state_dict(checkpoint["model_state"])

	# Freeze shared trunk — only train scout_insert_head
	for param in network.shared.parameters():
		param.requires_grad = False
	for param in network.value_head.parameters():
		param.requires_grad = False
	for param in network.action_type_head.parameters():
		param.requires_grad = False
	for param in network.play_start_head.parameters():
		param.requires_grad = False
	for param in network.play_end_head.parameters():
		param.requires_grad = False
	# Reinitialize scout_insert_head to test learning from scratch
	torch.nn.init.xavier_uniform_(network.scout_insert_head.weight)
	torch.nn.init.zeros_(network.scout_insert_head.bias)

	# Only optimize the scout head
	optimizer = torch.optim.Adam(network.scout_insert_head.parameters(), lr=LR)

	def _eval_adj_rate(network, n_samples=500):
		"""Measure P(adjacent_match) over scoutable states with a match available."""
		n_adj, n_total = 0, 0
		for _ in range(n_samples):
			game = _mid_round_state()
			if game is None:
				continue
			sample = _sample_scout(network, game)
			if sample is None:
				continue
			hand = sample["hand"]
			card = sample["card"]
			if not any(c[0] == card[0] for c in hand):
				continue
			n_total += 1
			pos = sample["insert_pos"]
			val = card[0]
			if (pos > 0 and hand[pos - 1][0] == val) or (pos < len(hand) and hand[pos][0] == val):
				n_adj += 1
		return n_adj / max(n_total, 1)

	# Measure initial quality with trained trunk + fresh head
	network.eval()
	init_rate = _eval_adj_rate(network)

	# Train
	for it in range(n_iters):
		network.eval()
		records = []
		for g in range(n_games):
			game = _mid_round_state()
			if game is None:
				continue
			sample = _sample_scout(network, game)
			if sample is None:
				continue
			if not any(c[0] == sample["card"][0] for c in sample["hand"]):
				continue
			forced_at_mask = np.zeros(ACTION_TYPE_SIZE, dtype=np.bool_)
			forced_at_mask[sample["action_type"]] = True
			sample["at_mask"] = forced_at_mask
			pos = sample["insert_pos"]
			val = sample["card"][0]
			hand = sample["hand"]
			adj = ((pos > 0 and hand[pos - 1][0] == val)
				   or (pos < len(hand) and hand[pos][0] == val))
			reward = 1.0 if adj else -1.0
			records.append(_make_scout_record(sample, reward=reward, game_id=g))
		if records:
			verbose = (it < 3 or it == n_iters - 1)
			if verbose:
				print(f"  [8 iter {it}]")
			# Use _train_iteration but with our optimizer
			advantages, returns, adv_std = compute_gae(records, gamma=0.99, lam=0.95)
			if verbose:
				rewards = [r.reward for r in records]
				mean_r = sum(rewards) / len(rewards)
				print(f"    records={len(records)}  mean_reward={mean_r:+.3f}  adv_std={adv_std:.4f}")
			batch = prepare_ppo_batch(records, advantages, returns=returns)
			for _ in range(PPO_EPOCHS):
				network.train()
				# Forward through frozen trunk + trainable head
				ppo_update(
					network, optimizer, batch,
					clip_epsilon=CLIP_EPSILON, entropy_bonus=ENTROPY_BONUS,
					value_loss_coeff=0.0,  # don't train frozen value head
					entropy_floors=ENTROPY_FLOORS,
					entropy_floor_coeff=ENTROPY_FLOOR_COEFF,
				)

	# Measure final
	network.eval()
	final_rate = _eval_adj_rate(network)

	passed = final_rate > init_rate + 0.05
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 8 (frozen trunk):      {status}  P(adj_match) {init_rate:.3f} -> {final_rate:.3f}  (target: improvement)")
	return passed

def probe_trivial_scout(n_iters=100, n_games=50):
	"""Probe 9: Can the scout head learn to always insert at position 0?
	Trivially easy control test — if this fails, the training mechanics are broken."""
	network = ScoutNetwork(layer_sizes=LAYER_SIZES)
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)

	def _eval_pos0_rate(network, n_samples=500):
		n_pos0, n_total = 0, 0
		for _ in range(n_samples):
			game = _mid_round_state()
			if game is None:
				continue
			sample = _sample_scout(network, game)
			if sample is None:
				continue
			n_total += 1
			if sample["insert_pos"] == 0:
				n_pos0 += 1
		return n_pos0 / max(n_total, 1)

	network.eval()
	init_rate = _eval_pos0_rate(network)

	for it in range(n_iters):
		network.eval()
		records = []
		for g in range(n_games):
			game = _mid_round_state()
			if game is None:
				continue
			sample = _sample_scout(network, game)
			if sample is None:
				continue
			forced_at_mask = np.zeros(ACTION_TYPE_SIZE, dtype=np.bool_)
			forced_at_mask[sample["action_type"]] = True
			sample["at_mask"] = forced_at_mask
			reward = 1.0 if sample["insert_pos"] == 0 else -1.0
			records.append(_make_scout_record(sample, reward=reward, game_id=g))
		if records:
			verbose = (it < 3 or it == n_iters - 1)
			if verbose:
				print(f"  [9 iter {it}]")
			_train_iteration(network, optimizer, records, verbose=verbose)

	network.eval()
	final_rate = _eval_pos0_rate(network)

	passed = final_rate > init_rate + 0.1
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 9 (trivial pos=0):     {status}  P(pos=0) {init_rate:.3f} -> {final_rate:.3f}  (target: improvement)")
	return passed

# --- Main ---

ALL_PROBES = {
	1: ("value", probe_value),
	2: ("play_start", probe_play_start),
	3: ("play_end", probe_play_end),
	4: ("full_chain", probe_full_chain),
	5: ("scout_insert", probe_scout_insert),
	55: ("scout_adjacent", probe_scout_adjacent),
	6: ("action_type", probe_action_type),
	7: ("gae_multistep", probe_gae_multistep),
	8: ("frozen_trunk_scout", probe_frozen_trunk_scout),
	9: ("trivial_scout", probe_trivial_scout),
}

def main():
	import argparse
	parser = argparse.ArgumentParser(description="Probe environments for Scout NN debugging")
	parser.add_argument("--iters", type=int, default=100, help="Training iterations per probe")
	parser.add_argument("--games", type=int, default=50, help="Games per iteration")
	parser.add_argument("--probe", type=int, nargs="*", default=None,
		help="Run specific probe(s) by number (e.g., --probe 5 6 7). Default: run all.")
	parser.add_argument("--layers", type=int, nargs="+", default=None,
		help="Override network layer sizes (e.g., --layers 512 256 256 128 128 128)")
	parser.add_argument("--entropy-floors", action="store_true",
		help="Enable entropy floors (same values as main.py training config)")
	args = parser.parse_args()

	global LAYER_SIZES, ENTROPY_FLOORS, ENTROPY_FLOOR_COEFF
	if args.layers:
		LAYER_SIZES = args.layers
	if args.entropy_floors:
		ENTROPY_FLOORS = {
			"action_type": 0.1,
			"play_start": 0.1,
			"play_end": 0.3,
			"scout_insert": 0.1,
		}
		ENTROPY_FLOOR_COEFF = 1.0
	probe_nums = args.probe if args.probe else sorted(ALL_PROBES.keys())
	print(f"Running probes {probe_nums} (iters={args.iters}, games={args.games}, layers={LAYER_SIZES})")
	print()
	results = []
	for num in probe_nums:
		if num not in ALL_PROBES:
			print(f"  Unknown probe {num}, skipping")
			continue
		name, fn = ALL_PROBES[num]
		results.append((name, fn(args.iters, args.games)))
	print()
	passed = sum(1 for _, p in results if p)
	print(f"Results: {passed}/{len(results)} passed")
	for name, p in results:
		print(f"  {'PASS' if p else 'FAIL'}  {name}")

if __name__ == "__main__":
	main()
