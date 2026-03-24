"""PPO variant tests to isolate why adjacent matching fails.

Each test targets a specific alternative explanation:
  graded:      Shaped reward (distance to nearest match) instead of binary +1/-1
  fixed_val:   "Insert adjacent to any card with value=5" (no need to read scouted card)
  big_net:     Adjacent matching with production-size [512,256,256,128,128,128] network
  hint:        Append scouted-card-value one-hot directly to scout head input

Usage: python scout-bot/probe_ppo_variants.py [--test graded|fixed_val|big_net|hint|all]
       [--iters N] [--games N] [--v2]
"""
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from game import Game
from encoding import (
	encode_state, get_legal_plays, get_action_type_mask,
	get_scout_insert_mask,
	HAND_SLOTS, PLAY_SLOTS, ACTION_TYPE_SIZE, SCOUT_INSERT_SIZE,
	CARD_VALUES, INPUT_SIZE,
	encode_state_v2, HAND_SLOTS_V2, SCOUT_INSERT_SIZE_V2,
	PLAY_START_SIZE, PLAY_START_SIZE_V2, INPUT_SIZE_V2,
	# V3
	encode_state_v3, HAND_SLOTS_V3, SCOUT_INSERT_SIZE_V3,
	PLAY_START_SIZE_V3, INPUT_SIZE_V3, PLAY_BUFFER_SLOTS_V3,
)
from network import ScoutNetwork, masked_sample
from training import StepRecord, compute_gae, prepare_ppo_batch, ppo_update

NUM_PLAYERS = 4
LR = 3e-4
PPO_EPOCHS = 4
ENTROPY_BONUS = 0.01
CLIP_EPSILON = 0.2
VALUE_LOSS_COEFF = 0.25
ENCODING_VERSION = 1

def _ev_hand_slots():
	if ENCODING_VERSION == 3: return HAND_SLOTS_V3
	if ENCODING_VERSION == 2: return HAND_SLOTS_V2
	return HAND_SLOTS

def _ev_insert_size():
	if ENCODING_VERSION == 3: return SCOUT_INSERT_SIZE_V3
	if ENCODING_VERSION == 2: return SCOUT_INSERT_SIZE_V2
	return SCOUT_INSERT_SIZE

def _ev_input_size():
	if ENCODING_VERSION == 3: return INPUT_SIZE_V3
	if ENCODING_VERSION == 2: return INPUT_SIZE_V2
	return INPUT_SIZE

def _make_network(layer_sizes=None):
	ls = layer_sizes or [64, 32]
	if ENCODING_VERSION == 3:
		return ScoutNetwork(INPUT_SIZE_V3, ls, encoding_version=3,
			play_start_size=PLAY_START_SIZE_V3, play_end_size=PLAY_START_SIZE_V3,
			scout_insert_size=SCOUT_INSERT_SIZE_V3)
	if ENCODING_VERSION == 2:
		return ScoutNetwork(INPUT_SIZE_V2, ls, encoding_version=2,
			play_start_size=PLAY_START_SIZE_V2, play_end_size=PLAY_START_SIZE_V2,
			scout_insert_size=SCOUT_INSERT_SIZE_V2)
	return ScoutNetwork(layer_sizes=ls)

def _mid_round_state():
	for _ in range(100):
		game = Game(NUM_PLAYERS)
		game.start_round()
		for p in range(NUM_PLAYERS):
			game.submit_flip_decision(p, do_flip=random.random() < 0.5)
		hand = game.players[0].hand
		legal_plays = get_legal_plays(hand, game.current_play)
		if legal_plays:
			start, end = random.choice(legal_plays)
			game.apply_play(start, end)
			return game
	return None

def _encode(game, player=1):
	if ENCODING_VERSION == 3:
		ho = random.randint(0, HAND_SLOTS_V3 - 1)
		po = random.randint(0, PLAY_BUFFER_SLOTS_V3 - 1) if game.current_play else 0
		return encode_state_v3(game, player, ho, po), ho, po
	if ENCODING_VERSION == 2:
		ho = random.randint(0, HAND_SLOTS_V2 - 1)
		return encode_state_v2(game, player, ho), ho, 0
	ho = random.randint(0, HAND_SLOTS - 1)
	po = random.randint(0, PLAY_SLOTS - 1) if game.current_play else 0
	return encode_state(game, player, ho, po), ho, po

def _sample_scout(network, game, player=1):
	_sis = _ev_insert_size()
	_hs = _ev_hand_slots()
	hand = game.players[player].hand
	play_cards = game.current_play.cards
	if len(play_cards) == 0:
		return None
	card = play_cards[0]
	action_type = 1
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
		"state": state, "value": value, "hidden": hidden,
		"at_logits": at_logits, "at_mask": at_mask, "action_type": action_type,
		"si_logits": si_logits, "si_mask": si_mask,
		"insert_slot": insert_slot, "insert_pos": insert_pos,
		"card": card, "hand": list(hand), "ho": ho,
	}

def _make_scout_record(sample, reward, game_id):
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
		player=1, round_num=0, game_id=game_id,
	)

def _is_adjacent_match(hand, card, insert_pos):
	val = card[0]
	if insert_pos > 0 and hand[insert_pos - 1][0] == val:
		return True
	if insert_pos < len(hand) and hand[insert_pos][0] == val:
		return True
	return False

def _has_any_match(hand, card):
	return any(c[0] == card[0] for c in hand)

def _train_iteration(network, optimizer, records, verbose=False):
	advantages, returns, adv_std = compute_gae(records, gamma=0.99, lam=0.95)
	if verbose:
		rewards = [r.reward for r in records]
		mean_r = sum(rewards) / len(rewards)
		print(f"    records={len(records)}  mean_reward={mean_r:+.3f}  adv_std={adv_std:.4f}")
	batch = prepare_ppo_batch(records, advantages, returns=returns)
	_pss = PLAY_START_SIZE_V3 if ENCODING_VERSION == 3 else (PLAY_START_SIZE_V2 if ENCODING_VERSION == 2 else PLAY_START_SIZE)
	for _ in range(PPO_EPOCHS):
		network.train()
		ppo_update(
			network, optimizer, batch,
			clip_epsilon=CLIP_EPSILON, entropy_bonus=ENTROPY_BONUS,
			value_loss_coeff=VALUE_LOSS_COEFF,
			play_start_size=_pss,
		)

def _eval_adj_rate(network, n_samples=500):
	network.eval()
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
	return n_adj / max(n_total, 1), n_total

# ============================================================
# Test: Graded reward (distance to nearest match)
# ============================================================

def test_graded_reward(n_iters=300, n_games=200):
	"""Instead of binary +1/-1, reward = -min_distance_to_nearest_matching_card.
	If the issue is binary reward sparsity, graded reward should help PPO
	by providing gradient direction (closer = better)."""
	print(f"\n=== Graded reward (v{ENCODING_VERSION}) ===")
	network = _make_network()
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)

	def _graded_reward(hand, card, insert_pos):
		"""Reward based on distance to nearest matching card. 1.0 = adjacent, decays with distance."""
		val = card[0]
		match_positions = [i for i, c in enumerate(hand) if c[0] == val]
		if not match_positions:
			return 0.0
		# Distance from insert_pos to nearest match (accounting for insert shifting)
		min_dist = min(
			min(abs(insert_pos - mp), abs(insert_pos - mp - 1))
			for mp in match_positions
		)
		if min_dist == 0:
			return 1.0
		# Decay: 1.0 at distance 0, approaches -1.0 at large distance
		return 1.0 - (2.0 * min_dist / len(hand))

	init_rate, _ = _eval_adj_rate(network)
	print(f"  Initial adj rate: {init_rate:.3f}")
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
			forced_at_mask = np.zeros(ACTION_TYPE_SIZE, dtype=np.bool_)
			forced_at_mask[sample["action_type"]] = True
			sample["at_mask"] = forced_at_mask
			reward = _graded_reward(sample["hand"], sample["card"], sample["insert_pos"])
			records.append(_make_scout_record(sample, reward=reward, game_id=g))
		if records:
			verbose = (it % 50 == 0 or it == n_iters - 1)
			if verbose:
				print(f"  [iter {it}]")
			_train_iteration(network, optimizer, records, verbose=verbose)
			if verbose:
				rate, n = _eval_adj_rate(network)
				print(f"  adj_rate={rate:.3f} (n={n})")
	final_rate, _ = _eval_adj_rate(network)
	passed = final_rate > init_rate + 0.05
	print(f"  RESULT: {'PASS' if passed else 'FAIL'}  {init_rate:.3f} -> {final_rate:.3f}")
	return passed

# ============================================================
# Test: Fixed-value matching ("insert adjacent to any 5")
# ============================================================

def test_fixed_value_match(n_iters=300, n_games=200):
	"""Target: insert adjacent to any card with value=5 (fixed, not from scouted card).
	Removes the need to read the scouted card's value from the encoding.
	If this passes: the bottleneck is extracting the scouted value, not matching.
	If this fails: PPO can't do positional matching against hand cards at all."""
	TARGET_VAL = 5
	print(f"\n=== Fixed-value match (always match val={TARGET_VAL}, v{ENCODING_VERSION}) ===")
	network = _make_network()
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)

	def _has_target_val(hand):
		return any(c[0] == TARGET_VAL for c in hand)

	def _is_adjacent_to_target(hand, insert_pos):
		if insert_pos > 0 and hand[insert_pos - 1][0] == TARGET_VAL:
			return True
		if insert_pos < len(hand) and hand[insert_pos][0] == TARGET_VAL:
			return True
		return False

	def _eval_rate(network, n_samples=500):
		network.eval()
		n_adj, n_total = 0, 0
		for _ in range(n_samples):
			game = _mid_round_state()
			if game is None:
				continue
			sample = _sample_scout(network, game)
			if sample is None:
				continue
			if not _has_target_val(sample["hand"]):
				continue
			n_total += 1
			if _is_adjacent_to_target(sample["hand"], sample["insert_pos"]):
				n_adj += 1
		return n_adj / max(n_total, 1), n_total

	init_rate, _ = _eval_rate(network)
	print(f"  Initial rate: {init_rate:.3f}")
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
			if not _has_target_val(sample["hand"]):
				continue
			forced_at_mask = np.zeros(ACTION_TYPE_SIZE, dtype=np.bool_)
			forced_at_mask[sample["action_type"]] = True
			sample["at_mask"] = forced_at_mask
			adj = _is_adjacent_to_target(sample["hand"], sample["insert_pos"])
			reward = 1.0 if adj else -1.0
			records.append(_make_scout_record(sample, reward=reward, game_id=g))
		if records:
			verbose = (it % 50 == 0 or it == n_iters - 1)
			if verbose:
				print(f"  [iter {it}]")
			_train_iteration(network, optimizer, records, verbose=verbose)
			if verbose:
				rate, n = _eval_rate(network)
				print(f"  rate={rate:.3f} (n={n})")
	final_rate, _ = _eval_rate(network)
	passed = final_rate > init_rate + 0.05
	print(f"  RESULT: {'PASS' if passed else 'FAIL'}  {init_rate:.3f} -> {final_rate:.3f}")
	return passed

# ============================================================
# Test: Large network on adjacent matching
# ============================================================

def test_big_network(n_iters=300, n_games=200):
	"""Probe 5b with production-size [512,256,256,128,128,128] network.
	If this passes: small networks lack capacity for PPO matching.
	If this fails: capacity isn't the issue."""
	print(f"\n=== Big network [512,256,256,128,128,128] (v{ENCODING_VERSION}) ===")
	network = _make_network(layer_sizes=[512, 256, 256, 128, 128, 128])
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)
	init_rate, _ = _eval_adj_rate(network)
	print(f"  Initial adj rate: {init_rate:.3f}")
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
			forced_at_mask = np.zeros(ACTION_TYPE_SIZE, dtype=np.bool_)
			forced_at_mask[sample["action_type"]] = True
			sample["at_mask"] = forced_at_mask
			adj = _is_adjacent_match(sample["hand"], sample["card"], sample["insert_pos"])
			reward = 1.0 if adj else -1.0
			records.append(_make_scout_record(sample, reward=reward, game_id=g))
		if records:
			verbose = (it % 50 == 0 or it == n_iters - 1)
			if verbose:
				print(f"  [iter {it}]")
			_train_iteration(network, optimizer, records, verbose=verbose)
			if verbose:
				rate, n = _eval_adj_rate(network)
				print(f"  adj_rate={rate:.3f} (n={n})")
	final_rate, _ = _eval_adj_rate(network)
	passed = final_rate > init_rate + 0.05
	print(f"  RESULT: {'PASS' if passed else 'FAIL'}  {init_rate:.3f} -> {final_rate:.3f}")
	return passed

# ============================================================
# Test: Scouted-value hint appended to scout head input
# ============================================================

class HintedScoutNetwork(ScoutNetwork):
	"""ScoutNetwork variant that appends a scouted-card-value one-hot to the
	scout_insert_head input, bypassing the trunk for this critical feature."""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		# Replace scout_insert_head with one that takes extra CARD_VALUES input
		old_in = self.scout_insert_head.in_features
		out = self.scout_insert_head.out_features
		self.scout_insert_head = nn.Linear(old_in + CARD_VALUES, out)
		self._hint = None

	def set_scout_hint(self, card_value):
		"""Set the scouted card value hint for the next scout_insert_logits call."""
		self._hint = card_value

	def scout_insert_logits(self, hidden, action_type):
		conditioned = self._build_conditioning(hidden, action_type, None)
		unbatched = conditioned.ndim == 1
		if unbatched:
			conditioned = conditioned.unsqueeze(0)
		hint_oh = torch.zeros(conditioned.shape[0], CARD_VALUES, device=conditioned.device)
		if self._hint is not None:
			hint_oh[:, self._hint] = 1.0
		conditioned = torch.cat([conditioned, hint_oh], dim=1)
		if unbatched:
			conditioned = conditioned.squeeze(0)
		return self.scout_insert_head(conditioned)

def test_hint(n_iters=300, n_games=200):
	"""Append scouted-card-value one-hot directly to scout head input.
	Bypasses the trunk for this feature — tests if PPO can do matching
	when the value-to-match is handed to it directly.
	If PASS: PPO can match, but can't extract the value from the trunk.
	If FAIL: PPO can't do matching even with the value handed to it."""
	print(f"\n=== Hint (scouted value appended to head, v{ENCODING_VERSION}) ===")
	ls = [64, 32]
	if ENCODING_VERSION == 3:
		network = HintedScoutNetwork(INPUT_SIZE_V3, ls, encoding_version=3,
			play_start_size=PLAY_START_SIZE_V3, play_end_size=PLAY_START_SIZE_V3,
			scout_insert_size=SCOUT_INSERT_SIZE_V3)
	elif ENCODING_VERSION == 2:
		network = HintedScoutNetwork(INPUT_SIZE_V2, ls, encoding_version=2,
			play_start_size=PLAY_START_SIZE_V2, play_end_size=PLAY_START_SIZE_V2,
			scout_insert_size=SCOUT_INSERT_SIZE_V2)
	else:
		network = HintedScoutNetwork(layer_sizes=ls)
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)

	def _sample_scout_hinted(network, game, player=1):
		"""Like _sample_scout but sets the hint before sampling."""
		_sis = _ev_insert_size()
		_hs = _ev_hand_slots()
		hand = game.players[player].hand
		play_cards = game.current_play.cards
		if len(play_cards) == 0:
			return None
		card = play_cards[0]
		action_type = 1
		state, ho, po = _encode(game, player)
		network.set_scout_hint(card[0])
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
			"card": card, "hand": list(hand), "ho": ho,
		}

	def _eval_rate_hinted(network, n_samples=500):
		network.eval()
		n_adj, n_total = 0, 0
		for _ in range(n_samples):
			game = _mid_round_state()
			if game is None:
				continue
			sample = _sample_scout_hinted(network, game)
			if sample is None:
				continue
			if not _has_any_match(sample["hand"], sample["card"]):
				continue
			n_total += 1
			if _is_adjacent_match(sample["hand"], sample["card"], sample["insert_pos"]):
				n_adj += 1
		return n_adj / max(n_total, 1), n_total

	init_rate, _ = _eval_rate_hinted(network)
	print(f"  Initial adj rate: {init_rate:.3f}")
	for it in range(n_iters):
		network.eval()
		records = []
		for g in range(n_games):
			game = _mid_round_state()
			if game is None:
				continue
			sample = _sample_scout_hinted(network, game)
			if sample is None:
				continue
			if not _has_any_match(sample["hand"], sample["card"]):
				continue
			forced_at_mask = np.zeros(ACTION_TYPE_SIZE, dtype=np.bool_)
			forced_at_mask[sample["action_type"]] = True
			sample["at_mask"] = forced_at_mask
			adj = _is_adjacent_match(sample["hand"], sample["card"], sample["insert_pos"])
			reward = 1.0 if adj else -1.0
			records.append(_make_scout_record(sample, reward=reward, game_id=g))
		if records:
			# The hint needs to be set during PPO update too for the forward pass
			# But ppo_update calls network.scout_insert_head via _build_batch_conditioning
			# which won't have the hint. We need to handle this differently.
			# Actually — ppo_update recomputes logits from stored states, so the
			# hint won't be available. The old_logits were computed WITH the hint,
			# but the new logits in ppo_update won't have it.
			# This means the standard ppo_update won't work for this test.
			# Let's do a simpler approach: direct policy gradient without PPO recomputation.
			# Use REINFORCE-style: just weight the old log probs by advantage.
			advantages, returns, adv_std = compute_gae(records, gamma=0.99, lam=0.95)
			verbose = (it % 50 == 0 or it == n_iters - 1)
			if verbose:
				rewards = [r.reward for r in records]
				mean_r = sum(rewards) / len(rewards)
				print(f"  [iter {it}]")
				print(f"    records={len(records)}  mean_reward={mean_r:+.3f}")

			# Manual REINFORCE update with the hint set per-sample
			for epoch in range(PPO_EPOCHS):
				network.train()
				total_loss = torch.tensor(0.0)
				for i, rec in enumerate(records):
					card_val = rec.state  # we need the actual card value...
					# Actually we stored the card in the sample but not in the record.
					# Let's just do a batch approach: re-encode with hints
					pass
			# This approach is getting complicated. Let me just test whether
			# the hint helps with supervised CE instead, which is cleaner.
			# If supervised CE + hint >> supervised CE without hint, it means
			# the trunk isn't passing the scouted value through.
			# But we already know supervised CE works WITHOUT the hint (test A passes).
			# So the hint test doesn't add information in the supervised case.
			# For PPO, the issue is that ppo_update can't use the hint.
			# Let me skip this test and replace with something more informative.
			break
	print("  SKIPPED: ppo_update can't propagate per-sample hints through batched recomputation")
	print("  (see test fixed_val instead — tests same hypothesis without architecture changes)")
	return False

# ============================================================
# Test: CE pretrain then unfreeze for PPO
# ============================================================

def test_ce_then_ppo(n_iters=300, n_games=200):
	"""CE pretrain full network, then switch to PPO with everything unfrozen.
	If features SURVIVE: PPO can't discover features but doesn't destroy them.
	If features DEGRADE: PPO gradient actively pushes trunk away from good features."""
	print(f"\n=== CE pretrain -> PPO unfreeze (v{ENCODING_VERSION}) ===")
	_sis = _ev_insert_size()
	_pss = PLAY_START_SIZE_V3 if ENCODING_VERSION == 3 else (PLAY_START_SIZE_V2 if ENCODING_VERSION == 2 else PLAY_START_SIZE)

	def _find_best_adjacent_slots(hand, card, ho):
		val = card[0]
		good_slots = []
		for pos in range(len(hand) + 1):
			if _is_adjacent_match(hand, card, pos):
				slot = (ho + pos) % _sis
				good_slots.append(slot)
		return good_slots

	# --- Phase 1: Supervised CE training ---
	print("  Phase 1: CE pre-training...")
	network = _make_network()
	optimizer = torch.optim.Adam(network.parameters(), lr=3e-3)

	for it in range(200):
		network.eval()
		states_list, targets_list, masks_list, at_list = [], [], [], []
		for g in range(500):
			game = _mid_round_state()
			if game is None:
				continue
			hand = game.players[1].hand
			play_cards = game.current_play.cards
			if not play_cards:
				continue
			card = play_cards[0]
			if not _has_any_match(hand, card):
				continue
			state, ho, po = _encode(game, player=1)
			si_mask = get_scout_insert_mask(game, ho, num_slots=_sis)
			good_slots = _find_best_adjacent_slots(hand, card, ho)
			if not good_slots:
				continue
			target = torch.zeros(_sis)
			for s in good_slots:
				target[s] = 1.0
			target = target / target.sum()
			states_list.append(state)
			targets_list.append(target)
			masks_list.append(torch.from_numpy(si_mask))
			at_list.append(1)
		if not states_list:
			continue
		states = torch.stack(states_list)
		targets = torch.stack(targets_list)
		masks = torch.stack(masks_list)
		at_tensor = torch.tensor(at_list, dtype=torch.long)
		for epoch in range(10):
			network.train()
			hidden = network(states)
			at_oh = F.one_hot(at_tensor, ACTION_TYPE_SIZE).float()
			start_oh = torch.zeros(len(states_list), _pss)
			cond = torch.cat([hidden, at_oh, start_oh], dim=1)
			logits = network.scout_insert_head(cond)
			masked_logits = logits.masked_fill(~masks, float('-inf'))
			log_probs = F.log_softmax(masked_logits, dim=-1)
			loss = -(targets * log_probs).nan_to_num(0.0).sum(dim=-1).mean()
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.5)
			optimizer.step()
		if it % 50 == 0:
			rate, _ = _eval_adj_rate(network)
			print(f"    CE iter {it:3d}  loss={loss.item():.4f}  adj_rate={rate:.3f}")

	ce_rate, _ = _eval_adj_rate(network)
	print(f"  CE done: adj_rate={ce_rate:.3f}")
	if ce_rate < 0.5:
		print("  CE pre-training failed")
		return False

	# --- Phase 2: Switch to PPO, everything unfrozen ---
	print("  Phase 2: Switching to PPO (all params unfrozen)...")
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)

	init_rate = ce_rate
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
			forced_at_mask = np.zeros(ACTION_TYPE_SIZE, dtype=np.bool_)
			forced_at_mask[sample["action_type"]] = True
			sample["at_mask"] = forced_at_mask
			adj = _is_adjacent_match(sample["hand"], sample["card"], sample["insert_pos"])
			reward = 1.0 if adj else -1.0
			records.append(_make_scout_record(sample, reward=reward, game_id=g))
		if records:
			verbose = (it % 50 == 0 or it == n_iters - 1)
			if verbose:
				print(f"  [PPO iter {it}]")
			_train_iteration(network, optimizer, records, verbose=verbose)
			if verbose:
				rate, n = _eval_adj_rate(network)
				print(f"  adj_rate={rate:.3f} (n={n})")

	final_rate, _ = _eval_adj_rate(network)
	survived = final_rate > ce_rate - 0.1
	improved = final_rate > ce_rate
	print(f"  RESULT: CE={ce_rate:.3f} -> PPO={final_rate:.3f}")
	if improved:
		print(f"  IMPROVED: PPO continued improving beyond CE")
	elif survived:
		print(f"  SURVIVED: features preserved under PPO")
	else:
		print(f"  DEGRADED: PPO destroyed CE features")
	return survived

# ============================================================
# Test: Gradient comparison (CE vs PPO on same batch)
# ============================================================

def test_gradient_compare(n_batches=20, n_games=500):
	"""Generate same batch, compute CE and PPO gradients on trunk, compare.
	Reports magnitude ratio and cosine similarity."""
	print(f"\n=== Gradient comparison CE vs PPO (v{ENCODING_VERSION}) ===")
	_sis = _ev_insert_size()
	_pss = PLAY_START_SIZE_V3 if ENCODING_VERSION == 3 else (PLAY_START_SIZE_V2 if ENCODING_VERSION == 2 else PLAY_START_SIZE)

	def _find_best_adjacent_slots(hand, card, ho):
		val = card[0]
		good_slots = []
		for pos in range(len(hand) + 1):
			if _is_adjacent_match(hand, card, pos):
				slot = (ho + pos) % _sis
				good_slots.append(slot)
		return good_slots

	network = _make_network()

	cos_sims = []
	mag_ratios = []

	for batch_idx in range(n_batches):
		# Generate samples
		samples = []
		for g in range(n_games):
			game = _mid_round_state()
			if game is None:
				continue
			sample = _sample_scout(network, game)
			if sample is None:
				continue
			if not _has_any_match(sample["hand"], sample["card"]):
				continue
			samples.append((game, sample))
		if len(samples) < 10:
			continue

		# --- CE gradient ---
		network.train()
		network.zero_grad()
		states_list, targets_list, masks_list = [], [], []
		for game, sample in samples:
			ho = sample["ho"]
			hand = sample["hand"]
			card = sample["card"]
			si_mask = sample["si_mask"]
			good_slots = _find_best_adjacent_slots(hand, card, ho)
			if not good_slots:
				continue
			target = torch.zeros(_sis)
			for s in good_slots:
				target[s] = 1.0
			target = target / target.sum()
			states_list.append(sample["state"])
			targets_list.append(target)
			masks_list.append(torch.from_numpy(si_mask) if isinstance(si_mask, np.ndarray) else si_mask)

		if not states_list:
			continue
		states = torch.stack(states_list)
		targets = torch.stack(targets_list)
		masks = torch.stack(masks_list)
		hidden = network(states)
		at_oh = torch.zeros(len(states_list), ACTION_TYPE_SIZE)
		at_oh[:, 1] = 1.0
		start_oh = torch.zeros(len(states_list), _pss)
		cond = torch.cat([hidden, at_oh, start_oh], dim=1)
		logits = network.scout_insert_head(cond)
		masked_logits = logits.masked_fill(~masks, float('-inf'))
		log_probs = F.log_softmax(masked_logits, dim=-1)
		ce_loss = -(targets * log_probs).nan_to_num(0.0).sum(dim=-1).mean()
		ce_loss.backward()

		ce_grad = torch.cat([p.grad.flatten() for p in network.shared.parameters() if p.grad is not None]).clone()

		# --- PPO gradient ---
		network.zero_grad()
		# Build PPO records from same samples
		records = []
		for game, sample in samples:
			forced_at_mask = np.zeros(ACTION_TYPE_SIZE, dtype=np.bool_)
			forced_at_mask[sample["action_type"]] = True
			sample["at_mask"] = forced_at_mask
			adj = _is_adjacent_match(sample["hand"], sample["card"], sample["insert_pos"])
			reward = 1.0 if adj else -1.0
			records.append(_make_scout_record(sample, reward=reward, game_id=0))

		advantages, returns, _ = compute_gae(records, gamma=0.99, lam=0.95)
		batch = prepare_ppo_batch(records, advantages, returns=returns)
		# Compute PPO policy loss manually (scout_insert only)
		states = torch.stack([s["state"] for _, s in samples])
		hidden = network(states)
		at_oh = torch.zeros(len(samples), ACTION_TYPE_SIZE)
		at_oh[:, 1] = 1.0
		start_oh = torch.zeros(len(samples), _pss)
		cond = torch.cat([hidden, at_oh, start_oh], dim=1)
		logits = network.scout_insert_head(cond)
		si_masks = torch.stack([
			torch.from_numpy(s["si_mask"]) if isinstance(s["si_mask"], np.ndarray) else s["si_mask"]
			for _, s in samples
		])
		masked_logits = logits.masked_fill(~si_masks, float('-inf'))
		log_probs = F.log_softmax(masked_logits, dim=-1)
		actions = torch.tensor([s["insert_slot"] for _, s in samples], dtype=torch.long)
		action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
		adv_tensor = torch.tensor(advantages[:len(samples)], dtype=torch.float32)
		if adv_tensor.std() > 1e-6:
			adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
		ppo_loss = -(action_log_probs * adv_tensor).mean()
		ppo_loss.backward()

		ppo_grad = torch.cat([p.grad.flatten() for p in network.shared.parameters() if p.grad is not None]).clone()

		# Compare
		cos_sim = F.cosine_similarity(ce_grad.unsqueeze(0), ppo_grad.unsqueeze(0)).item()
		ce_mag = ce_grad.norm().item()
		ppo_mag = ppo_grad.norm().item()
		ratio = ppo_mag / max(ce_mag, 1e-8)

		cos_sims.append(cos_sim)
		mag_ratios.append(ratio)

		if batch_idx < 3:
			print(f"  batch {batch_idx}: cos_sim={cos_sim:.4f}  CE_mag={ce_mag:.4f}  PPO_mag={ppo_mag:.4f}  ratio={ratio:.4f}")

	if cos_sims:
		mean_cos = sum(cos_sims) / len(cos_sims)
		mean_ratio = sum(mag_ratios) / len(mag_ratios)
		print(f"\n  Mean cosine similarity: {mean_cos:.4f}")
		print(f"  Mean magnitude ratio (PPO/CE): {mean_ratio:.4f}")
		print(f"  Interpretation:")
		if mean_cos > 0.3:
			print(f"    Gradients roughly aligned -- PPO points in similar direction as CE")
		elif mean_cos > -0.1:
			print(f"    Gradients nearly orthogonal -- PPO signal is unrelated to CE")
		else:
			print(f"    Gradients opposing -- PPO actively pushes away from CE solution")
	return True

# ============================================================
# Test: Frozen CE-trained trunk + PPO head
# ============================================================

def test_frozen_trunk(n_iters=300, n_games=200):
	"""Phase 1: Train network with supervised CE until adj matching works.
	Phase 2: Freeze trunk, reinit scout_insert_head, train with PPO.
	If PASS: PPO can learn the head given good features — failure is trunk learning.
	If FAIL: PPO can't even learn the head — failure is in the PPO signal itself."""
	print(f"\n=== Frozen trunk (CE pretrain -> freeze -> PPO head, v{ENCODING_VERSION}) ===")
	_sis = _ev_insert_size()
	_hs = _ev_hand_slots()
	_pss = PLAY_START_SIZE_V3 if ENCODING_VERSION == 3 else (PLAY_START_SIZE_V2 if ENCODING_VERSION == 2 else PLAY_START_SIZE)

	def _find_best_adjacent_slots(hand, card, ho):
		val = card[0]
		good_slots = []
		for pos in range(len(hand) + 1):
			if _is_adjacent_match(hand, card, pos):
				slot = (ho + pos) % _sis
				good_slots.append(slot)
		return good_slots

	# --- Phase 1: Supervised CE training ---
	print("  Phase 1: Supervised CE pre-training...")
	network = _make_network()
	optimizer = torch.optim.Adam(network.parameters(), lr=3e-3)

	for it in range(200):
		network.eval()
		states_list, targets_list, masks_list, at_list = [], [], [], []
		for g in range(500):
			game = _mid_round_state()
			if game is None:
				continue
			hand = game.players[1].hand
			play_cards = game.current_play.cards
			if not play_cards:
				continue
			card = play_cards[0]
			if not _has_any_match(hand, card):
				continue
			state, ho, po = _encode(game, player=1)
			si_mask = get_scout_insert_mask(game, ho, num_slots=_sis)
			good_slots = _find_best_adjacent_slots(hand, card, ho)
			if not good_slots:
				continue
			target = torch.zeros(_sis)
			for s in good_slots:
				target[s] = 1.0
			target = target / target.sum()
			states_list.append(state)
			targets_list.append(target)
			masks_list.append(torch.from_numpy(si_mask))
			at_list.append(1)
		if not states_list:
			continue
		states = torch.stack(states_list)
		targets = torch.stack(targets_list)
		masks = torch.stack(masks_list)
		at_tensor = torch.tensor(at_list, dtype=torch.long)
		for epoch in range(10):
			network.train()
			hidden = network(states)
			at_oh = F.one_hot(at_tensor, ACTION_TYPE_SIZE).float()
			start_oh = torch.zeros(len(states_list), _pss)
			cond = torch.cat([hidden, at_oh, start_oh], dim=1)
			logits = network.scout_insert_head(cond)
			masked_logits = logits.masked_fill(~masks, float('-inf'))
			log_probs = F.log_softmax(masked_logits, dim=-1)
			loss = -(targets * log_probs).nan_to_num(0.0).sum(dim=-1).mean()
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.5)
			optimizer.step()
		if it % 50 == 0:
			rate, _ = _eval_adj_rate(network)
			print(f"    CE iter {it:3d}  loss={loss.item():.4f}  adj_rate={rate:.3f}")

	ce_rate, _ = _eval_adj_rate(network)
	print(f"  CE training done: adj_rate={ce_rate:.3f}")
	if ce_rate < 0.5:
		print("  CE pre-training failed — can't test frozen trunk")
		return False

	# --- Phase 2: Freeze trunk, reinit head, train with PPO ---
	print("  Phase 2: Freezing trunk, reinitializing head, training with PPO...")
	for param in network.shared.parameters():
		param.requires_grad = False

	# Reinitialize scout_insert_head
	in_features = network.scout_insert_head.in_features
	out_features = network.scout_insert_head.out_features
	network.scout_insert_head = nn.Linear(in_features, out_features)

	# Optimizer only on unfrozen params
	optimizer = torch.optim.Adam(
		[p for p in network.parameters() if p.requires_grad], lr=LR)

	init_rate, _ = _eval_adj_rate(network)
	print(f"  PPO initial adj rate: {init_rate:.3f}")

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
			forced_at_mask = np.zeros(ACTION_TYPE_SIZE, dtype=np.bool_)
			forced_at_mask[sample["action_type"]] = True
			sample["at_mask"] = forced_at_mask
			adj = _is_adjacent_match(sample["hand"], sample["card"], sample["insert_pos"])
			reward = 1.0 if adj else -1.0
			records.append(_make_scout_record(sample, reward=reward, game_id=g))
		if records:
			verbose = (it % 50 == 0 or it == n_iters - 1)
			if verbose:
				print(f"  [PPO iter {it}]")
			_train_iteration(network, optimizer, records, verbose=verbose)
			if verbose:
				rate, n = _eval_adj_rate(network)
				print(f"  adj_rate={rate:.3f} (n={n})")

	final_rate, _ = _eval_adj_rate(network)
	passed = final_rate > init_rate + 0.05
	print(f"  RESULT: {'PASS' if passed else 'FAIL'}  {init_rate:.3f} -> {final_rate:.3f}")
	return passed

# ============================================================

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--test", type=str, default="all",
		help="Which test(s): graded, fixed_val, big_net, hint, or 'all'")
	parser.add_argument("--iters", type=int, default=300)
	parser.add_argument("--games", type=int, default=200)
	parser.add_argument("--v2", action="store_true")
	parser.add_argument("--v3", action="store_true",
		help="Use v3 encoding (scalar values + pairwise diffs, 285 dims)")
	args = parser.parse_args()

	global ENCODING_VERSION
	if args.v3:
		ENCODING_VERSION = 3
	elif args.v2:
		ENCODING_VERSION = 2

	tests = args.test.lower().split(",") if args.test != "all" else ["graded", "fixed_val", "big_net"]

	results = []
	for t in tests:
		t = t.strip()
		if t == "graded":
			results.append(("graded_reward", test_graded_reward(args.iters, args.games)))
		elif t == "fixed_val":
			results.append(("fixed_value_match", test_fixed_value_match(args.iters, args.games)))
		elif t == "big_net":
			results.append(("big_network", test_big_network(args.iters, args.games)))
		elif t == "hint":
			results.append(("hint", test_hint(args.iters, args.games)))
		elif t == "frozen_trunk":
			results.append(("frozen_trunk", test_frozen_trunk(args.iters, args.games)))
		elif t == "ce_then_ppo":
			results.append(("ce_then_ppo", test_ce_then_ppo(args.iters, args.games)))
		elif t == "grad_compare":
			results.append(("grad_compare", test_gradient_compare()))
		else:
			print(f"Unknown test: {t}")

	print(f"\n{'='*50}")
	print("SUMMARY:")
	for name, passed in results:
		print(f"  {'PASS' if passed else 'FAIL'}  {name}")

if __name__ == "__main__":
	main()
