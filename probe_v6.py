"""Probe environments for v6 flat-action-space Scout NN.
Incremental validation from encoding through full game loop.

Probes:
  0. Encoding sanity: valid tensors, mask/decode round-trip, dimensions
  1. Forward pass: shapes, no NaNs, masked softmax
  2. Value learning: can the value head learn a constant return?
  3. Policy preference: can the flat head learn to prefer play over scout?
  4. PPO mechanics: batch→update→gradient flow
  5. Rotation augmentation: permuted masks stay consistent
  6. Full game loop: rollouts produce valid records
  7. S&S round-trip: mask↔decode↔legality consistency for Scout & Show
  8. Play length preference: can the flat head learn to prefer longer plays?
  9. Rotation content: verify hand slot values actually shift correctly

Usage: python probe_v6.py [--probe N [N ...]] [--iters N] [--games N] [--layers N N ...]
"""
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from game import Game, Phase, Play
from encoding import (
	encode_state_v6, get_flat_action_mask, decode_flat_action,
	get_legal_plays, _has_any_legal_play, INPUT_SIZE_V6, FLAT_ACTION_SIZE,
	HAND_SLOTS_V6, NUM_VALUES_V6, FULL_PERM, HAND_SHIFT,
)
from network import FlatScoutNetwork, masked_sample
from training import (
	StepRecordV6, compute_gae,
	prepare_ppo_batch_v6, ppo_update_v6, augment_rotation_v6,
)

NUM_PLAYERS = 4
LAYER_SIZES = [64, 32]
LR = 3e-4
PPO_EPOCHS = 4
CLIP_EPSILON = 0.2
ENTROPY_BONUS = 0.01
VALUE_LOSS_COEFF = 0.25

# --- Helpers ---

def _make_network(layer_sizes=None):
	ls = layer_sizes or LAYER_SIZES
	return FlatScoutNetwork(INPUT_SIZE_V6, ls, encoding_version=6,
		attention={"dim": 32, "heads": 2, "layers": 1})

def _fresh_round():
	"""Create a game at the start of a round (TURN phase, no current play)."""
	game = Game(NUM_PLAYERS)
	game.start_round()
	for p in range(NUM_PLAYERS):
		game.submit_flip_decision(p, do_flip=random.random() < 0.5)
	return game

def _mid_round_state():
	"""Create a game state where current player can scout (play exists on table).
	Returns (game, player) or None."""
	for _ in range(100):
		game = _fresh_round()
		# Play a few turns to get a play on the table
		for _ in range(10):
			if game.phase != Phase.TURN:
				break
			p = game.current_player
			hand = game.players[p].hand
			legal_plays = get_legal_plays(hand, game.current_play)
			if legal_plays:
				s, e = random.choice(legal_plays)
				game.apply_play(s, e)
			elif game.current_play is not None:
				game.apply_scout(True, False, 0)
			else:
				break
			if game.current_play is not None and game.phase == Phase.TURN:
				return game, game.current_player
	return None

def _sample_action(network, game, player):
	"""Run network forward, sample from masked flat logits.
	Returns dict with state, action, mask, log_prob, value, hand_offset, or None."""
	hand = game.players[player].hand
	legal_plays = get_legal_plays(hand, game.current_play)
	ho = random.randint(0, HAND_SLOTS_V6 - 1)
	state = encode_state_v6(game, player, ho,
		forced_play=(game.phase == Phase.SNS_PLAY))
	mask = get_flat_action_mask(game, player, legal_plays, ho)
	if not mask.any():
		return None
	with torch.no_grad():
		hidden = network(state)
		value = network.value(hidden).item()
		logits = network.policy_logits(hidden)
		action, _ = masked_sample(logits, mask)
		# Compute log prob
		masked_logits = logits.masked_fill(~mask, float('-inf'))
		log_prob = torch.log_softmax(masked_logits, dim=-1)[action].item()
	return {
		"state": state, "action": action, "mask": mask.numpy(),
		"log_prob": log_prob, "value": value, "ho": ho,
		"legal_plays": legal_plays,
	}

def _make_record(sample, reward, game_id, player=0):
	return StepRecordV6(
		state=sample["state"],
		action=sample["action"],
		mask=sample["mask"],
		old_log_prob=sample["log_prob"],
		value=sample["value"],
		reward=reward,
		player=player,
		round_num=0,
		game_id=game_id,
		hand_offset=sample["ho"],
		play_length=None,
		scout_quality=None,
	)

def _train_iteration(network, optimizer, records, advantages=None, returns=None,
					 augment=False):
	"""GAE + PPO epochs. Returns avg metrics."""
	if advantages is None:
		advantages, returns = compute_gae(records, gamma=0.99, lam=0.95)
	if augment:
		# Match main.py: overwrite rec.value with GAE return, then augment
		for rec, ret in zip(records, returns):
			rec.value = ret
		records, advantages = augment_rotation_v6(records, advantages, network)
		returns = None  # prepare_ppo_batch_v6 uses rec.value
	batch = prepare_ppo_batch_v6(records, advantages, returns=returns)
	if batch is None:
		return {}
	ppo_sums = {}
	for _ in range(PPO_EPOCHS):
		network.train()
		m = ppo_update_v6(network, optimizer, batch,
			clip_epsilon=CLIP_EPSILON, entropy_bonus=ENTROPY_BONUS,
			value_loss_coeff=VALUE_LOSS_COEFF)
		for k, v in m.items():
			ppo_sums[k] = ppo_sums.get(k, 0.0) + v
	return {k: v / PPO_EPOCHS for k, v in ppo_sums.items()}

# --- Probe 0: Encoding sanity ---

def probe_encoding(n_iters=None, n_games=None):
	"""Check encoding dimensions, mask validity, and encode→mask→decode round-trip."""
	n_checks = 200
	passes = 0
	fails = []
	def fail(msg):
		fails.append(msg)
	# Check 1: encoding dimensions
	game = _fresh_round()
	state = encode_state_v6(game, 0, 0)
	if state.shape != (INPUT_SIZE_V6,):
		fail(f"state shape {state.shape} != ({INPUT_SIZE_V6},)")
	if torch.isnan(state).any():
		fail("NaN in encoded state")
	if state.dtype != torch.float32:
		fail(f"state dtype {state.dtype} != float32")
	passes += 1
	# Check 2: mask dimensions and basic validity
	hand = game.players[0].hand
	legal_plays = get_legal_plays(hand, game.current_play)
	ho = 0
	mask = get_flat_action_mask(game, 0, legal_plays, ho)
	if mask.shape != (FLAT_ACTION_SIZE,):
		fail(f"mask shape {mask.shape} != ({FLAT_ACTION_SIZE},)")
	if not mask.any():
		fail("mask has no legal actions at round start")
	# At round start (no current play), only play region should be active
	if mask[256:].any():
		fail("scout/sns actions legal at round start (no play on table)")
	passes += 1
	# Check 3: round-trip encode→mask→decode on many states
	for i in range(n_checks):
		game = _fresh_round()
		player = game.current_player
		hand = game.players[player].hand
		legal_plays = get_legal_plays(hand, game.current_play)
		ho = random.randint(0, HAND_SLOTS_V6 - 1)
		state = encode_state_v6(game, player, ho)
		mask = get_flat_action_mask(game, player, legal_plays, ho)
		# Every legal action should decode to valid game coords
		legal_set = set(legal_plays)
		for action_idx in range(FLAT_ACTION_SIZE):
			if not mask[action_idx]:
				continue
			decoded = decode_flat_action(action_idx, ho)
			if decoded["type"] == "play":
				if (decoded["start"], decoded["end"]) not in legal_set:
					fail(f"decoded play ({decoded['start']},{decoded['end']}) not in legal_plays")
					break
			# Scout/sns only possible with current_play, which is None at round start
		# Every legal play should have a corresponding mask entry
		for s, e in legal_plays:
			s_slot = (ho + s) % HAND_SLOTS_V6
			e_slot = (ho + e) % HAND_SLOTS_V6
			action_idx = s_slot * HAND_SLOTS_V6 + e_slot
			if not mask[action_idx]:
				fail(f"legal play ({s},{e}) not in mask (ho={ho}, idx={action_idx})")
				break
	passes += 1
	# Check 4: mid-round state has scout actions
	result = _mid_round_state()
	if result is None:
		fail("couldn't create mid-round state")
	else:
		game, player = result
		hand = game.players[player].hand
		legal_plays = get_legal_plays(hand, game.current_play)
		ho = random.randint(0, HAND_SLOTS_V6 - 1)
		mask = get_flat_action_mask(game, player, legal_plays, ho)
		if not mask[256:320].any():
			fail("no scout actions in mid-round state")
	passes += 1
	# Check 5: different hand offsets produce same set of decoded actions
	game = _fresh_round()
	player = game.current_player
	hand = game.players[player].hand
	legal_plays = get_legal_plays(hand, game.current_play)
	decoded_sets = []
	for ho in range(HAND_SLOTS_V6):
		mask = get_flat_action_mask(game, player, legal_plays, ho)
		decoded = set()
		for idx in range(FLAT_ACTION_SIZE):
			if mask[idx]:
				d = decode_flat_action(idx, ho)
				if d["type"] == "play":
					decoded.add(("play", d["start"], d["end"]))
		decoded_sets.append(decoded)
	if not all(s == decoded_sets[0] for s in decoded_sets[1:]):
		fail("different hand offsets produce different decoded play sets")
	passes += 1
	passed = len(fails) == 0
	status = "PASS" if passed else "FAIL"
	detail = f"  {len(fails)} failures" if fails else ""
	print(f"  Probe 0 (encoding sanity):     {status}  {passes} checks{detail}")
	for f in fails[:5]:
		print(f"    - {f}")
	return passed

# --- Probe 1: Forward pass ---

def probe_forward(n_iters=None, n_games=None):
	"""Check network output shapes, no NaNs, masked softmax validity."""
	network = _make_network()
	network.eval()
	fails = []
	def fail(msg):
		fails.append(msg)
	# Single state
	game = _fresh_round()
	state = encode_state_v6(game, 0, 0)
	hidden = network(state)
	if hidden.dim() != 1:
		fail(f"hidden dim {hidden.dim()} != 1 for single state")
	value = network.value(hidden)
	if value.shape != (1,):
		fail(f"value shape {value.shape} != (1,)")
	logits = network.policy_logits(hidden)
	if logits.shape != (FLAT_ACTION_SIZE,):
		fail(f"logits shape {logits.shape} != ({FLAT_ACTION_SIZE},)")
	if torch.isnan(logits).any():
		fail("NaN in logits")
	# Batched
	states = torch.stack([encode_state_v6(_fresh_round(), 0, random.randint(0, 15))
		for _ in range(16)])
	hidden_b = network(states)
	if hidden_b.shape[0] != 16:
		fail(f"batched hidden batch dim {hidden_b.shape[0]} != 16")
	logits_b = network.policy_logits(hidden_b)
	if logits_b.shape != (16, FLAT_ACTION_SIZE):
		fail(f"batched logits shape {logits_b.shape} != (16, {FLAT_ACTION_SIZE})")
	# Masked softmax: all prob on legal actions
	hand = game.players[0].hand
	legal_plays = get_legal_plays(hand, game.current_play)
	mask = get_flat_action_mask(game, 0, legal_plays, 0)
	masked_logits = logits.masked_fill(~mask, float('-inf'))
	probs = torch.softmax(masked_logits, dim=-1)
	illegal_prob = probs[~mask].sum().item()
	if illegal_prob > 1e-6:
		fail(f"illegal actions got prob {illegal_prob:.6f}")
	legal_prob = probs[mask].sum().item()
	if abs(legal_prob - 1.0) > 1e-4:
		fail(f"legal probs sum to {legal_prob:.6f} (should be 1.0)")
	passed = len(fails) == 0
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 1 (forward pass):        {status}")
	for f in fails[:5]:
		print(f"    - {f}")
	return passed

# --- Probe 2: Value learning ---

def probe_value(n_iters=100, n_games=50):
	"""Can the value head learn a constant return?
	Every state gets reward=+1.0, value should converge to ~1.0."""
	network = _make_network()
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)
	# Initial
	network.eval()
	init_vals = []
	for _ in range(n_games):
		game = _fresh_round()
		sample = _sample_action(network, game, game.current_player)
		if sample:
			init_vals.append(sample["value"])
	init_mean = sum(init_vals) / len(init_vals) if init_vals else 0.0
	# Train
	for it in range(n_iters):
		network.eval()
		records = []
		for g in range(n_games):
			game = _fresh_round()
			sample = _sample_action(network, game, game.current_player)
			if sample:
				records.append(_make_record(sample, reward=1.0, game_id=g))
		if records:
			_train_iteration(network, optimizer, records)
	# Final
	network.eval()
	final_vals = []
	for _ in range(n_games):
		game = _fresh_round()
		sample = _sample_action(network, game, game.current_player)
		if sample:
			final_vals.append(sample["value"])
	final_mean = sum(final_vals) / len(final_vals) if final_vals else 0.0
	passed = final_mean > 0.7
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 2 (value learning):      {status}  value {init_mean:.3f} -> {final_mean:.3f}  (target: >0.7)")
	return passed

# --- Probe 3: Policy preference ---

def probe_policy_preference(n_iters=100, n_games=50):
	"""Can the flat head learn to prefer play over scout?
	Mid-round states where both are legal. Reward +1 for play, -1 for scout."""
	network = _make_network()
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)
	def collect_episode():
		result = _mid_round_state()
		if result is None:
			return None
		game, player = result
		sample = _sample_action(network, game, player)
		if sample is None:
			return None
		action_type = decode_flat_action(sample["action"], sample["ho"])["type"]
		reward = 1.0 if action_type == "play" else -1.0
		return _make_record(sample, reward=reward, game_id=0, player=player), action_type
	# Initial play rate
	network.eval()
	init_plays = 0
	init_total = 0
	for _ in range(n_games * 2):
		ep = collect_episode()
		if ep:
			_, at = ep
			init_total += 1
			if at == "play":
				init_plays += 1
	init_rate = init_plays / max(init_total, 1)
	# Train
	for it in range(n_iters):
		network.eval()
		records = []
		for g in range(n_games):
			ep = collect_episode()
			if ep:
				rec, _ = ep
				rec = StepRecordV6(
					state=rec.state, action=rec.action, mask=rec.mask,
					old_log_prob=rec.old_log_prob, value=rec.value,
					reward=rec.reward, player=rec.player, round_num=0,
					game_id=g, hand_offset=rec.hand_offset,
					play_length=None, scout_quality=None,
				)
				records.append(rec)
		if records:
			_train_iteration(network, optimizer, records)
	# Final play rate
	network.eval()
	final_plays = 0
	final_total = 0
	for _ in range(n_games * 2):
		ep = collect_episode()
		if ep:
			_, at = ep
			final_total += 1
			if at == "play":
				final_plays += 1
	final_rate = final_plays / max(final_total, 1)
	passed = final_rate > 0.8
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 3 (play preference):     {status}  play_rate {init_rate:.2f} -> {final_rate:.2f}  (target: >0.8)")
	return passed

# --- Probe 4: PPO mechanics ---

def probe_ppo_mechanics(n_iters=None, n_games=None):
	"""Does prepare_ppo_batch_v6 → ppo_update_v6 produce gradients and reduce loss?"""
	network = _make_network()
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)
	fails = []
	def fail(msg):
		fails.append(msg)
	# Collect some records
	network.eval()
	records = []
	for g in range(50):
		game = _fresh_round()
		sample = _sample_action(network, game, game.current_player)
		if sample:
			records.append(_make_record(sample, reward=1.0, game_id=g))
	if len(records) < 10:
		fail(f"only {len(records)} records collected")
		print(f"  Probe 4 (PPO mechanics):       FAIL  insufficient records")
		return False
	# GAE
	advantages, returns = compute_gae(records, gamma=0.99, lam=0.95)
	if len(advantages) != len(records):
		fail(f"advantages length {len(advantages)} != records {len(records)}")
	# Batch
	batch = prepare_ppo_batch_v6(records, advantages, returns=returns)
	if batch is None:
		fail("prepare_ppo_batch_v6 returned None")
		print(f"  Probe 4 (PPO mechanics):       FAIL  batch is None")
		return False
	expected_keys = {"n", "states", "masks", "actions", "old_log_probs", "adv", "v_target"}
	missing = expected_keys - set(batch.keys())
	if missing:
		fail(f"batch missing keys: {missing}")
	if batch["n"] != len(records):
		fail(f"batch n={batch['n']} != {len(records)}")
	# PPO update should produce non-zero gradients
	network.train()
	metrics = ppo_update_v6(network, optimizer, batch,
		clip_epsilon=CLIP_EPSILON, entropy_bonus=ENTROPY_BONUS,
		value_loss_coeff=VALUE_LOSS_COEFF)
	if metrics["policy_loss"] == 0.0 and metrics["value_loss"] == 0.0:
		fail("both losses are exactly 0 after update")
	# Check gradients were applied (params should have changed)
	network.eval()
	records2 = []
	for g in range(50):
		game = _fresh_round()
		sample = _sample_action(network, game, game.current_player)
		if sample:
			records2.append(_make_record(sample, reward=1.0, game_id=g))
	# Run multiple epochs and check loss decreases
	initial_vloss = metrics["value_loss"]
	for _ in range(20):
		network.train()
		metrics = ppo_update_v6(network, optimizer, batch,
			clip_epsilon=CLIP_EPSILON, entropy_bonus=ENTROPY_BONUS,
			value_loss_coeff=VALUE_LOSS_COEFF)
	final_vloss = metrics["value_loss"]
	if final_vloss >= initial_vloss and initial_vloss > 0.01:
		fail(f"value loss didn't decrease: {initial_vloss:.4f} -> {final_vloss:.4f}")
	passed = len(fails) == 0
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 4 (PPO mechanics):       {status}  vloss {initial_vloss:.4f} -> {final_vloss:.4f}")
	for f in fails[:5]:
		print(f"    - {f}")
	return passed

# --- Probe 5: Rotation augmentation ---

def probe_rotation(n_iters=None, n_games=None):
	"""Does augment_rotation_v6 produce valid permutations with consistent masks?"""
	network = _make_network()
	network.eval()
	fails = []
	def fail(msg):
		fails.append(msg)
	# Collect a few records
	records = []
	for g in range(20):
		game = _fresh_round()
		sample = _sample_action(network, game, game.current_player)
		if sample:
			records.append(_make_record(sample, reward=1.0, game_id=g))
	if len(records) < 5:
		fail("too few records")
		print(f"  Probe 5 (rotation augment):    FAIL  insufficient records")
		return False
	advantages = [0.5] * len(records)
	aug_steps, aug_advs = augment_rotation_v6(records, advantages, network)
	# Should have 16x records (original + 15 rotations)
	expected = len(records) * 16
	if len(aug_steps) != expected:
		fail(f"augmented count {len(aug_steps)} != {expected} (16x {len(records)})")
	if len(aug_advs) != expected:
		fail(f"augmented advantages count {len(aug_advs)} != {expected}")
	# Check that augmented masks have same number of legal actions as originals
	for i, orig in enumerate(records):
		orig_count = orig.mask.sum()
		for k in range(1, 16):
			aug = aug_steps[k * len(records) + i]
			aug_count = aug.mask.sum()
			if aug_count != orig_count:
				fail(f"record {i} rotation {k}: mask count {aug_count} != orig {orig_count}")
				break
	# Check augmented states have correct shape and no NaNs
	for step in aug_steps[:50]:
		if step.state.shape != (INPUT_SIZE_V6,):
			fail(f"augmented state shape {step.state.shape}")
			break
		if torch.isnan(step.state).any():
			fail("NaN in augmented state")
			break
	# Check that decoded actions from augmented records match originals
	for i, orig in enumerate(records):
		orig_decoded = decode_flat_action(orig.action, orig.hand_offset)
		for k in range(1, 16):
			aug = aug_steps[k * len(records) + i]
			aug_decoded = decode_flat_action(aug.action, aug.hand_offset)
			if orig_decoded != aug_decoded:
				fail(f"record {i} rotation {k}: decoded action mismatch "
					 f"{orig_decoded} != {aug_decoded}")
				break
		else:
			continue
		break
	passed = len(fails) == 0
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 5 (rotation augment):    {status}  {len(records)} -> {len(aug_steps)} records")
	for f in fails[:5]:
		print(f"    - {f}")
	return passed

# --- Probe 6: Full game loop ---

def probe_full_game(n_iters=None, n_games=None):
	"""Does play_games_with_rollouts_v6 produce valid StepRecordV6 records?"""
	from training import play_games_with_rollouts_v6
	network = _make_network()
	network.eval()
	fails = []
	def fail(msg):
		fails.append(msg)
	try:
		records, advantages, _ = play_games_with_rollouts_v6(
			network, num_games=5, num_players=NUM_PLAYERS,
			rollouts_per_state=10, training_seats=NUM_PLAYERS)
	except Exception as e:
		fail(f"play_games_with_rollouts_v6 raised: {e}")
		print(f"  Probe 6 (full game loop):      FAIL  {e}")
		return False
	if len(records) == 0:
		fail("no records produced")
	if len(records) != len(advantages):
		fail(f"records ({len(records)}) != advantages ({len(advantages)})")
	# Validate records
	for i, rec in enumerate(records[:50]):
		if not isinstance(rec, StepRecordV6):
			fail(f"record {i} is {type(rec)}, not StepRecordV6")
			break
		if rec.state.shape != (INPUT_SIZE_V6,):
			fail(f"record {i} state shape {rec.state.shape}")
			break
		if not (0 <= rec.action < FLAT_ACTION_SIZE):
			fail(f"record {i} action {rec.action} out of range")
			break
		if rec.mask.shape != (FLAT_ACTION_SIZE,):
			fail(f"record {i} mask shape {rec.mask.shape}")
			break
		if not rec.mask[rec.action]:
			fail(f"record {i} action {rec.action} not in mask")
			break
	# Advantages should not all be identical
	if len(set(advantages)) <= 1 and len(advantages) > 10:
		fail("all advantages are identical")
	# Should be able to build a batch and run PPO
	batch = prepare_ppo_batch_v6(records, advantages)
	if batch is None:
		fail("prepare_ppo_batch_v6 returned None on game records")
	else:
		optimizer = torch.optim.Adam(network.parameters(), lr=LR)
		network.train()
		metrics = ppo_update_v6(network, optimizer, batch,
			clip_epsilon=CLIP_EPSILON, entropy_bonus=ENTROPY_BONUS,
			value_loss_coeff=VALUE_LOSS_COEFF)
		if all(v == 0.0 for v in metrics.values()):
			fail("all PPO metrics are 0 on game records")
	passed = len(fails) == 0
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 6 (full game loop):      {status}  {len(records)} records, {len(set(advantages))} unique advantages")
	for f in fails[:5]:
		print(f"    - {f}")
	return passed

# --- Probe 0b: S&S round-trip ---

def _create_sns_state():
	"""Create a state where S&S is legal for current player.
	Returns (game, player) or None."""
	for _ in range(200):
		game = _fresh_round()
		for _ in range(15):
			if game.phase not in (Phase.TURN, Phase.SNS_PLAY):
				break
			p = game.current_player
			hand = game.players[p].hand
			legal_plays = get_legal_plays(hand, game.current_play)
			if game.phase == Phase.SNS_PLAY:
				if legal_plays:
					s, e = random.choice(legal_plays)
					game.apply_play(s, e)
				continue
			if legal_plays:
				s, e = random.choice(legal_plays)
				game.apply_play(s, e)
			elif game.current_play is not None:
				game.apply_scout(True, False, 0)
			else:
				break
			# Check if current player now has S&S available
			if (game.phase == Phase.TURN and game.current_play is not None
					and game.players[game.current_player].sns_available):
				return game, game.current_player
	return None

def probe_sns_roundtrip(n_iters=None, n_games=None):
	"""Verify S&S mask entries decode correctly and match the legality check."""
	n_checks = 100
	fails = []
	sns_found = 0
	def fail(msg):
		fails.append(msg)
	for _ in range(n_checks):
		result = _create_sns_state()
		if result is None:
			continue
		game, player = result
		hand = game.players[player].hand
		play_cards = list(game.current_play.cards)
		legal_plays = get_legal_plays(hand, game.current_play)
		ho = random.randint(0, HAND_SLOTS_V6 - 1)
		mask = get_flat_action_mask(game, player, legal_plays, ho)
		# Check S&S region [320..383]
		sns_mask = mask[320:384]
		if not sns_mask.any():
			continue
		sns_found += 1
		H = HAND_SLOTS_V6
		for idx in range(64):
			if not sns_mask[idx]:
				continue
			action_idx = 320 + idx
			decoded = decode_flat_action(action_idx, ho)
			if decoded["type"] != "sns":
				fail(f"action {action_idx} decoded as {decoded['type']}, expected sns")
				break
			# Reconstruct what the mask builder checks
			card_choice = idx // H
			left_end = card_choice < 2
			flip = card_choice % 2 == 1
			insert_pos = decoded["insert_pos"]
			remaining = list(play_cards)
			card = remaining.pop(0) if left_end else remaining.pop()
			if flip:
				card = (card[1], card[0])
			reduced_play = Play.from_cards(remaining) if remaining else None
			new_hand = hand[:insert_pos] + [card] + hand[insert_pos:]
			if not _has_any_legal_play(new_hand, reduced_play):
				fail(f"S&S action {action_idx} (card_choice={card_choice}, pos={insert_pos}) "
					 f"decoded as legal but _has_any_legal_play returns False")
				break
		# Verify false entries are actually illegal
		for idx in range(64):
			if sns_mask[idx]:
				continue
			card_choice = idx // H
			slot = idx % H
			pos = (slot - ho) % H
			# Only check positions within hand range
			if pos > len(hand):
				continue
			left_end = card_choice < 2
			flip = card_choice % 2 == 1
			play_len = len(play_cards)
			# Skip if card_choice refers to right end of a single-card play
			if card_choice >= 2 and play_len <= 1:
				continue
			remaining = list(play_cards)
			card = remaining.pop(0) if left_end else remaining.pop()
			if flip:
				card = (card[1], card[0])
			reduced_play = Play.from_cards(remaining) if remaining else None
			new_hand = hand[:pos] + [card] + hand[pos:]
			if _has_any_legal_play(new_hand, reduced_play):
				fail(f"S&S pos={pos} card_choice={card_choice} is legal but mask says False")
				break
	if sns_found == 0:
		fail("couldn't create any S&S-legal states")
	passed = len(fails) == 0
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 0b (S&S round-trip):     {status}  {sns_found} states checked")
	for f in fails[:5]:
		print(f"    - {f}")
	return passed

# --- Probe 3b: Play-length preference ---

def probe_play_length(n_iters=100, n_games=50):
	"""Can the flat head learn to prefer longer plays?
	Round-start states only (no scout confound). Reward = play_length / hand_size."""
	network = _make_network()
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)
	def collect_play():
		"""Sample a play action from a fresh round, return (record, length) or None."""
		game = _fresh_round()
		player = game.current_player
		hand = game.players[player].hand
		sample = _sample_action(network, game, player)
		if sample is None:
			return None
		decoded = decode_flat_action(sample["action"], sample["ho"])
		if decoded["type"] != "play":
			return None
		length = (decoded["end"] - decoded["start"]) % HAND_SLOTS_V6 + 1
		# Only count if end >= start (valid contiguous play)
		if decoded["end"] < decoded["start"]:
			return None
		length = decoded["end"] - decoded["start"] + 1
		reward = length / max(len(hand), 1)
		return _make_record(sample, reward=reward, game_id=0, player=player), length
	# Initial avg length
	network.eval()
	init_lengths = []
	for _ in range(n_games * 2):
		result = collect_play()
		if result:
			_, length = result
			init_lengths.append(length)
	init_avg = sum(init_lengths) / max(len(init_lengths), 1)
	# Train
	for it in range(n_iters):
		network.eval()
		records = []
		for g in range(n_games):
			result = collect_play()
			if result:
				rec, _ = result
				rec = StepRecordV6(
					state=rec.state, action=rec.action, mask=rec.mask,
					old_log_prob=rec.old_log_prob, value=rec.value,
					reward=rec.reward, player=rec.player, round_num=0,
					game_id=g, hand_offset=rec.hand_offset,
					play_length=None, scout_quality=None,
				)
				records.append(rec)
		if records:
			_train_iteration(network, optimizer, records)
	# Final avg length
	network.eval()
	final_lengths = []
	for _ in range(n_games * 2):
		result = collect_play()
		if result:
			_, length = result
			final_lengths.append(length)
	final_avg = sum(final_lengths) / max(len(final_lengths), 1)
	passed = final_avg > init_avg + 0.3
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 3b (play length):        {status}  avg_length {init_avg:.2f} -> {final_avg:.2f}  (target: +0.3)")
	return passed

# --- Probe 5b: Rotation state content ---

def probe_rotation_content(n_iters=None, n_games=None):
	"""Verify rotation augmentation shifts hand slot content correctly."""
	H = HAND_SLOTS_V6
	N = NUM_VALUES_V6
	slot_size = N + 2
	fails = []
	def fail(msg):
		fails.append(msg)
	for _ in range(50):
		game = _fresh_round()
		player = game.current_player
		ho = random.randint(0, H - 1)
		state = encode_state_v6(game, player, ho)
		for k in range(1, H):
			shift = HAND_SHIFT[k]
			rotated = state[shift]
			# Each hand slot's content should have moved by k positions
			for s in range(H):
				orig_slot = s
				new_slot = (s + k) % H
				orig_start = orig_slot * slot_size
				new_start = new_slot * slot_size
				orig_vals = state[orig_start:orig_start + slot_size]
				rot_vals = rotated[new_start:new_start + slot_size]
				if not torch.allclose(orig_vals, rot_vals):
					fail(f"ho={ho} k={k}: slot {orig_slot} content doesn't match "
						 f"rotated slot {new_slot}")
					break
			else:
				# Also check bottom-face scalars shifted
				bottom_offset = H * slot_size
				for s in range(H):
					orig_val = state[bottom_offset + s]
					rot_val = rotated[bottom_offset + (s + k) % H]
					if abs(orig_val.item() - rot_val.item()) > 1e-6:
						fail(f"ho={ho} k={k}: bottom scalar slot {s} doesn't match "
							 f"rotated slot {(s + k) % H}")
						break
				else:
					# Non-hand portion should be unchanged
					non_hand_start = H * slot_size + H
					if not torch.allclose(state[non_hand_start:], rotated[non_hand_start:]):
						fail(f"ho={ho} k={k}: non-hand portion changed after rotation")
					continue
			break
		if fails:
			break
	passed = len(fails) == 0
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 5b (rotation content):   {status}")
	for f in fails[:5]:
		print(f"    - {f}")
	return passed

# --- Probe 10: Scout insert quality ---

def _hand_quality(hand):
	"""Score a hand by its longest available play (no opponent play to beat)."""
	plays = get_legal_plays(hand, None)
	if not plays:
		return 0
	return max(e - s + 1 for s, e in plays)

def _sample_scout_only(network, game, player):
	"""Sample a scout action (not play/S&S) from the flat head.
	Returns dict with state, action, mask, log_prob, value, hand_offset, decoded, or None."""
	hand = game.players[player].hand
	legal_plays = get_legal_plays(hand, game.current_play)
	ho = random.randint(0, HAND_SLOTS_V6 - 1)
	state = encode_state_v6(game, player, ho)
	mask = get_flat_action_mask(game, player, legal_plays, ho)
	# Restrict to scout region only (256-319)
	scout_mask = mask.clone()
	scout_mask[:256] = False
	scout_mask[320:] = False
	if not scout_mask.any():
		return None
	with torch.no_grad():
		hidden = network(state)
		value = network.value(hidden).item()
		logits = network.policy_logits(hidden)
		action, _ = masked_sample(logits, scout_mask)
		masked_logits = logits.masked_fill(~scout_mask, float('-inf'))
		log_prob = torch.log_softmax(masked_logits, dim=-1)[action].item()
	decoded = decode_flat_action(action, ho)
	# Resolve the actual scouted card
	play_cards = list(game.current_play.cards)
	card = play_cards[0] if decoded["left_end"] else play_cards[-1]
	if decoded["flip"]:
		card = (card[1], card[0])
	return {
		"state": state, "action": action, "mask": scout_mask.numpy(),
		"log_prob": log_prob, "value": value, "ho": ho,
		"insert_pos": decoded["insert_pos"], "card": card, "hand": list(hand),
	}

def probe_scout_quality(n_iters=100, n_games=50):
	"""Can the flat head learn to pick scout insertion positions that improve hand quality?
	Mid-round states where scouting is legal. Reward based on hand quality at chosen
	position relative to best/worst possible. Same metric as old probe 5."""
	network = _make_network()
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)

	def insert_qualities(hand, card):
		results = []
		for pos in range(len(hand) + 1):
			new_hand = hand[:pos] + [card] + hand[pos:]
			q = _hand_quality(new_hand)
			results.append((pos, q))
		qs = [q for _, q in results]
		return results, min(qs), max(qs)

	def eval_quality(n_samples=200):
		network.eval()
		total_q, max_q_sum, n = 0, 0, 0
		for _ in range(n_samples):
			result = _mid_round_state()
			if result is None:
				continue
			game, player = result
			sample = _sample_scout_only(network, game, player)
			if sample is None:
				continue
			pos = sample["insert_pos"]
			hand, card = sample["hand"], sample["card"]
			chosen_hand = hand[:pos] + [card] + hand[pos:]
			chosen_q = _hand_quality(chosen_hand)
			_, _, max_q = insert_qualities(hand, card)
			total_q += chosen_q
			max_q_sum += max_q
			n += 1
		if n == 0:
			return 0.0, 0.0, 0
		return total_q / n, max_q_sum / n, n

	init_q, init_max, init_n = eval_quality()

	for it in range(n_iters):
		network.eval()
		records = []
		for g in range(n_games):
			result = _mid_round_state()
			if result is None:
				continue
			game, player = result
			sample = _sample_scout_only(network, game, player)
			if sample is None:
				continue
			pos = sample["insert_pos"]
			hand, card = sample["hand"], sample["card"]
			_, min_q, max_q = insert_qualities(hand, card)
			chosen_hand = hand[:pos] + [card] + hand[pos:]
			chosen_q = _hand_quality(chosen_hand)
			if max_q == min_q:
				reward = 0.0
			else:
				reward = (chosen_q - min_q) / (max_q - min_q) * 2.0 - 1.0
			records.append(_make_record(sample, reward=reward, game_id=g, player=player))
		if records:
			_train_iteration(network, optimizer, records, augment=True)

	final_q, final_max, final_n = eval_quality()

	passed = final_q > init_q + 0.1
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 10 (scout quality):      {status}  chosen_q {init_q:.2f} -> {final_q:.2f}  "
		  f"(max={final_max:.2f}, n={final_n})  (target: +0.1)")
	return passed

# --- Probe 11: Scout adjacent matching ---

def probe_scout_adjacent(n_iters=100, n_games=50):
	"""Can the flat head learn to place a scouted card next to a matching value?
	Simpler pattern test than probe 10. Binary reward: +1 if adjacent match, -1 otherwise."""
	network = _make_network()
	optimizer = torch.optim.Adam(network.parameters(), lr=LR)

	def is_adjacent_match(hand, card, pos):
		val = card[0]
		if pos > 0 and hand[pos - 1][0] == val:
			return True
		if pos < len(hand) and hand[pos][0] == val:
			return True
		return False

	def has_any_match(hand, card):
		return any(c[0] == card[0] for c in hand)

	def eval_rate(n_samples=200):
		network.eval()
		n_adj, n_total = 0, 0
		for _ in range(n_samples):
			result = _mid_round_state()
			if result is None:
				continue
			game, player = result
			sample = _sample_scout_only(network, game, player)
			if sample is None:
				continue
			if not has_any_match(sample["hand"], sample["card"]):
				continue
			n_total += 1
			if is_adjacent_match(sample["hand"], sample["card"], sample["insert_pos"]):
				n_adj += 1
		return n_adj / max(n_total, 1), n_total

	init_rate, init_n = eval_rate()

	for it in range(n_iters):
		network.eval()
		records = []
		for g in range(n_games):
			result = _mid_round_state()
			if result is None:
				continue
			game, player = result
			sample = _sample_scout_only(network, game, player)
			if sample is None:
				continue
			if not has_any_match(sample["hand"], sample["card"]):
				continue
			adj = is_adjacent_match(sample["hand"], sample["card"], sample["insert_pos"])
			reward = 1.0 if adj else -1.0
			records.append(_make_record(sample, reward=reward, game_id=g, player=player))
		if records:
			_train_iteration(network, optimizer, records, augment=True)

	final_rate, final_n = eval_rate()

	passed = final_rate > init_rate + 0.05
	status = "PASS" if passed else "FAIL"
	print(f"  Probe 11 (scout adjacent):     {status}  P(adj) {init_rate:.3f} -> {final_rate:.3f}  "
		  f"(n={final_n})  (target: +0.05)")
	return passed

# --- Runner ---

ALL_PROBES = {
	0: ("encoding sanity", probe_encoding),
	1: ("forward pass", probe_forward),
	2: ("value learning", probe_value),
	3: ("play preference", probe_policy_preference),
	4: ("PPO mechanics", probe_ppo_mechanics),
	5: ("rotation augmentation", probe_rotation),
	6: ("full game loop", probe_full_game),
	7: ("S&S round-trip", probe_sns_roundtrip),
	8: ("play length preference", probe_play_length),
	9: ("rotation content", probe_rotation_content),
	10: ("scout insert quality", probe_scout_quality),
	11: ("scout adjacent match", probe_scout_adjacent),
}

def main():
	import argparse
	parser = argparse.ArgumentParser(description="V6 probe environments for Scout NN")
	parser.add_argument("--iters", type=int, default=100, help="Training iterations (probes 2-3)")
	parser.add_argument("--games", type=int, default=50, help="Games per iteration (probes 2-3)")
	parser.add_argument("--probe", type=int, nargs="*", default=None,
		help="Run specific probe(s) by number. Default: run all.")
	parser.add_argument("--layers", type=int, nargs="+", default=None,
		help="Override network layer sizes")
	args = parser.parse_args()
	global LAYER_SIZES
	if args.layers:
		LAYER_SIZES = args.layers
	probe_nums = args.probe if args.probe else sorted(ALL_PROBES.keys())
	print(f"Running v6 probes {probe_nums} (iters={args.iters}, games={args.games}, layers={LAYER_SIZES})")
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
