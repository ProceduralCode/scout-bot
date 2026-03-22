"""Diagnostic experiments to isolate why probe 5b (adjacent matching) fails.

Tests:
  A. Supervised probe — can the network learn adjacent matching with cross-entropy?
     If YES: trunk features suffice, PPO optimization is the bottleneck.
     If NO: the architecture can't represent the mapping.

  B. Value loss ablation — run probe 5b with value_loss_coeff=0.
     If helps: shared trunk interference is a major factor.

  C. Fixed hand_offset — run probe 5b with ho=0 always.
     If helps: the 20/21 slot misalignment matters.

  D. Simple conditional — "insert at pos 0 if scouted value <= 5, else pos=len(hand)"
     If PASS: PPO can learn conditionals, adjacent matching is too complex.
     If FAIL: PPO can't learn any conditional scout insertion.

  F. Standalone MLP — raw 475-dim state → MLP → 21 insert logits, no shared trunk.
     If PASS: encoding supports the task; trunk/head architecture is the bottleneck.
     If FAIL: encoding itself can't represent adjacent matching for any architecture.

  G. MLP scout head — shared trunk + 2-layer MLP head (replaces linear head).
     If PASS: trunk features are usable, linear head lacks capacity.
     If FAIL (and F passes): trunk compresses away the needed information.

  H. Standalone MLP with fixed ho=0, po=0 — eliminates rotation complexity.
     If PASS (and F fails): positional rotations are the bottleneck.
     If FAIL: encoding structure itself can't represent adjacent matching.

Usage: python scout-bot/probe_diagnostic.py [--test A|B|C|D|E|F|G|H|all] [--iters N] [--games N]
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
    get_scout_insert_mask, decode_slot_to_hand_index,
    HAND_SLOTS, PLAY_SLOTS, ACTION_TYPE_SIZE, SCOUT_INSERT_SIZE,
    CARD_VALUES, INPUT_SIZE,
)
from network import ScoutNetwork, masked_sample, CONDITIONING_SIZE
from training import StepRecord, compute_gae, prepare_ppo_batch, ppo_update

NUM_PLAYERS = 4
LR = 3e-4
PPO_EPOCHS = 4
ENTROPY_BONUS = 0.01
CLIP_EPSILON = 0.2
VALUE_LOSS_COEFF = 0.25

# --- Shared helpers (copied from probe.py to keep standalone) ---

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

def _encode(game, player=1, fixed_ho=None):
    ho = fixed_ho if fixed_ho is not None else random.randint(0, HAND_SLOTS - 1)
    po = random.randint(0, PLAY_SLOTS - 1) if game.current_play else 0
    return encode_state(game, player, ho, po), ho, po

def _sample_scout(network, game, player=1, fixed_ho=None):
    hand = game.players[player].hand
    play_cards = game.current_play.cards
    if len(play_cards) == 0:
        return None
    card = play_cards[0]
    action_type = 1

    state, ho, po = _encode(game, player, fixed_ho=fixed_ho)
    with torch.no_grad():
        hidden = network(state)
        value = network.value(hidden).item()
        at_logits = network.action_type_logits(hidden)
        legal_plays = get_legal_plays(hand, game.current_play)
        at_mask = get_action_type_mask(game, legal_plays)
        si_logits = network.scout_insert_logits(hidden, action_type)
        si_mask = get_scout_insert_mask(game, ho)
        if not si_mask.any():
            return None
        insert_slot, _ = masked_sample(si_logits, torch.from_numpy(si_mask))
        insert_pos = (insert_slot - ho) % SCOUT_INSERT_SIZE

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

def _eval_adj_rate(network, n_samples=500, fixed_ho=None):
    network.eval()
    n_adj, n_total = 0, 0
    for _ in range(n_samples):
        game = _mid_round_state()
        if game is None:
            continue
        sample = _sample_scout(network, game, fixed_ho=fixed_ho)
        if sample is None:
            continue
        if not _has_any_match(sample["hand"], sample["card"]):
            continue
        n_total += 1
        if _is_adjacent_match(sample["hand"], sample["card"], sample["insert_pos"]):
            n_adj += 1
    return n_adj / max(n_total, 1), n_total

def _train_iteration(network, optimizer, records, value_loss_coeff=VALUE_LOSS_COEFF, verbose=False):
    advantages, returns, adv_std = compute_gae(records, gamma=0.99, lam=0.95)
    if verbose:
        rewards = [r.reward for r in records]
        mean_r = sum(rewards) / len(rewards)
        print(f"    records={len(records)}  mean_reward={mean_r:+.3f}  adv_std={adv_std:.4f}")
    batch = prepare_ppo_batch(records, advantages, returns=returns)
    for _ in range(PPO_EPOCHS):
        network.train()
        ppo_update(
            network, optimizer, batch,
            clip_epsilon=CLIP_EPSILON, entropy_bonus=ENTROPY_BONUS,
            value_loss_coeff=value_loss_coeff,
        )

# ============================================================
# Test A: Supervised probe
# ============================================================

def _find_best_adjacent_slots(hand, card, ho):
    """Find all insert slots that are adjacent to a matching card value."""
    val = card[0]
    good_slots = []
    for pos in range(len(hand) + 1):
        if _is_adjacent_match(hand, card, pos):
            slot = (ho + pos) % SCOUT_INSERT_SIZE
            good_slots.append(slot)
    return good_slots

def test_supervised(n_iters=300, n_games=500, layer_sizes=None):
    """Train scout head with cross-entropy on correct adjacent positions.
    Uses masked logits, higher LR, multiple passes per batch, gradient diagnostics."""
    if layer_sizes is None:
        layer_sizes = [64, 32]
    print(f"\n=== Test A: Supervised probe (layers={layer_sizes}) ===")

    network = ScoutNetwork(layer_sizes=layer_sizes)
    # Higher LR for supervised — no trust region constraint
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-3)

    # Eval before training
    init_rate, _ = _eval_adj_rate(network)
    print(f"  Initial adj rate: {init_rate:.3f}")

    for it in range(n_iters):
        network.eval()
        states_list = []
        target_distributions = []
        action_types_list = []
        masks_list = []

        for g in range(n_games):
            game = _mid_round_state()
            if game is None:
                continue
            hand = game.players[1].hand
            play_cards = game.current_play.cards
            if len(play_cards) == 0:
                continue
            card = play_cards[0]
            if not _has_any_match(hand, card):
                continue

            ho = random.randint(0, HAND_SLOTS - 1)
            po = random.randint(0, PLAY_SLOTS - 1) if game.current_play else 0
            state = encode_state(game, 1, ho, po)
            si_mask = get_scout_insert_mask(game, ho)

            good_slots = _find_best_adjacent_slots(hand, card, ho)
            if not good_slots:
                continue

            target = torch.zeros(SCOUT_INSERT_SIZE)
            for s in good_slots:
                target[s] = 1.0
            target = target / target.sum()

            states_list.append(state)
            target_distributions.append(target)
            action_types_list.append(1)
            masks_list.append(torch.from_numpy(si_mask))

        if not states_list:
            continue

        states = torch.stack(states_list)
        targets = torch.stack(target_distributions)
        masks = torch.stack(masks_list)
        at_tensor = torch.tensor(action_types_list, dtype=torch.long)

        # Multiple passes over the same batch (standard supervised approach)
        for epoch in range(10):
            network.train()
            hidden = network(states)
            at_oh = F.one_hot(at_tensor, ACTION_TYPE_SIZE).float()
            start_oh = torch.zeros(len(states_list), 20)
            cond = torch.cat([hidden, at_oh, start_oh], dim=1)
            logits = network.scout_insert_head(cond)

            masked_logits = logits.masked_fill(~masks, float('-inf'))
            log_probs = F.log_softmax(masked_logits, dim=-1)
            loss = -(targets * log_probs).nan_to_num(0.0).sum(dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()

            # Gradient diagnostic on first iter
            if it == 0 and epoch == 0:
                trunk_grad = sum(p.grad.abs().mean().item() for p in network.shared.parameters() if p.grad is not None)
                head_grad = network.scout_insert_head.weight.grad.abs().mean().item()
                print(f"  Gradient check: trunk_mean_abs={trunk_grad:.6f}  head_mean_abs={head_grad:.6f}")

            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.5)
            optimizer.step()

        if it % 50 == 0 or it == n_iters - 1:
            rate, n = _eval_adj_rate(network)
            print(f"  iter {it:3d}  loss={loss.item():.4f}  adj_rate={rate:.3f} (n={n})")

    final_rate, _ = _eval_adj_rate(network)
    passed = final_rate > init_rate + 0.1
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}  {init_rate:.3f} -> {final_rate:.3f}")
    return passed

# ============================================================
# Test B: Value loss ablation
# ============================================================

def test_no_value_loss(n_iters=300, n_games=500, layer_sizes=None):
    """Run probe 5b with value_loss_coeff=0 to eliminate trunk gradient interference."""
    if layer_sizes is None:
        layer_sizes = [64, 32]
    print(f"\n=== Test B: No value loss (layers={layer_sizes}) ===")

    network = ScoutNetwork(layer_sizes=layer_sizes)
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
            _train_iteration(network, optimizer, records, value_loss_coeff=0.0,
                           verbose=(it < 3 or it == n_iters - 1 or it % 50 == 0))
            if it % 50 == 0 or it == n_iters - 1:
                rate, n = _eval_adj_rate(network)
                print(f"  iter {it:3d}  adj_rate={rate:.3f} (n={n})")

    final_rate, _ = _eval_adj_rate(network)
    passed = final_rate > init_rate + 0.05
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}  {init_rate:.3f} -> {final_rate:.3f}")
    return passed

# ============================================================
# Test C: Fixed hand_offset=0
# ============================================================

def test_fixed_offset(n_iters=300, n_games=500, layer_sizes=None):
    """Run probe 5b with ho=0 always — eliminates 20/21 misalignment."""
    if layer_sizes is None:
        layer_sizes = [64, 32]
    print(f"\n=== Test C: Fixed hand_offset=0 (layers={layer_sizes}) ===")

    network = ScoutNetwork(layer_sizes=layer_sizes)
    optimizer = torch.optim.Adam(network.parameters(), lr=LR)

    init_rate, _ = _eval_adj_rate(network, fixed_ho=0)
    print(f"  Initial adj rate: {init_rate:.3f}")

    for it in range(n_iters):
        network.eval()
        records = []
        for g in range(n_games):
            game = _mid_round_state()
            if game is None:
                continue
            sample = _sample_scout(network, game, fixed_ho=0)
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
            _train_iteration(network, optimizer, records,
                           verbose=(it < 3 or it == n_iters - 1 or it % 50 == 0))
            if it % 50 == 0 or it == n_iters - 1:
                rate, n = _eval_adj_rate(network, fixed_ho=0)
                print(f"  iter {it:3d}  adj_rate={rate:.3f} (n={n})")

    final_rate, _ = _eval_adj_rate(network, fixed_ho=0)
    passed = final_rate > init_rate + 0.05
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}  {init_rate:.3f} -> {final_rate:.3f}")
    return passed

# ============================================================
# Test D: Simple conditional ("even/odd" split)
# ============================================================

def test_simple_conditional(n_iters=300, n_games=500, layer_sizes=None):
    """Can PPO learn: 'if scouted value <= 5, insert at pos 0; else insert at pos=len(hand)'?
    This is a binary conditional — much simpler than adjacent matching."""
    if layer_sizes is None:
        layer_sizes = [64, 32]
    print(f"\n=== Test D: Simple conditional (layers={layer_sizes}) ===")

    network = ScoutNetwork(layer_sizes=layer_sizes)
    optimizer = torch.optim.Adam(network.parameters(), lr=LR)

    def _target_pos(hand, card):
        """Target: pos 0 if value<=5, pos len(hand) if value>5."""
        return 0 if card[0] <= 5 else len(hand)

    def _eval_conditional_rate(network, n_samples=500):
        network.eval()
        n_correct, n_total = 0, 0
        for _ in range(n_samples):
            game = _mid_round_state()
            if game is None:
                continue
            sample = _sample_scout(network, game)
            if sample is None:
                continue
            n_total += 1
            target = _target_pos(sample["hand"], sample["card"])
            if sample["insert_pos"] == target:
                n_correct += 1
        return n_correct / max(n_total, 1), n_total

    init_rate, _ = _eval_conditional_rate(network)
    print(f"  Initial correct rate: {init_rate:.3f}")

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
            target = _target_pos(sample["hand"], sample["card"])
            reward = 1.0 if sample["insert_pos"] == target else -1.0
            records.append(_make_scout_record(sample, reward=reward, game_id=g))
        if records:
            _train_iteration(network, optimizer, records,
                           verbose=(it < 3 or it == n_iters - 1 or it % 50 == 0))
            if it % 50 == 0 or it == n_iters - 1:
                rate, n = _eval_conditional_rate(network)
                print(f"  iter {it:3d}  correct_rate={rate:.3f} (n={n})")

    final_rate, _ = _eval_conditional_rate(network)
    passed = final_rate > init_rate + 0.1
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}  {init_rate:.3f} -> {final_rate:.3f}")
    return passed

# ============================================================
# Test E: Both fixes combined (no value loss + fixed ho)
# ============================================================

def test_combined_fixes(n_iters=300, n_games=500, layer_sizes=None):
    """Probe 5b with value_loss_coeff=0 AND ho=0."""
    if layer_sizes is None:
        layer_sizes = [64, 32]
    print(f"\n=== Test E: Combined (no value loss + fixed ho) (layers={layer_sizes}) ===")

    network = ScoutNetwork(layer_sizes=layer_sizes)
    optimizer = torch.optim.Adam(network.parameters(), lr=LR)

    init_rate, _ = _eval_adj_rate(network, fixed_ho=0)
    print(f"  Initial adj rate: {init_rate:.3f}")

    for it in range(n_iters):
        network.eval()
        records = []
        for g in range(n_games):
            game = _mid_round_state()
            if game is None:
                continue
            sample = _sample_scout(network, game, fixed_ho=0)
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
            _train_iteration(network, optimizer, records, value_loss_coeff=0.0,
                           verbose=(it < 3 or it == n_iters - 1 or it % 50 == 0))
            if it % 50 == 0 or it == n_iters - 1:
                rate, n = _eval_adj_rate(network, fixed_ho=0)
                print(f"  iter {it:3d}  adj_rate={rate:.3f} (n={n})")

    final_rate, _ = _eval_adj_rate(network, fixed_ho=0)
    passed = final_rate > init_rate + 0.05
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}  {init_rate:.3f} -> {final_rate:.3f}")
    return passed

# ============================================================
# Test F: Standalone MLP (no shared trunk)
# ============================================================

def _eval_adj_rate_standalone(mlp, n_samples=500, fixed_ho=None, fixed_po=None):
	"""Eval adjacent matching rate for a standalone MLP (raw state → logits)."""
	mlp.eval()
	n_adj, n_total = 0, 0
	for _ in range(n_samples):
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
		ho = fixed_ho if fixed_ho is not None else random.randint(0, HAND_SLOTS - 1)
		po = fixed_po if fixed_po is not None else (random.randint(0, PLAY_SLOTS - 1) if game.current_play else 0)
		state = encode_state(game, 1, ho, po)
		si_mask = get_scout_insert_mask(game, ho)
		if not si_mask.any():
			continue
		with torch.no_grad():
			logits = mlp(state)
			insert_slot, _ = masked_sample(logits, torch.from_numpy(si_mask))
			insert_pos = (insert_slot - ho) % SCOUT_INSERT_SIZE
		n_total += 1
		if _is_adjacent_match(hand, card, insert_pos):
			n_adj += 1
	return n_adj / max(n_total, 1), n_total

def test_standalone_mlp(n_iters=300, n_games=500, layer_sizes=None, fixed_ho=None, fixed_po=None):
	"""Raw 475-dim state → MLP → 21 insert logits. No shared trunk.
	Tests whether the encoding itself supports adjacent matching."""
	if layer_sizes is None:
		layer_sizes = [256, 128]
	offset_label = ""
	if fixed_ho is not None or fixed_po is not None:
		offset_label = f" ho={fixed_ho} po={fixed_po}"
	print(f"\n=== Test F: Standalone MLP (layers={layer_sizes}{offset_label}) ===")
	layers = []
	prev = INPUT_SIZE
	for h in layer_sizes:
		layers.append(nn.Linear(prev, h))
		layers.append(nn.ReLU())
		prev = h
	layers.append(nn.Linear(prev, SCOUT_INSERT_SIZE))
	mlp = nn.Sequential(*layers)
	optimizer = torch.optim.Adam(mlp.parameters(), lr=3e-3)
	init_rate, _ = _eval_adj_rate_standalone(mlp, fixed_ho=fixed_ho, fixed_po=fixed_po)
	print(f"  Initial adj rate: {init_rate:.3f}")
	for it in range(n_iters):
		mlp.eval()
		states_list = []
		target_distributions = []
		masks_list = []
		for g in range(n_games):
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
			ho = fixed_ho if fixed_ho is not None else random.randint(0, HAND_SLOTS - 1)
			po = fixed_po if fixed_po is not None else (random.randint(0, PLAY_SLOTS - 1) if game.current_play else 0)
			state = encode_state(game, 1, ho, po)
			si_mask = get_scout_insert_mask(game, ho)
			good_slots = _find_best_adjacent_slots(hand, card, ho)
			if not good_slots:
				continue
			target = torch.zeros(SCOUT_INSERT_SIZE)
			for s in good_slots:
				target[s] = 1.0
			target = target / target.sum()
			states_list.append(state)
			target_distributions.append(target)
			masks_list.append(torch.from_numpy(si_mask))
		if not states_list:
			continue
		states = torch.stack(states_list)
		targets = torch.stack(target_distributions)
		masks = torch.stack(masks_list)
		for epoch in range(10):
			mlp.train()
			logits = mlp(states)
			masked_logits = logits.masked_fill(~masks, float('-inf'))
			log_probs = F.log_softmax(masked_logits, dim=-1)
			loss = -(targets * log_probs).nan_to_num(0.0).sum(dim=-1).mean()
			optimizer.zero_grad()
			loss.backward()
			if it == 0 and epoch == 0:
				grads = [p.grad.abs().mean().item() for p in mlp.parameters() if p.grad is not None]
				print(f"  Gradient check: mean_abs={sum(grads)/len(grads):.6f}")
			torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=0.5)
			optimizer.step()
		if it % 50 == 0 or it == n_iters - 1:
			rate, n = _eval_adj_rate_standalone(mlp, fixed_ho=fixed_ho, fixed_po=fixed_po)
			print(f"  iter {it:3d}  loss={loss.item():.4f}  adj_rate={rate:.3f} (n={n})")
	final_rate, _ = _eval_adj_rate_standalone(mlp, fixed_ho=fixed_ho, fixed_po=fixed_po)
	passed = final_rate > init_rate + 0.1
	print(f"  RESULT: {'PASS' if passed else 'FAIL'}  {init_rate:.3f} -> {final_rate:.3f}")
	return passed

# ============================================================
# Test G: MLP scout head (replaces linear head, keeps trunk)
# ============================================================

def test_mlp_scout_head(n_iters=300, n_games=500, layer_sizes=None):
	"""Replace linear scout head with 2-layer MLP, keep shared trunk.
	Tests whether trunk features are usable with more head capacity."""
	if layer_sizes is None:
		layer_sizes = [64, 32]
	print(f"\n=== Test G: MLP scout head (trunk={layer_sizes}) ===")
	network = ScoutNetwork(layer_sizes=layer_sizes)
	cond_size = layer_sizes[-1] + CONDITIONING_SIZE
	network.scout_insert_head = nn.Sequential(
		nn.Linear(cond_size, 64),
		nn.ReLU(),
		nn.Linear(64, SCOUT_INSERT_SIZE),
	)
	optimizer = torch.optim.Adam(network.parameters(), lr=3e-3)
	init_rate, _ = _eval_adj_rate(network)
	print(f"  Initial adj rate: {init_rate:.3f}")
	for it in range(n_iters):
		network.eval()
		states_list = []
		target_distributions = []
		action_types_list = []
		masks_list = []
		for g in range(n_games):
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
			ho = random.randint(0, HAND_SLOTS - 1)
			po = random.randint(0, PLAY_SLOTS - 1) if game.current_play else 0
			state = encode_state(game, 1, ho, po)
			si_mask = get_scout_insert_mask(game, ho)
			good_slots = _find_best_adjacent_slots(hand, card, ho)
			if not good_slots:
				continue
			target = torch.zeros(SCOUT_INSERT_SIZE)
			for s in good_slots:
				target[s] = 1.0
			target = target / target.sum()
			states_list.append(state)
			target_distributions.append(target)
			action_types_list.append(1)
			masks_list.append(torch.from_numpy(si_mask))
		if not states_list:
			continue
		states = torch.stack(states_list)
		targets = torch.stack(target_distributions)
		masks = torch.stack(masks_list)
		at_tensor = torch.tensor(action_types_list, dtype=torch.long)
		for epoch in range(10):
			network.train()
			hidden = network(states)
			at_oh = F.one_hot(at_tensor, ACTION_TYPE_SIZE).float()
			start_oh = torch.zeros(len(states_list), 20)
			cond = torch.cat([hidden, at_oh, start_oh], dim=1)
			logits = network.scout_insert_head(cond)
			masked_logits = logits.masked_fill(~masks, float('-inf'))
			log_probs = F.log_softmax(masked_logits, dim=-1)
			loss = -(targets * log_probs).nan_to_num(0.0).sum(dim=-1).mean()
			optimizer.zero_grad()
			loss.backward()
			if it == 0 and epoch == 0:
				trunk_grad = sum(p.grad.abs().mean().item() for p in network.shared.parameters() if p.grad is not None)
				head_grads = [p.grad.abs().mean().item() for p in network.scout_insert_head.parameters() if p.grad is not None]
				print(f"  Gradient check: trunk={trunk_grad:.6f}  head={sum(head_grads)/len(head_grads):.6f}")
			torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.5)
			optimizer.step()
		if it % 50 == 0 or it == n_iters - 1:
			rate, n = _eval_adj_rate(network)
			print(f"  iter {it:3d}  loss={loss.item():.4f}  adj_rate={rate:.3f} (n={n})")
	final_rate, _ = _eval_adj_rate(network)
	passed = final_rate > init_rate + 0.1
	print(f"  RESULT: {'PASS' if passed else 'FAIL'}  {init_rate:.3f} -> {final_rate:.3f}")
	return passed

# ============================================================
# Test S: Rotation complexity sweep
# ============================================================

def test_rotation_sweep(n_iters=200, n_games=300, layer_sizes=None):
	"""Train standalone MLPs with increasing rotation counts.
	Shows how adj_rate degrades as ho×po combinations increase."""
	if layer_sizes is None:
		layer_sizes = [256, 128]
	configs = [
		(1, 1),    # 1 combo
		(2, 2),    # 4 combos
		(4, 2),    # 8 combos
		(4, 4),    # 16 combos
		(8, 4),    # 32 combos
		(10, 5),   # 50 combos
		(10, 10),  # 100 combos
		(20, 10),  # 200 combos (full)
	]
	print(f"\n=== Test S: Rotation complexity sweep (layers={layer_sizes}) ===")
	print(f"  {'ho':>3s}  {'po':>3s}  {'combos':>6s}  {'adj_rate':>8s}  {'loss':>8s}")
	print(f"  {'-'*3}  {'-'*3}  {'-'*6}  {'-'*8}  {'-'*8}")
	results = []
	for ho_count, po_count in configs:
		# Build fresh MLP
		layers = []
		prev = INPUT_SIZE
		for h in layer_sizes:
			layers.append(nn.Linear(prev, h))
			layers.append(nn.ReLU())
			prev = h
		layers.append(nn.Linear(prev, SCOUT_INSERT_SIZE))
		mlp = nn.Sequential(*layers)
		optimizer = torch.optim.Adam(mlp.parameters(), lr=3e-3)
		# Train
		last_loss = 0.0
		for it in range(n_iters):
			mlp.eval()
			states_list, targets_list, masks_list = [], [], []
			for g in range(n_games):
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
				ho = random.randint(0, ho_count - 1)
				po = random.randint(0, po_count - 1)
				state = encode_state(game, 1, ho, po)
				si_mask = get_scout_insert_mask(game, ho)
				good_slots = _find_best_adjacent_slots(hand, card, ho)
				if not good_slots:
					continue
				target = torch.zeros(SCOUT_INSERT_SIZE)
				for s in good_slots:
					target[s] = 1.0
				target = target / target.sum()
				states_list.append(state)
				targets_list.append(target)
				masks_list.append(torch.from_numpy(si_mask))
			if not states_list:
				continue
			states = torch.stack(states_list)
			targets = torch.stack(targets_list)
			masks = torch.stack(masks_list)
			for epoch in range(10):
				mlp.train()
				logits = mlp(states)
				masked_logits = logits.masked_fill(~masks, float('-inf'))
				log_probs = F.log_softmax(masked_logits, dim=-1)
				loss = -(targets * log_probs).nan_to_num(0.0).sum(dim=-1).mean()
				optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=0.5)
				optimizer.step()
			last_loss = loss.item()
		# Eval
		mlp.eval()
		n_adj, n_total = 0, 0
		for _ in range(500):
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
			ho = random.randint(0, ho_count - 1)
			po = random.randint(0, po_count - 1)
			state = encode_state(game, 1, ho, po)
			si_mask = get_scout_insert_mask(game, ho)
			if not si_mask.any():
				continue
			with torch.no_grad():
				logits = mlp(state)
				insert_slot, _ = masked_sample(logits, torch.from_numpy(si_mask))
				insert_pos = (insert_slot - ho) % SCOUT_INSERT_SIZE
			n_total += 1
			if _is_adjacent_match(hand, card, insert_pos):
				n_adj += 1
		rate = n_adj / max(n_total, 1)
		combos = ho_count * po_count
		print(f"  {ho_count:3d}  {po_count:3d}  {combos:6d}  {rate:8.3f}  {last_loss:8.4f}")
		results.append((ho_count, po_count, combos, rate, last_loss))
	return results

# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all",
                       help="Which test(s) to run: A,B,C,D,E or 'all'")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    args = parser.parse_args()

    layers = args.layers or [64, 32]
    tests = args.test.upper().split(",") if args.test != "all" else ["A", "B", "C", "D", "E", "F", "G", "H"]

    results = []
    for t in tests:
        t = t.strip()
        if t == "A":
            results.append(("supervised", test_supervised(args.iters, args.games, layers)))
        elif t == "B":
            results.append(("no_value_loss", test_no_value_loss(args.iters, args.games, layers)))
        elif t == "C":
            results.append(("fixed_ho", test_fixed_offset(args.iters, args.games, layers)))
        elif t == "D":
            results.append(("simple_conditional", test_simple_conditional(args.iters, args.games, layers)))
        elif t == "E":
            results.append(("combined_fixes", test_combined_fixes(args.iters, args.games, layers)))
        elif t == "F":
            # --layers controls MLP hidden sizes; default [256, 128] if not specified
            f_layers = args.layers or [256, 128]
            results.append(("standalone_mlp", test_standalone_mlp(args.iters, args.games, f_layers)))
        elif t == "G":
            results.append(("mlp_scout_head", test_mlp_scout_head(args.iters, args.games, layers)))
        elif t == "H":
            f_layers = args.layers or [256, 128]
            results.append(("standalone_fixed", test_standalone_mlp(args.iters, args.games, f_layers, fixed_ho=0, fixed_po=0)))
        elif t == "S":
            f_layers = args.layers or [256, 128]
            test_rotation_sweep(args.iters, args.games, f_layers)
        else:
            print(f"Unknown test: {t}")

    print(f"\n{'='*50}")
    print("SUMMARY:")
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

if __name__ == "__main__":
    main()
