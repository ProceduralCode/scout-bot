"""Diagnostic: why is scout_quality probe MSE dropping but scout_play_len flat?

Checks:
1. Target distribution: do scout actions have meaningful variance?
2. Network predictions: does the network separate play vs scout outputs?
3. Per-position analysis: does the network differentiate scout insert positions?
4. Eval path: does _sample_scout actually produce different choices?
"""
import os, sys, random
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from game import Game
from encoding import (
	encode_state_v6, get_flat_action_mask, get_legal_plays, decode_flat_action,
	HAND_SLOTS_V6, INPUT_SIZE_V6, FLAT_ACTION_SIZE,
)
from network import FlatScoutNetwork, masked_sample

H = HAND_SLOTS_V6  # 16

def load_network(checkpoint_path):
	ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
	cfg = ckpt.get("config", {})
	net = FlatScoutNetwork(
		INPUT_SIZE_V6, cfg["layer_sizes"],
		encoding_version=6, attention=cfg.get("attention"))
	net.load_state_dict(ckpt["model_state"])
	net.eval()
	return net, cfg

def mid_round_state():
	"""Create a scoutable game state (player 1's turn, current play exists)."""
	for _ in range(100):
		game = Game(4)
		game.start_round()
		for p in range(4):
			game.submit_flip_decision(p, do_flip=random.random() < 0.5)
		hand = game.players[0].hand
		legal_plays = get_legal_plays(hand, game.current_play)
		if legal_plays:
			start, end = random.choice(legal_plays)
			game.apply_play(start, end)
			return game
	return None

def compute_scout_quality_target(game, player, action_idx, hand_offset):
	"""Reproduce the probe target computation for a single scout/S&S action."""
	offset = 256 if action_idx < 320 else 320
	idx = action_idx - offset
	card_choice = idx // H
	slot = idx % H
	insert_pos = (slot - hand_offset) % H
	left_end = card_choice < 2
	flip = card_choice % 2 == 1
	play_cards = game.current_play.cards
	card = play_cards[0] if left_end else play_cards[-1]
	if flip:
		card = (card[1], card[0])
	hand = list(game.players[player].hand)
	new_hand = hand[:insert_pos] + [card] + hand[insert_pos:]
	plays = get_legal_plays(new_hand, None)
	max_len = 1
	for s, e in plays:
		if s <= insert_pos <= e:
			max_len = max(max_len, e - s + 1)
	return max_len, insert_pos

def main():
	ckpt_path = os.path.join(SCRIPT_DIR, "bots", "v8_5", "latest.pt")
	net, cfg = load_network(ckpt_path)
	print(f"Loaded checkpoint from {ckpt_path}")
	print(f"Config: layers={cfg['layer_sizes']}, attention={cfg.get('attention')}")
	print()

	# === Check 1: Target distribution across many states ===
	print("=" * 60)
	print("CHECK 1: Target distribution for scout actions")
	print("=" * 60)
	all_targets = []
	all_play_targets = []
	n_states = 50
	for _ in range(n_states):
		game = mid_round_state()
		if game is None:
			continue
		player = game.current_player
		ho = random.randint(0, H - 1)
		legal_plays = get_legal_plays(game.players[player].hand, game.current_play)
		mask = get_flat_action_mask(game, player, legal_plays, ho)
		legal = torch.where(mask)[0].tolist()
		for a in legal:
			if a < 256:
				all_play_targets.append(0.0)
			else:
				max_len, _ = compute_scout_quality_target(game, player, a, ho)
				all_targets.append(max_len / 5.0)

	all_targets = np.array(all_targets)
	print(f"Scout/S&S targets: n={len(all_targets)}")
	print(f"  mean={all_targets.mean():.3f}, std={all_targets.std():.3f}")
	print(f"  min={all_targets.min():.3f}, max={all_targets.max():.3f}")
	vals, counts = np.unique(all_targets, return_counts=True)
	for v, c in zip(vals, counts):
		print(f"  target={v:.2f} (max_len={v*5:.0f}): {c} ({100*c/len(all_targets):.1f}%)")
	print(f"Play targets: n={len(all_play_targets)} (all 0.0)")
	print()

	# === Check 2: Network predictions by action type ===
	print("=" * 60)
	print("CHECK 2: Network predictions by action type (50 states)")
	print("=" * 60)
	play_preds = []
	scout_preds = []
	scout_targets_matched = []  # (target, prediction) pairs
	for _ in range(50):
		game = mid_round_state()
		if game is None:
			continue
		player = game.current_player
		ho = random.randint(0, H - 1)
		legal_plays = get_legal_plays(game.players[player].hand, game.current_play)
		state = encode_state_v6(game, player, ho)
		mask = get_flat_action_mask(game, player, legal_plays, ho)
		with torch.no_grad():
			hidden = net(state)
			logits = net.policy_logits(hidden)
		outputs = logits.numpy()
		legal = torch.where(mask)[0].tolist()
		for a in legal:
			if a < 256:
				play_preds.append(outputs[a])
			else:
				scout_preds.append(outputs[a])
				max_len, _ = compute_scout_quality_target(game, player, a, ho)
				scout_targets_matched.append((max_len / 5.0, outputs[a]))

	play_preds = np.array(play_preds)
	scout_preds = np.array(scout_preds)
	print(f"Play outputs:  mean={play_preds.mean():.4f}, std={play_preds.std():.4f}")
	print(f"Scout outputs: mean={scout_preds.mean():.4f}, std={scout_preds.std():.4f}")
	print(f"Gap (scout - play): {scout_preds.mean() - play_preds.mean():.4f}")
	print()

	# === Check 3: Correlation between target and prediction ===
	print("=" * 60)
	print("CHECK 3: Target vs prediction correlation for scout actions")
	print("=" * 60)
	targets_arr = np.array([t for t, p in scout_targets_matched])
	preds_arr = np.array([p for t, p in scout_targets_matched])
	if len(targets_arr) > 2:
		corr = np.corrcoef(targets_arr, preds_arr)[0, 1]
		print(f"Pearson correlation: {corr:.4f}")
		# Group by target value
		for tgt_val in sorted(set(targets_arr)):
			mask_t = targets_arr == tgt_val
			pred_mean = preds_arr[mask_t].mean()
			pred_std = preds_arr[mask_t].std()
			print(f"  target={tgt_val:.2f}: mean_pred={pred_mean:.4f}, std_pred={pred_std:.4f}, n={mask_t.sum()}")
	print()

	# === Check 4: Per-state scout position analysis ===
	print("=" * 60)
	print("CHECK 4: Per-state analysis (5 detailed states)")
	print("=" * 60)
	for state_i in range(5):
		game = mid_round_state()
		if game is None:
			continue
		player = game.current_player
		hand = game.players[player].hand
		ho = random.randint(0, H - 1)
		legal_plays = get_legal_plays(hand, game.current_play)
		state = encode_state_v6(game, player, ho)
		mask = get_flat_action_mask(game, player, legal_plays, ho)
		with torch.no_grad():
			hidden = net(state)
			logits = net.policy_logits(hidden)
		outputs = logits.numpy()

		print(f"\n--- State {state_i} ---")
		print(f"  Hand: {[(c[0]) for c in hand]} (showing values)")
		play_cards = game.current_play.cards
		print(f"  Current play: {[(c[0], c[1]) for c in play_cards]}")
		print(f"  Hand offset: {ho}")

		# Show play actions
		play_legal = [a for a in torch.where(mask)[0].tolist() if a < 256]
		scout_legal = [a for a in torch.where(mask)[0].tolist() if 256 <= a < 320]
		sns_legal = [a for a in torch.where(mask)[0].tolist() if a >= 320]
		print(f"  Legal: {len(play_legal)} plays, {len(scout_legal)} scouts, {len(sns_legal)} S&S")

		if play_legal:
			play_outs = [outputs[a] for a in play_legal]
			print(f"  Play output range: [{min(play_outs):.4f}, {max(play_outs):.4f}], mean={np.mean(play_outs):.4f}")

		# Show each scout action: position, card, target, prediction
		if scout_legal:
			print(f"  Scout actions:")
			scout_data = []
			for a in scout_legal:
				max_len, ipos = compute_scout_quality_target(game, player, a, ho)
				scout_data.append((a, ipos, max_len, max_len / 5.0, outputs[a]))
			scout_data.sort(key=lambda x: x[1])  # sort by insert position
			for a, ipos, ml, tgt, pred in scout_data:
				decoded = decode_flat_action(a, ho)
				side = "L" if decoded["left_end"] else "R"
				flip = "F" if decoded["flip"] else " "
				print(f"    pos={ipos:2d} {side}{flip} -> max_len={ml}, target={tgt:.2f}, pred={pred:.4f}")

	# === Check 5: What does masked_sample actually pick? ===
	print()
	print("=" * 60)
	print("CHECK 5: masked_sample action distribution (200 trials)")
	print("=" * 60)
	chosen_targets = []
	for _ in range(200):
		game = mid_round_state()
		if game is None:
			continue
		player = game.current_player
		hand = game.players[player].hand
		ho = random.randint(0, H - 1)
		legal_plays = get_legal_plays(hand, game.current_play)
		state = encode_state_v6(game, player, ho)
		mask = get_flat_action_mask(game, player, legal_plays, ho)
		scout_mask = mask.clone()
		scout_mask[:256] = False
		scout_mask[320:] = False
		if not scout_mask.any():
			continue
		with torch.no_grad():
			hidden = net(state)
			logits = net.policy_logits(hidden)
			action_idx, _ = masked_sample(logits, scout_mask)
		max_len, _ = compute_scout_quality_target(game, player, action_idx, ho)
		chosen_targets.append(max_len)

	chosen_targets = np.array(chosen_targets)
	print(f"Network-chosen scout quality: mean={chosen_targets.mean():.3f}, n={len(chosen_targets)}")
	vals, counts = np.unique(chosen_targets, return_counts=True)
	for v, c in zip(vals, counts):
		print(f"  max_len={v:.0f}: {c} ({100*c/len(chosen_targets):.1f}%)")

	# Compare with random baseline
	random_targets = []
	for _ in range(200):
		game = mid_round_state()
		if game is None:
			continue
		player = game.current_player
		hand = game.players[player].hand
		ho = random.randint(0, H - 1)
		legal_plays = get_legal_plays(hand, game.current_play)
		mask = get_flat_action_mask(game, player, legal_plays, ho)
		scout_mask = mask.clone()
		scout_mask[:256] = False
		scout_mask[320:] = False
		if not scout_mask.any():
			continue
		scout_legal = torch.where(scout_mask)[0].tolist()
		action_idx = random.choice(scout_legal)
		max_len, _ = compute_scout_quality_target(game, player, action_idx, ho)
		random_targets.append(max_len)

	random_targets = np.array(random_targets)
	print(f"Random-chosen scout quality:  mean={random_targets.mean():.3f}, n={len(random_targets)}")
	vals, counts = np.unique(random_targets, return_counts=True)
	for v, c in zip(vals, counts):
		print(f"  max_len={v:.0f}: {c} ({100*c/len(random_targets):.1f}%)")

if __name__ == "__main__":
	main()
