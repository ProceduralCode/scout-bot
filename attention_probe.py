"""Attention pattern analysis: extract and analyze self-attention weights
from FlatScoutNetwork's attention layers.

Reports attention entropy, cross-block mass (hand vs scout entities),
strongest edges, and head specialization. Answers whether the attention
mechanism is doing useful work or just acting as a mixing layer.

Usage:
	python -u attention_probe.py [checkpoint_dir]
	python -u attention_probe.py bots/v7_8
"""

import sys
import os
import random
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from network import FlatScoutNetwork
from encoding import (
	INPUT_SIZE_V6, HAND_SLOTS_V6, NUM_ENTITIES_V6,
	GLOBAL_START_V6,
	encode_state_v6, get_flat_action_mask, get_legal_plays,
)
from game import Game, Phase

H = HAND_SLOTS_V6  # 16
N_ENTITIES = NUM_ENTITIES_V6  # 20
N_HAND = 16  # entities 0-15
N_SCOUT = 4  # entities 16-19

ENTITY_NAMES = [f"hand_{i}" for i in range(N_HAND)] + [f"scout_{i}" for i in range(N_SCOUT)]


def load_checkpoint(path):
	checkpoint = torch.load(path, weights_only=False, map_location='cpu')
	cfg = checkpoint.get("config", {})
	ls = cfg.get("layer_sizes", [512, 256, 128])
	net = FlatScoutNetwork(INPUT_SIZE_V6, ls, encoding_version=6,
		attention=cfg.get("attention"))
	net.load_state_dict(checkpoint["model_state"])
	net.eval()
	iteration = checkpoint.get("iteration", "?")
	return net, cfg, iteration


def extract_attention_weights(net, states_t):
	"""Replay the attention path, capturing per-head attention weights.
	Returns list of [B, num_heads, 20, 20] tensors, one per attention layer."""
	x = states_t
	if x.ndim == 1:
		x = x.unsqueeze(0)
	entities = x[:, net.entity_indices]  # [B, 20, 13]
	pos = net.position_onehots.expand(x.shape[0], -1, -1)
	entities = torch.cat([entities, pos], dim=2)  # [B, 20, 33]
	entities = net.entity_proj(entities)  # [B, 20, d_model]
	all_weights = []
	for layer in net.attn_layers:
		residual = entities
		entities_normed = layer['norm'](entities)
		entities_out, weights = layer['attn'](
			entities_normed, entities_normed, entities_normed,
			average_attn_weights=False)
		all_weights.append(weights)  # [B, heads, 20, 20]
		entities = residual + entities_out
	return all_weights


def collect_states(n_games=30, n_players=4, turns_per_game=8):
	"""Collect mid-game states for analysis."""
	states = []
	for _ in range(n_games):
		game = Game(n_players)
		game.start_round()
		for p in list(game.flips_remaining):
			game.submit_flip_decision(p, random.random() < 0.5)
		for _ in range(turns_per_game):
			if game.phase in (Phase.GAME_OVER, Phase.ROUND_OVER):
				break
			if game.phase == Phase.FLIP_DECISION:
				for p in list(game.flips_remaining):
					game.submit_flip_decision(p, random.random() < 0.5)
				continue
			p = game.current_player
			hand = game.players[p].hand
			legal = get_legal_plays(hand, game.current_play)
			if game.phase == Phase.SNS_PLAY:
				if legal:
					s, e = random.choice(legal)
					game.apply_play(s, e)
				else:
					game._advance_turn()
			elif not legal:
				if game.current_play is not None and len(hand) > 0:
					left_end = random.random() < 0.5
					flip = random.random() < 0.5
					insert_pos = random.randint(0, len(hand))
					game.apply_scout(left_end, flip, insert_pos)
				else:
					game._advance_turn()
			else:
				r = random.random()
				if r < 0.6:
					s, e = random.choice(legal)
					game.apply_play(s, e)
				elif game.current_play is not None and len(hand) > 0:
					left_end = random.random() < 0.5
					flip = random.random() < 0.5
					insert_pos = random.randint(0, len(hand))
					game.apply_scout(left_end, flip, insert_pos)
				else:
					s, e = random.choice(legal)
					game.apply_play(s, e)
		if game.phase not in (Phase.TURN, Phase.SNS_PLAY):
			continue
		p = game.current_player
		hand = game.players[p].hand
		if not hand:
			continue
		hand_offset = random.randint(0, H - 1)
		forced_play = game.phase == Phase.SNS_PLAY
		state = encode_state_v6(game, p, hand_offset, forced_play=forced_play)
		states.append(state)
	return states


def attention_entropy(weights):
	"""Per-query entropy of attention distribution. weights: [B, heads, Q, K].
	Returns [B, heads, Q]."""
	# Clamp to avoid log(0)
	w = weights.clamp(min=1e-10)
	return -(w * w.log()).sum(dim=-1)


def print_section(title):
	print(f"\n{'=' * 60}")
	print(f"  {title}")
	print(f"{'=' * 60}")


def run(checkpoint_dir):
	checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")
	if not os.path.exists(checkpoint_path):
		print(f"No checkpoint at {checkpoint_path}")
		sys.exit(1)

	print(f"Loading checkpoint: {checkpoint_path}")
	net, cfg, iteration = load_checkpoint(checkpoint_path)
	print(f"Iteration: {iteration}")
	print(f"Architecture: {cfg.get('layer_sizes')}, attention={cfg.get('attention')}")

	if not net._use_attention:
		print("\nThis checkpoint has no attention layers. Nothing to analyze.")
		return

	attn_cfg = cfg.get("attention", {})
	num_heads = attn_cfg.get("heads", "?")
	num_layers = attn_cfg.get("layers", "?")
	d_model = attn_cfg.get("dim", "?")
	print(f"Attention: {num_layers} layers, {num_heads} heads, d_model={d_model}")

	# Collect states
	random.seed(42)
	np.random.seed(42)
	torch.manual_seed(42)
	states = collect_states(n_games=50)
	print(f"Collected {len(states)} mid-game states")
	if len(states) < 5:
		print("Too few states collected — increase n_games")
		return

	states_t = torch.stack(states)
	with torch.no_grad():
		all_layer_weights = extract_attention_weights(net, states_t)
	# all_layer_weights: list of [B, heads, 20, 20]

	max_entropy = np.log(N_ENTITIES)  # uniform attention entropy

	for layer_idx, weights in enumerate(all_layer_weights):
		# weights: [B, heads, 20, 20]
		B, heads, Q, K = weights.shape
		print_section(f"Layer {layer_idx} ({heads} heads, max_entropy={max_entropy:.3f})")

		# Per-head analysis
		ent = attention_entropy(weights)  # [B, heads, Q]

		for h in range(heads):
			w_h = weights[:, h, :, :]  # [B, 20, 20]
			ent_h = ent[:, h, :]  # [B, 20]
			mean_ent = ent_h.mean().item()
			# Per-entity mean entropy
			entity_ent = ent_h.mean(dim=0)  # [20]

			# Cross-block attention mass (averaged over batch)
			w_mean = w_h.mean(dim=0)  # [20, 20]
			hand_to_hand = w_mean[:N_HAND, :N_HAND].sum().item()
			hand_to_scout = w_mean[:N_HAND, N_HAND:].sum().item()
			scout_to_hand = w_mean[N_HAND:, :N_HAND].sum().item()
			scout_to_scout = w_mean[N_HAND:, N_HAND:].sum().item()
			# Normalize: each query row sums to 1, so total mass = N_ENTITIES
			# Hand queries contribute N_HAND, scout queries contribute N_SCOUT
			hand_to_hand_frac = hand_to_hand / N_HAND
			hand_to_scout_frac = hand_to_scout / N_HAND
			scout_to_hand_frac = scout_to_hand / N_SCOUT
			scout_to_scout_frac = scout_to_scout / N_SCOUT

			print(f"\n  Head {h}:")
			print(f"    Mean entropy: {mean_ent:.3f} ({mean_ent/max_entropy:.1%} of max)")
			print(f"    Cross-block mass (fraction of each query block's attention):")
			print(f"      hand->hand:   {hand_to_hand_frac:.3f}  hand->scout:  {hand_to_scout_frac:.3f}")
			print(f"      scout->hand:  {scout_to_hand_frac:.3f}  scout->scout: {scout_to_scout_frac:.3f}")

			# Top-5 strongest edges (by mean attention weight across batch)
			flat = w_mean.flatten()
			top_vals, top_idxs = flat.topk(5)
			edges = []
			for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
				q, k = idx // N_ENTITIES, idx % N_ENTITIES
				edges.append((ENTITY_NAMES[q], ENTITY_NAMES[k], val))
			print(f"    Top-5 edges:")
			for q_name, k_name, val in edges:
				print(f"      {q_name:>8} -> {k_name:<8}  {val:.4f}")

			# Entity with lowest/highest mean entropy
			min_ent_idx = entity_ent.argmin().item()
			max_ent_idx = entity_ent.argmax().item()
			print(f"    Most focused query: {ENTITY_NAMES[min_ent_idx]} (H={entity_ent[min_ent_idx]:.3f})")
			print(f"    Most diffuse query: {ENTITY_NAMES[max_ent_idx]} (H={entity_ent[max_ent_idx]:.3f})")

	# Head specialization: pairwise cosine similarity of flattened attention patterns
	print_section("Head Specialization")
	for layer_idx, weights in enumerate(all_layer_weights):
		B, heads, Q, K = weights.shape
		if heads < 2:
			print(f"\n  Layer {layer_idx}: single head, no specialization analysis")
			continue
		# Flatten each head's mean attention pattern
		w_mean = weights.mean(dim=0)  # [heads, 20, 20]
		patterns = w_mean.reshape(heads, -1)  # [heads, 400]
		# Pairwise cosine similarity
		norms = patterns.norm(dim=1, keepdim=True).clamp(min=1e-8)
		normalized = patterns / norms
		cosine_sim = normalized @ normalized.T  # [heads, heads]
		print(f"\n  Layer {layer_idx} — head-pair cosine similarity:")
		# Header
		header = "      " + "".join(f"  H{h}" for h in range(heads))
		print(header)
		for h1 in range(heads):
			row = f"  H{h1}  "
			for h2 in range(heads):
				if h2 <= h1:
					row += f" {cosine_sim[h1, h2]:.2f}"
				else:
					row += "     "
			print(row)
		# Overall specialization: mean off-diagonal similarity
		mask = ~torch.eye(heads, dtype=torch.bool)
		mean_off_diag = cosine_sim[mask].mean().item()
		print(f"  Mean off-diagonal similarity: {mean_off_diag:.3f}")
		if mean_off_diag < 0.5:
			print(f"  -> Heads are specialized (low similarity)")
		elif mean_off_diag < 0.8:
			print(f"  -> Moderate specialization")
		else:
			print(f"  -> Heads are similar (low specialization, may be redundant)")

	# Overall summary
	print_section("Summary")
	for layer_idx, weights in enumerate(all_layer_weights):
		ent = attention_entropy(weights)  # [B, heads, Q]
		mean_ent = ent.mean().item()
		# How much attention goes to scout entities from hand entities?
		w_mean = weights.mean(dim=(0, 1))  # [20, 20] averaged over batch and heads
		hand_to_scout_frac = w_mean[:N_HAND, N_HAND:].sum().item() / N_HAND
		scout_to_hand_frac = w_mean[N_HAND:, :N_HAND].sum().item() / N_SCOUT
		print(f"Layer {layer_idx}: mean_entropy={mean_ent:.3f} ({mean_ent/max_entropy:.0%} of max), "
			f"hand->scout={hand_to_scout_frac:.3f}, scout->hand={scout_to_hand_frac:.3f}")
	# Characterization
	overall_ent = torch.cat([attention_entropy(w) for w in all_layer_weights]).mean().item()
	ent_ratio = overall_ent / max_entropy
	if ent_ratio > 0.9:
		print(f"\nAttention is nearly uniform ({ent_ratio:.0%} of max entropy) — "
			"likely not yet doing useful work")
	elif ent_ratio > 0.7:
		print(f"\nAttention is moderately focused ({ent_ratio:.0%} of max entropy) — "
			"some structure emerging")
	else:
		print(f"\nAttention is focused ({ent_ratio:.0%} of max entropy) — "
			"structured patterns present")
	print()


def main():
	if len(sys.argv) < 2:
		checkpoint_dir = os.path.join(SCRIPT_DIR, "bots", "v7_8")
	else:
		arg = sys.argv[1]
		if os.path.isabs(arg):
			checkpoint_dir = arg
		else:
			checkpoint_dir = os.path.join(SCRIPT_DIR, arg)
	run(checkpoint_dir)


if __name__ == "__main__":
	main()
