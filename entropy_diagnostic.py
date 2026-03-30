"""Diagnose scout entropy collapse in v6 flat action space.

Measures gradient contributions from each loss component on play vs scout logits
to determine whether the joint entropy bonus can structurally protect scout entropy.

Usage: python -u scout-bot/entropy_diagnostic.py v6_8/latest.pt
"""

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import random
import numpy as np
import torch
import torch.nn.functional as F

from network import FlatScoutNetwork, masked_sample
from encoding import (
	INPUT_SIZE_V6, HAND_SLOTS_V6,
	encode_state_v6, get_flat_action_mask, get_legal_plays,
	encode_hand_both_orientations_v6,
)
from game import Game, Phase

NUM_GAMES = 100
NUM_PLAYERS = 4
H = HAND_SLOTS_V6


def load_checkpoint(path):
	checkpoint = torch.load(path, weights_only=False)
	cfg = checkpoint.get("config", {})
	ls = cfg.get("layer_sizes", [512, 256, 128])
	net = FlatScoutNetwork(INPUT_SIZE_V6, ls, encoding_version=6,
		attention=cfg.get("attention"))
	net.load_state_dict(checkpoint["model_state"])
	iteration = checkpoint.get("iteration", "?")
	return net, iteration, cfg


def collect_states(net):
	"""Play games, collect (state_tensor, mask_tensor, action_taken, action_type) tuples."""
	entries = []
	for game_idx in range(NUM_GAMES):
		game = Game(NUM_PLAYERS)
		game.starting_player = random.randint(0, NUM_PLAYERS - 1)
		game.total_rounds = 1
		game.start_round()
		with torch.no_grad():
			for p in range(NUM_PLAYERS):
				ho = random.randint(0, H - 1)
				t_normal, t_flipped = encode_hand_both_orientations_v6(game, p, ho)
				h_normal = net(t_normal)
				h_flipped = net(t_flipped)
				v_normal = net.value(h_normal).item()
				v_flipped = net.value(h_flipped).item()
				game.submit_flip_decision(p, do_flip=v_flipped > v_normal)
			from encoding import decode_flat_action
			while game.phase in (Phase.TURN, Phase.SNS_PLAY):
				p = game.current_player
				hand = game.players[p].hand
				legal_plays = get_legal_plays(hand, game.current_play)
				forced_play = game.phase == Phase.SNS_PLAY
				state = encode_state_v6(game, p, 0, forced_play=forced_play)
				hidden = net(state)
				logits = net.policy_logits(hidden)
				mask_t = get_flat_action_mask(game, p, legal_plays, 0)
				if not mask_t.any():
					game._advance_turn()
					continue
				action_idx, _ = masked_sample(logits, mask_t)
				action = decode_flat_action(action_idx, 0)
				has_play = mask_t[:256].any()
				has_scout = mask_t[256:320].any()
				entries.append({
					"state": state,
					"mask": mask_t,
					"action": action_idx,
					"action_type": action["type"],
					"has_play": has_play.item(),
					"has_scout": has_scout.item(),
					"n_play_legal": mask_t[:256].sum().item(),
					"n_scout_legal": mask_t[256:320].sum().item(),
				})
				if action["type"] == "play":
					game.apply_play(action["start"], action["end"])
				elif action["type"] == "scout":
					game.apply_scout(action["left_end"], action["flip"], action["insert_pos"])
				elif action["type"] == "sns":
					game.apply_sns_scout(action["left_end"], action["flip"], action["insert_pos"])
	return entries


def analyze_probability_mass(net, entries):
	"""How probability mass is distributed between play and scout."""
	print(f"\n{'='*60}")
	print("ANALYSIS 1: Probability mass distribution")
	print(f"{'='*60}")
	# Only states where both play and scout are legal
	both = [e for e in entries if e["has_play"] and e["has_scout"]]
	print(f"  States with both play + scout legal: {len(both)}/{len(entries)}")
	states = torch.stack([e["state"] for e in both])
	masks = torch.stack([e["mask"] for e in both])
	with torch.no_grad():
		hidden = net(states)
		logits = net.policy_logits(hidden)
		masked_logits = logits.masked_fill(~masks, float('-inf'))
		probs = torch.softmax(masked_logits, dim=-1)
		play_mass = probs[:, :256].sum(dim=1)
		scout_mass = probs[:, 256:320].sum(dim=1)
		sns_mass = probs[:, 320:384].sum(dim=1)
	print(f"\n  Mean probability mass:")
	print(f"    Play:  {play_mass.mean():.4f} (std {play_mass.std():.4f})")
	print(f"    Scout: {scout_mass.mean():.4f} (std {scout_mass.std():.4f})")
	print(f"    SnS:   {sns_mass.mean():.4f} (std {sns_mass.std():.4f})")
	# Scout concentration: how many actions hold 90% of scout mass?
	scout_probs = probs[:, 256:320]
	scout_masks_t = masks[:, 256:320]
	concs = []
	for i in range(len(both)):
		legal = scout_probs[i][scout_masks_t[i]]
		if len(legal) == 0:
			continue
		sorted_p, _ = legal.sort(descending=True)
		cumsum = sorted_p.cumsum(0)
		n_for_90 = (cumsum < 0.9).sum().item() + 1
		concs.append(n_for_90)
	print(f"\n  Scout action concentration (actions for 90% of scout mass):")
	print(f"    Mean: {np.mean(concs):.1f}, Median: {np.median(concs):.0f}, "
		  f"Max legal: {np.mean([e['n_scout_legal'] for e in both]):.1f}")
	# Per-region entropy
	play_ent_list = []
	scout_ent_list = []
	for i in range(len(both)):
		# Play-conditional entropy
		play_legal = probs[i, :256][masks[i, :256]]
		if len(play_legal) > 0:
			play_cond = play_legal / play_legal.sum()
			play_ent_list.append(-(play_cond * torch.log(play_cond + 1e-8)).sum().item())
		# Scout-conditional entropy
		scout_legal = probs[i, 256:320][masks[i, 256:320]]
		if len(scout_legal) > 0:
			scout_cond = scout_legal / scout_legal.sum()
			scout_ent_list.append(-(scout_cond * torch.log(scout_cond + 1e-8)).sum().item())
	print(f"\n  Conditional entropy (within-region):")
	print(f"    Play:  {np.mean(play_ent_list):.4f} (std {np.std(play_ent_list):.4f})")
	print(f"    Scout: {np.mean(scout_ent_list):.4f} (std {np.std(scout_ent_list):.4f})")
	return both


def analyze_gradients(net, entries):
	"""Compare gradient magnitudes on play vs scout logits from each loss component."""
	print(f"\n{'='*60}")
	print("ANALYSIS 2: Gradient decomposition (play vs scout logits)")
	print(f"{'='*60}")
	both = [e for e in entries if e["has_play"] and e["has_scout"]]
	n_samples = min(300, len(both))
	indices = random.sample(range(len(both)), n_samples)
	states = torch.stack([both[i]["state"] for i in indices])
	masks = torch.stack([both[i]["mask"] for i in indices])
	actions = torch.tensor([both[i]["action"] for i in indices], dtype=torch.long)
	net.train()
	hidden = net(states)
	logits = net.policy_logits(hidden)
	logits.retain_grad()
	masked_logits = logits.masked_fill(~masks, float('-inf'))
	probs = torch.softmax(masked_logits, dim=-1)
	# --- Joint entropy (what the current loss uses) ---
	safe_log_probs = F.log_softmax(masked_logits, dim=-1).masked_fill(~masks, 0.0)
	joint_entropy = -(probs * safe_log_probs).sum(dim=-1).mean()
	# --- Per-region entropy ---
	play_masks_full = torch.zeros_like(masks)
	play_masks_full[:, :256] = masks[:, :256]
	scout_masks_full = torch.zeros_like(masks)
	scout_masks_full[:, 256:320] = masks[:, 256:320]
	play_masked = logits.masked_fill(~play_masks_full, float('-inf'))
	play_probs = torch.softmax(play_masked, dim=-1)
	play_log_probs = F.log_softmax(play_masked, dim=-1).masked_fill(~play_masks_full, 0.0)
	play_entropy = -(play_probs * play_log_probs).sum(dim=-1).mean()
	scout_masked = logits.masked_fill(~scout_masks_full, float('-inf'))
	scout_probs = torch.softmax(scout_masked, dim=-1)
	scout_log_probs = F.log_softmax(scout_masked, dim=-1).masked_fill(~scout_masks_full, 0.0)
	scout_entropy = -(scout_probs * scout_log_probs).sum(dim=-1).mean()
	# --- Policy loss ---
	log_probs_full = F.log_softmax(masked_logits, dim=-1)
	selected_log_probs = log_probs_full.gather(1, actions.unsqueeze(1)).squeeze(-1)
	policy_loss = -selected_log_probs.mean()
	# Measure gradients on the logits directly
	print(f"\n  Joint entropy: {joint_entropy.item():.4f}")
	print(f"  Play entropy:  {play_entropy.item():.4f}")
	print(f"  Scout entropy: {scout_entropy.item():.4f}")
	print(f"  Policy loss:   {policy_loss.item():.4f}")
	results = {}
	for name, loss_val, coeff in [
		("policy_loss", policy_loss, 1.0),
		("joint_entropy", -joint_entropy, 0.08),  # negative because we maximize
		("play_entropy", -play_entropy, 0.08),
		("scout_entropy", -scout_entropy, 0.08),
	]:
		if logits.grad is not None:
			logits.grad = None
		scaled = coeff * loss_val
		scaled.backward(retain_graph=True)
		if logits.grad is None:
			print(f"  WARNING: no grad for {name}")
			continue
		g = logits.grad.clone()
		# Gradient norms on play vs scout logits
		play_grad_norm = g[:, :256].norm().item()
		scout_grad_norm = g[:, 256:320].norm().item()
		sns_grad_norm = g[:, 320:384].norm().item()
		results[name] = {
			"play_grad": play_grad_norm,
			"scout_grad": scout_grad_norm,
			"sns_grad": sns_grad_norm,
		}
	print(f"\n  Gradient norms on logits (with coefficients applied):")
	print(f"  {'Component':<20} {'Play logits':>12} {'Scout logits':>13} {'Ratio P/S':>10}")
	print(f"  {'-'*55}")
	for name, r in results.items():
		ratio = r["play_grad"] / r["scout_grad"] if r["scout_grad"] > 1e-10 else float('inf')
		print(f"  {name:<20} {r['play_grad']:>12.6f} {r['scout_grad']:>13.6f} {ratio:>10.1f}")
	# Key comparison: can the entropy bonus gradient counteract the policy gradient on scout logits?
	if "policy_loss" in results and "joint_entropy" in results:
		pol_scout = results["policy_loss"]["scout_grad"]
		ent_scout = results["joint_entropy"]["scout_grad"]
		print(f"\n  === KEY METRIC ===")
		print(f"  Policy gradient on scout logits:         {pol_scout:.6f}")
		print(f"  Joint entropy gradient on scout logits:  {ent_scout:.6f}")
		if ent_scout > 1e-10:
			print(f"  Policy / Joint entropy ratio:            {pol_scout / ent_scout:.1f}x")
		if "scout_entropy" in results:
			sep_scout = results["scout_entropy"]["scout_grad"]
			print(f"  Scout-only entropy grad on scout logits: {sep_scout:.6f}")
			if sep_scout > 1e-10:
				print(f"  Scout-only / Joint entropy ratio:        {sep_scout / ent_scout:.1f}x")
	net.eval()
	return results


def analyze_gradient_by_action_type(net, entries):
	"""Decompose policy gradient on scout logits by which action type was taken.
	Tests whether play-sample gradients or scout-sample gradients dominate."""
	print(f"\n{'='*60}")
	print("ANALYSIS 4: Policy gradient on scout logits by action type")
	print(f"{'='*60}")
	both = [e for e in entries if e["has_play"] and e["has_scout"]]
	play_entries = [e for e in both if e["action_type"] == "play"]
	scout_entries = [e for e in both if e["action_type"] == "scout"]
	sns_entries = [e for e in both if e["action_type"] == "sns"]
	print(f"  Samples with both regions legal: {len(both)}")
	print(f"    play actions: {len(play_entries)}, scout actions: {len(scout_entries)}, sns: {len(sns_entries)}")
	results = {}
	net.train()
	for label, subset in [("play_samples", play_entries), ("scout_samples", scout_entries)]:
		if len(subset) < 10:
			print(f"  Skipping {label} (too few samples)")
			continue
		n = min(200, len(subset))
		chosen = random.sample(subset, n)
		states = torch.stack([e["state"] for e in chosen])
		masks = torch.stack([e["mask"] for e in chosen])
		actions = torch.tensor([e["action"] for e in chosen], dtype=torch.long)
		hidden = net(states)
		logits = net.policy_logits(hidden)
		logits.retain_grad()
		masked_logits = logits.masked_fill(~masks, float('-inf'))
		log_probs = F.log_softmax(masked_logits, dim=-1)
		selected = log_probs.gather(1, actions.unsqueeze(1)).squeeze(-1)
		policy_loss = -selected.mean()
		policy_loss.backward()
		g = logits.grad.clone()
		# Gradient on scout logits
		scout_grad = g[:, 256:320]
		scout_grad_norm = scout_grad.norm().item()
		# Per-sample gradient direction: average the per-sample scout gradients
		# to see if they point coherently or cancel out
		per_sample_norms = scout_grad.norm(dim=1)
		mean_grad = scout_grad.mean(dim=0)  # average across samples
		mean_grad_norm = mean_grad.norm().item()
		mean_per_sample_norm = per_sample_norms.mean().item()
		# Coherence: if gradients point the same way, mean_grad_norm ≈ mean_per_sample_norm
		# If they cancel, mean_grad_norm << mean_per_sample_norm
		coherence = mean_grad_norm / (mean_per_sample_norm + 1e-10)
		# Which scout logits get the largest mean gradient? (shows directionality)
		top_k = 5
		abs_mean = mean_grad.abs()
		top_vals, top_idx = abs_mean.topk(top_k)
		results[label] = {
			"n": n,
			"scout_grad_norm": scout_grad_norm,
			"mean_grad_norm": mean_grad_norm,
			"mean_per_sample_norm": mean_per_sample_norm,
			"coherence": coherence,
			"play_grad_norm": g[:, :256].norm().item(),
		}
		print(f"\n  {label} ({n} samples):")
		print(f"    Total grad norm on scout logits:  {scout_grad_norm:.6f}")
		print(f"    Mean per-sample grad norm:        {mean_per_sample_norm:.6f}")
		print(f"    Mean gradient norm (coherent):    {mean_grad_norm:.6f}")
		print(f"    Coherence (1.0=aligned, 0=cancel):{coherence:.4f}")
		print(f"    Top {top_k} scout logits by |mean grad|:")
		for i in range(top_k):
			idx = top_idx[i].item()
			val = mean_grad[idx].item()
			print(f"      logit {256+idx}: {val:+.6f}")
	if "play_samples" in results and "scout_samples" in results:
		p = results["play_samples"]
		s = results["scout_samples"]
		print(f"\n  === COMPARISON ===")
		print(f"  Play-sample grad on scout logits:  {p['scout_grad_norm']:.6f} (coherence {p['coherence']:.4f})")
		print(f"  Scout-sample grad on scout logits: {s['scout_grad_norm']:.6f} (coherence {s['coherence']:.4f})")
		ratio = s['scout_grad_norm'] / (p['scout_grad_norm'] + 1e-10)
		print(f"  Scout/Play gradient ratio:         {ratio:.1f}x")
		print(f"\n  Play-sample coherent grad on scout: {p['mean_grad_norm']:.6f}")
		print(f"  Scout-sample coherent grad on scout: {s['mean_grad_norm']:.6f}")
	net.eval()
	return results


def analyze_action_choices(entries):
	"""What actions does the network actually choose?"""
	print(f"\n{'='*60}")
	print("ANALYSIS 3: Action choice breakdown")
	print(f"{'='*60}")
	types = {}
	for e in entries:
		t = e["action_type"]
		types[t] = types.get(t, 0) + 1
	total = len(entries)
	for t, count in sorted(types.items()):
		print(f"  {t}: {count}/{total} ({count/total*100:.1f}%)")
	# For scout actions: which positions are chosen?
	scout_entries = [e for e in entries if e["action_type"] == "scout"]
	if scout_entries:
		action_counts = {}
		for e in scout_entries:
			a = e["action"]
			action_counts[a] = action_counts.get(a, 0) + 1
		sorted_actions = sorted(action_counts.items(), key=lambda x: -x[1])
		print(f"\n  Top 10 scout actions (of {len(action_counts)} unique):")
		for action_idx, count in sorted_actions[:10]:
			from encoding import decode_flat_action
			decoded = decode_flat_action(action_idx, 0)
			pct = count / len(scout_entries) * 100
			print(f"    action {action_idx}: {count}x ({pct:.1f}%) — "
				  f"left_end={decoded['left_end']}, flip={decoded['flip']}, "
				  f"insert={decoded['insert_pos']}")


def main():
	if len(sys.argv) < 2:
		print("Usage: python -u scout-bot/entropy_diagnostic.py <checkpoint>")
		sys.exit(1)
	path = os.path.join(SCRIPT_DIR, sys.argv[1])
	print(f"Loading {path}")
	net, iteration, cfg = load_checkpoint(path)
	print(f"Iteration: {iteration}")
	print(f"Config: layer_sizes={cfg.get('layer_sizes')}, "
		  f"entropy_bonus={cfg.get('entropy_bonus')}")
	print(f"\nCollecting states from {NUM_GAMES} self-play games...")
	entries = collect_states(net)
	print(f"Collected {len(entries)} decision states")
	analyze_probability_mass(net, entries)
	analyze_gradients(net, entries)
	analyze_action_choices(entries)
	analyze_gradient_by_action_type(net, entries)


if __name__ == "__main__":
	main()
