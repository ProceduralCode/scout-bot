"""Pipeline diagnostic: dissect one training iteration to examine signal flow.

Loads a checkpoint, generates games using the same pipeline as training,
computes GAE, then reports on credit assignment depth, signal dilution
by action type, value function accuracy by game stage, and reward distribution.

Usage:
	python -u pipeline_diagnostic.py [checkpoint_dir]
	python -u pipeline_diagnostic.py bots/v7_8
"""

import sys
import os
import math
import random
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from network import FlatScoutNetwork, masked_sample
from encoding import (
	INPUT_SIZE_V6, HAND_SLOTS_V6, FLAT_ACTION_SIZE,
	encode_state_v6, get_flat_action_mask, decode_flat_action,
	encode_hand_both_orientations_v6, get_legal_plays,
)
from training import (
	StepRecordV6, play_games_v6, play_games_with_rollouts_v6,
	compute_gae, _batched_masked_log_prob, _batched_masked_entropy,
)
from game import Game, Phase

# --- Config ---
NUM_GAMES = 100
NUM_PLAYERS = 4
TRAINING_SEATS = 4

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

def classify_action(action_idx):
	"""Classify flat action index into type string."""
	if action_idx < 256:
		return "play"
	elif action_idx < 320:
		return "scout"
	else:
		return "sns"

def compute_td_residuals(records, advantages, gamma):
	"""Compute per-step TD residuals: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t).
	Groups by (game_id, player) like compute_gae does."""
	groups = {}
	for i, rec in enumerate(records):
		key = (rec.game_id, rec.round_num, rec.player)
		groups.setdefault(key, []).append(i)
	td_residuals = [0.0] * len(records)
	for indices in groups.values():
		for t in range(len(indices)):
			idx = indices[t]
			if t < len(indices) - 1:
				next_value = records[indices[t + 1]].value
			else:
				next_value = 0.0
			td_residuals[idx] = records[idx].reward + gamma * next_value - records[idx].value
	return td_residuals

def analyze_episode_structure(records):
	"""Group records into episodes, compute per-step positions."""
	groups = {}
	for i, rec in enumerate(records):
		key = (rec.game_id, rec.player)
		groups.setdefault(key, []).append(i)
	# Assign step position within each episode
	step_positions = [0] * len(records)
	episode_lengths = []
	for indices in groups.values():
		episode_lengths.append(len(indices))
		for pos, idx in enumerate(indices):
			step_positions[idx] = pos
	return step_positions, episode_lengths, groups

def fmt(val, decimals=4):
	if isinstance(val, float):
		return f"{val:.{decimals}f}"
	return str(val)

def percentile(vals, p):
	if not vals:
		return 0.0
	s = sorted(vals)
	k = (len(s) - 1) * p / 100
	f = int(k)
	c = min(f + 1, len(s) - 1)
	d = k - f
	return s[f] * (1 - d) + s[c] * d

def print_section(title):
	print(f"\n{'=' * 60}")
	print(f"  {title}")
	print(f"{'=' * 60}")

def run_diagnostic(checkpoint_dir):
	checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")
	if not os.path.exists(checkpoint_path):
		print(f"No checkpoint at {checkpoint_path}")
		sys.exit(1)

	print(f"Loading checkpoint: {checkpoint_path}")
	net, cfg, iteration = load_checkpoint(checkpoint_path)
	if torch.cuda.is_available():
		net.cuda()
	print(f"Iteration: {iteration}")
	print(f"Architecture: {cfg.get('layer_sizes')}, attention={cfg.get('attention')}")

	gamma = cfg.get("gamma", 0.995)
	gae_lambda = cfg.get("gae_lambda", 0.98)
	temperature = cfg.get("sampling_temperature", 1.0)
	reward_dist = cfg.get("reward_distribution", "terminal")
	reward_mode = cfg.get("reward_mode", "game_score")
	rollout_frac = cfg.get("rollout_fraction", 0.0)
	print(f"gamma={gamma}, gae_lambda={gae_lambda}, temperature={temperature}")
	print(f"reward_distribution={reward_dist}, reward_mode={reward_mode}")
	print(f"rollout_fraction={rollout_frac}")

	# Generate games using the same pipeline as training
	n_rollout = int(NUM_GAMES * rollout_frac)
	n_gae = NUM_GAMES - n_rollout

	print(f"\nGenerating {n_gae} GAE games + {n_rollout} rollout games ({NUM_PLAYERS}p, {TRAINING_SEATS} training seats)...")

	gae_records = play_games_v6(
		net, n_gae, NUM_PLAYERS,
		training_seats=TRAINING_SEATS,
		reward_distribution=reward_dist,
		reward_mode=reward_mode,
		temperature=temperature)
	gae_advantages, gae_returns = compute_gae(
		gae_records, gamma=gamma, lam=gae_lambda)
	# Overwrite value with GAE return (value target), matching main.py
	for rec, ret in zip(gae_records, gae_returns):
		rec.predicted_value = rec.value  # preserve network prediction
		rec.value = ret

	ro_records = []
	ro_advantages = []
	if n_rollout > 0:
		ro_records, ro_advantages, _ = play_games_with_rollouts_v6(
			net, n_rollout, NUM_PLAYERS,
			rollouts_per_state=cfg.get("rollouts_per_state", 20),
			training_seats=TRAINING_SEATS,
			temperature=temperature)

	all_records = gae_records + ro_records
	all_advantages = gae_advantages + ro_advantages
	print(f"Total records: {len(all_records)} ({len(gae_records)} GAE + {len(ro_records)} rollout)")

	# Compute TD residuals (GAE records only — rollout records don't use GAE)
	td_residuals_gae = compute_td_residuals(gae_records, gae_advantages, gamma)

	# Assign step positions
	step_positions, episode_lengths, episode_groups = analyze_episode_structure(all_records)

	# =====================================================================
	#  Section 1: Episode Structure
	# =====================================================================
	print_section("Episode Structure")
	print(f"Episodes (player-games): {len(episode_lengths)}")
	print(f"Decisions per episode: mean={np.mean(episode_lengths):.1f}, "
		  f"median={np.median(episode_lengths):.0f}, "
		  f"min={min(episode_lengths)}, max={max(episode_lengths)}")
	# Histogram of episode lengths
	length_counts = {}
	for l in episode_lengths:
		length_counts[l] = length_counts.get(l, 0) + 1
	print("  Distribution:")
	for l in sorted(length_counts.keys()):
		bar = '#' * min(length_counts[l], 50)
		print(f"    {l:>2} decisions: {length_counts[l]:>4} episodes  {bar}")

	# Action type census
	type_counts = {"play": 0, "scout": 0, "sns": 0}
	for rec in all_records:
		type_counts[classify_action(rec.action)] += 1
	total = len(all_records)
	print(f"\n  Action type census:")
	for atype in ["play", "scout", "sns"]:
		c = type_counts[atype]
		print(f"    {atype:>5}: {c:>5} ({100*c/total:.1f}%)")

	# Action types by step position
	max_pos = max(step_positions) if step_positions else 0
	print(f"\n  Action types by step position (first 10):")
	for pos in range(min(10, max_pos + 1)):
		types_at_pos = {"play": 0, "scout": 0, "sns": 0}
		for i, sp in enumerate(step_positions):
			if sp == pos:
				types_at_pos[classify_action(all_records[i].action)] += 1
		total_at_pos = sum(types_at_pos.values())
		if total_at_pos == 0:
			continue
		parts = []
		for atype in ["play", "scout", "sns"]:
			c = types_at_pos[atype]
			if c > 0:
				parts.append(f"{atype}={100*c/total_at_pos:.0f}%")
		print(f"    step {pos:>2}: n={total_at_pos:>4}  {', '.join(parts)}")

	# =====================================================================
	#  Section 2: Credit Assignment — Advantage by Step Position
	# =====================================================================
	print_section("Credit Assignment: Advantage by Step Position")
	# Use GAE records only (rollout advantages have different semantics)
	gae_step_pos, _, _ = analyze_episode_structure(gae_records)

	pos_advs = {}  # position → list of advantages
	pos_td = {}    # position → list of TD residuals
	for i in range(len(gae_records)):
		pos = gae_step_pos[i]
		pos_advs.setdefault(pos, []).append(gae_advantages[i])
		pos_td.setdefault(pos, []).append(td_residuals_gae[i])

	max_gae_pos = max(pos_advs.keys()) if pos_advs else 0
	print(f"{'pos':>3}  {'n':>5}  {'mean_adv':>9}  {'|adv|':>9}  {'std_adv':>9}  {'|TD|':>9}  {'TD_std':>9}")
	print(f"{'---':>3}  {'---':>5}  {'-------':>9}  {'-----':>9}  {'-------':>9}  {'----':>9}  {'------':>9}")
	for pos in range(max_gae_pos + 1):
		advs = pos_advs.get(pos, [])
		tds = pos_td.get(pos, [])
		if not advs:
			continue
		n = len(advs)
		mean_a = np.mean(advs)
		abs_a = np.mean(np.abs(advs))
		std_a = np.std(advs)
		abs_td = np.mean(np.abs(tds))
		std_td = np.std(tds)
		print(f"{pos:>3}  {n:>5}  {mean_a:>9.4f}  {abs_a:>9.4f}  {std_a:>9.4f}  {abs_td:>9.4f}  {std_td:>9.4f}")

	# Correlation: does advantage magnitude decay with position?
	if max_gae_pos >= 2:
		positions_flat = []
		advs_flat = []
		for i in range(len(gae_records)):
			positions_flat.append(gae_step_pos[i])
			advs_flat.append(abs(gae_advantages[i]))
		corr = np.corrcoef(positions_flat, advs_flat)[0, 1]
		print(f"\n  Correlation(step_position, |advantage|): {corr:.4f}")
		print(f"  {'Negative = advantage decays with depth (credit assignment fading)' if corr < 0 else 'Non-negative = advantage does not decay with depth'}")

	# =====================================================================
	#  Section 3: Signal Dilution — Advantage by Action Type
	# =====================================================================
	print_section("Signal Dilution: Advantage by Action Type")
	type_advs = {"play": [], "scout": [], "sns": []}
	type_rewards = {"play": [], "scout": [], "sns": []}
	for i, rec in enumerate(all_records):
		atype = classify_action(rec.action)
		type_advs[atype].append(all_advantages[i])
		type_rewards[atype].append(rec.reward)

	print(f"{'type':>5}  {'n':>5}  {'mean_adv':>9}  {'|adv|':>9}  {'std_adv':>9}  {'p10':>9}  {'p90':>9}  {'mean_rew':>9}")
	print(f"{'----':>5}  {'---':>5}  {'-------':>9}  {'-----':>9}  {'-------':>9}  {'---':>9}  {'---':>9}  {'-------':>9}")
	for atype in ["play", "scout", "sns"]:
		advs = type_advs[atype]
		rews = type_rewards[atype]
		if not advs:
			continue
		n = len(advs)
		print(f"{atype:>5}  {n:>5}  {np.mean(advs):>9.4f}  {np.mean(np.abs(advs)):>9.4f}  "
			  f"{np.std(advs):>9.4f}  {percentile(advs, 10):>9.4f}  {percentile(advs, 90):>9.4f}  "
			  f"{np.mean(rews):>9.4f}")

	# Ratio of signal strength
	play_abs = np.mean(np.abs(type_advs["play"])) if type_advs["play"] else 0
	scout_abs = np.mean(np.abs(type_advs["scout"])) if type_advs["scout"] else 0
	if scout_abs > 1e-8:
		print(f"\n  |adv| ratio play/scout: {play_abs/scout_abs:.2f}x")
	if type_advs["scout"]:
		print(f"  Scout is {100*len(type_advs['scout'])/len(all_records):.1f}% of records "
			  f"with {100*scout_abs/(play_abs+1e-8):.1f}% of play's signal strength")

	# =====================================================================
	#  Section 4: Value Function by Game Stage
	# =====================================================================
	print_section("Value Function by Game Stage")
	# Use predicted_value (network output) not value (which got overwritten with GAE returns)
	pos_vpred = {}  # position → list of V(s) predictions
	pos_vreturn = {}  # position → list of GAE returns (value targets)
	for i in range(len(gae_records)):
		pos = gae_step_pos[i]
		pos_vpred.setdefault(pos, []).append(gae_records[i].predicted_value)
		pos_vreturn.setdefault(pos, []).append(gae_records[i].value)

	print(f"{'pos':>3}  {'n':>5}  {'V_pred':>9}  {'V_target':>9}  {'MAE':>9}  {'V_pred_std':>9}")
	print(f"{'---':>3}  {'---':>5}  {'------':>9}  {'--------':>9}  {'---':>9}  {'---------':>9}")
	for pos in range(max_gae_pos + 1):
		vpreds = pos_vpred.get(pos, [])
		vtargets = pos_vreturn.get(pos, [])
		if not vpreds:
			continue
		n = len(vpreds)
		mean_pred = np.mean(vpreds)
		mean_target = np.mean(vtargets)
		mae = np.mean(np.abs(np.array(vpreds) - np.array(vtargets)))
		std_pred = np.std(vpreds)
		print(f"{pos:>3}  {n:>5}  {mean_pred:>9.4f}  {mean_target:>9.4f}  {mae:>9.4f}  {std_pred:>9.4f}")

	# Overall value accuracy
	all_vpred = [r.predicted_value for r in gae_records]
	all_vtarget = [r.value for r in gae_records]
	if len(all_vpred) > 1:
		corr = np.corrcoef(all_vpred, all_vtarget)[0, 1]
		mae = np.mean(np.abs(np.array(all_vpred) - np.array(all_vtarget)))
		print(f"\n  GAE records — V_pred vs V_target (GAE return):")
		print(f"    corr={corr:.4f}, MAE={mae:.4f}")
		print(f"    V_pred at step 0: mean={np.mean(pos_vpred.get(0, [0])):.4f} (should be ~0 in self-play)")
		print(f"    V_target range: [{min(all_vtarget):.3f}, {max(all_vtarget):.3f}]")
		print(f"    V_pred range: [{min(all_vpred):.3f}, {max(all_vpred):.3f}]")

	# Rollout records: separate analysis
	if ro_records:
		ro_vpred = [r.predicted_value for r in ro_records]
		ro_vtarget = [r.value for r in ro_records]  # rollout ground truth
		if len(ro_vpred) > 1:
			corr_ro = np.corrcoef(ro_vpred, ro_vtarget)[0, 1]
			mae_ro = np.mean(np.abs(np.array(ro_vpred) - np.array(ro_vtarget)))
			print(f"\n  Rollout records — V_pred vs V_target (rollout ground truth):")
			print(f"    corr={corr_ro:.4f}, MAE={mae_ro:.4f}")
			print(f"    V_target range: [{min(ro_vtarget):.3f}, {max(ro_vtarget):.3f}]")
			print(f"    V_pred range: [{min(ro_vpred):.3f}, {max(ro_vpred):.3f}]")
			# How spread out are rollout targets vs GAE targets?
			print(f"\n  Target quality comparison:")
			print(f"    GAE target std:     {np.std(all_vtarget):.4f}")
			print(f"    Rollout target std: {np.std(ro_vtarget):.4f}")
			print(f"    V_pred std:         {np.std(all_vpred):.4f}")
			# What fraction of the target signal is the value function capturing?
			ev_gae = 1 - np.var(np.array(all_vpred) - np.array(all_vtarget)) / max(np.var(all_vtarget), 1e-8)
			ev_ro = 1 - np.var(np.array(ro_vpred) - np.array(ro_vtarget)) / max(np.var(ro_vtarget), 1e-8)
			print(f"    Explained variance (GAE):     {ev_gae:.4f}")
			print(f"    Explained variance (rollout): {ev_ro:.4f}")

	# =====================================================================
	#  Section 5: Reward Distribution
	# =====================================================================
	print_section("Reward Distribution")
	# Analyze where reward actually lands
	nonzero_rewards = [i for i, r in enumerate(all_records) if abs(r.reward) > 1e-8]
	zero_rewards = len(all_records) - len(nonzero_rewards)
	print(f"Records with nonzero reward: {len(nonzero_rewards)}/{len(all_records)} "
		  f"({100*len(nonzero_rewards)/len(all_records):.1f}%)")

	# Reward by step position (relative to episode end)
	# Compute distance-from-end for each record
	dist_from_end = [0] * len(all_records)
	for indices in episode_groups.values():
		ep_len = len(indices)
		for pos, idx in enumerate(indices):
			dist_from_end[idx] = ep_len - 1 - pos  # 0 = last step

	print(f"\n  Reward by distance from episode end:")
	dist_rewards = {}
	for i in range(len(all_records)):
		d = dist_from_end[i]
		dist_rewards.setdefault(d, []).append(all_records[i].reward)

	print(f"{'dist':>4}  {'n':>5}  {'mean_rew':>9}  {'|rew|':>9}  {'frac>0':>8}")
	print(f"{'----':>4}  {'---':>5}  {'-------':>9}  {'-----':>9}  {'------':>8}")
	for d in sorted(dist_rewards.keys())[:15]:
		rews = dist_rewards[d]
		n = len(rews)
		frac_nonzero = sum(1 for r in rews if abs(r) > 1e-8) / n
		print(f"{d:>4}  {n:>5}  {np.mean(rews):>9.4f}  {np.mean(np.abs(rews)):>9.4f}  {frac_nonzero:>8.1%}")

	# =====================================================================
	#  Section 6: Policy Entropy by Action Type
	# =====================================================================
	print_section("Policy Entropy by Action Type")
	# Compute entropy for each record by running a forward pass
	net.eval()
	dev = next(net.parameters()).device
	BATCH = 512
	all_entropy_play = []  # entropy over play actions [0:256]
	all_entropy_scout = []  # entropy over scout+sns actions [256:384]
	all_entropy_full = []  # full entropy
	with torch.no_grad():
		for start in range(0, len(all_records), BATCH):
			batch_recs = all_records[start:start+BATCH]
			states = torch.stack([r.state for r in batch_recs]).to(dev)
			masks = torch.from_numpy(np.stack([r.mask for r in batch_recs])).to(dev)
			hidden = net(states)
			logits = net.policy_logits(hidden)
			# Full entropy
			full_ent = _batched_masked_entropy(logits, masks)
			# Play-region entropy
			play_masks = masks.clone()
			play_masks[:, 256:] = False
			has_play = play_masks.any(dim=1)
			play_ent = torch.zeros(len(batch_recs), device=dev)
			if has_play.any():
				play_ent[has_play] = _batched_masked_entropy(
					logits[has_play], play_masks[has_play])
			# Scout-region entropy
			scout_masks = masks.clone()
			scout_masks[:, :256] = False
			has_scout = scout_masks.any(dim=1)
			scout_ent = torch.zeros(len(batch_recs), device=dev)
			if has_scout.any():
				scout_ent[has_scout] = _batched_masked_entropy(
					logits[has_scout], scout_masks[has_scout])
			for i, rec in enumerate(batch_recs):
				atype = classify_action(rec.action)
				all_entropy_full.append((atype, full_ent[i].item()))
				if has_play[i]:
					all_entropy_play.append(play_ent[i].item())
				if has_scout[i]:
					all_entropy_scout.append(scout_ent[i].item())

	# Entropy by chosen action type
	ent_by_type = {"play": [], "scout": [], "sns": []}
	for atype, ent in all_entropy_full:
		ent_by_type[atype].append(ent)

	print(f"{'type':>5}  {'n':>5}  {'mean_H':>9}  {'std_H':>9}")
	print(f"{'----':>5}  {'---':>5}  {'------':>9}  {'-----':>9}")
	for atype in ["play", "scout", "sns"]:
		ents = ent_by_type[atype]
		if ents:
			print(f"{atype:>5}  {len(ents):>5}  {np.mean(ents):>9.4f}  {np.std(ents):>9.4f}")

	print(f"\n  Play-region entropy (over play logits only): "
		  f"mean={np.mean(all_entropy_play):.4f}, std={np.std(all_entropy_play):.4f}")
	print(f"  Scout-region entropy (over scout+sns logits only): "
		  f"mean={np.mean(all_entropy_scout):.4f}, std={np.std(all_entropy_scout):.4f}")

	# =====================================================================
	#  Section 7: Per-Action-Type Gradient Magnitude (one PPO step)
	# =====================================================================
	print_section("Gradient Contribution by Action Type (one PPO step)")
	# Do a single PPO forward pass, compute per-sample loss, accumulate gradients
	# separately for play vs scout vs sns samples
	net.train()
	net.zero_grad()
	states_t = torch.stack([r.state for r in all_records]).to(dev)
	masks_t = torch.from_numpy(np.stack([r.mask for r in all_records])).to(dev)
	actions_t = torch.tensor([r.action for r in all_records], dtype=torch.long, device=dev)
	old_lps_t = torch.tensor([r.old_log_prob for r in all_records], dtype=torch.float32, device=dev)
	advs_t = torch.tensor(all_advantages, dtype=torch.float32, device=dev)

	# Forward pass
	hidden = net(states_t)
	logits = net.policy_logits(hidden)
	new_lps = _batched_masked_log_prob(logits, masks_t, actions_t)
	ratio = torch.exp(new_lps - old_lps_t)
	# Unclipped surrogate loss per sample (negative because we maximize)
	surr = ratio * advs_t
	# Classify each sample
	is_play = actions_t < 256
	is_scout = (actions_t >= 256) & (actions_t < 320)
	is_sns = actions_t >= 320

	grad_norms = {}
	for label, mask in [("play", is_play), ("scout", is_scout), ("sns", is_sns)]:
		if mask.sum() == 0:
			grad_norms[label] = 0.0
			continue
		net.zero_grad()
		loss = -surr[mask].mean()
		loss.backward(retain_graph=True)
		total_norm = 0.0
		for p in net.parameters():
			if p.grad is not None:
				total_norm += p.grad.norm().item() ** 2
		grad_norms[label] = total_norm ** 0.5

	net.zero_grad()
	# Combined loss for reference
	loss_all = -surr.mean()
	loss_all.backward()
	total_norm_all = sum(p.grad.norm().item() ** 2 for p in net.parameters() if p.grad is not None) ** 0.5

	print(f"{'type':>5}  {'n':>5}  {'grad_norm':>10}  {'fraction':>9}")
	print(f"{'----':>5}  {'---':>5}  {'---------':>10}  {'--------':>9}")
	for label in ["play", "scout", "sns"]:
		mask = {"play": is_play, "scout": is_scout, "sns": is_sns}[label]
		n = mask.sum().item()
		gn = grad_norms[label]
		frac = gn / total_norm_all if total_norm_all > 1e-8 else 0
		print(f"{label:>5}  {n:>5}  {gn:>10.4f}  {frac:>9.1%}")
	print(f"{'all':>5}  {len(all_records):>5}  {total_norm_all:>10.4f}  {'100.0%':>9}")

	net.eval()

	# =====================================================================
	#  Summary
	# =====================================================================
	print_section("Summary")
	scout_frac = type_counts["scout"] / total if total > 0 else 0
	scout_signal = np.mean(np.abs(type_advs["scout"])) if type_advs["scout"] else 0
	play_signal = np.mean(np.abs(type_advs["play"])) if type_advs["play"] else 0
	scout_grad = grad_norms.get("scout", 0)
	play_grad = grad_norms.get("play", 0)

	print(f"Scout decisions: {scout_frac:.1%} of batch")
	print(f"Scout |advantage|: {scout_signal:.4f} vs play |advantage|: {play_signal:.4f}"
		  f" (ratio: {scout_signal/play_signal:.2f}x)" if play_signal > 1e-8 else "")
	print(f"Scout grad norm: {scout_grad:.4f} vs play grad norm: {play_grad:.4f}"
		  f" (ratio: {scout_grad/play_grad:.2f}x)" if play_grad > 1e-8 else "")

	v0_pred = np.mean(pos_vpred.get(0, [0]))
	print(f"V(s) at step 0: {v0_pred:.4f} (ideal: ~0)")

	# Credit assignment summary
	if max_gae_pos >= 2:
		early_abs = np.mean(np.abs(pos_advs.get(0, [0]) + pos_advs.get(1, [0])))
		late_abs = np.mean(np.abs(
			pos_advs.get(max_gae_pos, [0]) + pos_advs.get(max(0, max_gae_pos - 1), [0])))
		print(f"|Advantage| early (pos 0-1): {early_abs:.4f}")
		print(f"|Advantage| late (pos {max(0,max_gae_pos-1)}-{max_gae_pos}): {late_abs:.4f}")
		print(f"Early/late ratio: {early_abs/late_abs:.2f}x" if late_abs > 1e-8 else "")

	print()

def main():
	if len(sys.argv) < 2:
		# Default to latest v7 dir
		checkpoint_dir = os.path.join(SCRIPT_DIR, "bots", "v7_8")
	else:
		arg = sys.argv[1]
		if os.path.isabs(arg):
			checkpoint_dir = arg
		else:
			checkpoint_dir = os.path.join(SCRIPT_DIR, arg)
	run_diagnostic(checkpoint_dir)

if __name__ == "__main__":
	main()
