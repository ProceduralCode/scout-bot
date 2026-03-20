"""Diagnostic script: checks value function, advantages, and gradient flow.
Usage: python diagnostic.py <checkpoint_path>"""
import sys
import torch
import numpy as np
from encoding import INPUT_SIZE
from network import ScoutNetwork
from training import play_game, compute_gae, OpponentPool, ppo_update

def main():
	if len(sys.argv) < 2:
		print("Usage: python diagnostic.py <checkpoint_path>")
		return
	path = sys.argv[1]
	ckpt = torch.load(path, weights_only=False)
	cfg = ckpt.get("config", {})
	ls = cfg.get("layer_sizes", [cfg.get("first_hidden_size", 256), cfg.get("hidden_size", 128)])
	network = ScoutNetwork(input_size=INPUT_SIZE, layer_sizes=ls)
	network.load_state_dict(ckpt["model_state"])
	num_players = cfg.get("num_players", 4)
	training_seats = cfg.get("training_seats", 3)
	gamma = cfg.get("gamma", 0.99)
	gae_lambda = cfg.get("gae_lambda", 0.95)
	# Restore opponent pool if available
	pool_nets = []
	if "opponent_pool" in ckpt:
		pool = OpponentPool()
		pool.load_state_dicts(ckpt["opponent_pool"], network)
		pool_nets = pool.versions
	print(f"Loaded: layers={ls}, players={num_players}, training_seats={training_seats}")
	print(f"Opponent pool: {len(pool_nets)} versions")
	print()
	# Play games and collect records
	n_games = 200
	network.eval()
	all_records = []
	for g in range(n_games):
		opponents = [pool_nets[torch.randint(len(pool_nets), (1,)).item()]
					 for _ in range(num_players - training_seats)] if pool_nets else None
		records = play_game(network, num_players, opponent_pool=opponents,
							training_seats=training_seats)
		for r in records:
			r.game_id = g
		all_records.extend(records)
	print(f"Played {n_games} games, {len(all_records)} total records")
	print()
	# === 1. Value function analysis ===
	print("=" * 60)
	print("VALUE FUNCTION ANALYSIS")
	print("=" * 60)
	values = [r.value for r in all_records]
	print(f"  Mean:   {np.mean(values):.4f}")
	print(f"  Std:    {np.std(values):.4f}")
	print(f"  Min:    {np.min(values):.4f}")
	print(f"  Max:    {np.max(values):.4f}")
	print(f"  Range:  {np.max(values) - np.min(values):.4f}")
	if np.std(values) < 0.01:
		print("  ** VALUE FUNCTION IS NEAR-CONSTANT — this is the problem **")
	print()
	# === 2. GAE advantage analysis ===
	print("=" * 60)
	print("GAE ADVANTAGE ANALYSIS")
	print("=" * 60)
	advantages, returns, adv_std = compute_gae(all_records, gamma=gamma, lam=gae_lambda)
	print(f"  Raw advantage std: {adv_std:.6f}")
	print(f"  Returns mean: {np.mean(returns):.4f}")
	print(f"  Returns std:  {np.std(returns):.4f}")
	print()
	# Explained variance
	v_pred = np.array(values)
	v_target = np.array(returns)
	var_returns = np.var(v_target)
	if var_returns < 1e-8:
		ev = 0.0
	else:
		ev = 1 - np.var(v_target - v_pred) / var_returns
	print(f"  Explained variance: {ev:.4f}")
	if ev < 0.1:
		print("  ** VALUE FUNCTION EXPLAINS ALMOST NOTHING — advantages are noise **")
	print()
	# === 3. Advantages by action type ===
	print("=" * 60)
	print("ADVANTAGES BY ACTION TYPE")
	print("=" * 60)
	# Group by action type category
	type_names = {0: "play", 1: "scout", 2: "scout", 3: "scout", 4: "scout",
				  5: "sns", 6: "sns", 7: "sns", 8: "sns"}
	by_type = {"play": [], "scout": [], "sns": []}
	for i, r in enumerate(all_records):
		by_type[type_names[r.action_type]].append(advantages[i])
	for name in ("play", "scout", "sns"):
		vals = by_type[name]
		if vals:
			print(f"  {name:6s}: n={len(vals):5d}  mean={np.mean(vals):+.4f}  std={np.std(vals):.4f}")
		else:
			print(f"  {name:6s}: (none)")
	print()
	# Within-round advantage variance (do different actions in same round get different advantages?)
	print("=" * 60)
	print("WITHIN-ROUND ADVANTAGE VARIANCE")
	print("=" * 60)
	from collections import defaultdict
	groups = defaultdict(list)
	for i, r in enumerate(all_records):
		groups[(r.game_id, r.round_num, r.player)].append(advantages[i])
	within_stds = []
	for key, advs in groups.items():
		if len(advs) > 1:
			within_stds.append(np.std(advs))
	if within_stds:
		print(f"  Mean within-group std:   {np.mean(within_stds):.4f}")
		print(f"  Median within-group std: {np.median(within_stds):.4f}")
		print(f"  Groups with std < 0.01:  {sum(1 for s in within_stds if s < 0.01)} / {len(within_stds)}")
		if np.mean(within_stds) < 0.1:
			print("  ** LOW WITHIN-ROUND VARIANCE — all steps in a round get similar advantages **")
	print()
	# === 4. Action distribution ===
	print("=" * 60)
	print("ACTION DISTRIBUTION")
	print("=" * 60)
	n_total = len(all_records)
	n_play = sum(1 for r in all_records if r.action_type == 0)
	n_scout = sum(1 for r in all_records if 1 <= r.action_type <= 4)
	n_sns = sum(1 for r in all_records if 5 <= r.action_type <= 8)
	print(f"  Play:  {n_play:5d} ({100*n_play/n_total:.1f}%)")
	print(f"  Scout: {n_scout:5d} ({100*n_scout/n_total:.1f}%)")
	print(f"  S&S:   {n_sns:5d} ({100*n_sns/n_total:.1f}%)")
	print()
	# === 5. Per-head entropy (action probabilities) ===
	print("=" * 60)
	print("ACTION TYPE ENTROPY (is policy near-uniform or converged?)")
	print("=" * 60)
	at_entropies = []
	ps_entropies = []
	pe_entropies = []
	si_entropies = []
	for r in all_records:
		# Action type entropy
		masked = r.action_type_logits.clone()
		masked[~r.action_type_mask] = float('-inf')
		probs = torch.softmax(masked, dim=-1)
		n_legal = r.action_type_mask.sum().item()
		ent = -(probs * torch.log(probs + 1e-8)).sum().item()
		max_ent = np.log(n_legal) if n_legal > 1 else 1.0
		at_entropies.append(ent / max_ent)  # normalized entropy
		if r.play_start is not None:
			masked = r.play_start_logits.clone()
			masked[~r.play_start_mask] = float('-inf')
			probs = torch.softmax(masked, dim=-1)
			n_legal = r.play_start_mask.sum().item()
			ent = -(probs * torch.log(probs + 1e-8)).sum().item()
			max_ent = np.log(n_legal) if n_legal > 1 else 1.0
			ps_entropies.append(ent / max_ent)
		if r.play_end is not None:
			masked = r.play_end_logits.clone()
			masked[~r.play_end_mask] = float('-inf')
			probs = torch.softmax(masked, dim=-1)
			n_legal = r.play_end_mask.sum().item()
			ent = -(probs * torch.log(probs + 1e-8)).sum().item()
			max_ent = np.log(n_legal) if n_legal > 1 else 1.0
			pe_entropies.append(ent / max_ent)
		if r.scout_insert is not None:
			masked = r.scout_insert_logits.clone()
			masked[~r.scout_insert_mask] = float('-inf')
			probs = torch.softmax(masked, dim=-1)
			n_legal = r.scout_insert_mask.sum().item()
			ent = -(probs * torch.log(probs + 1e-8)).sum().item()
			max_ent = np.log(n_legal) if n_legal > 1 else 1.0
			si_entropies.append(ent / max_ent)
	print(f"  (normalized: 1.0 = uniform among legal moves, 0.0 = deterministic)")
	print(f"  Action type:  {np.mean(at_entropies):.3f}")
	if ps_entropies:
		print(f"  Play start:   {np.mean(ps_entropies):.3f}")
	if pe_entropies:
		print(f"  Play end:     {np.mean(pe_entropies):.3f}")
	if si_entropies:
		print(f"  Scout insert: {np.mean(si_entropies):.3f}")
	print()
	# === 6. Gradient flow ===
	print("=" * 60)
	print("GRADIENT FLOW (per-layer gradient norms)")
	print("=" * 60)
	network.train()
	# Use a subset for gradient check
	subset = all_records[:500]
	sub_adv = advantages[:500]
	sub_ret = returns[:500]
	_ = ppo_update(network, torch.optim.Adam(network.parameters(), lr=1e-10),
				   subset, sub_adv, returns=sub_ret)
	for name, param in network.named_parameters():
		if param.grad is not None:
			print(f"  {name:40s}  grad_norm={param.grad.norm().item():.6f}  param_norm={param.norm().item():.4f}")
		else:
			print(f"  {name:40s}  NO GRADIENT")
	print()

if __name__ == "__main__":
	main()
