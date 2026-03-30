"""Empirical verification of compute_gae().
Plays a few games, manually computes GAE, and compares to compute_gae() output."""

import torch
import random
import numpy as np

from network import FlatScoutNetwork
from encoding import INPUT_SIZE_V6
from training import play_games_v6, compute_gae

GAMMA = 0.99
LAM = 0.95
NUM_GAMES = 3
NUM_PLAYERS = 4

def manual_gae(records, gamma, lam):
	"""Independent reimplementation of GAE for verification."""
	groups: dict[tuple[int, int, int], list[int]] = {}
	for i, rec in enumerate(records):
		groups.setdefault((rec.game_id, rec.round_num, rec.player), []).append(i)
	advantages = [0.0] * len(records)
	returns = [0.0] * len(records)
	for key, indices in groups.items():
		n = len(indices)
		# Walk backward through this player's decisions
		gae = 0.0
		for t in range(n - 1, -1, -1):
			idx = indices[t]
			r = records[idx].reward
			v = records[idx].value
			v_next = records[indices[t + 1]].value if t + 1 < n else 0.0
			delta = r + gamma * v_next - v
			gae = delta + gamma * lam * gae
			advantages[idx] = gae
			returns[idx] = gae + v
	# Normalize
	mean = sum(advantages) / len(advantages)
	std = (sum((a - mean) ** 2 for a in advantages) / len(advantages)) ** 0.5
	normalized = [(a - mean) / (std + 1e-8) for a in advantages]
	return normalized, returns, std

def main():
	random.seed(42)
	torch.manual_seed(42)
	np.random.seed(42)
	net = FlatScoutNetwork(INPUT_SIZE_V6, [512, 256, 128],
		encoding_version=6, attention={"dim": 32, "heads": 2, "layers": 1})
	print(f"Playing {NUM_GAMES} games with {NUM_PLAYERS} players...\n")
	records = play_games_v6(net, NUM_GAMES, NUM_PLAYERS, training_seats=NUM_PLAYERS,
		reward_distribution=0.7, reward_mode="game_score")
	print(f"Total records: {len(records)}\n")
	# Group and print records
	groups: dict[tuple[int, int, int], list[int]] = {}
	for i, rec in enumerate(records):
		groups.setdefault((rec.game_id, rec.round_num, rec.player), []).append(i)
	print("=== Records by group ===")
	for key in sorted(groups.keys()):
		indices = groups[key]
		print(f"\nGroup {key} (game, round, player) — {len(indices)} steps:")
		print(f"  {'idx':>4}  {'reward':>8}  {'value':>8}")
		for idx in indices:
			r = records[idx]
			print(f"  {idx:>4}  {r.reward:>8.4f}  {r.value:>8.4f}")
	# Run both implementations
	lib_adv, lib_ret = compute_gae(records, gamma=GAMMA, lam=LAM)
	man_adv, man_ret, man_std = manual_gae(records, gamma=GAMMA, lam=LAM)
	# Compare
	lib_std = np.std(lib_adv)
	print(f"\n=== Comparison ===")
	print(f"Raw advantage std — library: {lib_std:.6f}, manual: {man_std:.6f}")
	max_adv_diff = 0.0
	max_ret_diff = 0.0
	for i in range(len(records)):
		adv_diff = abs(lib_adv[i] - man_adv[i])
		ret_diff = abs(lib_ret[i] - man_ret[i])
		max_adv_diff = max(max_adv_diff, adv_diff)
		max_ret_diff = max(max_ret_diff, ret_diff)
	print(f"Max advantage diff: {max_adv_diff:.2e}")
	print(f"Max return diff:    {max_ret_diff:.2e}")
	if max_adv_diff < 1e-6 and max_ret_diff < 1e-6:
		print("\nVERIFIED: compute_gae matches manual implementation exactly.")
	else:
		print("\nMISMATCH DETECTED — printing details:")
		for i in range(len(records)):
			if abs(lib_adv[i] - man_adv[i]) > 1e-6 or abs(lib_ret[i] - man_ret[i]) > 1e-6:
				r = records[i]
				print(f"  idx={i} game={r.game_id} player={r.player} "
					  f"lib_adv={lib_adv[i]:.6f} man_adv={man_adv[i]:.6f} "
					  f"lib_ret={lib_ret[i]:.6f} man_ret={man_ret[i]:.6f}")
	# Spot-check: manually walk through one group in detail
	first_key = sorted(groups.keys())[0]
	indices = groups[first_key]
	print(f"\n=== Detailed walkthrough for group {first_key} ===")
	# Compute step by step, forward for readability
	n = len(indices)
	step_deltas = []
	step_gaes = []
	gae = 0.0
	for t in range(n - 1, -1, -1):
		idx = indices[t]
		r = records[idx].reward
		v = records[idx].value
		v_next = records[indices[t + 1]].value if t + 1 < n else 0.0
		delta = r + GAMMA * v_next - v
		gae = delta + GAMMA * LAM * gae
		step_deltas.append((t, idx, r, v, v_next, delta, gae))
	step_deltas.reverse()
	vnext_hdr = "V(s')"
	print(f"  {'t':>2}  {'idx':>4}  {'reward':>8}  {'V(s)':>8}  {vnext_hdr:>8}  {'delta':>10}  {'gae':>10}  {'return':>10}  {'lib_adv':>10}")
	for t, idx, r, v, v_next, delta, gae_val in step_deltas:
		ret = gae_val + v
		print(f"  {t:>2}  {idx:>4}  {r:>8.4f}  {v:>8.4f}  {v_next:>8.4f}  {delta:>10.6f}  {gae_val:>10.6f}  {ret:>10.6f}  {lib_adv[idx]:>10.6f}")

if __name__ == "__main__":
	main()
