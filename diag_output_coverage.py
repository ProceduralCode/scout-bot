"""Diagnostic: per-output training signal before and after curation.

Plays 1000 games, shows coverage for a random 1/10 subset (baseline)
vs curate_samples with multiplier=10.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import random
import torch
from training import play_games_q_v6, curate_samples
from network import FlatScoutNetwork
from encoding import INPUT_SIZE_V6, FULL_PERM, HAND_SLOTS_V6

H = HAND_SLOTS_V6
FLAT = 384

network = FlatScoutNetwork(INPUT_SIZE_V6, [256, 128],
	encoding_version=6, attention={"dim": 32, "heads": 2, "layers": 1})
ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots", "q_v1", "latest.pt")
if os.path.exists(ckpt_path):
	ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
	network.load_state_dict(ckpt["model_state"])
network.cuda()
network.eval()

MULTIPLIER = 10

print(f"Playing {100 * MULTIPLIER} games...")
all_samples = play_games_q_v6(network, 100 * MULTIPLIER, 4, training_seats=4,
	temperature=0.0, epsilon=0.05)
print(f"  {len(all_samples)} total samples")

# Baseline: random subsample (simulates no curation)
target = len(all_samples) // MULTIPLIER
baseline = random.sample(all_samples, target)

# Curated subset
curated = curate_samples(all_samples, MULTIPLIER)

print(f"  Baseline (random): {len(baseline)} samples")
print(f"  Curated: {len(curated)} samples")

# ── Analysis helper ──────────────────────────────────────────────────────────

def simulate_rollouts(samples):
	"""Simulate rollout action selection, return per-output aug target counts."""
	K, EXTRA = 10, 2
	for sample in samples:
		legal = np.where(sample.action_mask)[0]
		outputs = sample.network_outputs[legal]
		k = min(K, len(legal))
		top_idx = legal[np.argsort(outputs)[-k:][::-1]]
		selected = set(top_idx.tolist())
		selected.add(sample.action_taken)
		remaining = [a for a in legal if a not in selected]
		n_extra = min(EXTRA, len(remaining))
		if n_extra > 0:
			selected.update(random.sample(remaining, n_extra))
		sample.rolled_actions = sorted(selected)
	aug_counts = np.zeros(FLAT, dtype=int)
	aug_legal = np.zeros(FLAT, dtype=int)
	for s in samples:
		legal = np.where(s.action_mask)[0]
		for a in s.rolled_actions:
			for k in range(H):
				inv_perm = FULL_PERM[(H - k) % H]
				aug_counts[inv_perm[a].item()] += 1
		for a in legal:
			for k in range(H):
				inv_perm = FULL_PERM[(H - k) % H]
				aug_legal[inv_perm[a].item()] += 1
	return aug_counts, aug_legal

def play_length(a):
	if a >= 256:
		return None
	return (a % H - a // H) % H + 1

def report(label, aug_counts, aug_legal):
	print(f"\n{'='*60}")
	print(f"  {label}")
	print(f"{'='*60}")
	# By play length
	print("\n  Play outputs by length:")
	for length in range(1, 17):
		indices = [a for a in range(256) if play_length(a) == length]
		counts = aug_counts[indices]
		legal = aug_legal[indices]
		total = counts.sum()
		legal_total = legal.sum()
		if total == 0 and legal_total == 0:
			continue
		nonzero = (counts > 0).sum()
		print(f"    play-{length:2d}: targets={total:6,}  nonzero={nonzero:3d}/{len(indices)}  "
			  f"mean={counts.mean():7.1f}  |  legal={legal_total:7,}")
	# Scout and S&S
	for name, lo, hi in [("scout", 256, 320), ("S&S", 320, 384)]:
		counts = aug_counts[lo:hi]
		legal = aug_legal[lo:hi]
		print(f"    {name:8s}: targets={counts.sum():6,}  nonzero={(counts > 0).sum():3d}/{hi-lo}  "
			  f"mean={counts.mean():7.1f}  |  legal={legal.sum():7,}")
	# Overall
	print(f"\n  Overall:")
	print(f"    Total targets: {aug_counts.sum():,}")
	print(f"    Zero-signal outputs: {(aug_counts == 0).sum()}/384")
	pcts = np.percentile(aug_counts, [0, 5, 25, 50, 75, 95, 100])
	print(f"    Percentiles (p0/p5/p25/p50/p75/p95/p100):")
	print(f"      {pcts[0]:.0f} / {pcts[1]:.0f} / {pcts[2]:.0f} / {pcts[3]:.0f} / "
		  f"{pcts[4]:.0f} / {pcts[5]:.0f} / {pcts[6]:.0f}")
	nonzero = aug_counts[aug_counts > 0]
	if len(nonzero) > 0:
		print(f"    Nonzero range: {nonzero.min()}–{nonzero.max()}  "
			  f"ratio={nonzero.max()/nonzero.min():.1f}x")

# ── Run ──────────────────────────────────────────────────────────────────────

b_counts, b_legal = simulate_rollouts(baseline)
report("BASELINE (random 1/10)", b_counts, b_legal)

c_counts, c_legal = simulate_rollouts(curated)
report(f"CURATED (multiplier={MULTIPLIER})", c_counts, c_legal)
