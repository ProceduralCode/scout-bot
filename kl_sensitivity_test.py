"""KL sensitivity test: verify that low entropy causes KL blowup per optimizer step.

Loads a checkpoint, generates one mini-batch of training data, and measures
the KL divergence produced by a single optimizer step at various learning rates.

If the theory is correct: at the current (low) entropy, even one step at lr=3e-4
produces KL >> 0.015 (the kl_target), explaining why kl_batch_frac collapsed.

Usage:
	python -u kl_sensitivity_test.py [checkpoint_dir]
"""

import sys
import os
import copy
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from network import FlatScoutNetwork
from encoding import INPUT_SIZE_V6, FLAT_ACTION_SIZE
from training import (
	play_games_v6, play_games_with_rollouts_v6, compute_gae,
	augment_rotation_v6, prepare_ppo_batch_v6,
	_batched_masked_log_prob, _batched_masked_entropy,
)


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


def measure_kl_after_step(network, batch, lr, clip_epsilon=0.2,
						  entropy_bonus=0.03, value_loss_coeff=0.5,
						  max_grad_norm=0.5):
	"""Do one PPO gradient step at the given LR, measure KL, then restore weights."""
	saved_state = copy.deepcopy(network.state_dict())
	optimizer = torch.optim.Adam(network.parameters(), lr=lr)
	network.train()
	dev = next(network.parameters()).device
	states = batch["states"].to(dev)
	masks = batch["masks"].to(dev)
	actions = batch["actions"].to(dev)
	old_lps = batch["old_log_probs"].to(dev)
	adv = batch["adv"].to(dev)
	v_target = batch["v_target"].to(dev)
	# Forward
	hidden = network(states)
	v_pred = network.value(hidden).squeeze(-1)
	logits = network.policy_logits(hidden)
	new_lps = _batched_masked_log_prob(logits, masks, actions)
	entropy = _batched_masked_entropy(logits, masks)
	log_ratio = new_lps - old_lps
	ratio = torch.exp(log_ratio)
	surr1 = ratio * adv
	surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
	policy_loss = -torch.min(surr1, surr2).mean()
	value_loss = torch.nn.functional.mse_loss(v_pred, v_target)
	loss = policy_loss + value_loss_coeff * value_loss - entropy_bonus * entropy.mean()
	# Step
	optimizer.zero_grad()
	loss.backward()
	grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=max_grad_norm).item()
	optimizer.step()
	# Measure KL between old and new policy on the same batch
	network.eval()
	with torch.no_grad():
		new_hidden = network(states)
		new_logits = network.policy_logits(new_hidden)
		new_lps_after = _batched_masked_log_prob(new_logits, masks, actions)
		new_entropy = _batched_masked_entropy(new_logits, masks)
		# KL(old || new) ≈ E[ratio - 1 - log(ratio)]
		log_ratio_after = new_lps_after - old_lps
		ratio_after = torch.exp(log_ratio_after)
		approx_kl = ((ratio_after - 1) - log_ratio_after).mean().item()
	# Restore
	network.load_state_dict(saved_state)
	network.eval()
	return {
		"lr": lr,
		"approx_kl": approx_kl,
		"grad_norm": grad_norm,
		"entropy_before": entropy.mean().item(),
		"entropy_after": new_entropy.mean().item(),
		"policy_loss": policy_loss.item(),
	}


def run(checkpoint_dir):
	checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")
	if not os.path.exists(checkpoint_path):
		print(f"No checkpoint at {checkpoint_path}")
		sys.exit(1)
	print(f"Loading checkpoint: {checkpoint_path}")
	net, cfg, iteration = load_checkpoint(checkpoint_path)
	print(f"Iteration: {iteration}")
	kl_target = cfg.get("kl_target", 0.015)
	base_lr = cfg.get("learning_rate", 0.0003)
	temperature = cfg.get("sampling_temperature", 1.0)
	rollout_frac = cfg.get("rollout_fraction", 0.0)
	print(f"kl_target={kl_target}, base_lr={base_lr}, temperature={temperature}")
	if torch.cuda.is_available():
		net.cuda()

	# Generate one mini-batch of training data
	n_games = 50
	n_rollout = int(n_games * rollout_frac)
	n_gae = n_games - n_rollout
	print(f"\nGenerating {n_gae} GAE + {n_rollout} rollout games...")
	gae_records = play_games_v6(
		net, n_gae, cfg.get("num_players", 4),
		training_seats=cfg.get("training_seats", 4),
		reward_distribution=cfg.get("reward_distribution", "terminal"),
		reward_mode=cfg.get("reward_mode", "game_score"),
		temperature=temperature)
	gae_advantages, gae_returns = compute_gae(
		gae_records, gamma=cfg.get("gamma", 0.995),
		lam=cfg.get("gae_lambda", 0.98))
	for rec, ret in zip(gae_records, gae_returns):
		rec.value = ret
	ro_records, ro_advantages = [], []
	if n_rollout > 0:
		ro_records, ro_advantages, _ = play_games_with_rollouts_v6(
			net, n_rollout, cfg.get("num_players", 4),
			rollouts_per_state=cfg.get("rollouts_per_state", 20),
			training_seats=cfg.get("training_seats", 4),
			temperature=temperature)
	all_records = gae_records + ro_records
	all_advantages = gae_advantages + ro_advantages
	# Augment and batch
	gae_vloss_weight = cfg.get("gae_vloss_weight", 1.0)
	v_weights = [gae_vloss_weight] * len(gae_records) + [1.0] * len(ro_records)
	aug_steps, aug_advs, aug_vw = augment_rotation_v6(all_records, all_advantages, net, v_weights)
	batch = prepare_ppo_batch_v6(aug_steps, aug_advs, aug_vw)
	n_samples = batch["n"]
	print(f"Batch: {n_samples} samples (after 16x augmentation)")

	# Measure current entropy
	dev = next(net.parameters()).device
	with torch.no_grad():
		states = batch["states"].to(dev)
		masks = batch["masks"].to(dev)
		hidden = net(states)
		logits = net.policy_logits(hidden)
		ent = _batched_masked_entropy(logits, masks)
		# Play vs scout entropy
		play_mask = masks.clone()
		play_mask[:, 256:] = False
		has_play = play_mask.any(dim=1)
		scout_mask = masks.clone()
		scout_mask[:, :256] = False
		has_scout = scout_mask.any(dim=1)
		play_ent = _batched_masked_entropy(logits[has_play], play_mask[has_play]).mean().item() if has_play.any() else 0
		scout_ent = _batched_masked_entropy(logits[has_scout], scout_mask[has_scout]).mean().item() if has_scout.any() else 0
	print(f"\nCurrent policy entropy: {ent.mean().item():.4f}")
	print(f"  play entropy:  {play_ent:.4f}")
	print(f"  scout entropy: {scout_ent:.4f}")

	# Use a mini-batch-sized subset (same as training)
	mbs = cfg.get("mini_batch_size", 16384)
	if n_samples > mbs:
		idx = torch.randperm(n_samples)[:mbs]
		mini = {k: v[idx] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
		mini["n"] = mbs
	else:
		mini = batch
	print(f"Mini-batch size: {mini['n']}")

	# Test at multiple learning rates
	print(f"\n{'='*60}")
	print(f"  KL per single optimizer step at various learning rates")
	print(f"  kl_target = {kl_target}")
	print(f"{'='*60}")
	print(f"\n{'lr':>12}  {'approx_kl':>10}  {'vs_target':>10}  {'grad_norm':>10}  {'ent_after':>10}")
	print(f"{'--':>12}  {'--------':>10}  {'---------':>10}  {'---------':>10}  {'---------':>10}")
	test_lrs = [base_lr * mult for mult in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]]
	for lr in test_lrs:
		result = measure_kl_after_step(
			net, mini, lr,
			clip_epsilon=cfg.get("clip_epsilon", 0.2),
			entropy_bonus=cfg.get("entropy_bonus", 0.03),
			value_loss_coeff=cfg.get("value_loss_coeff", 0.5))
		ratio = result["approx_kl"] / kl_target
		marker = " <-- base LR" if abs(lr - base_lr) < 1e-8 else ""
		print(f"{lr:>12.6f}  {result['approx_kl']:>10.6f}  {ratio:>9.1f}x  {result['grad_norm']:>10.4f}  {result['entropy_after']:>10.4f}{marker}")

	print(f"\nIf base LR produces KL >> {kl_target}, the theory is confirmed:")
	print(f"low entropy -> sharp logits -> same LR produces large KL -> KL early stopping")
	print(f"kills most mini-batches -> noisy truncated updates -> regression.")
	print()


def main():
	if len(sys.argv) < 2:
		checkpoint_dir = os.path.join(SCRIPT_DIR, "bots", "v7_9")
	else:
		arg = sys.argv[1]
		if os.path.isabs(arg):
			checkpoint_dir = arg
		else:
			checkpoint_dir = os.path.join(SCRIPT_DIR, arg)
	run(checkpoint_dir)


if __name__ == "__main__":
	main()
