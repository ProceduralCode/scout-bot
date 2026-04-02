"""Instrument rollout_numba step loop to find where time goes."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch import Tensor
from numba import cuda
from training import play_games_q_v6, rollout_multi_action_v6
from network import FlatScoutNetwork, batched_masked_sample
from encoding import INPUT_SIZE_V6
from gpu_engine import from_snapshots, repeat_state
from numba_engine import (
	compute_legal_plays_kernel, compute_action_masks_kernel,
	encode_states_kernel, apply_actions_kernel,
	compute_scores_tensor, _grid, TPB, H, FLAT_ACTION_SIZE, MAX_STEPS,
)

network = FlatScoutNetwork(INPUT_SIZE_V6, [256, 128],
	encoding_version=6, attention={"dim": 32, "heads": 2, "layers": 1})
ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots", "q_v1", "latest.pt")
if os.path.exists(ckpt_path):
	ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
	network.load_state_dict(ckpt["model_state"])
network.cuda()
network.eval()

# Warmup
print("Warmup...")
warmup = play_games_q_v6(network, 5, 4, training_seats=4, temperature=0.0, epsilon=0.05)
rollout_multi_action_v6(warmup, network, 4,
	rollout_actions_per_sample=3, rollout_actions_random_extra=1,
	rollouts_per_action=5, rollout_temperature=1.0, chunk_pairs=64)
torch.cuda.synchronize()

# Collect samples and prepare one chunk worth of games
print("Collecting samples...")
samples = play_games_q_v6(network, 100, 4, training_seats=4,
	temperature=0.0, epsilon=0.05)
# Select actions (step 1 of rollout_multi_action_v6)
import numpy as np, random
from encoding import decode_flat_action
from training import _apply_action_to_game
from game import Phase

for sample in samples:
	legal = np.where(sample.action_mask)[0]
	outputs = sample.network_outputs[legal]
	k = min(10, len(legal))
	top_idx = legal[np.argsort(outputs)[-k:][::-1]]
	selected = set(top_idx.tolist())
	selected.add(sample.action_taken)
	remaining = [a for a in legal if a not in selected]
	n_extra = min(2, len(remaining))
	if n_extra > 0:
		selected.update(random.sample(remaining, n_extra))
	sample.rolled_actions = sorted(selected)

# Build games for one chunk (chunk_pairs=512)
CHUNK_PAIRS = 512
ROLLOUTS = 30
pairs = []
for si, sample in enumerate(samples):
	for ai, action_idx in enumerate(sample.rolled_actions):
		pairs.append((si, ai, action_idx))

chunk = pairs[:CHUNK_PAIRS]
games = []
for si, ai, action_idx in chunk:
	sample = samples[si]
	g = sample.snapshot.clone()
	action = decode_flat_action(action_idx, sample.hand_offset)
	_apply_action_to_game(g, action)
	if g.phase not in (Phase.ROUND_OVER, Phase.GAME_OVER):
		games.append(g)

print(f"Chunk: {len(chunk)} pairs, {len(games)} need rollout")
print(f"After repeat: B = {len(games)} x {ROLLOUTS} = {len(games) * ROLLOUTS}")

# Convert to GPU and repeat
gpu_state = from_snapshots(games, device='cuda')
gpu_state = repeat_state(gpu_state, ROLLOUTS)
B = gpu_state.done.shape[0]
dev = gpu_state.done.device
print(f"B = {B}, grid = {_grid(B)}")

# Pre-allocate buffers (same as rollout_numba)
legal_buf = torch.zeros(B, H, H, dtype=torch.bool, device=dev)
mask_buf = torch.zeros(B, FLAT_ACTION_SIZE, dtype=torch.bool, device=dev)
encode_buf = torch.zeros(B, 309, dtype=torch.float32, device=dev)

d_hands_show = cuda.as_cuda_array(gpu_state.hands_show)
d_hands_hide = cuda.as_cuda_array(gpu_state.hands_hide)
d_hand_len = cuda.as_cuda_array(gpu_state.hand_len)
d_play_show = cuda.as_cuda_array(gpu_state.play_show)
d_play_hide = cuda.as_cuda_array(gpu_state.play_hide)
d_play_len = cuda.as_cuda_array(gpu_state.play_len)
d_play_owner = cuda.as_cuda_array(gpu_state.play_owner)
d_play_type = cuda.as_cuda_array(gpu_state.play_type)
d_play_strength = cuda.as_cuda_array(gpu_state.play_strength)
d_current_player = cuda.as_cuda_array(gpu_state.current_player)
d_phase = cuda.as_cuda_array(gpu_state.phase)
d_scouts_since_play = cuda.as_cuda_array(gpu_state.scouts_since_play)
d_sns_available = cuda.as_cuda_array(gpu_state.sns_available)
d_num_players = cuda.as_cuda_array(gpu_state.num_players)
d_collected = cuda.as_cuda_array(gpu_state.collected)
d_scout_tokens = cuda.as_cuda_array(gpu_state.scout_tokens)
d_round_ender = cuda.as_cuda_array(gpu_state.round_ender)
d_done = cuda.as_cuda_array(gpu_state.done)
d_legal = cuda.as_cuda_array(legal_buf)
d_mask = cuda.as_cuda_array(mask_buf)
d_encode = cuda.as_cuda_array(encode_buf)

grid = _grid(B)
FCHUNK = 1024
temperature = 1.0

# Instrumented rollout loop — use CUDA events for accurate GPU timing
print(f"\nRunning instrumented rollout (B={B})...\n")

# Accumulators
t_check = 0.0      # active.any() check
t_randint = 0.0     # hand_offsets generation
t_legal = 0.0       # compute_legal_plays_kernel
t_masks = 0.0       # compute_action_masks_kernel
t_encode = 0.0      # encode_states_kernel
t_forward = 0.0     # network forward + policy_logits
t_sampling = 0.0    # action sampling + no_action check
t_apply = 0.0       # apply_actions_kernel
total_steps = 0
active_counts = []

network.eval()
with torch.no_grad():
	torch.cuda.synchronize()
	t_total_start = time.perf_counter()

	for step in range(MAX_STEPS):
		# Check active
		torch.cuda.synchronize()
		t0 = time.perf_counter()
		active = ~gpu_state.done
		n_active = active.sum().item()
		t_check += time.perf_counter() - t0
		if n_active == 0:
			break
		active_counts.append(n_active)
		total_steps += 1

		# Randint
		torch.cuda.synchronize()
		t0 = time.perf_counter()
		hand_offsets = torch.randint(0, H, (B,), device=dev, dtype=torch.long)
		d_offsets = cuda.as_cuda_array(hand_offsets)
		torch.cuda.synchronize()
		t_randint += time.perf_counter() - t0

		# Legal plays kernel
		torch.cuda.synchronize()
		t0 = time.perf_counter()
		compute_legal_plays_kernel[grid, TPB](
			d_hands_show, d_hand_len, d_current_player,
			d_play_len, d_play_type, d_play_strength,
			d_done, d_legal, B,
		)
		torch.cuda.synchronize()
		t_legal += time.perf_counter() - t0

		# Action masks kernel
		torch.cuda.synchronize()
		t0 = time.perf_counter()
		compute_action_masks_kernel[grid, TPB](
			d_hands_show, d_hand_len, d_current_player,
			d_play_show, d_play_hide, d_play_len, d_play_type,
			d_phase, d_sns_available, d_num_players,
			d_legal, d_offsets, d_mask, B,
		)
		torch.cuda.synchronize()
		t_masks += time.perf_counter() - t0

		# Encode states kernel
		torch.cuda.synchronize()
		t0 = time.perf_counter()
		encode_states_kernel[grid, TPB](
			d_hands_show, d_hands_hide, d_hand_len, d_current_player,
			d_play_show, d_play_hide, d_play_len, d_play_owner,
			d_play_type, d_play_strength, d_phase,
			d_scouts_since_play, d_sns_available, d_num_players,
			d_collected, d_scout_tokens, d_offsets, d_encode, B,
		)
		torch.cuda.synchronize()
		t_encode += time.perf_counter() - t0

		# Forward pass
		torch.cuda.synchronize()
		t0 = time.perf_counter()
		if B <= FCHUNK:
			h = network(encode_buf)
			logits = network.policy_logits(h)
		else:
			logits = torch.empty(B, FLAT_ACTION_SIZE, device=dev)
			for start in range(0, B, FCHUNK):
				end = min(start + FCHUNK, B)
				h_chunk = network(encode_buf[start:end])
				logits[start:end] = network.policy_logits(h_chunk)
		torch.cuda.synchronize()
		t_forward += time.perf_counter() - t0

		# Sampling + no_action handling
		torch.cuda.synchronize()
		t0 = time.perf_counter()
		has_action = mask_buf.any(dim=1)
		no_action = active & ~has_action
		if no_action.any():
			adv_cp = ((gpu_state.current_player.long() + 1) %
				gpu_state.num_players.long()).to(torch.int8)
			gpu_state.current_player = torch.where(
				no_action, adv_cp, gpu_state.current_player)
			d_current_player = cuda.as_cuda_array(gpu_state.current_player)
		if temperature == 0.0:
			actions = logits.masked_fill(~mask_buf, float('-inf')).argmax(dim=1)
		elif temperature != 1.0:
			actions = batched_masked_sample(logits / temperature, mask_buf)
		else:
			actions = batched_masked_sample(logits, mask_buf)
		d_actions = cuda.as_cuda_array(actions)
		apply_active = active & has_action
		d_apply_active = cuda.as_cuda_array(apply_active)
		torch.cuda.synchronize()
		t_sampling += time.perf_counter() - t0

		# Apply actions kernel
		torch.cuda.synchronize()
		t0 = time.perf_counter()
		apply_actions_kernel[grid, TPB](
			d_hands_show, d_hands_hide, d_hand_len,
			d_play_show, d_play_hide, d_play_len, d_play_owner,
			d_play_type, d_play_strength, d_current_player,
			d_phase, d_scouts_since_play, d_sns_available,
			d_num_players, d_collected, d_scout_tokens,
			d_round_ender, d_done, d_actions, d_offsets,
			d_apply_active, B,
		)
		torch.cuda.synchronize()
		t_apply += time.perf_counter() - t0

	torch.cuda.synchronize()
	t_total = time.perf_counter() - t_total_start

print(f"Total steps: {total_steps}")
print(f"Active games: start={active_counts[0]}, mid={active_counts[len(active_counts)//2]}, "
	  f"end={active_counts[-1] if active_counts else 0}")
print(f"Total chunk time: {t_total:.3f}s")
print()
print(f"{'Phase':<25} {'Total':>8} {'Per-step':>10} {'%':>6}")
print("-" * 55)
phases = [
	("active check", t_check),
	("randint", t_randint),
	("legal_plays kernel", t_legal),
	("action_masks kernel", t_masks),
	("encode_states kernel", t_encode),
	("forward pass", t_forward),
	("sampling + no_action", t_sampling),
	("apply_actions kernel", t_apply),
]
accounted = sum(t for _, t in phases)
for name, t in phases:
	per_step = t / total_steps * 1000 if total_steps else 0
	pct = t / t_total * 100 if t_total else 0
	print(f"{name:<25} {t:>7.3f}s {per_step:>8.1f}ms {pct:>5.1f}%")
print("-" * 55)
print(f"{'accounted':<25} {accounted:>7.3f}s {'':>10} {accounted/t_total*100:>5.1f}%")
print(f"{'overhead (loop etc)':<25} {t_total - accounted:>7.3f}s {'':>10} {(t_total-accounted)/t_total*100:>5.1f}%")
print(f"{'TOTAL':<25} {t_total:>7.3f}s")

# Extrapolate to full iteration
n_chunks = 98  # ~48K pairs / 512
print(f"\nExtrapolated full iteration: {t_total * n_chunks:.0f}s (for {n_chunks} chunks)")
