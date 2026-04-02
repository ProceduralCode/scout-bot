"""Benchmark: how often should rollout_numba recompute active indices?

Compares checking every step vs every N steps. The tradeoff:
- Checking less often: saves torch.where cost but forward-passes done games.
- Checking every step: max compaction benefit but more torch.where calls.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch import Tensor
from numba import cuda
from training import play_games_q_v6, rollout_multi_action_v6
from network import FlatScoutNetwork, batched_masked_sample
from encoding import INPUT_SIZE_V6
from gpu_engine import from_snapshots, repeat_state, GpuGameState, compute_scores_tensor
from numba_engine import (
	compute_legal_plays_kernel, compute_action_masks_kernel,
	encode_states_kernel, apply_actions_kernel,
	_grid, TPB, H, FLAT_ACTION_SIZE, MAX_STEPS,
)

# ── Setup ────────────────────────────────────────────────────────────────────

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

# Build a chunk of games
print("Collecting samples...")
import numpy as np, random
from encoding import decode_flat_action
from training import _apply_action_to_game
from game import Phase

samples = play_games_q_v6(network, 100, 4, training_seats=4,
	temperature=0.0, epsilon=0.05)
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


# ── Rollout variant with configurable check interval ────────────────────────

def rollout_check_interval(state: GpuGameState, network, check_every: int,
                           max_steps: int = MAX_STEPS, temperature: float = 1.0) -> Tensor:
	"""Like rollout_numba but recomputes active indices every check_every steps."""
	B = state.done.shape[0]
	dev = state.done.device

	legal_buf = torch.zeros(B, H, H, dtype=torch.bool, device=dev)
	mask_buf = torch.zeros(B, FLAT_ACTION_SIZE, dtype=torch.bool, device=dev)
	encode_buf = torch.zeros(B, 309, dtype=torch.float32, device=dev)

	d_hands_show = cuda.as_cuda_array(state.hands_show)
	d_hands_hide = cuda.as_cuda_array(state.hands_hide)
	d_hand_len = cuda.as_cuda_array(state.hand_len)
	d_play_show = cuda.as_cuda_array(state.play_show)
	d_play_hide = cuda.as_cuda_array(state.play_hide)
	d_play_len = cuda.as_cuda_array(state.play_len)
	d_play_owner = cuda.as_cuda_array(state.play_owner)
	d_play_type = cuda.as_cuda_array(state.play_type)
	d_play_strength = cuda.as_cuda_array(state.play_strength)
	d_current_player = cuda.as_cuda_array(state.current_player)
	d_phase = cuda.as_cuda_array(state.phase)
	d_scouts_since_play = cuda.as_cuda_array(state.scouts_since_play)
	d_sns_available = cuda.as_cuda_array(state.sns_available)
	d_num_players = cuda.as_cuda_array(state.num_players)
	d_collected = cuda.as_cuda_array(state.collected)
	d_scout_tokens = cuda.as_cuda_array(state.scout_tokens)
	d_round_ender = cuda.as_cuda_array(state.round_ender)
	d_done = cuda.as_cuda_array(state.done)
	d_legal = cuda.as_cuda_array(legal_buf)
	d_mask = cuda.as_cuda_array(mask_buf)
	d_encode = cuda.as_cuda_array(encode_buf)

	grid = _grid(B)
	logits = torch.zeros(B, FLAT_ACTION_SIZE, device=dev)
	CHUNK = 1 << 10

	network.eval()
	with torch.no_grad():
		step = 0
		active_idx = None
		while True:
			# Recompute active indices every check_every steps
			if step % check_every == 0:
				active_idx = torch.where(~state.done)[0]
				if active_idx.shape[0] == 0:
					break
				active = ~state.done
				n_active = active_idx.shape[0]
			if step >= max_steps:
				break
			step += 1

			hand_offsets = torch.randint(0, H, (B,), device=dev, dtype=torch.long)
			d_offsets = cuda.as_cuda_array(hand_offsets)

			compute_legal_plays_kernel[grid, TPB](
				d_hands_show, d_hand_len, d_current_player,
				d_play_len, d_play_type, d_play_strength,
				d_done, d_legal, B,
			)
			compute_action_masks_kernel[grid, TPB](
				d_hands_show, d_hand_len, d_current_player,
				d_play_show, d_play_hide, d_play_len, d_play_type,
				d_phase, d_sns_available, d_num_players,
				d_legal, d_offsets, d_mask, B,
			)
			encode_states_kernel[grid, TPB](
				d_hands_show, d_hands_hide, d_hand_len, d_current_player,
				d_play_show, d_play_hide, d_play_len, d_play_owner,
				d_play_type, d_play_strength, d_phase,
				d_scouts_since_play, d_sns_available, d_num_players,
				d_collected, d_scout_tokens, d_offsets, d_encode, B,
			)

			# Forward pass — only on (possibly stale) active games
			if n_active <= CHUNK:
				h = network(encode_buf[active_idx])
				logits[active_idx] = network.policy_logits(h)
			else:
				for start in range(0, n_active, CHUNK):
					end = min(start + CHUNK, n_active)
					idx = active_idx[start:end]
					h_chunk = network(encode_buf[idx])
					logits[idx] = network.policy_logits(h_chunk)

			has_action = mask_buf.any(dim=1)
			no_action = active & ~has_action
			if no_action.any():
				adv_cp = ((state.current_player.long() + 1) %
					state.num_players.long()).to(torch.int8)
				state.current_player = torch.where(
					no_action, adv_cp, state.current_player)
				d_current_player = cuda.as_cuda_array(state.current_player)

			if temperature == 0.0:
				actions = logits.masked_fill(~mask_buf, float('-inf')).argmax(dim=1)
			elif temperature != 1.0:
				actions = batched_masked_sample(logits / temperature, mask_buf)
			else:
				actions = batched_masked_sample(logits, mask_buf)
			d_actions = cuda.as_cuda_array(actions)
			apply_active = active & has_action
			d_apply_active = cuda.as_cuda_array(apply_active)

			apply_actions_kernel[grid, TPB](
				d_hands_show, d_hands_hide, d_hand_len,
				d_play_show, d_play_hide, d_play_len, d_play_owner,
				d_play_type, d_play_strength, d_current_player,
				d_phase, d_scouts_since_play, d_sns_available,
				d_num_players, d_collected, d_scout_tokens,
				d_round_ender, d_done, d_actions, d_offsets,
				d_apply_active, B,
			)

	return compute_scores_tensor(state)


# ── Benchmark ────────────────────────────────────────────────────────────────

check_intervals = [1, 2, 5, 10]
TRIALS = 3

for interval in check_intervals:
	times = []
	for trial in range(TRIALS):
		gpu_state = from_snapshots(games, device='cuda')
		gpu_state = repeat_state(gpu_state, ROLLOUTS)
		torch.cuda.synchronize()
		t0 = time.time()
		scores = rollout_check_interval(gpu_state, network, check_every=interval, temperature=1.0)
		torch.cuda.synchronize()
		elapsed = time.time() - t0
		times.append(elapsed)
	B = len(games) * ROLLOUTS
	avg = sum(times) / len(times)
	best = min(times)
	print(f"check_every={interval:2d}:  avg={avg:.3f}s  best={best:.3f}s  (B={B})")
