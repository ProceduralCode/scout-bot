"""Benchmark GPU rollout with torch.compile via triton-windows."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import random
import time
import torch
from game import Game, Phase
from encoding import get_legal_plays, INPUT_SIZE_V6
from network import FlatScoutNetwork, batched_masked_sample
from gpu_engine import (
	from_snapshots, encode_states, compute_legal_plays, compute_action_masks,
	apply_actions, rollout_gpu, compute_scores, H, MAX_STEPS, GpuGameState,
)

def make_snapshots(num_players, count, seed_start=0):
	games = []
	for i in range(seed_start, seed_start + count * 20):
		random.seed(i)
		g = Game(num_players)
		g.start_round()
		for p in range(num_players):
			g.submit_flip_decision(p, do_flip=False)
		for _ in range(random.randint(1, 3)):
			if g.phase != Phase.TURN:
				break
			p = g.current_player
			legal = get_legal_plays(g.players[p].hand, g.current_play)
			if not legal:
				g._advance_turn()
				continue
			g.apply_play(*random.choice(legal))
		if g.phase in (Phase.TURN, Phase.SNS_PLAY):
			games.append(g)
		if len(games) >= count:
			break
	return games


def bench_cpu(snapshots, network, label=""):
	"""Benchmark CPU Cython path."""
	from training import rollout_from_states_batched_v6
	network.cpu()
	network.eval()
	t0 = time.time()
	with torch.no_grad():
		scores = rollout_from_states_batched_v6(snapshots, network)
	elapsed = time.time() - t0
	done = sum(1 for s in scores if any(v != 0 for v in s))
	print(f"  CPU   [{label}]: {elapsed:.2f}s, {done}/{len(snapshots)} finished")
	return elapsed


def bench_eager(snapshots, network, label=""):
	"""Benchmark without torch.compile."""
	network.cuda()
	network.eval()
	# Warmup
	with torch.no_grad():
		warm = from_snapshots(snapshots[:2], device='cuda')
		rollout_gpu(warm, network, max_steps=5)
	torch.cuda.synchronize()
	t0 = time.time()
	with torch.no_grad():
		state = from_snapshots(snapshots, device='cuda')
		scores = rollout_gpu(state, network)
	torch.cuda.synchronize()
	elapsed = time.time() - t0
	done = state.done.sum().item()
	print(f"  Eager [{label}]: {elapsed:.2f}s, {done}/{len(snapshots)} finished")
	return elapsed


def make_compiled_rollout():
	"""Create a compiled version of rollout_gpu's inner loop."""
	compiled_legal = torch.compile(compute_legal_plays)
	compiled_masks = torch.compile(compute_action_masks, fullgraph=False)
	compiled_encode = torch.compile(encode_states)
	compiled_apply = torch.compile(apply_actions, fullgraph=False)

	def rollout_compiled(state: GpuGameState, network, max_steps=MAX_STEPS):
		B = state.done.shape[0]
		dev = state.done.device
		network.eval()
		with torch.no_grad():
			for step in range(max_steps):
				active = ~state.done
				hand_offsets = torch.randint(0, H, (B,), device=dev, dtype=torch.long)
				legal = compiled_legal(state)
				masks = compiled_masks(state, legal, hand_offsets)
				encoded = compiled_encode(state, hand_offsets)
				h = network(encoded)
				logits = network.policy_logits(h)
				has_action = masks.any(dim=1)
				no_action = active & ~has_action
				adv_cp = ((state.current_player.long() + 1) %
					state.num_players.long()).to(torch.int8)
				state.current_player = torch.where(
					no_action, adv_cp, state.current_player)
				actions = batched_masked_sample(logits, masks)
				compiled_apply(state, actions, hand_offsets, active & has_action)
		return compute_scores(state)
	return rollout_compiled


def bench_compiled(snapshots, network, rollout_fn, label=""):
	"""Benchmark with torch.compile."""
	network.cuda()
	network.eval()
	# Warmup (triggers JIT compilation)
	print(f"  Compiling [{label}]...", end=" ", flush=True)
	t_comp = time.time()
	with torch.no_grad():
		warm = from_snapshots(snapshots[:5], device='cuda')
		rollout_fn(warm, network, max_steps=3)
	torch.cuda.synchronize()
	print(f"done ({time.time() - t_comp:.1f}s)")

	t0 = time.time()
	with torch.no_grad():
		state = from_snapshots(snapshots, device='cuda')
		scores = rollout_fn(state, network)
	torch.cuda.synchronize()
	elapsed = time.time() - t0
	done = state.done.sum().item()
	print(f"  Compiled [{label}]: {elapsed:.2f}s, {done}/{len(snapshots)} finished")
	return elapsed


if __name__ == '__main__':
	print(f"CUDA: {torch.cuda.get_device_name()}")
	print(f"PyTorch: {torch.__version__}\n")

	net = FlatScoutNetwork(input_size=INPUT_SIZE_V6, layer_sizes=[512, 256, 128], attention=None)
	print(f"Network params: {sum(p.numel() for p in net.parameters()):,}\n")

	compiled_rollout = make_compiled_rollout()

	# Warmup compile once at B=200 so JIT cost doesn't pollute benchmarks
	print("Warmup compilation at B=200...")
	warm_snaps = make_snapshots(4, 200, seed_start=999)
	net.cuda()
	with torch.no_grad():
		ws = from_snapshots(warm_snaps[:5], device='cuda')
		compiled_rollout(ws, net, max_steps=3)
	torch.cuda.synchronize()
	net.cpu()
	print("Done.\n")

	for B in [50, 200, 500]:
		print(f"--- B={B} ---")
		snaps = make_snapshots(4, B, seed_start=B * 100)
		t_cpu = bench_cpu([s.clone() for s in snaps], net, f"B={B}")
		t_eager = bench_eager([s.clone() for s in snaps], net, f"B={B}")
		t_compiled = bench_compiled([s.clone() for s in snaps], net, compiled_rollout, f"B={B}")
		print(f"  Compiled vs CPU: {t_cpu / t_compiled:.2f}x")
		print(f"  Compiled vs Eager: {t_eager / t_compiled:.2f}x")
		net.cpu()
		print()
