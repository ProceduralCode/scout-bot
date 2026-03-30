"""Profile GPU rollout with torch.compile to see if fusion helps."""
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
	apply_actions, compute_scores, H, MAX_STEPS,
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


if __name__ == '__main__':
	B = 200
	snaps = make_snapshots(4, B)
	net = FlatScoutNetwork(input_size=INPUT_SIZE_V6, layer_sizes=[512, 256, 128], attention=None)
	net.cuda()
	net.eval()

	# Try compiling
	print("Compiling functions...")
	t_comp = time.time()
	compiled_encode = torch.compile(encode_states)
	compiled_legal = torch.compile(compute_legal_plays)
	# compute_action_masks uses data-dependent control flow (if can_sns.any()),
	# so it may not compile cleanly. Try with fullgraph=False.
	compiled_masks = torch.compile(compute_action_masks, fullgraph=False)
	# apply_actions has data-dependent control flow too
	compiled_apply = torch.compile(apply_actions, fullgraph=False)
	print(f"Compile setup: {time.time() - t_comp:.1f}s")

	state = from_snapshots(snaps, device='cuda')

	# Warmup (triggers actual compilation)
	print("Warmup (JIT compile on first call)...")
	t_warm = time.time()
	with torch.no_grad():
		ho = torch.randint(0, H, (B,), device='cuda', dtype=torch.long)
		legal = compiled_legal(state)
		masks = compiled_masks(state, legal, ho)
		encoded = compiled_encode(state, ho)
		h = net(encoded)
		logits = net.policy_logits(h)
		actions = batched_masked_sample(logits, masks)
		has_action = masks.any(dim=1)
		compiled_apply(state, actions, ho, ~state.done & has_action)
	torch.cuda.synchronize()
	print(f"Warmup: {time.time() - t_warm:.1f}s")

	# Reset state for fresh run
	state = from_snapshots(snaps, device='cuda')

	# Profile compiled version
	n_steps = 50
	times = {'legal': 0, 'masks': 0, 'encode': 0, 'network': 0, 'sample': 0, 'apply': 0}

	with torch.no_grad():
		for step in range(n_steps):
			active = ~state.done
			if not active.any():
				break
			ho = torch.randint(0, H, (B,), device='cuda', dtype=torch.long)

			torch.cuda.synchronize()
			t0 = time.perf_counter()
			legal = compiled_legal(state)
			torch.cuda.synchronize()
			times['legal'] += time.perf_counter() - t0

			t0 = time.perf_counter()
			masks = compiled_masks(state, legal, ho)
			torch.cuda.synchronize()
			times['masks'] += time.perf_counter() - t0

			t0 = time.perf_counter()
			encoded = compiled_encode(state, ho)
			torch.cuda.synchronize()
			times['encode'] += time.perf_counter() - t0

			t0 = time.perf_counter()
			h = net(encoded)
			logits = net.policy_logits(h)
			torch.cuda.synchronize()
			times['network'] += time.perf_counter() - t0

			t0 = time.perf_counter()
			has_action = masks.any(dim=1)
			actions = batched_masked_sample(logits, masks)
			torch.cuda.synchronize()
			times['sample'] += time.perf_counter() - t0

			t0 = time.perf_counter()
			compiled_apply(state, actions, ho, active & has_action)
			torch.cuda.synchronize()
			times['apply'] += time.perf_counter() - t0

	actual_steps = min(step + 1, n_steps)
	total = sum(times.values())
	print(f"\nB={B}, {actual_steps} steps, total={total:.3f}s ({total/actual_steps*1000:.1f}ms/step)\n")
	print(f"{'Component':<12} {'Total':>8} {'Per-step':>10} {'%':>6}")
	print("-" * 40)
	for name, t in sorted(times.items(), key=lambda x: -x[1]):
		print(f"{name:<12} {t:>7.3f}s {t/actual_steps*1000:>8.1f}ms {t/total*100:>5.1f}%")
