"""Test encode_states_kernel against gpu_engine.encode_states (PyTorch reference)."""
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import random
import torch
from numba import cuda as ncuda
from game import Game, Phase
from encoding import get_legal_plays, HAND_SLOTS_V6
from gpu_engine import from_snapshots, encode_states as ref_encode
from numba_engine import encode_states_kernel, H, TPB, _grid

DEVICE = 'cuda'

def make_games_with_play(num_players, count, seed_start=0):
	games = []
	for i in range(seed_start, seed_start + count * 10):
		random.seed(i)
		g = Game(num_players)
		g.start_round()
		for p in range(num_players):
			g.submit_flip_decision(p, do_flip=False)
		for _ in range(random.randint(1, 4)):
			if g.phase not in (Phase.TURN, Phase.SNS_PLAY):
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

def run_numba_encode(state, hand_offsets):
	B = state.done.shape[0]
	out = torch.zeros(B, 309, dtype=torch.float32, device=DEVICE)
	encode_states_kernel[_grid(B), TPB](
		ncuda.as_cuda_array(state.hands_show),
		ncuda.as_cuda_array(state.hands_hide),
		ncuda.as_cuda_array(state.hand_len),
		ncuda.as_cuda_array(state.current_player),
		ncuda.as_cuda_array(state.play_show),
		ncuda.as_cuda_array(state.play_hide),
		ncuda.as_cuda_array(state.play_len),
		ncuda.as_cuda_array(state.play_owner),
		ncuda.as_cuda_array(state.play_type),
		ncuda.as_cuda_array(state.play_strength),
		ncuda.as_cuda_array(state.phase),
		ncuda.as_cuda_array(state.scouts_since_play),
		ncuda.as_cuda_array(state.sns_available),
		ncuda.as_cuda_array(state.num_players),
		ncuda.as_cuda_array(state.collected),
		ncuda.as_cuda_array(state.scout_tokens),
		ncuda.as_cuda_array(hand_offsets),
		ncuda.as_cuda_array(out),
		B,
	)
	ncuda.synchronize()
	return out

def check(games, label):
	B = len(games)
	state = from_snapshots(games, device=DEVICE)
	random.seed(42)
	hand_offsets = torch.tensor(
		[random.randint(0, H - 1) for _ in range(B)],
		dtype=torch.long, device=DEVICE,
	)
	ref = ref_encode(state, hand_offsets).cpu()
	numba_out = run_numba_encode(state, hand_offsets).cpu()

	diff = (ref - numba_out).abs()
	max_diff = diff.max().item()
	if max_diff > 1e-5:
		# Find worst game/dim
		for b in range(B):
			game_diff = diff[b]
			gmax = game_diff.max().item()
			if gmax > 1e-5:
				dim = game_diff.argmax().item()
				print(f"FAIL [{label}]: max diff {max_diff:.2e}")
				print(f"  game {b} dim {dim}: ref={ref[b,dim]:.6f} numba={numba_out[b,dim]:.6f}")
				# Identify which segment
				if dim < 192:
					slot = dim // 12
					within = dim % 12
					print(f"  -> hand_top slot={slot} within={within}")
				elif dim < 208:
					print(f"  -> hand_bottom slot={dim - 192}")
				elif dim < 260:
					off = dim - 208
					if off < 44:
						c, v = off // 11, off % 11
						print(f"  -> scout one-hot c={c} v={v}")
					elif off < 48:
						print(f"  -> scout top scalar c={off - 44}")
					else:
						print(f"  -> scout bot scalar c={off - 48}")
				elif dim < 281:
					print(f"  -> play_buffer offset={dim - 260}")
				else:
					print(f"  -> metadata offset={dim - 281}")
				return False
	print(f"PASS [{label}]: {B} games, max diff {max_diff:.2e}")
	return True

def main():
	ok = True
	# Fresh games (no play)
	fresh = []
	for i in range(20):
		random.seed(i)
		g = Game(4)
		g.start_round()
		for p in range(4):
			g.submit_flip_decision(p, do_flip=False)
		fresh.append(g)
	ok &= check(fresh, "no play (fresh)")
	# With plays
	for n in [3, 4, 5]:
		ok &= check(make_games_with_play(n, 15, seed_start=n*100), f"{n}p with play")
	# Mixed
	mixed = []
	for n in [3, 4, 5]:
		mixed += make_games_with_play(n, 5, seed_start=n*200)
	ok &= check(mixed, "mixed")
	print()
	print("All passed." if ok else "SOME TESTS FAILED.")

if __name__ == '__main__':
	main()
