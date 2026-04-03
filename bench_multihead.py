"""Benchmark: single-head bmm vs multi-head bmm (isolate reshape overhead)."""

import sys, os, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
import torch.nn.functional as F
from network import FlatScoutNetwork
from encoding import INPUT_SIZE_V6

B = 1024
WARMUP = 20
TRIALS = 200
LAYER_SIZES = [256, 128]

configs = [
	# (dim, heads, layers)
	(64, 1, 2),    # single-head baseline
	(64, 4, 2),    # 4 heads × 16 head_dim
	(64, 8, 2),    # 8 heads × 8 head_dim
]

def bench_forward(net, x, label):
	with torch.no_grad():
		for _ in range(WARMUP):
			h = net(x); net.policy_logits(h)
		torch.cuda.synchronize()
		t0 = time.perf_counter()
		for _ in range(TRIALS):
			h = net(x); net.policy_logits(h)
		torch.cuda.synchronize()
		ms = (time.perf_counter() - t0) / TRIALS * 1000
	throughput = B / ms * 1000
	print(f"  {label:<40s} {ms:6.2f}ms  {throughput:>10,.0f} samples/sec")
	return ms

x = torch.randn(B, INPUT_SIZE_V6, device='cuda')

print(f"B={B}, warmup={WARMUP}, trials={TRIALS}")
print(f"All configs: d=64, L=2, layer_sizes={LAYER_SIZES}\n")

for dim, heads, layers in configs:
	attn_cfg = {"dim": dim, "heads": heads, "layers": layers}
	hd = dim // heads

	# Single-head bmm (no reshape, current default)
	net_sh = FlatScoutNetwork(INPUT_SIZE_V6, LAYER_SIZES, attention=attn_cfg).cuda().eval()
	ms_sh = bench_forward(net_sh, x, f"h={heads} hd={hd} single-head bmm")

	# Multi-head bmm (reshape into B*H groups, still manual bmm)
	net_mh = FlatScoutNetwork(INPUT_SIZE_V6, LAYER_SIZES, attention=attn_cfg).cuda().eval()
	net_mh._use_multihead_bmm = True
	ms_mh = bench_forward(net_mh, x, f"h={heads} hd={hd} multi-head bmm")

	# SDPA for comparison
	net_sdpa = FlatScoutNetwork(INPUT_SIZE_V6, LAYER_SIZES, attention=attn_cfg).cuda().eval()
	net_sdpa._use_sdpa = True
	ms_sdpa = bench_forward(net_sdpa, x, f"h={heads} hd={hd} SDPA")

	print(f"    multi-head bmm overhead: {ms_mh - ms_sh:+.2f}ms ({(ms_mh/ms_sh - 1)*100:+.1f}%)")
	print(f"    SDPA overhead:           {ms_sdpa - ms_sh:+.2f}ms ({(ms_sdpa/ms_sh - 1)*100:+.1f}%)")
	print()
