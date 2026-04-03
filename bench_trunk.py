"""Benchmark FC trunk sizes with d=64 h=1 L=2 attention."""

import sys, os, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
from network import FlatScoutNetwork
from encoding import INPUT_SIZE_V6

B = 1024
WARMUP = 20
TRIALS = 200
ATTN = {"dim": 64, "heads": 1, "layers": 2}

trunk_configs = [
	[256, 128],                     # current
	[512, 256, 128],                # wider first layer
	[512, 512, 256, 128],           # wider + deeper
	[1024, 512, 256, 128],          # match attention output width
	[1024, 512, 512, 256, 128],     # large
	[512, 256, 256, 128, 128],      # more residual pairs
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
	print(f"  {label:<45s} {ms:6.2f}ms  {throughput:>10,.0f}/sec")
	return ms

x = torch.randn(B, INPUT_SIZE_V6, device='cuda')
print(f"B={B}, attention=d{ATTN['dim']} h{ATTN['heads']} L{ATTN['layers']}")
print(f"Trunk input: {20 * ATTN['dim'] + 49} features\n")

results = []
for layers in trunk_configs:
	net = FlatScoutNetwork(INPUT_SIZE_V6, layers, attention=ATTN).cuda().eval()
	params = sum(p.numel() for p in net.parameters())
	trunk_params = sum(p.numel() for p in net.shared.parameters())
	label = f"{str(layers):<30s} ({params:>7,} total, {trunk_params:>7,} trunk)"
	ms = bench_forward(net, x, label)
	results.append((layers, params, trunk_params, ms))

print(f"\n{'='*70}")
print(f"{'Layers':<30s} {'params':>8s} {'time':>8s} {'delta':>8s}")
print(f"{'-'*70}")
base_ms = results[0][3]
for layers, params, trunk_params, ms in results:
	delta = ms - base_ms
	print(f"{str(layers):<30s} {params:>7,}  {ms:>7.2f}ms {delta:>+7.2f}ms")
