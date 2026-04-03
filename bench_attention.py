"""Benchmark attention: dims × heads × implementations (manual bmm vs SDPA)."""

import sys, os, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
from network import FlatScoutNetwork
from encoding import INPUT_SIZE_V6

B = 1024
WARMUP = 20
TRIALS = 200
LAYER_SIZES = [256, 128]

configs = [
	# (dim, heads, layers)
	(20, 4, 2),    # current v8_3
	(32, 4, 2),    # 4 heads × 8 head_dim
	(64, 4, 2),    # 4 heads × 16 head_dim
	(64, 8, 2),    # 8 heads × 8 head_dim
	(128, 4, 2),   # 4 heads × 32 head_dim
	(128, 8, 2),   # 8 heads × 16 head_dim
	(256, 8, 2),   # 8 heads × 32 head_dim
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
	print(f"  {label:<35s} {ms:6.2f}ms  {throughput:>10,.0f} samples/sec")
	return ms

x = torch.randn(B, INPUT_SIZE_V6, device='cuda')

# FC baseline
net_fc = FlatScoutNetwork(INPUT_SIZE_V6, LAYER_SIZES, attention=None).cuda().eval()
print(f"B={B}, warmup={WARMUP}, trials={TRIALS}\n")
fc_ms = bench_forward(net_fc, x, "FC only (no attention)")
print()

results = []
for dim, heads, layers in configs:
	attn_cfg = {"dim": dim, "heads": heads, "layers": layers}
	label_base = f"d={dim} h={heads} L={layers}"
	trunk_input = 20 * dim + 49
	params_attn = sum(p.numel() for p in
		FlatScoutNetwork(INPUT_SIZE_V6, LAYER_SIZES, attention=attn_cfg).parameters())
	print(f"--- {label_base} (trunk_in={trunk_input}, params={params_attn:,}) ---")

	# Manual bmm (single-head, current implementation)
	net = FlatScoutNetwork(INPUT_SIZE_V6, LAYER_SIZES, attention=attn_cfg).cuda().eval()
	ms_bmm = bench_forward(net, x, "manual bmm (single-head)")

	# SDPA (multi-head, fused kernel)
	net_sdpa = FlatScoutNetwork(INPUT_SIZE_V6, LAYER_SIZES, attention=attn_cfg).cuda().eval()
	net_sdpa._use_sdpa = True
	ms_sdpa = bench_forward(net_sdpa, x, "SDPA (multi-head)")

	results.append((label_base, trunk_input, ms_bmm, ms_sdpa))
	print()

# Summary table
print("=" * 75)
print(f"{'Config':<25s} {'trunk_in':>8s} {'bmm':>8s} {'SDPA':>8s} {'best':>8s} {'vs FC':>8s}")
print("-" * 75)
print(f"{'FC only':<25s} {'309':>8s} {fc_ms:>7.2f}ms {'':>8s} {fc_ms:>7.2f}ms {'1.0x':>8s}")
for label, trunk_in, ms_bmm, ms_sdpa in results:
	best = min(ms_bmm, ms_sdpa)
	print(f"{label:<25s} {trunk_in:>8d} {ms_bmm:>7.2f}ms {ms_sdpa:>7.2f}ms {best:>7.2f}ms {best/fc_ms:>7.1f}x")
