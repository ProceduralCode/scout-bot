"""Benchmark network forward pass throughput at different batch sizes.
Finds the optimal chunk size for rollout_numba.

Usage: python -u scout-bot/bench_batch_size.py [checkpoint_dir]
"""
import sys, os, time
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from network import FlatScoutNetwork
from encoding import INPUT_SIZE_V6

def bench(network, batch_sizes, warmup=3, trials=10):
	dev = next(network.parameters()).device
	network.eval()
	results = []
	with torch.no_grad():
		for B in batch_sizes:
			x = torch.randn(B, INPUT_SIZE_V6, device=dev)
			# Warmup
			for _ in range(warmup):
				h = network(x)
				logits = network.policy_logits(h)
			torch.cuda.synchronize()
			# Timed trials
			t0 = time.perf_counter()
			for _ in range(trials):
				h = network(x)
				logits = network.policy_logits(h)
			torch.cuda.synchronize()
			elapsed = time.perf_counter() - t0
			avg_ms = elapsed / trials * 1000
			throughput = B / (elapsed / trials)
			results.append((B, avg_ms, throughput))
			print(f"  B={B:>7,}  {avg_ms:>8.2f} ms  {throughput:>12,.0f} samples/sec")
			del x
			torch.cuda.empty_cache()
	return results

def main():
	ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "bots", "v7_10")
	ckpt_path = os.path.join(ckpt_dir, "latest.pt")
	if not os.path.exists(ckpt_path):
		ckpt_path = ckpt_dir  # maybe they passed the file directly

	print(f"Loading checkpoint: {ckpt_path}")
	ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
	cfg = ckpt.get("config", {})
	ls = cfg.get("layer_sizes", [256, 128])
	attn = cfg.get("attention")
	net = FlatScoutNetwork(INPUT_SIZE_V6, ls, encoding_version=6, attention=attn)
	net.load_state_dict(ckpt["model_state"])
	net.cuda()
	net.eval()
	print(f"Network: layers={ls}, attention={'yes' if attn else 'no'}")
	print(f"GPU: {torch.cuda.get_device_name()}")
	print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
	print()

	batch_sizes = [
		1,         # 1
		4,         # 4
		16,        # 16
		64,        # 64
		256,       # 256
		512,       # 512
		1 << 10,   # 1K
		1 << 11,   # 2K
		1 << 12,   # 4K
		1 << 13,   # 8K
		1 << 14,   # 16K
		1 << 15,   # 32K
		1 << 16,   # 64K
		1 << 17,   # 128K
		1 << 18,   # 256K
		1 << 19,   # 512K
	]

	# Filter out sizes that would OOM (rough estimate: ~2KB per sample for forward pass)
	vram_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
	max_safe = int(vram_mb * 1000 / 4)  # very rough: 4 bytes per float, ~1000 floats per sample
	batch_sizes = [b for b in batch_sizes if b <= max_safe]

	print("Forward pass benchmark (forward + policy_logits):")
	results = bench(net, batch_sizes)

	# Find peak throughput
	best = max(results, key=lambda r: r[2])
	print(f"\nPeak throughput: B={best[0]:,} -> {best[2]:,.0f} samples/sec")
	print(f"Recommended CHUNK size: {best[0]}")

	# Also test chunked vs unchunked for a realistic total batch
	total_B = 400_000
	if total_B <= max_safe:
		print(f"\n--- Chunked vs unchunked for B={total_B:,} ---")
		x = torch.randn(total_B, INPUT_SIZE_V6, device='cuda')
		warmup_trials = 2
		timed_trials = 5

		# Unchunked
		for _ in range(warmup_trials):
			h = net(x); net.policy_logits(h)
		torch.cuda.synchronize()
		t0 = time.perf_counter()
		for _ in range(timed_trials):
			h = net(x); net.policy_logits(h)
		torch.cuda.synchronize()
		unchunked_ms = (time.perf_counter() - t0) / timed_trials * 1000
		print(f"  Unchunked:  {unchunked_ms:.1f} ms")

		# Chunked at peak
		chunk = best[0]
		for _ in range(warmup_trials):
			for start in range(0, total_B, chunk):
				end = min(start + chunk, total_B)
				h = net(x[start:end]); net.policy_logits(h)
		torch.cuda.synchronize()
		t0 = time.perf_counter()
		for _ in range(timed_trials):
			for start in range(0, total_B, chunk):
				end = min(start + chunk, total_B)
				h = net(x[start:end]); net.policy_logits(h)
		torch.cuda.synchronize()
		chunked_ms = (time.perf_counter() - t0) / timed_trials * 1000
		print(f"  Chunked@{chunk//1024}K: {chunked_ms:.1f} ms")
		print(f"  Speedup: {unchunked_ms / chunked_ms:.2f}x")

		del x
		torch.cuda.empty_cache()

if __name__ == "__main__":
	main()
