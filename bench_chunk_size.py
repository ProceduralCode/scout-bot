"""Benchmark forward pass chunk sizes for the current network.
Tests throughput at different batch sizes. Hard 20s timeout."""

import sys, os, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
from network import FlatScoutNetwork
from encoding import INPUT_SIZE_V6
from main import Q_PARAMS

cfg = Q_PARAMS
net = FlatScoutNetwork(
    input_size=INPUT_SIZE_V6, layer_sizes=cfg["layer_sizes"],
    attention=cfg.get("attention"),
).cuda().eval()

print(f"Network: layers={cfg['layer_sizes']}, attention={cfg.get('attention')}")
print(f"GPU: {torch.cuda.get_device_name()}")

# Test individual batch sizes (throughput)
print("\n--- Throughput per batch size ---")
with torch.no_grad():
    for B in [256, 512, 1024, 2048, 4096, 7168, 8192]:
        x = torch.randn(B, INPUT_SIZE_V6, device='cuda')
        # Warmup
        for _ in range(3):
            h = net(x); net.policy_logits(h)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            h = net(x); net.policy_logits(h)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 20 * 1000
        tput = B / ms * 1000
        print(f"  B={B:>5d}  {ms:>6.2f}ms  {tput:>10,.0f} samples/sec")
        del x

# Test chunked processing of 7168 samples (matching production)
TOTAL = 7168
x_full = torch.randn(TOTAL, INPUT_SIZE_V6, device='cuda')
print(f"\n--- Processing {TOTAL} samples with different chunk sizes ---")
with torch.no_grad():
    for chunk in [512, 1024, 2048, 4096, 7168]:
        # Warmup
        for _ in range(2):
            for s in range(0, TOTAL, chunk):
                e = min(s + chunk, TOTAL)
                h = net(x_full[s:e]); net.policy_logits(h)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            for s in range(0, TOTAL, chunk):
                e = min(s + chunk, TOTAL)
                h = net(x_full[s:e]); net.policy_logits(h)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 10 * 1000
        n_chunks = (TOTAL + chunk - 1) // chunk
        print(f"  chunk={chunk:>5d}  {n_chunks} chunks  {ms:>6.2f}ms")
