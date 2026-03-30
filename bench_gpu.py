"""Benchmark CPU vs GPU inference at the batch sizes that occur during rollouts.

The key question: rollouts batch ~346 states per forward pass on average. Is GPU
faster than CPU at this batch size for this model?

Usage: python -u bench_gpu.py [checkpoint_path]"""

import sys
import os
import time
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from network import FlatScoutNetwork
from encoding import INPUT_SIZE_V6

BATCH_SIZES = [1, 16, 64, 128, 256, 346, 512, 1024]
WARMUP = 20
REPS = 100

def load_network(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    layer_sizes = cfg.get("layer_sizes", [512, 256, 128])
    attention = cfg.get("attention", {})
    net = FlatScoutNetwork(input_size=INPUT_SIZE_V6, layer_sizes=layer_sizes, attention=attention)
    net.load_state_dict(checkpoint["model_state"])
    net.eval()
    return net.to(device)

def bench(net, batch_size, device, warmup=WARMUP, reps=REPS):
    x = torch.randn(batch_size, INPUT_SIZE_V6, device=device)
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            hidden = net(x)
            net.policy_logits(hidden)
    if device.type == "cuda":
        torch.cuda.synchronize()
    # Timed
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            hidden = net(x)
            net.policy_logits(hidden)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / reps * 1000  # ms per call

def bench_with_transfer(net_gpu, batch_size, reps=REPS):
    """Simulate the real workload: states are built on CPU, transferred to GPU for inference."""
    warmup = WARMUP
    x_cpu = torch.randn(batch_size, INPUT_SIZE_V6)
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            x = x_cpu.to("cuda")
            hidden = net_gpu(x)
            net_gpu.policy_logits(hidden)
    torch.cuda.synchronize()
    # Timed
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            x = x_cpu.to("cuda")
            hidden = net_gpu(x)
            net_gpu.policy_logits(hidden)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / reps * 1000

def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "bots/v7_3/latest.pt")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cpu = torch.device("cpu")
    net_cpu = load_network(checkpoint_path, cpu)

    print(f"\n{'Batch':>8} | {'CPU (ms)':>10} | {'GPU (ms)':>10} | {'GPU+xfer (ms)':>14} | {'Speedup (GPU+xfer)':>18}")
    print("-" * 72)

    if torch.cuda.is_available():
        cuda = torch.device("cuda")
        net_gpu = load_network(checkpoint_path, cuda)
    else:
        net_gpu = None

    for bs in BATCH_SIZES:
        cpu_ms = bench(net_cpu, bs, cpu)
        if net_gpu is not None:
            gpu_ms = bench(net_gpu, bs, cuda)
            xfer_ms = bench_with_transfer(net_gpu, bs)
            speedup = cpu_ms / xfer_ms
            print(f"{bs:>8} | {cpu_ms:>10.3f} | {gpu_ms:>10.3f} | {xfer_ms:>14.3f} | {speedup:>17.2f}x")
        else:
            print(f"{bs:>8} | {cpu_ms:>10.3f} | {'N/A':>10} | {'N/A':>14} | {'N/A':>18}")

    print("\nNote: 'GPU+xfer' is the realistic cost — states are built on CPU and transferred per call.")
    print(f"      Average batch size during rollouts is ~346 (from profiling).")

if __name__ == "__main__":
    main()
