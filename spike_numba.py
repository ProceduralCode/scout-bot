"""Spike: verify Numba CUDA ↔ PyTorch tensor interchange, measure kernel launch overhead."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import torch
import numpy as np
from numba import cuda

# ── Basic kernel: increment every element by 1 ──────────────────────────────

@cuda.jit
def increment_kernel(arr, n):
	i = cuda.grid(1)
	if i < n:
		arr[i] += 1

# ── Test 1: PyTorch → Numba → PyTorch round-trip ────────────────────────────

print("=== Test 1: PyTorch <-> Numba round-trip ===")
t = torch.zeros(1024, dtype=torch.int32, device='cuda')
d_arr = cuda.as_cuda_array(t)
threads = 256
blocks = (1024 + threads - 1) // threads
increment_kernel[blocks, threads](d_arr, 1024)
cuda.synchronize()
assert t.sum().item() == 1024, f"Expected 1024, got {t.sum().item()}"
print("PASS: Numba kernel wrote to PyTorch tensor, result correct.")

# ── Test 2: Multiple dtypes ─────────────────────────────────────────────────

print("\n=== Test 2: dtype compatibility ===")
for dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.float32, torch.bool]:
	t = torch.zeros(64, dtype=dtype, device='cuda')
	try:
		d = cuda.as_cuda_array(t)
		print(f"  {str(dtype):20s} -> OK")
	except Exception as e:
		print(f"  {str(dtype):20s} -> FAIL: {e}")

# ── Test 3: 2D tensor ───────────────────────────────────────────────────────

@cuda.jit
def fill_2d_kernel(arr, rows, cols):
	b = cuda.grid(1)
	if b < rows:
		for j in range(cols):
			arr[b, j] = b * cols + j

print("\n=== Test 3: 2D tensor (B=5000, C=16) ===")
B, C = 5000, 16
t2d = torch.zeros(B, C, dtype=torch.int32, device='cuda')
d2d = cuda.as_cuda_array(t2d)
threads = 256
blocks = (B + threads - 1) // threads
fill_2d_kernel[blocks, threads](d2d, B, C)
cuda.synchronize()
expected = torch.arange(B, device='cuda').unsqueeze(1) * C + torch.arange(C, device='cuda').unsqueeze(0)
assert torch.equal(t2d, expected.int()), "2D mismatch"
print("PASS: 2D tensor read/write works.")

# ── Test 4: Kernel launch overhead ──────────────────────────────────────────

@cuda.jit
def noop_kernel():
	pass

print("\n=== Test 4: Kernel launch overhead ===")
# Warmup
for _ in range(100):
	noop_kernel[1, 1]()
cuda.synchronize()

# Measure single launch
N_LAUNCHES = 10000
cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_LAUNCHES):
	noop_kernel[1, 1]()
cuda.synchronize()
elapsed = time.perf_counter() - t0
per_launch_us = elapsed / N_LAUNCHES * 1e6
print(f"  No-op kernel: {per_launch_us:.1f} us/launch ({N_LAUNCHES} launches in {elapsed*1e3:.1f} ms)")

# Measure with actual work (B=5000 threads, small per-thread work)
B = 5000
t_work = torch.zeros(B, dtype=torch.int32, device='cuda')
d_work = cuda.as_cuda_array(t_work)
for _ in range(100):
	increment_kernel[(B + 255) // 256, 256](d_work, B)
cuda.synchronize()

cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_LAUNCHES):
	increment_kernel[(B + 255) // 256, 256](d_work, B)
cuda.synchronize()
elapsed = time.perf_counter() - t0
per_launch_us = elapsed / N_LAUNCHES * 1e6
print(f"  B=5000 increment: {per_launch_us:.1f} us/launch ({N_LAUNCHES} launches in {elapsed*1e3:.1f} ms)")

# Simulate one rollout step: 4 kernel launches
print(f"\n  Projected per-step overhead (4 launches): {per_launch_us * 4:.1f} us")
print(f"  Projected 100 steps: {per_launch_us * 4 * 100 / 1000:.1f} ms")

# ── Test 5: as_cuda_array on contiguous slices ──────────────────────────────

print("\n=== Test 5: Contiguous slice handling ===")
t3d = torch.zeros(10, 5, 16, dtype=torch.int8, device='cuda')
# Contiguous slice (first dim)
try:
	d = cuda.as_cuda_array(t3d[0])
	print("  t3d[0] (contiguous)     -> OK")
except Exception as e:
	print(f"  t3d[0] (contiguous)     -> FAIL: {e}")
# Non-contiguous slice
try:
	d = cuda.as_cuda_array(t3d[:, 0, :])
	print("  t3d[:, 0, :] (non-contig) -> OK")
except Exception as e:
	print(f"  t3d[:, 0, :] (non-contig) -> FAIL: {e}")
# Make contiguous first
try:
	d = cuda.as_cuda_array(t3d[:, 0, :].contiguous())
	print("  .contiguous() workaround  -> OK")
except Exception as e:
	print(f"  .contiguous() workaround  -> FAIL: {e}")

print("\n=== Spike complete ===")
