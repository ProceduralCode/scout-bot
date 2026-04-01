# Numba Engine Plan

## Task & State

Planning phase complete for rewriting GPU rollout engine game logic as Numba CUDA kernels. Spec doc written, Numba installed and verified working. Implementation not yet started.

The PyTorch tensor-op GPU engine (`gpu_engine.py`, steps 1-6) is functionally complete and all tests pass. But it's slower than CPU Cython due to Python dispatch overhead — ~100 tensor ops per step, each with 10-50μs dispatch. torch.compile with triton-windows helped partially but hit a ceiling.

## What Changed

### New files
- `scout-bot/ai-specs/numba-engine.md` — full spec for the Numba engine (architecture, kernel specs, expected performance, implementation steps)
- `scout-bot/bench_gpu_scaling.py` — benchmarks GPU rollout at B=500 to B=50,000
- `scout-bot/bench_multiprocess.py` — benchmarks multiprocessing with persistent worker pools

### Installed
- `numba 0.64.0` (with `llvmlite 0.46.0`) — verified CUDA works on the RTX 3060

## Decisions

- **Numba CUDA kernels over other approaches.** Evaluated: torch.compile (1.3-3.5x, not enough), CUDA Graphs (requires full buffer pre-allocation rewrite), multiprocessing (2.7x, poor parallel efficiency), Cython nogil + ONNX (deceptively expensive — existing Cython uses Python objects), C/LibTorch (most from-scratch). Numba wins because gpu_engine.py is already a reference implementation to translate from, and zero-copy PyTorch interop via `as_cuda_array()` means no data interchange overhead.
- **One kernel per function, one thread per game.** 4 Numba kernel launches per step instead of ~200 PyTorch ops. Separate kernels for testability; can fuse later.
- **PyTorch stays for network inference + sampling.** Already efficient, no reason to rewrite.
- **New file `numba_engine.py`**, gpu_engine.py stays as reference/test oracle.

## Benchmarks from this session

GPU scaling (torch.compile):
| B | CPU (Cython) | GPU compiled | GPU throughput |
|---|---|---|---|
| 500 | 1.84s | 8.01s* | 62 g/s |
| 1,000 | 3.93s | 2.22s | 450 g/s |
| 2,000 | 7.26s | 3.10s | 646 g/s |
| 5,000 | ~3.6s | 6.10s | 820 g/s |
| 10,000 | ~37s | 11.12s | 899 g/s |
| 50,000 | ~183s | 52.06s | 960 g/s |
*B=500 includes torch.compile recompilation

GPU throughput plateaus at ~960 games/s regardless of batch size. Max speedup ~3.5x over CPU.

Multiprocessing (persistent pools, B=5,000): 8 workers = 2.71x (682 games/s). ~34% parallel efficiency.

## Next Steps

Start implementation per the spec (`scout-bot/ai-specs/numba-engine.md`):

1. **Spike** — write a trivial Numba CUDA kernel that reads/writes a PyTorch CUDA tensor, measure launch overhead
2. **compute_legal_plays_kernel** — translate from gpu_engine.py, test against it
3. **encode_states_kernel** — largest function, algorithmically simple
4. **compute_action_masks_kernel** — most complex (S&S hypothetical hands)
5. **apply_actions_kernel**
6. **rollout_numba** — wire loop, benchmark
7. **Integration into training.py**

## Watch Out

- Numba CUDA kernels can't use Python objects, dynamic allocation, or recursion. All game logic must be expressed as scalar ops on arrays with compile-time-known sizes (H=16, MAX_P=5 are constants — this is fine).
- `numba.cuda.as_cuda_array()` wraps PyTorch tensors zero-copy but requires contiguous tensors.
- The S&S mask computation involves 4 card choices × 17 insert positions × 136 start/end pairs = ~9,248 iterations per game in the inner loop. At B=5,000, each thread does this independently — GPU handles it fine.
