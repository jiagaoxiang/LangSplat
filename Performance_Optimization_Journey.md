## LangSplat on AMD GPUs: Performance Optimization Journey (15 -> 145 iter/s)

### Overview

Over the course of one week, we achieved a **~10x throughput improvement** for LangSplat language feature training on AMD MI325X GPUs through three categories of optimization. Each addressed a different bottleneck in the pipeline.

| Stage | Throughput | Bottleneck Removed |
|-------|------------|-------------------|
| Baseline (single GPU, `python train.py`) | ~15 iter/s | -- |
| After `OMP_NUM_THREADS=1` | ~25 iter/s | CPU thread contention |
| After GPU caching of language features | ~45 iter/s | Per-iteration disk I/O + CPU preprocessing |
| After gsplat integration | ~145 iter/s | Unoptimized CUDA rasterization kernels |

---

### 1. OMP_NUM_THREADS and CPU Thread Contention (15 -> 25 iter/s, ~1.7x)

**Commit:** `7864a69` (LangSplat)

**What happened:**
Running `python train.py` on a single GPU gave ~15 iter/s. Switching to multi-GPU DDP via `torchrun --nproc_per_node=8` bumped throughput to ~25 iter/s *per GPU*. This was surprising because DDP adds communication overhead (`all_reduce`), so per-GPU throughput should drop, not rise.

The clue: `torchrun` auto-sets `OMP_NUM_THREADS=1` when `nproc_per_node > 1` (see [`torch/distributed/run.py` line 850](https://github.com/pytorch/pytorch/blob/main/torch/distributed/run.py#L850)). Importantly, **single-GPU torchrun (`--nproc_per_node=1`) does NOT set this** -- the guard is `nproc_per_node > 1`. Plain `python` doesn't set it either.

**Root cause:**
Without `OMP_NUM_THREADS=1`, PyTorch's internal thread pool (controlled by `torch.get_num_threads()`) defaults to the number of CPU cores. On the MI325X node with a 128-core AMD EPYC, this means PyTorch dispatches every CPU operation (tensor indexing, `meshgrid`, L1 loss reduction, etc.) across up to 128 threads via `at::parallel_for`.

The thread pool itself is created once and reused -- threads aren't spawned per operation. However, each dispatch involves **partitioning the work, waking threads, and hitting a synchronization barrier** at the end. LangSplat's training loop has many short-lived CPU operations (mask indexing, loss computation) on small tensors interleaved with GPU kernel launches. For these microsecond-scale operations, the per-dispatch overhead of coordinating 128 threads far exceeds the actual computation, and the frequent synchronization barriers serialize the pipeline.

Setting `OMP_NUM_THREADS=1` makes PyTorch use a single-threaded code path, bypassing the thread pool dispatch entirely. For small-tensor ops this is faster because it eliminates the coordination overhead.

**Verification:** Setting `OMP_NUM_THREADS=1 python train.py` on a single GPU reproduces the same ~25 iter/s, confirming this is purely a CPU threading effect, not a DDP benefit.

**Code:**

```bash
# launch_baremetal.sh (commit eaca59e)
OMP_NUM_THREADS=1 torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    ...
```

**Takeaway:** Always set `OMP_NUM_THREADS=1` for GPU-bound training with short CPU ops, especially on high-core-count AMD EPYC nodes. Don't rely on `torchrun` to do it -- it only kicks in when `nproc_per_node > 1`.

---

### 2. GPU-Side Caching of Language Features (25 -> 45 iter/s, ~1.8x)

**Commit:** `7864a69` (LangSplat)
**File:** `scene/cameras.py`

**What happened:**
Every training iteration, `Camera.get_language_feature()` was:
1. Loading two `.npy` files from NFS disk (~10-50MB each)
2. Running CPU preprocessing: `meshgrid`, segment map indexing, reshaping (~70MB of heap allocations)
3. Copying the result to GPU via `tensor.cuda()`

This caused two problems:
- **I/O bottleneck:** NFS reads on every iteration, even for the same camera revisited across epochs.
- **glibc heap leak:** `np.load()` allocates on the heap. glibc's `malloc` uses `brk()` for allocations < 128KB, which **never returns memory to the OS** even after `free()`. With ~70MB of intermediates freed per iteration and 8 DDP workers at 30,000 iterations each, RSS grew monotonically until the Linux OOM killer terminated the job.

**The fix (two-part):**

**Part A -- GPU-side caching:** Cache `(point_feature.cuda(), mask.cuda())` on first access. Subsequent calls return the cached GPU tensors with zero I/O or CPU work. Memory cost: ~5GB per process for 299 cameras at 832x1264x3 -- trivial on MI325X with 256GB HBM.

```python
# scene/cameras.py (commit 7864a69)
def get_language_feature(self, language_feature_dir, feature_level):
    cache_key = feature_level
    if hasattr(self, '_lf_cache') and cache_key in self._lf_cache:
        return self._lf_cache[cache_key]

    # ... load from disk, preprocess on CPU ...

    # Cache on GPU -- subsequent calls are free
    result = (point_feature.cuda(), mask.cuda())
    if not hasattr(self, '_lf_cache'):
        self._lf_cache = {}
    self._lf_cache[cache_key] = result
    return result
```

**Part B -- MALLOC_MMAP_THRESHOLD:** Force glibc to use `mmap()` for allocations >= 64KB. Unlike `brk()`-based heap, `mmap()`'d regions are properly returned to the OS on `free()`, preventing the OOM.

```python
# train_ddp.py (commit 7864a69)
# CRITICAL: Must be set BEFORE importing numpy/torch.
import os
os.environ.setdefault('MALLOC_MMAP_THRESHOLD_', '65536')
```

**Why 1.8x:**
The 25 -> 45 iter/s jump shows that disk I/O + CPU preprocessing consumed roughly **45% of each iteration** before caching. After caching, the first epoch is slow (cold cache), but all subsequent epochs hit cache with zero cost. The remaining time is dominated by GPU rasterization (forward + backward), which becomes the new bottleneck.

---

### 3. gsplat Integration (45 -> 145 iter/s, ~3.2x)

**Commits:**
- gsplat repo: `413afe1` (language features support), `bc2e041` + `09a451d` (setup.py fixes)
- LangSplat repo: `b365699` (renderer rewrite)

**What happened:**
Replaced the `langsplat-rasterization` backend (forked from Inria's original 3DGS CUDA rasterizer, hipified for ROCm) with `ROCm/gsplat`, which has purpose-built AMD kernel optimizations. The `gaussian_renderer/__init__.py` was fully rewritten to call `gsplat.rasterization()` directly.

**Why 3.2x -- the AMD optimizations in gsplat:**

| Optimization | langsplat-rasterization | gsplat (ROCm) | Impact |
|---|---|---|---|
| **Tile size** | 16x16 hardcoded | 8x8 default | 8x8 = 64 threads = 1 wavefront on CDNA architecture. 16x16 = 256 threads = 4 wavefronts competing for the same CU's register file and LDS |
| **Backward warp reduction** | `atomicAdd` to global memory | DPP via `rocprim::warp_reduce` | DPP performs intra-wavefront reduction in registers using hardware crosslane ops, avoiding global memory atomics. This is the **single biggest speedup** for the backward pass |
| **Launch bounds** | Unspecified | `__launch_bounds__(64)` on backward kernel | Compiler knows exactly 1 wavefront/block, enabling maximum register allocation per thread |
| **Projection** | Separate kernels for cov3D, cov2D, projection | Fused `projection_ewa_3dgs_fused_fwd_kernel` | Single kernel launch for the entire world-to-camera + covariance + projection + conic pipeline |
| **Stream management** | CUDA-style via hipify shim | Native `c10::hip::HIPStream` | Eliminates the masquerading-as-CUDA overhead layer |

**The language features integration:**
Rather than threading a `language_feature` parameter through every kernel (as langsplat-rasterization does), we leveraged gsplat's existing N-D channel support (`CDIM` template). Language features (3ch) are concatenated with RGB (3ch) = 6 channels, padded to CDIM=8. No kernel changes needed -- all AMD optimizations apply automatically.

```python
# gsplat/rendering.py (commit 413afe1)
if language_features is not None:
    _language_features_dim = language_features.shape[-1]
    colors = torch.cat([colors, lf], dim=-1)  # [N, 3+3] = [N, 6]
```

**Why 3.2x makes sense:**
After I/O caching (step 2), iteration time was dominated by GPU rasterization: forward (~40%) + backward (~45%) + optimizer/misc (~15%). gsplat's DPP warp reductions accelerate the backward pass by ~4-5x, the 8x8 tile size improves forward occupancy by ~2x, and the fused projection eliminates kernel launch overhead. Combined, the rasterization portion (~85% of iteration) speeds up ~3-4x, yielding the overall 3.2x.

---

### Summary

**15 iter/s -> 25 -> 45 -> 145 iter/s (9.7x cumulative)**

| | OMP_NUM_THREADS=1 | GPU Caching | gsplat Integration |
|---|---|---|---|
| **Bottleneck** | CPU thread overhead | Disk I/O + CPU preprocessing | Unoptimized rasterization kernels |
| **Fix** | Single OMP thread | Cache on GPU HBM | AMD DPP + fused kernels + 8x8 tiles |
| **Speedup** | 1.7x | 1.8x | 3.2x |
| **Cumulative** | 1.7x | 3.0x | 9.7x |
