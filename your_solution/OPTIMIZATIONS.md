# INT4 Quantize + GEMM Kernel Optimizations

Starting point: `kernel_original.cu` — a naive baseline with one thread per output element and two global memory passes for quantization.

Target hardware: **NVIDIA RTX A6000** (SM86, 84 SMs, 768 GB/s GDDR6, 6 MB L2)

---

## Quantization Kernel (`quantize_int4_kernel`)

### 1. Single-pass register caching with `half2` vectorized loads
**Original:** Two separate loops over the group — one to find `max_abs`, one to quantize. Each element read from global memory twice.

**Optimized:** Load the entire group into a `float vals[128]` register array in a single pass, finding `max_abs` simultaneously. The second loop reads from registers (free) instead of global memory.

Loads are vectorized using `half2` — reads two FP16 values per instruction instead of one, halving load count and fusing the `fmaxf` for both values in one iteration.

```cpp
// Before: two global memory passes
for (int i = 0; i < group_size; i++) {
    float val = __half2float(input[rowK + k_start + i]);  // load 1
    max_abs = fmaxf(max_abs, fabsf(val));
}
for (int i = 0; i < group_size; i += 2) {
    float val_even = __half2float(input[rowK + k_start + i]);  // load 2 (redundant)
    ...
}

// After: single pass, half2 loads
for (int i = 0; i < group_size; i += 2) {
    half2 h2 = *reinterpret_cast<const half2*>(&input[rowK + k_start + i]);
    float v0 = __half2float(__low2half(h2));
    float v1 = __half2float(__high2half(h2));
    vals[i] = v0; vals[i+1] = v1;
    max_abs = fmaxf(max_abs, fmaxf(fabsf(v0), fabsf(v1)));
}
```

### 2. Input validation guards
Added `TORCH_CHECK(group_size <= 128)` to prevent silent buffer overflow from the fixed-size `vals[128]` register array.

---

## GEMM Kernel (`gemm_int4_kernel` → `gemm_int4_mma_kernel`)

The original naive kernel assigned one thread per output element with no data reuse. The optimized kernel is a complete replacement using NVIDIA tensor cores.

### 3. INT4 Tensor Core MMA (m16n8k64)
Replaced scalar INT4 dot products with `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32` PTX instructions. Each warp computes a 16×8 output tile using 64 INT4 elements per step, running natively on the A6000's third-generation tensor cores (SM86).

```ptx
mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
    {c0,c1,c2,c3}, {a.x,a.y,a.z,a.w}, {b.x,b.y}, {c0,c1,c2,c3};
```

Fallback path for SM75 uses `m8n8k32` instructions for compatibility.

### 4. Double-buffered async tile loading (`cp.async`)
Used `cp.async.ca.shared.global` to pipeline global→shared memory transfers with compute. While the current k-tile is being processed by the tensor cores, the next tile is already being loaded asynchronously — hiding memory latency.

```
tile k:   [load k+1 async] → [cp_wait] → [compute k] → repeat
```

Two shared memory buffers alternate (ping-pong) so loads and compute never conflict.

### 5. Scale caching in shared memory
**Original:** Each warp fetched `scales_A` and `scales_B` directly from global memory inside the hot loop — scattered reads across rows and columns of the scale matrices.

**Optimized:** At the start of each k-tile iteration, all 256 threads cooperatively load the 128 A-scales and 128 B-scales for the current block tile into shared memory. The inner `nt` loop then reads from SMEM (a few cycles) instead of global memory.

```cpp
if (tid < BLOCK_M) sScaleA[tid] = scales_A[(bm + tid) * num_groups + g];
if (tid < BLOCK_N) sScaleB[tid] = scales_B[(bn + tid) * num_groups + g];
__syncthreads();
// inner loop reads sScaleA[local_m_lo], sScaleB[local_c0], etc.
```

### 6. Merged sync barriers
Scale loads were originally issued after a `__syncthreads()`, requiring a second sync before compute — 3 barriers per tile. By issuing scale loads *before* `cp_wait + __syncthreads()`, a single barrier covers both the async tile data and the scale SMEM stores, reducing to 2 barriers per tile.

For K=3072 (48 tiles) this saves 48 `__syncthreads()` calls per block.

### 7. L2 XOR swizzle
**Problem:** Default linear block rasterization means all blocks in the same row run consecutively — good for A reuse but B tiles are fetched cold every time. On the A6000's 6 MB L2, neither A nor B fits, making this critical.

**Solution:** XOR the lower 3 bits of `blockIdx.y` into `blockIdx.x` before computing `bm`/`bn`. This creates a diagonal access pattern where blocks scheduled close together in time access a stripe of the output matrix, improving reuse of both A rows and B columns in L2.

```cpp
constexpr int LOG_SWIZZLE = 3;
const int swizzled_bx = blockIdx.x ^ ((blockIdx.y & 7));
const int bn = swizzled_bx * BLOCK_N;
```

The XOR is a bijection when `gridDim.x` is a multiple of 8 — enforced by a `TORCH_CHECK` at launch. All benchmark shapes satisfy this (gridDim.x ∈ {24, 72, 96}).

### 8. Precomputed scale products
`sa_lo` and `sa_hi` are loop-invariant across the `TILES_N` loop. Rather than computing `sa_lo * sb0`, `sa_lo * sb1`, etc. inline in the accumulate expressions, all 8 `sa*sb` products are precomputed before the accumulation lines. This reduces multiply count by 16 per tile per warp and gives the compiler better scheduling latitude alongside MMA latency.

```cpp
float slo0 = sa_lo * sb0, slo1 = sa_lo * sb1;
float slo2 = sa_lo * sb2, slo3 = sa_lo * sb3;
float shi0 = sa_hi * sb0, shi1 = sa_hi * sb1;
float shi2 = sa_hi * sb2, shi3 = sa_hi * sb3;

acc[nt][0][0] += (float)p0[0] * slo0;  // not sa_lo * sb0
```

### 9. `half2` output stores
Adjacent output pairs `(c0, c1)` and `(c2, c3)` are written using a single 32-bit `half2` store instead of two 16-bit stores, halving store instruction count. `c0` is always even-aligned (bn is a multiple of 128, tile offsets are even), guaranteeing `half2` alignment.

```cpp
*reinterpret_cast<half2*>(&C[m_lo * N + c0]) =
    __halves2half2(__float2half(acc[nt][0][0]), __float2half(acc[nt][0][1]));
```

### 10. Aligned-shape fast path
All benchmark shapes have M and N as multiples of `BLOCK_M`/`BLOCK_N` (128). The writeback loop's boundary checks (`if (m_lo < M)`, `if (c0+1 < N)`, etc.) are never taken but still consume predicate evaluation per thread. A runtime check selects a branchless fast path for aligned shapes:

```cpp
const bool aligned = (M % BLOCK_M == 0) && (N % BLOCK_N == 0);
if (aligned) {
    // unconditional half2 stores, no bounds checks
} else {
    // original guarded stores
}
```

### 11. A6000-specific: 96 KB shared memory opt-in
**Critical for occupancy.** Our kernel uses ~25 KB of shared memory per block. Under Ampere's default 48 KB carveout, only **1 block per SM** can run concurrently — leaving 84 SMs severely underutilized with limited ability to hide GDDR6 memory latency.

By opting into the 96 KB carveout:

```cpp
cudaFuncSetAttribute(gemm_int4_mma_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
cudaFuncSetAttribute(gemm_int4_mma_kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared);
```

The SM can now schedule **3 blocks concurrently** (3 × 25 KB = 75 KB < 96 KB), tripling the warp pool available to hide memory latency — the primary bottleneck on the A6000's 768 GB/s GDDR6.

---

## Weight Quantization (`quantize.py`)

### 12. Percentile clipping instead of max-abs scaling
**Original (reference):** `scale = max(|x|) / 7` — a single outlier can dominate the scale and crush quantization resolution for 99%+ of values.

**Optimized:** `scale = percentile_99.9(|x|) / 7` — clips outliers before computing the scale, recovering resolution for the bulk of the weight distribution. Implemented using `torch.kthvalue` (partial select, O(n)) rather than `torch.quantile` (full sort, O(n log n)).

```python
k = max(1, int(0.999 * group_size))  # e.g. k=63 for group_size=64
clip = torch.kthvalue(abs_w, k, dim=-1, keepdim=True).values
scale = clip / 7.0
```

This improves cosine similarity between the INT4 GEMM output and FP16 reference, providing margin above the correctness threshold.

---

## Summary Table

| Optimization | Where | Type |
|---|---|---|
| Single-pass register caching | quantize kernel | Memory traffic |
| `half2` vectorized loads | quantize kernel | Instruction count |
| INT4 tensor core MMA (m16n8k64) | GEMM kernel | Compute throughput |
| Double-buffered `cp.async` | GEMM kernel | Latency hiding |
| Scale caching in shared memory | GEMM kernel | Memory traffic |
| Merged sync barriers | GEMM kernel | Synchronization overhead |
| L2 XOR swizzle | GEMM kernel | Cache utilization |
| Precomputed scale products | GEMM kernel | Instruction count |
| `half2` output stores | GEMM kernel | Instruction count |
| Aligned-shape fast path | GEMM kernel | Branch elimination |
| 96 KB SMEM opt-in (A6000) | GEMM launcher | Occupancy |
| Percentile clipping + kthvalue | quantize.py | Quantization quality |

---

## References

### Tensor Cores & MMA
- **NVIDIA PTX ISA Reference** — `mma.sync.aligned` instruction documentation for s4/s8/f16 operand types  
  https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma
- **NVIDIA Ampere Architecture Whitepaper** (2020) — third-generation tensor cores, SM86 capabilities, shared memory carveout configuration  
  https://images.nvidia.com/aio-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
- **CUTLASS 3.x** (NVIDIA, 2023) — reference implementation for double-buffered `cp.async` pipelines, warp-level MMA tiling, and shared memory swizzle patterns  
  https://github.com/NVIDIA/cutlass

### Async Memory & Pipelining
- **NVIDIA `cp.async` Programming Guide** — `cp.async.ca.shared.global`, commit/wait group semantics, double-buffering patterns  
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies
- **"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking"** — Jia et al., 2018. Analysis of shared memory bank conflicts, warp scheduling, and memory latency hiding relevant to Ampere successors  
  https://arxiv.org/abs/1804.06826

### INT4 GEMM
- **MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models** — Frantar & Alistarh, 2024. Introduces weight layout transformation for INT4 MMA, 4-stage async pipelines, and scale interleaving for LLM inference. Directly relevant to this kernel's design.  
  https://arxiv.org/abs/2408.11743
- **FlashAttention-2** — Dao, 2023. Establishes the pattern of fused kernel design, tiled shared memory accumulation, and minimizing HBM/GDDR reads that underpins this kernel's tiling strategy.  
  https://arxiv.org/abs/2307.08691

### Quantization
- **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** — Frantar et al., 2022. Established INT4 weight quantization as viable for transformer inference; motivates the per-group symmetric quantization scheme used here.  
  https://arxiv.org/abs/2210.17323
- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** — Lin et al., 2023. Shows that protecting salient weight channels (high activation magnitude) via per-channel scaling improves INT4 accuracy — basis for the percentile clipping approach in `quantize.py`.  
  https://arxiv.org/abs/2306.00978
- **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models** — Xiao et al., 2022 (NVIDIA/MIT). Demonstrates that migrating quantization difficulty from activations to weights via smoothing improves joint A×W INT8/INT4 accuracy.  
  https://arxiv.org/abs/2211.10438

### L2 Cache Swizzle
- **CUTLASS TileScheduler with Swizzle** — NVIDIA, 2023. Documents the XOR-based CTA rasterization pattern used to improve L2 reuse in large GEMM workloads.  
  https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/tile_scheduler.hpp
- **"Optimizing Parallel Reduction in CUDA"** — Harris, NVIDIA, 2007. Classic reference for shared memory access patterns, bank conflict avoidance, and warp-level efficiency — foundational to the SMEM padding and access strategies used here.  
  https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
