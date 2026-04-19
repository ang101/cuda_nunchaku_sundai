#include <cuda_fp16.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// ============================================================
// INT4 QUANTIZATION
// ============================================================

// Baseline quantizer: per-row, per-group symmetric INT4 quantization.
// Packs two signed INT4 values per byte:
//   low nibble  = even element
//   high nibble = odd element
__global__ void quantize_int4_kernel(
    const half* __restrict__ input,   // [M, K]
    uint8_t* __restrict__ output,     // [M, K/2]
    half* __restrict__ scales,        // [M, num_groups]
    int M,
    int K,
    int group_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int group = blockIdx.y;

    if (row >= M) return;

    int num_groups = K / group_size;
    int k_start = group * group_size;
    int rowK = row * K;
    int rowK2 = row * (K / 2);

    // Single pass with half2 vectorized loads; vals[] cached in registers.
    // float[128] per thread costs ~64 regs at group_size=64 — acceptable occupancy.
    // For group_size > 128 switch to a shared-memory buffer instead.
    float vals[128];
    float max_abs = 0.0f;
    for (int i = 0; i < group_size; i += 2) {
        half2 h2 = *reinterpret_cast<const half2*>(&input[rowK + k_start + i]);
        float v0 = __half2float(__low2half(h2));
        float v1 = __half2float(__high2half(h2));
        vals[i]   = v0;
        vals[i+1] = v1;
        max_abs = fmaxf(max_abs, fmaxf(fabsf(v0), fabsf(v1)));
    }

    float scale = max_abs / 7.0f;
    scales[row * num_groups + group] = __float2half(scale);

    float rscale = (max_abs > 0.0f) ? (7.0f / max_abs) : 0.0f;

    int out_offset = rowK2 + k_start / 2;
    for (int i = 0; i < group_size; i += 2) {
        float val_even = vals[i];
        float val_odd  = vals[i + 1];

        int q_even = __float2int_rn(val_even * rscale);
        int q_odd  = __float2int_rn(val_odd * rscale);

        q_even = max(-8, min(7, q_even));
        q_odd  = max(-8, min(7, q_odd));

        uint8_t packed = (uint8_t)((q_odd & 0xF) << 4) | (uint8_t)(q_even & 0xF);
        output[out_offset + i / 2] = packed;
    }
}

std::vector<torch::Tensor> quantize_int4_custom(torch::Tensor input, int group_size) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kHalf, "input must be float16");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");

    int M = input.size(0);
    int K = input.size(1);

    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(group_size % 2 == 0, "group_size must be even");
    TORCH_CHECK(group_size <= 128, "group_size must be <= 128 for current kernel (increase vals[] or switch to shared memory)");

    auto output = torch::empty(
        {M, K / 2},
        torch::TensorOptions().dtype(torch::kUInt8).device(input.device())
    );
    int num_groups = K / group_size;
    auto scales = torch::empty(
        {M, num_groups},
        torch::TensorOptions().dtype(torch::kHalf).device(input.device())
    );

    dim3 block(256);
    dim3 grid((M + 255) / 256, num_groups);

    quantize_int4_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        output.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
        M, K, group_size
    );

    return {output, scales};
}

// ============================================================
// INT4 GEMM: MMA / TENSOR CORE VERSION
// Requires SM80+ for the main path.
// ============================================================

static constexpr int BLOCK_M   = 128;
static constexpr int BLOCK_N   = 128;
static constexpr int BLOCK_K   = 64;
static constexpr int WARP_SZ   = 32;
static constexpr int NUM_WARPS = 8;
static constexpr int WARP_M    = BLOCK_M / NUM_WARPS;   // 16
static constexpr int TILES_N   = BLOCK_N / 16;          // 8
static constexpr int SMEM_STRIDE = BLOCK_K / 2 + 16;    // 48 bytes per row

__device__ __forceinline__ void mma_s4(uint4 a, uint2 b, int (&c)[4]) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
#else
    asm volatile("{"
        ".reg .b32 t0,t1,t2,t3;\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {t0,t1},{%4},{%8},{%0,%1};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {t2,t3},{%5},{%8},{%2,%3};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1},{%6},{%9},{t0,t1};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%2,%3},{%7},{%9},{t2,t3};\n"
        "}\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
#endif
}

__device__ __forceinline__ void cp_async_16(void *dst, const void *src, bool pred) {
    unsigned s = __cvta_generic_to_shared(dst);
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p,%2,0;\n"
        "  @p cp.async.ca.shared.global [%0],[%1],16;\n"
        "  @!p st.shared.v4.u32 [%0],{0,0,0,0}; }\n"
        :: "r"(s), "l"(src), "r"((int)pred));
}

__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_wait(int n) {
    if (n == 0) asm volatile("cp.async.wait_group 0;\n");
    else        asm volatile("cp.async.wait_group 1;\n");
}

__device__ __forceinline__ uint4 load_a_frag(const uint8_t *base, int stride) {
    int lane = threadIdx.x % WARP_SZ;
    int row_lo = lane / 4;
    int row_hi = row_lo + 8;
    int col = (lane % 4) * 4;

    uint4 a;
    a.x = *(const uint32_t*)(base + row_lo * stride + col);
    a.y = *(const uint32_t*)(base + row_hi * stride + col);
    a.z = *(const uint32_t*)(base + row_lo * stride + 16 + col);
    a.w = *(const uint32_t*)(base + row_hi * stride + 16 + col);
    return a;
}

__device__ __forceinline__ uint2 load_b_frag(const uint8_t *base, int stride) {
    int lane = threadIdx.x % WARP_SZ;
    int row = lane / 4;
    int col = (lane % 4) * 4;

    uint2 b;
    b.x = *(const uint32_t*)(base + row * stride + col);
    b.y = *(const uint32_t*)(base + row * stride + 16 + col);
    return b;
}

__global__ void gemm_int4_mma_kernel(
    const uint8_t *__restrict__ A,
    const uint8_t *__restrict__ B,
    const half    *__restrict__ scales_A,
    const half    *__restrict__ scales_B,
    half          *__restrict__ C,
    int M, int N, int K, int group_size
) {
    // L2 swizzle: XOR lower LOG_SWIZZLE bits of blockIdx.y into blockIdx.x.
    // Requires gridDim.x to be a multiple of (1<<LOG_SWIZZLE) — enforced in launcher.
    static constexpr int LOG_SWIZZLE = 3;  // swizzle factor = 8 blocks
    const int swizzled_bx = blockIdx.x ^ ((blockIdx.y & ((1 << LOG_SWIZZLE) - 1)));
    const int bm = blockIdx.y * BLOCK_M;
    const int bn = swizzled_bx * BLOCK_N;
    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SZ;
    const int laneId = tid % WARP_SZ;
    const int halfK = K / 2;
    const int num_groups = K / group_size;
    const int num_k_tiles = K / BLOCK_K;

    extern __shared__ uint8_t smem[];
    const int tileA = BLOCK_M * SMEM_STRIDE;
    const int tileB = BLOCK_N * SMEM_STRIDE;
    uint8_t *sA0 = smem;
    uint8_t *sB0 = smem + tileA;
    uint8_t *sA1 = smem + tileA + tileB;
    uint8_t *sB1 = sA1 + tileA;
    uint8_t *sA[2] = {sA0, sA1};
    uint8_t *sB[2] = {sB0, sB1};

    half *sScaleA = reinterpret_cast<half*>(smem + 2 * (tileA + tileB));
    half *sScaleB = sScaleA + BLOCK_M;

    float acc[TILES_N][2][4];
    #pragma unroll
    for (int j = 0; j < TILES_N; j++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            acc[j][h][0] = 0.f;
            acc[j][h][1] = 0.f;
            acc[j][h][2] = 0.f;
            acc[j][h][3] = 0.f;
        }
    }

    auto load_tile = [&](int kt, int s) {
        int kb = kt * (BLOCK_K / 2);

        {
            int row = tid / 2;
            int half_sel = tid % 2;
            bool p = (bm + row < M) && (kb + half_sel * 16 < halfK);
            cp_async_16(
                sA[s] + row * SMEM_STRIDE + half_sel * 16,
                A + (size_t)(bm + row) * halfK + kb + half_sel * 16,
                p
            );
        }

        {
            int row = tid / 2;
            int half_sel = tid % 2;
            bool p = (bn + row < N) && (kb + half_sel * 16 < halfK);
            cp_async_16(
                sB[s] + row * SMEM_STRIDE + half_sel * 16,
                B + (size_t)(bn + row) * halfK + kb + half_sel * 16,
                p
            );
        }

        cp_commit();
    };

    if (num_k_tiles > 0) load_tile(0, 0);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int s = kt & 1;

        if (kt + 1 < num_k_tiles) {
            load_tile(kt + 1, (kt + 1) & 1);
        }

        // Issue scale loads before the sync so they are covered by a single barrier
        // instead of two — saves one __syncthreads() per k-tile (48 syncs for K=3072).
        int g = (kt * BLOCK_K) / group_size;
        if (tid < BLOCK_M)
            sScaleA[tid] = (bm + tid < M) ? scales_A[(bm + tid) * num_groups + g]
                                          : __float2half(0.f);
        if (tid < BLOCK_N)
            sScaleB[tid] = (bn + tid < N) ? scales_B[(bn + tid) * num_groups + g]
                                          : __float2half(0.f);

        cp_wait(kt + 1 < num_k_tiles ? 1 : 0);
        __syncthreads();  // covers both cp.async tile data and scale SMEM stores

        int local_m_lo = warpId * WARP_M + laneId / 4;
        int local_m_hi = local_m_lo + 8;
        float sa_lo = __half2float(sScaleA[local_m_lo]);
        float sa_hi = __half2float(sScaleA[local_m_hi]);

        uint4 af = load_a_frag(sA[s] + warpId * WARP_M * SMEM_STRIDE, SMEM_STRIDE);

        #pragma unroll
        for (int nt = 0; nt < TILES_N; nt++) {
            int n_off = nt * 16;
            int local_c0 = n_off + (laneId % 4) * 2;

            uint2 bf0 = load_b_frag(sB[s] + (n_off + 0) * SMEM_STRIDE, SMEM_STRIDE);
            uint2 bf1 = load_b_frag(sB[s] + (n_off + 8) * SMEM_STRIDE, SMEM_STRIDE);

            int p0[4] = {0, 0, 0, 0};
            int p1[4] = {0, 0, 0, 0};

            mma_s4(af, bf0, p0);
            mma_s4(af, bf1, p1);

            // Opt #2: precompute sa*sb products outside the accumulate lines —
            // sa_lo/sa_hi are loop-invariant; fusing here lets the compiler
            // schedule the multiplies alongside MMA latency.
            float sb0 = __half2float(sScaleB[local_c0]);
            float sb1 = __half2float(sScaleB[local_c0 + 1]);
            float sb2 = __half2float(sScaleB[local_c0 + 8]);
            float sb3 = __half2float(sScaleB[local_c0 + 9]);
            float slo0 = sa_lo * sb0, slo1 = sa_lo * sb1;
            float slo2 = sa_lo * sb2, slo3 = sa_lo * sb3;
            float shi0 = sa_hi * sb0, shi1 = sa_hi * sb1;
            float shi2 = sa_hi * sb2, shi3 = sa_hi * sb3;

            acc[nt][0][0] += (float)p0[0] * slo0;
            acc[nt][0][1] += (float)p0[1] * slo1;
            acc[nt][0][2] += (float)p0[2] * shi0;
            acc[nt][0][3] += (float)p0[3] * shi1;

            acc[nt][1][0] += (float)p1[0] * slo2;
            acc[nt][1][1] += (float)p1[1] * slo3;
            acc[nt][1][2] += (float)p1[2] * shi2;
            acc[nt][1][3] += (float)p1[3] * shi3;
        }

        __syncthreads();
    }

    int m_lo = bm + warpId * WARP_M + laneId / 4;
    int m_hi = m_lo + 8;

    // Opt #1: fast path skips all boundary checks for shapes where M and N are
    // multiples of BLOCK_M/BLOCK_N (all benchmark shapes satisfy this).
    const bool aligned = (M % BLOCK_M == 0) && (N % BLOCK_N == 0);

    #pragma unroll
    for (int nt = 0; nt < TILES_N; nt++) {
        int c0 = bn + nt * 16 + (laneId % 4) * 2;
        int c2 = c0 + 8;

        if (aligned) {
            *reinterpret_cast<half2*>(&C[m_lo * N + c0]) =
                __halves2half2(__float2half(acc[nt][0][0]), __float2half(acc[nt][0][1]));
            *reinterpret_cast<half2*>(&C[m_lo * N + c2]) =
                __halves2half2(__float2half(acc[nt][1][0]), __float2half(acc[nt][1][1]));
            *reinterpret_cast<half2*>(&C[m_hi * N + c0]) =
                __halves2half2(__float2half(acc[nt][0][2]), __float2half(acc[nt][0][3]));
            *reinterpret_cast<half2*>(&C[m_hi * N + c2]) =
                __halves2half2(__float2half(acc[nt][1][2]), __float2half(acc[nt][1][3]));
        } else {
            if (m_lo < M) {
                if (c0 + 1 < N)
                    *reinterpret_cast<half2*>(&C[m_lo * N + c0]) =
                        __halves2half2(__float2half(acc[nt][0][0]), __float2half(acc[nt][0][1]));
                else if (c0 < N)
                    C[m_lo * N + c0] = __float2half(acc[nt][0][0]);
                if (c2 + 1 < N)
                    *reinterpret_cast<half2*>(&C[m_lo * N + c2]) =
                        __halves2half2(__float2half(acc[nt][1][0]), __float2half(acc[nt][1][1]));
                else if (c2 < N)
                    C[m_lo * N + c2] = __float2half(acc[nt][1][0]);
            }
            if (m_hi < M) {
                if (c0 + 1 < N)
                    *reinterpret_cast<half2*>(&C[m_hi * N + c0]) =
                        __halves2half2(__float2half(acc[nt][0][2]), __float2half(acc[nt][0][3]));
                else if (c0 < N)
                    C[m_hi * N + c0] = __float2half(acc[nt][0][2]);
                if (c2 + 1 < N)
                    *reinterpret_cast<half2*>(&C[m_hi * N + c2]) =
                        __halves2half2(__float2half(acc[nt][1][2]), __float2half(acc[nt][1][3]));
                else if (c2 < N)
                    C[m_hi * N + c2] = __float2half(acc[nt][1][2]);
            }
        }
    }
}

torch::Tensor gemm_int4_mma(
    torch::Tensor A_packed,
    torch::Tensor B_packed,
    torch::Tensor scales_A,
    torch::Tensor scales_B,
    int group_size
) {
    TORCH_CHECK(A_packed.is_cuda() && B_packed.is_cuda(), "A_packed and B_packed must be CUDA tensors");
    TORCH_CHECK(A_packed.dtype() == torch::kUInt8, "A_packed must be uint8");
    TORCH_CHECK(B_packed.dtype() == torch::kUInt8, "B_packed must be uint8");
    TORCH_CHECK(scales_A.dtype() == torch::kHalf, "scales_A must be float16");
    TORCH_CHECK(scales_B.dtype() == torch::kHalf, "scales_B must be float16");

    int M = A_packed.size(0);
    int K = A_packed.size(1) * 2;
    int N = B_packed.size(0);

    TORCH_CHECK(B_packed.size(1) * 2 == K, "A and B must have the same K dimension");
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(K % BLOCK_K == 0, "K must be divisible by BLOCK_K");
    TORCH_CHECK(BLOCK_K % group_size == 0,
        "BLOCK_K must be divisible by group_size for mma kernel");

    // Swizzle safety: XOR swizzle is a bijection only when gridDim.x is a
    // multiple of the swizzle factor (1 << LOG_SWIZZLE = 8).
    constexpr int SWIZZLE_FACTOR = 1 << 3;  // must match LOG_SWIZZLE in kernel
    int grid_n = (N + BLOCK_N - 1) / BLOCK_N;
    TORCH_CHECK(grid_n % SWIZZLE_FACTOR == 0,
        "N must produce a gridDim.x that is a multiple of 8 for L2 swizzle. "
        "Got gridDim.x=", grid_n, ". Ensure N is a multiple of ", BLOCK_N * SWIZZLE_FACTOR, ".");

    auto C = torch::empty(
        {M, N},
        torch::TensorOptions().dtype(torch::kHalf).device(A_packed.device())
    );

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(WARP_SZ * NUM_WARPS);
    int smem = 2 * (BLOCK_M * SMEM_STRIDE + BLOCK_N * SMEM_STRIDE)
             + (BLOCK_M + BLOCK_N) * (int)sizeof(half);

    // Opt-in to the 96KB shared memory carveout on Ampere (default is 48KB).
    // Our kernel uses ~25KB — under 48KB a single block per SM is all we get.
    // With 96KB we fit 3 blocks per SM, tripling occupancy and hiding more
    // memory latency on the A6000's GDDR6 bandwidth.
    cudaFuncSetAttribute(
        gemm_int4_mma_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem
    );
    cudaFuncSetAttribute(
        gemm_int4_mma_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared
    );

    gemm_int4_mma_kernel<<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
        A_packed.data_ptr<uint8_t>(),
        B_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K, group_size
    );

    return C;
}

// ============================================================
// DEFAULT DISPATCH
// ============================================================

torch::Tensor gemm_int4_custom(
    torch::Tensor A_packed,
    torch::Tensor B_packed,
    torch::Tensor scales_A,
    torch::Tensor scales_B,
    int group_size
) {
    return gemm_int4_mma(A_packed, B_packed, scales_A, scales_B, group_size);
}