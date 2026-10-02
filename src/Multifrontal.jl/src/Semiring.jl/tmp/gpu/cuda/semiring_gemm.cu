// Semiring GEMM  C ← C ⊕ A ⊗ B  (column-major, leading dimensions, any m, n, k).
//
// Same contract as SemiringGPU.sgemx_gpu! / Semiring.sgemx!(s, Val(:N), Val(:N), C, A, B):
// out-of-range entries of A are padded with the semiring zero and those of B with
// one; the accumulator starts at zero and is ⊕-ed into C at the end
// (C[i, j] = acc ⊕ C[i, j]).
//
// Structure (see cuda/NOTES.md for the sources of each idea):
//   - CUTLASS/cuASR 2-stage SIMT mainloop: the next k-tile is fetched from global
//     memory into registers while the current one is consumed from shared memory,
//     then stored into the other shared buffer; one __syncthreads per k-tile.
//   - cuASR warp tiling: each warp owns a WM × WN tile; its lanes form an LM × LN
//     grid, and each lane owns 4-wide groups of rows / columns, so fragment reads
//     from shared memory are LDS.128 with at most one wavefront per warp.
//   - shared layouts As[k][m] (A is m-contiguous: no transpose) and Bs[k][n + 4]
//     (B is k-contiguous: transposing store, padded by 4 words so it is free of
//     bank conflicts, as cuASR pads transposed operands).
//   - TropicalGemm-style interior fast path: tiles that are fully inside m, n, k
//     are fetched with 128-bit loads and no bounds checks; the others with
//     coalesced, checked scalar loads (the branch is uniform per block).
//   - vectorized (128-bit) read-modify-write of C in interior blocks (ROME).
#include "semiring.cuh"

namespace sr {

#ifndef MINB_SMALL
#define MINB_SMALL 3
#endif

template <int BM_, int BN_, int BK_, int WARPS_M_, int WARPS_N_, int LM_>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, BK = BK_;
    static constexpr int WARPS_M = WARPS_M_, WARPS_N = WARPS_N_;
    static constexpr int NT = WARPS_M * WARPS_N * 32;
    static constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
    static constexpr int LM = LM_, LN = 32 / LM_;
    static constexpr int TM = WM / LM, TN = WN / LN;
    static constexpr int VM = TM < 4 ? TM : 4, VN = TN < 4 ? TN : 4;
    static constexpr int PADB = 4;
    // occupancy target for __launch_bounds__ (FP32): 2 blocks of 256 threads (≤ 128 registers) for
    // 8 × 8 and 8 × 4 thread tiles and BK = 32; MINB_SMALL (3: ≤ 85 registers) for the memory-bound
    // 4 × 4 tiles with BK ≤ 16 (+6–20% on skinny shapes; measured with build-mb2, cuda/NOTES.md)
    template <class T> static constexpr int minblocks() {
        return sizeof(T) != 4 ? 1 : (TM * TN <= 16 && BK < 32 ? MINB_SMALL * 256 / NT : 2 * 256 / NT);
    }
    static_assert(WM * WARPS_M == BM && WN * WARPS_N == BN, "warp tiling");
    static_assert(TM * LM == WM && TN * LN == WN, "lane tiling");
    static_assert(TM % VM == 0 && TN % VN == 0, "vector groups");
    static_assert(BK % 4 == 0 && BM % 4 == 0, "vector loads");
};

// 4 contiguous elements (16-byte aligned for float, 2 × 16 bytes for double)
template <class T> struct V4 { T x[4]; };

__device__ __forceinline__ V4<float> ldg4(const float* p) {
    float4 v = __ldg(reinterpret_cast<const float4*>(p));
    return {{v.x, v.y, v.z, v.w}};
}
__device__ __forceinline__ V4<double> ldg4(const double* p) {
    double2 a = __ldg(reinterpret_cast<const double2*>(p));
    double2 b = __ldg(reinterpret_cast<const double2*>(p) + 1);
    return {{a.x, a.y, b.x, b.y}};
}
__device__ __forceinline__ V4<float> ld4(const float* p) {
    float4 v = *reinterpret_cast<const float4*>(p);
    return {{v.x, v.y, v.z, v.w}};
}
__device__ __forceinline__ V4<double> ld4(const double* p) {
    double2 a = reinterpret_cast<const double2*>(p)[0];
    double2 b = reinterpret_cast<const double2*>(p)[1];
    return {{a.x, a.y, b.x, b.y}};
}
__device__ __forceinline__ void st4(float* p, const V4<float>& v) {
    *reinterpret_cast<float4*>(p) = make_float4(v.x[0], v.x[1], v.x[2], v.x[3]);
}
__device__ __forceinline__ void st4(double* p, const V4<double>& v) {
    reinterpret_cast<double2*>(p)[0] = make_double2(v.x[0], v.x[1]);
    reinterpret_cast<double2*>(p)[1] = make_double2(v.x[2], v.x[3]);
}

// NV contiguous elements from shared memory (NV ∈ {1, 2, 4}); 16-byte vector when possible
template <int NV, class T>
__device__ __forceinline__ void lds(T* out, const T* p) {
    if constexpr (NV == 4) {
        V4<T> v = ld4(p);
#pragma unroll
        for (int i = 0; i < 4; ++i) out[i] = v.x[i];
    } else {
#pragma unroll
        for (int i = 0; i < NV; ++i) out[i] = p[i];
    }
}

template <class S, class T, class CF>
__global__ void __launch_bounds__(CF::NT, CF::template minblocks<T>())
sgemm_kernel(int m, int n, int k, const T* __restrict__ A, int64_t lda, const T* __restrict__ B, int64_t ldb,
             T* __restrict__ C, int64_t ldc, bool vecA, bool vecB, bool vecC) {
    constexpr int BM = CF::BM, BN = CF::BN, BK = CF::BK, NT = CF::NT;
    constexpr int WM = CF::WM, WN = CF::WN, LM = CF::LM, LN = CF::LN;
    constexpr int TM = CF::TM, TN = CF::TN, VM = CF::VM, VN = CF::VN;
    constexpr int LDB = BN + CF::PADB;
    // elements of the A (B) tile per thread: scalar mode SA, vector mode 4 VA
    constexpr int SA = (BM * BK + NT - 1) / NT, VA = (BM * BK / 4 + NT - 1) / NT;
    constexpr int SB = (BK * BN + NT - 1) / NT, VB = (BK * BN / 4 + NT - 1) / NT;
    constexpr int RA = SA > 4 * VA ? SA : 4 * VA;
    constexpr int RB = SB > 4 * VB ? SB : 4 * VB;

    __shared__ __align__(16) T As[2][BK][BM];
    __shared__ __align__(16) T Bs[2][BK][LDB];

    const int t = threadIdx.x;
    const int warp = t / 32, lane = t % 32;
    const int wm = warp % CF::WARPS_M, wn = warp / CF::WARPS_M;
    const int lm = lane % LM, ln = lane / LM;
    const int i0 = blockIdx.x * BM, j0 = blockIdx.y * BN;
    const bool innerA = i0 + BM <= m, innerB = j0 + BN <= n;
    const bool inner = innerA && innerB;

    const T z = S::template zero<T>();
    const T u = S::template one<T>();

    T acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = z;

    T ra[RA], rb[RB];

    // ----- global → registers -----
    auto fetch = [&](int k0, bool& va, bool& vb) {
        const bool fullk = k0 + BK <= k;
        va = fullk && innerA && vecA;   // the A tile only needs rows i0 : i0+BM in range
        vb = fullk && innerB && vecB;
        if (va) {
#pragma unroll
            for (int l = 0; l < VA; ++l) {
                const int e = t + l * NT;
                if ((BM * BK / 4) % NT == 0 || e < BM * BK / 4) {
                    const int r = (e % (BM / 4)) * 4, c = e / (BM / 4);
                    V4<T> v = ldg4(A + (int64_t)(i0 + r) + (int64_t)(k0 + c) * lda);
#pragma unroll
                    for (int q = 0; q < 4; ++q) ra[4 * l + q] = v.x[q];
                }
            }
        } else {
#pragma unroll
            for (int l = 0; l < SA; ++l) {
                const int e = t + l * NT;
                if ((BM * BK) % NT == 0 || e < BM * BK) {
                    const int r = e % BM, c = e / BM;
                    const int gi = i0 + r, gc = k0 + c;
                    ra[l] = (gi < m && gc < k) ? __ldg(A + (int64_t)gi + (int64_t)gc * lda) : z;
                }
            }
        }
        if (vb) {
#pragma unroll
            for (int l = 0; l < VB; ++l) {
                const int e = t + l * NT;
                if ((BK * BN / 4) % NT == 0 || e < BK * BN / 4) {
                    const int kq = (e % (BK / 4)) * 4, j = e / (BK / 4);
                    V4<T> v = ldg4(B + (int64_t)(k0 + kq) + (int64_t)(j0 + j) * ldb);
#pragma unroll
                    for (int q = 0; q < 4; ++q) rb[4 * l + q] = v.x[q];
                }
            }
        } else {
#pragma unroll
            for (int l = 0; l < SB; ++l) {
                const int e = t + l * NT;
                if ((BK * BN) % NT == 0 || e < BK * BN) {
                    const int kk = e % BK, j = e / BK;
                    const int gk = k0 + kk, gj = j0 + j;
                    rb[l] = (gk < k && gj < n) ? __ldg(B + (int64_t)gk + (int64_t)gj * ldb) : u;
                }
            }
        }
    };

    // ----- registers → shared buffer -----
    auto stash = [&](int buf, bool va, bool vb) {
        if (va) {
#pragma unroll
            for (int l = 0; l < VA; ++l) {
                const int e = t + l * NT;
                if ((BM * BK / 4) % NT == 0 || e < BM * BK / 4) {
                    const int r = (e % (BM / 4)) * 4, c = e / (BM / 4);
                    V4<T> v;
#pragma unroll
                    for (int q = 0; q < 4; ++q) v.x[q] = ra[4 * l + q];
                    st4(&As[buf][c][r], v);
                }
            }
        } else {
#pragma unroll
            for (int l = 0; l < SA; ++l) {
                const int e = t + l * NT;
                if ((BM * BK) % NT == 0 || e < BM * BK) As[buf][e / BM][e % BM] = ra[l];
            }
        }
        if (vb) {
#pragma unroll
            for (int l = 0; l < VB; ++l) {
                const int e = t + l * NT;
                if ((BK * BN / 4) % NT == 0 || e < BK * BN / 4) {
                    const int kq = (e % (BK / 4)) * 4, j = e / (BK / 4);
#pragma unroll
                    for (int q = 0; q < 4; ++q) Bs[buf][kq + q][j] = rb[4 * l + q];
                }
            }
        } else {
#pragma unroll
            for (int l = 0; l < SB; ++l) {
                const int e = t + l * NT;
                if ((BK * BN) % NT == 0 || e < BK * BN) Bs[buf][e % BK][e / BK] = rb[l];
            }
        }
    };

    const int arow = wm * WM + lm * VM;   // first row of the thread's fragment (within the block tile)
    const int bcol = wn * WN + ln * VN;
    const int ntile = (k + BK - 1) / BK;

    bool va, vb;
    fetch(0, va, vb);
    stash(0, va, vb);
    __syncthreads();

    for (int q = 0; q < ntile; ++q) {
        const int cur = q & 1;
        if (q + 1 < ntile) fetch((q + 1) * BK, va, vb);

#pragma unroll
        // two k-steps at a time: acc ← (acc ⊕ a₀b₀) ⊕ a₁b₁ written as one expression, so that
        // on sm_100 ptxas fuses the two ⊕ = min/max into one 3-input FMNMX3 (same result, bit
        // for bit: it is the same pair of minnum operations). On sm_120 it stays 2 × FMNMX.
#pragma unroll
        for (int kk = 0; kk < BK; kk += 2) {
            T a0[TM], b0[TN], a1[TM], b1[TN];
#pragma unroll
            for (int g = 0; g < TM / VM; ++g) lds<VM>(a0 + g * VM, &As[cur][kk][arow + g * LM * VM]);
#pragma unroll
            for (int g = 0; g < TN / VN; ++g) lds<VN>(b0 + g * VN, &Bs[cur][kk][bcol + g * LN * VN]);
#pragma unroll
            for (int g = 0; g < TM / VM; ++g) lds<VM>(a1 + g * VM, &As[cur][kk + 1][arow + g * LM * VM]);
#pragma unroll
            for (int g = 0; g < TN / VN; ++g) lds<VN>(b1 + g * VN, &Bs[cur][kk + 1][bcol + g * LN * VN]);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = S::muladd(a1[i], b1[j], S::muladd(a0[i], b0[j], acc[i][j]));
        }

        if (q + 1 < ntile) stash(cur ^ 1, va, vb);
        __syncthreads();
    }

    // ----- epilogue: C ← acc ⊕ C -----
    // Two phases: all loads of the thread's part of C first, then all stores. Written
    // as one load-⊕-store per element (as in sgemx_kernel2!), the compiler cannot
    // prove that a store does not alias the next load (ldc is a runtime value), so
    // every load waits for the previous store: for small k the GEMM is bound by
    // this read-modify-write of C.
    // Only for thread tiles of ≤ 32 values: for 8 × 8 tiles the extra registers make
    // the mainloop spill, and those tiles are compute-bound anyway.
    constexpr bool TWO_PHASE = TM * TN <= 32;
    const bool vc = VM == 4 && inner && vecC;
    if constexpr (TWO_PHASE) {
        T cv[TM][TN];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gj = j0 + bcol + (j / VN) * LN * VN + j % VN;
#pragma unroll
            for (int g = 0; g < TM / VM; ++g) {
                const int gi = i0 + arow + g * LM * VM;
                const T* p = C + (int64_t)gi + (int64_t)gj * ldc;
                if (vc) {
                    V4<T> c = ld4(p);
#pragma unroll
                    for (int v = 0; v < 4; ++v) cv[g * VM + v][j] = c.x[v];
                } else {
#pragma unroll
                    for (int v = 0; v < VM; ++v) cv[g * VM + v][j] = (gj < n && gi + v < m) ? p[v] : z;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gj = j0 + bcol + (j / VN) * LN * VN + j % VN;
#pragma unroll
            for (int g = 0; g < TM / VM; ++g) {
                const int gi = i0 + arow + g * LM * VM;
                T* p = C + (int64_t)gi + (int64_t)gj * ldc;
                if (vc) {
                    V4<T> c;
#pragma unroll
                    for (int v = 0; v < 4; ++v) c.x[v] = S::add(acc[g * VM + v][j], cv[g * VM + v][j]);
                    st4(p, c);
                } else {
#pragma unroll
                    for (int v = 0; v < VM; ++v)
                        if (gj < n && gi + v < m) p[v] = S::add(acc[g * VM + v][j], cv[g * VM + v][j]);
                }
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gj = j0 + bcol + (j / VN) * LN * VN + j % VN;
#pragma unroll
            for (int g = 0; g < TM / VM; ++g) {
                const int gi = i0 + arow + g * LM * VM;
                T* p = C + (int64_t)gi + (int64_t)gj * ldc;
                if (vc) {
                    V4<T> c = ld4(p);
#pragma unroll
                    for (int v = 0; v < 4; ++v) c.x[v] = S::add(acc[g * VM + v][j], c.x[v]);
                    st4(p, c);
                } else if (gj < n) {
#pragma unroll
                    for (int v = 0; v < VM; ++v)
                        if (gi + v < m) p[v] = S::add(acc[g * VM + v][j], p[v]);
                }
            }
        }
    }
}

// tile configurations (TILING codes for sr_gemm)
using CfgL = Cfg<128, 128, 8, 2, 4, 8>;     // 1: 256 thr, 8 × 8 per thread (cuASR 128×128×8, warp 64×32)
using CfgM = Cfg<128, 64, 8, 2, 4, 8>;      // 2: 256 thr, 8 × 4
using CfgS = Cfg<64, 64, 8, 2, 4, 8>;       // 3: 256 thr, 4 × 4
using CfgN32 = Cfg<128, 32, 8, 4, 2, 8>;    // 4: 256 thr, 4 × 4, skinny n ≤ 32
using CfgN16 = Cfg<256, 16, 8, 8, 1, 8>;    // 5: 256 thr, 4 × 4, skinny n ≤ 16
using CfgL16 = Cfg<128, 128, 16, 2, 4, 8>;  // 6: as 1 with BK = 16
using CfgM16 = Cfg<128, 64, 16, 2, 4, 8>;   // 7: as 2 with BK = 16
using CfgN16b = Cfg<128, 16, 8, 4, 1, 8>;   // 8: 128 thr, 4 × 4, n ≤ 16
using CfgN32k = Cfg<128, 32, 32, 4, 2, 8>;  // 9: as 4 with BK = 32 (TropicalGemm's FP32 BK)
using CfgN16k = Cfg<128, 16, 32, 4, 1, 8>;  // 10: as 8 with BK = 32

static int nsm() {
    static int v = 0;
    if (v == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev);
    }
    return v;
}

static inline int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// measured on the RTX 5060 Laptop (bench/bench_cuda_backend.jl gemm, cuda/NOTES.md)
int choose_tiling(int m, int n, int k) {
    const int64_t nsm_ = nsm();
    if (n <= 16) return 8;                                          // 128 × 16
    if (n <= 32) return 4;                                          // 128 × 32
    if (cdiv(m, 128) * cdiv(n, 64) < nsm_) return 3;                // too few 128 × 64 tiles: 64 × 64
    if (k <= 64 && (int64_t)m * n >= (int64_t(1) << 24)) return 2;  // C read-modify-write from DRAM dominates: 128 × 64, BK = 8
    if (k >= 256 && cdiv(m, 128) * cdiv(n, 128) >= 4 * nsm_) return 6;  // large: 128 × 128, BK = 16
    return 7;                                                       // 128 × 64, BK = 16
}

template <class S, class T, class CF>
static int run(int m, int n, int k, const T* A, int64_t lda, const T* B, int64_t ldb, T* C, int64_t ldc, cudaStream_t st) {
    auto al = [](const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; };
    constexpr int64_t V = 16 / sizeof(T);  // elements per 16-byte segment
    const bool vecA = al(A) && lda % V == 0;
    const bool vecB = al(B) && ldb % V == 0;
    const bool vecC = al(C) && ldc % V == 0;
    dim3 grid((unsigned)cdiv(m, CF::BM), (unsigned)cdiv(n, CF::BN));
    sgemm_kernel<S, T, CF><<<grid, CF::NT, 0, st>>>(m, n, k, A, lda, B, ldb, C, ldc, vecA, vecB, vecC);
    return (int)cudaGetLastError();
}

template <class S, class T>
int gemm(int tiling, int m, int n, int k, const void* A, int64_t lda, const void* B, int64_t ldb, void* C, int64_t ldc,
         cudaStream_t st) {
    if (m <= 0 || n <= 0 || k <= 0) return 0;
    const T* a = static_cast<const T*>(A);
    const T* b = static_cast<const T*>(B);
    T* c = static_cast<T*>(C);
    if (tiling == 0) tiling = choose_tiling(m, n, k);
    if (sizeof(T) == 8) {  // FP64: 8 × 8 tiles spill, BK ≥ 16 overflows 48 KB of static shared memory
        if (tiling == 1 || tiling == 6 || tiling == 7) tiling = 2;
        if (tiling == 9) tiling = 4;
        if (tiling == 10) tiling = 8;
    }
    switch (tiling) {
        case 1: if constexpr (sizeof(T) == 8) return -2; else return run<S, T, CfgL>(m, n, k, a, lda, b, ldb, c, ldc, st);
        case 2: return run<S, T, CfgM>(m, n, k, a, lda, b, ldb, c, ldc, st);
        case 3: return run<S, T, CfgS>(m, n, k, a, lda, b, ldb, c, ldc, st);
        case 4: return run<S, T, CfgN32>(m, n, k, a, lda, b, ldb, c, ldc, st);
        case 5: return run<S, T, CfgN16>(m, n, k, a, lda, b, ldb, c, ldc, st);
        case 6: if constexpr (sizeof(T) == 4) return run<S, T, CfgL16>(m, n, k, a, lda, b, ldb, c, ldc, st); else return -2;
        case 7: if constexpr (sizeof(T) == 4) return run<S, T, CfgM16>(m, n, k, a, lda, b, ldb, c, ldc, st); else return -2;
        case 8: return run<S, T, CfgN16b>(m, n, k, a, lda, b, ldb, c, ldc, st);
        case 9: if constexpr (sizeof(T) == 4) return run<S, T, CfgN32k>(m, n, k, a, lda, b, ldb, c, ldc, st); else return -2;
        case 10: if constexpr (sizeof(T) == 4) return run<S, T, CfgN16k>(m, n, k, a, lda, b, ldb, c, ldc, st); else return -2;
        default: return -2;
    }
}

}  // namespace sr

extern "C" {

// C ← C ⊕ A ⊗ B; returns a cudaError_t (0 = success), -1 unknown semiring/type, -2 unknown tiling
int sr_gemm(int sr_code, int dt_code, int tiling, int m, int n, int k, const void* A, int64_t lda, const void* B, int64_t ldb,
            void* C, int64_t ldc, cudaStream_t stream) {
    SR_DISPATCH(sr_code, dt_code, (sr::gemm<S, T>(tiling, m, n, k, A, lda, B, ldb, C, ldc, stream)));
}

int sr_gemm_choose_tiling(int m, int n, int k) { return sr::choose_tiling(m, n, k); }

}
