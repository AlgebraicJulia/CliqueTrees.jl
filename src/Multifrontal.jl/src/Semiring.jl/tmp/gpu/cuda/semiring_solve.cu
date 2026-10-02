// Batched multifrontal solve kernels (closure / blocked queries) and the small
// helpers of the dense large-front path, for the CUDA C++ backend.
//
// Layout and conventions are those of SemiringGPU.GPUSLU (src/sgetrs.jl):
//   - C is nrhs × n, column-major, leading dimension ldc ("row layout": thread t
//     owns row t, so a warp reads 32 consecutive words of a column);
//   - every index array (Rptr, Sptr, Stgt, Dptr, Lptr, order, idx, pnt, cinvp,
//     sources) is the Julia Int64 array, holding 1-based values;
//   - front f's residual columns are Rptr[f] : Rptr[f+1]-1 (nn of them), its
//     separator columns Stgt[Sptr[f] : Sptr[f+1]-1] (na of them), its diagonal
//     block Dval[Dptr[f] + (j-1) nn + (k-1)] (column-major nn × nn) and its
//     off-diagonal block Lval[Lptr[f] + ...] (L₂₁ is na × nn, U₁₂ is nn × na).
//
// Two variants of each sweep kernel:
//   variant 0 ("port"): a line-by-line port of the Julia kernel (same loop order,
//                       same loads and stores of C);
//   variant 1 ("reg"):  residual values kept in registers for fronts with nn ≤ 8
//                       (template on the exact nn, uniform per block), so each
//                       separator value C[t, Stgt[r]] is loaded once instead of nn
//                       times, 4 separator loads in flight; the ⊕-accumulation order
//                       is unchanged, so the results are bit-identical to variant 0.
//                       (The same kernel written in Julia runs as fast: cuda/NOTES.md.)
// Both read the factor and the index arrays through the read-only path (__ldg; no
// measurable difference with plain loads, -DSR_NO_LDG).
#include "semiring.cuh"

namespace sr {

template <class T> struct Bits;
template <> struct Bits<float> { using U = unsigned int; };
template <> struct Bits<double> { using U = unsigned long long; };

__device__ __forceinline__ unsigned int tobits(float x) { return __float_as_uint(x); }
__device__ __forceinline__ unsigned long long tobits(double x) { return (unsigned long long)__double_as_longlong(x); }
__device__ __forceinline__ float frombits(unsigned int x) { return __uint_as_float(x); }
__device__ __forceinline__ double frombits(unsigned long long x) { return __longlong_as_double((long long)x); }

// *p ← *p ⊕ m, atomically: a compare-and-swap loop around ⊕ (as atomic_splus! in sgetrs.jl)
template <class S, class T>
__device__ __forceinline__ void atomic_splus(T* p, T m) {
    using U = typename Bits<T>::U;
    T old = *p;
    while (true) {
        T nw = S::add(old, m);
        if (tobits(nw) == tobits(old)) return;
        U prev = atomicCAS(reinterpret_cast<U*>(p), tobits(old), tobits(nw));
        if (prev == tobits(old)) return;
        old = frombits(prev);
    }
}

// read-only loads of the factor and index arrays (-DSR_NO_LDG: plain loads, as the Julia kernels)
#ifdef SR_NO_LDG
template <class X> __device__ __forceinline__ X ro(const X* p) { return *p; }
#else
template <class X> __device__ __forceinline__ X ro(const X* p) { return __ldg(p); }
#endif

// ===== U sweep, one front, one row (upward_front! in sgetrs.jl) =====
//
//   C₁ ← C₁ U₁₁*          (forward substitution, scaled by the stars of the diagonal)
//   C₂ ← C₂ ⊕ C₁ U₁₂      (atomic when siblings may scatter into the same column)
template <class S, class T, bool SCALE, bool ATOMIC>
__device__ __forceinline__ void upward_front_port(T* __restrict__ C, int64_t ldc, int64_t t, int64_t f, const int64_t* Rptr,
                                                  const int64_t* Sptr, const int64_t* Stgt, const int64_t* Dptr,
                                                  const int64_t* Lptr, const T* Dval, const T* Lval) {
    const int64_t Rp = ro(Rptr + f - 1), nn = ro(Rptr + f) - Rp;
    const int64_t Sp = ro(Sptr + f - 1), na = ro(Sptr + f) - Sp;
    const T* D = Dval + ro(Dptr + f - 1) - 1;
    const T* L = Lval + ro(Lptr + f - 1) - 1;
    T* Ct = C + (t - 1);
#define CC(col) Ct[((col) - 1) * ldc]
    for (int64_t j = 0; j < nn; ++j) {
        T acc = CC(Rp + j);
        for (int64_t k = 0; k < j; ++k) acc = S::muladd(CC(Rp + k), ro(D + j * nn + k), acc);
        if (SCALE) acc = S::prod(acc, S::star(ro(D + j * nn + j)));
        CC(Rp + j) = acc;
    }
    for (int64_t r = 0; r < na; ++r) {
        T m = S::template zero<T>();
        for (int64_t j = 0; j < nn; ++j) m = S::muladd(CC(Rp + j), ro(L + r * nn + j), m);
        const int64_t c = ro(Stgt + Sp - 1 + r);
        if (ATOMIC) atomic_splus<S>(&CC(c), m);
        else CC(c) = S::add(CC(c), m);
    }
#undef CC
}

// the same with the nn ≤ NN residual values of the row in registers
template <class S, class T, bool SCALE, bool ATOMIC, int NN>
__device__ __forceinline__ void upward_front_reg(T* __restrict__ C, int64_t ldc, int64_t t, int64_t Rp, int64_t Sp, int64_t na,
                                                 const T* D, const T* L, const int64_t* Stgt) {
    T* Ct = C + (t - 1);
    T x[NN];
#pragma unroll
    for (int j = 0; j < NN; ++j) x[j] = Ct[(Rp + j - 1) * ldc];
#pragma unroll
    for (int j = 0; j < NN; ++j) {
        T acc = x[j];
#pragma unroll
        for (int k = 0; k < j; ++k) acc = S::muladd(x[k], ro(D + j * NN + k), acc);
        if (SCALE) acc = S::prod(acc, S::star(ro(D + j * NN + j)));
        x[j] = acc;
        Ct[(Rp + j - 1) * ldc] = acc;
    }
    for (int64_t r = 0; r < na; ++r) {
        T m = S::template zero<T>();
#pragma unroll
        for (int j = 0; j < NN; ++j) m = S::muladd(x[j], ro(L + r * NN + j), m);
        const int64_t c = ro(Stgt + Sp - 1 + r);
        T* p = Ct + (c - 1) * ldc;
        if (ATOMIC) atomic_splus<S>(p, m);
        else *p = S::add(*p, m);
    }
}

template <class S, class T, bool SCALE, bool ATOMIC>
__device__ __forceinline__ void upward_front_var1(T* __restrict__ C, int64_t ldc, int64_t t, int64_t f, const int64_t* Rptr,
                                                  const int64_t* Sptr, const int64_t* Stgt, const int64_t* Dptr,
                                                  const int64_t* Lptr, const T* Dval, const T* Lval) {
    const int64_t Rp = ro(Rptr + f - 1), nn = ro(Rptr + f) - Rp;
    const int64_t Sp = ro(Sptr + f - 1), na = ro(Sptr + f) - Sp;
    const T* D = Dval + ro(Dptr + f - 1) - 1;
    const T* L = Lval + ro(Lptr + f - 1) - 1;
    switch (nn) {
        case 1: upward_front_reg<S, T, SCALE, ATOMIC, 1>(C, ldc, t, Rp, Sp, na, D, L, Stgt); return;
        case 2: upward_front_reg<S, T, SCALE, ATOMIC, 2>(C, ldc, t, Rp, Sp, na, D, L, Stgt); return;
        case 3: upward_front_reg<S, T, SCALE, ATOMIC, 3>(C, ldc, t, Rp, Sp, na, D, L, Stgt); return;
        case 4: upward_front_reg<S, T, SCALE, ATOMIC, 4>(C, ldc, t, Rp, Sp, na, D, L, Stgt); return;
        case 5: upward_front_reg<S, T, SCALE, ATOMIC, 5>(C, ldc, t, Rp, Sp, na, D, L, Stgt); return;
        case 6: upward_front_reg<S, T, SCALE, ATOMIC, 6>(C, ldc, t, Rp, Sp, na, D, L, Stgt); return;
        case 7: upward_front_reg<S, T, SCALE, ATOMIC, 7>(C, ldc, t, Rp, Sp, na, D, L, Stgt); return;
        case 8: upward_front_reg<S, T, SCALE, ATOMIC, 8>(C, ldc, t, Rp, Sp, na, D, L, Stgt); return;
        default: upward_front_port<S, T, SCALE, ATOMIC>(C, ldc, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval); return;
    }
}

// upward_kernel!: one block per (front order[off + bx], chunk of rows)
template <class S, class T, int VAR>
__global__ void upward_kernel(int64_t nrows, T* __restrict__ C, int64_t ldc, const int64_t* __restrict__ order, int64_t off,
                              const int64_t* __restrict__ Rptr, const int64_t* __restrict__ Sptr, const int64_t* __restrict__ Stgt,
                              const int64_t* __restrict__ Dptr, const int64_t* __restrict__ Lptr, const T* __restrict__ Dval,
                              const T* __restrict__ Lval) {
    const int64_t t = threadIdx.x + 1 + (int64_t)blockIdx.y * blockDim.x;
    if (t > nrows) return;
    const int64_t f = ro(order + off + blockIdx.x);
    constexpr bool SCALE = !S::integral;
    if (VAR == 0) upward_front_port<S, T, SCALE, true>(C, ldc, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval);
    else upward_front_var1<S, T, SCALE, true>(C, ldc, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval);
}

// upward_path_kernel!: row t is e_v, v = cinvp[sources[t]]; walk the path of fronts from idx[v]
// to the top of the tree (istop) or the root; the thread owns its row (no atomics)
template <class S, class T, int VAR>
__global__ void upward_path_kernel(int64_t nrows, T* __restrict__ C, int64_t ldc, const int64_t* __restrict__ sources,
                                   const int64_t* __restrict__ cinvp, const int64_t* __restrict__ idx, const int64_t* __restrict__ pnt,
                                   const bool* __restrict__ istop, const int64_t* __restrict__ Rptr, const int64_t* __restrict__ Sptr,
                                   const int64_t* __restrict__ Stgt, const int64_t* __restrict__ Dptr, const int64_t* __restrict__ Lptr,
                                   const T* __restrict__ Dval, const T* __restrict__ Lval) {
    const int64_t t = threadIdx.x + 1 + (int64_t)blockIdx.x * blockDim.x;
    if (t > nrows) return;
    constexpr bool SCALE = !S::integral;
    const int64_t v = ro(cinvp + ro(sources + t - 1) - 1);
    C[(t - 1) + (v - 1) * ldc] = S::template one<T>();
    int64_t f = ro(idx + v - 1);
    while (f != 0 && !istop[f - 1]) {
        if (VAR == 0) upward_front_port<S, T, SCALE, false>(C, ldc, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval);
        else upward_front_var1<S, T, SCALE, false>(C, ldc, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval);
        f = ro(pnt + f - 1);
    }
}

// ===== L sweep (downward_kernel_simple! in sgetrs.jl) =====
//
//   C₁ ← C₁ ⊕ C₂ L₂₁
//   C₁ ← C₁ L₁₁*          (unit lower: backward substitution)
template <class S, class T>
__device__ __forceinline__ void downward_front_port(T* __restrict__ C, int64_t ldc, int64_t t, int64_t Rp, int64_t nn, int64_t Sp,
                                                    int64_t na, const T* D, const T* L, const int64_t* Stgt) {
    T* Ct = C + (t - 1);
#define CC(col) Ct[((col) - 1) * ldc]
    for (int64_t j = 0; j < nn; ++j) {
        T acc = CC(Rp + j);
        for (int64_t r = 0; r < na; ++r) acc = S::muladd(CC(ro(Stgt + Sp - 1 + r)), ro(L + j * na + r), acc);
        CC(Rp + j) = acc;
    }
    for (int64_t j = nn - 1; j >= 0; --j) {
        T acc = CC(Rp + j);
        for (int64_t k = j + 1; k < nn; ++k) acc = S::muladd(CC(Rp + k), ro(D + j * nn + k), acc);
        CC(Rp + j) = acc;
    }
#undef CC
}

template <class S, class T, int NN>
__device__ __forceinline__ void downward_front_reg(T* __restrict__ C, int64_t ldc, int64_t t, int64_t Rp, int64_t Sp, int64_t na,
                                                   const T* D, const T* L, const int64_t* Stgt) {
    T* Ct = C + (t - 1);
    T x[NN];
#pragma unroll
    for (int j = 0; j < NN; ++j) x[j] = Ct[(Rp + j - 1) * ldc];
    // separator loads issued 4 at a time (4 loads in flight per thread), then the updates
    // in the original order r = 0, 1, 2, ...
    const int64_t* St = Stgt + Sp - 1;
    int64_t r = 0;
    for (; r + 3 < na; r += 4) {
        const T c0 = Ct[(ro(St + r) - 1) * ldc];
        const T c1 = Ct[(ro(St + r + 1) - 1) * ldc];
        const T c2 = Ct[(ro(St + r + 2) - 1) * ldc];
        const T c3 = Ct[(ro(St + r + 3) - 1) * ldc];
#pragma unroll
        for (int j = 0; j < NN; ++j) {
            x[j] = S::muladd(c0, ro(L + j * na + r), x[j]);
            x[j] = S::muladd(c1, ro(L + j * na + r + 1), x[j]);
            x[j] = S::muladd(c2, ro(L + j * na + r + 2), x[j]);
            x[j] = S::muladd(c3, ro(L + j * na + r + 3), x[j]);
        }
    }
    for (; r < na; ++r) {
        const T c = Ct[(ro(St + r) - 1) * ldc];
#pragma unroll
        for (int j = 0; j < NN; ++j) x[j] = S::muladd(c, ro(L + j * na + r), x[j]);
    }
#pragma unroll
    for (int j = NN - 1; j >= 0; --j) {
#pragma unroll
        for (int k = j + 1; k < NN; ++k) x[j] = S::muladd(x[k], ro(D + j * NN + k), x[j]);
    }
#pragma unroll
    for (int j = 0; j < NN; ++j) Ct[(Rp + j - 1) * ldc] = x[j];
}

// row t through front f of the L sweep (variant 0: port, 1: registers for nn ≤ 8)
template <class S, class T, int VAR>
__device__ __forceinline__ void downward_front(T* __restrict__ C, int64_t ldc, int64_t t, int64_t f, const int64_t* Rptr,
                                               const int64_t* Sptr, const int64_t* Stgt, const int64_t* Dptr, const int64_t* Lptr,
                                               const T* Dval, const T* Lval) {
    const int64_t Rp = ro(Rptr + f - 1), nn = ro(Rptr + f) - Rp;
    const int64_t Sp = ro(Sptr + f - 1), na = ro(Sptr + f) - Sp;
    const T* D = Dval + ro(Dptr + f - 1) - 1;
    const T* L = Lval + ro(Lptr + f - 1) - 1;
    if (VAR == 1) {
        switch (nn) {
            case 1: downward_front_reg<S, T, 1>(C, ldc, t, Rp, Sp, na, D, L, Stgt); return;
            case 2: downward_front_reg<S, T, 2>(C, ldc, t, Rp, Sp, na, D, L, Stgt); return;
            case 3: downward_front_reg<S, T, 3>(C, ldc, t, Rp, Sp, na, D, L, Stgt); return;
            case 4: downward_front_reg<S, T, 4>(C, ldc, t, Rp, Sp, na, D, L, Stgt); return;
            case 5: downward_front_reg<S, T, 5>(C, ldc, t, Rp, Sp, na, D, L, Stgt); return;
            case 6: downward_front_reg<S, T, 6>(C, ldc, t, Rp, Sp, na, D, L, Stgt); return;
            case 7: downward_front_reg<S, T, 7>(C, ldc, t, Rp, Sp, na, D, L, Stgt); return;
            case 8: downward_front_reg<S, T, 8>(C, ldc, t, Rp, Sp, na, D, L, Stgt); return;
            default: break;
        }
    }
    downward_front_port<S, T>(C, ldc, t, Rp, nn, Sp, na, D, L, Stgt);
}

// downward_kernel_simple!: one block per (front order[off + bx], chunk of rows)
template <class S, class T, int VAR>
__global__ void downward_kernel(int64_t nrows, T* __restrict__ C, int64_t ldc, const int64_t* __restrict__ order, int64_t off,
                                const int64_t* __restrict__ Rptr, const int64_t* __restrict__ Sptr, const int64_t* __restrict__ Stgt,
                                const int64_t* __restrict__ Dptr, const int64_t* __restrict__ Lptr, const T* __restrict__ Dval,
                                const T* __restrict__ Lval) {
    const int64_t t = threadIdx.x + 1 + (int64_t)blockIdx.y * blockDim.x;
    if (t > nrows) return;
    const int64_t f = ro(order + off + blockIdx.x);
    downward_front<S, T, VAR>(C, ldc, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval);
}

// persistent_down_kernel! (sgetrs.jl): the L sweep below the top of the tree in one launch.
// A persistent grid takes tickets in order; ticket q is front rest[q / nchunk] on row chunk
// q % nchunk (fronts by depth, parents first). A block first waits until the parent of its
// front has finished all of its chunks (per-front counters), unless the parent is in the
// top (already done). Tickets are handed out in order, so every front waited on is held by
// a running block: no deadlock. C is read with plain (coherent) loads, never __ldg.
template <class S, class T, int VAR>
__global__ void persistent_down_kernel(int64_t nrows, T* __restrict__ C, int64_t ldc, const int64_t* __restrict__ rest, int64_t nrest,
                                       int64_t nchunk, const int64_t* __restrict__ pnt, const bool* __restrict__ istop,
                                       unsigned int* counters, unsigned int* ticket, const int64_t* __restrict__ Rptr,
                                       const int64_t* __restrict__ Sptr, const int64_t* __restrict__ Stgt, const int64_t* __restrict__ Dptr,
                                       const int64_t* __restrict__ Lptr, const T* __restrict__ Dval, const T* __restrict__ Lval) {
    __shared__ unsigned int qs;
    // 32-bit ticket arithmetic (64-bit division is emulated), as sgetrs.jl since 2026-10-01 23:35
#ifdef SR_PERSIST_I64
    using QT = int64_t;   // 64-bit ticket arithmetic (sgetrs.jl before 23:35)
#else
    using QT = unsigned int;
#endif
    const QT nitems32 = (QT)(nrest * nchunk);
    const QT nchunk32 = (QT)nchunk;
    const int tid = threadIdx.x;
    while (true) {
        if (tid == 0) qs = atomicAdd(ticket, 1u);
        __syncthreads();
        const QT q = qs;
        __syncthreads();
        if (q >= nitems32) break;
        const int64_t f = ro(rest + q / nchunk32);
        const int64_t c = q % nchunk32;
        const int64_t p = ro(pnt + f - 1);
        if (tid == 0 && p != 0 && !istop[p - 1]) {
            while (atomicAdd(counters + p - 1, 0u) < (unsigned int)nchunk) {
            }
            __threadfence();
        }
        __syncthreads();
        const int64_t t = c * blockDim.x + tid + 1;
        if (t <= nrows) downward_front<S, T, VAR>(C, ldc, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval);
        __threadfence();
        __syncthreads();
        if (tid == 0) atomicAdd(counters + f - 1, 1u);
    }
}

// upward_path_warp_kernel! (sgetrs.jl): one warp per source row. Variant 0: lane 0 solves with
// U₁₁, then the lanes split the separator update. Variant 1: every lane solves with U₁₁ in
// registers (redundantly, nn ≤ 8; lane 0 stores), so the separator update uses the
// register values. Same operations, same order.
template <class S, class T, int VAR>
__global__ void upward_path_warp_kernel(int64_t nrows, T* __restrict__ C, int64_t ldc, const int64_t* __restrict__ sources,
                                        const int64_t* __restrict__ cinvp, const int64_t* __restrict__ idx, const int64_t* __restrict__ pnt,
                                        const bool* __restrict__ istop, const int64_t* __restrict__ Rptr, const int64_t* __restrict__ Sptr,
                                        const int64_t* __restrict__ Stgt, const int64_t* __restrict__ Dptr, const int64_t* __restrict__ Lptr,
                                        const T* __restrict__ Dval, const T* __restrict__ Lval) {
    constexpr bool SCALE = !S::integral;
    const int lane = threadIdx.x % 32;
    const int64_t t = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32 + 1;
    if (t > nrows) return;
    T* Ct = C + (t - 1);
    const int64_t v = ro(cinvp + ro(sources + t - 1) - 1);
    if (lane == 0) Ct[(v - 1) * ldc] = S::template one<T>();
    __syncwarp();
    __threadfence_block();
    int64_t f = ro(idx + v - 1);
    while (f != 0 && !istop[f - 1]) {
        const int64_t Rp = ro(Rptr + f - 1), nn = ro(Rptr + f) - Rp;
        const int64_t Sp = ro(Sptr + f - 1), na = ro(Sptr + f) - Sp;
        const T* D = Dval + ro(Dptr + f - 1) - 1;
        const T* L = Lval + ro(Lptr + f - 1) - 1;
        bool done = false;
        if (VAR == 1 && nn <= 8) {
#define WARP_CASE(NN)                                                                                       \
    case NN: {                                                                                              \
        T x[NN];                                                                                            \
        _Pragma("unroll") for (int j = 0; j < NN; ++j) x[j] = Ct[(Rp + j - 1) * ldc];                       \
        _Pragma("unroll") for (int j = 0; j < NN; ++j) {                                                    \
            T acc = x[j];                                                                                   \
            _Pragma("unroll") for (int k = 0; k < j; ++k) acc = S::muladd(x[k], ro(D + j * NN + k), acc);   \
            if (SCALE) acc = S::prod(acc, S::star(ro(D + j * NN + j)));                                     \
            x[j] = acc;                                                                                     \
        }                                                                                                   \
        __syncwarp();                                                                                       \
        if (lane == 0) {                                                                                    \
            _Pragma("unroll") for (int j = 0; j < NN; ++j) Ct[(Rp + j - 1) * ldc] = x[j];                   \
        }                                                                                                   \
        for (int64_t r = lane; r < na; r += 32) {                                                           \
            T m = S::template zero<T>();                                                                    \
            _Pragma("unroll") for (int j = 0; j < NN; ++j) m = S::muladd(x[j], ro(L + r * NN + j), m);      \
            T* p = Ct + (ro(Stgt + Sp - 1 + r) - 1) * ldc;                                                  \
            *p = S::add(*p, m);                                                                             \
        }                                                                                                   \
        done = true;                                                                                        \
    } break;
            switch (nn) { WARP_CASE(1) WARP_CASE(2) WARP_CASE(3) WARP_CASE(4) WARP_CASE(5) WARP_CASE(6) WARP_CASE(7) WARP_CASE(8) default: break; }
#undef WARP_CASE
        }
        if (!done) {
            if (lane == 0) {
                for (int64_t j = 0; j < nn; ++j) {
                    T acc = Ct[(Rp + j - 1) * ldc];
                    for (int64_t k = 0; k < j; ++k) acc = S::muladd(Ct[(Rp + k - 1) * ldc], ro(D + j * nn + k), acc);
                    if (SCALE) acc = S::prod(acc, S::star(ro(D + j * nn + j)));
                    Ct[(Rp + j - 1) * ldc] = acc;
                }
            }
            __syncwarp();
            __threadfence_block();
            for (int64_t r = lane; r < na; r += 32) {
                T m = S::template zero<T>();
                for (int64_t j = 0; j < nn; ++j) m = S::muladd(Ct[(Rp + j - 1) * ldc], ro(L + r * nn + j), m);
                T* p = Ct + (ro(Stgt + Sp - 1 + r) - 1) * ldc;
                *p = S::add(*p, m);
            }
        }
        __syncwarp();
        __threadfence_block();
        f = ro(pnt + f - 1);
    }
}

// ===== dense-path helpers =====

// strsx_diag_kernel!: X ← X A* with one thread per row (A is b × b)
//   uplo = 'U': forward,  X[:, j] ← (X[:, j] ⊕ Σ_{k<j} X[:, k] A[k, j]) A[j, j]*   (scaled if SCALE)
//   uplo = 'L': backward, X[:, j] ←  X[:, j] ⊕ Σ_{k>j} X[:, k] A[k, j]             (unit)
template <class S, class T, bool SCALE, bool UPPER>
__global__ void strsx_diag_kernel(int64_t m, int64_t b, T* __restrict__ X, int64_t ldx, const T* __restrict__ A, int64_t lda) {
    const int64_t t = threadIdx.x + (int64_t)blockIdx.x * blockDim.x;
    if (t >= m) return;
    T* Xt = X + t;
    if (UPPER) {
        for (int64_t j = 0; j < b; ++j) {
            T acc = Xt[j * ldx];
            for (int64_t k = 0; k < j; ++k) acc = S::muladd(Xt[k * ldx], ro(A + k + j * lda), acc);
            if (SCALE) acc = S::prod(acc, S::star(ro(A + j + j * lda)));
            Xt[j * ldx] = acc;
        }
    } else {
        for (int64_t j = b - 1; j >= 0; --j) {
            T acc = Xt[j * ldx];
            for (int64_t k = j + 1; k < b; ++k) acc = S::muladd(Xt[k * ldx], ro(A + k + j * lda), acc);
            Xt[j * ldx] = acc;
        }
    }
}

template <class T>
__global__ void fill_kernel(int64_t m, int64_t len, T* __restrict__ X, int64_t ldx, T v) {
    const int64_t e = threadIdx.x + (int64_t)blockIdx.x * blockDim.x;
    if (e < len) X[(e % m) + (e / m) * ldx] = v;
}

template <class T>
__global__ void fill_contig_kernel(int64_t len, T* __restrict__ X, T v) {
    int64_t e = threadIdx.x + (int64_t)blockIdx.x * blockDim.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; e < len; e += stride) X[e] = v;
}

template <class T>
__global__ void copy_kernel(int64_t m, int64_t len, T* __restrict__ X, int64_t ldx, const T* __restrict__ Y, int64_t ldy) {
    const int64_t e = threadIdx.x + (int64_t)blockIdx.x * blockDim.x;
    if (e < len) {
        const int64_t i = e % m, j = e / m;
        X[i + j * ldx] = Y[i + j * ldy];
    }
}

template <class S, class T>
__global__ void identity_kernel(int64_t m, int64_t len, T* __restrict__ X, int64_t ldx) {
    const int64_t e = threadIdx.x + (int64_t)blockIdx.x * blockDim.x;
    if (e < len) {
        const int64_t i = e % m, j = e / m;
        X[i + j * ldx] = i == j ? S::template one<T>() : S::template zero<T>();
    }
}

// M[:, r] ← C[:, Stgt[Sp + r - 1]]   (one block column per r)
template <class T>
__global__ void gather_kernel(int64_t m, T* __restrict__ M, int64_t ldm, const T* __restrict__ C, int64_t ldc,
                              const int64_t* __restrict__ Stgt, int64_t Sp) {
    const int64_t t = threadIdx.x + (int64_t)blockIdx.y * blockDim.x;
    const int64_t r = blockIdx.x;
    if (t < m) M[t + r * ldm] = C[t + (ro(Stgt + Sp - 1 + r) - 1) * ldc];
}

// C[:, Stgt[Sp + r - 1]] ← C[:, Stgt[Sp + r - 1]] ⊕ M[:, r]
template <class S, class T>
__global__ void scatteradd_kernel(int64_t m, T* __restrict__ C, int64_t ldc, const T* __restrict__ M, int64_t ldm,
                                  const int64_t* __restrict__ Stgt, int64_t Sp) {
    const int64_t t = threadIdx.x + (int64_t)blockIdx.y * blockDim.x;
    const int64_t r = blockIdx.x;
    if (t < m) {
        T* p = C + t + (ro(Stgt + Sp - 1 + r) - 1) * ldc;
        *p = S::add(*p, M[t + r * ldm]);
    }
}

static inline int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
static inline int err() { return (int)cudaGetLastError(); }

template <class S, class T>
int upward_batched(int var, int64_t nrows, void* C, int64_t ldc, const int64_t* order, int64_t off, int64_t nfl, const int64_t* Rptr,
                   const int64_t* Sptr, const int64_t* Stgt, const int64_t* Dptr, const int64_t* Lptr, const void* Dval,
                   const void* Lval, int tb, cudaStream_t st) {
    if (nfl <= 0 || nrows <= 0) return 0;
    dim3 grid((unsigned)nfl, (unsigned)cdiv(nrows, tb));
    auto k = var == 0 ? upward_kernel<S, T, 0> : upward_kernel<S, T, 1>;
    k<<<grid, tb, 0, st>>>(nrows, (T*)C, ldc, order, off, Rptr, Sptr, Stgt, Dptr, Lptr, (const T*)Dval, (const T*)Lval);
    return err();
}

template <class S, class T>
int downward_batched(int var, int64_t nrows, void* C, int64_t ldc, const int64_t* order, int64_t off, int64_t nfl, const int64_t* Rptr,
                     const int64_t* Sptr, const int64_t* Stgt, const int64_t* Dptr, const int64_t* Lptr, const void* Dval,
                     const void* Lval, int tb, cudaStream_t st) {
    if (nfl <= 0 || nrows <= 0) return 0;
    dim3 grid((unsigned)nfl, (unsigned)cdiv(nrows, tb));
    auto k = var == 0 ? downward_kernel<S, T, 0> : downward_kernel<S, T, 1>;
    k<<<grid, tb, 0, st>>>(nrows, (T*)C, ldc, order, off, Rptr, Sptr, Stgt, Dptr, Lptr, (const T*)Dval, (const T*)Lval);
    return err();
}

template <class S, class T>
int upward_path(int var, int64_t nrows, void* C, int64_t ldc, const int64_t* sources, const int64_t* cinvp, const int64_t* idx,
                const int64_t* pnt, const bool* istop, const int64_t* Rptr, const int64_t* Sptr, const int64_t* Stgt,
                const int64_t* Dptr, const int64_t* Lptr, const void* Dval, const void* Lval, int tb, cudaStream_t st) {
    if (nrows <= 0) return 0;
    auto k = var == 0 ? upward_path_kernel<S, T, 0> : upward_path_kernel<S, T, 1>;
    k<<<(unsigned)cdiv(nrows, tb), tb, 0, st>>>(nrows, (T*)C, ldc, sources, cinvp, idx, pnt, istop, Rptr, Sptr, Stgt, Dptr, Lptr,
                                               (const T*)Dval, (const T*)Lval);
    return err();
}

template <class S, class T>
int persistent_down(int var, int64_t nrows, void* C, int64_t ldc, const int64_t* rest, int64_t nrest, int tb, int nblk, const int64_t* pnt,
                    const bool* istop, unsigned int* counters, int64_t nf, unsigned int* ticket, const int64_t* Rptr, const int64_t* Sptr,
                    const int64_t* Stgt, const int64_t* Dptr, const int64_t* Lptr, const void* Dval, const void* Lval, cudaStream_t st) {
    if (nrest <= 0 || nrows <= 0) return 0;
    const int64_t nchunk = cdiv(nrows, tb);
    cudaMemsetAsync(counters, 0, sizeof(unsigned int) * nf, st);
    cudaMemsetAsync(ticket, 0, sizeof(unsigned int), st);
    auto k = var == 0 ? persistent_down_kernel<S, T, 0> : persistent_down_kernel<S, T, 1>;
    k<<<(unsigned)nblk, tb, 0, st>>>(nrows, (T*)C, ldc, rest, nrest, nchunk, pnt, istop, counters, ticket, Rptr, Sptr, Stgt, Dptr, Lptr,
                                     (const T*)Dval, (const T*)Lval);
    return err();
}

template <class S, class T>
int upward_path_warp(int var, int64_t nrows, void* C, int64_t ldc, const int64_t* sources, const int64_t* cinvp, const int64_t* idx,
                     const int64_t* pnt, const bool* istop, const int64_t* Rptr, const int64_t* Sptr, const int64_t* Stgt,
                     const int64_t* Dptr, const int64_t* Lptr, const void* Dval, const void* Lval, int tb, cudaStream_t st) {
    if (nrows <= 0) return 0;
    auto k = var == 0 ? upward_path_warp_kernel<S, T, 0> : upward_path_warp_kernel<S, T, 1>;
    k<<<(unsigned)cdiv(32 * nrows, tb), tb, 0, st>>>(nrows, (T*)C, ldc, sources, cinvp, idx, pnt, istop, Rptr, Sptr, Stgt, Dptr, Lptr,
                                                     (const T*)Dval, (const T*)Lval);
    return err();
}

template <class S, class T>
int strsx_diag(int upper, int64_t m, int64_t b, void* X, int64_t ldx, const void* A, int64_t lda, int threads, cudaStream_t st) {
    if (m <= 0 || b <= 0) return 0;
    constexpr bool SCALE = !S::integral;
    const int tb = threads > 0 ? threads : (int)(cdiv(m, 32) * 32 < 128 ? cdiv(m, 32) * 32 : 128);
    if (upper) strsx_diag_kernel<S, T, SCALE, true><<<(unsigned)cdiv(m, tb), tb, 0, st>>>(m, b, (T*)X, ldx, (const T*)A, lda);
    else strsx_diag_kernel<S, T, false, false><<<(unsigned)cdiv(m, tb), tb, 0, st>>>(m, b, (T*)X, ldx, (const T*)A, lda);
    return err();
}

template <class S, class T>
int identity(int64_t m, int64_t n, void* X, int64_t ldx, cudaStream_t st) {
    const int64_t len = m * n;
    if (len <= 0) return 0;
    identity_kernel<S, T><<<(unsigned)cdiv(len, 256), 256, 0, st>>>(m, len, (T*)X, ldx);
    return err();
}

template <class S, class T>
int scatteradd(int64_t m, int64_t ncol, void* C, int64_t ldc, const void* M, int64_t ldm, const int64_t* Stgt, int64_t Sp,
               cudaStream_t st) {
    if (m <= 0 || ncol <= 0) return 0;
    const int tb = (int)(cdiv(m, 32) * 32 < 256 ? cdiv(m, 32) * 32 : 256);
    dim3 grid((unsigned)ncol, (unsigned)cdiv(m, tb));
    scatteradd_kernel<S, T><<<grid, tb, 0, st>>>(m, (T*)C, ldc, (const T*)M, ldm, Stgt, Sp);
    return err();
}

template <class T>
int fill(int64_t m, int64_t n, void* X, int64_t ldx, double v, cudaStream_t st) {
    const int64_t len = m * n;
    if (len <= 0) return 0;
    if (ldx == m) {
        const int64_t nb = cdiv(len, 256);
        fill_contig_kernel<T><<<(unsigned)(nb < 4096 ? nb : 4096), 256, 0, st>>>(len, (T*)X, (T)v);
    } else {
        fill_kernel<T><<<(unsigned)cdiv(len, 256), 256, 0, st>>>(m, len, (T*)X, ldx, (T)v);
    }
    return err();
}

template <class T>
int copy(int64_t m, int64_t n, void* X, int64_t ldx, const void* Y, int64_t ldy, cudaStream_t st) {
    const int64_t len = m * n;
    if (len <= 0) return 0;
    copy_kernel<T><<<(unsigned)cdiv(len, 256), 256, 0, st>>>(m, len, (T*)X, ldx, (const T*)Y, ldy);
    return err();
}

template <class T>
int gather(int64_t m, int64_t ncol, void* M, int64_t ldm, const void* C, int64_t ldc, const int64_t* Stgt, int64_t Sp, cudaStream_t st) {
    if (m <= 0 || ncol <= 0) return 0;
    const int tb = (int)(cdiv(m, 32) * 32 < 256 ? cdiv(m, 32) * 32 : 256);
    dim3 grid((unsigned)ncol, (unsigned)cdiv(m, tb));
    gather_kernel<T><<<grid, tb, 0, st>>>(m, (T*)M, ldm, (const T*)C, ldc, Stgt, Sp);
    return err();
}

}  // namespace sr

#define DT_DISPATCH(dt_code, CALL)                                        \
    do {                                                                  \
        if ((dt_code) == sr::DT_F32) { using T = float; return CALL; }    \
        if ((dt_code) == sr::DT_F64) { using T = double; return CALL; }   \
        return -1;                                                        \
    } while (0)

extern "C" {

int sr_upward_batched(int sr_code, int dt, int var, int64_t nrows, void* C, int64_t ldc, const int64_t* order, int64_t off, int64_t nfl,
                      const int64_t* Rptr, const int64_t* Sptr, const int64_t* Stgt, const int64_t* Dptr, const int64_t* Lptr,
                      const void* Dval, const void* Lval, int tb, cudaStream_t st) {
    SR_DISPATCH(sr_code, dt, (sr::upward_batched<S, T>(var, nrows, C, ldc, order, off, nfl, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, tb, st)));
}

int sr_downward_batched(int sr_code, int dt, int var, int64_t nrows, void* C, int64_t ldc, const int64_t* order, int64_t off, int64_t nfl,
                        const int64_t* Rptr, const int64_t* Sptr, const int64_t* Stgt, const int64_t* Dptr, const int64_t* Lptr,
                        const void* Dval, const void* Lval, int tb, cudaStream_t st) {
    SR_DISPATCH(sr_code, dt, (sr::downward_batched<S, T>(var, nrows, C, ldc, order, off, nfl, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, tb, st)));
}

int sr_upward_path(int sr_code, int dt, int var, int64_t nrows, void* C, int64_t ldc, const int64_t* sources, const int64_t* cinvp,
                   const int64_t* idx, const int64_t* pnt, const bool* istop, const int64_t* Rptr, const int64_t* Sptr,
                   const int64_t* Stgt, const int64_t* Dptr, const int64_t* Lptr, const void* Dval, const void* Lval, int tb,
                   cudaStream_t st) {
    SR_DISPATCH(sr_code, dt, (sr::upward_path<S, T>(var, nrows, C, ldc, sources, cinvp, idx, pnt, istop, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, tb, st)));
}

int sr_persistent_down(int sr_code, int dt, int var, int64_t nrows, void* C, int64_t ldc, const int64_t* rest, int64_t nrest, int tb,
                       int nblk, const int64_t* pnt, const bool* istop, unsigned int* counters, int64_t nf, unsigned int* ticket,
                       const int64_t* Rptr, const int64_t* Sptr, const int64_t* Stgt, const int64_t* Dptr, const int64_t* Lptr,
                       const void* Dval, const void* Lval, cudaStream_t st) {
    SR_DISPATCH(sr_code, dt, (sr::persistent_down<S, T>(var, nrows, C, ldc, rest, nrest, tb, nblk, pnt, istop, counters, nf, ticket, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, st)));
}

int sr_upward_path_warp(int sr_code, int dt, int var, int64_t nrows, void* C, int64_t ldc, const int64_t* sources, const int64_t* cinvp,
                        const int64_t* idx, const int64_t* pnt, const bool* istop, const int64_t* Rptr, const int64_t* Sptr,
                        const int64_t* Stgt, const int64_t* Dptr, const int64_t* Lptr, const void* Dval, const void* Lval, int tb,
                        cudaStream_t st) {
    SR_DISPATCH(sr_code, dt, (sr::upward_path_warp<S, T>(var, nrows, C, ldc, sources, cinvp, idx, pnt, istop, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, tb, st)));
}

int sr_strsx_diag(int sr_code, int dt, int upper, int64_t m, int64_t b, void* X, int64_t ldx, const void* A, int64_t lda, int threads,
                  cudaStream_t st) {
    SR_DISPATCH(sr_code, dt, (sr::strsx_diag<S, T>(upper, m, b, X, ldx, A, lda, threads, st)));
}

int sr_identity(int sr_code, int dt, int64_t m, int64_t n, void* X, int64_t ldx, cudaStream_t st) {
    SR_DISPATCH(sr_code, dt, (sr::identity<S, T>(m, n, X, ldx, st)));
}

int sr_scatteradd(int sr_code, int dt, int64_t m, int64_t ncol, void* C, int64_t ldc, const void* M, int64_t ldm, const int64_t* Stgt,
                  int64_t Sp, cudaStream_t st) {
    SR_DISPATCH(sr_code, dt, (sr::scatteradd<S, T>(m, ncol, C, ldc, M, ldm, Stgt, Sp, st)));
}

int sr_fill(int dt, int64_t m, int64_t n, void* X, int64_t ldx, double v, cudaStream_t st) {
    DT_DISPATCH(dt, (sr::fill<T>(m, n, X, ldx, v, st)));
}

int sr_copy(int dt, int64_t m, int64_t n, void* X, int64_t ldx, const void* Y, int64_t ldy, cudaStream_t st) {
    DT_DISPATCH(dt, (sr::copy<T>(m, n, X, ldx, Y, ldy, st)));
}

int sr_gather(int dt, int64_t m, int64_t ncol, void* M, int64_t ldm, const void* C, int64_t ldc, const int64_t* Stgt, int64_t Sp,
              cudaStream_t st) {
    DT_DISPATCH(dt, (sr::gather<T>(m, ncol, M, ldm, C, ldc, Stgt, Sp, st)));
}

}
