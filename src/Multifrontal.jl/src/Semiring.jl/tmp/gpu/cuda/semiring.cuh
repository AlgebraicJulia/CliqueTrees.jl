// Semirings for the CUDA C++ backend of SemiringGPU.
//
// Each struct reproduces, bit for bit, what the Julia kernels compute with
// CliqueTrees.Multifrontal.Semiring on the GPU (the X86 host branch of
// Semiring.jl, plus the CUDA.@device_override of vmin/vmax → llvm.minnum /
// llvm.maxnum in src/SemiringGPU.jl):
//
//   MinPlus   a ⊕ b = minnum(a, b)            a ⊗ b = a + b
//             smuladd(a, b, c) = minnum(a + b, c)     (NaN from +∞ + -∞ is absorbed)
//             sprod(a, b)      = isnan(a + b) ? +∞ : a + b
//             sstar(a)         = a >= 0 ? 0 : -∞,  zero = +∞, one = 0,  not integral
//   MaxMin    a ⊕ b = maxnum(a, b)            a ⊗ b = minnum(a, b)
//             smuladd(a, b, c) = maxnum(minnum(a, b), c)
//             sstar(a)         = one,  zero = -∞, one = +∞,  integral
//   PlusProd  a ⊕ b = a + b                   a ⊗ b = a * b
//             smuladd(a, b, c) = fma(a, b, c)  (Julia's muladd contracts to fma.rn)
//             sstar(a)         = a < 1 ? 1 / (1 - a) : +∞,  zero = 0, one = 1,  not integral
//
// fminf/fmaxf are PTX min.f32/max.f32, exactly what llvm.minnum/maxnum lower to.
// Build without --use_fast_math (it would turn 1/x into an approximation and
// flush denormals); --fmad=false so that no a*b+c is contracted behind our
// back (fma is always explicit).
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace sr {

__device__ __forceinline__ float vmin(float a, float b) { return fminf(a, b); }
__device__ __forceinline__ double vmin(double a, double b) { return fmin(a, b); }
__device__ __forceinline__ float vmax(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ double vmax(double a, double b) { return fmax(a, b); }
__device__ __forceinline__ float vfma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
__device__ __forceinline__ double vfma(double a, double b, double c) { return __fma_rn(a, b, c); }

template <class T> __device__ __forceinline__ T inf();
template <> __device__ __forceinline__ float inf<float>() { return __int_as_float(0x7f800000); }
template <> __device__ __forceinline__ double inf<double>() { return __longlong_as_double(0x7ff0000000000000LL); }

struct MinPlus {
    static constexpr bool integral = false;
    template <class T> __device__ __forceinline__ static T zero() { return inf<T>(); }
    template <class T> __device__ __forceinline__ static T one() { return T(0); }
    template <class T> __device__ __forceinline__ static T add(T a, T b) { return vmin(a, b); }
    template <class T> __device__ __forceinline__ static T muladd(T a, T b, T c) { return vmin(a + b, c); }
    template <class T> __device__ __forceinline__ static T prod(T a, T b) {
        T c = a + b;
        return c != c ? zero<T>() : c;
    }
    template <class T> __device__ __forceinline__ static T star(T a) { return a >= T(0) ? one<T>() : -inf<T>(); }
};

struct MaxMin {
    static constexpr bool integral = true;
    template <class T> __device__ __forceinline__ static T zero() { return -inf<T>(); }
    template <class T> __device__ __forceinline__ static T one() { return inf<T>(); }
    template <class T> __device__ __forceinline__ static T add(T a, T b) { return vmax(a, b); }
    template <class T> __device__ __forceinline__ static T muladd(T a, T b, T c) { return vmax(vmin(a, b), c); }
    template <class T> __device__ __forceinline__ static T prod(T a, T b) { return vmin(a, b); }
    template <class T> __device__ __forceinline__ static T star(T) { return one<T>(); }
};

struct PlusProd {
    static constexpr bool integral = false;
    template <class T> __device__ __forceinline__ static T zero() { return T(0); }
    template <class T> __device__ __forceinline__ static T one() { return T(1); }
    template <class T> __device__ __forceinline__ static T add(T a, T b) { return a + b; }
    template <class T> __device__ __forceinline__ static T muladd(T a, T b, T c) { return vfma(a, b, c); }
    template <class T> __device__ __forceinline__ static T prod(T a, T b) { return a * b; }
    template <class T> __device__ __forceinline__ static T star(T a) { return a < T(1) ? T(1) / (T(1) - a) : inf<T>(); }
};

// semiring and element-type codes used by the extern "C" entry points
enum { SR_MINPLUS = 0, SR_MAXMIN = 1, SR_PLUSPROD = 2 };
enum { DT_F32 = 0, DT_F64 = 1 };

}  // namespace sr

// dispatch a templated call over (semiring code, dtype code); evaluates to an int error code
#define SR_DISPATCH(sr_code, dt_code, CALL)                                                     \
    do {                                                                                        \
        if ((sr_code) == sr::SR_MINPLUS && (dt_code) == sr::DT_F32) { using S = sr::MinPlus; using T = float; return CALL; }   \
        if ((sr_code) == sr::SR_MINPLUS && (dt_code) == sr::DT_F64) { using S = sr::MinPlus; using T = double; return CALL; }  \
        if ((sr_code) == sr::SR_MAXMIN && (dt_code) == sr::DT_F32) { using S = sr::MaxMin; using T = float; return CALL; }     \
        if ((sr_code) == sr::SR_MAXMIN && (dt_code) == sr::DT_F64) { using S = sr::MaxMin; using T = double; return CALL; }    \
        if ((sr_code) == sr::SR_PLUSPROD && (dt_code) == sr::DT_F32) { using S = sr::PlusProd; using T = float; return CALL; } \
        if ((sr_code) == sr::SR_PLUSPROD && (dt_code) == sr::DT_F64) { using S = sr::PlusProd; using T = double; return CALL; }\
        return -1;                                                                              \
    } while (0)
