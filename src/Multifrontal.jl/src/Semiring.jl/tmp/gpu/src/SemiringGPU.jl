module SemiringGPU

# GPU kernels for CliqueTrees.Multifrontal.Semiring.
#
# sgemx_gpu!(s, C, A, B) computes C ← C ⊕ A ⊗ B in the semiring s, with
# the same semantics as the CPU kernel Semiring.sgemx!. The kernel is
# generic: it only calls smuladd, szero, and sone, so every semiring
# whose operations compile for the GPU works unchanged.

using CUDA
using CliqueTrees
using SparseArrays: SparseMatrixCSC

const Semiring = CliqueTrees.Multifrontal.Semiring

using .Semiring: AbstractSemiring, smuladd, splus, szero, sone

export sgemx_gpu!, GPUSLU, rmul_gpu!, sssp_gpu!, SSSPPlan, sgetrf_gpu!, mlu_gpu, FactorPlan, factorize!, closure_gpu!, closure_gpu

# ===== tiling =====
#
# Each thread block computes a BM × BN tile of C, streaming BK-deep
# panels of A and B through shared memory. Each of its TX × TY threads
# keeps a TM × TN register tile of C; the rows owned by thread (tx, ty)
# are tx, tx + TX, ..., and its columns are ty, ty + TY, ..., so that a
# warp reads consecutive shared-memory words (no bank conflicts) or
# one broadcast word.

struct Tiling{BM, BN, BK, TM, TN} end

const TILING_LARGE = Tiling{128, 128, 8, 8, 8}()   # 256 threads
const TILING_SMALL = Tiling{64, 64, 8, 4, 4}()     # 256 threads
const TILING_N16 = Tiling{256, 16, 8, 16, 1}()     # 256 threads, skinny outputs (n ≤ 16)
const TILING_N32 = Tiling{128, 32, 8, 8, 2}()      # 256 threads, skinny outputs (n ≤ 32)

function sgemx_gpu!(s::AbstractSemiring, C::AbstractMatrix{V}, A::AbstractMatrix{V}, B::AbstractMatrix{V};
        tiling::Tiling = choose_tiling(size(C, 1), size(C, 2))) where {V}
    @assert size(C, 1) == size(A, 1)
    @assert size(C, 2) == size(B, 2)
    @assert size(A, 2) == size(B, 1)

    m = size(C, 1)
    n = size(C, 2)
    k = size(A, 2)

    if m > 0 && n > 0 && k > 0
        launch!(s, C, A, B, tiling)
    end

    return C
end

# Small outputs (e.g. most frontal updates) cannot fill the GPU with
# 128 × 128 tiles.
const NSM = Ref(0)

function choose_tiling(m::Integer, n::Integer)
    if iszero(NSM[])
        NSM[] = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    end

    if n <= 16
        return TILING_N16
    elseif n <= 32
        return TILING_N32
    elseif cld(m, 128) * cld(n, 128) >= 2 * NSM[]
        return TILING_LARGE
    else
        return TILING_SMALL
    end
end

function launch!(s::AbstractSemiring, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, ::Tiling{BM, BN, BK, TM, TN}) where {BM, BN, BK, TM, TN}
    TX = BM ÷ TM
    TY = BN ÷ TN
    blocks = (cld(size(C, 1), BM), cld(size(C, 2), BN))
    @cuda threads = TX * TY blocks = blocks sgemx_kernel!(s, C, A, B, Val(BM), Val(BN), Val(BK), Val(TM), Val(TN))
    return
end

# ===== kernel =====

function sgemx_kernel!(s::AbstractSemiring, C::AbstractMatrix{V}, A::AbstractMatrix{V}, B::AbstractMatrix{V},
        ::Val{BM}, ::Val{BN}, ::Val{BK}, ::Val{TM}, ::Val{TN}) where {V, BM, BN, BK, TM, TN}
    TX = BM ÷ TM
    TY = BN ÷ TN
    NT = TX * TY

    m = size(C, 1)
    n = size(C, 2)
    k = size(A, 2)

    # As[i, p] = A[i0 + i, k0 + p]; Bs[p, j] = B[k0 + p, j0 + j]
    As = CuStaticSharedArray(V, (BM, BK))
    Bs = CuStaticSharedArray(V, (BK, BN))

    t  = threadIdx().x - 1
    tx = t % TX
    ty = t ÷ TX
    i0 = (blockIdx().x - 1) * BM
    j0 = (blockIdx().y - 1) * BN

    # Out-of-range entries of A are padded with 0 and those of B with 1:
    # 0 ⊗ 1 = 0 is exact (no overflow for integer tropical types) and is
    # absorbed by ⊕.
    z = szero(s, V, Val(:N))
    u = sone(s, V, Val(:N))

    acc = ntuple(_ -> z, Val(TM * TN))

    @inbounds for k0 in 0:BK:(k - 1)
        # global → shared, coalesced along the columns of A and B
        e = t

        while e < BM * BK
            i = e % BM
            p = e ÷ BM
            gi = i0 + i + 1
            gp = k0 + p + 1
            As[i + 1, p + 1] = (gi <= m && gp <= k) ? A[gi, gp] : z
            e += NT
        end

        e = t

        while e < BK * BN
            p = e % BK
            j = e ÷ BK
            gp = k0 + p + 1
            gj = j0 + j + 1
            Bs[p + 1, j + 1] = (gp <= k && gj <= n) ? B[gp, gj] : u
            e += NT
        end

        sync_threads()

        for p in 1:BK
            a = ntuple(r -> (@inbounds As[tx + (r - 1) * TX + 1, p]), Val(TM))
            b = ntuple(c -> (@inbounds Bs[p, ty + (c - 1) * TY + 1]), Val(TN))

            acc = rank1_update(s, acc, a, b)
        end

        sync_threads()
    end

    store_tile!(s, C, acc, i0 + tx + 1, j0 + ty + 1, Val(TX), Val(TY), Val(TM))
    return
end

# C[i, j] ← acc ⊕ C[i, j] for the rows i = i1, i1 + TX, ... and columns
# j = j1, j1 + TY, ... of one thread. Unrolled, so that acc is indexed
# by constants and stays in registers.
@generated function store_tile!(s::AbstractSemiring, C::AbstractMatrix, acc::NTuple{N}, i1::Int, j1::Int,
        ::Val{TX}, ::Val{TY}, ::Val{TM}) where {N, TX, TY, TM}
    stores = Expr[]

    for e in 1:N
        r = (e - 1) % TM
        c = (e - 1) ÷ TM

        push!(stores, quote
            gi = i1 + $(r * TX)
            gj = j1 + $(c * TY)

            if gi <= m && gj <= n
                C[gi, gj] = splus(s, acc[$e], C[gi, gj], Val(:N))
            end
        end)
    end

    return quote
        $(Expr(:meta, :inline))
        m = size(C, 1)
        n = size(C, 2)
        @inbounds begin
            $(stores...)
        end
        return
    end
end

# acc ← acc ⊕ a ⊗ bᵀ on a TM × TN register tile stored column-major in a
# tuple. A separate function, so that `acc` is never captured by a
# closure that reassigns it (which would box it).
# Unrolled explicitly: an ntuple closure of TM * TN calls is not always
# inlined, and an outlined call per multiply-add is ruinous on the GPU.
@generated function rank1_update(s::AbstractSemiring, acc::NTuple{N, V}, a::NTuple{TM, V}, b::NTuple{TN, V}) where {N, V, TM, TN}
    terms = Expr[]

    for c in 1:TN, r in 1:TM
        push!(terms, :(smuladd(s, a[$r], b[$c], acc[$(r + (c - 1) * TM)], Val(:N), Val(:N))))
    end

    return quote
        $(Expr(:meta, :inline))
        @inbounds return ($(terms...),)
    end
end


include("sgetrs.jl")
include("sgetrf.jl")

end
