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

export sgemx_gpu!, GPUSLU, rmul_gpu!, sssp_gpu!, SSSPPlan, sgetrf_gpu!, mlu_gpu, FactorPlan, factorize!, closure_gpu!, closure_gpu, precompute_ops!

# ===== device overrides =====
#
# Semiring.vmin(x, y) is ifelse(x < y, x, y) on x86 hosts (a host-side
# @static choice that the GPU compilation inherits), which ptxas compiles
# to FSETP + FSEL. On the GPU, llvm.minnum gives the same result whenever
# the accumulator is not NaN (in particular +∞ + -∞ = NaN is absorbed, as
# upstream intends) and is a single FMNMX instruction. This halves the cost
# of a min-plus multiply-add.
#
CUDA.@device_override @inline Semiring.vmin(x::Float32, y::Float32) = ccall("llvm.minnum.f32", llvmcall, Float32, (Float32, Float32), x, y)
CUDA.@device_override @inline Semiring.vmax(x::Float32, y::Float32) = ccall("llvm.maxnum.f32", llvmcall, Float32, (Float32, Float32), x, y)
CUDA.@device_override @inline Semiring.vmin(x::Float64, y::Float64) = ccall("llvm.minnum.f64", llvmcall, Float64, (Float64, Float64), x, y)
CUDA.@device_override @inline Semiring.vmax(x::Float64, y::Float64) = ccall("llvm.maxnum.f64", llvmcall, Float64, (Float64, Float64), x, y)

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

# overwrite = true computes C ← A ⊗ B instead (C is not read). C may then be A itself (C = A ⊗ B in
# place) when the output is one tile wide (size(C, 2) ≤ BN), since every block reads all of its rows
# of A before it writes them; see inplace_ok.
function sgemx_gpu!(s::AbstractSemiring, C::AbstractMatrix{V}, A::AbstractMatrix{V}, B::AbstractMatrix{V};
        tiling::Tiling = choose_tiling(size(C, 1), size(C, 2)), overwrite::Bool = false) where {V}
    @assert size(C, 1) == size(A, 1)
    @assert size(C, 2) == size(B, 2)
    @assert size(A, 2) == size(B, 1)

    m = size(C, 1)
    n = size(C, 2)
    k = size(A, 2)

    if m > 0 && n > 0 && k > 0
        launch!(s, C, A, B, tiling, Val(overwrite))
    elseif overwrite && m > 0 && n > 0
        fill!(C, szero(s, V, Val(:N)))
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
    elseif n <= 64                    # a 128-wide tile would be at least half empty (2.2× slower on 27000 × 64 × 64)
        return TILING_SMALL
    elseif cld(m, 128) * cld(n, 128) >= 2 * NSM[]
        return TILING_LARGE
    else
        return TILING_SMALL
    end
end

# can C = A ⊗ B be computed in place (C === A) with this tiling?
inplace_ok(n::Integer, ::Tiling{BM, BN}) where {BM, BN} = GEMM_VERSION[] == 2 && n <= BN

function launch!(s::AbstractSemiring, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, ::Tiling{BM, BN, BK, TM, TN}, ::Val{OW} = Val(false)) where {BM, BN, BK, TM, TN, OW}
    TX = BM ÷ TM
    TY = BN ÷ TN
    blocks = (cld(size(C, 1), BM), cld(size(C, 2), BN))
    if GEMM_VERSION[] == 2
        @cuda threads = TX * TY blocks = blocks sgemx_kernel2!(s, C, A, B, Val(BM), Val(BN), Val(BK), Val(TM), Val(TN), Val(OW))
    else
        OW && fill!(C, szero(s, eltype(C), Val(:N)))
        @cuda threads = TX * TY blocks = blocks sgemx_kernel!(s, C, A, B, Val(BM), Val(BN), Val(BK), Val(TM), Val(TN))
    end
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

# ===== kernel v2: pipelined =====
#
# The same tile shapes as sgemx_kernel!, with the structure of cuASR /
# CUTLASS's 2-stage SIMT GEMM and TropicalGEMM's thread mapping:
#
#   - two shared-memory buffers: while the block computes on one k-panel,
#     the next panel is fetched from global memory into registers and then
#     stored into the other buffer, so there is one barrier per panel;
#   - each thread owns groups of VW = 4 contiguous rows (columns) of its
#     register tile, so its shared-memory reads are contiguous 4-vectors;
#   - Bs is stored n-contiguous (Bs[j, p]) and padded by 4 to avoid bank
#     conflicts on the transposing store (cuASR pads transposed layouts).
#
# Row r ∈ 1:TM of the register tile is row g VW TX + VW tx + v of the
# block tile, with g = (r - 1) ÷ VW and v = (r - 1) % VW (VW = 1 gives the
# strided mapping of sgemx_kernel!).

const GEMM_VERSION = Ref(2)
const BPAD = 4

function sgemx_kernel2!(s::AbstractSemiring, C::AbstractMatrix{V}, A::AbstractMatrix{V}, B::AbstractMatrix{V},
        ::Val{BM}, ::Val{BN}, ::Val{BK}, ::Val{TM}, ::Val{TN}, ::Val{OW} = Val(false)) where {V, BM, BN, BK, TM, TN, OW}
    TX = BM ÷ TM
    TY = BN ÷ TN
    NT = TX * TY
    VWM = TM % 4 == 0 ? 4 : 1
    VWN = TN % 4 == 0 ? 4 : 1
    NA = cld(BM * BK, NT)       # A elements each thread prefetches per panel
    NB = cld(BK * BN, NT)

    m = size(C, 1)
    n = size(C, 2)
    k = size(A, 2)

    As = CuStaticSharedArray(V, (BM, BK, 2))
    Bs = CuStaticSharedArray(V, (BN + BPAD, BK, 2))

    t  = threadIdx().x - 1
    tx = t % TX
    ty = t ÷ TX
    i0 = (blockIdx().x - 1) * BM
    j0 = (blockIdx().y - 1) * BN

    z = szero(s, V, Val(:N))
    u = sone(s, V, Val(:N))
    acc = ntuple(_ -> z, Val(TM * TN))
    npanel = cld(k, BK)

    @inbounds begin
        ra = fetch_a(A, i0, 0, m, k, t, z, Val(BM), Val(BK), Val(NT), Val(NA))
        rb = fetch_b(B, j0, 0, n, k, t, u, Val(BN), Val(BK), Val(NT), Val(NB))
        stash_a!(As, ra, 1, t, Val(BM), Val(BK), Val(NT), Val(NA))
        stash_b!(Bs, rb, 1, t, Val(BN), Val(BK), Val(NT), Val(NB))
        sync_threads()

        for q in 1:npanel
            cur = isodd(q) ? 1 : 2
            nxt = 3 - cur

            if q < npanel
                k0 = q * BK
                ra = fetch_a(A, i0, k0, m, k, t, z, Val(BM), Val(BK), Val(NT), Val(NA))
                rb = fetch_b(B, j0, k0, n, k, t, u, Val(BN), Val(BK), Val(NT), Val(NB))
            end

            for p in 1:BK
                a = load_frag(As, p, cur, tx, Val(TX), Val(TM), Val(VWM))
                b = load_frag(Bs, p, cur, ty, Val(TY), Val(TN), Val(VWN))
                acc = rank1_update(s, acc, a, b)
            end

            if q < npanel
                stash_a!(As, ra, nxt, t, Val(BM), Val(BK), Val(NT), Val(NA))
                stash_b!(Bs, rb, nxt, t, Val(BN), Val(BK), Val(NT), Val(NB))
            end

            sync_threads()
        end
    end

    store_tile2!(s, C, acc, i0, j0, tx, ty, Val(TX), Val(TY), Val(TM), Val(VWM), Val(VWN), Val(OW))
    return
end

# the NA elements of the A panel (rows i0+1:i0+BM, columns k0+1:k0+BK) that thread t loads, coalesced along columns
@generated function fetch_a(A, i0, k0, m, k, t, z, ::Val{BM}, ::Val{BK}, ::Val{NT}, ::Val{NA}) where {BM, BK, NT, NA}
    loads = [quote
        e = t + $(l * NT)
        i = e % $BM; p = e ÷ $BM
        gi = i0 + i + 1; gp = k0 + p + 1
        (e < $(BM * BK) && gi <= m && gp <= k) ? A[gi, gp] : z
    end for l in 0:(NA - 1)]
    return :($(Expr(:meta, :inline)); @inbounds ($(loads...),))
end

@generated function fetch_b(B, j0, k0, n, k, t, u, ::Val{BN}, ::Val{BK}, ::Val{NT}, ::Val{NB}) where {BN, BK, NT, NB}
    loads = [quote
        e = t + $(l * NT)
        p = e % $BK; j = e ÷ $BK
        gp = k0 + p + 1; gj = j0 + j + 1
        (e < $(BK * BN) && gp <= k && gj <= n) ? B[gp, gj] : u
    end for l in 0:(NB - 1)]
    return :($(Expr(:meta, :inline)); @inbounds ($(loads...),))
end

@generated function stash_a!(As, ra, buf, t, ::Val{BM}, ::Val{BK}, ::Val{NT}, ::Val{NA}) where {BM, BK, NT, NA}
    stores = [quote
        e = t + $(l * NT)
        if e < $(BM * BK)
            As[e % $BM + 1, e ÷ $BM + 1, buf] = ra[$(l + 1)]
        end
    end for l in 0:(NA - 1)]
    return :($(Expr(:meta, :inline)); @inbounds begin $(stores...) end; nothing)
end

@generated function stash_b!(Bs, rb, buf, t, ::Val{BN}, ::Val{BK}, ::Val{NT}, ::Val{NB}) where {BN, BK, NT, NB}
    stores = [quote
        e = t + $(l * NT)
        if e < $(BK * BN)
            Bs[e ÷ $BK + 1, e % $BK + 1, buf] = rb[$(l + 1)]
        end
    end for l in 0:(NB - 1)]
    return :($(Expr(:meta, :inline)); @inbounds begin $(stores...) end; nothing)
end

# register fragment: entries r = 1:TM of row (or column) p of the panel, VW-contiguous groups
@generated function load_frag(S, p, buf, tx, ::Val{TX}, ::Val{TM}, ::Val{VW}) where {TX, TM, VW}
    loads = [:(S[$((r - 1) ÷ VW * VW * TX + (r - 1) % VW + 1) + $VW * tx, p, buf]) for r in 1:TM]
    return :($(Expr(:meta, :inline)); @inbounds ($(loads...),))
end

@generated function store_tile2!(s, C, acc::NTuple{N}, i0, j0, tx, ty, ::Val{TX}, ::Val{TY}, ::Val{TM}, ::Val{VWM}, ::Val{VWN}, ::Val{OW} = Val(false)) where {N, TX, TY, TM, VWM, VWN, OW}
    stores = Expr[]

    for e in 1:N
        r = (e - 1) % TM + 1
        c = (e - 1) ÷ TM + 1
        ri = (r - 1) ÷ VWM * VWM * TX + (r - 1) % VWM + 1
        cj = (c - 1) ÷ VWN * VWN * TY + (c - 1) % VWN + 1

        push!(stores, quote
            gi = i0 + $ri + $VWM * tx
            gj = j0 + $cj + $VWN * ty

            if gi <= m && gj <= n
                C[gi, gj] = $(OW ? :(acc[$e]) : :(splus(s, acc[$e], C[gi, gj], Val(:N))))
            end
        end)
    end

    return quote
        $(Expr(:meta, :inline))
        m = size(C, 1); n = size(C, 2)
        @inbounds begin
            $(stores...)
        end
        return
    end
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


include("amalgamate.jl")
include("sgetrs.jl")
include("layered.jl")
include("sgetrf.jl")

end
