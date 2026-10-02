# ARCHIVED (2026-10-02): the C++ backend experiment of cuda/NOTES.md. It targets the solver API before
# the GPUConfig refactor and is kept for the record; it is not part of the package and not maintained.
# CUDA C++ backend for SemiringGPU: the same algorithms as sgemx_gpu! and
# closure_gpu! / sssp_gpu!, with every kernel replaced by a hand-written CUDA C++
# kernel from cuda/ (libsemiring_cuda.so, built by cuda/build.sh), called with
# @ccall on CUDA.jl's task-local stream.
#
#   include("src/SemiringGPU.jl"); include("src/cuda_backend.jl")
#   using .SemiringCUDA
#   sgemx_cuda!(s, C, A, B)            # C ← C ⊕ A ⊗ B (column-major, strided views allowed)
#   closure_cuda!(D, G)                # == closure_gpu!(D, G)
#   sssp_cuda!(X, G, sources)          # == sssp_gpu!(X, G, sources)
#
# Supported: MinPlus, MaxMin, PlusProd × Float32, Float64. Index type Int64 (as GPUSLU).
#
# The schedule (levels, large fronts, precomputed operators when G.ops[] is set,
# workspaces) is that of src/sgetrs.jl, line for line; only the kernels differ.
# It follows SemiringGPU.PERSISTENT[] / PATH_WARP[] (the developer's persistent L
# sweep and warp path walk) unless `persistent` / `path_warp` are passed.
# `variant` selects the batched sweep kernels: 0 = line-by-line port of the Julia
# kernels, 1 = residual values in registers (bit-identical results).
module SemiringCUDA

using CUDA
using ..SemiringGPU
using ..SemiringGPU: Semiring, GPUSLU, @phase, ispositive, trsm_workspace, ops_workspace, permutecols_gpu!, upload, STRSX_GPU_NB, STRSX_INV_MIN
using .Semiring: AbstractSemiring, MinPlus, MaxMin, PlusProd, szero, sone, isintegral

export sgemx_cuda!, closure_cuda!, closure_cuda, sssp_cuda!

# ENV["SEMIRING_CUDA_LIB"] overrides the library (e.g. a PTX-only build, or another arch)
const LIB = get(ENV, "SEMIRING_CUDA_LIB", joinpath(@__DIR__, "..", "cuda", "build", "libsemiring_cuda.so"))

isfile(LIB) || error("$LIB not found: run cuda/build.sh")

srcode(::MinPlus) = Cint(0)
srcode(::MaxMin) = Cint(1)
srcode(::PlusProd) = Cint(2)
srcode(s) = error("SemiringCUDA: unsupported semiring $(typeof(s))")
dtcode(::Type{Float32}) = Cint(0)
dtcode(::Type{Float64}) = Cint(1)
dtcode(T) = error("SemiringCUDA: unsupported element type $T")

stream() = Ptr{Cvoid}(CUDA.stream().handle)

function check(r::Cint)
    r == 0 || error("SemiringCUDA: kernel launch failed (code $r)")
    return nothing
end

# device pointer and leading dimension of a column-major (possibly strided) matrix
@inline dptr(X::AbstractArray{T}) where {T} = reinterpret(CuPtr{Cvoid}, Base.unsafe_convert(CuPtr{T}, X))
@inline function ld(X::AbstractMatrix)
    st = strides(X)
    @assert st[1] == 1 "SemiringCUDA: matrices must be column-major with unit row stride"
    return Int64(st[2])
end
@inline iptr(v::CuVector{Int64}) = reinterpret(CuPtr{Cvoid}, pointer(v))

# ===== GEMM =====

"""
    sgemx_cuda!(s, C, A, B; tiling = 0)

C ← C ⊕ A ⊗ B with the CUDA C++ kernel. `tiling` 0 chooses; 1 = 128×128 (8×8 per
thread), 2 = 128×64, 3 = 64×64, 4 = 128×32, 5 = 256×16, 6/7 = 1/2 with BK = 16,
8 = 128×16 (128 threads), 9/10 = 4/8 with BK = 32 (see cuda/NOTES.md).
"""
function sgemx_cuda!(s::AbstractSemiring, C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}; tiling::Integer = 0) where {T}
    m, n = size(C)
    k = size(A, 2)
    @assert size(A, 1) == m && size(B) == (k, n)
    (m > 0 && n > 0 && k > 0) || return C
    GC.@preserve C A B begin
        r = @ccall LIB.sr_gemm(srcode(s)::Cint, dtcode(T)::Cint, Cint(tiling)::Cint, Cint(m)::Cint, Cint(n)::Cint, Cint(k)::Cint,
            dptr(A)::CuPtr{Cvoid}, ld(A)::Int64, dptr(B)::CuPtr{Cvoid}, ld(B)::Int64, dptr(C)::CuPtr{Cvoid}, ld(C)::Int64,
            stream()::Ptr{Cvoid})::Cint
    end
    check(r)
    return C
end

choose_tiling(m::Integer, n::Integer, k::Integer) = @ccall LIB.sr_gemm_choose_tiling(Cint(m)::Cint, Cint(n)::Cint, Cint(k)::Cint)::Cint

# ===== helpers =====

function fill_cuda!(X::AbstractMatrix{T}, v) where {T}
    isempty(X) && return X
    GC.@preserve X begin
        r = @ccall LIB.sr_fill(dtcode(T)::Cint, Int64(size(X, 1))::Int64, Int64(size(X, 2))::Int64, dptr(X)::CuPtr{Cvoid}, ld(X)::Int64,
            Float64(v)::Cdouble, stream()::Ptr{Cvoid})::Cint
    end
    check(r)
    return X
end

function copy_cuda!(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T}
    @assert size(X) == size(Y)
    isempty(X) && return X
    GC.@preserve X Y begin
        r = @ccall LIB.sr_copy(dtcode(T)::Cint, Int64(size(X, 1))::Int64, Int64(size(X, 2))::Int64, dptr(X)::CuPtr{Cvoid}, ld(X)::Int64,
            dptr(Y)::CuPtr{Cvoid}, ld(Y)::Int64, stream()::Ptr{Cvoid})::Cint
    end
    check(r)
    return X
end

function identity_cuda!(s::AbstractSemiring, X::AbstractMatrix{T}) where {T}
    GC.@preserve X begin
        r = @ccall LIB.sr_identity(srcode(s)::Cint, dtcode(T)::Cint, Int64(size(X, 1))::Int64, Int64(size(X, 2))::Int64,
            dptr(X)::CuPtr{Cvoid}, ld(X)::Int64, stream()::Ptr{Cvoid})::Cint
    end
    check(r)
    return X
end

# M[:, r] ← C[:, Stgt[Sp + r - 1]]
function gather_cuda!(M::AbstractMatrix{T}, C::AbstractMatrix{T}, Stgt::CuVector{Int64}, Sp) where {T}
    GC.@preserve M C Stgt begin
        r = @ccall LIB.sr_gather(dtcode(T)::Cint, Int64(size(M, 1))::Int64, Int64(size(M, 2))::Int64, dptr(M)::CuPtr{Cvoid}, ld(M)::Int64,
            dptr(C)::CuPtr{Cvoid}, ld(C)::Int64, iptr(Stgt)::CuPtr{Cvoid}, Int64(Sp)::Int64, stream()::Ptr{Cvoid})::Cint
    end
    check(r)
    return M
end

# C[:, Stgt[Sp + r - 1]] ← C[:, Stgt[Sp + r - 1]] ⊕ M[:, r]
function scatteradd_cuda!(s::AbstractSemiring, C::AbstractMatrix{T}, M::AbstractMatrix{T}, Stgt::CuVector{Int64}, Sp) where {T}
    GC.@preserve M C Stgt begin
        r = @ccall LIB.sr_scatteradd(srcode(s)::Cint, dtcode(T)::Cint, Int64(size(M, 1))::Int64, Int64(size(M, 2))::Int64,
            dptr(C)::CuPtr{Cvoid}, ld(C)::Int64, dptr(M)::CuPtr{Cvoid}, ld(M)::Int64, iptr(Stgt)::CuPtr{Cvoid}, Int64(Sp)::Int64,
            stream()::Ptr{Cvoid})::Cint
    end
    check(r)
    return C
end

function strsx_diag_cuda!(s::AbstractSemiring, upper::Bool, X::AbstractMatrix{T}, A::AbstractMatrix{T}, threads::Integer) where {T}
    GC.@preserve X A begin
        r = @ccall LIB.sr_strsx_diag(srcode(s)::Cint, dtcode(T)::Cint, Cint(upper)::Cint, Int64(size(X, 1))::Int64, Int64(size(A, 1))::Int64,
            dptr(X)::CuPtr{Cvoid}, ld(X)::Int64, dptr(A)::CuPtr{Cvoid}, ld(A)::Int64, Cint(threads)::Cint, stream()::Ptr{Cvoid})::Cint
    end
    check(r)
    return X
end

#
# strsx_gpu! of src/sgetrs.jl with C++ kernels: X ← X A*, blocked over 64-column
# diagonal blocks, diagonal blocks applied by inversion when X has > 64 rows.
#
function strsx_cuda!(s::AbstractSemiring, upper::Bool, X::AbstractMatrix{T}, A::AbstractMatrix{T}; nb::Int = STRSX_GPU_NB, tiling = 0) where {T}
    n = size(A, 1)
    m = size(X, 1)
    inv = m > STRSX_INV_MIN
    blocks = upper ? [(j0, min(j0 + nb - 1, n)) for j0 in 1:nb:n] : [(max(j1 - nb + 1, 1), j1) for j1 in n:-nb:1]

    for (j0, j1) in blocks
        J = j0:j1
        b = j1 - j0 + 1

        if inv
            Tb, W = trsm_workspace(T, m * b)
            Tj = view(Tb, 1:b, 1:b)
            identity_cuda!(s, Tj)
            strsx_diag_cuda!(s, upper, Tj, view(A, J, J), 32 * cld(b, 32))
            Wj = reshape(view(W, 1:(m * b)), m, b)
            fill_cuda!(Wj, szero(s, T, Val(:N)))
            sgemx_cuda!(s, Wj, view(X, :, J), Tj; tiling)
            copy_cuda!(view(X, :, J), Wj)
        else
            strsx_diag_cuda!(s, upper, view(X, :, J), view(A, J, J), 0)
        end

        if upper && j1 < n
            sgemx_cuda!(s, view(X, :, (j1 + 1):n), view(X, :, J), view(A, J, (j1 + 1):n); tiling)
        elseif !upper && j0 > 1
            sgemx_cuda!(s, view(X, :, 1:(j0 - 1)), view(X, :, J), view(A, J, 1:(j0 - 1)); tiling)
        end
    end

    return X
end

# ===== batched sweeps =====

function upward_batched_cuda!(G::GPUSLU{Sem, T, Int64}, C::CuMatrix{T}, order::CuVector{Int64}, off::Int, nfl::Int, tb::Int, variant::Int) where {Sem, T}
    GC.@preserve G C order begin
        r = @ccall LIB.sr_upward_batched(srcode(G.s)::Cint, dtcode(T)::Cint, Cint(variant)::Cint, Int64(size(C, 1))::Int64,
            dptr(C)::CuPtr{Cvoid}, ld(C)::Int64, iptr(order)::CuPtr{Cvoid}, Int64(off)::Int64, Int64(nfl)::Int64,
            iptr(G.Rptr)::CuPtr{Cvoid}, iptr(G.Sptr)::CuPtr{Cvoid}, iptr(G.Stgt)::CuPtr{Cvoid}, iptr(G.Dptr)::CuPtr{Cvoid}, iptr(G.Lptr)::CuPtr{Cvoid},
            dptr(G.UDval)::CuPtr{Cvoid}, dptr(G.ULval)::CuPtr{Cvoid}, Cint(tb)::Cint, stream()::Ptr{Cvoid})::Cint
    end
    check(r)
end

function downward_batched_cuda!(G::GPUSLU{Sem, T, Int64}, C::CuMatrix{T}, order::CuVector{Int64}, off::Int, nfl::Int, tb::Int, variant::Int) where {Sem, T}
    GC.@preserve G C order begin
        r = @ccall LIB.sr_downward_batched(srcode(G.s)::Cint, dtcode(T)::Cint, Cint(variant)::Cint, Int64(size(C, 1))::Int64,
            dptr(C)::CuPtr{Cvoid}, ld(C)::Int64, iptr(order)::CuPtr{Cvoid}, Int64(off)::Int64, Int64(nfl)::Int64,
            iptr(G.Rptr)::CuPtr{Cvoid}, iptr(G.Sptr)::CuPtr{Cvoid}, iptr(G.Stgt)::CuPtr{Cvoid}, iptr(G.Dptr)::CuPtr{Cvoid}, iptr(G.Lptr)::CuPtr{Cvoid},
            dptr(G.LDval)::CuPtr{Cvoid}, dptr(G.LLval)::CuPtr{Cvoid}, Cint(tb)::Cint, stream()::Ptr{Cvoid})::Cint
    end
    check(r)
end

function upward_path_cuda!(G::GPUSLU{Sem, T, Int64}, C::CuMatrix{T}, sources::CuVector{Int64}, tb::Int, variant::Int) where {Sem, T}
    GC.@preserve G C sources begin
        r = @ccall LIB.sr_upward_path(srcode(G.s)::Cint, dtcode(T)::Cint, Cint(variant)::Cint, Int64(size(C, 1))::Int64,
            dptr(C)::CuPtr{Cvoid}, ld(C)::Int64, iptr(sources)::CuPtr{Cvoid}, iptr(G.cinvp)::CuPtr{Cvoid}, iptr(G.idx)::CuPtr{Cvoid},
            iptr(G.pnt)::CuPtr{Cvoid}, reinterpret(CuPtr{Cvoid}, pointer(G.istop))::CuPtr{Cvoid},
            iptr(G.Rptr)::CuPtr{Cvoid}, iptr(G.Sptr)::CuPtr{Cvoid}, iptr(G.Stgt)::CuPtr{Cvoid}, iptr(G.Dptr)::CuPtr{Cvoid}, iptr(G.Lptr)::CuPtr{Cvoid},
            dptr(G.UDval)::CuPtr{Cvoid}, dptr(G.ULval)::CuPtr{Cvoid}, Cint(tb)::Cint, stream()::Ptr{Cvoid})::Cint
    end
    check(r)
end

function upward_path_warp_cuda!(G::GPUSLU{Sem, T, Int64}, C::CuMatrix{T}, sources::CuVector{Int64}, variant::Int) where {Sem, T}
    GC.@preserve G C sources begin
        r = @ccall LIB.sr_upward_path_warp(srcode(G.s)::Cint, dtcode(T)::Cint, Cint(variant)::Cint, Int64(size(C, 1))::Int64,
            dptr(C)::CuPtr{Cvoid}, ld(C)::Int64, iptr(sources)::CuPtr{Cvoid}, iptr(G.cinvp)::CuPtr{Cvoid}, iptr(G.idx)::CuPtr{Cvoid},
            iptr(G.pnt)::CuPtr{Cvoid}, reinterpret(CuPtr{Cvoid}, pointer(G.istop))::CuPtr{Cvoid},
            iptr(G.Rptr)::CuPtr{Cvoid}, iptr(G.Sptr)::CuPtr{Cvoid}, iptr(G.Stgt)::CuPtr{Cvoid}, iptr(G.Dptr)::CuPtr{Cvoid}, iptr(G.Lptr)::CuPtr{Cvoid},
            dptr(G.UDval)::CuPtr{Cvoid}, dptr(G.ULval)::CuPtr{Cvoid}, Cint(128)::Cint, stream()::Ptr{Cvoid})::Cint
    end
    check(r)
end

const NSM = Ref(0)
nsm() = iszero(NSM[]) ? (NSM[] = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)) : NSM[]

# persistent_down_kernel! of sgetrs.jl, with its plan (SemiringGPU.persistent_plan) and launch shape
function persistent_down_cuda!(G::GPUSLU{Sem, T, Int64}, C::CuMatrix{T}, plan, variant::Int) where {Sem, T}
    tb = SemiringGPU.PERSIST_TB
    nitems = plan.nrest * cld(size(C, 1), tb)
    nblk = min(nitems, nsm() * SemiringGPU.PERSIST_BLOCKS_PER_SM)
    GC.@preserve G C plan begin
        r = @ccall LIB.sr_persistent_down(srcode(G.s)::Cint, dtcode(T)::Cint, Cint(variant)::Cint, Int64(size(C, 1))::Int64,
            dptr(C)::CuPtr{Cvoid}, ld(C)::Int64, iptr(plan.rest)::CuPtr{Cvoid}, Int64(plan.nrest)::Int64, Cint(tb)::Cint, Cint(nblk)::Cint,
            iptr(G.pnt)::CuPtr{Cvoid}, reinterpret(CuPtr{Cvoid}, pointer(G.istop))::CuPtr{Cvoid},
            reinterpret(CuPtr{Cvoid}, pointer(plan.counters))::CuPtr{Cvoid}, Int64(length(plan.counters))::Int64,
            reinterpret(CuPtr{Cvoid}, pointer(plan.ticket))::CuPtr{Cvoid},
            iptr(G.Rptr)::CuPtr{Cvoid}, iptr(G.Sptr)::CuPtr{Cvoid}, iptr(G.Stgt)::CuPtr{Cvoid}, iptr(G.Dptr)::CuPtr{Cvoid}, iptr(G.Lptr)::CuPtr{Cvoid},
            dptr(G.LDval)::CuPtr{Cvoid}, dptr(G.LLval)::CuPtr{Cvoid}, stream()::Ptr{Cvoid})::Cint
    end
    check(r)
end

# the schedule switches of src/sgetrs.jl (absent in older versions = the level-by-level schedule)
jlflag(name::Symbol) = isdefined(SemiringGPU, name) && getfield(SemiringGPU, name)[]

# ===== large fronts (upward_large! / downward_large! of src/sgetrs.jl) =====

function upward_large_cuda!(G::GPUSLU{<:Any, T}, C::CuMatrix{T}, M::CuMatrix{T}, f, tiling) where {T}
    s = G.s
    Rp = G.hRptr[f]; nn = G.hRptr[f + 1] - Rp
    Sp = G.hSptr[f]; na = G.hSptr[f + 1] - Sp
    Dp = G.hDptr[f]; Lp = G.hLptr[f]

    C₁ = view(C, :, Rp:(Rp + nn - 1))
    ops = G.ops[]

    if !isnothing(ops) && haskey(ops.off, f)
        KU = reshape(view(ops.KU, ops.off[f][2]:(ops.off[f][2] + nn * (nn + na) - 1)), nn, nn + na)
        W₁ = ops_workspace(ops, T, size(C, 1), nn)
        copy_cuda!(W₁, C₁)
        fill_cuda!(C₁, szero(s, T, Val(:N)))
        sgemx_cuda!(s, C₁, W₁, view(KU, :, 1:nn); tiling)

        if ispositive(na)
            M₂ = view(M, :, 1:na)
            fill_cuda!(M₂, szero(s, T, Val(:N)))
            sgemx_cuda!(s, M₂, W₁, view(KU, :, (nn + 1):(nn + na)); tiling)
            scatteradd_cuda!(s, C, M₂, G.Stgt, Sp)
        end

        return
    end

    D₁₁ = reshape(view(G.UDval, Dp:(Dp + nn * nn - 1)), nn, nn)
    strsx_cuda!(s, true, C₁, D₁₁; tiling)

    if ispositive(na)
        U₁₂ = reshape(view(G.ULval, Lp:(Lp + nn * na - 1)), nn, na)
        M₂ = view(M, :, 1:na)
        fill_cuda!(M₂, szero(s, T, Val(:N)))
        sgemx_cuda!(s, M₂, C₁, U₁₂; tiling)
        scatteradd_cuda!(s, C, M₂, G.Stgt, Sp)
    end

    return
end

function downward_large_cuda!(G::GPUSLU{<:Any, T}, C::CuMatrix{T}, M::CuMatrix{T}, f, tiling) where {T}
    s = G.s
    Rp = G.hRptr[f]; nn = G.hRptr[f + 1] - Rp
    Sp = G.hSptr[f]; na = G.hSptr[f + 1] - Sp
    Dp = G.hDptr[f]; Lp = G.hLptr[f]

    C₁ = view(C, :, Rp:(Rp + nn - 1))
    ops = G.ops[]

    if !isnothing(ops) && haskey(ops.off, f)
        KL = reshape(view(ops.KL, ops.off[f][1]:(ops.off[f][1] + (nn + na) * nn - 1)), nn + na, nn)
        W = ops_workspace(ops, T, size(C, 1), nn + na)
        copy_cuda!(view(W, :, 1:nn), C₁)
        ispositive(na) && gather_cuda!(view(W, :, (nn + 1):(nn + na)), C, G.Stgt, Sp)
        fill_cuda!(C₁, szero(s, T, Val(:N)))
        sgemx_cuda!(s, C₁, W, KL; tiling)
        return
    end

    if ispositive(na)
        L₂₁ = reshape(view(G.LLval, Lp:(Lp + nn * na - 1)), na, nn)
        M₂ = view(M, :, 1:na)
        gather_cuda!(M₂, C, G.Stgt, Sp)
        sgemx_cuda!(s, C₁, M₂, L₂₁; tiling)
    end

    D₁₁ = reshape(view(G.LDval, Dp:(Dp + nn * nn - 1)), nn, nn)
    strsx_cuda!(s, false, C₁, D₁₁; tiling)
    return
end

# ===== sweeps, sssp, closure =====

function downward_sweep_cuda!(G::GPUSLU, W::CuMatrix, M::CuMatrix, tb::Int, timer, variant::Int, tiling, persistent::Bool)
    if persistent
        plan = SemiringGPU.persistent_plan(G)
        #
        #   the top of the tree, level by level (large fronts dense, small batched)
        #
        for (small, large) in plan.toplevels
            for f in large
                @phase timer :L_dense downward_large_cuda!(G, W, M, f, tiling)
            end

            if ispositive(length(small))
                @phase timer :L_batched downward_batched_cuda!(G, W, small, 0, length(small), tb, variant)
            end
        end
        #
        #   everything below the top in one persistent launch
        #
        if ispositive(plan.nrest)
            @phase timer :L_persistent persistent_down_cuda!(G, W, plan, variant)
        end

        return W
    end

    for l in 1:(length(G.downptr) - 1)
        for f in G.downlarge[l]
            @phase timer :L_dense downward_large_cuda!(G, W, M, f, tiling)
        end

        strt = G.downptr[l]
        nfl = G.downptr[l + 1] - strt

        if ispositive(nfl)
            @phase timer :L_batched downward_batched_cuda!(G, W, G.down, strt - 1, nfl, tb, variant)
        end
    end

    return W
end

"""
    sssp_cuda!(X, G, sources; W = similar(X), M, nthreads = 64, timer = nothing, permute = true, variant = 1, tiling = 0,
               path_warp = SemiringGPU.PATH_WARP[], persistent = SemiringGPU.PERSISTENT[])

sssp_gpu! with the CUDA C++ kernels (same schedule). By default the schedule follows the
switches of src/sgetrs.jl (warp-per-source path walk, persistent L sweep); pass
path_warp / persistent to choose explicitly.
"""
function sssp_cuda!(X::CuMatrix{T}, G::GPUSLU{Sem, T, Int64}, sources::CuVector{Int64}; W::CuMatrix{T} = similar(X),
        M::CuMatrix{T} = CuMatrix{T}(undef, size(X, 1), G.maxna), nthreads::Int = 64, timer = nothing, permute::Bool = true,
        variant::Int = 1, tiling::Integer = 0, path_warp::Bool = jlflag(:PATH_WARP), persistent::Bool = jlflag(:PERSISTENT)) where {Sem, T}
    @assert size(X) == (length(sources), G.n)
    @assert size(W) == size(X)
    @assert permute || W === X

    s = G.s
    nrhs = size(X, 1)
    tb = min(nthreads, 32 * cld(nrhs, 32))
    #
    #   W ← B Q⁻¹ U*,  B = [e_{s₁}; …; e_{s_k}]
    #
    @phase timer :fill fill_cuda!(W, szero(s, T, Val(:N)))
    if path_warp
        @phase timer :U_path upward_path_warp_cuda!(G, W, sources, variant)
    else
        @phase timer :U_path upward_path_cuda!(G, W, sources, tb, variant)
    end
    #
    #   the top of the tree, for all rows at once
    #
    for l in 1:(length(G.topptr) - 1)
        strt = G.topptr[l]
        nfl = G.topptr[l + 1] - strt

        if ispositive(nfl)
            @phase timer :U_top_batched upward_batched_cuda!(G, W, G.top, strt - 1, nfl, tb, variant)
        end

        for f in G.toplarge[l]
            @phase timer :U_top_dense upward_large_cuda!(G, W, M, f, tiling)
        end
    end
    #
    #   W ← W L*
    #
    downward_sweep_cuda!(G, W, M, tb, timer, variant, tiling, persistent)
    permute && @phase timer :permute permutecols_gpu!(X, W, G.rperm)
    return X
end

"""
    closure_cuda!(D, G; M, timer = nothing, variant = 1, tiling = 0)

closure_gpu!(D, G) with the CUDA C++ kernels: D[i, j] = A*[rperm[i], rperm[j]].
"""
function closure_cuda!(D::CuMatrix{T}, G::GPUSLU{Sem, T, Int64}; timer = nothing, M::CuMatrix{T} = CuMatrix{T}(undef, G.n, G.maxna),
        variant::Int = 1, tiling::Integer = 0, kw...) where {Sem, T}
    @assert size(D) == (G.n, G.n)
    sources = upload(Array(G.rperm))
    return sssp_cuda!(D, G, sources; W = D, M, timer, permute = false, variant, tiling, kw...)
end

closure_cuda(G::GPUSLU{Sem, T}; kw...) where {Sem, T} = closure_cuda!(CuMatrix{T}(undef, G.n, G.n), G; kw...)

end
