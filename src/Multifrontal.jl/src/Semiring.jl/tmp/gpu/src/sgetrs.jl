# GPU solves with a ChordalSLU factor.
#
#   rmul_gpu!(B, G)    B ← B A*    (B is nrhs × n: the row layout)
#
# This is the GPU counterpart of rmul!(B, F) = sgetrs!(F, Val(:R), Val(:N), B):
# k single-source queries at once, X = B U* L*. Row t of B holds the k-th
# right-hand side, so the values of one vertex are contiguous and a warp
# that maps its threads to right-hand sides reads and writes coalesced
# memory.
#
# The solve is level-scheduled over the elimination tree:
#
#   B ← B U*   leaves to root, one launch per height. A block owns one
#              front: it solves with U₁₁ on its residual columns and
#              ⊕-scatters C₁ U₁₂ into its separator columns. Siblings may
#              scatter into the same column, so the scatter is an atomic
#              compare-and-swap loop around splus (generic, any semiring
#              whose elements are 4 or 8 bytes).
#
#   B ← B L*   root to leaves, one launch per depth. A block gathers its
#              separator columns, which its ancestors have finished,
#              adds M₂ L₂₁, and solves with L₁₁. No atomics.
#
# The factorization itself stays on the CPU.

const MF = CliqueTrees.Multifrontal

using .Semiring: ChordalSLU, sprod, sstar, isintegral

#
# @phase timer :name ex   runs ex, and if timer is a Dict accumulates its
# GPU time under :name (this synchronizes; use it only for profiling)
#
macro phase(timer, name, ex)
    return quote
        if isnothing($(esc(timer)))
            $(esc(ex))
        else
            t = CUDA.@elapsed $(esc(ex))
            $(esc(timer))[$name] = get($(esc(timer)), $name, 0.0) + t
        end
    end
end

struct GPUSLU{Sem <: AbstractSemiring, T, I}
    s::Sem
    n::Int
    nf::Int
    # symbolic structure, on the device and (for large fronts) on the host
    Rptr::CuVector{I}           # residual of front f: Rptr[f]:Rptr[f + 1] - 1
    Sptr::CuVector{I}           # separator of front f: Stgt[Sptr[f]:Sptr[f + 1] - 1]
    Stgt::CuVector{I}
    Dptr::CuVector{I}
    Lptr::CuVector{I}
    hRptr::Vector{I}
    hSptr::Vector{I}
    hDptr::Vector{I}
    hLptr::Vector{I}
    # numeric factor
    LDval::CuVector{T}
    LLval::CuVector{T}
    UDval::CuVector{T}
    ULval::CuVector{T}
    # schedule: level l of the upward sweep (by height, leaves first) has
    # small fronts up[upptr[l]:upptr[l + 1] - 1], batched in one launch,
    # and large fronts uplarge[l], each solved with dense kernels
    up::CuVector{I}
    upptr::Vector{Int}
    uplarge::Vector{Vector{I}}
    # downward sweep, by depth, roots first
    down::CuVector{I}
    downptr::Vector{Int}
    downlarge::Vector{Vector{I}}
    maxna::Int                  # largest separator of a large front
    idx::CuVector{I}            # front of each vertex (elimination order)
    pnt::CuVector{I}            # parent of each front (0 at a root)
    # the top of the tree: large fronts and all their ancestors. The path
    # walk of sssp_gpu! stops there, and the top is swept level by level.
    istop::CuVector{Bool}
    top::CuVector{I}
    topptr::Vector{Int}
    toplarge::Vector{Vector{I}}
    cinvp::CuVector{I}
    rperm::CuVector{I}
end

#
# A front is large when its solve work nn (nn + na) per right-hand side
# reaches `large`: one thread per right-hand side would then serialize too
# much, so it is solved with tiled dense kernels instead.
#
function GPUSLU(F::ChordalSLU{Sem, T, I}; large::Integer = 2048, factor = nothing) where {Sem, T, I}
    if !iszero(MF.ne(F.S.N))
        error("GPUSLU: coupling between strongly connected components is not supported yet")
    end

    S = F.S.S
    nf = Int(MF.nv(S.res))
    pnt = S.pnt
    Rptr = MF.pointers(S.res)
    Sptr = MF.pointers(S.sep)
    #
    # height (leaves = 1) and depth (roots = 1) of every front;
    # fronts are postordered, so parents come after their children
    #
    height = ones(Int, nf)
    depth = ones(Int, nf)

    for f in 1:nf
        p = pnt[f]
        @assert iszero(p) || p > f

        if !iszero(p)
            height[p] = max(height[p], height[f] + 1)
        end
    end

    for f in nf:-1:1
        p = pnt[f]

        if !iszero(p)
            depth[f] = depth[p] + 1
        end
    end

    islarge = falses(nf)
    maxna = 0

    for f in 1:nf
        nn = Rptr[f + 1] - Rptr[f]
        na = Sptr[f + 1] - Sptr[f]

        if nn * (nn + na) >= large
            islarge[f] = true
            maxna = max(maxna, na)
        end
    end

    istop = falses(nf)

    for f in 1:nf
        if islarge[f]
            g = f

            while !iszero(g) && !istop[g]
                istop[g] = true
                g = pnt[g]
            end
        end
    end

    up, upptr, uplarge = levels(height, islarge, I)
    down, downptr, downlarge = levels(depth, islarge, I)
    top, topptr, toplarge = levels(height, islarge, I, istop)

    return GPUSLU{Sem, T, I}(
        F.s, size(F, 1), nf,
        upload(Rptr), upload(Sptr), upload(MF.targets(S.sep)), upload(S.Dptr), upload(S.Lptr),
        Vector{I}(Rptr), Vector{I}(Sptr), Vector{I}(S.Dptr), Vector{I}(S.Lptr),
        (isnothing(factor) ? (upload(F.LDval), upload(F.LLval), upload(F.UDval), upload(F.ULval)) : factor)...,
        CuVector(up), upptr, uplarge, CuVector(down), downptr, downlarge, maxna,
        upload(view(S.idx, 1:size(F, 1))), upload(view(pnt, 1:nf)),
        CuVector(Vector{Bool}(istop)), CuVector(top), topptr, toplarge,
        upload(F.cinvp), upload(F.rperm),
    )
end

function levels(key::Vector{Int}, islarge::BitVector, ::Type{I}, keep::BitVector = trues(length(key))) where {I}
    nl = maximum(key; init = 0)
    ptr = zeros(Int, nl + 1)
    large = [I[] for _ in 1:nl]

    for (f, k) in enumerate(key)
        if !keep[f]
            continue
        elseif islarge[f]
            push!(large[k], f)
        else
            ptr[k + 1] += 1
        end
    end

    ptr[1] = 1
    cumsum!(ptr, ptr)

    order = Vector{I}(undef, ptr[end] - 1)
    next = copy(ptr)

    for (f, k) in enumerate(key)
        if keep[f] && !islarge[f]
            order[next[k]] = f
            next[k] += 1
        end
    end

    return order, ptr, large
end

nlevels(G::GPUSLU) = length(G.upptr) - 1

nlarge(G::GPUSLU) = sum(length, G.uplarge)

# ===== rmul_gpu! =====

function rmul_gpu!(B::CuMatrix{T}, G::GPUSLU{Sem, T, I}; W::CuMatrix{T} = similar(B), nthreads::Int = 64, timer = nothing) where {Sem, T, I}
    @assert size(B, 2) == G.n
    @assert size(W) == size(B)

    s = G.s
    trans = Val(:N)
    scale = Val(!isintegral(s))
    nrhs = size(B, 1)
    tb = min(nthreads, 32 * cld(nrhs, 32))
    nb = cld(nrhs, tb)
    M = CuMatrix{T}(undef, nrhs, G.maxna)
    #
    #   W ← B Q⁻¹
    #
    @phase timer :permute permutecols_gpu!(W, B, G.cinvp)
    #
    #   W ← W U*
    #
    kernel = @cuda launch = false upward_kernel!(s, trans, scale, W, G.up, 0, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.UDval, G.ULval)

    for l in 1:nlevels(G)
        strt = G.upptr[l]
        nfl = G.upptr[l + 1] - strt

        if ispositive(nfl)
            @phase timer :U_batched kernel(s, trans, scale, W, G.up, strt - 1, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.UDval, G.ULval; threads = tb, blocks = (nfl, nb))
        end

        for f in G.uplarge[l]
            @phase timer :U_dense upward_large!(G, W, M, f, trans, scale)
        end
    end
    #
    #   W ← W L*
    #
    downward_sweep!(G, W, M, trans, tb, nb, timer)
    #
    #   B ← W P⁻¹
    #
    @phase timer :permute permutecols_gpu!(B, W, G.rperm)
    return B
end

# ===== sssp_gpu! =====

#
# Single-source queries from `sources` (k vertices): row t of X ← e_{sources[t]} A*.
# Same as rmul_gpu! on B = [e_{s₁}; …; e_{s_k}], but the U sweep walks only
# the k root paths instead of the whole tree.
#
function sssp_gpu!(X::CuMatrix{T}, G::GPUSLU{Sem, T, I}, sources::CuVector{<:Integer}; W::CuMatrix{T} = similar(X),
        M::CuMatrix{T} = CuMatrix{T}(undef, size(X, 1), G.maxna), nthreads::Int = 64, timer = nothing, permute::Bool = true) where {Sem, T, I}
    @assert size(X) == (length(sources), G.n)
    @assert size(W) == size(X)
    @assert permute || W === X

    s = G.s
    trans = Val(:N)
    scale = Val(!isintegral(s))
    nrhs = size(X, 1)
    tb = min(nthreads, 32 * cld(nrhs, 32))
    nb = cld(nrhs, tb)
    #
    #   W ← B Q⁻¹ U*,  B = [e_{s₁}; …; e_{s_k}]
    #
    @phase timer :fill fill!(W, szero(s, T, trans))
    @phase timer :U_path @cuda threads = tb blocks = nb upward_path_kernel!(s, trans, scale, W, sources, G.cinvp, G.idx, G.pnt, G.istop,
        G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.UDval, G.ULval)
    #
    #   the top of the tree, for all rows at once
    #
    kernel = @cuda launch = false upward_kernel!(s, trans, scale, W, G.top, 0, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.UDval, G.ULval)

    for l in 1:(length(G.topptr) - 1)
        strt = G.topptr[l]
        nfl = G.topptr[l + 1] - strt

        if ispositive(nfl)
            @phase timer :U_top_batched kernel(s, trans, scale, W, G.top, strt - 1, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.UDval, G.ULval; threads = tb, blocks = (nfl, nb))
        end

        for f in G.toplarge[l]
            @phase timer :U_top_dense upward_large!(G, W, M, f, trans, scale)
        end
    end
    #
    #   W ← W L*
    #
    downward_sweep!(G, W, M, trans, tb, nb, timer)
    #
    #   X ← W P⁻¹   (unless the result is wanted in elimination coordinates)
    #
    permute && @phase timer :permute permutecols_gpu!(X, W, G.rperm)
    return X
end

# ===== closure_gpu! =====

#
# The whole closure A* on the GPU, in elimination coordinates:
#
#   D[i, j] = A*[rperm[i], rperm[j]]     (symmetric pattern: rperm = cperm)
#
# Every vertex is a source, all at once (one block of n right-hand sides),
# with D itself as the work matrix: the U sweep walks n root paths, the top
# of the tree and the L sweep run with n rows, so the large fronts are big
# GEMMs. Reading D in the original labels is a permutation.
#
function closure_gpu!(D::CuMatrix{T}, G::GPUSLU{Sem, T, I}; timer = nothing, M::CuMatrix{T} = CuMatrix{T}(undef, G.n, G.maxna)) where {Sem, T, I}
    @assert size(D) == (G.n, G.n)
    sources = upload(Array(G.rperm))
    return sssp_gpu!(D, G, sources; W = D, M, timer, permute = false)
end

closure_gpu(G::GPUSLU{Sem, T}; kw...) where {Sem, T} = closure_gpu!(CuMatrix{T}(undef, G.n, G.n), G; kw...)

# ===== SSSPPlan =====

#
# sssp_gpu! for a fixed number k of sources, recorded once as a CUDA graph.
# A solve launches thousands of small kernels (one per level, several per
# large front); replaying the graph removes the host-side launch cost of
# each one. The buffers are owned by the plan: P(sources) overwrites P.X.
#
struct SSSPPlan{Sem, T, I}
    G::GPUSLU{Sem, T, I}
    X::CuMatrix{T}
    W::CuMatrix{T}
    M::CuMatrix{T}
    sources::CuVector{Int}
    exec::CuGraphExec
end

function SSSPPlan(G::GPUSLU{Sem, T, I}, k::Integer) where {Sem, T, I}
    X = CuMatrix{T}(undef, k, G.n)
    W = similar(X)
    M = CuMatrix{T}(undef, k, G.maxna)
    sources = CUDA.ones(Int, k)
    # compile every kernel before capturing
    sssp_gpu!(X, G, sources; W, M)
    CUDA.synchronize()
    graph = CUDA.capture() do
        sssp_gpu!(X, G, sources; W, M)
    end
    return SSSPPlan{Sem, T, I}(G, X, W, M, sources, CUDA.instantiate(graph))
end

function (P::SSSPPlan)(sources::AbstractVector{<:Integer})
    @assert length(sources) == length(P.sources)
    copyto!(P.sources, sources)
    CUDA.launch(P.exec)
    return P.X
end

function downward_sweep!(G::GPUSLU, W::CuMatrix, M::CuMatrix, trans::Val, tb::Int, nb::Int, timer)
    # function barrier: each kernel variant compiles its own sweep
    if DOWN_VARIANT[] === :simple
        return downward_sweep!(downward_kernel_simple!, G, W, M, trans, tb, nb, timer)
    else
        return downward_sweep!(downward_kernel_blocked!, G, W, M, trans, tb, nb, timer)
    end
end

function downward_sweep!(kf::F, G::GPUSLU, W::CuMatrix, M::CuMatrix, trans::Val, tb::Int, nb::Int, timer) where {F}
    s = G.s
    kernel = @cuda launch = false kf(s, trans, W, G.down, 0, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.LDval, G.LLval)

    for l in 1:(length(G.downptr) - 1)
        for f in G.downlarge[l]
            @phase timer :L_dense downward_large!(G, W, M, f, trans)
        end

        strt = G.downptr[l]
        nfl = G.downptr[l + 1] - strt

        if ispositive(nfl)
            @phase timer :L_batched kernel(s, trans, W, G.down, strt - 1, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.LDval, G.LLval; threads = tb, blocks = (nfl, nb))
        end
    end

    return W
end

ispositive(x) = x > zero(x)

# ===== large fronts =====
#
# Kernels on the default stream run in order, so a large front never runs
# concurrently with the batched kernel of its level, and its scatter needs
# no atomics.

#
#   C₁ ← C₁ U₁₁*
#   C₂ ← C₂ ⊕ C₁ U₁₂
#
function upward_large!(G::GPUSLU{<:Any, T}, C::CuMatrix{T}, M::CuMatrix{T}, f, trans::Val, scale::Val) where {T}
    s = G.s
    Rp = G.hRptr[f]; nn = G.hRptr[f + 1] - Rp
    Sp = G.hSptr[f]; na = G.hSptr[f + 1] - Sp
    Dp = G.hDptr[f]; Lp = G.hLptr[f]

    C₁ = view(C, :, Rp:(Rp + nn - 1))
    D₁₁ = reshape(view(G.UDval, Dp:(Dp + nn * nn - 1)), nn, nn)
    strsx_gpu!(s, trans, scale, Val(:U), C₁, D₁₁)

    if ispositive(na)
        U₁₂ = reshape(view(G.ULval, Lp:(Lp + nn * na - 1)), nn, na)
        M₂ = view(M, :, 1:na)
        fill!(M₂, szero(s, T, trans))
        sgemx_gpu!(s, M₂, C₁, U₁₂)
        scatteradd_gpu!(s, trans, C, M₂, G.Stgt, Sp)
    end

    return
end

#
#   C₁ ← C₁ ⊕ C₂ L₂₁
#   C₁ ← C₁ L₁₁*
#
function downward_large!(G::GPUSLU{<:Any, T}, C::CuMatrix{T}, M::CuMatrix{T}, f, trans::Val) where {T}
    s = G.s
    Rp = G.hRptr[f]; nn = G.hRptr[f + 1] - Rp
    Sp = G.hSptr[f]; na = G.hSptr[f + 1] - Sp
    Dp = G.hDptr[f]; Lp = G.hLptr[f]

    C₁ = view(C, :, Rp:(Rp + nn - 1))

    if ispositive(na)
        L₂₁ = reshape(view(G.LLval, Lp:(Lp + nn * na - 1)), na, nn)
        M₂ = view(M, :, 1:na)
        gather_gpu!(M₂, C, G.Stgt, Sp)
        sgemx_gpu!(s, C₁, M₂, L₂₁)
    end

    D₁₁ = reshape(view(G.LDval, Dp:(Dp + nn * nn - 1)), nn, nn)
    strsx_gpu!(s, trans, Val(false), Val(:L), C₁, D₁₁)
    return
end

# ===== strsx_gpu! =====

const STRSX_GPU_NB = 64

#
# Dense right triangular solve X ← X A*, blocked over 64-column diagonal
# blocks:
#
#   uplo = :U   forward,  X[:, j] ← (X[:, j] ⊕ Σ_{k<j} X[:, k] A[k, j]) A[j, j]*
#   uplo = :L   backward, X[:, j] ←  X[:, j] ⊕ Σ_{k>j} X[:, k] A[k, j]      (unit)
#
# A diagonal block is applied by inversion (diagonal-block inversion, as in
# GPU supernodal solvers): its closure T = A[J, J]* is formed by solving
# with the identity (one small kernel, b threads), and X[:, J] ← X[:, J] T is
# a semiring GEMM. The trailing update is a GEMM as well. Over exact
# arithmetic (tropical with integer values, bottleneck, Boolean) this is
# identical to substitution; with floating-point values it re-associates the
# products, so results may differ in the last bits.
# When X has few rows, substitution with one thread per row is used instead.
#
function strsx_gpu!(s::AbstractSemiring, trans::Val, scale::Val, uplo::Val{UPLO}, X::AbstractMatrix{T}, A::AbstractMatrix; nb::Int = STRSX_GPU_NB) where {UPLO, T}
    n = size(A, 1)
    m = size(X, 1)
    inv = m > STRSX_INV_MIN

    blocks = UPLO === :U ? [(j0, min(j0 + nb - 1, n)) for j0 in 1:nb:n] : [(max(j1 - nb + 1, 1), j1) for j1 in n:-nb:1]

    for (j0, j1) in blocks
        J = j0:j1
        b = j1 - j0 + 1

        if inv
            Tb, W = trsm_workspace(T, m * b)
            Tj = view(Tb, 1:b, 1:b)
            identity_gpu!(s, Tj)
            @cuda threads = 32 * cld(b, 32) strsx_diag_kernel!(s, trans, scale, uplo, Tj, view(A, J, J))
            Wj = reshape(view(W, 1:(m * b)), m, b)
            fill!(Wj, szero(s, T, Val(:N)))
            sgemx_gpu!(s, Wj, view(X, :, J), Tj)
            copy_gpu!(view(X, :, J), Wj)
        else
            tb = min(128, 32 * cld(m, 32))
            @cuda threads = tb blocks = cld(m, tb) strsx_diag_kernel!(s, trans, scale, uplo, view(X, :, J), view(A, J, J))
        end

        if UPLO === :U && j1 < n
            sgemx_gpu!(s, view(X, :, (j1 + 1):n), view(X, :, J), view(A, J, (j1 + 1):n))
        elseif UPLO === :L && j0 > 1
            sgemx_gpu!(s, view(X, :, 1:(j0 - 1)), view(X, :, J), view(A, J, 1:(j0 - 1)))
        end
    end

    return X
end

const STRSX_INV_MIN = 64

# Scratch for diagonal-block inversion, per element type and stream (so
# that concurrent streams never share it). It only grows outside stream
# capture (plans warm up before capturing).
const TRSM_WS = Dict{Tuple{DataType, UInt}, Tuple{CuMatrix, CuVector}}()

function trsm_workspace(::Type{T}, len::Int) where {T}
    key = (T, UInt(CUDA.stream().handle))
    Tb, W = get(TRSM_WS, key, (nothing, nothing))

    if isnothing(Tb) || length(W) < len
        @assert !CUDA.is_capturing() "trsm workspace must grow before graph capture"
        Tb = CuMatrix{T}(undef, STRSX_GPU_NB, STRSX_GPU_NB)
        W = CuVector{T}(undef, max(len, isnothing(W) ? 0 : 2 * length(W)))
        CUDA.enable_synchronization!(Tb, false)
        CUDA.enable_synchronization!(W, false)
        TRSM_WS[key] = (Tb, W)
    end

    return Tb::CuMatrix{T}, W::CuVector{T}
end

# X ← semiring identity
function identity_gpu!(s::AbstractSemiring, X::AbstractMatrix{T}) where {T}
    function kernel(s, X)
        e = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        m = size(X, 1)

        if e <= length(X)
            i = (e - 1) % m + 1
            j = (e - 1) ÷ m + 1
            @inbounds X[i, j] = i == j ? sone(s, T, Val(:N)) : szero(s, T, Val(:N))
        end

        return
    end

    @cuda threads = 256 blocks = cld(length(X), 256) kernel(s, X)
    return X
end

# X ← Y (both may be strided views)
function copy_gpu!(X::AbstractMatrix, Y::AbstractMatrix)
    function kernel(X, Y)
        e = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        m = size(X, 1)

        if e <= length(X)
            i = (e - 1) % m + 1
            j = (e - 1) ÷ m + 1
            @inbounds X[i, j] = Y[i, j]
        end

        return
    end

    @cuda threads = 256 blocks = cld(length(X), 256) kernel(X, Y)
    return X
end

function strsx_diag_kernel!(s::AbstractSemiring, trans::Val, ::Val{SCALE}, ::Val{UPLO}, X::AbstractMatrix, A::AbstractMatrix) where {SCALE, UPLO}
    t = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n = size(A, 1)

    if t > size(X, 1)
        return
    end

    @inbounds if UPLO === :U
        for j in 1:n
            acc = X[t, j]

            for k in 1:(j - 1)
                acc = smuladd(s, X[t, k], A[k, j], acc, Val(:N), trans)
            end

            if SCALE
                acc = sprod(s, acc, sstar(s, A[j, j]), Val(:N), trans)
            end

            X[t, j] = acc
        end
    else
        for j in n:-1:1
            acc = X[t, j]

            for k in (j + 1):n
                acc = smuladd(s, X[t, k], A[k, j], acc, Val(:N), trans)
            end

            X[t, j] = acc
        end
    end

    return
end

# C[:, Stgt[Sp + r - 1]] ← C[:, Stgt[Sp + r - 1]] ⊕ M[:, r]
function scatteradd_gpu!(s::AbstractSemiring, trans::Val, C::AbstractMatrix, M::AbstractMatrix, Stgt::CuVector, Sp)
    function kernel(s, trans, C, M, Stgt, Sp)
        t = threadIdx().x + (blockIdx().y - 1) * blockDim().x
        r = blockIdx().x

        if t <= size(M, 1)
            @inbounds begin
                j = Stgt[Sp + r - 1]
                C[t, j] = splus(s, C[t, j], M[t, r], trans)
            end
        end

        return
    end

    tb = min(256, 32 * cld(size(M, 1), 32))
    @cuda threads = tb blocks = (size(M, 2), cld(size(M, 1), tb)) kernel(s, trans, C, M, Stgt, Sp)
    return C
end

# M[:, r] ← C[:, Stgt[Sp + r - 1]]
function gather_gpu!(M::AbstractMatrix, C::AbstractMatrix, Stgt::CuVector, Sp)
    function kernel(M, C, Stgt, Sp)
        t = threadIdx().x + (blockIdx().y - 1) * blockDim().x
        r = blockIdx().x

        if t <= size(M, 1)
            @inbounds M[t, r] = C[t, Stgt[Sp + r - 1]]
        end

        return
    end

    tb = min(256, 32 * cld(size(M, 1), 32))
    @cuda threads = tb blocks = (size(M, 2), cld(size(M, 1), tb)) kernel(M, C, Stgt, Sp)
    return M
end

# dst[:, perm[j]] = src[:, j], as permutecols! on the CPU
function permutecols_gpu!(dst::CuMatrix, src::CuMatrix, perm::CuVector)
    function kernel(dst, src, perm)
        t = threadIdx().x + (blockIdx().y - 1) * blockDim().x
        j = blockIdx().x

        if t <= size(src, 1)
            @inbounds dst[t, perm[j]] = src[t, j]
        end

        return
    end

    tb = min(256, 32 * cld(size(src, 1), 32))
    @cuda threads = tb blocks = (size(src, 2), cld(size(src, 1), tb)) kernel(dst, src, perm)
    return dst
end

# ===== kernels =====

#
# One block per (front f, chunk of right-hand sides); thread t owns row t.
#
#   C₁ ← C₁ U₁₁*
#   C₂ ← C₂ ⊕ C₁ U₁₂       (atomic)
#
function upward_kernel!(s::AbstractSemiring, trans::Val, scale::Val, C::AbstractMatrix{T}, order, off::Int,
        Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval) where {T}
    t = threadIdx().x + (blockIdx().y - 1) * blockDim().x

    if t <= size(C, 1)
        f = @inbounds order[off + blockIdx().x]
        upward_front!(s, trans, scale, C, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, Val(true))
    end

    return
end

#
# Row t of C through front f of the U sweep. ATOMIC selects an atomic
# scatter (other threads may update the same entries) or a plain one
# (this thread owns row t).
#
@inline function upward_front!(s::AbstractSemiring, trans::Val, ::Val{SCALE}, C::AbstractMatrix{T}, t, f,
        Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, ::Val{ATOMIC}) where {SCALE, T, ATOMIC}
    @inbounds begin
        Rp = Rptr[f]; nn = Rptr[f + 1] - Rp
        Sp = Sptr[f]; na = Sptr[f + 1] - Sp
        Dp = Dptr[f]; Lp = Lptr[f]

        for j in 1:nn
            acc = C[t, Rp + j - 1]

            for k in 1:(j - 1)
                acc = smuladd(s, C[t, Rp + k - 1], Dval[Dp + (j - 1) * nn + k - 1], acc, Val(:N), trans)
            end

            if SCALE
                acc = sprod(s, acc, sstar(s, Dval[Dp + (j - 1) * nn + j - 1]), Val(:N), trans)
            end

            C[t, Rp + j - 1] = acc
        end

        for r in 1:na
            m = szero(s, T, trans)

            for j in 1:nn
                m = smuladd(s, C[t, Rp + j - 1], Lval[Lp + (r - 1) * nn + j - 1], m, Val(:N), trans)
            end

            c = Stgt[Sp + r - 1]

            if ATOMIC
                atomic_splus!(s, trans, C, t, c, m)
            else
                C[t, c] = splus(s, C[t, c], m, trans)
            end
        end
    end

    return
end

#
# Path walk for unit right-hand sides: row t of C is zero except for a one
# at vertex v, so the U sweep only touches the fronts on the path from the
# front of v to its root (the reach of e_v, the upward search space of a
# contraction-hierarchy query). One thread per row walks its own path and
# owns its row, so no atomics are needed.
#
function upward_path_kernel!(s::AbstractSemiring, trans::Val, scale::Val, C::AbstractMatrix{T}, sources, cinvp, idx, pnt, istop,
        Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval) where {T}
    t = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if t <= size(C, 1)
        @inbounds begin
            v = cinvp[sources[t]]
            C[t, v] = sone(s, T, trans)
            f = idx[v]

            while !iszero(f) && !istop[f]
                upward_front!(s, trans, scale, C, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, Val(false))
                f = pnt[f]
            end
        end
    end

    return
end

#
#   C₁ ← C₁ ⊕ C₂ L₂₁
#   C₁ ← C₁ L₁₁*
#
function downward_large!(G::GPUSLU{<:Any, T}, C::CuMatrix{T}, M::CuMatrix{T}, f, trans::Val) where {T}
    s = G.s
    Rp = G.hRptr[f]; nn = G.hRptr[f + 1] - Rp
    Sp = G.hSptr[f]; na = G.hSptr[f + 1] - Sp
    Dp = G.hDptr[f]; Lp = G.hLptr[f]

    C₁ = view(C, :, Rp:(Rp + nn - 1))

    if ispositive(na)
        L₂₁ = reshape(view(G.LLval, Lp:(Lp + nn * na - 1)), na, nn)
        M₂ = view(M, :, 1:na)
        gather_gpu!(M₂, C, G.Stgt, Sp)
        sgemx_gpu!(s, C₁, M₂, L₂₁)
    end

    D₁₁ = reshape(view(G.LDval, Dp:(Dp + nn * nn - 1)), nn, nn)
    strsx_gpu!(s, trans, Val(false), Val(:L), C₁, D₁₁)
    return
end

# ===== strsx_gpu! =====

const STRSX_GPU_NB = 64

#
# Dense right triangular solve X ← X A*, blocked: a diagonal block is
# solved with one thread per row of X, and its contribution to the rest
# of X is a semiring GEMM.
#
#   uplo = :U   forward,  X[:, j] ← (X[:, j] ⊕ Σ_{k<j} X[:, k] A[k, j]) A[j, j]*
#   uplo = :L   backward, X[:, j] ←  X[:, j] ⊕ Σ_{k>j} X[:, k] A[k, j]      (unit)
#
function strsx_gpu!(s::AbstractSemiring, trans::Val, scale::Val, uplo::Val{UPLO}, X::AbstractMatrix, A::AbstractMatrix; nb::Int = STRSX_GPU_NB) where {UPLO}
    n = size(A, 1)
    m = size(X, 1)
    tb = min(128, 32 * cld(m, 32))

    if UPLO === :U
        for j0 in 1:nb:n
            j1 = min(j0 + nb - 1, n)
            J = j0:j1
            @cuda threads = tb blocks = cld(m, tb) strsx_diag_kernel!(s, trans, scale, uplo, view(X, :, J), view(A, J, J))

            if j1 < n
                sgemx_gpu!(s, view(X, :, (j1 + 1):n), view(X, :, J), view(A, J, (j1 + 1):n))
            end
        end
    else
        for j1 in n:-nb:1
            j0 = max(j1 - nb + 1, 1)
            J = j0:j1
            @cuda threads = tb blocks = cld(m, tb) strsx_diag_kernel!(s, trans, scale, uplo, view(X, :, J), view(A, J, J))

            if j0 > 1
                sgemx_gpu!(s, view(X, :, 1:(j0 - 1)), view(X, :, J), view(A, J, 1:(j0 - 1)))
            end
        end
    end

    return X
end

function strsx_diag_kernel!(s::AbstractSemiring, trans::Val, ::Val{SCALE}, ::Val{UPLO}, X::AbstractMatrix, A::AbstractMatrix) where {SCALE, UPLO}
    t = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n = size(A, 1)

    if t > size(X, 1)
        return
    end

    @inbounds if UPLO === :U
        for j in 1:n
            acc = X[t, j]

            for k in 1:(j - 1)
                acc = smuladd(s, X[t, k], A[k, j], acc, Val(:N), trans)
            end

            if SCALE
                acc = sprod(s, acc, sstar(s, A[j, j]), Val(:N), trans)
            end

            X[t, j] = acc
        end
    else
        for j in n:-1:1
            acc = X[t, j]

            for k in (j + 1):n
                acc = smuladd(s, X[t, k], A[k, j], acc, Val(:N), trans)
            end

            X[t, j] = acc
        end
    end

    return
end

# C[:, Stgt[Sp + r - 1]] ← C[:, Stgt[Sp + r - 1]] ⊕ M[:, r]
function scatteradd_gpu!(s::AbstractSemiring, trans::Val, C::AbstractMatrix, M::AbstractMatrix, Stgt::CuVector, Sp)
    function kernel(s, trans, C, M, Stgt, Sp)
        t = threadIdx().x + (blockIdx().y - 1) * blockDim().x
        r = blockIdx().x

        if t <= size(M, 1)
            @inbounds begin
                j = Stgt[Sp + r - 1]
                C[t, j] = splus(s, C[t, j], M[t, r], trans)
            end
        end

        return
    end

    tb = min(256, 32 * cld(size(M, 1), 32))
    @cuda threads = tb blocks = (size(M, 2), cld(size(M, 1), tb)) kernel(s, trans, C, M, Stgt, Sp)
    return C
end

# M[:, r] ← C[:, Stgt[Sp + r - 1]]
function gather_gpu!(M::AbstractMatrix, C::AbstractMatrix, Stgt::CuVector, Sp)
    function kernel(M, C, Stgt, Sp)
        t = threadIdx().x + (blockIdx().y - 1) * blockDim().x
        r = blockIdx().x

        if t <= size(M, 1)
            @inbounds M[t, r] = C[t, Stgt[Sp + r - 1]]
        end

        return
    end

    tb = min(256, 32 * cld(size(M, 1), 32))
    @cuda threads = tb blocks = (size(M, 2), cld(size(M, 1), tb)) kernel(M, C, Stgt, Sp)
    return M
end

# dst[:, perm[j]] = src[:, j], as permutecols! on the CPU
function permutecols_gpu!(dst::CuMatrix, src::CuMatrix, perm::CuVector)
    function kernel(dst, src, perm)
        t = threadIdx().x + (blockIdx().y - 1) * blockDim().x
        j = blockIdx().x

        if t <= size(src, 1)
            @inbounds dst[t, perm[j]] = src[t, j]
        end

        return
    end

    tb = min(256, 32 * cld(size(src, 1), 32))
    @cuda threads = tb blocks = (size(src, 2), cld(size(src, 1), tb)) kernel(dst, src, perm)
    return dst
end

# ===== kernels =====

#
# One block per (front f, chunk of right-hand sides); thread t owns row t.
#
#   C₁ ← C₁ U₁₁*
#   C₂ ← C₂ ⊕ C₁ U₁₂       (atomic)
#
function upward_kernel!(s::AbstractSemiring, trans::Val, ::Val{SCALE}, C::AbstractMatrix{T}, order, off::Int,
        Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval) where {SCALE, T}
    t = threadIdx().x + (blockIdx().y - 1) * blockDim().x

    if t > size(C, 1)
        return
    end

    @inbounds begin
        f = order[off + blockIdx().x]
        Rp = Rptr[f]; nn = Rptr[f + 1] - Rp
        Sp = Sptr[f]; na = Sptr[f + 1] - Sp
        Dp = Dptr[f]; Lp = Lptr[f]

        for j in 1:nn
            acc = C[t, Rp + j - 1]

            for k in 1:(j - 1)
                acc = smuladd(s, C[t, Rp + k - 1], Dval[Dp + (j - 1) * nn + k - 1], acc, Val(:N), trans)
            end

            if SCALE
                acc = sprod(s, acc, sstar(s, Dval[Dp + (j - 1) * nn + j - 1]), Val(:N), trans)
            end

            C[t, Rp + j - 1] = acc
        end

        for r in 1:na
            m = szero(s, T, trans)

            for j in 1:nn
                m = smuladd(s, C[t, Rp + j - 1], Lval[Lp + (r - 1) * nn + j - 1], m, Val(:N), trans)
            end

            atomic_splus!(s, trans, C, t, Stgt[Sp + r - 1], m)
        end
    end

    return
end

#
#   C₁ ← C₁ ⊕ C₂ L₂₁
#   C₁ ← C₁ L₁₁*           (unit diagonal)
#
# Up to DOWN_NB residual columns of row t are kept in registers: each
# gathered separator value C[t, sep[r]] is loaded once for all of them, and
# when nn ≤ DOWN_NB the unit-lower solve with L₁₁ runs in registers too.
#
const DOWN_NB = 8

# :blocked (register chunks) or :simple (one residual column at a time)
const DOWN_VARIANT = Ref(:simple)


function downward_kernel_simple!(s::AbstractSemiring, trans::Val, C::AbstractMatrix{T}, order, off::Int,
        Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval) where {T}
    t = threadIdx().x + (blockIdx().y - 1) * blockDim().x

    if t > size(C, 1)
        return
    end

    @inbounds begin
        f = order[off + blockIdx().x]
        Rp = Rptr[f]; nn = Rptr[f + 1] - Rp
        Sp = Sptr[f]; na = Sptr[f + 1] - Sp
        Dp = Dptr[f]; Lp = Lptr[f]

        for j in 1:nn
            acc = C[t, Rp + j - 1]

            for r in 1:na
                acc = smuladd(s, C[t, Stgt[Sp + r - 1]], Lval[Lp + (j - 1) * na + r - 1], acc, Val(:N), trans)
            end

            C[t, Rp + j - 1] = acc
        end

        for j in nn:-1:1
            acc = C[t, Rp + j - 1]

            for k in (j + 1):nn
                acc = smuladd(s, C[t, Rp + k - 1], Dval[Dp + (j - 1) * nn + k - 1], acc, Val(:N), trans)
            end

            C[t, Rp + j - 1] = acc
        end
    end

    return
end

function downward_kernel_blocked!(s::AbstractSemiring, trans::Val, C::AbstractMatrix{T}, order, off::Int,
        Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval) where {T}
    t = threadIdx().x + (blockIdx().y - 1) * blockDim().x

    if t > size(C, 1)
        return
    end

    @inbounds begin
        f = order[off + blockIdx().x]
        Rp = Rptr[f]; nn = Rptr[f + 1] - Rp
        Sp = Sptr[f]; na = Sptr[f + 1] - Sp
        Dp = Dptr[f]; Lp = Lptr[f]
        # the chunk width is uniform over the block (one front per block), so no divergence
        if nn == 1
            downward_chunks!(s, trans, C, t, Rp, nn, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(1))
        elseif nn == 2
            downward_chunks!(s, trans, C, t, Rp, nn, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(2))
        elseif nn <= 4
            downward_chunks!(s, trans, C, t, Rp, nn, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(4))
        else
            downward_chunks!(s, trans, C, t, Rp, nn, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(DOWN_NB))
        end

        if nn > DOWN_NB
            for j in nn:-1:1
                acc = C[t, Rp + j - 1]

                for k in (j + 1):nn
                    acc = smuladd(s, C[t, Rp + k - 1], Dval[Dp + (j - 1) * nn + k - 1], acc, Val(:N), trans)
                end

                C[t, Rp + j - 1] = acc
            end
        end
    end

    return
end

@inline function downward_chunks!(s, trans, C::AbstractMatrix{T}, t, Rp, nn, Sp, na, Dp, Lp, Stgt, Dval, Lval, ::Val{NB}) where {T, NB}
    z = szero(s, T, Val(:N))

    @inbounds for j0 in 1:NB:nn
        acc = load_chunk(C, t, Rp + j0 - 1, nn - j0 + 1, z, Val(NB))

        for r in 1:na
            c = C[t, Stgt[Sp + r - 1]]
            acc = gemv_chunk(s, trans, acc, c, Lval, Lp + (j0 - 1) * na + r - 1, na, nn - j0 + 1, Val(NB))
        end

        if nn <= NB
            acc = trsm_chunk(s, trans, acc, Dval, Dp, nn, Val(NB))
        end

        store_chunk!(C, t, Rp + j0 - 1, nn - j0 + 1, acc, Val(NB))
    end

    return
end

@generated function load_chunk(C, t, c0, len, z, ::Val{NB}) where {NB}
    return :($(Expr(:meta, :inline)); @inbounds ($((:($i <= len ? C[t, c0 + $(i - 1)] : z) for i in 1:NB)...),))
end

@generated function store_chunk!(C, t, c0, len, acc, ::Val{NB}) where {NB}
    stores = [:($i <= len && (C[t, c0 + $(i - 1)] = acc[$i])) for i in 1:NB]
    return :($(Expr(:meta, :inline)); @inbounds begin $(stores...) end; nothing)
end

# acc[i] ← acc[i] ⊕ c L[base + (i - 1) na]   for i ≤ len
@generated function gemv_chunk(s, trans, acc::NTuple{NB}, c, Lval, base, na, len, ::Val{NB}) where {NB}
    return :($(Expr(:meta, :inline)); @inbounds ($((:($i <= len ? smuladd(s, c, Lval[base + $(i - 1) * na], acc[$i], Val(:N), trans) : acc[$i]) for i in 1:NB)...),))
end

# backward unit-lower solve on the first nn entries: acc[j] ← acc[j] ⊕ Σ_{k>j} acc[k] D[k, j]
@generated function trsm_chunk(s, trans, acc::NTuple{NB}, Dval, Dp, nn, ::Val{NB}) where {NB}
    names = [Symbol(:a, i) for i in 1:NB]
    body = Expr[]
    push!(body, :(($(names...),) = acc))

    for j in NB:-1:1, k in (j + 1):NB
        push!(body, :(if $k <= nn
            $(names[j]) = smuladd(s, $(names[k]), Dval[Dp + ($j - 1) * nn + $k - 1], $(names[j]), Val(:N), trans)
        end))
    end

    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $(body...)
        end
        return ($(names...),)
    end
end

#
#   C[i, j] ← C[i, j] ⊕ m, atomically
#
@inline function atomic_splus!(s::AbstractSemiring, trans::Val, C::CuDeviceMatrix{T}, i, j, m::T) where {T}
    U = sizeof(T) == 4 ? UInt32 : UInt64
    ptr = reinterpret(Core.LLVMPtr{U, CUDA.AS.Global}, pointer(C, i + (j - 1) * size(C, 1)))
    old = @inbounds C[i, j]

    while true
        new = splus(s, old, m, trans)

        if new === old
            return
        end

        prev = CUDA.atomic_cas!(ptr, reinterpret(U, old), reinterpret(U, new))

        if prev == reinterpret(U, old)
            return
        end

        old = reinterpret(T, prev)
    end
end
