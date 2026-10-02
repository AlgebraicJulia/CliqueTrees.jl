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
    ops::Base.RefValue{Any}     # precomputed operators of the large fronts (precompute_ops!), or nothing
    cache::Dict{Symbol, Any}    # lazily built schedules (e.g. the persistent sweep plan)
end

#
# A front is large when its solve work nn (nn + na) per right-hand side
# reaches `large`: one thread per right-hand side would then serialize too
# much, so it is solved with tiled dense kernels instead.
#
function GPUSLU(F::ChordalSLU{Sem, T, I}; large::Integer = 2048, factor = nothing, amalgamate::Integer = config().merge, alpha::Real = config().merge_alpha) where {Sem, T, I}
    if !iszero(MF.ne(F.S.N))
        error("GPUSLU: coupling between strongly connected components (a directed graph whose components " *
              "reach one another) is not supported on the GPU yet; use the CPU solver")
    end

    if !(isbitstype(T) && sizeof(T) in (4, 8))
        error("GPUSLU: element type $T is not supported on the GPU (the atomic ⊕ needs 4- or 8-byte isbits elements)")
    end

    check_semiring(F.s, T)

    S = F.S.S
    n = size(F, 1)
    nf = Int(MF.nv(S.res))
    pnt = Vector{I}(view(S.pnt, 1:nf))
    idx = Vector{I}(view(S.idx, 1:n))
    Rptr = Vector{I}(view(MF.pointers(S.res), 1:(nf + 1)))
    Sptr = Vector{I}(view(MF.pointers(S.sep), 1:(nf + 1)))
    Stgt = Vector{I}(view(MF.targets(S.sep), 1:(Sptr[end] - 1)))
    Dptr = Vector{I}(view(S.Dptr, 1:(nf + 1)))
    Lptr = Vector{I}(view(S.Lptr, 1:(nf + 1)))

    if amalgamate > 1
        #
        # merge chains of small fronts for the solve (see amalgamate.jl): the merge is cached per
        # symbolic factorization, and the factor is rearranged on the GPU
        #
        A = amalgamation(amalgamation_key(S.Dptr), I, amalgamate, alpha, Rptr, Sptr, Stgt, Dptr, Lptr, pnt, idx)
        if !isnothing(A)
            vals = isnothing(factor) ? (upload(F.LDval), upload(F.LLval), upload(F.UDval), upload(F.ULval)) : factor
            factor = amalgamate_values(A, F.s, vals...)
            nf = A.nf; pnt = A.pnt; idx = A.idx
            Rptr = A.Rptr; Sptr = A.Sptr; Stgt = A.Stgt; Dptr = A.Dptr; Lptr = A.Lptr
        end
    end
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
        upload(Rptr), upload(Sptr), upload(Stgt), upload(Dptr), upload(Lptr),
        Rptr, Sptr, Dptr, Lptr,
        (isnothing(factor) ? (upload(F.LDval), upload(F.LLval), upload(F.UDval), upload(F.ULval)) : factor)...,
        CuVector(up), upptr, uplarge, CuVector(down), downptr, downlarge, maxna,
        upload(idx), upload(pnt),
        CuVector(Vector{Bool}(istop)), CuVector(top), topptr, toplarge,
        upload(F.cinvp), upload(F.rperm), Ref{Any}(nothing), Dict{Symbol, Any}(),
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
            @phase timer :U_batched kernel(s, trans, scale, W, G.up, strt - 1, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.UDval, G.ULval; threads = tb, blocks = (nfl, rhs_blocks(nrhs, tb)))
        end

        for f in G.uplarge[l]
            @phase timer :U_dense upward_large!(G, W, M, f, trans, scale)
        end
    end
    #
    #   W ← W L*
    #
    downward_sweep!(G, W, M, trans, tb, rhs_blocks(nrhs, tb), timer)
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
    skip = config().skip_fill && rowmajor_path(W)

    if skip
        @phase timer :fill fill_top!(W, G)
    else
        @phase timer :fill fill!(W, szero(s, T, trans))
    end

    @phase timer :U_path @cuda threads = 128 blocks = cld(32 * nrhs, 128) upward_path_warp_kernel!(s, trans, scale, W, sources, G.cinvp, G.idx, G.pnt, G.istop,
        G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.UDval, G.ULval, Val(skip))
    #
    #   the top of the tree, for all rows at once
    #
    kernel = @cuda launch = false upward_kernel!(s, trans, scale, W, G.top, 0, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.UDval, G.ULval)

    for l in 1:(length(G.topptr) - 1)
        strt = G.topptr[l]
        nfl = G.topptr[l + 1] - strt

        if ispositive(nfl)
            @phase timer :U_top_batched kernel(s, trans, scale, W, G.top, strt - 1, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.UDval, G.ULval; threads = tb, blocks = (nfl, rhs_blocks(nrhs, tb)))
        end

        for f in G.toplarge[l]
            @phase timer :U_top_dense upward_large!(G, W, M, f, trans, scale)
        end
    end
    #
    #   W ← W L*
    #
    downward_sweep!(G, W, M, trans, tb, rhs_blocks(nrhs, tb), timer; zr = skip ? sources : nothing)
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
function closure_gpu!(D::CuMatrix{T}, G::GPUSLU{Sem, T, I}; timer = nothing, M::CuMatrix{T} = (need_memory(G.n * G.maxna * sizeof(T), "closure workspace"); CuMatrix{T}(undef, G.n, G.maxna))) where {Sem, T, I}
    @assert size(D) == (G.n, G.n)
    sources = upload(Array(G.rperm))
    return sssp_gpu!(D, G, sources; W = D, M, timer, permute = false)
end

function closure_gpu(G::GPUSLU{Sem, T}; kw...) where {Sem, T}
    need_memory(G.n^2 * sizeof(T), "the $(G.n) × $(G.n) closure")
    return closure_gpu!(CuMatrix{T}(undef, G.n, G.n), G; kw...)
end

#
# Fail early, with the sizes, when an allocation cannot fit (instead of an
# out-of-memory error deep inside a sweep).
#
function need_memory(bytes::Integer, what::AbstractString)
    free = CUDA.free_memory()

    if bytes > free                     # memory cached by CUDA.jl's pool is not free: release it and look again
        GC.gc(false); CUDA.reclaim()
        free = CUDA.free_memory()
    end

    if bytes > free
        error("not enough GPU memory for $what: needs $(round(bytes / 2^30; digits = 2)) GiB, " *
              "$(round(free / 2^30; digits = 2)) GiB available; use smaller blocks of right-hand sides (sssp_gpu!)")
    end

    return
end

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
    exec = capture_graph(() -> sssp_gpu!(X, G, sources; W, M))
    return SSSPPlan{Sem, T, I}(G, X, W, M, sources, exec)
end

function (P::SSSPPlan)(sources::AbstractVector{<:Integer})
    @assert length(sources) == length(P.sources)
    copyto!(P.sources, sources)
    CUDA.launch(P.exec)
    return P.X
end

# zr: the sources of the rows of W when W was not filled (skip_fill), so that the sweep below the top
# treats the residual entries of fronts off a row's root path as the semiring zero; nothing otherwise
function downward_sweep!(G::GPUSLU, W::CuMatrix, M::CuMatrix, trans::Val, tb::Int, nb::Int, timer; zr = nothing)
    s = G.s
    kernel = @cuda launch = false downward_kernel_reg!(s, trans, W, G.down, 0, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.LDval, G.LLval)
    @assert isnothing(zr) || rowmajor_path(W)
    skip = !isnothing(zr)
    zargs = skip ? (Val(true), zr, G.cinvp, G.idx, first_descendants(G)) : (Val(false), nothing, nothing, nothing, nothing)

    if rowmajor_path(W)
        #
        # the top of the tree level by level (large fronts dense, small batched), then everything
        # below it as one layered launch per layer (layered.jl)
        #
        plan = sweep_plan(G)

        for (small, large) in plan.toplevels
            for f in large
                @phase timer :L_dense downward_large!(G, W, M, f, trans)
            end

            if ispositive(length(small))
                @phase timer :L_batched kernel(s, trans, W, small, 0, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.LDval, G.LLval; threads = tb, blocks = (length(small), nb))
            end
        end

        if ispositive(plan.nrest)
            sp = config().layer_cache ? slot_plan(G, slot_count(eltype(W), SLOT_TB * SLOT_Q), SLOT_TB * SLOT_Q) : nothing
            isnothing(sp) || return layered_slot_sweep!(G, W, sp, trans, timer, zr)

            for (regptr, regfronts, nreg) in layer_plan(G).layers
                @phase timer :L_layered @cuda threads = ROWMAJOR_TB blocks = (cld(size(W, 1), ROWMAJOR_TB), nreg) maxregs = ROWMAJOR_MAXREGS layered_down_kernel!(s, trans, W, regptr, regfronts,
                    G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.LDval, G.LLval, zargs...)
            end
        end

        return W
    end
    #
    # few right-hand sides: every level of the tree is one batched launch
    #
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

# ===== sweep plan =====
#
# The fronts of the top of the tree by depth (small ones batched, large ones dense), and how many
# fronts lie below it (the layered part of the L sweep).
#
struct SweepPlan{I}
    toplevels::Vector{Tuple{CuVector{I}, Vector{I}}}   # by depth: (small top fronts, large fronts)
    nrest::Int
end

function sweep_plan(G::GPUSLU{<:Any, <:Any, I}) where {I}
    get!(G.cache, :sweep) do
        istop = Array(G.istop)
        down = Array(G.down)
        toplevels = Tuple{CuVector{I}, Vector{I}}[]
        nrest = 0

        for l in 1:(length(G.downptr) - 1)
            fs = down[G.downptr[l]:(G.downptr[l + 1] - 1)]
            small = filter(f -> istop[f], fs)
            large = filter(f -> istop[f], G.downlarge[l])
            nrest += count(f -> !istop[f], fs)
            @assert all(f -> istop[f], G.downlarge[l])
            (isempty(small) && isempty(large)) || push!(toplevels, (CuVector{I}(small), large))
        end

        SweepPlan{I}(toplevels, nrest)
    end
end

@inline function downward_front!(s::AbstractSemiring, trans::Val, C::AbstractMatrix{T}, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, zi::Bool = false) where {T}
    @inbounds begin
        Rp = Rptr[f]; nn = Rptr[f + 1] - Rp
        Sp = Sptr[f]; na = Sptr[f + 1] - Sp
        Dp = Dptr[f]; Lp = Lptr[f]

        for j in 1:nn
            acc = zi ? szero(s, T, trans) : C[t, Rp + j - 1]

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

# ===== row-major L sweep =====
#
# Below the top of the tree, every row of the L sweep is independent, so one thread can carry its
# row through all remaining fronts in reverse postorder (parents before children, depth-first),
# in one launch with no level barriers. A child then reads its parent's freshly written columns
# from L1/L2 instead of DRAM (in level order they are re-read a level later, after eviction).
# Same per-front operations as the level kernels; worth it when there are many rows (the closure).
#
const ROWMAJOR_TB = 128
const ROWMAJOR_MAXREGS = 48           # 32–64 are within ±3% on L4, RTX PRO 6000, B200 (the sweep is bound by memory)
rowmajor_path(W::AbstractMatrix) = size(W, 1) >= config().layered_min_rows

# ===== skipping the fill of W =====
#
# After the U sweep, row t of W is nonzero only in the columns of the fronts on the root path of its
# source's front ft (the U sweep only touches those). So instead of filling all of W with the semiring
# zero, fill the columns of the top of the tree (where every row is processed densely), let the path
# walk zero its own path columns, and let the L sweep below the top treat the residual entries of a
# front f that is not an ancestor of ft as zero without reading them. f is an ancestor of ft (or ft)
# iff fd[f] ≤ ft ≤ f, with fd[f] the first front of f's subtree in postorder. Every entry of W is
# still written by the L sweep, with the same operations: the result is bit-identical, and one write
# of W (the fill) and most reads of residual entries are saved.

function first_descendants(G::GPUSLU{<:Any, <:Any, I}) where {I}
    get!(G.cache, :firstdesc) do
        pnt = Array(G.pnt)
        fsz = ones(Int, G.nf)

        for f in 1:G.nf
            p = pnt[f]
            iszero(p) || (fsz[p] += fsz[f])
        end

        upload(I[f - fsz[f] + 1 for f in 1:G.nf])
    end
end

function top_columns(G::GPUSLU)
    get!(G.cache, :topcols) do
        istop = Array(G.istop)
        upload(Int32[c for f in 1:G.nf if istop[f] for c in G.hRptr[f]:(G.hRptr[f + 1] - 1)])
    end
end

function fill_top_kernel!(W, cols, z)
    t = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = blockIdx().y

    @inbounds while j <= length(cols)
        t <= size(W, 1) && (W[t, cols[j]] = z)
        j += gridDim().y
    end

    return
end

function fill_top!(W::CuMatrix{T}, G::GPUSLU) where {T}
    cols = top_columns(G)
    launch2d(fill_top_kernel!, size(W, 1), length(cols), W, cols, szero(G.s, T, Val(:N)))
    return W
end
# at most 48 registers: 10 blocks of 128 per SM instead of 9 at 56 (on sm_120: 33k rows in one wave, not 30k)
# ===== warp-cooperative path walk =====
#
# One warp per source instead of one thread: lane 0 solves with the
# (small) U₁₁ of each front on the path, then the 32 lanes split the
# separator update C[t, sep] ⊕= C₁ U₁₂ (distinct columns of the same row,
# so no conflicts). Same operations as upward_path_kernel!.
#

function upward_path_warp_kernel!(s::AbstractSemiring, trans::Val, ::Val{SCALE}, C::AbstractMatrix{T}, sources, cinvp, idx, pnt, istop,
        Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, ::Val{ZERO} = Val(false)) where {SCALE, T, ZERO}
    lane = (threadIdx().x - 1) % 32
    t = ((blockIdx().x - 1) * blockDim().x + threadIdx().x - 1) ÷ 32 + 1

    if t > size(C, 1)
        return
    end

    @inbounds begin
        v = cinvp[sources[t]]

        if ZERO                                 # W was not filled: zero this row's columns of the path below the top
            g = idx[v]

            while !iszero(g) && !istop[g]
                Rp = Rptr[g]; nn = Rptr[g + 1] - Rp
                j = lane

                while j < nn
                    C[t, Rp + j] = szero(s, T, trans)
                    j += 32
                end

                g = pnt[g]
            end

            sync_warp(); threadfence_block()
        end

        lane == 0 && (C[t, v] = sone(s, T, trans))
        sync_warp(); threadfence_block()
        f = idx[v]

        while !iszero(f) && !istop[f]
            Rp = Rptr[f]; nn = Rptr[f + 1] - Rp
            Sp = Sptr[f]; na = Sptr[f + 1] - Sp
            Dp = Dptr[f]; Lp = Lptr[f]

            if lane == 0
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
            end

            sync_warp(); threadfence_block()
            r = lane + 1

            while r <= na
                m = szero(s, T, trans)

                for j in 1:nn
                    m = smuladd(s, C[t, Rp + j - 1], Lval[Lp + (r - 1) * nn + j - 1], m, Val(:N), trans)
                end

                c = Stgt[Sp + r - 1]
                C[t, c] = splus(s, C[t, c], m, trans)
                r += 32
            end

            sync_warp(); threadfence_block()
            f = pnt[f]
        end
    end

    return
end

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
    ops = G.ops[]

    if !isnothing(ops) && haskey(ops.off, f)
        #
        #   [C₁ | M₂] ← C₁ [U₁₁* | U₁₁* U₁₂]      (one GEMM per part, no triangular solve)
        #
        KU = reshape(view(ops.KU, ops.off[f][2]:(ops.off[f][2] + nn * (nn + na) - 1)), nn, nn + na)

        #
        #   C[:, sep] ← C[:, sep] ⊕ C₁ (U₁₁* U₁₂)   (through an index view: no buffer, no scatter)
        #   C₁ ← C₁ U₁₁*                              (overwrite; in place when one tile wide)
        #
        ispositive(na) && sgemx_gpu!(s, colview(C, view(G.Stgt, Sp:(Sp + na - 1))), C₁, view(KU, :, (nn + 1):(nn + na)))

        if inplace_ok(nn)
            sgemx_gpu!(s, C₁, C₁, view(KU, :, 1:nn); overwrite = true)
        else
            W₁ = ops_workspace(ops, T, size(C, 1), nn)
            copy_gpu!(W₁, C₁)
            sgemx_gpu!(s, C₁, W₁, view(KU, :, 1:nn); overwrite = true)
        end

        return
    end
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
    ops = G.ops[]

    if !isnothing(ops) && haskey(ops.off, f)
        #
        #   C₁ ← [C₁ | C₂] [L₁₁* ; L₂₁ L₁₁*]      (one gather + one GEMM, no triangular solve)
        #
        KL = reshape(view(ops.KL, ops.off[f][1]:(ops.off[f][1] + (nn + na) * nn - 1)), nn + na, nn)

        if inplace_ok(nn)
            #
            #   C₁ ← C[:, [res; sep]] [L₁₁* ; L₂₁ L₁₁*]   (in place, through an index view)
            #
            sgemx_gpu!(s, C₁, colview(C, ops.cols[f]), KL; overwrite = true, inplace = true)   # mightalias cannot see it
            return
        end

        W = ops_workspace(ops, T, size(C, 1), nn + na)
        copy_gpu!(view(W, :, 1:nn), C₁)
        ispositive(na) && gather_gpu!(view(W, :, (nn + 1):(nn + na)), C, G.Stgt, Sp)
        fill!(C₁, szero(s, T, trans))
        sgemx_gpu!(s, C₁, W, KL)
        return
    end

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

# ===== precomputed operators of the large fronts =====
#
# The factor does not change between solves, so the triangular solves of
# a large front can be folded into its off-diagonal block once (the
# "partitioned inverse" of GPU sparse triangular solvers):
#
#   L sweep:   C₁ ← (C₁ ⊕ C₂ L₂₁) L₁₁*  =  [C₁ | C₂] K_L,   K_L = [L₁₁* ; L₂₁ L₁₁*]
#   U sweep:   [C₁ | M₂] ← C₁ [U₁₁* | U₁₁* U₁₂] = C₁ K_U
#
# so each large front costs a gather and a GEMM instead of a blocked
# triangular solve (~5 launches per 64 columns). K_L and K_U have the size
# of the front's part of the factor. Over exact arithmetic the results are
# identical; with floating point the products are re-associated.
#
struct SolveOps{T}
    KL::CuVector{T}
    KU::CuVector{T}
    off::Dict{Int, Tuple{Int, Int}}     # front → (offset in KL, offset in KU)
    work::Base.RefValue{CuVector{T}}
    cols::Dict{Int, CuVector{Int}}      # front → its columns [res; sep]
end

# C[:, idx] without a bounds check (which would read idx on the host)
colview(C::CuMatrix, idx::AbstractVector) = SubArray(C, (Base.Slice(axes(C, 1)), idx))


const PRECOMPUTE_STREAMS = 8

function precompute_ops!(G::GPUSLU{Sem, T}) where {Sem, T}
    s = G.s
    trans = Val(:N)
    scale = Val(!isintegral(s))
    fronts = sort!(reduce(vcat, G.uplarge; init = Int[]))
    off = Dict{Int, Tuple{Int, Int}}()
    q = 1

    for f in fronts
        nn = G.hRptr[f + 1] - G.hRptr[f]; na = G.hSptr[f + 1] - G.hSptr[f]
        off[f] = (q, q)
        q += nn * (nn + na)
    end

    KL = CuVector{T}(undef, max(q - 1, 1))
    KU = CuVector{T}(undef, max(q - 1, 1))

    # the fronts are independent: spread them over streams so their small kernels overlap. The arrays
    # shared by the streams are ordered here (fork and join events), not by CUDA.jl's implicit sync.
    ns = min(PRECOMPUTE_STREAMS, length(fronts))
    streams = get!(() -> [CuStream() for _ in 1:PRECOMPUTE_STREAMS], G.cache, :streams)
    shared = (KL, KU, G.LDval, G.UDval, G.LLval, G.ULval)
    foreach(x -> CUDA.enable_synchronization!(x, false), shared)
    fork = CuEvent(CUDA.EVENT_DISABLE_TIMING)
    record(fork, CUDA.stream())
    foreach(st -> CUDA.wait(fork, st), streams[1:ns])
    try
        with(TUNING => false, SCRATCH_OWNER => G.cache) do      # no clean timing on concurrent streams; G owns their scratch
            for (i, f) in enumerate(fronts)
                CUDA.stream!(streams[mod1(i, ns)]) do
                    nn = Int(G.hRptr[f + 1] - G.hRptr[f]); na = Int(G.hSptr[f + 1] - G.hSptr[f])
                    Dp = Int(G.hDptr[f]); Lp = Int(G.hLptr[f]); o = off[f][1]
                    L₁₁ = reshape(view(G.LDval, Dp:(Dp + nn * nn - 1)), nn, nn)
                    U₁₁ = reshape(view(G.UDval, Dp:(Dp + nn * nn - 1)), nn, nn)

                    Kl = reshape(view(KL, o:(o + (nn + na) * nn - 1)), nn + na, nn)
                    T₁ = view(Kl, 1:nn, :)
                    identity_gpu!(s, T₁)
                    strsx_gpu!(s, trans, Val(false), Val(:L), T₁, L₁₁)          # T₁ = L₁₁*

                    Ku = reshape(view(KU, o:(o + nn * (nn + na) - 1)), nn, nn + na)
                    T₂ = view(Ku, :, 1:nn)
                    identity_gpu!(s, T₂)
                    strsx_gpu!(s, trans, scale, Val(:U), T₂, U₁₁)               # T₂ = U₁₁*

                    if ispositive(na)
                        L₂₁ = reshape(view(G.LLval, Lp:(Lp + nn * na - 1)), na, nn)
                        U₁₂ = reshape(view(G.ULval, Lp:(Lp + nn * na - 1)), nn, na)
                        B₁ = view(Kl, (nn + 1):(nn + na), :)
                        fill!(B₁, szero(s, T, trans)); sgemx_gpu!(s, B₁, L₂₁, T₁)  # L₂₁ L₁₁*
                        B₂ = view(Ku, :, (nn + 1):(nn + na))
                        fill!(B₂, szero(s, T, trans)); sgemx_gpu!(s, B₂, T₂, U₁₂)  # U₁₁* U₁₂
                    end
    
                end
            end
        end
    finally
        join_streams!(streams[1:ns])
        foreach(x -> CUDA.enable_synchronization!(x, true), shared[3:end])
    end

    cols = Dict{Int, CuVector{Int}}()
    hStgt = Array(G.Stgt)

    for f in fronts
        Rp = Int(G.hRptr[f]); Sp = Int(G.hSptr[f])
        cols[f] = CuVector{Int}(vcat(Rp:(Int(G.hRptr[f + 1]) - 1), hStgt[Sp:(Int(G.hSptr[f + 1]) - 1)]))
    end

    G.ops[] = SolveOps{T}(KL, KU, off, Ref{CuVector{T}}(CuVector{T}(undef, 1)), cols)
    return G
end

# a contiguous m × w scratch matrix (grows outside graph capture only)
function ops_workspace(ops::SolveOps{T}, ::Type{T}, m::Int, w::Int) where {T}
    if length(ops.work[]) < m * w
        @assert !CUDA.is_capturing() "ops workspace must grow before graph capture"
        ops.work[] = CuVector{T}(undef, m * w)
        CUDA.enable_synchronization!(ops.work[], false)
    end

    return reshape(view(ops.work[], 1:(m * w)), m, w)
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
            diag_solve!(s, trans, scale, uplo, Tj, view(A, J, J))
            Wj = reshape(view(W, 1:(m * b)), m, b)
            fill!(Wj, szero(s, T, Val(:N)))
            sgemx_gpu!(s, Wj, view(X, :, J), Tj)
            copy_gpu!(view(X, :, J), Wj)
        else
            diag_solve!(s, trans, scale, uplo, view(X, :, J), view(A, J, J))
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

# Scratch for diagonal-block inversion, per stream (so that concurrent
# streams never share it) and element type. It only grows outside stream
# capture (plans warm up before capturing).
#
# Each table belongs to the owner of the streams that use it: the plan that
# runs work on its own streams (SCRATCH_OWNER, set by FactorPlan and
# precompute_ops!), else the current task, whose task-local stream it is.
# Owners are held weakly, so the scratch is freed with its plan or task.
# (A table keyed weakly by the stream itself would never free anything:
# a CUDA.jl array keeps a reference to the last stream that used it. And
# a table keyed by stream handle, as before, never freed anything and could
# hand a reused handle a stale buffer, possibly from another device.)
const TRSM_WS = WeakKeyDict{Any, Dict{Tuple{CuStream, DataType}, Tuple{CuMatrix, CuVector}}}()

const SCRATCH_OWNER = ScopedValue{Any}(nothing)

function trsm_workspace(::Type{T}, len::Int) where {T}
    owner = something(SCRATCH_OWNER[], current_task())
    key = (CUDA.stream(), T)
    Tb, W = lock(TRSM_WS) do
        get(get!(Dict{Tuple{CuStream, DataType}, Tuple{CuMatrix, CuVector}}, TRSM_WS, owner), key, (nothing, nothing))
    end

    if isnothing(Tb) || length(W) < len
        @assert !CUDA.is_capturing() "trsm workspace must grow before graph capture"
        Tb = CuMatrix{T}(undef, STRSX_GPU_NB, STRSX_GPU_NB)
        W = CuVector{T}(undef, max(len, isnothing(W) ? 0 : 2 * length(W)))
        CUDA.enable_synchronization!(Tb, false)
        CUDA.enable_synchronization!(W, false)
        lock(() -> (TRSM_WS[owner][key] = (Tb, W)), TRSM_WS)
    end

    keep = CAPTURED[]
    isnothing(keep) || push!(keep, Tb, W)
    return Tb::CuMatrix{T}, W::CuVector{T}
end

# ===== graph capture =====
#
# A CUDA graph replays raw device pointers, so every scratch buffer that a
# recorded graph uses must live as long as the graph, even when the stream
# that owned it (and with it its entry in TRSM_WS) is gone. While capturing,
# trsm_workspace adds its buffers to CAPTURED[], and capture_graph ties them
# to the executable graph.

const CAPTURED = ScopedValue{Union{Nothing, Base.IdSet{Any}}}(nothing)

const GRAPH_SCRATCH = WeakKeyDict{CuGraphExec, Base.IdSet{Any}}()

# record f() on the current stream and instantiate it
function capture_graph(f)
    keep = Base.IdSet{Any}()
    graph = with(() -> CUDA.capture(f), CAPTURED => keep)
    exec = CUDA.instantiate(graph)
    lock(() -> (GRAPH_SCRATCH[exec] = keep), GRAPH_SCRATCH)
    return exec
end

# X ← semiring identity
function identity_gpu!(s::AbstractSemiring, X::AbstractMatrix{T}) where {T}
    function kernel(s, X)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j = blockIdx().y

        if i <= size(X, 1)
            @inbounds while j <= size(X, 2)
                X[i, j] = i == j ? sone(s, T, Val(:N)) : szero(s, T, Val(:N))
                j += gridDim().y
            end
        end

        return
    end

    launch2d(kernel, size(X, 1), size(X, 2), s, X)
    return X
end

# X ← Y (both may be strided views)
function copy_gpu!(X::AbstractMatrix, Y::AbstractMatrix)
    function kernel(X, Y)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j = blockIdx().y

        if i <= size(X, 1)
            @inbounds while j <= size(X, 2)
                X[i, j] = Y[i, j]
                j += gridDim().y
            end
        end

        return
    end

    launch2d(kernel, size(X, 1), size(X, 2), X, Y)
    return X
end

#
# Launch kernel(args...) over an m × ncols index space: rows on the x axis,
# columns on the y axis (strided past the 65535 limit), so that kernels
# need no (emulated, 64-bit) integer division to recover (i, j).
#
function launch2d(kernel, m::Integer, ncols::Integer, args...)
    if m > 0 && ncols > 0
        tb = min(256, 32 * cld(m, 32))
        @cuda threads = tb blocks = (cld(m, tb), min(ncols, 65535)) kernel(args...)
    end

    return
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

# ===== diagonal-block triangular solves in shared memory =====
#
# strsx_diag_kernel! gives each row of X one thread that walks the b ≤ 64 columns of the block in
# order, reading X and A from global memory: a chain of ~b²/2 dependent multiply-adds per thread, with
# only b threads (64 for the inversion of a diagonal block), ~40 µs per call. Here the block lives in
# shared memory (padded against bank conflicts), each row gets DIAG_KS threads that split the inner
# sum, and the partial sums are combined with warp shuffles: one barrier per column, ~b steps of a few
# cycles. The products are the same; only the order of the ⊕ reduction differs, so the result is
# bit-identical for idempotent ⊕ (min, max) and equal up to rounding for (+, ×).
#
const DIAG_KS = 16                       # threads per row (a power of 2 dividing 32)
const DIAG_NB = 64

@inline function ks_reduce(s::AbstractSemiring, x::T) where {T}
    o = DIAG_KS ÷ 2
    while o >= 1
        x = splus(s, x, shfl_xor_sync(0xffffffff, x, o), Val(:N))
        o ÷= 2
    end
    return x
end

# X ← X A* (UPLO = :U, upper A, columns left to right) or X ← X A (strictly lower part of A, unit
# diagonal, columns right to left; as strsx_diag_kernel!), for size(X, 1) ≤ 64 and size(A, 1) ≤ 64
function strsx_diag_shared_kernel!(s::AbstractSemiring, trans::Val, ::Val{SCALE}, ::Val{UPLO}, X::AbstractMatrix{T}, A::AbstractMatrix) where {SCALE, UPLO, T}
    Xs = CuStaticSharedArray(T, (DIAG_NB + 1, DIAG_NB))
    As = CuStaticSharedArray(T, (DIAG_NB + 1, DIAG_NB))
    m = size(X, 1); n = size(A, 1)
    tid = threadIdx().x - 1; nt = blockDim().x

    @inbounds begin
        e = tid
        while e < n * n
            As[e % n + 1, e ÷ n + 1] = A[e % n + 1, e ÷ n + 1]
            e += nt
        end
        e = tid
        while e < m * n
            Xs[e % m + 1, e ÷ m + 1] = X[e % m + 1, e ÷ m + 1]
            e += nt
        end
        sync_threads()

        q = tid % DIAG_KS
        r = tid ÷ DIAG_KS + 1                  # row of X
        z = szero(s, T, trans)

        for jj in 1:n
            j = UPLO === :U ? jj : n - jj + 1
            part = z

            if r <= m
                if UPLO === :U
                    k = 1 + q
                    while k < j
                        part = smuladd(s, Xs[r, k], As[k, j], part, Val(:N), trans)
                        k += DIAG_KS
                    end
                else
                    k = j + 1 + q
                    while k <= n
                        part = smuladd(s, Xs[r, k], As[k, j], part, Val(:N), trans)
                        k += DIAG_KS
                    end
                end
            end

            part = ks_reduce(s, part)

            if r <= m && q == 0
                acc = splus(s, Xs[r, j], part, Val(:N))
                SCALE && UPLO === :U && (acc = sprod(s, acc, sstar(s, As[j, j]), Val(:N), trans))
                Xs[r, j] = acc
            end

            sync_threads()
        end

        e = tid
        while e < m * n
            X[e % m + 1, e ÷ m + 1] = Xs[e % m + 1, e ÷ m + 1]
            e += nt
        end
    end

    return
end

# X ← A* X for a lower unit triangular A (as strsx_left_diag_kernel!), for size(X, 2) ≤ 64, size(A, 1) ≤ 64
function strsx_left_diag_shared_kernel!(s::AbstractSemiring, trans::Val, X::AbstractMatrix{T}, A::AbstractMatrix) where {T}
    Xs = CuStaticSharedArray(T, (DIAG_NB + 1, DIAG_NB))
    As = CuStaticSharedArray(T, (DIAG_NB + 1, DIAG_NB))
    n = size(A, 1); m = size(X, 2)
    tid = threadIdx().x - 1; nt = blockDim().x

    @inbounds begin
        e = tid
        while e < n * n
            As[e % n + 1, e ÷ n + 1] = A[e % n + 1, e ÷ n + 1]
            e += nt
        end
        e = tid
        while e < n * m
            Xs[e % n + 1, e ÷ n + 1] = X[e % n + 1, e ÷ n + 1]
            e += nt
        end
        sync_threads()

        q = tid % DIAG_KS
        c = tid ÷ DIAG_KS + 1                  # column of X
        z = szero(s, T, Val(:N))

        for i in 1:n
            part = z

            if c <= m
                k = 1 + q
                while k < i
                    part = smuladd(s, As[i, k], Xs[k, c], part, Val(:N), trans)
                    k += DIAG_KS
                end
            end

            part = ks_reduce(s, part)
            (c <= m && q == 0) && (Xs[i, c] = splus(s, Xs[i, c], part, Val(:N)))
            sync_threads()
        end

        e = tid
        while e < n * m
            X[e % n + 1, e ÷ n + 1] = Xs[e % n + 1, e ÷ n + 1]
            e += nt
        end
    end

    return
end

diag_threads(rows::Integer) = DIAG_KS * 32 * cld(max(rows, 1), 32)    # whole warps of whole row groups

# the diagonal solve of strsx_gpu! / strsx_left_gpu! on one ≤ 64 × 64 block
function diag_solve!(s::AbstractSemiring, trans::Val, scale::Val, uplo::Val, X::AbstractMatrix{T}, A::AbstractMatrix) where {T}
    if size(X, 1) <= DIAG_NB && size(A, 1) <= DIAG_NB && sizeof(T) <= 4      # 2 padded 64² blocks: 33 KB in 32 bits
        @cuda threads = diag_threads(size(X, 1)) strsx_diag_shared_kernel!(s, trans, scale, uplo, X, A)
    else
        tb = min(128, 32 * cld(size(X, 1), 32))
        @cuda threads = tb blocks = cld(size(X, 1), tb) strsx_diag_kernel!(s, trans, scale, uplo, X, A)
    end
    return X
end

function diag_solve_left!(s::AbstractSemiring, trans::Val, X::AbstractMatrix{T}, A::AbstractMatrix) where {T}
    if size(X, 2) <= DIAG_NB && size(A, 1) <= DIAG_NB && sizeof(T) <= 4
        @cuda threads = diag_threads(size(X, 2)) strsx_left_diag_shared_kernel!(s, trans, X, A)
    else
        tb = min(128, 32 * cld(size(X, 2), 32))
        @cuda threads = tb blocks = cld(size(X, 2), tb) strsx_left_diag_kernel!(s, trans, X, A)
    end
    return X
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
    f = @inbounds order[off + blockIdx().x]
    t = threadIdx().x + (blockIdx().y - 1) * blockDim().x

    while t <= size(C, 1)                       # each thread may own several rows (coarsening)
        upward_front!(s, trans, scale, C, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, Val(true))
        t += blockDim().x * gridDim().y
    end

    return
end

# blocks of tb rows covering nrhs rows (the batched kernels loop over rows, so fewer blocks also work)
rhs_blocks(nrhs::Integer, tb::Integer) = max(1, cld(nrhs, tb))

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
#   C₁ ← C₁ ⊕ C₂ L₂₁
#   C₁ ← C₁ L₁₁*           (unit diagonal)
#
# Up to DOWN_NB residual columns of row t are kept in registers: each
# gathered separator value C[t, sep[r]] is loaded once for all of them, and
# when nn ≤ DOWN_NB the unit-lower solve with L₁₁ runs in registers too.
#
const DOWN_NB = 8



#
# Register-blocked L sweep: one compile-time residual width per front (nn ≤ 8, no masked lanes),
# the residual values of row t in registers, and four separator loads in flight. Same operations,
# same order per entry as downward_front!. (Found by the C++ port experiment: its "reg" kernel, written
# in Julia here, is as fast as the C++ one.)
#
# zi: the residual entries C[t, res] are known to be the semiring zero (not read; see skip_fill)
@generated function down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, ::Val{NN}, zi::Bool = false) where {NN}
    xs = [Symbol(:x, j) for j in 1:NN]
    loads = [:($(xs[j]) = zi ? szero(s, eltype(C), trans) : C[t, Rp + $(j - 1)]) for j in 1:NN]
    upd = [:($(xs[j]) = smuladd(s, c, Lval[Lp + $(j - 1) * na + r - 1], $(xs[j]), Val(:N), trans)) for j in 1:NN]
    solve = Expr[]

    for j in NN:-1:1, k in (j + 1):NN
        push!(solve, :($(xs[j]) = smuladd(s, $(xs[k]), Dval[Dp + $(j - 1) * NN + $(k - 1)], $(xs[j]), Val(:N), trans)))
    end

    stores = [:(C[t, Rp + $(j - 1)] = $(xs[j])) for j in 1:NN]

    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $(loads...)
            r = 1

            while r + 3 <= na
                c1 = C[t, Stgt[Sp + r - 1]]; c2 = C[t, Stgt[Sp + r]]; c3 = C[t, Stgt[Sp + r + 1]]; c4 = C[t, Stgt[Sp + r + 2]]
                c = c1; $(upd...); r += 1
                c = c2; $(upd...); r += 1
                c = c3; $(upd...); r += 1
                c = c4; $(upd...); r += 1
            end

            while r <= na
                c = C[t, Stgt[Sp + r - 1]]
                $(upd...)
                r += 1
            end

            $(solve...)
            $(stores...)
        end

        return
    end
end

@inline function downward_front_reg!(s::AbstractSemiring, trans::Val, C::AbstractMatrix, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, zi::Bool = false)
    @inbounds begin
        Rp = Rptr[f]; nn = Rptr[f + 1] - Rp
        Sp = Sptr[f]; na = Sptr[f + 1] - Sp
        Dp = Dptr[f]; Lp = Lptr[f]
    end

    if nn == 1
        down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(1), zi)
    elseif nn == 2
        down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(2), zi)
    elseif nn == 3
        down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(3), zi)
    elseif nn == 4
        down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(4), zi)
    elseif nn <= 8
        nn == 5 ? down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(5), zi) :
        nn == 6 ? down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(6), zi) :
        nn == 7 ? down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(7), zi) :
                  down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(8), zi)
    else
        downward_front!(s, trans, C, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, zi)
    end

    return
end

function downward_kernel_reg!(s::AbstractSemiring, trans::Val, C::AbstractMatrix{T}, order, off::Int,
        Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval) where {T}
    f = @inbounds order[off + blockIdx().x]
    t = threadIdx().x + (blockIdx().y - 1) * blockDim().x

    while t <= size(C, 1)
        downward_front_reg!(s, trans, C, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval)
        t += blockDim().x * gridDim().y
    end

    return
end

#
#   C[i, j] ← C[i, j] ⊕ m, atomically
#
# C[i, j] ← C[i, j] ⊕ m atomically. When ⊕ is min, max or + on a machine type, this is one native
# reduction (RED: fire and forget, no returned value); otherwise a compare-and-swap loop.
@inline atomic_splus!(s::AbstractSemiring, trans::Val, C::CuDeviceMatrix{T}, i, j, m::T) where {T} =
    atomic_splus!(atomic_kind(s, trans, T), s, trans, C, i, j, m)

atomic_kind(s, trans, T) = Val(:cas)
atomic_kind(::Semiring.MinPlus, ::Val{:N}, ::Type{<:Union{Float32, Float64, Int32, Int64}}) = Val(:min)
atomic_kind(::Semiring.DualQuantale{Semiring.MinPlus}, ::Val{:N}, ::Type{<:Union{Float32, Float64, Int32, Int64}}) = Val(:max)   # MaxPlus
atomic_kind(::Semiring.DualQuantale{Semiring.MinMax}, ::Val{:N}, ::Type{<:Union{Float32, Float64, Int32, Int64}}) = Val(:max)   # MaxMin
atomic_kind(::Semiring.PlusProd, ::Val{:N}, ::Type{<:Union{Float32, Float64}}) = Val(:add)

@inline atomic_ptr(::Type{U}, C, i, j) where {U} = reinterpret(Core.LLVMPtr{U, CUDA.AS.Global}, pointer(C, i + (j - 1) * size(C, 1)))

@inline function atomic_splus!(::Val{:add}, s, trans, C::CuDeviceMatrix{T}, i, j, m::T) where {T}
    CUDA.atomic_add!(atomic_ptr(T, C, i, j), m)
    return
end

# integers: native min / max. IEEE floats: their order is the signed-integer order of the bits for
# x ≥ 0 and the reversed unsigned order for x < 0, so min is a signed min when m ≥ 0 and an unsigned
# max when m < 0 (and the converse for max). Exact; a NaN m is dropped, as by minnum / maxnum.
for (K, op, rop) in ((:min, :atomic_min!, :atomic_max!), (:max, :atomic_max!, :atomic_min!))
    @eval @inline function atomic_splus!(::Val{$(QuoteNode(K))}, s, trans, C::CuDeviceMatrix{T}, i, j, m::T) where {T}
        if T <: Integer
            CUDA.$op(atomic_ptr(T, C, i, j), m)
        elseif !isnan(m)
            S = sizeof(T) == 4 ? Int32 : Int64
            U = sizeof(T) == 4 ? UInt32 : UInt64

            if signbit(m)
                CUDA.$rop(atomic_ptr(U, C, i, j), reinterpret(U, m))
            else
                CUDA.$op(atomic_ptr(S, C, i, j), reinterpret(S, m))
            end
        end

        return
    end
end

@inline function atomic_splus!(::Val{:cas}, s::AbstractSemiring, trans::Val, C::CuDeviceMatrix{T}, i, j, m::T) where {T}
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
