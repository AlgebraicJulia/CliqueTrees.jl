# GPU factorization of the closure, A* = U* L*.
#
#   sgetrf_gpu!(s, A)        dense LU of a CuMatrix, in place (as Semiring.sgetrf!)
#   mlu_gpu(s, A)            hybrid sparse factorization: the many small fronts
#                            at the bottom of the elimination tree on the CPU,
#                            the large fronts near the root (and all their
#                            ancestors) on the GPU. Returns (F, G): the CPU
#                            factor and a GPUSLU whose factor lives on the GPU.
#
# Every kernel performs the same semiring operations in the same per-entry
# order as the CPU code, so idempotent semirings (tropical, bottleneck)
# reproduce the CPU factor bit for bit.

using .Semiring: AbstractSemiring, ChordalSLU, splus, sprod, sstar, szero, isintegral, spool_mt,
    sgetrf_loop!, sgetrf_loop_1!

# ===== dense LU =====

const SGETRF_GPU_NB = 64
const LU_DIAG_THREADS = 1024             # 256 left most of the b³/3 updates of a 64-block serial per thread


#
# A ← L + U with A* = U* L*: right-looking and blocked, as Semiring.sgetrf_mt!
#
#   Akk ← LU of Akk                 one thread block, Akk in shared memory
#   Akn ← Lkk* Akn                  left lower unit TRSM
#   Ank ← Ank Ukk*                  right upper TRSM
#   Ann ← Ann ⊕ Ank Akn             semiring GEMM
#
function sgetrf_gpu!(s::AbstractSemiring, A::AbstractMatrix{T}; nb::Int = SGETRF_GPU_NB) where {T}
    @assert size(A, 1) == size(A, 2)
    @assert nb <= SGETRF_GPU_NB

    n = size(A, 1)
    scale = Val(!isintegral(s))

    for k in 1:nb:n
        b = min(nb, n - k + 1)
        J = k:(k + b - 1)
        Akk = view(A, J, J)
        @phase FTIMER[] :lu_diag @cuda threads = LU_DIAG_THREADS sgetrf_diag_kernel!(s, scale, Akk)

        if k + b <= n
            R = (k + b):n
            @phase FTIMER[] :lu_trsm strsx_left_gpu!(s, Val(:N), Akk, view(A, J, R))
            @phase FTIMER[] :lu_trsm strsx_gpu!(s, Val(:N), scale, Val(:U), view(A, R, J), Akk)
            @phase FTIMER[] :lu_gemm sgemx_gpu!(s, view(A, R, R), view(A, R, J), view(A, J, R))
        end
    end

    return A
end

#
# Unblocked LU of a b × b block (b ≤ 64), as Semiring.sgetrf2!:
#
#   for p = 1, …, b:
#       A[k, p] ← A[k, p] A[p, p]*              k > p
#       A[k, j] ← A[k, j] ⊕ A[k, p] A[p, j]     k, j > p
#
function sgetrf_diag_kernel!(s::AbstractSemiring, ::Val{SCALE}, A::AbstractMatrix{T}) where {SCALE, T}
    S = CuStaticSharedArray(T, (SGETRF_GPU_NB, SGETRF_GPU_NB))
    b = size(A, 1)
    t = threadIdx().x
    nt = blockDim().x

    @inbounds begin
        e = t

        while e <= b * b
            i = (e - 1) % b + 1
            j = (e - 1) ÷ b + 1
            S[i, j] = A[i, j]
            e += nt
        end

        sync_threads()

        for p in 1:b
            if SCALE
                sp = sstar(s, S[p, p])
                k = p + t

                while k <= b
                    S[k, p] = sprod(s, S[k, p], sp, Val(:N), Val(:N))
                    k += nt
                end

                sync_threads()
            end

            m = b - p
            e = t

            while e <= m * m
                k = p + (e - 1) % m + 1
                j = p + (e - 1) ÷ m + 1
                S[k, j] = smuladd(s, S[k, p], S[p, j], S[k, j], Val(:N), Val(:N))
                e += nt
            end

            sync_threads()
        end

        e = t

        while e <= b * b
            i = (e - 1) % b + 1
            j = (e - 1) ÷ b + 1
            A[i, j] = S[i, j]
            e += nt
        end
    end

    return
end

#
# Left lower unit triangular solve X ← A* X, blocked over rows:
#
#   X[i, :] ← X[i, :] ⊕ Σ_{k<i} A[i, k] X[k, :]
#
# with diagonal-block inversion as in strsx_gpu!: T = A[J, J]* (solve with
# the identity), X[J, :] ← T X[J, :] as a GEMM, then the trailing GEMM.
#
function strsx_left_gpu!(s::AbstractSemiring, trans::Val, A::AbstractMatrix, X::AbstractMatrix{T}; nb::Int = STRSX_GPU_NB) where {T}
    n = size(A, 1)
    m = size(X, 2)
    inv = m > STRSX_INV_MIN

    for i0 in 1:nb:n
        i1 = min(i0 + nb - 1, n)
        J = i0:i1
        b = i1 - i0 + 1

        if inv
            Tb, W = trsm_workspace(T, b * m)
            Tj = view(Tb, 1:b, 1:b)
            identity_gpu!(s, Tj)
            diag_solve_left!(s, trans, Tj, view(A, J, J))
            Wj = reshape(view(W, 1:(b * m)), b, m)
            fill!(Wj, szero(s, T, Val(:N)))
            sgemx_gpu!(s, Wj, Tj, view(X, J, :))
            copy_gpu!(view(X, J, :), Wj)
        else
            diag_solve_left!(s, trans, view(X, J, :), view(A, J, J))
        end

        if i1 < n
            sgemx_gpu!(s, view(X, (i1 + 1):n, :), view(A, (i1 + 1):n, J), view(X, J, :))
        end
    end

    return X
end

function strsx_left_diag_kernel!(s::AbstractSemiring, trans::Val, X::AbstractMatrix, A::AbstractMatrix)
    c = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n = size(A, 1)

    if c <= size(X, 2)
        @inbounds for i in 1:n
            acc = X[i, c]

            for k in 1:(i - 1)
                acc = smuladd(s, A[i, k], X[k, c], acc, Val(:N), trans)
            end

            X[i, c] = acc
        end
    end

    return
end

# ===== hybrid sparse factorization =====

struct HybridTimes
    cpu::Float64        # bottom fronts on the CPU
    upload::Float64     # factor arrays and boundary updates to the GPU
    gpu::Float64        # top fronts on the GPU (wall time, synchronized)
    host::Float64       # host time spent issuing the GPU work (or launching the graph)
    ntop::Int
    topwork::Float64    # fraction of the multiply-adds done on the GPU
end

#
# Everything that depends only on the sparsity pattern: which fronts go to
# the GPU (the top: fronts with nn + na ≥ `large` and all their ancestors),
# the CPU and GPU workspaces, the persistent device buffers, and the list of
# GPU front tasks. With `graph = true` the GPU work is captured as a CUDA
# graph after the first factorization, and replayed by later ones (new
# weights, same pattern), which removes the host cost of the ~50 small
# launches per front.
#
#
# One CPU thread's share of the bottom forest: whole subtrees (contiguous
# postorder ranges), its own update stack, and where its boundary updates
# (left on its stack) go in the device buffer Mb.
#
struct BottomWorker{T, I}
    ranges::Vector{UnitRange{Int}}
    Mptr::MF.FVector{I}
    Mval::MF.FVector{T}
    Fval::MF.FVector{T}
    pool::Any
    nbndval::Int
    base::Int
end

mutable struct FactorPlan{Sem <: AbstractSemiring, T, I}
    F::ChordalSLU{Sem, T, I}
    istop::BitVector
    topwork::Float64
    nt::Int
    # CPU, bottom fronts: independent subtrees spread over threads
    workers::Vector{BottomWorker{T, I}}
    nbndval::Int                    # total length of the boundary updates
    # device buffers
    LD::CuVector{T}
    LL::CuVector{T}
    UD::CuVector{T}
    UL::CuVector{T}
    Mb::CuVector{T}
    Mg::CuVector{T}
    Fg::CuVector{T}
    relptr::CuVector{I}
    reltgt::CuVector{I}
    # GPU, top fronts in postorder: (n₁, n₂, Dp, Lp, out, kids), kids = [(from_gpu, offset, na, rel pointer)]
    tasks::Vector{Tuple{Int, Int, Int, Int, Int, Vector{Tuple{Bool, Int, Int, Int}}}}
    # concurrency: with nstreams > 1 every top front has its own update slot,
    # and the tasks of each level run on nstreams streams (one graph branch each)
    levels::Vector{Vector{Int}}
    streams::Vector{CuStream}
    Fbufs::Vector{CuVector{T}}
    graph::Bool
    exec::Union{Nothing, CuGraphExec}
    # merged chains of top fronts (config().factor_merge), or nothing: the merge and the merged factor arrays
    amal::Any
    merged::NTuple{4, CuVector{T}}
end

function FactorPlan(F::ChordalSLU{Sem, T, I}; large::Integer = 256, nt::Integer = Threads.nthreads(), graph::Bool = true,
        nstreams::Integer = 8, maxbytes::Integer = 2^31, merge::Integer = config().factor_merge, alpha::Real = 0.5) where {Sem, T, I}
    s = F.s
    check_semiring(s, T)
    S = F.S.S
    nf = Int(MF.nv(S.res))
    pnt = S.pnt
    Rptr = MF.pointers(S.res)
    Sptr = MF.pointers(S.sep)
    relptr = MF.pointers(S.rel)
    chdptr = MF.pointers(S.chd)
    chdtgt = MF.targets(S.chd)

    nn(f) = Int(Rptr[f + 1] - Rptr[f])
    na(f) = Int(Sptr[f + 1] - Sptr[f])
    children(f) = (Int(chdtgt[p]) for p in chdptr[f]:(chdptr[f + 1] - 1))

    istop = falses(nf)

    for f in 1:nf
        if nn(f) + na(f) >= large
            g = f

            while !iszero(g) && !istop[g]
                istop[g] = true
                g = pnt[g]
            end
        end
    end

    work(f) = nn(f)^3 / 3 + nn(f)^2 * na(f) + nn(f) * na(f)^2
    topwork = sum(work(f) for f in 1:nf if istop[f]; init = 0.0) / max(sum(work, 1:nf), 1.0)
    #
    # the bottom forest: subtrees rooted at the bottom fronts whose parent is
    # in the top (or that are roots), balanced over nt threads by work
    #
    sz = ones(Int, nf)

    for f in 1:nf
        iszero(pnt[f]) || (sz[pnt[f]] += sz[f])
    end

    roots = [f for f in 1:nf if !istop[f] && (iszero(pnt[f]) || istop[pnt[f]])]
    weight = [sum(g -> work(g) + 64, (f - sz[f] + 1):f) for f in roots]
    nw = max(1, min(nt, length(roots)))
    lists = [Int[] for _ in 1:nw]; load = zeros(nw)

    for r in sortperm(weight; rev = true)
        w = argmin(load)
        push!(lists[w], roots[r]); load[w] += weight[r]
    end
    #
    # simulate each worker's update stack: the updates of its subtree roots
    # whose parent is in the top (the boundary) stay on it, at offsets that
    # the simulation reproduces exactly
    #
    bndoff = zeros(Int, nf)
    workers = BottomWorker{T, I}[]
    base = 1

    for list in lists
        sort!(list)
        ranges = [(f - sz[f] + 1):f for f in list]
        cstack = Tuple{Int, Int}[]; cpeak = 0; cdepth = 0; nFcpu = 2      # (offset, length)

        for r in ranges, f in r
            for _ in children(f)
                pop!(cstack)
            end

            if ispositive(na(f))
                off = isempty(cstack) ? 1 : sum(cstack[end])
                push!(cstack, (off, na(f)^2))
                cpeak = max(cpeak, off + na(f)^2 - 1)
                cdepth = max(cdepth, length(cstack))

                if !iszero(pnt[f]) && istop[pnt[f]]
                    bndoff[f] = base + off - 1
                end
            end

            nFcpu = max(nFcpu, (nn(f) + na(f))^2)
        end

        nb = isempty(cstack) ? 0 : sum(cstack[end]) - 1
        push!(workers, BottomWorker{T, I}(ranges, MF.FVector{I}(undef, cdepth + 1), MF.FVector{T}(undef, max(cpeak, 1)),
            MF.FVector{T}(undef, nFcpu), spool_mt(s, T, 1), nb, base))
        base += nb
    end

    nbndval = base - 1
    #
    # merge chains of top fronts (see config().factor_merge): the GPU then factors merged fronts, in a merged
    # copy of the factor (gathered before and scattered back after the top), and every child's update
    # is assembled at its positions in the merged front of its parent
    #
    amal = nothing

    if merge > 1 && any(istop)
        Stgt = MF.targets(S.sep)
        amal = amalgamation(amalgamation_key(S.Dptr), I, merge, alpha, Vector{I}(view(Rptr, 1:(nf + 1))), Vector{I}(view(Sptr, 1:(nf + 1))),
            Vector{I}(view(Stgt, 1:(Sptr[nf + 1] - 1))), Vector{I}(view(S.Dptr, 1:(nf + 1))), Vector{I}(view(S.Lptr, 1:(nf + 1))),
            Vector{I}(view(pnt, 1:nf)), Vector{I}(view(S.idx, 1:size(F, 1))); allowed = istop, compact = true)     # cached per symbolic factorization
        isnothing(amal) || amal.nf == nf && (amal = nothing)        # nothing merged
    end

    if !isnothing(amal)
        return merged_plan(F, amal, istop, topwork, nt, workers, nbndval, bndoff, nstreams, maxbytes, graph)
    end
    #
    # the top fronts, in postorder, with a stack of update matrices
    #
    gstack = Tuple{Int, Int}[]; gpeak = 0
    nFgpu = 1
    tasks = Tuple{Int, Int, Int, Int, Int, Vector{Tuple{Bool, Int, Int, Int}}}[]

    for f in 1:nf
        if istop[f]
            kids = Tuple{Bool, Int, Int, Int}[]

            for c in Iterators.reverse(collect(children(f)))
                if istop[c]
                    off, _ = pop!(gstack)
                    push!(kids, (true, off, na(c), Int(relptr[c])))
                else
                    push!(kids, (false, bndoff[c], na(c), Int(relptr[c])))
                end
            end

            out = isempty(gstack) ? 1 : sum(gstack[end])
            push!(tasks, (nn(f), na(f), Int(S.Dptr[f]), Int(S.Lptr[f]), out, kids))

            if ispositive(na(f))
                push!(gstack, (out, na(f)^2))
                gpeak = max(gpeak, out + na(f)^2 - 1)
            end

            nFgpu = max(nFgpu, (nn(f) + na(f))^2)
        end
    end
    #
    # concurrent schedule: static update slots, levels by height in the top
    #
    topsize = sum(f -> istop[f] ? na(f)^2 : 0, 1:nf; init = 0)
    nstreams = (nstreams > 1 && topsize * sizeof(T) <= maxbytes) ? Int(nstreams) : 1
    levels = Vector{Int}[]

    if nstreams > 1
        slot = zeros(Int, nf); theight = zeros(Int, nf); q = 1
        tasks = empty(tasks)

        for f in 1:nf
            istop[f] || continue
            slot[f] = q; q += na(f)^2
            theight[f] = 1 + maximum((theight[c] for c in children(f) if istop[c]); init = 0)
            kids = [istop[c] ? (true, slot[c], na(c), Int(relptr[c])) : (false, bndoff[c], na(c), Int(relptr[c])) for c in Iterators.reverse(collect(children(f)))]
            push!(tasks, (nn(f), na(f), Int(S.Dptr[f]), Int(S.Lptr[f]), slot[f], kids))
            length(levels) < theight[f] && push!(levels, Int[])
            push!(levels[theight[f]], length(tasks))
        end

        gpeak = topsize
    end

    P = FactorPlan{Sem, T, I}(
        F, istop, topwork, nt, workers, nbndval,
        CuVector{T}(undef, length(F.LDval)), CuVector{T}(undef, length(F.LLval)),
        CuVector{T}(undef, length(F.UDval)), CuVector{T}(undef, length(F.ULval)),
        CuVector{T}(undef, max(nbndval, 1)), CuVector{T}(undef, max(gpeak, 1)), CuVector{T}(undef, 1),
        upload(relptr), upload(MF.targets(S.rel)),
        tasks, levels, [CuStream() for _ in 1:nstreams],
        [CuVector{T}(undef, nFgpu) for _ in 1:nstreams], graph, nothing,
        nothing, ntuple(_ -> CuVector{T}(undef, 0), 4),
    )
    #
    # the plan orders every cross-stream access itself (fork event, level
    # barriers, join), so CUDA.jl's implicit synchronization on stream
    # switches is disabled for its buffers; it would also break capture
    #
    for x in (P.LD, P.LL, P.UD, P.UL, P.Mb, P.Mg, P.relptr, P.reltgt, P.Fbufs...)
        CUDA.enable_synchronization!(x, false)
    end

    return P
end

# config().factor_merge: merge chains of top fronts in the GPU factorization up to this width (1 = off).
# The top of the elimination tree is nearly a chain (about 40 levels of 1 or 2 fronts on a 3D grid),
# and most top fronts have 1–8 pivots, so the factorization is a sequence of ~10 tiny launches per
# front. Merged fronts are factored as one dense front (the same eliminations; the padding is the
# semiring zero).

function merged_plan(F::ChordalSLU{Sem, T, I}, A::Amalgamation, istop, topwork, nt, workers, nbndval, bndoff, nstreams, maxbytes, graph) where {Sem, T, I}
    S = F.S.S
    nf = Int(MF.nv(S.res))
    Sptr = MF.pointers(S.sep); Stgt = MF.targets(S.sep)
    chdptr = MF.pointers(S.chd); chdtgt = MF.targets(S.chd)
    na(f) = Int(Sptr[f + 1] - Sptr[f])
    children(f) = (Int(chdtgt[p]) for p in chdptr[f]:(chdptr[f + 1] - 1))
    sep(f) = view(Stgt, Sptr[f]:(Sptr[f + 1] - 1))
    ng = A.nf; grp = A.group
    ufirst = zeros(Int, ng); ulast = zeros(Int, ng)

    for f in 1:nf
        q = grp[f]
        iszero(ufirst[q]) && (ufirst[q] = f)
        ulast[q] = f
    end

    uw(q) = Int(A.Rptr[q + 1] - A.Rptr[q])
    una(q) = Int(A.Sptr[q + 1] - A.Sptr[q])
    utop(q) = istop[ufirst[q]]
    newrel = I[]
    #
    # positions of the vertices `verts` in unit q's front: residual vertices first, then its separator
    #
    function positions!(q, verts)
        r0 = Int(A.Rptr[q]); w = uw(q)
        sg = view(A.Stgt, A.Sptr[q]:(A.Sptr[q + 1] - 1))

        for u in verts
            if r0 <= u < r0 + w
                push!(newrel, I(u - r0 + 1))
            else
                k = searchsortedfirst(sg, u)
                @assert k <= length(sg) && sg[k] == u "merged factorization: separator vertex outside the parent's front"
                push!(newrel, I(w + k))
            end
        end
    end

    externals(q) = sort!([c for m in ufirst[q]:ulast[q] for c in children(m) if grp[c] != q])
    tasks = Tuple{Int, Int, Int, Int, Int, Vector{Tuple{Bool, Int, Int, Int}}}[]
    levels = Vector{Int}[]
    nFgpu = 1
    topsize = sum(q -> utop(q) ? una(q)^2 : 0, 1:ng; init = 0)
    nstreams = (nstreams > 1 && topsize * sizeof(T) <= maxbytes) ? Int(nstreams) : 1

    if nstreams > 1
        slot = zeros(Int, ng); theight = zeros(Int, ng); qq = 1

        for q in 1:ng
            utop(q) || continue
            slot[q] = qq; qq += una(q)^2
            ks = externals(q)
            theight[q] = 1 + maximum((theight[grp[c]] for c in ks if istop[c]); init = 0)
            kids = Tuple{Bool, Int, Int, Int}[]

            for c in Iterators.reverse(ks)
                rp = length(newrel) + 1
                positions!(q, sep(c))
                push!(kids, istop[c] ? (true, slot[grp[c]], na(c), rp) : (false, bndoff[c], na(c), rp))
            end

            push!(tasks, (uw(q), una(q), Int(A.Dptr[q]), Int(A.Lptr[q]), slot[q], kids))
            length(levels) < theight[q] && push!(levels, Int[])
            push!(levels[theight[q]], length(tasks))
            nFgpu = max(nFgpu, (uw(q) + una(q))^2)
        end

        gpeak = topsize
    else
        gstack = Tuple{Int, Int}[]; gpeak = 0

        for q in 1:ng
            utop(q) || continue
            ks = externals(q)
            offs = Dict{Int, Int}()

            for c in Iterators.reverse(ks)
                istop[c] && (offs[c] = first(pop!(gstack)))
            end

            kids = Tuple{Bool, Int, Int, Int}[]

            for c in Iterators.reverse(ks)
                rp = length(newrel) + 1
                positions!(q, sep(c))
                push!(kids, istop[c] ? (true, offs[c], na(c), rp) : (false, bndoff[c], na(c), rp))
            end

            out = isempty(gstack) ? 1 : sum(gstack[end])
            push!(tasks, (uw(q), una(q), Int(A.Dptr[q]), Int(A.Lptr[q]), out, kids))

            if ispositive(una(q))
                push!(gstack, (out, una(q)^2))
                gpeak = max(gpeak, out + una(q)^2 - 1)
            end

            nFgpu = max(nFgpu, (uw(q) + una(q))^2)
        end
    end

    merged = (CuVector{T}(undef, length(A.mLD)), CuVector{T}(undef, length(A.mLL)), CuVector{T}(undef, length(A.mUD)), CuVector{T}(undef, length(A.mUL)))
    P = FactorPlan{Sem, T, I}(
        F, istop, topwork, nt, workers, nbndval,
        CuVector{T}(undef, length(F.LDval)), CuVector{T}(undef, length(F.LLval)),
        CuVector{T}(undef, length(F.UDval)), CuVector{T}(undef, length(F.ULval)),
        CuVector{T}(undef, max(nbndval, 1)), CuVector{T}(undef, max(gpeak, 1)), CuVector{T}(undef, 1),
        upload(MF.pointers(S.rel)), upload(newrel),
        tasks, levels, [CuStream() for _ in 1:nstreams],
        [CuVector{T}(undef, nFgpu) for _ in 1:nstreams], graph, nothing,
        A, merged,
    )

    for x in (P.LD, P.LL, P.UD, P.UL, P.Mb, P.Mg, P.relptr, P.reltgt, P.Fbufs..., merged...)
        CUDA.enable_synchronization!(x, false)
    end

    return P
end

# the factor arrays the top fronts are factored in: (LD, UD, LL, UL)
top_arrays(P::FactorPlan) = isnothing(P.amal) ? (P.LD, P.UD, P.LL, P.UL) : (P.merged[1], P.merged[3], P.merged[2], P.merged[4])

ntop(P::FactorPlan) = count(P.istop)

# CPU: the bottom fronts, with Semiring.sgetrf_loop!, one task per worker
# (the subtrees are disjoint, so the workers write disjoint parts of F)
function factor_bottom!(P::FactorPlan{Sem, T, I}) where {Sem, T, I}
    F = P.F; s = F.s; S = F.S.S
    Rptr = MF.pointers(S.res)

    @sync for W in P.workers
        Threads.@spawn begin
            Mptr = W.Mptr; Mval = W.Mval
            ns = zero(I); Mptr[one(I)] = one(I)

            for r in W.ranges, f in r
                if isone(Rptr[f + 1] - Rptr[f])
                    ns = sgetrf_loop_1!(s, F.LDval, F.UDval, F.LLval, F.ULval, S.Dptr, S.Lptr, Mptr, Mval, W.Fval, S.res, S.rel, S.chd, ns, I(f))
                else
                    ns = sgetrf_loop!(s, F.LDval, F.UDval, F.LLval, F.ULval, S.Dptr, S.Lptr, Mptr, Mval, W.Fval, S.res, S.rel, S.chd, W.pool, 1, ns, I(f))
                end
            end

            @assert Mptr[ns + one(I)] - one(I) == W.nbndval
        end
    end

    return P
end

function upload!(P::FactorPlan)
    F = P.F

    for (d, h) in ((P.LD, F.LDval), (P.LL, F.LLval), (P.UD, F.UDval), (P.UL, F.ULval))
        GC.@preserve h unsafe_copyto!(pointer(d), pointer(h), length(h))
    end

    for W in P.workers
        if ispositive(W.nbndval)
            GC.@preserve W unsafe_copyto!(pointer(P.Mb, W.base), pointer(W.Mval), W.nbndval)
        end
    end

    CUDA.synchronize()
    return P
end

# GPU: issue the top fronts, ordered after the current stream. The GEMMs here are issued on several
# streams, so they use the heuristic kernel choice (no clean timing for the autotuner).
function factor_top!(P::FactorPlan)
    with(TUNING => false, SCRATCH_OWNER => P) do              # concurrent streams: no autotuning; P owns their scratch
        if !isnothing(P.amal)
            A = P.amal; z = szero(P.F.s, eltype(P.LD), Val(:N))
            LDm, LLm, UDm, ULm = P.merged
            amalgamate_gather!(LDm, P.LD, P.LL, A.mLD, z); amalgamate_gather!(LLm, P.LL, P.LL, A.mLL, z)
            amalgamate_gather!(UDm, P.UD, P.UL, A.mUD, z); amalgamate_gather!(ULm, P.UL, P.UL, A.mUL, z)
            factor_top_streams!(P)
            amalgamate_scatter!(P.LD, P.LL, LDm, A.mLD); amalgamate_scatter!(P.LL, P.LL, LLm, A.mLL)
            amalgamate_scatter!(P.UD, P.UL, UDm, A.mUD); amalgamate_scatter!(P.UL, P.UL, ULm, A.mUL)
            return P
        end

        return factor_top_streams!(P)
    end
end

function factor_top_streams!(P::FactorPlan{Sem, T}) where {Sem, T}
    s = P.F.s
    scale = Val(!isintegral(s))

    LD, UD, LL, UL = top_arrays(P)

    run(t, Fbuf) = let (n₁, n₂, Dp, Lp, out, kids) = P.tasks[t]
        factor_front_gpu!(s, scale, Fbuf, LD, UD, LL, UL, P.Mg, out, n₁, n₂, Dp, Lp,
            ((k[1] ? P.Mg : P.Mb, k[2], k[3], k[4]) for k in kids), P.relptr, P.reltgt; direct = !isempty(P.levels))
    end

    ns = length(P.streams)

    if ns == 1
        fork = CuEvent(CUDA.EVENT_DISABLE_TIMING)
        record(fork, CUDA.stream())
        CUDA.wait(fork, P.streams[1])

        CUDA.stream!(P.streams[1]) do
            for t in eachindex(P.tasks)
                run(t, P.Fbufs[1])
            end
        end
        join_streams!(P.streams[1:1])
        return P
    end
    #
    # fork: every stream starts after the current one
    #
    fork = CuEvent(CUDA.EVENT_DISABLE_TIMING)
    record(fork, CUDA.stream())

    for st in P.streams
        CUDA.wait(fork, st)
    end

    for level in P.levels
        used = min(ns, length(level))

        for (i, t) in enumerate(level)
            k = mod1(i, ns)
            CUDA.stream!(() -> run(t, P.Fbufs[k]), P.streams[k])
        end
        #
        # barrier: the next level starts when this one is done on every stream
        #
        evs = [CuEvent(CUDA.EVENT_DISABLE_TIMING) for _ in 1:used]

        for k in 1:used
            record(evs[k], P.streams[k])
        end

        for j in 1:ns, k in 1:used
            j == k || CUDA.wait(evs[k], P.streams[j])
        end
    end

    join_streams!(P.streams)
    return P
end

# the current stream waits for all of `streams`
function join_streams!(streams)
    for st in streams
        ev = CuEvent(CUDA.EVENT_DISABLE_TIMING)
        record(ev, st)
        CUDA.wait(ev, CUDA.stream())
    end
end

#
# Numeric factorization with the plan; F must hold the entries of A
# (copyto!(P.F, A)). The first call issues the GPU work directly (and
# compiles the kernels); with P.graph it then records the graph that later
# calls replay.
#
function factorize!(P::FactorPlan; download::Bool = false)
    tcpu = @elapsed factor_bottom!(P)
    tup = @elapsed upload!(P)
    thost = 0.0

    tgpu = @elapsed begin
        thost = @elapsed if isnothing(P.exec)
            factor_top!(P)
        else
            CUDA.launch(P.exec)
        end

        CUDA.synchronize()
    end

    if P.graph && isnothing(P.exec)
        P.exec = capture_graph(() -> factor_top!(P))
    end

    if download
        F = P.F

        for (h, d) in ((F.LDval, P.LD), (F.LLval, P.LL), (F.UDval, P.UD), (F.ULval, P.UL))
            GC.@preserve h unsafe_copyto!(pointer(h), pointer(d), length(h))
        end

        CUDA.synchronize()
    end

    return HybridTimes(tcpu, tup, tgpu, thost, ntop(P), P.topwork)
end

#
# CPU-only reference with the same subtree parallelism: the bottom subtrees
# in parallel (factor_bottom!), then the top fronts in postorder on the CPU
# with Semiring.sgetrf_loop! (multithreaded dense kernels). The boundary
# updates are pushed onto the top stack in postorder, so the stack holds
# every top front's children in the order sgetrf_loop! pops them.
#
function factorize_cpu!(P::FactorPlan{Sem, T, I}) where {Sem, T, I}
    F = P.F; s = F.s; S = F.S.S
    nf = Int(MF.nv(S.res))
    Rptr = MF.pointers(S.res); Sptr = MF.pointers(S.sep); pnt = S.pnt
    tbottom = @elapsed factor_bottom!(P)

    ttop = @elapsed begin
        # where each boundary update lives
        owner = Dict{Int, Tuple{BottomWorker{T, I}, Int}}()

        for W in P.workers
            # replay the worker's stack to find its boundary offsets
            stack = Tuple{Int, Int}[]

            for r in W.ranges, f in r
                for p in MF.pointers(S.chd)[f]:(MF.pointers(S.chd)[f + 1] - 1)
                    pop!(stack)
                end

                na = Int(Sptr[f + 1] - Sptr[f])

                if ispositive(na)
                    off = isempty(stack) ? 1 : sum(stack[end])
                    push!(stack, (off, na^2))
                    (!iszero(pnt[f]) && P.istop[pnt[f]]) && (owner[f] = (W, off))
                end
            end
        end

        size = 0; peak = 0; depth = 0; maxdepth = 0; nF = 2
        sizes = Int[]

        for f in 1:nf
            if P.istop[f]
                for _ in MF.pointers(S.chd)[f]:(MF.pointers(S.chd)[f + 1] - 1)
                    size -= pop!(sizes); depth -= 1
                end
            elseif !haskey(owner, f)
                continue
            end

            na = Int(Sptr[f + 1] - Sptr[f])

            if ispositive(na)
                push!(sizes, na^2); size += na^2; depth += 1
                peak = max(peak, size); maxdepth = max(maxdepth, depth)
            end

            P.istop[f] && (nF = max(nF, (Int(Rptr[f + 1] - Rptr[f]) + na)^2))
        end

        Mptr = MF.FVector{I}(undef, maxdepth + 1); Mval = MF.FVector{T}(undef, max(peak, 1)); Fval = MF.FVector{T}(undef, nF)
        pool = spool_mt(s, T, P.nt)
        ns = zero(I); Mptr[one(I)] = one(I)

        for f in 1:nf
            if P.istop[f]
                if isone(Rptr[f + 1] - Rptr[f])
                    ns = sgetrf_loop_1!(s, F.LDval, F.UDval, F.LLval, F.ULval, S.Dptr, S.Lptr, Mptr, Mval, Fval, S.res, S.rel, S.chd, ns, I(f))
                else
                    ns = sgetrf_loop!(s, F.LDval, F.UDval, F.LLval, F.ULval, S.Dptr, S.Lptr, Mptr, Mval, Fval, S.res, S.rel, S.chd, pool, P.nt, ns, I(f))
                end
            elseif haskey(owner, f)
                W, off = owner[f]
                len = Int(Sptr[f + 1] - Sptr[f])^2
                ns += one(I)
                Mptr[ns + one(I)] = Mptr[ns] + len
                copyto!(Mval, Int(Mptr[ns]), W.Mval, off, len)
            end
        end
    end

    return tbottom, ttop
end

GPUSLU(P::FactorPlan; large::Integer = 2048) = GPUSLU(P.F; large, factor = (P.LD, P.LL, P.UD, P.UL))

#
# One-shot hybrid factorization: F ← factor of A (F must hold the entries of
# A, as after copyto!(F, A)). Returns a GPUSLU with the factor on the GPU and
# the phase times. With `download = true` the GPU part is copied back into F.
#
function sgetrf_gpu!(F::ChordalSLU; large::Integer = 256, nt::Integer = Threads.nthreads(), download::Bool = false, solve_large::Integer = 2048, nstreams::Integer = 1)
    P = FactorPlan(F; large, nt, graph = false, nstreams)
    times = factorize!(P; download)
    return GPUSLU(P; large = solve_large), times
end

function mlu_gpu(s::AbstractSemiring, A::SparseMatrixCSC; alg = MF.DEFAULT_ELIMINATION_ALGORITHM, kw...)
    F = ChordalSLU(s, A; alg)
    copyto!(F, A)
    G, times = sgetrf_gpu!(F; kw...)
    return F, G, times
end

#
# Host → device copy of a dense vector (or contiguous view). CuVector(x) on
# the FixedSizeArrays used by CliqueTrees takes a slow generic path (~40×
# slower than the raw copy).
#
function upload(v::AbstractVector{T}) where {T}
    d = CuVector{T}(undef, length(v))

    if !isempty(v)
        GC.@preserve v unsafe_copyto!(pointer(d), pointer(v), length(v))
    end

    return d
end

#
# Factorize one front on the current stream:
#
#   F ← 0;  F ← F ⊕ inj M_c injᵀ for each child c (kids: (buffer, offset, na, rel pointer))
#   L₁₁ ← LU of (F₁₁ ⊕ original entries);  U₁₁ ← upper part of L₁₁
#   L₂₁ ← (L₂₁ ⊕ F₂₁) U₁₁*;  U₁₂ ← L₁₁* (U₁₂ ⊕ F₁₂)
#   M ← F₂₂ ⊕ L₂₁ U₁₂, written to Mg[out:out + na² - 1]
#
function factor_front_gpu!(s::AbstractSemiring, scale::Val, Fbuf::CuVector{T}, LD, UD, LL, UL, Mg::CuVector{T}, out::Int,
        n₁::Int, n₂::Int, Dp::Int, Lp::Int, kids, relptr_g, reltgt_g; direct::Bool = false) where {T}
    nj = n₁ + n₂
    L₁₁ = reshape(view(LD, Dp:(Dp + n₁ * n₁ - 1)), n₁, n₁)
    U₁₁ = reshape(view(UD, Dp:(Dp + n₁ * n₁ - 1)), n₁, n₁)

    # (only with static update slots: on the single-stream stack the parent's update may overlap its children's)
    if direct && config().direct_assembly && config().fused_front && ispositive(n₂) && n₁ <= DIAG_NB && sizeof(T) <= 4
        #
        # no front matrix: the original entries are already in L₁₁/U₁₁, L₂₁ and U₁₂; the update matrix
        # starts empty, and every child's update is added straight into the block it belongs to
        #
        L₂₁ = reshape(view(LL, Lp:(Lp + n₁ * n₂ - 1)), n₂, n₁)
        U₁₂ = reshape(view(UL, Lp:(Lp + n₁ * n₂ - 1)), n₁, n₂)
        M = reshape(view(Mg, out:(out + n₂ * n₂ - 1)), n₂, n₂)
        @phase FTIMER[] :assemble merge_diag_gpu!(L₁₁, U₁₁)
        @phase FTIMER[] :assemble fill!(M, szero(s, T, Val(:N)))

        for (buf, off, nac, rp) in kids
            @phase FTIMER[] :assemble extendadd_direct_gpu!(s, L₁₁, L₂₁, U₁₂, M, buf, off, nac, reltgt_g, rp)
        end

        Tb, W = trsm_workspace(T, n₁ * n₁ + 2 * n₁ * n₂)
        TU = view(Tb, 1:n₁, 1:n₁)
        TL = reshape(view(W, 1:(n₁ * n₁)), n₁, n₁)
        @phase FTIMER[] :lu_diag @cuda threads = diag_threads(n₁) front_diag_kernel!(s, scale, L₁₁, TL, TU)
        @phase FTIMER[] :assemble copyupper_gpu!(U₁₁, L₁₁)
        front_panels!(s, L₂₁, U₁₂, TL, TU, W, n₁, n₂)
        @phase FTIMER[] :schur_gemm sgemx_gpu!(s, M, L₂₁, U₁₂)
        return
    end

    Fj = reshape(view(Fbuf, 1:(nj * nj)), nj, nj)
    @phase FTIMER[] :assemble fill!(Fj, szero(s, T, Val(:N)))

    for (buf, off, nac, rp) in kids
        @phase FTIMER[] :assemble extendadd_gpu!(s, Fj, buf, off, nac, relptr_g, reltgt_g, rp)
    end

    @phase FTIMER[] :assemble combine_gpu!(s, L₁₁, U₁₁, Fj)

    if config().fused_front && ispositive(n₂) && n₁ <= DIAG_NB && sizeof(T) <= 4
        #
        # LU and both closures of the diagonal block in one kernel; the panel solves are GEMMs
        #
        Tb, W = trsm_workspace(T, n₁ * n₁ + 2 * n₁ * n₂)
        TU = view(Tb, 1:n₁, 1:n₁)
        TL = reshape(view(W, 1:(n₁ * n₁)), n₁, n₁)
        @phase FTIMER[] :lu_diag @cuda threads = diag_threads(n₁) front_diag_kernel!(s, scale, L₁₁, TL, TU)
        @phase FTIMER[] :assemble copyupper_gpu!(U₁₁, L₁₁)
        L₂₁ = reshape(view(LL, Lp:(Lp + n₁ * n₂ - 1)), n₂, n₁)
        U₁₂ = reshape(view(UL, Lp:(Lp + n₁ * n₂ - 1)), n₁, n₂)
        @phase FTIMER[] :assemble addblock_gpu!(s, L₂₁, Fj, n₁, 0)
        @phase FTIMER[] :assemble addblock_gpu!(s, U₁₂, Fj, 0, n₁)
        front_panels!(s, L₂₁, U₁₂, TL, TU, W, n₁, n₂)
        M = reshape(view(Mg, out:(out + n₂ * n₂ - 1)), n₂, n₂)
        @phase FTIMER[] :assemble addblock_gpu!(s, M, Fj, n₁, n₁; overwrite = true)
        @phase FTIMER[] :schur_gemm sgemx_gpu!(s, M, L₂₁, U₁₂)
        return
    end

    sgetrf_gpu!(s, L₁₁)
    @phase FTIMER[] :assemble copyupper_gpu!(U₁₁, L₁₁)

    if ispositive(n₂)
        L₂₁ = reshape(view(LL, Lp:(Lp + n₁ * n₂ - 1)), n₂, n₁)
        U₁₂ = reshape(view(UL, Lp:(Lp + n₁ * n₂ - 1)), n₁, n₂)
        @phase FTIMER[] :assemble addblock_gpu!(s, L₂₁, Fj, n₁, 0)
        @phase FTIMER[] :assemble addblock_gpu!(s, U₁₂, Fj, 0, n₁)
        @phase FTIMER[] :panel_trsm strsx_gpu!(s, Val(:N), scale, Val(:U), L₂₁, L₁₁)
        @phase FTIMER[] :panel_trsm strsx_left_gpu!(s, Val(:N), L₁₁, U₁₂)
        M = reshape(view(Mg, out:(out + n₂ * n₂ - 1)), n₂, n₂)
        @phase FTIMER[] :assemble addblock_gpu!(s, M, Fj, n₁, n₁; overwrite = true)
        @phase FTIMER[] :schur_gemm sgemx_gpu!(s, M, L₂₁, U₁₂)
    end

    return
end

# ===== fused diagonal block of a small front =====
#
# For a front with n₁ ≤ 64 pivots the old path issues ~11 launches for the diagonal block and the two
# panel solves (LU kernel; for each panel identity, diagonal solve, fill, GEMM, copy). This kernel does
# the LU of the block and both closures TL = L₁₁* (unit lower) and TU = U₁₁* in one launch, with the
# block in shared memory, so that the panel solves are plain GEMMs: L₂₁ ← L₂₁ TU (in place) and
# U₁₂ ← TL U₁₂. These are the same operations as the inversion path of strsx_gpu! / strsx_left_gpu!
# (which the old path takes for panels of more than 64 rows or columns).
#

function front_diag_kernel!(s::AbstractSemiring, ::Val{SCALE}, A::AbstractMatrix{T}, TL::AbstractMatrix{T}, TU::AbstractMatrix{T}) where {SCALE, T}
    S = CuStaticSharedArray(T, (DIAG_NB + 1, DIAG_NB))
    X = CuStaticSharedArray(T, (DIAG_NB + 1, DIAG_NB))
    b = size(A, 1)
    tid = threadIdx().x - 1; nt = blockDim().x
    z = szero(s, T, Val(:N)); o = sone(s, T, Val(:N))

    @inbounds begin
        e = tid
        while e < b * b
            S[e % b + 1, e ÷ b + 1] = A[e % b + 1, e ÷ b + 1]
            e += nt
        end
        sync_threads()
        #
        # LU, as sgetrf_diag_kernel!
        #
        for p in 1:b
            if SCALE
                sp = sstar(s, S[p, p])
                k = p + 1 + tid
                while k <= b
                    S[k, p] = sprod(s, S[k, p], sp, Val(:N), Val(:N))
                    k += nt
                end
                sync_threads()
            end

            m = b - p
            e = tid
            while e < m * m
                k = p + e % m + 1
                j = p + e ÷ m + 1
                S[k, j] = smuladd(s, S[k, p], S[p, j], S[k, j], Val(:N), Val(:N))
                e += nt
            end
            sync_threads()
        end

        e = tid
        while e < b * b
            A[e % b + 1, e ÷ b + 1] = S[e % b + 1, e ÷ b + 1]
            e += nt
        end
        q = tid % DIAG_KS
        r = tid ÷ DIAG_KS + 1
        #
        # TU = U₁₁*: X ← I, then column by column X[r, j] ← (X[r, j] ⊕ ⊕_{k<j} X[r, k] S[k, j]) S[j, j]*
        #
        e = tid
        while e < b * b
            X[e % b + 1, e ÷ b + 1] = e % b == e ÷ b ? o : z
            e += nt
        end
        sync_threads()

        for j in 1:b
            part = z
            if r <= b
                k = 1 + q
                while k < j
                    part = smuladd(s, X[r, k], S[k, j], part, Val(:N), Val(:N))
                    k += DIAG_KS
                end
            end
            part = ks_reduce(s, part)
            if r <= b && q == 0
                acc = splus(s, X[r, j], part, Val(:N))
                SCALE && (acc = sprod(s, acc, sstar(s, S[j, j]), Val(:N), Val(:N)))
                X[r, j] = acc
            end
            sync_threads()
        end

        e = tid
        while e < b * b
            TU[e % b + 1, e ÷ b + 1] = X[e % b + 1, e ÷ b + 1]
            e += nt
        end
        sync_threads()
        #
        # TL = L₁₁*: X ← I, then row by row X[i, c] ← X[i, c] ⊕ ⊕_{k<i} S[i, k] X[k, c]
        #
        e = tid
        while e < b * b
            X[e % b + 1, e ÷ b + 1] = e % b == e ÷ b ? o : z
            e += nt
        end
        sync_threads()
        c = r

        for i in 1:b
            part = z
            if c <= b
                k = 1 + q
                while k < i
                    part = smuladd(s, S[i, k], X[k, c], part, Val(:N), Val(:N))
                    k += DIAG_KS
                end
            end
            part = ks_reduce(s, part)
            (c <= b && q == 0) && (X[i, c] = splus(s, X[i, c], part, Val(:N)))
            sync_threads()
        end

        e = tid
        while e < b * b
            TL[e % b + 1, e ÷ b + 1] = X[e % b + 1, e ÷ b + 1]
            e += nt
        end
    end

    return
end

# L₂₁ ← L₂₁ U₁₁* and U₁₂ ← L₁₁* U₁₂ as GEMMs with the closures TU, TL (W: scratch of n₁² + 2 n₁ n₂)
function front_panels!(s, L₂₁, U₁₂, TL, TU, W, n₁, n₂)
    if inplace_ok(n₁)
        @phase FTIMER[] :panel_trsm sgemx_gpu!(s, L₂₁, L₂₁, TU; overwrite = true)        # in place, one tile wide
    else
        Y = reshape(view(W, (n₁ * n₁ + n₁ * n₂ + 1):(n₁ * n₁ + 2 * n₁ * n₂)), n₂, n₁)
        @phase FTIMER[] :panel_trsm copy_gpu!(Y, L₂₁)
        @phase FTIMER[] :panel_trsm sgemx_gpu!(s, L₂₁, Y, TU; overwrite = true)
    end

    X = reshape(view(W, (n₁ * n₁ + 1):(n₁ * n₁ + n₁ * n₂)), n₁, n₂)
    @phase FTIMER[] :panel_trsm copy_gpu!(X, U₁₂)
    @phase FTIMER[] :panel_trsm sgemx_gpu!(s, U₁₂, TL, X; overwrite = true)
    return
end


# L₁₁[i, j] ← U₁₁[i, j] for i ≤ j: the diagonal block in one array (as combine_gpu! without a front matrix)
function merge_diag_gpu!(L::AbstractMatrix, U::AbstractMatrix)
    function kernel(L, U)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j = blockIdx().y

        if i <= size(L, 1)
            @inbounds while j <= size(L, 2)
                i <= j && (L[i, j] = U[i, j])
                j += gridDim().y
            end
        end

        return
    end

    launch2d(kernel, size(L, 1), size(L, 2), L, U)
    return L
end

# a child's update added straight into the blocks of its parent: positions reltgt[rp:rp+na-1] in the
# parent's front (1:n₁ residual, then separator) select L₁₁, L₂₁, U₁₂ or the parent's update M
function extendadd_direct_gpu!(s::AbstractSemiring, L11, L21, U12, M, buf::CuVector, off::Int, na::Int, reltgt, rp::Int)
    function kernel(s, L11, L21, U12, M, buf, off, na, reltgt, rp)
        v = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        w = blockIdx().y
        n₁ = size(L11, 1)

        if v <= na
            @inbounds begin
                i = Int(reltgt[rp + v - 1]); j = Int(reltgt[rp + w - 1])
                x = buf[off + (w - 1) * na + v - 1]

                if i <= n₁ && j <= n₁
                    L11[i, j] = splus(s, L11[i, j], x, Val(:N))
                elseif j <= n₁
                    L21[i - n₁, j] = splus(s, L21[i - n₁, j], x, Val(:N))
                elseif i <= n₁
                    U12[i, j - n₁] = splus(s, U12[i, j - n₁], x, Val(:N))
                else
                    M[i - n₁, j - n₁] = splus(s, M[i - n₁, j - n₁], x, Val(:N))
                end
            end
        end

        return
    end

    tb = min(256, 32 * cld(na, 32))
    @cuda threads = tb blocks = (cld(na, tb), na) kernel(s, L11, L21, U12, M, buf, off, na, reltgt, rp)
    return
end

# ===== front kernels =====

# F[inj[v], inj[w]] ← F[inj[v], inj[w]] ⊕ M[v, w], M = reshape(buf[off:off + na² - 1], na, na)
function extendadd_gpu!(s::AbstractSemiring, F::AbstractMatrix, buf::CuVector, off::Int, na::Int, relptr, reltgt, rp::Int)
    function kernel(s, F, buf, off, na, reltgt, rp)
        v = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        w = blockIdx().y

        if v <= na
            @inbounds begin
                iv = reltgt[rp + v - 1]
                iw = reltgt[rp + w - 1]
                F[iv, iw] = splus(s, F[iv, iw], buf[off + (w - 1) * na + v - 1], Val(:N))
            end
        end

        return
    end

    tb = min(256, 32 * cld(na, 32))
    @cuda threads = tb blocks = (cld(na, tb), na) kernel(s, F, buf, off, na, reltgt, rp)
    return F
end

# L₁₁[i, j] ← F[i, j] ⊕ (i > j ? L₁₁[i, j] : U₁₁[i, j])
function combine_gpu!(s::AbstractSemiring, L::AbstractMatrix, U::AbstractMatrix, F::AbstractMatrix)
    function kernel(s, L, U, F)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j = blockIdx().y

        if i <= size(L, 1)
            @inbounds while j <= size(L, 2)
                L[i, j] = splus(s, F[i, j], i > j ? L[i, j] : U[i, j], Val(:N))
                j += gridDim().y
            end
        end

        return
    end

    launch2d(kernel, size(L, 1), size(L, 2), s, L, U, F)
    return L
end

# U[i, j] ← L[i, j] for i ≤ j
function copyupper_gpu!(U::AbstractMatrix, L::AbstractMatrix)
    function kernel(U, L)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j = blockIdx().y

        if i <= size(L, 1)
            @inbounds while j <= size(L, 2)
                i <= j && (U[i, j] = L[i, j])
                j += gridDim().y
            end
        end

        return
    end

    launch2d(kernel, size(L, 1), size(L, 2), U, L)
    return U
end

# X ← X ⊕ F[i0 .+ (1:m), j0 .+ (1:n)]   (or X ← F[…] with overwrite)
function addblock_gpu!(s::AbstractSemiring, X::AbstractMatrix, F::AbstractMatrix, i0::Int, j0::Int; overwrite::Bool = false)
    function kernel(s, X, F, i0, j0, ::Val{OVERWRITE}) where {OVERWRITE}
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j = blockIdx().y

        if i <= size(X, 1)
            @inbounds while j <= size(X, 2)
                X[i, j] = OVERWRITE ? F[i0 + i, j0 + j] : splus(s, X[i, j], F[i0 + i, j0 + j], Val(:N))
                j += gridDim().y
            end
        end

        return
    end

    launch2d(kernel, size(X, 1), size(X, 2), s, X, F, i0, j0, Val(overwrite))
    return X
end
