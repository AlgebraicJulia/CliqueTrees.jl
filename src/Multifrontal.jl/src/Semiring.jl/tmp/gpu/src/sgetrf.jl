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

    @step "top selection" begin
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
    end
    #
    # the bottom forest: subtrees rooted at the bottom fronts whose parent is
    # in the top (or that are roots), balanced over nt threads by work
    #
    @step "bottom forest" begin
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
    end
    #
    # simulate each worker's update stack: the updates of its subtree roots
    # whose parent is in the top (the boundary) stay on it, at offsets that
    # the simulation reproduces exactly
    #
    @step "worker stacks" begin
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
    end
    #
    # merge chains of top fronts (see config().factor_merge): the GPU then factors merged fronts, in a merged
    # copy of the factor (gathered before and scattered back after the top), and every child's update
    # is assembled at its positions in the merged front of its parent
    #
    @step "top merge" begin
        amal = nothing

        if merge > 1 && any(istop)
            Stgt = MF.targets(S.sep)
            amal = amalgamation(amalgamation_key(S.Dptr), I, merge, alpha, Vector{I}(view(Rptr, 1:(nf + 1))), Vector{I}(view(Sptr, 1:(nf + 1))),
                Vector{I}(view(Stgt, 1:(Sptr[nf + 1] - 1))), Vector{I}(view(S.Dptr, 1:(nf + 1))), Vector{I}(view(S.Lptr, 1:(nf + 1))),
                Vector{I}(view(pnt, 1:nf)), Vector{I}(view(S.idx, 1:size(F, 1))); allowed = istop, compact = true)     # cached per symbolic factorization
            isnothing(amal) || amal.nf == nf && (amal = nothing)        # nothing merged
        end
    end
    if !isnothing(amal)
        return @step "merged plan" merged_plan(F, amal, istop, topwork, nt, workers, nbndval, bndoff, nstreams, maxbytes, graph)
    end
    #
    # the top fronts, in postorder, with a stack of update matrices
    #
    @step "top schedule" begin
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
    end
    @step "device buffers" begin
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
            @step "merge gather" begin
                amalgamate_gather!(LDm, P.LD, P.LL, A.mLD, z); amalgamate_gather!(LLm, P.LL, P.LL, A.mLL, z)
                amalgamate_gather!(UDm, P.UD, P.UL, A.mUD, z); amalgamate_gather!(ULm, P.UL, P.UL, A.mUL, z)
            end
            @step "fronts" factor_top_streams!(P)
            @step "merge scatter" begin
                amalgamate_scatter!(P.LD, P.LL, LDm, A.mLD); amalgamate_scatter!(P.LL, P.LL, LLm, A.mLL)
                amalgamate_scatter!(P.UD, P.UL, UDm, A.mUD); amalgamate_scatter!(P.UL, P.UL, ULm, A.mUL)
            end
            return P
        end

        return @step "fronts" factor_top_streams!(P)
    end
end

function factor_top_streams!(P::FactorPlan{Sem, T}) where {Sem, T}
    use_batched_top(P) && return factor_top_batched!(P)
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
    tcpu = @elapsed @step "bottom (CPU)" factor_bottom!(P)
    tup = @elapsed @step "upload" upload!(P)
    thost = 0.0

    tgpu = @elapsed @step "top (GPU)" begin
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
            ci, cj = cm_index(e, b)
            S[ci, cj] = A[ci, cj]
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
                ci, cj = cm_index(e, m)
                k = p + ci
                j = p + cj
                S[k, j] = smuladd(s, S[k, p], S[p, j], S[k, j], Val(:N), Val(:N))
                e += nt
            end
            sync_threads()
        end

        e = tid
        while e < b * b
            ci, cj = cm_index(e, b)
            A[ci, cj] = S[ci, cj]
            e += nt
        end
        q = tid % DIAG_KS
        r = tid ÷ DIAG_KS + 1
        #
        # TU = U₁₁*: X ← I, then column by column X[r, j] ← (X[r, j] ⊕ ⊕_{k<j} X[r, k] S[k, j]) S[j, j]*
        #
        e = tid
        while e < b * b
            ci, cj = cm_index(e, b)
            X[ci, cj] = ci == cj ? o : z
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
            ci, cj = cm_index(e, b)
            TU[ci, cj] = X[ci, cj]
            e += nt
        end
        sync_threads()
        #
        # TL = L₁₁*: X ← I, then row by row X[i, c] ← X[i, c] ⊕ ⊕_{k<i} S[i, k] X[k, c]
        #
        e = tid
        while e < b * b
            ci, cj = cm_index(e, b)
            X[ci, cj] = ci == cj ? o : z
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
            ci, cj = cm_index(e, b)
            TL[ci, cj] = X[ci, cj]
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

# ===== batched kernels: diagonal blocks, tiles, assembly =====
#
# The top of the tree is factored level by level, and each launch below covers the same step of every
# front of a level (and precompute_ops! forms the operators of all large fronts in a few launches):
# a launch is a list of tasks, one per thread block. This replaces the ~30 launches per 64 pivots of
# the per-front path, most of them single-block kernels on a GPU of 100+ SMs, by 3 launches per 64
# pivots per level. A task names its matrices by device address (Int64; the matrices of one launch
# live in different arrays) and leading dimension.

# element k (from 1) of the T array at device address a
@inline gptr(::Type{T}, a::Int64) where {T} = reinterpret(Core.LLVMPtr{T, 1}, a)
@inline gload(::Type{T}, a::Int64, k::Int64) where {T} = unsafe_load(gptr(T, a), k, Val(Base.datatype_alignment(T)))
@inline gstore!(a::Int64, x::T, k::Int64) where {T} = unsafe_store!(gptr(T, a), x, k, Val(Base.datatype_alignment(T)))

# the device address of x[i] (host side)
devaddr(x::CuArray, i::Integer = 1) = reinterpret(Int64, pointer(x, i))

#
# Register tiles: a thread of a 256-thread block holds 4 × 4 entries of a 64 × 64 block, rows
# tx + 16a + 1 and columns 4ty + c + 1 (tx = tid mod 16, ty = tid ÷ 16, a, c ∈ 0:3), as entry
# 4c + a + 1 of a 16-tuple. Warp w holds the columns 8w + 1:8w + 8: a column vector in shared memory
# is read without bank conflicts, a row vector is a broadcast, and global rows are coalesced.
#
const TB_NT = 256

@inline tile_row(tx, a) = tx + 16a + 1
@inline tile_col(ty, c) = 4ty + c + 1

# X[i, j] ← X[i, j] ⊕ l[i] u[j] on the thread's entries
@inline function rank1(s::AbstractSemiring, X::NTuple{16, T}, l::NTuple{4, T}, u::NTuple{4, T}) where {T}
    return ntuple(k -> smuladd(s, l[((k - 1) & 3) + 1], u[((k - 1) >> 2) + 1], X[k], Val(:N), Val(:N)), Val(16))
end

# t[i + 1] for a runtime i ∈ 0:3, by selects (a runtime index into a tuple would go through local memory)
@inline pick4(t::NTuple{4}, i) = ifelse(i == 0, t[1], ifelse(i == 1, t[2], ifelse(i == 2, t[3], t[4])))

# the thread's entries of the b × b block at sl / su (strictly lower part from sl, the rest from su); zero outside
@inline function load_block(::Type{T}, sl::Int64, su::Int64, ld::Int64, b::Int64, tx, ty, z::T) where {T}
    return ntuple(Val(16)) do k
        i = tile_row(tx, (k - 1) & 3); j = tile_col(ty, (k - 1) >> 2)
        (i <= b && j <= b) ? gload(T, i > j ? sl : su, i + (j - 1) * ld) : z
    end
end

@inline function ident_block(b::Int64, tx, ty, z::T, o::T) where {T}
    return ntuple(k -> (i = tile_row(tx, (k - 1) & 3); ifelse(i == tile_col(ty, (k - 1) >> 2) && i <= b, o, z)), Val(16))
end

# X[1:m, 1:n] (or its upper part) → a, leading dimension ld
@inline function store_block!(a::Int64, X::NTuple{16}, ld::Int64, m::Int64, n::Int64, tx, ty, upper::Bool)
    Base.Cartesian.@nexprs 16 k -> begin
        i = tile_row(tx, (k - 1) & 3); j = tile_col(ty, (k - 1) >> 2)
        (i <= m && j <= n && (!upper || i <= j)) && gstore!(a, X[k], i + (j - 1) * ld)
    end
    return
end

# the owners of column q0 + 1 write it (of S and of XU) to buffer nb; the owners of row q0 + 1 write it (of S and of XL)
@inline function publish_col!(CL, CX, S::NTuple{16}, XU::NTuple{16}, q0, nb, tx, ty)
    if ty == q0 >> 2
        c = q0 & 3
        Base.Cartesian.@nexprs 4 a -> begin
            @inbounds CL[tile_row(tx, a - 1), nb] = pick4((S[a], S[a + 4], S[a + 8], S[a + 12]), c)
            @inbounds CX[tile_row(tx, a - 1), nb] = pick4((XU[a], XU[a + 4], XU[a + 8], XU[a + 12]), c)
        end
    end
    return
end

@inline function publish_row!(RU, RX, S::NTuple{16}, XL::NTuple{16}, q0, nb, tx, ty)
    if tx == q0 & 15
        a = q0 >> 4
        Base.Cartesian.@nexprs 4 c -> begin
            @inbounds RU[tile_col(ty, c - 1), nb] = pick4((S[4c - 3], S[4c - 2], S[4c - 1], S[4c]), a)
            @inbounds RX[tile_col(ty, c - 1), nb] = pick4((XL[4c - 3], XL[4c - 2], XL[4c - 1], XL[4c]), a)
        end
    end
    return
end

# the thread's rows of column vector C (zero above row p + 1 if masked, times sp if SCALE)
@inline function col_vals(s::AbstractSemiring, ::Val{SCALE}, C, nb, p, tx, z::T, sp::T, masked::Bool) where {SCALE, T}
    return ntuple(Val(4)) do a
        i = tile_row(tx, a - 1)
        v = @inbounds C[i, nb]
        v = (masked & (i <= p)) ? z : v
        SCALE ? sprod(s, v, sp, Val(:N), Val(:N)) : v
    end
end

@inline function row_vals(R, nb, p, ty, z::T, masked::Bool) where {T}
    return ntuple(c -> (j = tile_col(ty, c - 1); v = @inbounds R[j, nb]; (masked & (j <= p)) ? z : v), Val(4))
end

# the owners of column q0 + 1 keep its new values v: rows below q0 + 1 (lower) or all
@inline function keep_col(X::NTuple{16, T}, v::NTuple{4, T}, q0, tx, ty, lower::Bool) where {T}
    own = ty == q0 >> 2
    c = q0 & 3
    return ntuple(Val(16)) do k
        a = (k - 1) & 3
        ifelse(own & (((k - 1) >> 2) == c) & (!lower | (tile_row(tx, a) > q0 + 1)), v[a + 1], X[k])
    end
end

#
# One b × b diagonal block (b ≤ 64) per thread block, in registers:
#
#   LU = true    S ← LU of S in place (as sgetrf_diag_kernel!), TL = L*, TU = U*
#   LU = false   only the closures TL = L* (L: the strictly lower part of S, unit) and TU = U* of a factor
#
# all three right-looking, in one pass over the pivots p = 1, …, b:
#
#   S[i, p] ← S[i, p] S[p, p]*  (i > p, LU)          XU[:, p] ← XU[:, p] S[p, p]*      (SCALE)
#   S[i, j] ← S[i, j] ⊕ S[i, p] S[p, j]              (i, j > p, LU)
#   XL[i, c] ← XL[i, c] ⊕ S[i, p] XL[p, c]           (i > p; XL = I at the start)
#   XU[r, j] ← XU[r, j] ⊕ XU[r, p] S[p, j]           (j > p; XU = I at the start)
#
# Column p and row p of S, row p of XL and column p of XU are final when step p starts; their owners
# publish them in shared memory (two buffers, one barrier per pivot). The products are those of
# sgetrf_diag_kernel! and strsx_*_diag_shared_kernel!; XL and XU sum them in pivot order instead of a
# tree, the same for idempotent ⊕ (min, max). A warp skips the updates that are void for its columns:
# XL[p, c] is zero for c > p and S[p, j] is not used for j ≤ p.
#
# A task: (sl, su, lds, b, a, u, tl, ldtl, tu, ldtu). S is read from sl (strictly lower part) and su
# (the rest), leading dimension lds; with LU the factored block goes to a, and its upper part to u if
# u ≠ 0 (as copyupper_gpu!). TL and TU go to tl and tu if nonzero.
#
function diag_block_kernel!(s::AbstractSemiring, ::Val{SCALE}, ::Val{LU}, ::Type{T}, tasks, toff::Int32) where {SCALE, LU, T}
    sl, su, lds, b, aout, uout, tl, ldtl, tu, ldtu = @inbounds tasks[toff + blockIdx().x]
    CL = CuStaticSharedArray(T, (DIAG_NB, 2))              # column p of S, row p of S, row p of XL, column p of XU
    RU = CuStaticSharedArray(T, (DIAG_NB, 2))
    RX = CuStaticSharedArray(T, (DIAG_NB, 2))
    CX = CuStaticSharedArray(T, (DIAG_NB, 2))
    tid = Int(threadIdx().x) - 1
    tx = tid & 15; ty = tid >> 4; w8 = (tid >> 5) << 3     # the warp's columns: w8 + 1:w8 + 8
    z = szero(s, T, Val(:N)); o = sone(s, T, Val(:N))

    S = load_block(T, sl, su, lds, b, tx, ty, z)
    XL = ident_block(b, tx, ty, z, o)
    XU = XL
    publish_col!(CL, CX, S, XU, 0, 1, tx, ty)
    publish_row!(RU, RX, S, XL, 0, 1, tx, ty)
    sync_threads()

    for p in 1:b
        nb = ((p - 1) & 1) + 1
        sp = SCALE ? sstar(s, @inbounds(CL[p, nb])) : o
        l = col_vals(s, Val(SCALE && LU), CL, nb, p, tx, z, sp, true)        # S[i, p] (i > p), scaled
        xu = col_vals(s, Val(SCALE), CX, nb, p, tx, z, sp, false)            # XU[:, p], scaled
        u = row_vals(RU, nb, p, ty, z, true)                                 # S[p, j] (j > p)
        xl = row_vals(RX, nb, p, ty, z, false)                               # XL[p, :]

        if SCALE
            LU && (S = keep_col(S, l, p - 1, tx, ty, true))
            XU = keep_col(XU, xu, p - 1, tx, ty, false)
        end

        if w8 + 8 > p && w8 < b
            LU && (S = rank1(s, S, l, u))
            XU = rank1(s, XU, xu, u)
        end

        w8 < p && (XL = rank1(s, XL, l, xl))

        if p < b
            publish_col!(CL, CX, S, XU, p, 3 - nb, tx, ty)
            publish_row!(RU, RX, S, XL, p, 3 - nb, tx, ty)
        end

        sync_threads()
    end

    if LU
        store_block!(aout, S, lds, b, b, tx, ty, false)
        uout != 0 && store_block!(uout, S, lds, b, b, tx, ty, true)
    end

    tl != 0 && store_block!(tl, XL, ldtl, b, b, tx, ty, false)
    tu != 0 && store_block!(tu, XU, ldtu, b, b, tx, ty, false)
    return
end

# Xs[1:64, 1:n] ← the m × n matrix at a (rows past m are zero)
@inline function load_tile!(Xs, a::Int64, ld::Int64, m::Int64, n::Int64, tid, z::T) where {T}
    e = tid

    while e < (n << 6)
        i = e & 63; j = e >> 6
        @inbounds Xs[i + 1, j + 1] = i < m ? gload(T, a, i + 1 + j * ld) : z
        e += TB_NT
    end

    return
end

@inline function put_tile!(Xs, X::NTuple{16}, tx, ty)
    Base.Cartesian.@nexprs 16 k -> (@inbounds Xs[tile_row(tx, (k - 1) & 3), tile_col(ty, (k - 1) >> 2)] = X[k])
    return
end

# X ← X ⊕ As[:, 1:k] Bs[1:k, :] on the thread's entries
@inline function tile_mac(s::AbstractSemiring, X::NTuple{16, T}, As, Bs, k::Int64, tx, ty) where {T}
    for kk in 1:k
        av = ntuple(a -> @inbounds(As[tile_row(tx, a - 1), kk]), Val(4))
        bv = ntuple(c -> @inbounds(Bs[kk, tile_col(ty, c - 1)]), Val(4))
        X = rank1(s, X, av, bv)
    end

    return X
end

#
# One tile C[1:m, 1:n] (m, n ≤ 64) per thread block:
#
#   X ← ⊕_q A_q B_q                 (products q, each of depth k_q ≤ 64)
#   X ← X D (D: n × n) or D X (D: m × m), if asked
#   C ← X, or C ← C ⊕ X             (and C2 ← the same, if C2 ≠ 0)
#
# A task: (c, ldc, c2, m, n, flags, q1, nq, d, ldd) with flags 1 overwrite, 2 X D, 4 D X; a product:
# (a, lda, b, ldb, k). Both operands of a product are in shared memory before C is written, so a task
# with one product may overwrite its own A or B (a panel solve in place).
#
function tile_kernel!(s::AbstractSemiring, ::Type{T}, tasks, prods, toff::Int32) where {T}
    c, ldc, c2, m, n, flags, q1, nq, d, ldd = @inbounds tasks[toff + blockIdx().x]
    As = CuStaticSharedArray(T, (64, 64))
    Bs = CuStaticSharedArray(T, (64, 64))
    tid = Int(threadIdx().x) - 1
    tx = tid & 15; ty = tid >> 4
    z = szero(s, T, Val(:N))
    X = ntuple(_ -> z, Val(16))

    for q in q1:(q1 + nq - 1)
        a, lda, b, ldb, k = @inbounds prods[q]
        load_tile!(As, a, lda, m, k, tid, z)
        load_tile!(Bs, b, ldb, k, n, tid, z)
        sync_threads()
        X = tile_mac(s, X, As, Bs, k, tx, ty)
        sync_threads()
    end

    if flags & 2 != 0
        put_tile!(As, X, tx, ty)
        load_tile!(Bs, d, ldd, n, n, tid, z)
        sync_threads()
        X = tile_mac(s, ntuple(_ -> z, Val(16)), As, Bs, n, tx, ty)
    elseif flags & 4 != 0
        put_tile!(Bs, X, tx, ty)
        load_tile!(As, d, ldd, m, m, tid, z)
        sync_threads()
        X = tile_mac(s, ntuple(_ -> z, Val(16)), As, Bs, m, tx, ty)
    end

    ow = flags & 1 != 0

    Base.Cartesian.@nexprs 16 k -> begin
        i = tile_row(tx, (k - 1) & 3); j = tile_col(ty, (k - 1) >> 2)

        if i <= m && j <= n
            e = i + (j - 1) * ldc
            v = ow ? X[k] : splus(s, gload(T, c, e), X[k], Val(:N))
            gstore!(c, v, e)
            c2 != 0 && gstore!(c2, v, e)
        end
    end

    return
end

const ASM_NT = 256

#
# Assembly of the fronts of a level, straight into the factor (as extendadd_direct_gpu!), one thread
# block per column j of a front [L₁₁ U₁₂; L₂₁ M]:
#
#   L₁₁[1:j, j] ← U₁₁[1:j, j]  (j ≤ n₁: the diagonal block in one array)        M[:, j - n₁] ← 0  (j > n₁)
#   F[rel_c[v], j] ← F[rel_c[v], j] ⊕ M_c[v, w]   for each child c with rel_c[w] = j, v = 1:na_c
#
# The children are taken in their order, with a barrier between two (they may add to the same
# entries), so the sums are those of one extendadd launch per child, without races. rel_c is
# increasing, so w is found by bisection. A front: (l11, u11, l21, u12, m, n₁, n₂, k1, nk); a child:
# (address of M_c, na_c, rel pointer).
#
function assemble_kernel!(s::AbstractSemiring, ::Type{T}, fronts, kids, reltgt, foff::Int32) where {T}
    l11, u11, l21, u12, mm, n1, n2, k1, nk = @inbounds fronts[foff + blockIdx().y]
    j = Int(blockIdx().x)
    j > n1 + n2 && return
    HK = CuStaticSharedArray(Int32, ASM_NT)               # the children of a chunk that hold column j, in order
    HW = CuStaticSharedArray(Int32, ASM_NT)               # and its position in them
    WC = CuStaticSharedArray(Int32, ASM_NT >> 5)          # hits per warp
    tid = Int(threadIdx().x) - 1
    lane = tid & 31; wid = tid >> 5
    z = szero(s, T, Val(:N))

    @inbounds begin
        i = tid + 1

        if j <= n1
            while i <= j
                gstore!(l11, gload(T, u11, i + (j - 1) * n1), i + (j - 1) * n1)
                i += ASM_NT
            end
        else
            while i <= n2
                gstore!(mm, z, i + (j - n1 - 1) * n2)
                i += ASM_NT
            end
        end

        sync_threads()
        c0 = k1

        while c0 < k1 + nk
            kc = c0 + tid
            w = 0

            if kc < k1 + nk
                _, na, rp = kids[kc]
                lo = rp; hi = rp + na - 1

                if reltgt[lo] <= j <= reltgt[hi]
                    while lo < hi
                        mid = (lo + hi) >> 1
                        reltgt[mid] < j ? (lo = mid + 1) : (hi = mid)
                    end

                    reltgt[lo] == j && (w = lo - rp + 1)
                end
            end

            mask = vote_ballot_sync(0xffffffff, w > 0)
            lane == 0 && (WC[wid + 1] = count_ones(mask))
            sync_threads()
            base = 0; total = 0

            for v in 1:(ASM_NT >> 5)
                x = Int(WC[v])
                v <= wid && (base += x)
                total += x
            end

            if w > 0
                h = base + count_ones(mask & ((UInt32(1) << lane) - UInt32(1))) + 1
                HK[h] = kc; HW[h] = w
            end

            sync_threads()

            for h in 1:total
                mc, na, rp = kids[HK[h]]
                cw = Int(HW[h]) - 1
                v = tid + 1

                while v <= na
                    r = Int(reltgt[rp + v - 1])
                    x = gload(T, mc, v + cw * na)

                    if r <= n1
                        a, e = j <= n1 ? (l11, r + (j - 1) * n1) : (u12, r + (j - n1 - 1) * n1)
                    else
                        a, e = j <= n1 ? (l21, r - n1 + (j - 1) * n2) : (mm, r - n1 + (j - n1 - 1) * n2)
                    end

                    gstore!(a, splus(s, gload(T, a, e), x, Val(:N)), e)
                    v += ASM_NT
                end

                sync_threads()
            end

            c0 += ASM_NT
            sync_threads()
        end
    end

    return
end

# the start of every front's assembly, for all top fronts in one launch (they are independent): the
# first phase of assemble_kernel!, one thread block per column
function assemble_init_kernel!(s::AbstractSemiring, ::Type{T}, fronts) where {T}
    l11, u11, _, _, mm, n1, n2, _, _ = @inbounds fronts[blockIdx().y]
    j = Int(blockIdx().x)
    j > n1 + n2 && return
    i = Int(threadIdx().x)

    if j <= n1
        while i <= j
            gstore!(l11, gload(T, u11, i + (j - 1) * n1), i + (j - 1) * n1)
            i += ASM_NT
        end
    else
        while i <= n2
            gstore!(mm, szero(s, T, Val(:N)), i + (j - n1 - 1) * n2)
            i += ASM_NT
        end
    end

    return
end

# x ⊕= y at element e of the array at a, atomically, when ⊕ is min or max (atomic_kind; as
# atomic_splus!: native integer min / max on the bits of an IEEE float, exact, a NaN y dropped)
@inline function gatomic_splus!(::Val{K}, a::Int64, e::Int64, y::T) where {K, T}
    p = a + (e - 1) * sizeof(T)

    if T <: Integer
        K === :min ? CUDA.atomic_min!(gptr(T, p), y) : CUDA.atomic_max!(gptr(T, p), y)
    elseif !isnan(y)
        S = sizeof(T) == 4 ? Int32 : Int64
        U = sizeof(T) == 4 ? UInt32 : UInt64

        if signbit(y)
            K === :min ? CUDA.atomic_max!(gptr(U, p), reinterpret(U, y)) : CUDA.atomic_min!(gptr(U, p), reinterpret(U, y))
        else
            K === :min ? CUDA.atomic_min!(gptr(S, p), reinterpret(S, y)) : CUDA.atomic_max!(gptr(S, p), reinterpret(S, y))
        end
    end

    return
end

#
# Assembly of all children of a level at once, for ⊕ = min or max: the order of the ⊕ does not matter
# and the atomic ⊕ is exact, so every entry of every child is added in parallel, with no barrier. A
# task: (k1, k2, w0, w1), columns w0:w1 of child k1 (k1 = k2), or all of children k1:k2 (w1 = 0); a
# child: (address of M_c, na_c, rel pointer, front).
#
function assemble_atomic_kernel!(kind::Val, ::Type{T}, fronts, kids, tasks, reltgt, toff::Int32) where {T}
    k1, k2, w0, w1 = @inbounds tasks[toff + blockIdx().x]

    @inbounds for kc in k1:k2
        mc, na, rp, fr = kids[kc]
        l11, _, l21, u12, mm, n1, n2, _, _ = fronts[fr]
        lo = iszero(w1) ? 1 : w0
        hi = iszero(w1) ? na : w1
        e = Int(threadIdx().x) - 1
        len = (hi - lo + 1) * na

        while e < len
            v, w = cm_index(e, na)
            w += lo - 1
            r = Int(reltgt[rp + v - 1]); j = Int(reltgt[rp + w - 1])
            x = gload(T, mc, v + (w - 1) * na)

            if r <= n1
                a, ee = j <= n1 ? (l11, r + (j - 1) * n1) : (u12, r + (j - n1 - 1) * n1)
            else
                a, ee = j <= n1 ? (l21, r - n1 + (j - 1) * n2) : (mm, r - n1 + (j - n1 - 1) * n2)
            end

            gatomic_splus!(kind, a, ee, x)
            e += ASM_NT
        end
    end

    return
end

# ===== the top, level by level =====
#
# factor_top_batched! factors the top fronts of a plan with static update slots (P.levels) with the
# batched kernels above. Per level:
#
#   assembly              one launch: every child of every front of the level
#   for each 64-pivot step k (the fronts with at least k blocks of pivots, J = their k-th block):
#     diagonal blocks     one launch: L₁₁[J, J] ← LU, TL = L[J, J]*, TU = U[J, J]*, U₁₁[J, J] ← its upper part
#     panels              one launch: [L₁₁[J, R] | U₁₂[J, :]] ← TL [⋯] (also into U₁₁[J, R]), [L₁₁[R, J]; L₂₁[:, J]] ← [⋯] TU
#     trailing update     one launch: L₁₁[R, R], U₁₂[R, :], L₂₁[:, R] ⊕= (panel) (panel)        (R: the pivots after J)
#   Schur complements     one launch: M ← M ⊕ L₂₁ U₁₂
#
# (with the merged-diagonal start of every front, merge_diag_gpu! and the zero M, in one launch for the
# whole top). This is the blocked right-looking LU of each front with inverted diagonal blocks, as the
# per-front path (sgetrf_gpu!, strsx_gpu!, strsx_left_gpu!), with the same products; for idempotent ⊕
# the factor is bit-identical.
#
# The tables of tasks depend only on the plan, so they are built on the first factorization and kept
# (TOP_BATCH), which also lets a recorded graph replay them.

const BATCHED_TOP = Ref(true)          # (A/B switch for benchmarks: false runs the per-front path)

struct TopBatch
    fronts::CuVector{NTuple{9, Int64}}
    kids::CuVector{NTuple{4, Int64}}
    ktasks::CuVector{NTuple{4, Int64}}
    diag::CuVector{NTuple{10, Int64}}
    tiles::CuVector{NTuple{10, Int64}}
    prods::CuVector{NTuple{5, Int64}}
    work::CuVector
    launches::Vector{NTuple{4, Int}}     # (kind, first task - 1, tasks, grid width)
    maxw::Int                            # widest front (n₁ + n₂)
end

const TB_INIT, TB_ASM, TB_ASM_ORDERED, TB_DIAG, TB_PANEL, TB_TRAIL, TB_SCHUR = 1, 2, 3, 4, 5, 6, 7

const TOP_BATCH = WeakKeyDict{Any, TopBatch}()

use_batched_top(P::FactorPlan{Sem, T}) where {Sem, T} =
    BATCHED_TOP[] && !isempty(P.levels) && config().direct_assembly && config().fused_front && sizeof(T) <= 4

# children of an atomic assembly per task: whole children while they are small, else ≥ 8192 entries
const ASM_TASK = 8192

function top_batch(P::FactorPlan{Sem, T}) where {Sem, T}
    tb = lock(() -> get(TOP_BATCH, P, nothing), TOP_BATCH)
    isnothing(tb) || return tb
    @assert !CUDA.is_capturing() "the batched schedule must be built before graph capture"
    nb = 64; sz = sizeof(T)
    nslot = maximum(length, P.levels)
    work = CuVector{T}(undef, 2 * nb * nb * nslot)
    CUDA.enable_synchronization!(work, false)
    LD, UD, LL, UL = top_arrays(P)
    # (pointer() is slow, ~0.2 µs, and the plan's array fields are not concrete: the addresses as Int64)
    bLD::Int64, bUD::Int64, bLL::Int64, bUL::Int64, bMg::Int64, bMb::Int64, bW::Int64 = devaddr.((LD, UD, LL, UL, P.Mg, P.Mb, work))
    ordered = atomic_kind(P.F.s, Val(:N), T) isa Union{Val{:add}, Val{:cas}}
    fronts = NTuple{9, Int64}[]; kids = NTuple{4, Int64}[]; ktasks = NTuple{4, Int64}[]
    diag = NTuple{10, Int64}[]; tiles = NTuple{10, Int64}[]; prods = NTuple{5, Int64}[]
    launches = NTuple{4, Int}[]
    index = zeros(Int, length(P.tasks))
    maxw = 0
    #
    # every front: (L₁₁, U₁₁, L₂₁, U₁₂, M, n₁, n₂, first child, children), the children with their front
    #
    for level in P.levels, t in level
        n1, n2, Dp, Lp, out, ks = P.tasks[t]
        k1 = length(kids) + 1

        for (gpu, off, na, rp) in ks
            push!(kids, ((gpu ? bMg : bMb) + (off - 1) * sz, na, rp, length(fronts) + 1))
        end

        push!(fronts, (bLD + (Dp - 1) * sz, bUD + (Dp - 1) * sz, bLL + (Lp - 1) * sz, bUL + (Lp - 1) * sz, bMg + (out - 1) * sz, n1, n2, k1, length(ks)))
        index[t] = length(fronts)
        maxw = max(maxw, n1 + n2)
    end

    ordered || push!(launches, (TB_INIT, 0, length(fronts), maxw))
    addr(base, p, ld, i, j) = base + (p - 1 + (i - 1) + (j - 1) * ld) * sz          # of X[i, j], X = the ld-row matrix at base[p]

    for level in P.levels
        f1 = index[first(level)]                # the fronts of a level are consecutive
        lw = maximum(t -> P.tasks[t][1] + P.tasks[t][2], level)

        if ordered
            push!(launches, (TB_ASM_ORDERED, f1 - 1, length(level), lw))
        else
            q0 = length(ktasks)
            kc = fronts[f1][8]; kend = fronts[index[last(level)]][8] + fronts[index[last(level)]][9] - 1
            while kc <= kend
                na = kids[kc][2]

                if na * na >= ASM_TASK                      # a large child: groups of columns
                    w = max(1, ASM_TASK ÷ na)

                    for w0 in 1:w:na
                        push!(ktasks, (kc, kc, w0, min(w0 + w - 1, na)))
                    end

                    kc += 1
                else                                        # small children: as many as fill a task
                    k2 = kc; len = na * na

                    while k2 < kend && len + kids[k2 + 1][2]^2 <= ASM_TASK && kids[k2 + 1][2]^2 < ASM_TASK
                        k2 += 1; len += kids[k2][2]^2
                    end

                    push!(ktasks, (kc, k2, 1, 0))
                    kc = k2 + 1
                end
            end

            push!(launches, (TB_ASM, q0, length(ktasks) - q0, 0))
        end

        maxp = maximum(t -> cld(P.tasks[t][1], nb), level)

        for k in 1:maxp
            j0 = (k - 1) * nb + 1
            d0 = length(diag); p0 = length(tiles)
            #
            # diagonal blocks: TL, TU of the level's slot-th front at work[(2slot - 2) nb² + 1], [(2slot - 1) nb² + 1]
            #
            for (slot, t) in enumerate(level)
                n1, n2, Dp, Lp = P.tasks[t]
                j0 <= n1 || continue
                b = min(nb, n1 - j0 + 1)
                a = addr(bLD, Dp, n1, j0, j0)
                push!(diag, (a, a, n1, b, a, addr(bUD, Dp, n1, j0, j0), bW + (2slot - 2) * nb * nb * sz, nb,
                    bW + (2slot - 1) * nb * nb * sz, nb))
            end

            push!(launches, (TB_DIAG, d0, length(diag) - d0, 0))
            #
            # panels, in place
            #
            for (slot, t) in enumerate(level)
                n1, n2, Dp, Lp = P.tasks[t]
                j0 <= n1 || continue
                b = min(nb, n1 - j0 + 1); j1 = j0 + b - 1
                wl = bW + (2slot - 2) * nb * nb * sz; wu = bW + (2slot - 1) * nb * nb * sz

                for c0 in (j1 + 1):nb:n1                    # L₁₁[J, c] ← TL L₁₁[J, c], also into U₁₁
                    w = min(nb, n1 - c0 + 1); x = addr(bLD, Dp, n1, j0, c0)
                    push!(prods, (wl, nb, x, n1, b))
                    push!(tiles, (x, n1, addr(bUD, Dp, n1, j0, c0), b, w, 1, length(prods), 1, 0, 0))
                end

                for c0 in 1:nb:n2                           # U₁₂[J, c] ← TL U₁₂[J, c]
                    w = min(nb, n2 - c0 + 1); x = addr(bUL, Lp, n1, j0, c0)
                    push!(prods, (wl, nb, x, n1, b))
                    push!(tiles, (x, n1, 0, b, w, 1, length(prods), 1, 0, 0))
                end

                for r0 in (j1 + 1):nb:n1                    # L₁₁[r, J] ← L₁₁[r, J] TU
                    h = min(nb, n1 - r0 + 1); x = addr(bLD, Dp, n1, r0, j0)
                    push!(prods, (x, n1, wu, nb, b))
                    push!(tiles, (x, n1, 0, h, b, 1, length(prods), 1, 0, 0))
                end

                for r0 in 1:nb:n2                           # L₂₁[r, J] ← L₂₁[r, J] TU
                    h = min(nb, n2 - r0 + 1); x = addr(bLL, Lp, n2, r0, j0)
                    push!(prods, (x, n2, wu, nb, b))
                    push!(tiles, (x, n2, 0, h, b, 1, length(prods), 1, 0, 0))
                end
            end

            push!(launches, (TB_PANEL, p0, length(tiles) - p0, 0))
            #
            # trailing update of the pivots after J (the update matrix M waits for the Schur complement)
            #
            r0t = length(tiles)

            for t in level
                n1, n2, Dp, Lp = P.tasks[t]
                j0 + nb <= n1 || continue
                b = nb; j1 = j0 + b - 1

                for c0 in (j1 + 1):nb:n1, r0 in (j1 + 1):nb:n1                  # L₁₁[R, R]
                    push!(prods, (addr(bLD, Dp, n1, r0, j0), n1, addr(bLD, Dp, n1, j0, c0), n1, b))
                    push!(tiles, (addr(bLD, Dp, n1, r0, c0), n1, 0, min(nb, n1 - r0 + 1), min(nb, n1 - c0 + 1), 0, length(prods), 1, 0, 0))
                end

                for c0 in 1:nb:n2, r0 in (j1 + 1):nb:n1                          # U₁₂[R, :]
                    push!(prods, (addr(bLD, Dp, n1, r0, j0), n1, addr(bUL, Lp, n1, j0, c0), n1, b))
                    push!(tiles, (addr(bUL, Lp, n1, r0, c0), n1, 0, min(nb, n1 - r0 + 1), min(nb, n2 - c0 + 1), 0, length(prods), 1, 0, 0))
                end

                for c0 in (j1 + 1):nb:n1, r0 in 1:nb:n2                          # L₂₁[:, R]
                    push!(prods, (addr(bLL, Lp, n2, r0, j0), n2, addr(bLD, Dp, n1, j0, c0), n1, b))
                    push!(tiles, (addr(bLL, Lp, n2, r0, c0), n2, 0, min(nb, n2 - r0 + 1), min(nb, n1 - c0 + 1), 0, length(prods), 1, 0, 0))
                end
            end

            push!(launches, (TB_TRAIL, r0t, length(tiles) - r0t, 0))
        end
        #
        # Schur complements M ← M ⊕ L₂₁ U₁₂, one tile per 64 × 64 block of M, a product per 64 pivots
        #
        s0 = length(tiles)

        for t in level
            n1, n2, Dp, Lp, out = P.tasks[t]

            for c0 in 1:nb:n2, r0 in 1:nb:n2
                q1 = length(prods) + 1

                for j0 in 1:nb:n1
                    push!(prods, (addr(bLL, Lp, n2, r0, j0), n2, addr(bUL, Lp, n1, j0, c0), n1, min(nb, n1 - j0 + 1)))
                end

                push!(tiles, (addr(bMg, out, n2, r0, c0), n2, 0, min(nb, n2 - r0 + 1), min(nb, n2 - c0 + 1), 0, q1, length(prods) - q1 + 1, 0, 0))
            end
        end

        push!(launches, (TB_SCHUR, s0, length(tiles) - s0, 0))
    end

    up(v) = (d = CuVector(isempty(v) ? [ntuple(_ -> Int64(0), fieldcount(eltype(v)))] : v); CUDA.enable_synchronization!(d, false); d)
    tb = TopBatch(up(fronts), up(kids), up(ktasks), up(diag), up(tiles), up(prods), work, filter(l -> l[3] > 0, launches), maxw)
    lock(() -> (TOP_BATCH[P] = tb), TOP_BATCH)
    return tb
end

function factor_top_batched!(P::FactorPlan{Sem, T}) where {Sem, T}
    s = P.F.s
    scale = Val(!isintegral(s))
    kind = atomic_kind(s, Val(:N), T)
    B = top_batch(P)

    for (what, off, n, w) in B.launches
        o = Int32(off)

        if what == TB_INIT
            @phase FTIMER[] :assemble @cuda threads = ASM_NT blocks = (w, n) assemble_init_kernel!(s, T, B.fronts)
        elseif what == TB_ASM
            @phase FTIMER[] :assemble @cuda threads = ASM_NT blocks = n assemble_atomic_kernel!(kind, T, B.fronts, B.kids, B.ktasks, P.reltgt, o)
        elseif what == TB_ASM_ORDERED
            @phase FTIMER[] :assemble @cuda threads = ASM_NT blocks = (w, n) assemble_kernel!(s, T, B.fronts, B.kids, P.reltgt, o)
        elseif what == TB_DIAG
            @phase FTIMER[] :lu_diag @cuda threads = TB_NT blocks = n diag_block_kernel!(s, scale, Val(true), T, B.diag, o)
        elseif what == TB_PANEL
            @phase FTIMER[] :panel_trsm @cuda threads = TB_NT blocks = n tile_kernel!(s, T, B.tiles, B.prods, o)
        elseif what == TB_TRAIL
            @phase FTIMER[] :lu_gemm @cuda threads = TB_NT blocks = n tile_kernel!(s, T, B.tiles, B.prods, o)
        else
            @phase FTIMER[] :schur_gemm @cuda threads = TB_NT blocks = n tile_kernel!(s, T, B.tiles, B.prods, o)
        end
    end

    return P
end
