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
        nlarge = count(istop)               # the top of the large fronts (and their ancestors)
        #
        # Balance (config().factor_balance): the CPU factors the bottom forest one subtree per thread, so a
        # subtree with much of the work keeps one thread busy while the others idle. (When no front
        # reaches `large` there is no top, and the bottom forest is the whole tree: one subtree, on one
        # thread.) For b = 1, 2, 4, … ≤ factor_balance, the subtrees with more than 1/(b nt) of the work
        # join the top; the b kept is the one whose CPU saving (the heaviest bottom subtree, or the even
        # share, before and after) exceeds the GPU top levels it adds by the most, a level counted as
        # config().factor_level_work, or none. Work here counts a front as at least 256 multiply-adds
        # (its fixed cost on the CPU, ~0.1 µs, against ~0.3 ns per multiply-add). Subtree work grows
        # toward the roots, so the fronts marked are closed under ancestors, as the top must be.
        #
        cf = config()

        if cf.factor_balance > 0 && nt > 1
            wsub = [work(f) + 256 for f in 1:nf]

            for f in 1:nf
                iszero(pnt[f]) || (wsub[pnt[f]] += wsub[f])
            end

            total = sum(f -> iszero(pnt[f]) ? wsub[f] : 0.0, 1:nf; init = 0.0)
            theight = zeros(Int, nf)
            #
            # (the CPU's critical path in work, the levels of the top) with the top `top`
            #
            function cpu_levels(top)
                fill!(theight, 0); levels = 0; heavy = 0.0; rest = 0.0

                for f in 1:nf                       # postorder: children first
                    p = pnt[f]

                    if top[f]
                        levels = max(levels, theight[f] + 1)
                        iszero(p) || (theight[p] = max(theight[p], theight[f] + 1))
                    elseif iszero(p) || top[p]      # a bottom subtree
                        heavy = max(heavy, wsub[f]); rest += wsub[f]
                    end
                end

                return max(heavy, rest / nt), levels
            end

            c0, l0 = cpu_levels(istop)
            best = 0.0; keep = nothing; b = 1

            while b <= cf.factor_balance
                top = istop .| (wsub .> total / (b * nt))
                c, l = cpu_levels(top)
                gain = (c0 - c) - cf.factor_level_work * max(l - l0, 1)
                gain > best && (best = gain; keep = top)
                b *= 2
            end

            isnothing(keep) || copyto!(istop, keep)
        end

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

        if merge > 1 && any(istop) && nlarge >= cf.factor_merge_min
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

GPUSLU(P::FactorPlan; large::Integer = 2048, structure = nothing) = GPUSLU(P.F; large, factor = (P.LD, P.LL, P.UD, P.UL), structure)

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
# the per-front path, most of them single-block kernels on a GPU of 100+ SMs, by 2 launches per 64
# pivots per level. A task names its matrices by device address (Int64; the matrices of one launch
# live in different arrays) and leading dimension.

# element k (from 1) of the T array at device address a
@inline gptr(::Type{T}, a::Int64) where {T} = reinterpret(Core.LLVMPtr{T, 1}, a)
@inline gload(::Type{T}, a::Int64, k::Int64) where {T} = unsafe_load(gptr(T, a), k, Val(Base.datatype_alignment(T)))
@inline gstore!(a::Int64, x::T, k::Int64) where {T} = unsafe_store!(gptr(T, a), x, k, Val(Base.datatype_alignment(T)))

# the device address of x[i] (host side)
devaddr(x::CuArray, i::Integer = 1) = reinterpret(Int64, pointer(x, i))

#
# Register tiles: a thread of a 256-thread block holds 4 × 4 entries of a 64 × 64 block, rows 4tx + a + 1
# and columns 4ty + c + 1 (tx = tid mod 16, ty = tid ÷ 16, a, c ∈ 0:3), as entry 4c + a + 1 of a 16-tuple.
# The thread's part of a column (or row) vector in shared memory is then one 128-bit load (ld4), conflict
# free across a warp, and warp w holds the columns 8w + 1:8w + 8.
#
const TB_NT = 256

@inline tile_row(tx, a) = 4tx + a + 1
@inline tile_col(ty, c) = 4ty + c + 1

# X[i, j] ← X[i, j] ⊕ l[i] u[j] on the thread's entries
@inline function rank1(s::AbstractSemiring, X::NTuple{16, T}, l::NTuple{4, T}, u::NTuple{4, T}) where {T}
    return ntuple(k -> smuladd(s, l[((k - 1) & 3) + 1], u[((k - 1) >> 2) + 1], X[k], Val(:N), Val(:N)), Val(16))
end

# two rank-1 updates, X ⊕ l0 u0 ⊕ l1 u1; with op = Val(:min) / Val(:max) (min-plus / max-plus Float32 on
# compute capability 10.x, pair_ok) as 2 adds and one 3-input min / max per entry (FMNMX3; rank2_min3, as
# GEMM kernel v6): the same values. (sm_120 has no FMNMX3: there the 3-input form is 30-50% slower.)
@inline rank2(s::AbstractSemiring, ::Nothing, X::NTuple{16}, l0, u0, l1, u1) = rank1(s, rank1(s, X, l0, u0), l1, u1)
@inline rank2(s::AbstractSemiring, op::Val, X::NTuple{16}, l0, u0, l1, u1) = rank2_min3(op, X, l0, u0, l1, u1)

# the op argument of rank2 for semiring s and element type T (host side)
pair_op(s, ::Type{T}) where {T} = pair_ok(s, T) ? min3_op(s) : nothing

# the thread's entries of the b × b block at sl / su (strictly lower part from sl, the rest from su); zero outside
@inline function load_block(::Type{T}, sl::Int64, su::Int64, ld::Int64, b::Int64, tx, ty, z::T) where {T}
    return ntuple(Val(16)) do k
        i = tile_row(tx, (k - 1) & 3); j = tile_col(ty, (k - 1) >> 2)
        (i <= b && j <= b) ? gload(T, i > j ? sl : su, i + (j - 1) * ld) : z
    end
end

@inline smul4(s, x::NTuple{4, T}, y::T) where {T} = ntuple(a -> sprod(s, x[a], y, Val(:N), Val(:N)), Val(4))

# a[i] = column p as published (unscaled): L[i, p] below p (scaled with LU), XU[i, p] above p and 1 at p (scaled)
@inline function scale_col(s, ::Val{SCALE}, ::Val{LU}, av::NTuple{4, T}, sp::T, p, tx) where {SCALE, LU, T}
    SCALE || return av
    LU && return smul4(s, av, sp)
    return ntuple(a -> tile_row(tx, a - 1) > p ? av[a] : sprod(s, av[a], sp, Val(:N), Val(:N)), Val(4))
end

# W restarted at zero in row Q + 1 right of p (the owner of row p) or in column Q + 1 below p (of column p)
@inline restart_row(::Val{Q}, W::NTuple{16, T}, p, ty, z::T) where {Q, T} =
    ntuple(k -> ((k - 1) & 3) == Q && tile_col(ty, (k - 1) >> 2) > p ? z : W[k], Val(16))
@inline restart_col(::Val{Q}, W::NTuple{16, T}, p, tx, z::T) where {Q, T} =
    ntuple(k -> ((k - 1) >> 2) == Q && tile_row(tx, (k - 1) & 3) > p ? z : W[k], Val(16))

#
# One pivot p = 4p4 + Q + 1 of diag_block!: W ⊕= a b, with the owners of row and column p restarting
# their S entries as X first (unless ⊕ is idempotent and there is no scaling: then L[i, p] ⊕ L[i, p] 1 and
# U[p, j] ⊕ 1 U[p, j] are already the X entries); then the owners of row and column p + 1 publish them.
# (Q static: the thread's row or column p is its entry Q + 1.)
#
@inline function diag_pivot(s::AbstractSemiring, ::Val{SCALE}, ::Val{LU}, ::Val{IDEM}, ::Val{Q}, W::NTuple{16, T}, Sf::NTuple{16, T},
        CH, RH, DG, p4, b, tx, ty, z::T, o::T) where {SCALE, LU, IDEM, Q, T}
    p = 4p4 + Q + 1
    sp = SCALE ? sstar(s, @inbounds(DG[p])) : o
    av = scale_col(s, Val(SCALE), Val(LU), ld4(CH, 4tx + 1 + 64(p - 1)), sp, p, tx)     # a[i]: column p (L below p, XU above)
    bv = ld4(RH, 4ty + 1 + DIAG_LD * (p - 1))                                           # b[j]: row p (U right of p, XL left), 1 at p

    if !(LU && IDEM && !SCALE)
        tx == p4 && (W = restart_row(Val(Q), W, p, ty, z))
        ty == p4 && (W = restart_col(Val(Q), W, p, tx, z))
    end

    W = rank1(s, W, av, bv)
    pn = p + 1

    if pn <= b                                             # publish row and column p + 1 (S from Sf with LU = false)
        qn = (Q + 1) & 3; pn4 = (pn - 1) >> 2

        if tx == pn4
            Base.Cartesian.@nexprs 4 c -> begin
                k = 4c - 3 + qn; j = tile_col(ty, c - 1)
                @inbounds RH[j, pn] = j == pn ? o : (j > pn && !LU) ? Sf[k] : W[k]
                j == pn && (@inbounds DG[pn] = LU ? W[k] : Sf[k])
            end
        end

        if ty == pn4
            Base.Cartesian.@nexprs 4 a -> begin
                k = 4qn + a; i = tile_row(tx, a - 1)
                @inbounds CH[i, pn] = i == pn ? o : (i > pn && !LU) ? Sf[k] : W[k]
            end
        end
    end

    sync_threads()
    return W
end

#
# Pivots p, p + 1 = 4p4 + Q + 1, 4p4 + Q + 2 (Q = 0 or 2) with one barrier, for LU with idempotent ⊕ and no
# scaling (no restarts): row and column p + 1 were published before pivot p, and every thread applies
# pivot p to them for its own rows and columns (a1 = a + a0 S[p, p + 1], b1 = b + L[p + 1, p] b0, the
# owners' operations); then W ⊕= a0 b0 ⊕ a1 b1 (rank2). The kept row and column p + 1 are those before
# pivot p, corrected in the same way when the results are read (pair_fix).
#
@inline function diag_pair(s::AbstractSemiring, op, ::Val{Q}, W::NTuple{16, T}, CH, RH, DG, p4, b, tx, ty, z::T, o::T) where {Q, T}
    p = 4p4 + Q + 1
    av = ld4(CH, 4tx + 1 + 64(p - 1)); bv = ld4(RH, 4ty + 1 + DIAG_LD * (p - 1))

    if p + 1 <= b
        g = @inbounds RH[p + 1, p]; h = @inbounds CH[p + 1, p]            # S[p, p + 1], L[p + 1, p]
        ar = ld4(CH, 4tx + 1 + 64p); br = ld4(RH, 4ty + 1 + DIAG_LD * p)            # as published before pivot p
        a1 = ntuple(a -> tile_row(tx, a - 1) == p + 1 ? o : smuladd(s, av[a], g, ar[a], Val(:N), Val(:N)), Val(4))
        b1 = ntuple(c -> tile_col(ty, c - 1) == p + 1 ? o : smuladd(s, h, bv[c], br[c], Val(:N), Val(:N)), Val(4))
        W = rank2(s, op, W, av, bv, a1, b1)
    else
        W = rank1(s, W, av, bv)
    end

    if p + 2 <= b                                          # publish rows and columns p + 2, p + 3
        q4 = (p + 1) >> 2; qn = (Q + 2) & 3                # (one thread group holds both)

        if tx == q4
            Base.Cartesian.@nexprs 4 c -> begin
                j = tile_col(ty, c - 1)
                @inbounds RH[j, p + 2] = j == p + 2 ? o : W[4c - 3 + qn]
                @inbounds RH[j, p + 3] = j == p + 3 ? o : W[4c - 2 + qn]
                j == p + 2 && (@inbounds DG[p + 2] = W[4c - 3 + qn])
                j == p + 3 && (@inbounds DG[p + 3] = W[4c - 2 + qn])
            end
        end

        if ty == q4
            Base.Cartesian.@nexprs 4 a -> begin
                i = tile_row(tx, a - 1)
                @inbounds CH[i, p + 2] = i == p + 2 ? o : W[4qn + a]
                @inbounds CH[i, p + 3] = i == p + 3 ? o : W[4qn + 4 + a]
            end
        end
    end

    sync_threads()
    return W
end

# the kept value of row (rows = true) or column q at k, final: for q even, pivot q - 1 applied (diag_pair)
@inline function pair_fix(s, ::Val{PAIRS}, CH, RH, q, k, rows::Bool) where {PAIRS}
    @inbounds if rows
        v = RH[k, q]
        PAIRS && iseven(q) && k != q - 1 && (v = smuladd(s, CH[q, q - 1], RH[k, q - 1], v, Val(:N), Val(:N)))
    else
        v = CH[k, q]
        PAIRS && iseven(q) && k != q - 1 && (v = smuladd(s, CH[k, q - 1], RH[q, q - 1], v, Val(:N), Val(:N)))
    end
    return v
end

const DIAG_LD = 68                                         # rows of RH: the output reads RH[j, i] along i (4-way, not 32-way, conflicts)

#
# One b × b diagonal block (b ≤ 64) per thread block, in registers:
#
#   LU = true    S ← LU of S in place (as sgetrf_diag_kernel!), TL = L*, TU = U*
#   LU = false   only the closures TL = L* (L: the strictly lower part of S, unit) and TU = U* of a factor
#
# all three right-looking, over the pivots p = 1, …, b:
#
#   S[i, p] ← S[i, p] S[p, p]*  (i > p, LU)          XU[r, p] ← XU[r, p] S[p, p]*      (r < p, SCALE)
#   S[i, j] ← S[i, j] ⊕ S[i, p] S[p, j]              (i, j > p, LU)
#   XL[i, j] ← XL[i, j] ⊕ S[i, p] XL[p, j]           (i > p ≥ j; XL = I at the start)
#   XU[i, j] ← XU[i, j] ⊕ XU[i, p] S[p, j]           (i ≤ p < j; XU = I at the start)
#
# The three updates of a pivot touch disjoint entries, so they are one rank-1 update W ⊕= a b of one
# accumulator per entry, with a[i] = S[i, p] (i > p), XU[i, p] (i ≤ p) and b[j] = S[p, j] (j > p),
# XL[p, j] (j ≤ p): W[i, j] is S[i, j] while p < min(i, j), then XL[i, j] or XU[i, j] (restarted at
# zero) while p < max(i, j), and then no longer used. An entry's S is final when p reaches min(i, j)
# and its X when p reaches max(i, j): the entries of row and column p, which their owners publish at
# the end of pivot p - 1 (one barrier per pivot, or per two with diag_pair). Every published row and
# column is kept (RH, CH), and
# the results are read from them at the end: U and XL from the rows, L and XU from the columns. Every
# entry costs one multiply-add per pivot instead of up to three, with no masks. The products are those
# of sgetrf_diag_kernel! and strsx_*_diag_shared_kernel!; XL and XU sum them in pivot order instead of
# a tree, the same for idempotent ⊕ (min, max). (With LU = false, S is the factor itself: Sf.)
# Per 64 × 64 block: 10-14 µs, against 42 µs for sgetrf_diag_kernel! and 34 µs for each of the two
# closure kernels on a B200.
#
# A task: (sl, su, lds, b, a, u, tl, ldtl, tu, ldtu). S is read from sl (strictly lower part) and su
# (the rest), leading dimension lds; with LU the factored block goes to a, and its upper part to u if
# u ≠ 0 (as copyupper_gpu!). TL and TU go to tl and tu if nonzero. IDEM: ⊕ is idempotent (min, max).
#
function diag_block_kernel!(s::AbstractSemiring, op, scale::Val, lu::Val, idem::Val, ::Type{T}, tasks, toff::Int32) where {T}
    diag_block!(s, op, scale, lu, idem, T, batch_shared(T)..., (@inbounds tasks[toff + blockIdx().x])...)
    return
end

# the shared memory of the batched kernels: As, Bt of tile_task! are CH, RH, DG of diag_block! (one block may do both)
@inline batch_shared(::Type{T}) where {T} =
    (CuStaticSharedArray(T, (DIAG_NB, DIAG_NB)), CuStaticSharedArray(T, (DIAG_LD, DIAG_NB)), CuStaticSharedArray(T, DIAG_NB))

@inline function diag_block!(s::AbstractSemiring, op, ::Val{SCALE}, ::Val{LU}, ::Val{IDEM}, ::Type{T}, CH, RH, DG, sl::Int64, su::Int64, lds::Int64, b::Int64,
        aout::Int64, uout::Int64, tl::Int64, ldtl::Int64, tu::Int64, ldtu::Int64) where {SCALE, LU, IDEM, T}
    # CH[:, p]: column p as published; RH[:, p]: row p; DG[p] = S[p, p]
    tid = Int(threadIdx().x) - 1
    tx = tid & 15; ty = tid >> 4
    z = szero(s, T, Val(:N)); o = sone(s, T, Val(:N))

    Sf = load_block(T, sl, su, lds, b, tx, ty, z)
    W = LU ? Sf : ntuple(_ -> z, Val(16))
    #
    # publish pivot 1, as diag_pivot does for p + 1
    #
    if tx == 0
        Base.Cartesian.@nexprs 4 c -> begin
            j = tile_col(ty, c - 1)
            @inbounds RH[j, 1] = j == 1 ? o : Sf[4c - 3]
            j == 1 && (@inbounds DG[1] = Sf[1])
        end
    end

    if ty == 0
        Base.Cartesian.@nexprs 4 a -> (i = tile_row(tx, a - 1); @inbounds CH[i, 1] = i == 1 ? o : Sf[a])
    end

    sync_threads()
    sv = Val(SCALE); lv = Val(LU); iv = Val(IDEM)
    PAIRS = LU && IDEM && !SCALE

    if PAIRS                                               # (also publish row and column 2 before pivot 1)
        if tx == 0
            Base.Cartesian.@nexprs 4 c -> begin
                j = tile_col(ty, c - 1)
                @inbounds RH[j, 2] = j == 2 ? o : Sf[4c - 2]
                j == 2 && (@inbounds DG[2] = Sf[4c - 2])
            end
        end

        ty == 0 && Base.Cartesian.@nexprs 4 a -> (i = tile_row(tx, a - 1); @inbounds CH[i, 2] = i == 2 ? o : Sf[4 + a])
        sync_threads()

        for p4 in 0:((b - 1) >> 2)
            W = diag_pair(s, op, Val(0), W, CH, RH, DG, p4, b, tx, ty, z, o)
            4p4 + 3 <= b && (W = diag_pair(s, op, Val(2), W, CH, RH, DG, p4, b, tx, ty, z, o))
        end
    else
        for p4 in 0:((b - 1) >> 2)
            W = diag_pivot(s, sv, lv, iv, Val(0), W, Sf, CH, RH, DG, p4, b, tx, ty, z, o)
            4p4 + 2 <= b && (W = diag_pivot(s, sv, lv, iv, Val(1), W, Sf, CH, RH, DG, p4, b, tx, ty, z, o))
            4p4 + 3 <= b && (W = diag_pivot(s, sv, lv, iv, Val(2), W, Sf, CH, RH, DG, p4, b, tx, ty, z, o))
            4p4 + 4 <= b && (W = diag_pivot(s, sv, lv, iv, Val(3), W, Sf, CH, RH, DG, p4, b, tx, ty, z, o))
        end
    end
    #
    # the results, from the published rows and columns (coalesced: consecutive threads, consecutive rows)
    #
    e = tid
    pv = Val(PAIRS)

    @inbounds while e < DIAG_NB * DIAG_NB
        i = (e & 63) + 1; j = (e >> 6) + 1

        if i <= b && j <= b
            r = pair_fix(s, pv, CH, RH, i, j, true)       # row i at j: U[i, j] (j > i), XL[i, j] (j < i)
            c = pair_fix(s, pv, CH, RH, j, i, false)      # column j at i: L[i, j] (i > j), XU[i, j] (i < j)

            if LU                                          # U right of the diagonal, L (scaled) left of it
                d = DG[i]
                PAIRS && iseven(i) && (d = smuladd(s, CH[i, i - 1], RH[i, i - 1], d, Val(:N), Val(:N)))
                v = i < j ? r : i == j ? d : (SCALE ? sprod(s, c, sstar(s, DG[j]), Val(:N), Val(:N)) : c)
                gstore!(aout, v, i + (j - 1) * lds)
                (uout != 0 && i <= j) && gstore!(uout, v, i + (j - 1) * lds)
            end

            tl != 0 && gstore!(tl, i > j ? r : i == j ? o : z, i + (j - 1) * ldtl)

            if tu != 0                                     # XU (scaled), XU[i, i] = S[i, i]*
                v = i < j ? (SCALE ? sprod(s, c, sstar(s, DG[j]), Val(:N), Val(:N)) : c) :
                    i == j ? (SCALE ? sprod(s, o, sstar(s, DG[i]), Val(:N), Val(:N)) : o) : z
                gstore!(tu, v, i + (j - 1) * ldtu)
            end
        end

        e += TB_NT
    end

    return
end

# ⊕ is idempotent (min or max: the atomic kinds), for diag_block! (host side)
idem_plus(s, ::Type{T}) where {T} = atomic_kind(s, Val(:N), T) isa Union{Val{:min}, Val{:max}}

#
# In the tile kernel each step of a product reads 4 contiguous values of A (As[i, kk], i contiguous) and of
# B (Bt[j, kk] = B[kk, j], rows padded to 68 for the transposing store) as two 128-bit shared loads, for
# 16 multiply-adds.
#
const BT_LD = 68

# the 16 entries of the m × n matrix at a that thread tid moves into shared memory: rows tid mod 64,
# columns tid ÷ 64 + 4r (zero outside)
@inline function fetch_tile(::Type{T}, a::Int64, ld::Int64, m::Int64, n::Int64, tid, z::T) where {T}
    i = tid & 63; j0 = tid >> 6
    return ntuple(r -> (j = j0 + 4(r - 1); (i < m) & (j < n) ? gload(T, a, i + 1 + j * ld) : z), Val(16))
end

@inline function stash_a!(As, x::NTuple{16}, tid)
    i = (tid & 63) + 1; j0 = tid >> 6
    Base.Cartesian.@nexprs 16 r -> (@inbounds As[i, j0 + 4r - 3] = x[r])
    return
end

@inline function stash_b!(Bt, x::NTuple{16}, tid)           # Bt[j, kk] = B[kk, j]
    kk = (tid & 63) + 1; j0 = tid >> 6
    Base.Cartesian.@nexprs 16 r -> (@inbounds Bt[j0 + 4r - 3, kk] = x[r])
    return
end

# X[e:e + 3] for e - 1 a multiple of 4 (in an array aligned to 16 bytes): one 128-bit load for 4-byte numbers
@inline function ld4(X::CuDeviceArray{T}, e::Int64) where {T}
    if sizeof(T) == 4 && T <: Union{Float32, Int32, UInt32}
        v = unsafe_load(reinterpret(Core.LLVMPtr{NTuple{4, VecElement{T}}, CUDA.AS.Shared}, pointer(X, e)), 1, Val(16))
        return (v[1].value, v[2].value, v[3].value, v[4].value)
    else
        return @inbounds (X[e], X[e + 1], X[e + 2], X[e + 3])
    end
end

# X ← X ⊕ As[:, 1:k] Bt[:, 1:k]ᵀ on the thread's entries, two steps at a time
@inline function tile_mac(s::AbstractSemiring, op, X::NTuple{16, T}, As, Bt, k::Int64, tx, ty) where {T}
    ea = 4tx + 1; eb = 4ty + 1

    for _ in 1:(k >> 1)
        X = rank2(s, op, X, ld4(As, ea), ld4(Bt, eb), ld4(As, ea + 64), ld4(Bt, eb + BT_LD))
        ea += 128; eb += 2BT_LD
    end

    isodd(k) && (X = rank1(s, X, ld4(As, ea), ld4(Bt, eb)))
    return X
end

#
# One tile C[1:m, 1:n] (m, n ≤ 64) per thread block:
#
#   X ← A B                         (A: m × k, B: k × n, k in chunks of 64)
#   X ← X D (D: n × n) or D X (D: m × m), if asked
#   C ← X, or C ← C ⊕ X             (and C2 ← the same, if C2 ≠ 0)
#
# A task: (c, ldc, c2, m, n, flags, a, lda, b, ldb, k, d, ldd), flags 1 overwrite, 2 X D, 4 D X. The next
# chunk of A and B is fetched into registers while the current one is computed. With k ≤ 64 both
# operands are in shared memory before C is written, so the task may overwrite its own A or B (a panel
# solve in place).
#
function tile_kernel!(s::AbstractSemiring, op, ::Type{T}, tasks, toff::Int32) where {T}
    As, Bt, _ = batch_shared(T)
    tile_task!(s, op, T, As, Bt, (@inbounds tasks[toff + blockIdx().x])...)
    return
end

@inline function tile_task!(s::AbstractSemiring, op, ::Type{T}, As, Bt, c::Int64, ldc::Int64, c2::Int64, m::Int64, n::Int64, flags::Int64,
        a::Int64, lda::Int64, b::Int64, ldb::Int64, k::Int64, d::Int64, ldd::Int64) where {T}
    tid = Int(threadIdx().x) - 1
    tx = tid & 15; ty = tid >> 4
    z = szero(s, T, Val(:N))
    X = ntuple(_ -> z, Val(16))
    sz = sizeof(T)

    if k > 0
        ra = fetch_tile(T, a, lda, m, min(k, 64), tid, z)
        rb = fetch_tile(T, b, ldb, min(k, 64), n, tid, z)
        q0 = 0

        while q0 < k
            kq = min(64, k - q0)
            stash_a!(As, ra, tid); stash_b!(Bt, rb, tid)
            sync_threads()

            if q0 + 64 < k                                   # the next chunk: columns of A, rows of B
                kn = min(64, k - q0 - 64)
                ra = fetch_tile(T, a + (q0 + 64) * lda * sz, lda, m, kn, tid, z)
                rb = fetch_tile(T, b + (q0 + 64) * sz, ldb, kn, n, tid, z)
            end

            X = tile_mac(s, op, X, As, Bt, kq, tx, ty)
            sync_threads()
            q0 += 64
        end
    end

    if flags & 6 != 0
        if flags & 2 != 0                                    # X D: As ← X, Bt ← D
            Base.Cartesian.@nexprs 16 kk -> (@inbounds As[4tx + ((kk - 1) & 3) + 1, 4ty + ((kk - 1) >> 2) + 1] = X[kk])
            stash_b!(Bt, fetch_tile(T, d, ldd, n, n, tid, z), tid)
            kd = n
        else                                                 # D X: As ← D, Bt ← Xᵀ
            Base.Cartesian.@nexprs 16 kk -> (@inbounds Bt[4ty + ((kk - 1) >> 2) + 1, 4tx + ((kk - 1) & 3) + 1] = X[kk])
            stash_a!(As, fetch_tile(T, d, ldd, m, m, tid, z), tid)
            kd = m
        end

        sync_threads()
        X = tile_mac(s, op, ntuple(_ -> z, Val(16)), As, Bt, kd, tx, ty)
    end

    ow = flags & 1 != 0

    Base.Cartesian.@nexprs 16 kk -> begin
        i = 4tx + ((kk - 1) & 3) + 1; j = 4ty + ((kk - 1) >> 2) + 1

        if i <= m && j <= n
            e = i + (j - 1) * ldc
            v = ow ? X[kk] : splus(s, gload(T, c, e), X[kk], Val(:N))
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
# entries), so the sums are those of one extendadd launch per child, without races, for every semiring.
# (Adding all children at once with atomic min / max was no faster on an RTX 5060 Laptop, and 2.6×
# slower on email-Enron, where thousands of children share hub entries.) rel_c is increasing, so w is
# found by bisection. A front: (l11, u11, l21, u12, m, n₁, n₂, k1, nk); a child:
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
                HK[h] = kc % Int32; HW[h] = w % Int32
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
# (the assembly also starts each front: merge_diag_gpu! and the zero M; the diagonal blocks after the
# first are factored in the trailing update before them). This is the blocked right-looking LU of each
# front with inverted diagonal blocks, as the per-front path (sgetrf_gpu!, strsx_gpu!, strsx_left_gpu!),
# with the same products; for idempotent ⊕ the factor is bit-identical.
#
# A launch is a grid of (tiles, fronts of the level): each thread block finds its tile from the table of
# fronts, the step and its index (level_tile_kernel!), so the host only lists the launches. The table
# depends only on the plan: it is built on the first factorization and kept (TOP_BATCH), which also lets
# a recorded graph replay it.

const BATCHED_TOP = Ref(true)          # (A/B switch for benchmarks: false runs the per-front path)

struct TopBatch
    fronts::CuVector{NTuple{9, Int64}}       # (L₁₁, U₁₁, L₂₁, U₁₂, M, n₁, n₂, first child, children), levels in order
    kids::CuVector{NTuple{3, Int64}}         # (M_c, na_c, rel pointer)
    work::CuVector                           # TL, TU of each front of a level
    launches::Vector{NTuple{5, Int}}         # (kind, first front - 1, fronts, grid width, step)
end

const TB_ASM, TB_DIAG, TB_PANEL, TB_TRAIL, TB_SCHUR = 1, 2, 3, 4, 5

const TOP_BATCH = WeakKeyDict{Any, TopBatch}()

use_batched_top(P::FactorPlan{Sem, T}) where {Sem, T} =
    BATCHED_TOP[] && !isempty(P.levels) && config().direct_assembly && config().fused_front && sizeof(T) <= 4

# the tiles of a front of n₁ pivots and n₂ separator vertices in a launch at step k (as level_tile)
function level_tiles(kind, n1, n2, k)
    j1 = min(64k, n1); nr = cld(n1 - j1, 64); nc = cld(n2, 64)
    kind == TB_PANEL && return 64 * (k - 1) < n1 ? 2 * (nr + nc) : 0
    kind == TB_TRAIL && return nr * (nr + 2nc)
    return nc * nc
end

function top_batch(P::FactorPlan{Sem, T}) where {Sem, T}
    tb = lock(() -> get(TOP_BATCH, P, nothing), TOP_BATCH)
    isnothing(tb) || return tb
    @assert !CUDA.is_capturing() "the batched schedule must be built before graph capture"
    nb = 64; sz = sizeof(T)
    work = CuVector{T}(undef, 2 * nb * nb * maximum(length, P.levels))
    CUDA.enable_synchronization!(work, false)
    LD, UD, LL, UL = top_arrays(P)
    # (pointer() is slow, ~0.2 µs, and the plan's array fields are not concrete: the addresses as Int64)
    bLD::Int64, bUD::Int64, bLL::Int64, bUL::Int64, bMg::Int64, bMb::Int64 = devaddr.((LD, UD, LL, UL, P.Mg, P.Mb))
    fronts = NTuple{9, Int64}[]; kids = NTuple{3, Int64}[]
    launches = NTuple{5, Int}[]

    for level in P.levels, t in level
        n1, n2, Dp, Lp, out, ks = P.tasks[t]
        k1 = length(kids) + 1

        for (gpu, off, na, rp) in ks
            push!(kids, ((gpu ? bMg : bMb) + (off - 1) * sz, na, rp))
        end

        push!(fronts, (bLD + (Dp - 1) * sz, bUD + (Dp - 1) * sz, bLL + (Lp - 1) * sz, bUL + (Lp - 1) * sz, bMg + (out - 1) * sz, n1, n2, k1, length(ks)))
    end

    f0 = 0                                       # the fronts of a level are consecutive

    for level in P.levels
        nf = length(level)
        dims = [(P.tasks[t][1], P.tasks[t][2]) for t in level]

        push!(launches, (TB_ASM, f0, nf, maximum(sum, dims), 0))

        for k in 1:maximum(d -> cld(d[1], nb), dims)
            k == 1 && push!(launches, (TB_DIAG, f0, nf, 1, k))                # (the next ones: in the trailing update)

            for kind in (TB_PANEL, TB_TRAIL)
                w = maximum(d -> level_tiles(kind, d..., k), dims)
                w > 0 && push!(launches, (kind, f0, nf, w, k))
            end
        end

        w = maximum(d -> level_tiles(TB_SCHUR, d..., 0), dims)
        w > 0 && push!(launches, (TB_SCHUR, f0, nf, w, 0))
        f0 += nf
    end

    up(v) = (d = CuVector(isempty(v) ? [ntuple(_ -> Int64(0), fieldcount(eltype(v)))] : v); CUDA.enable_synchronization!(d, false); d)
    tb = TopBatch(up(fronts), up(kids), work, launches)
    lock(() -> (TOP_BATCH[P] = tb), TOP_BATCH)
    return tb
end

#
# The tile task (as tile_task!) of tile t (from 0) of front f at step k, and whether there is one:
#
#   panels (k)    L₁₁[J, c] ← TL L₁₁[J, c] (also into U₁₁), U₁₂[J, c] ← TL U₁₂[J, c],
#                 L₁₁[r, J] ← L₁₁[r, J] TU, L₂₁[r, J] ← L₂₁[r, J] TU            (in place)
#   trailing (k)  L₁₁[r, c], U₁₂[r, c], L₂₁[r, c] ⊕= X[r, J] Y[J, c]             (r, c after J)
#   Schur         M[r, c] ⊕= L₂₁[r, :] U₁₂[:, c]
#
# over the 64-blocks r, c (in this order; level_tiles counts them). TL, TU are at wl, wu.
#
@inline function level_tile(::Val{KIND}, f::NTuple{9, Int64}, k::Int64, t::Int64, wl::Int64, wu::Int64, ::Type{T}) where {KIND, T}
    l11, u11, l21, u12, mm, n1, n2, _, _ = f
    sz = sizeof(T)
    at(x, ld, i, j) = x + ((i - 1) + (j - 1) * ld) * sz
    none = (false, ntuple(_ -> Int64(0), Val(13)))
    nc = cld(n2, 64)

    if KIND == TB_SCHUR
        t < nc * nc || return none
        i, j = cm_index(t, nc)
        r0 = 64i - 63; c0 = 64j - 63
        return (true, (at(mm, n2, r0, c0), n2, Int64(0), min(64, n2 - r0 + 1), min(64, n2 - c0 + 1), Int64(0),
            at(l21, n2, r0, 1), n2, at(u12, n1, 1, c0), n1, n1, Int64(0), Int64(0)))
    end

    j0 = 64k - 63
    j0 <= n1 || return none
    b = min(64, n1 - j0 + 1); j1 = j0 + b - 1; nr = cld(n1 - j1, 64)

    if KIND == TB_PANEL
        if t < nr                                    # L₁₁[J, c], also into U₁₁
            c0 = j1 + 1 + 64t; x = at(l11, n1, j0, c0)
            return (true, (x, n1, at(u11, n1, j0, c0), b, min(64, n1 - c0 + 1), Int64(1), wl, Int64(64), x, n1, b, Int64(0), Int64(0)))
        end

        t -= nr

        if t < nc                                    # U₁₂[J, c]
            c0 = 1 + 64t; x = at(u12, n1, j0, c0)
            return (true, (x, n1, Int64(0), b, min(64, n2 - c0 + 1), Int64(1), wl, Int64(64), x, n1, b, Int64(0), Int64(0)))
        end

        t -= nc

        if t < nr                                    # L₁₁[r, J]
            r0 = j1 + 1 + 64t; x = at(l11, n1, r0, j0)
            return (true, (x, n1, Int64(0), min(64, n1 - r0 + 1), b, Int64(1), x, n1, wu, Int64(64), b, Int64(0), Int64(0)))
        end

        t -= nr

        if t < nc                                    # L₂₁[r, J]
            r0 = 1 + 64t; x = at(l21, n2, r0, j0)
            return (true, (x, n2, Int64(0), min(64, n2 - r0 + 1), b, Int64(1), x, n2, wu, Int64(64), b, Int64(0), Int64(0)))
        end

        return none
    end

    if t < nr * nr                                   # L₁₁[R, R]
        i, j = cm_index(t, nr)
        r0 = j1 + 64i - 63; c0 = j1 + 64j - 63
        return (true, (at(l11, n1, r0, c0), n1, Int64(0), min(64, n1 - r0 + 1), min(64, n1 - c0 + 1), Int64(0),
            at(l11, n1, r0, j0), n1, at(l11, n1, j0, c0), n1, b, Int64(0), Int64(0)))
    end

    t -= nr * nr

    if t < nr * nc                                   # U₁₂[R, :]
        i, j = cm_index(t, nr)
        r0 = j1 + 64i - 63; c0 = 64j - 63
        return (true, (at(u12, n1, r0, c0), n1, Int64(0), min(64, n1 - r0 + 1), min(64, n2 - c0 + 1), Int64(0),
            at(l11, n1, r0, j0), n1, at(u12, n1, j0, c0), n1, b, Int64(0), Int64(0)))
    end

    t -= nr * nc

    if t < nc * nr                                   # L₂₁[:, R]
        i, j = cm_index(t, nc)
        r0 = 64i - 63; c0 = j1 + 64j - 63
        return (true, (at(l21, n2, r0, c0), n2, Int64(0), min(64, n2 - r0 + 1), min(64, n1 - c0 + 1), Int64(0),
            at(l21, n2, r0, j0), n2, at(l11, n1, j0, c0), n1, b, Int64(0), Int64(0)))
    end

    return none
end

# panels, trailing updates or Schur complements of the fronts foff + 1, … of a level: block (t, f) does
# tile t of front f; TL, TU of front f at work + 2 (f - 1) 64² (as written by level_diag). The first tile
# of a trailing update is the next diagonal block, L₁₁[J + 1, J + 1]: its block then factors it at once
# (level_diag for step k + 1), while the other blocks finish the update.
function level_tile_kernel!(s::AbstractSemiring, op, kind::Val{KIND}, scale::Val, idem::Val, ::Type{T}, fronts, foff::Int32, k::Int64,
        work::Int64) where {KIND, T}
    As, Bt, DG = batch_shared(T)
    f = Int(blockIdx().y); t = Int(blockIdx().x) - 1
    fr = @inbounds fronts[foff + f]
    wl = work + 2 * (f - 1) * 4096 * sizeof(T)
    ok, task = level_tile(kind, fr, k, t, wl, wl + 4096 * sizeof(T), T)
    ok || return
    tile_task!(s, op, T, As, Bt, task...)

    if KIND == TB_TRAIL && t == 0
        sync_threads()                                      # (the block's global writes are visible to it)
        level_diag(s, op, scale, idem, T, As, Bt, DG, fr, k + 1, wl)
    end

    return
end

# the k-th diagonal block of each front of a level that has one (as diag_block_kernel!, LU)
function level_diag_kernel!(s::AbstractSemiring, op, scale::Val, idem::Val, ::Type{T}, fronts, foff::Int32, k::Int64, work::Int64) where {T}
    f = Int(blockIdx().x)
    level_diag(s, op, scale, idem, T, batch_shared(T)..., @inbounds(fronts[foff + f]), k, work + 2 * (f - 1) * 4096 * sizeof(T))
    return
end

@inline function level_diag(s, op, scale::Val, idem::Val, ::Type{T}, CH, RH, DG, fr::NTuple{9, Int64}, k::Int64, wl::Int64) where {T}
    l11, u11, _, _, _, n1, _, _, _ = fr
    j0 = 64k - 63
    j0 <= n1 || return
    sz = sizeof(T)
    e = ((j0 - 1) + (j0 - 1) * n1) * sz
    diag_block!(s, op, scale, Val(true), idem, T, CH, RH, DG, l11 + e, l11 + e, n1, min(64, n1 - j0 + 1), l11 + e, u11 + e, wl, Int64(64),
        wl + 4096 * sz, Int64(64))
    return
end

# (op, a Val or nothing that only the device decides, through a function barrier: a union-typed op
# in the launch loop made the host compiler crash, LLVM 18 LazyCallGraph)
factor_top_batched!(P::FactorPlan{Sem, T}) where {Sem, T} = factor_top_batched!(P, pair_op(P.F.s, T))

function factor_top_batched!(P::FactorPlan{Sem, T}, op) where {Sem, T}
    s = P.F.s
    scale = Val(!isintegral(s))
    B = top_batch(P)
    work::Int64 = devaddr(B.work)
    idem = Val(idem_plus(s, T))

    for (what, off, n, w, k) in B.launches
        o = Int32(off)

        if what == TB_ASM
            @phase FTIMER[] :assemble @cuda threads = ASM_NT blocks = (w, n) assemble_kernel!(s, T, B.fronts, B.kids, P.reltgt, o)
        elseif what == TB_DIAG
            @phase FTIMER[] :lu_diag @cuda threads = TB_NT blocks = n level_diag_kernel!(s, op, scale, idem, T, B.fronts, o, k, work)
        elseif what == TB_PANEL
            @phase FTIMER[] :panel_trsm @cuda threads = TB_NT blocks = (w, n) level_tile_kernel!(s, op, Val(TB_PANEL), scale, idem, T, B.fronts, o, k, work)
        elseif what == TB_TRAIL
            @phase FTIMER[] :lu_gemm @cuda threads = TB_NT blocks = (w, n) level_tile_kernel!(s, op, Val(TB_TRAIL), scale, idem, T, B.fronts, o, k, work)
        else
            @phase FTIMER[] :schur_gemm @cuda threads = TB_NT blocks = (w, n) level_tile_kernel!(s, op, Val(TB_SCHUR), scale, idem, T, B.fronts, o, k, work)
        end
    end

    return P
end
