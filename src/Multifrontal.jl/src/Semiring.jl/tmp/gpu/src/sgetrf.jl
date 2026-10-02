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

# set to a Dict{Symbol, Float64} to profile the factorization kernels by type (synchronizes)
const FTIMER = Ref{Any}(nothing)

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
        @phase FTIMER[] :lu_diag @cuda threads = 256 sgetrf_diag_kernel!(s, scale, Akk)

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
            @cuda threads = 32 * cld(b, 32) strsx_left_diag_kernel!(s, trans, Tj, view(A, J, J))
            Wj = reshape(view(W, 1:(b * m)), b, m)
            fill!(Wj, szero(s, T, Val(:N)))
            sgemx_gpu!(s, Wj, Tj, view(X, J, :))
            copy_gpu!(view(X, J, :), Wj)
        else
            tb = min(128, 32 * cld(m, 32))
            @cuda threads = tb blocks = cld(m, tb) strsx_left_diag_kernel!(s, trans, view(X, J, :), view(A, J, J))
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
end

function FactorPlan(F::ChordalSLU{Sem, T, I}; large::Integer = 256, nt::Integer = Threads.nthreads(), graph::Bool = true,
        nstreams::Integer = 8, maxbytes::Integer = 2^31) where {Sem, T, I}
    s = F.s
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

# GPU: issue the top fronts, ordered after the current stream
function factor_top!(P::FactorPlan{Sem, T}) where {Sem, T}
    s = P.F.s
    scale = Val(!isintegral(s))

    run(t, Fbuf) = let (n₁, n₂, Dp, Lp, out, kids) = P.tasks[t]
        factor_front_gpu!(s, scale, Fbuf, P.LD, P.UD, P.LL, P.UL, P.Mg, out, n₁, n₂, Dp, Lp,
            ((k[1] ? P.Mg : P.Mb, k[2], k[3], k[4]) for k in kids), P.relptr, P.reltgt)
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
        g = CUDA.capture() do
            factor_top!(P)
        end

        P.exec = CUDA.instantiate(g)
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
        n₁::Int, n₂::Int, Dp::Int, Lp::Int, kids, relptr_g, reltgt_g) where {T}
    nj = n₁ + n₂
    Fj = reshape(view(Fbuf, 1:(nj * nj)), nj, nj)
    @phase FTIMER[] :assemble fill!(Fj, szero(s, T, Val(:N)))

    for (buf, off, nac, rp) in kids
        @phase FTIMER[] :assemble extendadd_gpu!(s, Fj, buf, off, nac, relptr_g, reltgt_g, rp)
    end

    L₁₁ = reshape(view(LD, Dp:(Dp + n₁ * n₁ - 1)), n₁, n₁)
    U₁₁ = reshape(view(UD, Dp:(Dp + n₁ * n₁ - 1)), n₁, n₁)
    @phase FTIMER[] :assemble combine_gpu!(s, L₁₁, U₁₁, Fj)
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
