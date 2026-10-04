# Fast fill-reducing orderings for large sparse graphs (APSP / multifrontal LU front end):
#
#   HubAMF     AMF on the graph without its hubs (degree > alpha·√n), the hubs ordered last.
#   BFSND      nested dissection with breadth-first level-structure separators for the top levels of the
#              separator tree, halo AMF on the parts, the parts ordered in parallel.
#   AutoOrder  the per-graph choice between HubAMF, BFSND and AMF from degree statistics (and, for
#              BFSND, the depth of one breadth-first search), O(n) to decide.
#   Natural    the identity order (leaves of a partial nested dissection kept as dense fronts).
#
# halo_amf orders part of a graph with MUMPS' halo AMF (AMFLib): the halo vertices count in the degrees
# and fill of the others but are eliminated last.

"""
    Natural <: EliminationAlgorithm

The identity ordering.
"""
struct Natural <: EliminationAlgorithm end

function permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::Natural) where {V}
    n = nv(graph); order = collect(oneto(V(n)))
    return order, copy(order)
end

# ===== halo AMF =====

"""
    halo_amf(n0, n1, xadj, adjncy)

Approximate minimum fill order of the vertices `1:n0` of the graph with `n0 + n1` vertices whose adjacency
is `xadj`, `adjncy` (1-based, symmetric, no self loops; the edges among the last `n1` vertices may be
omitted). The last `n1` vertices (the halo) count in the degrees and fill of the others but are not
ordered. Returns the order of `1:n0` (a vector of length `n0`).
"""
function halo_amf(n0::Integer, n1::Integer, xadj::AbstractVector{E}, adjncy::AbstractVector{V}) where {V <: Integer, E <: Integer}
    n = V(n0 + n1); m = E(xadj[n + 1] - 1)
    norig = n; nbbuck = twice(norig)
    iwlen = m + E(4n + min(10000, m + n))
    len = FVector{V}(undef, n); pe = FVector{E}(undef, n + 1); iw = FVector{V}(undef, iwlen)
    nvv = FVector{V}(undef, n); elen = FVector{V}(undef, n); last = FVector{V}(undef, n)
    degree = FVector{V}(undef, n); wf = FVector{E}(undef, n); next = FVector{V}(undef, n)
    w = FVector{Int}(undef, n); head = FVector{V}(undef, nbbuck + two(V))

    @inbounds for p in oneto(m)
        iw[p] = adjncy[p]
    end

    @inbounds for i in oneto(n)
        pe[i] = xadj[i]; nvv[i] = one(V)
        d = V(xadj[i + 1] - xadj[i])
        len[i] = i <= n0 ? d : (iszero(d) ? -norig - one(V) : -d)
    end

    @inbounds pe[n + 1] = xadj[n + 1]
    AMFLib.hamf_impl!(norig, n, zero(V), nbbuck, iwlen, pe, m + one(E), len, iw, nvv, elen, last, degree, wf, next, w, head)

    # elen[i] = ± the position of i; keep the vertices 1:n0 in that order
    pos = head

    @inbounds for j in oneto(norig)
        pos[j] = zero(V)
    end

    @inbounds for i in oneto(n)
        pos[abs(elen[i])] = i
    end

    order = Vector{V}(undef, n0); k = 0

    @inbounds for j in oneto(norig)
        i = pos[j]

        if ispositive(i) && i <= n0
            k += 1; order[k] = i
        end
    end

    @assert k == n0
    return order
end

# the graph's adjacency as 1-based Int32 vectors (no self loops); I = Int64 if it does not fit
function int_adjacency(graph::AbstractGraph)
    g = BipartiteGraph(graph); n = nv(g); ptr = pointers(g); tgt = targets(g); m = ptr[n + 1] - 1
    I = max(n, m) < typemax(Int32) ? Int32 : Int64
    xadj = Vector{Int}(undef, n + 1); adj = Vector{I}(undef, m); q = 0

    @inbounds for v in oneto(n)
        xadj[v] = q + 1

        for p in ptr[v]:ptr[v + 1] - 1
            w = tgt[p]
            w != v && (q += 1; adj[q] = w)
        end
    end

    @inbounds xadj[n + 1] = q + 1
    resize!(adj, q)
    return n, xadj, adj
end

# ===== HubAMF =====

"""
    HubAMF(alg = AMF(); alpha = 2.0)

Order the hubs — the vertices of degree greater than `max(16, alpha·√n)` — last (by increasing degree),
and the rest of the graph with `alg`, as AMD does with its dense rows. On graphs with a power-law degree
distribution AMF spends most of its time updating the hubs, which end up in the root front anyway.
"""
struct HubAMF{A <: EliminationAlgorithm} <: EliminationAlgorithm
    alg::A
    alpha::Float64
end

HubAMF(alg::EliminationAlgorithm = AMF(); alpha::Real = 2.0) = HubAMF(alg, Float64(alpha))

function permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::HubAMF) where {V}
    n, xadj, adj = int_adjacency(graph)
    thr = max(16.0, alg.alpha * sqrt(n))
    order = hubamf(n, xadj, adj, thr, alg.alg, weights)
    out = convert(Vector{V}, order)
    return out, invperm(out)
end

function hubamf(n::Int, xadj::Vector{Int}, adj::Vector{I}, thr::Float64, alg::EliminationAlgorithm, weights) where {I}
    nhub = 0

    @inbounds for v in oneto(n)
        xadj[v + 1] - xadj[v] > thr && (nhub += 1)
    end

    if iszero(nhub)
        return leaforder(n, xadj, adj, alg, weights)
    end

    map = zeros(I, n); keep = Vector{I}(undef, n - nhub); hubs = Vector{I}(undef, nhub); k = 0; h = 0

    @inbounds for v in oneto(n)
        if xadj[v + 1] - xadj[v] > thr
            h += 1; hubs[h] = v
        else
            k += 1; keep[k] = v; map[v] = k
        end
    end

    sxadj = Vector{Int}(undef, k + 1); sxadj[1] = 1; q = 0

    @inbounds for (i, v) in enumerate(keep)
        for p in xadj[v]:xadj[v + 1] - 1
            !iszero(map[adj[p]]) && (q += 1)
        end

        sxadj[i + 1] = q + 1
    end

    sadj = Vector{I}(undef, q); q = 0

    @inbounds for v in keep, p in xadj[v]:xadj[v + 1] - 1
        w = map[adj[p]]
        !iszero(w) && (q += 1; sadj[q] = w)
    end

    so = leaforder(k, sxadj, sadj, alg, view(weights, keep))
    sort!(hubs; by = v -> xadj[v + 1] - xadj[v])
    order = Vector{I}(undef, n)

    @inbounds for i in oneto(k)
        order[i] = keep[so[i]]
    end

    @inbounds for j in oneto(nhub)
        order[k + j] = hubs[j]
    end

    return order
end

# order a graph given as (n, xadj, adj) with alg (AMF: AMFLib directly, no copy of the graph)
function leaforder(n::Int, xadj::Vector{Int}, adj::Vector{I}, alg::EliminationAlgorithm, weights) where {I}
    if alg isa AMF
        iszero(n) && return I[]
        return halo_amf(I(n), zero(I), xadj, adj)
    else
        g = BipartiteGraph{I, Int}(n, n, length(adj), xadj, adj)
        order, _ = permutation(weights, g, alg)
        return convert(Vector{I}, order)
    end
end

# ===== BFSND =====

"""
    BFSND(; leaf = AMF(), levels = 4, minsize = 1024, maxsep = 0.1, fm = 0, threads = true)

Nested dissection for the top `levels` levels of the separator tree, with breadth-first level-structure
separators (George & Liu): from a pseudo-peripheral vertex, the level that best balances the two sides
against its size, thinned (a vertex without a neighbor beyond the level moves to the near side) and
optionally improved by `fm` passes of node Fiduccia–Mattheyses refinement. The parts are ordered with
`leaf` — halo AMF when `leaf` is `AMF()`, so that a part's order sees the separators around it — and the
two sides of every separator in parallel when `threads`. A subgraph is not split when it has fewer than
`minsize` vertices or no separator with at most `maxsep` of its vertices (a graph without small
separators, e.g. a social network, is then ordered with `leaf` after two breadth-first searches).
Disconnected subgraphs are split into their components. Separator vertices are ordered last, in
breadth-first order.

Breadth-first separators are about as small as METIS' on grids and regular meshes (where this order has
10–25% less fill than AMF) and much larger on irregular meshes and geometric graphs (where it has more).
"""
Base.@kwdef struct BFSND{A <: EliminationAlgorithm} <: EliminationAlgorithm
    leaf::A = AMF()
    levels::Int = 4
    minsize::Int = 1024
    maxsep::Float64 = 0.1
    fm::Int = 0
    fmlimit::Int = 64
    balance::Float64 = 0.65
    threads::Bool = true
    depthcheck::Float64 = 0.0   # (AutoOrder) at the root: no split unless BFS depth ≥ depthcheck·√n
end

struct NDState{I}
    xadj::Vector{Int}
    adj::Vector{I}
    owner::Vector{Int32}     # subproblem of each vertex (0: in a separator)
    level::Vector{Int32}     # breadth-first level within its subproblem
    loc::Vector{I}           # index within its part
    part::Vector{Int8}       # side during a split: 0, 1, or 2 (separator)
    ca::Vector{Int32}        # FM: neighbors on side 0 / side 1 of a separator vertex
    cb::Vector{Int32}
    locked::Vector{Bool}
    order::Vector{I}
    nextid::Threads.Atomic{Int32}
end

function permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::BFSND) where {V}
    n, xadj, adj = int_adjacency(graph)
    order = bfsnd(n, xadj, adj, alg, weights)
    out = convert(Vector{V}, order)
    return out, invperm(out)
end

function bfsnd(n::Int, xadj::Vector{Int}, adj::Vector{I}, alg::BFSND, weights) where {I}
    n < alg.minsize && return leaforder(n, xadj, adj, alg.leaf, weights)
    fm = alg.fm > 0
    st = NDState{I}(xadj, adj, ones(Int32, n), zeros(Int32, n), Vector{I}(undef, n), Vector{Int8}(undef, n),
        fm ? zeros(Int32, n) : Int32[], fm ? zeros(Int32, n) : Int32[], fm ? zeros(Bool, n) : Bool[],
        Vector{I}(undef, n), Threads.Atomic{Int32}(1))
    ndsplit!(st, alg, collect(oneto(I(n))), Int32(1), 1, 0, weights)
    return st.order
end

newid!(st::NDState) = Threads.atomic_add!(st.nextid, Int32(1)) + Int32(1)

# breadth-first search from s over subproblem id; levels from 1, queue[1:count]
function bfs!(st::NDState{I}, id::Int32, s::I, queue::Vector{I}) where {I}
    xadj = st.xadj; adj = st.adj; owner = st.owner; level = st.level

    @inbounds begin
        head = 1; tail = 1; queue[1] = s; level[s] = Int32(1)

        while head <= tail
            v = queue[head]; head += 1; lv = level[v] + Int32(1)

            for p in xadj[v]:xadj[v + 1] - 1
                w = adj[p]

                if owner[w] == id && iszero(level[w])
                    level[w] = lv; tail += 1; queue[tail] = w
                end
            end
        end
    end

    return tail
end

function clearlevels!(st::NDState, vs)
    level = st.level

    @inbounds for v in vs
        level[v] = Int32(0)
    end
end

function ndsplit!(st::NDState{I}, alg::BFSND, vs::Vector{I}, id::Int32, lo::Int, depth::Int, weights) where {I}
    k = length(vs)

    if depth >= alg.levels || k < alg.minsize
        return ndleaf!(st, alg, vs, id, lo, weights)
    end

    xadj = st.xadj; adj = st.adj; owner = st.owner; level = st.level; part = st.part
    queue = Vector{I}(undef, k)

    # a pseudo-peripheral vertex (George & Liu): from a vertex of minimum degree, then the vertex of minimum
    # degree in the last level, while the depth grows (at most two more searches)
    s = vs[1]; ds = xadj[s + 1] - xadj[s]

    @inbounds for v in vs
        d = xadj[v + 1] - xadj[v]
        d < ds && (s = v; ds = d)
    end

    clearlevels!(st, vs)
    cnt = bfs!(st, id, s, queue)
    cnt < k && return ndcomponents!(st, alg, vs, id, lo, depth, queue, cnt, weights)

    # (the diameter is at most twice this eccentricity: fail the depth check without more searches)
    if iszero(depth) && 2level[queue[k]] < alg.depthcheck * sqrt(k)
        return ndleaf!(st, alg, vs, id, lo, weights)
    end

    for sweep in 1:2
        last = level[queue[k]]; u = queue[k]; du = xadj[u + 1] - xadj[u]

        @inbounds for i in k:-1:1
            v = queue[i]; level[v] != last && break
            d = xadj[v + 1] - xadj[v]
            d < du && (u = v; du = d)
        end

        clearlevels!(st, vs)
        bfs!(st, id, u, queue)
        level[queue[k]] <= last && break
    end

    depthL = Int(level[queue[k]])

    if depthL < 3 || (iszero(depth) && depthL < alg.depthcheck * sqrt(k))
        return ndleaf!(st, alg, vs, id, lo, weights)
    end

    sizes = zeros(Int, depthL)

    @inbounds for v in vs
        sizes[level[v]] += 1
    end

    # the level l minimizing |S| / min(|A|, |B|) with both sides ≥ 1/4
    best = 0; bestr = Inf; c = sizes[1]

    @inbounds for l in 2:depthL - 1
        a = c; b = k - c - sizes[l]

        if 4a >= k && 4b >= k
            r = sizes[l] / min(a, b)
            r < bestr && (best = l; bestr = r)
        end

        c += sizes[l]
    end

    if iszero(best) || sizes[best] > alg.maxsep * k
        return ndleaf!(st, alg, vs, id, lo, weights)
    end

    sl = Int32(best)

    @inbounds for v in vs
        lv = level[v]

        if lv < sl
            part[v] = Int8(0)
        elseif lv > sl
            part[v] = Int8(1)
        else
            up = false

            for p in xadj[v]:xadj[v + 1] - 1
                w = adj[p]

                if owner[w] == id && level[w] == sl + Int32(1)
                    up = true; break
                end
            end

            part[v] = up ? Int8(2) : Int8(0)
        end
    end

    alg.fm > 0 && fmrefine!(st, alg, vs, id)

    nA = 0; nB = 0

    @inbounds for v in vs
        pv = part[v]; iszero(pv) ? (nA += 1) : isone(pv) && (nB += 1)
    end

    nS = k - nA - nB
    A = Vector{I}(undef, nA); B = Vector{I}(undef, nB); ia = 0; ib = 0; is = 0
    ida = newid!(st); idb = newid!(st)

    # (breadth-first order: separator vertices in it, sides in it as the next searches' input)
    @inbounds for v in queue
        pv = part[v]

        if iszero(pv)
            ia += 1; A[ia] = v; owner[v] = ida
        elseif isone(pv)
            ib += 1; B[ib] = v; owner[v] = idb
        else
            is += 1; st.order[lo + nA + nB + is - 1] = v; owner[v] = Int32(0)
        end
    end

    if alg.threads
        t = Threads.@spawn ndsplit!(st, alg, A, ida, lo, depth + 1, weights)
        ndsplit!(st, alg, B, idb, lo + nA, depth + 1, weights)
        wait(t)
    else
        ndsplit!(st, alg, A, ida, lo, depth + 1, weights)
        ndsplit!(st, alg, B, idb, lo + nA, depth + 1, weights)
    end

    return
end

# a disconnected subproblem: its large components in parallel, the small ones together as one part
function ndcomponents!(st::NDState{I}, alg::BFSND, vs::Vector{I}, id::Int32, lo::Int, depth::Int, queue::Vector{I}, cnt::Int, weights) where {I}
    level = st.level; owner = st.owner
    comps = Vector{I}[queue[1:cnt]]

    @inbounds for v in vs
        if iszero(level[v])
            c = bfs!(st, id, v, queue)
            push!(comps, queue[1:c])
        end
    end

    tasks = Task[]; pos = lo; small = Vector{I}[]

    for c in comps
        if length(c) < alg.minsize
            push!(small, c)
            continue
        end

        cid = newid!(st)

        @inbounds for v in c
            owner[v] = cid
        end

        if alg.threads
            let c = c, cid = cid, p = pos
                push!(tasks, Threads.@spawn ndsplit!(st, alg, c, cid, p, depth, weights))
            end
        else
            ndsplit!(st, alg, c, cid, pos, depth, weights)
        end

        pos += length(c)
    end

    if !isempty(small)
        rest = reduce(vcat, small); rid = newid!(st)

        @inbounds for v in rest
            owner[v] = rid
        end

        ndleaf!(st, alg, rest, rid, pos, weights)
    end

    foreach(wait, tasks)
    return
end

# a part: halo AMF (halo: the adjacent separator vertices), or alg.leaf on the induced subgraph
function ndleaf!(st::NDState{I}, alg::BFSND, vs::Vector{I}, id::Int32, lo::Int, weights) where {I}
    k = length(vs); iszero(k) && return
    xadj = st.xadj; adj = st.adj; owner = st.owner; loc = st.loc
    usehalo = alg.leaf isa AMF

    @inbounds for (i, v) in enumerate(vs)
        loc[v] = I(i)
    end

    halo = I[]

    if usehalo
        @inbounds for v in vs, p in xadj[v]:xadj[v + 1] - 1
            w = adj[p]; owner[w] != id && push!(halo, w)
        end

        sort!(halo); unique!(halo)
    end

    h = length(halo); nn = k + h
    lx = Vector{Int}(undef, nn + 1); lx[1] = 1
    hdeg = zeros(Int, h)

    @inbounds for (i, v) in enumerate(vs)
        d = 0

        for p in xadj[v]:xadj[v + 1] - 1
            w = adj[p]

            if owner[w] == id
                d += 1
            elseif usehalo
                d += 1; hdeg[searchsortedfirst(halo, w)] += 1
            end
        end

        lx[i + 1] = lx[i] + d
    end

    @inbounds for j in oneto(h)
        lx[k + j + 1] = lx[k + j] + hdeg[j]
    end

    la = Vector{I}(undef, lx[nn + 1] - 1); fp = lx[1:nn]

    @inbounds for (i, v) in enumerate(vs), p in xadj[v]:xadj[v + 1] - 1
        w = adj[p]

        if owner[w] == id
            la[fp[i]] = loc[w]; fp[i] += 1
        elseif usehalo
            j = k + searchsortedfirst(halo, w)
            la[fp[i]] = I(j); fp[i] += 1
            la[fp[j]] = I(i); fp[j] += 1
        end
    end

    o = usehalo ? halo_amf(I(k), I(h), lx, la) : leaforder(k, lx, la, alg.leaf, view(weights, vs))

    @inbounds for i in oneto(k)
        st.order[lo + i - 1] = vs[o[i]]
    end

    return
end

# node Fiduccia–Mattheyses refinement of a separator (unit weights; METIS' FM_2WayNodeRefine): move a separator
# vertex to the smaller side, pulling its neighbors on the other side into the separator; hill-climb, keep the best
function fmrefine!(st::NDState{I}, alg::BFSND, vs::Vector{I}, id::Int32) where {I}
    xadj = st.xadj; adj = st.adj; owner = st.owner; part = st.part; ca = st.ca; cb = st.cb; locked = st.locked
    k = length(vs); maxside = floor(Int, alg.balance * k)
    nA = 0; nB = 0; nS = 0

    @inbounds for v in vs
        pv = part[v]; iszero(pv) ? (nA += 1) : isone(pv) ? (nB += 1) : (nS += 1)
    end

    heapA = Tuple{Int32, I}[]; heapB = Tuple{Int32, I}[]
    moved = I[]; pulled = I[]; mark = Int[]

    for pass in 1:alg.fm
        empty!(heapA); empty!(heapB); empty!(moved); empty!(pulled); empty!(mark)

        @inbounds for v in vs
            locked[v] = false

            if part[v] == 2
                a = Int32(0); b = Int32(0)

                for p in xadj[v]:xadj[v + 1] - 1
                    w = adj[p]; owner[w] == id || continue
                    pw = part[w]; iszero(pw) ? (a += one(Int32)) : isone(pw) && (b += one(Int32))
                end

                ca[v] = a; cb[v] = b
                fmpush!(heapA, (b, v)); fmpush!(heapB, (a, v))
            end
        end

        bestS = nS; bestbal = abs(nA - nB); bestpos = 0; nobetter = 0

        while true
            T = nA <= nB ? 0 : 1; v = zero(I); key = Int32(0)

            for attempt in 1:2
                h = iszero(T) ? heapA : heapB

                while !isempty(h)
                    key, v = fmpop!(h)
                    (part[v] == 2 && !locked[v] && key == (iszero(T) ? cb[v] : ca[v])) && break
                    v = zero(I)
                end

                !iszero(v) && (iszero(T) ? nA : nB) + 1 <= maxside && break
                !iszero(v) && (fmpush!(h, (key, v)); v = zero(I))
                T = 1 - T
            end

            iszero(v) && break
            O = Int8(1 - T)
            part[v] = Int8(T); locked[v] = true; nS -= 1
            iszero(T) ? (nA += 1) : (nB += 1)
            push!(moved, v); push!(mark, length(pulled))

            @inbounds for p in xadj[v]:xadj[v + 1] - 1
                w = adj[p]; owner[w] == id || continue
                pw = part[w]

                if pw == 2
                    if iszero(T)
                        ca[w] += one(Int32); locked[w] || fmpush!(heapB, (ca[w], w))
                    else
                        cb[w] += one(Int32); locked[w] || fmpush!(heapA, (cb[w], w))
                    end
                elseif pw == O
                    part[w] = Int8(2); nS += 1
                    iszero(O) ? (nA -= 1) : (nB -= 1)
                    push!(pulled, w)
                    a = Int32(0); b = Int32(0)

                    for q in xadj[w]:xadj[w + 1] - 1
                        x = adj[q]; owner[x] == id || continue
                        px = part[x]

                        if iszero(px)
                            a += one(Int32)
                        elseif isone(px)
                            b += one(Int32)
                        elseif x != w
                            if iszero(O)
                                ca[x] -= one(Int32); locked[x] || fmpush!(heapB, (ca[x], x))
                            else
                                cb[x] -= one(Int32); locked[x] || fmpush!(heapA, (cb[x], x))
                            end
                        end
                    end

                    ca[w] = a; cb[w] = b
                    fmpush!(heapA, (b, w)); fmpush!(heapB, (a, w))
                end
            end

            bal = abs(nA - nB)

            if nS < bestS || (nS == bestS && bal < bestbal)
                bestS = nS; bestbal = bal; bestpos = length(moved); nobetter = 0
            else
                nobetter += 1
                nobetter > alg.fmlimit && break
            end
        end

        @inbounds for i in length(moved):-1:bestpos + 1
            v = moved[i]; T = part[v]; O = Int8(1 - T)

            for j in mark[i] + 1:(i < length(moved) ? mark[i + 1] : length(pulled))
                w = pulled[j]; part[w] = O; nS -= 1
                iszero(O) ? (nA += 1) : (nB += 1)
            end

            part[v] = Int8(2); nS += 1
            iszero(T) ? (nA -= 1) : (nB -= 1)
        end

        iszero(bestpos) && break
    end

    return
end

function fmpush!(h::Vector{Tuple{Int32, I}}, x) where {I}
    push!(h, x); i = length(h)

    @inbounds while i > 1
        p = i >> 1; h[p] <= h[i] && break
        h[p], h[i] = h[i], h[p]; i = p
    end

    return h
end

function fmpop!(h::Vector{Tuple{Int32, I}}) where {I}
    @inbounds begin
        top = h[1]; x = pop!(h)

        if !isempty(h)
            h[1] = x; i = 1; n = length(h)

            while true
                l = 2i; r = l + 1; m = i
                l <= n && h[l] < h[m] && (m = l)
                r <= n && h[r] < h[m] && (m = r)
                m == i && break
                h[m], h[i] = h[i], h[m]; i = m
            end
        end
    end

    return top
end

# ===== AutoOrder =====

"""
    AutoOrder(; alpha = 2.0, strong = 5.0, alpha_strong = 1.0, lattice = true, nd = BFSND(), minsize = 1000)

A per-graph (per-component) choice, made in O(n) from the degrees:

  - hubs (maximum degree > `alpha·√n`): `HubAMF(; alpha)`, or `HubAMF(; alpha = alpha_strong)` when the
    maximum degree exceeds `strong·√n` (a heavy tail: more of the high-degree vertices end in the root front);
  - lattice-like (≥ 75% of the vertices of the most common degree, coefficient of variation of the degrees
    ≤ 0.15, maximum degree ≤ 8): `nd` (BFSND), which on grids and regular meshes has both less fill than
    AMF and runs in parallel; for most common degree ≤ 4 the breadth-first search from a pseudo-peripheral
    vertex must also reach depth ≥ 1.5√n (a planar lattice), or AMF is used;
  - otherwise (and below `minsize` vertices): AMF.
"""
Base.@kwdef struct AutoOrder{D <: BFSND} <: EliminationAlgorithm
    alpha::Float64 = 2.0
    strong::Float64 = 5.0
    alpha_strong::Float64 = 1.0
    lattice::Bool = true
    nd::D = BFSND()
    minsize::Int = 1000
end

function permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::AutoOrder) where {V}
    n, xadj, adj = int_adjacency(graph)
    order = autoorder(n, xadj, adj, alg, weights)
    out = convert(Vector{V}, order)
    return out, invperm(out)
end

# the choice of AutoOrder from the degrees: (:amf, _) | (:hub, alpha) | (:nd, depth check)
function autochoice(n::Int, xadj::Vector{Int}, alg::AutoOrder)
    n < alg.minsize && return :amf, 0.0
    dmax = 0; s1 = 0.0; s2 = 0.0

    @inbounds for v in oneto(n)
        d = xadj[v + 1] - xadj[v]; dmax = max(dmax, d); s1 += d; s2 += d * d
    end

    dmax > alg.strong * sqrt(n) && return :hub, alg.alpha_strong
    dmax > alg.alpha * sqrt(n) && return :hub, alg.alpha
    alg.lattice || return :amf, 0.0
    dmax <= 8 || return :amf, 0.0
    cnt = zeros(Int, dmax + 1)

    @inbounds for v in oneto(n)
        cnt[xadj[v + 1] - xadj[v] + 1] += 1
    end

    mode = argmax(cnt) - 1; mean = s1 / n; cv = sqrt(max(0.0, s2 / n - mean^2)) / mean
    (cnt[mode + 1] >= 0.75n && cv <= 0.15) || return :amf, 0.0
    return :nd, mode <= 4 ? 1.5 : 0.0
end

function autoorder(n::Int, xadj::Vector{Int}, adj::Vector{I}, alg::AutoOrder, weights) where {I}
    choice, x = autochoice(n, xadj, alg)

    if choice === :hub
        return hubamf(n, xadj, adj, max(16.0, x * sqrt(n)), AMF(), weights)
    elseif choice === :nd
        nd = alg.nd
        nd = BFSND(nd.leaf, nd.levels, nd.minsize, nd.maxsep, nd.fm, nd.fmlimit, nd.balance, nd.threads, x)
        return bfsnd(n, xadj, adj, nd, weights)
    else
        return leaforder(n, xadj, adj, AMF(), weights)
    end
end

function Base.show(io::IO, ::MIME"text/plain", alg::HubAMF)
    print(io, "HubAMF(alpha = $(alg.alpha)) of $(alg.alg)")
end
