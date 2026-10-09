"""
    DissectionAlgorithm

A vertex separator algorithm.
"""
abstract type DissectionAlgorithm end

"""
    METISND <: DissectionAlgorithm

    METISND(; nseps=-1, seed=-1)

Compute a vertex separator using the graph partitioning library METIS.

### Parameters

  - `nseps`: number of different separators computed at each level of nested dissection
  - `seed`: random seed

### References

  - Karypis, George, and Vipin Kumar. "A fast and high quality multilevel scheme for partitioning irregular graphs." *SIAM Journal on Scientific Computing* 20.1 (1998): 359-392.
"""
@kwdef struct METISND <: DissectionAlgorithm
    nseps::Int = -1
    seed::Int = -1
end

"""
    KaHyParND{O} <: DissectionAlgorithm

    KaHyParND(order; beta=1.0)

Compute a vertex separator using the hypergraph partitioning library KaHyPar. A β-quasi-clique cover is constructed
using a greedy algorithm controlled by the parameters `order` and `beta`.

### Parameters

  - `order`: tie breaking strategy (`Forward` or `Reverse`).
  - `beta`: quasi-clique parameter

### References

  - Çatalyürek, Ümit V., Cevdet Aykanat, and Enver Kayaaslan. "Hypergraph partitioning-based fill-reducing ordering for symmetric matrices." *SIAM Journal on Scientific Computing* 33.4 (2011): 1996-2023.
  - Kaya, Oguz, et al. "Fill-in reduction in sparse matrix factorizations using hypergraphs".
"""
struct KaHyParND{O <: Ordering} <: DissectionAlgorithm
    order::O
    beta::Float64
end

function KaHyParND(order::Ordering = Forward; beta::Number = 1.0)
    return KaHyParND(order, beta)
end

# Turn a bipartition `hpart` of the cliques of a clique cover into a vertex
# separator of the covered graph: a vertex is in A (part 0) or B (part 1) if
# all of its cliques lie on that side, and in S (part 2) if they lie on both.
# The cliques of each side are numbered 2, 3, ... in `hproject0` and
# `hproject1`; number 1 is left for S, which is a clique of both children.
# Returns the number of cliques of each child, counting S.
function hseparator!(
        hproject0::AbstractVector{V},
        hproject1::AbstractVector{V},
        hpart::AbstractVector,
        part::AbstractVector{V},
        hgraph::AbstractGraph,
    ) where {V}
    @assert nov(hgraph) <= length(hproject0)
    @assert nov(hgraph) <= length(hproject1)
    @assert nv(hgraph) <= length(part)

    h0 = one(V)
    h1 = one(V)

    for hv in outvertices(hgraph)
        hvv = hpart[hv]

        if iszero(hvv)
            hproject0[hv] = h0 += one(V)
            hproject1[hv] = zero(V)
        else
            hproject1[hv] = h1 += one(V)
            hproject0[hv] = zero(V)
        end
    end

    # V = W ∪ B
    for v in vertices(hgraph)
        vv = three(V)

        for hv in neighbors(hgraph, v)
            hvv = hpart[hv]

            if iszero(hvv) # v ∈ W
                if isthree(vv)
                    vv = zero(V)
                elseif isone(vv)  # v ∈ W ∩ B
                    vv = two(V)
                end
            else          # v ∈ B
                if isthree(vv)
                    vv = one(V)
                elseif iszero(vv) # v ∈ W ∩ B
                    vv = two(V)
                end
            end
        end

        if isthree(vv)
            vv = zero(V)
        end

        part[v] = vv
    end

    return h0, h1
end

# Merge the twins of a simple graph. If there are none, return the graph
# itself with the identity map, so that its vertex order is kept.
function compresstwins(weights::AbstractVector{W}, graph::BipartiteGraph{V, E}) where {W, V, E}
    n = nv(graph)
    cmpgraph, project = compress(graph, Val(true))

    if nv(cmpgraph) < n
        cmpweights = compressweights(weights, project)
    else
        nn = n + one(V)
        cmpgraph = BipartiteGraph{V, E}(n, n, ne(graph))
        project = BipartiteGraph{V, V}(n, n, n)
        cmpweights = FVector{W}(undef, n)

        @inbounds for v in oneto(nn)
            pointers(cmpgraph)[v] = pointers(graph)[v]
            pointers(project)[v] = v
        end

        @inbounds for p in oneto(ne(graph))
            targets(cmpgraph)[p] = targets(graph)[p]
        end

        @inbounds for v in oneto(n)
            targets(project)[v] = v
            cmpweights[v] = weights[v]
        end
    end

    return cmpgraph, cmpweights, project
end

# The separator S = { v : part[v] = 2 } of a twin-free graph, together with
#
#     count[v] = | N(v) ∩ S |
#
# and, for each x ∈ S, the size and checksum of N(x) ∩ A (side 0) and N(x) ∩ B
# (side 1): the keys of S in the two children.
function twinfreekeys!(
        count::AbstractVector{V},
        degree0::AbstractVector{V},
        degree1::AbstractVector{V},
        part::AbstractVector{V},
        graph::AbstractGraph{V},
    ) where {V}
    @assert nv(graph) <= length(count)
    @assert nv(graph) <= length(part)

    # S = W ∩ B
    n2 = zero(V)

    @inbounds for v in vertices(graph)
        if istwo(part[v])
            n2 += one(V)
        end
    end

    @assert n2 <= length(degree0)
    @assert n2 <= length(degree1)
    label2 = FVector{V}(undef, n2); t2 = zero(V)

    @inbounds for v in vertices(graph)
        if istwo(part[v])
            t2 += one(V); label2[t2] = v
        end
    end

    @inbounds for v in vertices(graph)
        count[v] = zero(V)
    end

    checksum0 = FVector{UInt64}(undef, n2)
    checksum1 = FVector{UInt64}(undef, n2)

    @inbounds for i in oneto(n2)
        x = label2[i]
        d0 = zero(V); h0 = zero(UInt64)
        d1 = zero(V); h1 = zero(UInt64)

        for w in neighbors(graph, x)
            count[w] += one(V); pw = part[w]

            if iszero(pw)    # w ∈ W - B
                d0 += one(V); h0 += twinhash(w)
            elseif isone(pw) # w ∈ B - W
                d1 += one(V); h1 += twinhash(w)
            end
        end

        degree0[i] = d0; checksum0[i] = h0
        degree1[i] = d1; checksum1[i] = h1
    end

    return label2, checksum0, checksum1
end

# splitmix64 finalizer
@inline function twinhash(v::Integer)
    x = convert(UInt64, v) * 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    return x ⊻ (x >> 31)
end

# v ∈ S ∪ A*, where A* = { u on `side` : S ⊆ N(u) }
@inline function twinfreeinx(v::V, side::V, n2::V, part::AbstractVector{V}, count::AbstractVector{V}) where {V}
    @inbounds pv = part[v]
    return istwo(pv) || (pv == side && ispositive(n2) && @inbounds count[v] == n2)
end

# Split a twin-free graph G, i.e. a graph in which no two vertices have the
# same closed neighborhood, along a vertex separator, and compress the two
# children. The children are again twin-free, so the input graph only needs
# to be compressed once, before the dissection starts.
#
# Let V = A ∪ S ∪ B, where S separates A from B, and let
#
#     G₀ := G[A ∪ S] + K(S)
#
# be the first child. Its closed neighborhoods are
#
#     N₀[u] = N[u]               for u ∈ A
#     N₀[x] = (N(x) ∩ A) ∪ S     for x ∈ S.
#
# Hence two vertices of A are twins in G₀ if and only if they are twins in G,
# which never happens. A vertex u ∈ A can be the twin of a vertex x ∈ S only
# if S ⊆ N[u]. Call the set of such vertices A*. Every y ∈ S ∪ A* satisfies
#
#     N₀[y] = (N[y] ∩ A) ∪ S,
#
# so the twin classes of G₀ are the singletons in A - A*, together with the
# classes of S ∪ A* under the key
#
#     y ↦ N[y] ∩ A.
#
# We group S ∪ A* by the size and checksum of each key, confirm each group
# exactly with a marker, and then build the quotient graph directly from G.
# The keys of S for both children are read in a single pass over N(S).

# The twin classes of a child of a twin-free graph (steps 1–3 below), without
# building the child. On return, `project` maps every vertex v of the child
# (part[v] ∈ {side, 2}) to its class, the classes are numbered by their first
# vertex, `xrep[u]` is the first vertex of class u, and `class`/`size` describe
# the classes of S ∪ A*. Returns the number of classes and of vertices.
function twinfreeclasses!(
        mark::AbstractVector{V},
        class::AbstractVector{V},
        size::AbstractVector{V},
        xrep::AbstractVector{V},
        project::AbstractVector{V},
        sdegree::AbstractVector{V},
        schecksum::AbstractVector{UInt64},
        count::AbstractVector{V},
        label2::AbstractVector{V},
        side::V,
        part::AbstractVector{V},
        graph::AbstractGraph{V},
    ) where {V}
    @assert nv(graph) <= length(mark)
    @assert nv(graph) <= length(class)
    @assert nv(graph) <= length(size)
    @assert nv(graph) <= length(xrep)
    @assert nv(graph) <= length(project)
    n = nv(graph); n2 = convert(V, length(label2))

    # v ∈ S ∪ A*
    @inline inx(v::V) = twinfreeinx(v, side, n2, part, count)

    ##############################
    # 1. X = S ∪ A* and its keys #
    ##############################

    nx = zero(V)

    if ispositive(n2)
        @inbounds for x in label2
            nx += one(V); xrep[nx] = x
        end

        # A* ⊆ N(x) for every x ∈ S
        @inbounds for w in neighbors(graph, first(label2))
            if part[w] == side && count[w] == n2
                nx += one(V); xrep[nx] = w
            end
        end
    end

    xdegree = FVector{V}(undef, nx)
    xchecksum = FVector{UInt64}(undef, nx)

    @inbounds for i in oneto(n2)
        xdegree[i] = sdegree[i]
        xchecksum[i] = schecksum[i]
    end

    @inbounds for i in n2 + one(V):nx # y ∈ A*
        y = xrep[i]; d = one(V); h = twinhash(y)

        for w in neighbors(graph, y)
            if part[w] == side
                d += one(V); h += twinhash(w)
            end
        end

        xdegree[i] = d; xchecksum[i] = h
    end

    #################################################
    # 2. group X by key size and checksum, and      #
    #    confirm each group exactly with a marker   #
    #################################################

    xorder = FVector{V}(undef, nx)

    @inbounds for i in oneto(nx)
        xorder[i] = i
    end

    sort!(xorder; by = i -> (@inbounds (xdegree[i], xchecksum[i])))

    @inbounds for v in oneto(n)
        mark[v] = zero(V)
    end

    nc = zero(V); tag = zero(V); lo = one(V)

    @inbounds while lo <= nx
        # xorder[lo:hi] have the same key size and checksum
        i = xorder[lo]; hi = lo

        while hi < nx && xdegree[xorder[hi + one(V)]] == xdegree[i] && xchecksum[xorder[hi + one(V)]] == xchecksum[i]
            hi += one(V)
        end

        next = hi + one(V)

        while lo <= hi
            # mark the key of the leader
            y = xrep[xorder[lo]]; tag += one(V)

            for w in neighbors(graph, y)
                if part[w] == side
                    mark[w] = tag
                end
            end

            if part[y] == side
                mark[y] = tag
            end

            nc += one(V); class[y] = nc; size[nc] = one(V)

            # compare the other keys against it; keys of equal size
            # are equal iff one is contained in the other
            keep = lo

            for k in lo + one(V):hi
                z = xrep[xorder[k]]; same = part[z] != side || mark[z] == tag

                if same
                    for w in neighbors(graph, z)
                        if part[w] == side && mark[w] != tag
                            same = false; break
                        end
                    end
                end

                if same
                    class[z] = nc; size[nc] += one(V)
                else
                    keep += one(V); xorder[keep] = xorder[k]
                end
            end

            lo += one(V); hi = keep
        end

        lo = next
    end

    #####################################
    # 3. number the classes of the child #
    #####################################

    # `mark` becomes the class → child map
    newid = mark; nsub = zero(V); ncmp = zero(V)

    @inbounds for c in oneto(nc)
        newid[c] = zero(V)
    end

    @inbounds for v in vertices(graph)
        pv = part[v]

        if pv == side || istwo(pv)
            nsub += one(V)

            if inx(v)
                c = class[v]

                if iszero(newid[c])
                    ncmp += one(V); newid[c] = ncmp; xrep[ncmp] = v
                end

                project[v] = newid[c]
            else
                ncmp += one(V); project[v] = ncmp; xrep[ncmp] = v
            end
        end
    end

    return ncmp, nsub
end

function qcc(::Type{V}, ::Type{E}, graph, beta::Number, order::Ordering) where {V, E}
    simple = simplegraph(V, E, graph)
    return qcc!(simple, beta, order)
end

# Fill-in reduction in sparse matrix factorizations using hypergraphs
# Kaya, Kayaaslan, Ucar, and Duff
# Algorithm 1: QCC(G, β)
#
# Construct a β-quasi-clique cover.
# The complexity is O( ∑ |N(v)|² ) ≤ O( Δ|E| ).
function qcc!(graph::BipartiteGraph{V, E}, beta::W, order::Ordering) where {W, V, E}
    @assert zero(W) < beta <= one(W)
    n = nv(graph); m = ne(graph); mm = m + one(E)
    marker = zeros(V, n)

    #### bucket queue (degree) ########################
    degree = FVector{V}(undef, n)
    deghead = FVector{V}(undef, n)
    degprev = FVector{V}(undef, n)
    degnext = FVector{V}(undef, n)

    @inbounds for deg in oneto(n)
        deghead[deg] = zero(V)
    end

    function degset(deg::V)
        @inbounds head = view(deghead, deg + one(V))
        return DoublyLinkedList(head, degprev, degnext)
    end
    ###################################################

    #### bucket queue (score) #########################
    score = FVector{E}(undef, n)
    scrhead = FVector{V}(undef, mm)
    scrprev = FVector{V}(undef, n)
    scrnext = FVector{V}(undef, n)

    @inbounds for scr in oneto(mm)
        scrhead[scr] = zero(V)
    end

    function scrset(scr::E)
        @inbounds head = view(scrhead, scr + one(E))
        return DoublyLinkedList(head, scrprev, scrnext)
    end
    ###################################################

    #### clique cover #################################
    #          cliques
    #          [ x x ]
    # vertices [   x ]
    #          [ x   ]
    ptrC = FVector{E}(undef, mm)
    tgtC = FVector{V}(undef, twice(m))
    vC = one(V); ptrC[vC] = pC = one(E)
    ###################################################

    degmax = zero(V)

    @inbounds for v in vertices(graph)
        deg = eltypedegree(graph, v)
        scr = zero(E)

        degree[v] = deg; pushfirst!(degset(deg), v)
        score[v] = scr; pushfirst!(scrset(scr), v)

        degmax = max(degmax, deg)
    end

    @inbounds while ispositive(m)
        ppC = pC; mC = scrmax = zero(E)

        while isempty(degset(degmax))
            degmax -= one(V)
        end

        v = first(degset(degmax))

        while true
            marker[v] = vC; tgtC[ppC] = v; ppC += one(E)

            for w in neighbors(graph, v)
                if ispositive(w)
                    if marker[w] < vC # w ∈ B
                        scr = score[w]
                        delete!(scrset(scr), w)
                        score[w] = scr += one(E)

                        if !isempty(scrset(scr))
                            ww = first(scrset(scr))

                            if lt(order, degree[ww], degree[w])
                                delete!(scrset(scr), ww)
                                pushfirst!(scrset(scr), w)
                                w = ww
                            end
                        end

                        pushfirst!(scrset(scr), w)
                    else
                        mC += one(E)  # w ∈ C
                    end
                end
            end

            scrmax += one(E)

            while isempty(scrset(scrmax))
                scrmax -= one(E)
            end

            # |E(C)| + score(v)
            left = convert(W, mC + scrmax)

            # |C| (|C| + 1)
            # -------------
            #       2
            right = convert(W, half((ppC - pC) * (ppC - pC + one(E))))

            #   |E(C)| + score(v)
            # 2 ----------------- < β
            #     |C| (|C| + 1)
            left < beta * right && break
            v = popfirst!(scrset(scrmax))
            score[v] = scr = zero(E); pushfirst!(scrset(scr), v)
        end

        while pC < ppC
            v = tgtC[pC]
            deg = degree[v]; delete!(degset(deg), v)
            pstart = pointers(graph)[v]
            pstop = pointers(graph)[v + one(V)] - one(E)

            for p in pstart:pstop
                w = targets(graph)[p]

                if ispositive(w)
                    if marker[w] < vC  # w ∈ B
                        scr = score[w]
                        delete!(scrset(scr), w)
                        score[w] = scr = zero(E)
                        pushfirst!(scrset(scr), w)
                    else               # w ∈ C
                        targets(graph)[p] = zero(V)
                        deg -= one(V); m -= one(E)
                    end
                end
            end

            degree[v] = deg; pushfirst!(degset(deg), v)
            pC += one(E)
        end

        vC += one(V); ptrC[vC] = pC
    end

    nC = vC - one(V)
    mC = pC - one(E)
    return BipartiteGraph(n, nC, mC, ptrC, tgtC)
end

function Base.show(io::IO, ::MIME"text/plain", alg::METISND)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "METISND:")
    println(io, " "^indent * "    nseps: $(alg.nseps)")
    println(io, " "^indent * "    seed: $(alg.seed)")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::KaHyParND{O}) where {O}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "KaHyParND{$O}:")
    println(io, " "^indent * "    order: $(alg.order)")
    println(io, " "^indent * "    beta: $(alg.beta)")
    return
end

"""
    DEFAULT_DISSECTION_ALGORITHM = METISND()

The default dissection algorithm.
"""
const DEFAULT_DISSECTION_ALGORITHM = METISND()

# Nested dissection on one global quotient graph.
#
# The dissection stores, once, a twin-free graph G, together with the
# separators of the ancestors of the node being processed: a stack of
# elements, one per level. A subproblem is a sorted list W of vertices of G,
# together with its twin classes. Its graph is
#
#     G'[W] = G[W] + Σ K(e ∩ W),
#
# where the sum ranges over the separators e on the stack. It is realized on
# demand, already compressed (one vertex per twin class), into buffers that
# are reused by the next realization. Before a node at level L is processed,
# the stack is cut back to its first L separators: the separators of the
# node's ancestors.
#
# The children of a node are numbered and compressed by `twinfreeclasses!`, so
# that every realized graph is twin-free, and the classes of a child are unions
# of classes of its parent.
#
# The state of the dissection is a collection of arrays and scalars, owned
# by the caller and passed to every routine.
#
#   - separators of the ancestors (a stack):
#     - `nelm`: number of separators on the stack
#     - `elmptr`: the members of the separator of the level-(l - 1) ancestor
#       are `pinvtx[elmptr[l]:elmptr[l + 1] - 1]`
#     - `pinvtx`: members of the separators, in stack order
#     - `pinlvl`: level of the separator of each member
#     - `pinnext`: the next (older) entry of the same vertex, or 0
#     - `pinhead`: the newest entry of each vertex, or 0
#   - realization of the current subproblem (W, cls):
#     - `tag`: a running marker tag
#     - `stamp`: `stamp[v] = tag` if and only if v ∈ W
#     - `vclass`: the class of each v ∈ W
#     - `mask`: bit l of `mask[c]` is set if and only if the first vertex of
#       class c lies in the separator of the level-l ancestor
#     - `clsptr`, `clstgt`: the classes whose first vertex lies in the separator
#       of the level-(l - 1) ancestor are `clstgt[clsptr[l]:clsptr[l + 1] - 1]`,
#       in increasing order
#     - `lblptr`, `lbltgt`: the vertices of class c are
#       `lbltgt[lblptr[c]:lblptr[c + 1] - 1]`, in increasing order
#     - `pointer`, `target`: the realized graph
#     - `marker`: marker array
#
# The arrays `elmptr` and `clsptr` have length maxlevel + 2, `marker` has
# length max(n, maxlevel + 2), `pointer` and `lblptr` have length n + 1, and
# the other arrays over vertices or classes have length n. The arrays `pinvtx`,
# `pinlvl`, `pinnext`, `clstgt`, and `target` grow as needed.

# Cut the stack back to its first `level` separators.
function popelements!(
        nelm::AbstractScalar{V},
        elmptr::AbstractVector{V},
        pinvtx::AbstractVector{V},
        pinnext::AbstractVector{V},
        pinhead::AbstractVector{V},
        level::V,
    ) where {V}
    @inbounds if nelm[] > level
        pstart = elmptr[level + one(V)]
        pstop = elmptr[nelm[] + one(V)] - one(V)

        # the newest entry of each vertex is on top
        for p in pstop:-one(V):pstart
            pinhead[pinvtx[p]] = pinnext[p]
        end

        nelm[] = level
    end

    return
end

# Push the separator `members` of a node at `level` (= `nelm`).
function pushelement!(
        nelm::AbstractScalar{V},
        elmptr::AbstractVector{V},
        pinvtx::Vector{V},
        pinlvl::Vector{V},
        pinnext::Vector{V},
        pinhead::AbstractVector{V},
        members::AbstractVector{V},
        level::V,
    ) where {V}
    @assert nelm[] == level
    @assert level + two(V) <= length(elmptr)
    @inbounds p = elmptr[level + one(V)] - one(V)
    q = p + convert(V, length(members))

    if q > length(pinvtx)
        len = max(q, twice(length(pinvtx)))
        resize!(pinvtx, len)
        resize!(pinlvl, len)
        resize!(pinnext, len)
    end

    @inbounds for v in members
        p += one(V)
        pinvtx[p] = v
        pinlvl[p] = level
        pinnext[p] = pinhead[v]
        pinhead[v] = p
    end

    @inbounds elmptr[level + two(V)] = p + one(V)
    nelm[] = level + one(V)
    return
end

# Steps 1 and 2 of `realize!`: the classes of W (weights, vertices, masks, and
# their lists per separator), without their neighborhoods. The stack must hold
# the separators of the ancestors of the subproblem, `level` of them.
function prepare!(
        tag::AbstractScalar{Int},
        stamp::AbstractVector{Int},
        vclass::AbstractVector{V},
        marker::AbstractVector{V},
        mask::AbstractVector{UInt64},
        clsptr::AbstractVector{V},
        clstgt::Vector{V},
        lblptr::AbstractVector{V},
        lbltgt::AbstractVector{V},
        nelm::AbstractScalar{V},
        elmptr::AbstractVector{V},
        pinlvl::AbstractVector{V},
        pinnext::AbstractVector{V},
        pinhead::AbstractVector{V},
        weights::AbstractVector{W},
        vertexset::AbstractVector{V},
        cls::AbstractVector{V},
        nc::V,
        level::V,
        graph::AbstractGraph{V},
    ) where {W, V}
    @assert nelm[] == level
    @assert level + one(V) <= length(clsptr)
    @assert nc <= length(mask)
    n = convert(V, length(vertexset)); t = tag[] += 1

    ###########################
    # 1. the twin classes of W #
    ###########################

    cmpweights = FVector{W}(undef, nc)

    @inbounds for c in oneto(nc)
        cmpweights[c] = zero(W); lblptr[c + one(V)] = zero(V)
    end

    @inbounds for i in oneto(n)
        v = vertexset[i]; c = cls[i]
        stamp[v] = t; vclass[v] = c
        lblptr[c + one(V)] += one(V); cmpweights[c] += weights[v]
    end

    @inbounds lblptr[begin] = one(V)

    @inbounds for c in oneto(nc)
        lblptr[c + one(V)] += lblptr[c]
        marker[c] = lblptr[c]
    end

    @inbounds for i in oneto(n)
        c = cls[i]; lbltgt[marker[c]] = vertexset[i]; marker[c] += one(V)
    end

    label = BipartiteGraph(convert(V, nv(graph)), nc, n, lblptr, lbltgt)

    #######################################################
    # 2. the separators on the stack, as lists of classes #
    #######################################################

    @inbounds npin = elmptr[level + one(V)] - one(V)

    if npin > length(clstgt)
        resize!(clstgt, npin)
    end

    @inbounds for l in oneto(level + one(V))
        clsptr[l] = zero(V)
    end

    @inbounds for c in oneto(nc)
        p = pinhead[lbltgt[lblptr[c]]]; μ = zero(UInt64)

        while ispositive(p)
            l = pinlvl[p]; μ |= one(UInt64) << l
            clsptr[l + two(V)] += one(V)
            p = pinnext[p]
        end

        mask[c] = μ
    end

    @inbounds clsptr[begin] = one(V)

    @inbounds for l in oneto(level)
        clsptr[l + one(V)] += clsptr[l]
        marker[l] = clsptr[l]
    end

    @inbounds for c in oneto(nc)
        p = pinhead[lbltgt[lblptr[c]]]

        while ispositive(p)
            l = pinlvl[p] + one(V)
            clstgt[marker[l]] = c; marker[l] += one(V)
            p = pinnext[p]
        end
    end

    return cmpweights, label
end

# Steps 3 and 4 of `realize!`, after `prepare!`.
function connect!(
        tag::AbstractScalar{Int},
        stamp::AbstractVector{Int},
        vclass::AbstractVector{V},
        marker::AbstractVector{V},
        clsptr::AbstractVector{V},
        clstgt::AbstractVector{V},
        lblptr::AbstractVector{V},
        lbltgt::AbstractVector{V},
        pointer::AbstractVector{V},
        target::Vector{V},
        elmptr::AbstractVector{V},
        pinvtx::AbstractVector{V},
        pinlvl::AbstractVector{V},
        pinnext::AbstractVector{V},
        pinhead::AbstractVector{V},
        nc::V,
        level::V,
        graph::AbstractGraph{V},
    ) where {V}
    @assert nc < length(pointer)
    t = tag[]

    ##################################
    # 3. the neighborhoods of classes #
    ##################################

    @inbounds for c in oneto(nc)
        marker[c] = zero(V)
    end

    @inbounds pointer[begin] = p = one(V)

    @inbounds for c in oneto(nc)
        v = lbltgt[lblptr[c]]; marker[c] = c

        for w in neighbors(graph, v)
            if stamp[w] == t
                u = vclass[w]

                if marker[u] != c
                    marker[u] = c

                    if p > length(target)
                        resize!(target, twice(length(target)))
                    end

                    target[p] = u; p += one(V)
                end
            end
        end

        q = pinhead[v]

        while ispositive(q)
            l = pinlvl[q] + one(V)
            rstart = clsptr[l]; rstop = clsptr[l + one(V)] - one(V)

            if p + (rstop - rstart) >= length(target)
                resize!(target, twice(length(target)) + rstop - rstart + one(V))
            end

            for r in rstart:rstop
                u = clstgt[r]

                if marker[u] != c
                    marker[u] = c; target[p] = u; p += one(V)
                end
            end

            q = pinnext[q]
        end

        pointer[c + one(V)] = p
    end

    cmpgraph = BipartiteGraph(nc, nc, p - one(V), pointer, target)

    #####################################################
    # 4. the classes meeting the parent's separator, S #
    #####################################################

    k = zero(V)

    if ispositive(level)
        @inbounds for c in oneto(nc)
            marker[c] = zero(V)
        end

        @inbounds for q in elmptr[level]:(elmptr[level + one(V)] - one(V))
            u = vclass[pinvtx[q]]

            if iszero(marker[u])
                marker[u] = one(V); k += one(V)
            end
        end
    end

    clique = FVector{V}(undef, k); k = zero(V)

    if ispositive(level)
        @inbounds for c in oneto(nc)
            if isone(marker[c])
                k += one(V); clique[k] = c
            end
        end
    end

    return cmpgraph, clique
end

# Realize the compressed graph of the subproblem (W, cls) at level `level`:
#
#   - `graph`:   one vertex per twin class; the neighborhood of a class is the
#                closed neighborhood of its first vertex in G'[W]
#   - `weights`: the total weight of each class
#   - `label`:   the vertices of each class, in increasing order
#   - `clique`:  the classes meeting the parent's separator, in increasing order
#
# The classes whose first vertex lies in a separator e suffice to realize
# K(e ∩ W): if a class meets e only through a later vertex, then its first
# vertex is adjacent to every vertex of e ∩ W through some other edge of
# G'[W], since the vertices of a class have the same closed neighborhood.
function realize!(
        tag::AbstractScalar{Int},
        stamp::AbstractVector{Int},
        vclass::AbstractVector{V},
        marker::AbstractVector{V},
        mask::AbstractVector{UInt64},
        clsptr::AbstractVector{V},
        clstgt::Vector{V},
        lblptr::AbstractVector{V},
        lbltgt::AbstractVector{V},
        pointer::AbstractVector{V},
        target::Vector{V},
        nelm::AbstractScalar{V},
        elmptr::AbstractVector{V},
        pinvtx::AbstractVector{V},
        pinlvl::AbstractVector{V},
        pinnext::AbstractVector{V},
        pinhead::AbstractVector{V},
        weights::AbstractVector,
        vertexset::AbstractVector{V},
        cls::AbstractVector{V},
        nc::V,
        level::V,
        graph::AbstractGraph{V},
    ) where {V}
    cmpweights, label = prepare!(tag, stamp, vclass, marker, mask, clsptr, clstgt,
        lblptr, lbltgt, nelm, elmptr, pinlvl, pinnext, pinhead, weights,
        vertexset, cls, nc, level, graph)

    cmpgraph, clique = connect!(tag, stamp, vclass, marker, clsptr, clstgt,
        lblptr, lbltgt, pointer, target, elmptr, pinvtx, pinlvl, pinnext,
        pinhead, nc, level, graph)

    return cmpgraph, cmpweights, label, clique
end

###############################################
# Splitting a node without realizing its graph #
###############################################
#
# After `prepare!`, a class t is listed for the level-l separator if and only
# if bit l of `mask[t]` is set, so the union of the separators in `mask[c]` is
#
#     U(c) = { t : mask[t] ∩ mask[c] ≠ ∅ },
#
# which contains c whenever mask[c] ≠ ∅. Let G(c) be the classes of the
# neighbors in G of the first vertex of c, other than c. The neighborhood of
# c in the realized graph is
#
#     N(c) = (U(c) - {c}) ∪ G(c),
#
# and t ∈ G(c) lies in U(c) if and only if mask[t] ∩ mask[c] ≠ ∅. So the size
# and checksum of N(c) ∩ X, for a part X, are read from a table of |U ∩ X| and
# its checksum for each distinct mask, plus one scan of G(c); membership in
# N(c) is a mark or a mask test. No edge of a clique K(e) is ever enumerated.
#
# The masks need one bit per level: this applies when maxlevel < 64.
#
#   - G(c):
#     - `gtag`: a running marker tag, one per scan
#     - `gmark`: `gmark[t] = gtag` if and only if t ∈ G(c)
#     - `gbuf`: G(c), without repetition
#   - mask tables (per subproblem):
#     - `column`: the column of `mask[c]`
#     - `ucount`: `ucount[p + 1, k]` = |U ∩ X_p| for the k-th distinct mask
#     - `uchecksum`: the sum of `twinhash(t)` over the same set

# The distinct masks of the classes, and the tables `ucount` and `uchecksum`.
# If `part` is `nothing`, all classes count as part 0.
function masktable(mask::AbstractVector{UInt64}, part::Union{Nothing, AbstractVector{V}}, nc::V) where {V}
    order = FVector{V}(undef, nc)
    column = FVector{V}(undef, nc)
    value = FVector{UInt64}(undef, nc)

    @inbounds for c in oneto(nc)
        order[c] = c
    end

    sort!(order; by = c -> (@inbounds mask[c]))
    nm = zero(V)

    @inbounds for i in oneto(nc)
        c = order[i]; μ = mask[c]

        if iszero(nm) || value[nm] != μ
            nm += one(V); value[nm] = μ
        end

        column[c] = nm
    end

    owncount = FMatrix{Int}(undef, 3, nm)
    ownchecksum = FMatrix{UInt64}(undef, 3, nm)
    ucount = FMatrix{Int}(undef, 3, nm)
    uchecksum = FMatrix{UInt64}(undef, 3, nm)

    @inbounds for k in oneto(nm), p in 1:3
        owncount[p, k] = 0; ownchecksum[p, k] = zero(UInt64)
        ucount[p, k] = 0; uchecksum[p, k] = zero(UInt64)
    end

    @inbounds for c in oneto(nc)
        k = column[c]; p = isnothing(part) ? 1 : part[c] + 1
        owncount[p, k] += 1; ownchecksum[p, k] += twinhash(c)
    end

    @inbounds for k in oneto(nm), j in oneto(nm)
        if !iszero(value[j] & value[k])
            for p in 1:3
                ucount[p, k] += owncount[p, j]
                uchecksum[p, k] += ownchecksum[p, j]
            end
        end
    end

    return column, ucount, uchecksum
end

# G(c), without repetition, in `gbuf[1:k]`; returns k.
function gclasses!(
        gtag::AbstractScalar{Int},
        gmark::AbstractVector{Int},
        gbuf::AbstractVector{V},
        tag::AbstractScalar{Int},
        stamp::AbstractVector{Int},
        vclass::AbstractVector{V},
        lblptr::AbstractVector{V},
        lbltgt::AbstractVector{V},
        graph::AbstractGraph{V},
        c::V,
    ) where {V}
    t = tag[]; g = gtag[] += 1; k = zero(V)
    @inbounds v = lbltgt[lblptr[c]]; @inbounds gmark[c] = g

    @inbounds for w in neighbors(graph, v)
        if stamp[w] == t
            u = vclass[w]

            if gmark[u] != g
                gmark[u] = g; k += one(V); gbuf[k] = u
            end
        end
    end

    return k
end

# The number of arcs of the realized graph, after `prepare!`.
function implicitarcs(
        gtag::AbstractScalar{Int},
        gmark::AbstractVector{Int},
        gbuf::AbstractVector{V},
        tag::AbstractScalar{Int},
        stamp::AbstractVector{Int},
        vclass::AbstractVector{V},
        mask::AbstractVector{UInt64},
        lblptr::AbstractVector{V},
        lbltgt::AbstractVector{V},
        graph::AbstractGraph{V},
        nc::V,
    ) where {V}
    column, ucount, _ = masktable(mask, nothing, nc); m = 0

    @inbounds for c in oneto(nc)
        μ = mask[c]; m += ucount[1, column[c]] - !iszero(μ)
        k = gclasses!(gtag, gmark, gbuf, tag, stamp, vclass, lblptr, lbltgt, graph, c)

        for i in oneto(k)
            m += iszero(mask[gbuf[i]] & μ)
        end
    end

    return convert(V, m)
end

# `twinfreekeys!`, after `prepare!`.
function implicitkeys!(
        count::AbstractVector{V},
        degree0::AbstractVector{V},
        degree1::AbstractVector{V},
        gtag::AbstractScalar{Int},
        gmark::AbstractVector{Int},
        gbuf::AbstractVector{V},
        tag::AbstractScalar{Int},
        stamp::AbstractVector{Int},
        vclass::AbstractVector{V},
        mask::AbstractVector{UInt64},
        lblptr::AbstractVector{V},
        lbltgt::AbstractVector{V},
        column::AbstractVector{V},
        ucount::AbstractMatrix{Int},
        uchecksum::AbstractMatrix{UInt64},
        part::AbstractVector{V},
        graph::AbstractGraph{V},
        nc::V,
    ) where {V}
    n2 = zero(V)

    @inbounds for c in oneto(nc)
        if istwo(part[c])
            n2 += one(V)
        end
    end

    label2 = FVector{V}(undef, n2); t2 = zero(V)

    @inbounds for c in oneto(nc)
        if istwo(part[c])
            t2 += one(V); label2[t2] = c
        end
    end

    # count[c] = | N(c) ∩ S |
    @inbounds for c in oneto(nc)
        μ = mask[c]; x = ucount[3, column[c]] - (istwo(part[c]) && !iszero(μ))
        k = gclasses!(gtag, gmark, gbuf, tag, stamp, vclass, lblptr, lbltgt, graph, c)

        for i in oneto(k)
            u = gbuf[i]
            x += istwo(part[u]) && iszero(mask[u] & μ)
        end

        count[c] = x
    end

    checksum0 = FVector{UInt64}(undef, n2)
    checksum1 = FVector{UInt64}(undef, n2)

    @inbounds for i in oneto(n2)
        x = label2[i]; μ = mask[x]; j = column[x]
        d0 = ucount[1, j]; h0 = uchecksum[1, j]
        d1 = ucount[2, j]; h1 = uchecksum[2, j]
        k = gclasses!(gtag, gmark, gbuf, tag, stamp, vclass, lblptr, lbltgt, graph, x)

        for r in oneto(k)
            u = gbuf[r]

            if iszero(mask[u] & μ)
                pu = part[u]

                if iszero(pu)    # u ∈ A
                    d0 += 1; h0 += twinhash(u)
                elseif isone(pu) # u ∈ B
                    d1 += 1; h1 += twinhash(u)
                end
            end
        end

        degree0[i] = d0; checksum0[i] = h0
        degree1[i] = d1; checksum1[i] = h1
    end

    return label2, checksum0, checksum1
end

# `twinfreeclasses!`, after `prepare!`.
function implicitclasses!(
        mark::AbstractVector{V},
        class::AbstractVector{V},
        size::AbstractVector{V},
        xrep::AbstractVector{V},
        project::AbstractVector{V},
        sdegree::AbstractVector{V},
        schecksum::AbstractVector{UInt64},
        count::AbstractVector{V},
        label2::AbstractVector{V},
        side::V,
        part::AbstractVector{V},
        gtag::AbstractScalar{Int},
        gmark::AbstractVector{Int},
        gbuf::AbstractVector{V},
        tag::AbstractScalar{Int},
        stamp::AbstractVector{Int},
        vclass::AbstractVector{V},
        mask::AbstractVector{UInt64},
        clsptr::AbstractVector{V},
        clstgt::AbstractVector{V},
        lblptr::AbstractVector{V},
        lbltgt::AbstractVector{V},
        column::AbstractVector{V},
        ucount::AbstractMatrix{Int},
        uchecksum::AbstractMatrix{UInt64},
        graph::AbstractGraph{V},
        nc::V,
    ) where {V}
    n = nc; n2 = convert(V, length(label2)); p = side + one(V)

    # v ∈ S ∪ A*
    @inline inx(v::V) = twinfreeinx(v, side, n2, part, count)

    ##############################
    # 1. X = S ∪ A* and its keys #
    ##############################

    nx = zero(V)

    if ispositive(n2)
        @inbounds for x in label2
            nx += one(V); xrep[nx] = x
        end

        # A* = { v ∈ A : S ⊆ N(v) }
        @inbounds for v in oneto(n)
            if part[v] == side && count[v] == n2
                nx += one(V); xrep[nx] = v
            end
        end
    end

    xdegree = FVector{V}(undef, nx)
    xchecksum = FVector{UInt64}(undef, nx)

    @inbounds for i in oneto(n2)
        xdegree[i] = sdegree[i]
        xchecksum[i] = schecksum[i]
    end

    @inbounds for i in (n2 + one(V)):nx # y ∈ A*: the key is N[y] ∩ A
        y = xrep[i]; μ = mask[y]; j = column[y]
        d = ucount[p, j] + iszero(μ)
        h = uchecksum[p, j] + (iszero(μ) ? twinhash(y) : zero(UInt64))
        k = gclasses!(gtag, gmark, gbuf, tag, stamp, vclass, lblptr, lbltgt, graph, y)

        for r in oneto(k)
            u = gbuf[r]

            if part[u] == side && iszero(mask[u] & μ)
                d += 1; h += twinhash(u)
            end
        end

        xdegree[i] = d; xchecksum[i] = h
    end

    #################################################
    # 2. group X by key size and checksum, and      #
    #    confirm each group exactly                 #
    #################################################

    xorder = FVector{V}(undef, nx)

    @inbounds for i in oneto(nx)
        xorder[i] = i
    end

    sort!(xorder; by = i -> (@inbounds (xdegree[i], xchecksum[i])))

    @inbounds for v in oneto(n)
        mark[v] = zero(V)
    end

    nc = zero(V); stamp2 = zero(V); lo = one(V)

    @inbounds while lo <= nx
        # xorder[lo:hi] have the same key size and checksum
        i = xorder[lo]; hi = lo

        while hi < nx && xdegree[xorder[hi + one(V)]] == xdegree[i] && xchecksum[xorder[hi + one(V)]] == xchecksum[i]
            hi += one(V)
        end

        next = hi + one(V)

        while lo <= hi
            # the key of the leader y: the marked classes, and
            # the classes of U(y) on this side (a mask test)
            y = xrep[xorder[lo]]; μy = mask[y]; stamp2 += one(V)
            k = gclasses!(gtag, gmark, gbuf, tag, stamp, vclass, lblptr, lbltgt, graph, y)

            for r in oneto(k)
                u = gbuf[r]

                if part[u] == side
                    mark[u] = stamp2
                end
            end

            if part[y] == side
                mark[y] = stamp2
            end

            nc += one(V); class[y] = nc; size[nc] = one(V)

            # compare the other keys against it; keys of equal size
            # are equal iff one is contained in the other
            keep = lo

            for kk in (lo + one(V)):hi
                z = xrep[xorder[kk]]; μz = mask[z]
                same = part[z] != side || mark[z] == stamp2 || !iszero(mask[z] & μy)

                if same
                    k = gclasses!(gtag, gmark, gbuf, tag, stamp, vclass, lblptr, lbltgt, graph, z)

                    for r in oneto(k)
                        u = gbuf[r]

                        if part[u] == side && mark[u] != stamp2 && iszero(mask[u] & μy)
                            same = false; break
                        end
                    end
                end

                # the separators in mask[z] but not in mask[y]
                extra = μz & ~μy

                while same && !iszero(extra)
                    l = convert(V, trailing_zeros(extra)) + one(V)
                    extra &= extra - one(UInt64)

                    for r in clsptr[l]:(clsptr[l + one(V)] - one(V))
                        u = clstgt[r]

                        if u != z && part[u] == side && mark[u] != stamp2 && iszero(mask[u] & μy)
                            same = false; break
                        end
                    end
                end

                if same
                    class[z] = nc; size[nc] += one(V)
                else
                    keep += one(V); xorder[keep] = xorder[kk]
                end
            end

            lo += one(V); hi = keep
        end

        lo = next
    end

    #####################################
    # 3. number the classes of the child #
    #####################################

    newid = mark; nsub = zero(V); ncmp = zero(V)

    @inbounds for c in oneto(nc)
        newid[c] = zero(V)
    end

    @inbounds for v in oneto(n)
        pv = part[v]

        if pv == side || istwo(pv)
            nsub += one(V)

            if inx(v)
                c = class[v]

                if iszero(newid[c])
                    ncmp += one(V); newid[c] = ncmp; xrep[ncmp] = v
                end

                project[v] = newid[c]
            else
                ncmp += one(V); project[v] = ncmp; xrep[ncmp] = v
            end
        end
    end

    return ncmp, nsub
end

###########################################
# Nested dissection without a node stack #
###########################################
#
# The children of a node P, with vertex sets A ∪ S and B ∪ S, return orderings
# that overlap in S. Dropping S from each, which is what P does when it
# stitches them together, leaves the residuals: orderings of A and of B. Along
# the path from the root to the current node, every ancestor holds at most one
# such piece (the vertex set of a child not yet started, or the residual of a
# child that has finished), and these pieces are disjoint from each other and
# from the vertex set of the current node. So they fit in one array of length
# n, used as a stack of segments.
#
# A node at level L owns the segment that starts at base[L]. When it is split,
# it writes A and then B there, and its child 1 (B ∪ S) starts at base + |A|.
# When child 1 returns, its residual (an ordering of B) occupies B's slots;
# A is read out, the residual of child 1 moves down to base, and child 0
# (A ∪ S) starts at base + |B|. When child 0 returns, its residual follows,
# and P writes its own residual (its ordering, minus its parent's separator)
# at base.
#
# The twin classes of the nodes on the path are kept in a union-find with
# rollback over the vertices: the classes of a child are unions of the
# classes of its parent, so the classes of a node are the singletons of the
# twin-free graph, merged by the merges of its ancestors. When a node is
# split, the merges of child 1 are applied, and those of child 0 are kept
# until child 1 returns.
#
#   - working orderings:
#     - `segment`: the stack of segments, length n
#     - `base`, `na`, `nb`, `side`: per level, the start of the node's
#       segment, the sizes of A and B, and the child being processed
#   - classes:
#     - `ufp`: union-find parent of each vertex; a root is its own parent
#     - `mrg`: pairs (r, t) of roots: the merges of the children not applied
#     - `mlog`: the roots relinked so far, in order (to roll back)
#     - `rmark`, `rid`: marker array and class number of each root

# The root of the class of v.
@inline function ufind(ufp::AbstractVector{V}, v::V) where {V}
    @inbounds while ufp[v] != v
        v = ufp[v]
    end

    return v
end

# The classes of the sorted vertex set W = vertexset[1:nw]: cls[i] is the
# class of vertexset[i], numbered by their first vertex. Returns the number
# of classes.
function classify!(
        cls::AbstractVector{V},
        rmark::AbstractVector{Int},
        rid::AbstractVector{V},
        ufp::AbstractVector{V},
        vertexset::AbstractVector{V},
        nw::V,
        t::Int,
    ) where {V}
    nc = zero(V)

    @inbounds for i in oneto(nw)
        r = ufind(ufp, vertexset[i])

        if rmark[r] != t
            rmark[r] = t; nc += one(V); rid[r] = nc
        end

        cls[i] = rid[r]
    end

    return nc
end

# Apply the merges mrg[mstart:mstop] (pairs), logging the relinked roots.
# Returns the new length of the log.
function applymerges!(
        ufp::AbstractVector{V},
        mlog::Vector{V},
        nlog::V,
        mrg::AbstractVector{V},
        mstart::V,
        mstop::V,
    ) where {V}
    nnew = nlog + half(mstop - mstart + one(V))

    if nnew > length(mlog)
        resize!(mlog, max(nnew, twice(length(mlog))))
    end

    @inbounds for p in mstart:two(V):mstop
        r = mrg[p]; ufp[r] = mrg[p + one(V)]
        nlog += one(V); mlog[nlog] = r
    end

    return nlog
end

# Undo the merges logged after position `nstart - 1`. Returns the new length
# of the log.
function rollback!(ufp::AbstractVector{V}, mlog::AbstractVector{V}, nlog::V, nstart::V) where {V}
    @inbounds for p in nlog:-one(V):nstart
        r = mlog[p]; ufp[r] = r
    end

    return nstart - one(V)
end

# Merge the sorted lists a[astart:astop] and b[bstart:bstop] into out.
# Returns the length of the result.
function mergesorted!(
        out::AbstractVector{V},
        a::AbstractVector{V},
        astart::V,
        astop::V,
        b::AbstractVector{V},
        bstart::V,
        bstop::V,
    ) where {V}
    i = astart; j = bstart; k = zero(V)

    @inbounds while i <= astop && j <= bstop
        if a[i] < b[j]
            k += one(V); out[k] = a[i]; i += one(V)
        else
            k += one(V); out[k] = b[j]; j += one(V)
        end
    end

    @inbounds while i <= astop
        k += one(V); out[k] = a[i]; i += one(V)
    end

    @inbounds while j <= bstop
        k += one(V); out[k] = b[j]; j += one(V)
    end

    return k
end

# Split the subproblem (W, cls) along a vertex separator of its realized graph,
# with part[c] ∈ {0, 1, 2} for each class c: see `segmentchildren!`.
function segmentsplit!(
        work0::AbstractVector{V},
        work1::AbstractVector{V},
        work2::AbstractVector{V},
        work3::AbstractVector{V},
        work4::AbstractVector{V},
        work5::AbstractVector{V},
        work6::AbstractVector{V},
        work7::AbstractVector{V},
        work8::AbstractVector{V},
        marker::AbstractVector{V},
        segment::AbstractVector{V},
        base::V,
        mrg::Vector{V},
        nmrg::V,
        ufp::AbstractVector{V},
        lblptr::AbstractVector{V},
        lbltgt::AbstractVector{V},
        nelm::AbstractScalar{V},
        elmptr::AbstractVector{V},
        pinvtx::Vector{V},
        pinlvl::Vector{V},
        pinnext::Vector{V},
        pinhead::AbstractVector{V},
        vertexset::AbstractVector{V},
        cls::AbstractVector{V},
        part::AbstractVector{V},
        nc::V,
        level::V,
        graph::AbstractGraph{V},
    ) where {V}
    count = work0; degree0 = work1; degree1 = work2
    mark = work3; class = work4; size = work5; xrep = work6
    project0 = work7; project1 = work8

    label2, checksum0, checksum1 = twinfreekeys!(count, degree0, degree1, part, graph)

    nc0, _ = twinfreeclasses!(mark, class, size, xrep, project0,
        degree0, checksum0, count, label2, zero(V), part, graph)

    nc1, _ = twinfreeclasses!(mark, class, size, xrep, project1,
        degree1, checksum1, count, label2, one(V), part, graph)

    return segmentchildren!(marker, segment, base, mrg, nmrg, ufp, lblptr, lbltgt,
        nelm, elmptr, pinvtx, pinlvl, pinnext, pinhead, vertexset, cls, part,
        project0, project1, nc0, nc1, nc, level)
end

# `segmentsplit!`, after `prepare!`: the graph of (W, cls) is never realized.
function implicitsegmentsplit!(
        work0::AbstractVector{V},
        work1::AbstractVector{V},
        work2::AbstractVector{V},
        work3::AbstractVector{V},
        work4::AbstractVector{V},
        work5::AbstractVector{V},
        work6::AbstractVector{V},
        work7::AbstractVector{V},
        work8::AbstractVector{V},
        marker::AbstractVector{V},
        segment::AbstractVector{V},
        base::V,
        mrg::Vector{V},
        nmrg::V,
        ufp::AbstractVector{V},
        gtag::AbstractScalar{Int},
        gmark::AbstractVector{Int},
        gbuf::AbstractVector{V},
        tag::AbstractScalar{Int},
        stamp::AbstractVector{Int},
        vclass::AbstractVector{V},
        mask::AbstractVector{UInt64},
        clsptr::AbstractVector{V},
        clstgt::AbstractVector{V},
        lblptr::AbstractVector{V},
        lbltgt::AbstractVector{V},
        nelm::AbstractScalar{V},
        elmptr::AbstractVector{V},
        pinvtx::Vector{V},
        pinlvl::Vector{V},
        pinnext::Vector{V},
        pinhead::AbstractVector{V},
        vertexset::AbstractVector{V},
        cls::AbstractVector{V},
        part::AbstractVector{V},
        nc::V,
        level::V,
        graph::AbstractGraph{V},
    ) where {V}
    count = work0; degree0 = work1; degree1 = work2
    mark = work3; class = work4; size = work5; xrep = work6
    project0 = work7; project1 = work8

    column, ucount, uchecksum = masktable(mask, part, nc)

    label2, checksum0, checksum1 = implicitkeys!(count, degree0, degree1,
        gtag, gmark, gbuf, tag, stamp, vclass, mask, lblptr, lbltgt,
        column, ucount, uchecksum, part, graph, nc)

    nc0, _ = implicitclasses!(mark, class, size, xrep, project0, degree0,
        checksum0, count, label2, zero(V), part, gtag, gmark, gbuf, tag, stamp,
        vclass, mask, clsptr, clstgt, lblptr, lbltgt, column, ucount, uchecksum,
        graph, nc)

    nc1, _ = implicitclasses!(mark, class, size, xrep, project1, degree1,
        checksum1, count, label2, one(V), part, gtag, gmark, gbuf, tag, stamp,
        vclass, mask, clsptr, clstgt, lblptr, lbltgt, column, ucount, uchecksum,
        graph, nc)

    return segmentchildren!(marker, segment, base, mrg, nmrg, ufp, lblptr, lbltgt,
        nelm, elmptr, pinvtx, pinlvl, pinnext, pinhead, vertexset, cls, part,
        project0, project1, nc0, nc1, nc, level)
end

# Given the class maps `project0` and `project1` of the children (from
# `twinfreeclasses!` or `implicitclasses!`):
#
#   - A and B are written, sorted, to segment[base:base + na + nb - 1]
#   - S is pushed onto the separator stack
#   - the merges that turn the classes of W into those of child 1 (B ∪ S) and
#     of child 0 (A ∪ S) are appended to `mrg`, in that order
#
# Returns na, nb, and the ends of the two lists of merges.
function segmentchildren!(
        marker::AbstractVector{V},
        segment::AbstractVector{V},
        base::V,
        mrg::Vector{V},
        nmrg::V,
        ufp::AbstractVector{V},
        lblptr::AbstractVector{V},
        lbltgt::AbstractVector{V},
        nelm::AbstractScalar{V},
        elmptr::AbstractVector{V},
        pinvtx::Vector{V},
        pinlvl::Vector{V},
        pinnext::Vector{V},
        pinhead::AbstractVector{V},
        vertexset::AbstractVector{V},
        cls::AbstractVector{V},
        part::AbstractVector{V},
        project0::AbstractVector{V},
        project1::AbstractVector{V},
        nc0::V,
        nc1::V,
        nc::V,
        level::V,
    ) where {V}
    ###################################
    # 1. A and B to the segment, and S #
    ###################################

    na = zero(V); nb = zero(V); ns = zero(V)

    @inbounds for i in eachindex(vertexset)
        pc = part[cls[i]]

        if iszero(pc)
            na += one(V)
        elseif isone(pc)
            nb += one(V)
        else
            ns += one(V)
        end
    end

    members = FVector{V}(undef, ns)
    ia = base - one(V); ib = base + na - one(V); is = zero(V)

    @inbounds for i in eachindex(vertexset)
        v = vertexset[i]; pc = part[cls[i]]

        if iszero(pc)
            ia += one(V); segment[ia] = v
        elseif isone(pc)
            ib += one(V); segment[ib] = v
        else
            is += one(V); members[is] = v
        end
    end

    pushelement!(nelm, elmptr, pinvtx, pinlvl, pinnext, pinhead, members, level)

    #####################################
    # 2. the merges of child 1 and child 0 #
    #####################################

    # at most one merge per class of W, for each child
    if nmrg + four(V) * nc > length(mrg)
        resize!(mrg, max(nmrg + four(V) * nc, twice(length(mrg))))
    end

    stop1 = zero(V); stop0 = zero(V)

    for (side, project, ncs) in ((one(V), project1, nc1), (zero(V), project0, nc0))
        # marker[u] = the first class of W in class u of the child
        @inbounds for u in oneto(ncs)
            marker[u] = zero(V)
        end

        @inbounds for c in oneto(nc)
            pc = part[c]

            if pc == side || istwo(pc)
                u = project[c]

                if iszero(marker[u])
                    marker[u] = c
                else
                    r = ufind(ufp, lbltgt[lblptr[c]])
                    t = ufind(ufp, lbltgt[lblptr[marker[u]]])
                    nmrg += one(V); mrg[nmrg] = r
                    nmrg += one(V); mrg[nmrg] = t
                end
            end
        end

        if isone(side)
            stop1 = nmrg
        else
            stop0 = nmrg
        end
    end

    return na, nb, stop1, stop0
end

# Write the class ordering order[1:nc], expanded to vertices, to `segment`
# from `base`, leaving out the vertices v with rmark[v] = t. Returns the
# number of vertices written.
function writeresidual!(
        segment::AbstractVector{V},
        base::V,
        order::AbstractVector{V},
        nc::V,
        label::AbstractGraph{V},
        rmark::AbstractVector{Int},
        t::Int,
    ) where {V}
    p = base - one(V)

    @inbounds for i in oneto(nc)
        for v in neighbors(label, order[i])
            if rmark[v] != t
                p += one(V); segment[p] = v
            end
        end
    end

    return p - base + one(V)
end

################################################
# A global clique cover for KaHyPar (elements) #
################################################
#
# KaHyPar partitions the cliques of a cover of G'[W]: its hypernodes are the
# cliques, its nets are the classes of W, and the net of a class is the set of
# cliques containing one of its vertices. A vertex lies in the separator if
# its cliques lie on both sides.
#
# The cliques are elements: the cliques of a cover of the twin-free graph G
# (computed once), and the separators on the stack. When a node at level L is
# split, each of its elements goes to one side, and the separator of the node
# goes to both. An element is active at a node if it went to the node's side
# at every split along the path. Each element records the last split:
#
#   - `clev`, `cside`, `cepoch`: for the cliques of the cover
#   - `slev`, `sside`, `sepoch`: for the separators, by position on the stack
#
# meaning "active in the child at level `lev` on side `side` (2: both) of the
# node with epoch `epoch`". The nodes are numbered (their epochs) as they are
# reached, so an element is active at a node at level L, with parent epoch E
# and side s, if and only if
#
#     lev = L, epoch = E, and side ∈ {s, 2}.
#
# Records written below a node concern elements active there. Apart from the
# separator of its parent, which goes to both children, these are not active
# at its sibling, so the only record to restore when the second child starts
# is that of the parent's separator.
#
# The hypernodes of a node are numbered as in a recursive construction: the
# separators on the stack, newest first, then the cliques of the cover, in
# increasing order. The pins of each net are in increasing order.

@inline function isactive(lev::V, side::V, epoch::Int, level::V, pepoch::Int, pside::V) where {V}
    return lev == level && epoch == pepoch && (istwo(side) || side == pside)
end

# The hypergraph of the current node, after `prepare!`: the pins of net c are
# htgt[hptr[c]:hptr[c + 1] - 1], and hypernode i is the element helm[i]
# (positive: a clique of the cover; negative: minus a position on the stack).
# Returns the number of hypernodes and of pins.
function hrealize!(
        hptr::AbstractVector{E},
        htgt::Vector{HV},
        helm::AbstractVector{V},
        hid::AbstractVector{HV},
        sid::AbstractVector{HV},
        hmark::AbstractVector{Int},
        htag::AbstractScalar{Int},
        cmark::AbstractVector{Int},
        ctag::AbstractScalar{Int},
        clev::AbstractVector{V},
        cside::AbstractVector{V},
        cepoch::AbstractVector{Int},
        slev::AbstractVector{V},
        sside::AbstractVector{V},
        sepoch::AbstractVector{Int},
        lblptr::AbstractVector{V},
        lbltgt::AbstractVector{V},
        nelm::AbstractScalar{V},
        pinlvl::AbstractVector{V},
        pinnext::AbstractVector{V},
        pinhead::AbstractVector{V},
        cover::AbstractGraph,
        nc::V,
        level::V,
        pepoch::Int,
        pside::V,
    ) where {E, HV, V}
    ######################################
    # 1. the separators, newest first    #
    ######################################

    hn = zero(V)

    @inbounds for k in nelm[]:-one(V):one(V)
        if isactive(slev[k], sside[k], sepoch[k], level, pepoch, pside)
            hn += one(V); sid[k] = convert(HV, hn); helm[hn] = -k
        else
            sid[k] = zero(HV)
        end
    end

    ns = hn

    ##########################################
    # 2. the cliques of the cover, in order  #
    ##########################################

    t = ctag[] += 1

    @inbounds for c in oneto(nc), q in lblptr[c]:(lblptr[c + one(V)] - one(V))
        for e in neighbors(cover, lbltgt[q])
            if cmark[e] != t
                cmark[e] = t

                if isactive(clev[e], cside[e], cepoch[e], level, pepoch, pside)
                    hn += one(V); helm[hn] = convert(V, e)
                else
                    hid[e] = zero(HV)
                end
            end
        end
    end

    sort!(view(helm, (ns + one(V)):hn))

    @inbounds for i in (ns + one(V)):hn
        hid[helm[i]] = convert(HV, i)
    end

    #######################################
    # 3. the nets: the pins of each class #
    #######################################

    @inbounds hptr[begin] = p = one(E)

    @inbounds for c in oneto(nc)
        t = htag[] += 1; pstart = p

        for q in lblptr[c]:(lblptr[c + one(V)] - one(V))
            v = lbltgt[q]

            for e in neighbors(cover, v)
                i = hid[e]

                if ispositive(i) && hmark[i] != t
                    hmark[i] = t

                    if p > length(htgt)
                        resize!(htgt, twice(length(htgt)))
                    end

                    htgt[p] = i; p += one(E)
                end
            end

            r = pinhead[v]

            while ispositive(r)
                i = sid[pinlvl[r] + one(V)]

                if ispositive(i) && hmark[i] != t
                    hmark[i] = t

                    if p > length(htgt)
                        resize!(htgt, twice(length(htgt)))
                    end

                    htgt[p] = i; p += one(E)
                end

                r = pinnext[r]
            end
        end

        sort!(view(htgt, pstart:(p - one(E))))
        hptr[c + one(V)] = p
    end

    return hn, p - one(E)
end

# After KaHyPar has put hypernode i on side hpart[i], record the split of the
# node at `level` with epoch `epoch`.
function hrecord!(
        clev::AbstractVector{V},
        cside::AbstractVector{V},
        cepoch::AbstractVector{Int},
        slev::AbstractVector{V},
        sside::AbstractVector{V},
        sepoch::AbstractVector{Int},
        helm::AbstractVector{V},
        hpart::AbstractVector,
        hn::V,
        level::V,
        epoch::Int,
    ) where {V}
    @inbounds for i in oneto(hn)
        e = helm[i]; s = convert(V, hpart[i])

        if ispositive(e)
            clev[e] = level + one(V); cside[e] = s; cepoch[e] = epoch
        else
            slev[-e] = level + one(V); sside[-e] = s; sepoch[-e] = epoch
        end
    end

    return
end
