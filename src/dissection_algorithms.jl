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

function hpartition!(
        work00::AbstractScalar{V},
        work01::AbstractVector{V},
        work02::AbstractVector{V},
        work03::AbstractVector{V},
        work04::AbstractVector{V},
        work05::AbstractVector{V},
        work06::AbstractVector{V},
        work07::AbstractVector{V},
        work08::AbstractVector{E},
        work09::AbstractVector{E},
        work10::AbstractVector{V},
        work11::AbstractVector{V},
        work12::AbstractVector{V},
        work13::AbstractVector{V},
        hproject0::AbstractVector{V},
        hproject1::AbstractVector{V},
        hpart::AbstractVector{V},
        part::AbstractVector{V},
        weights::AbstractVector{W},
        hgraph::AbstractGraph{HV},
        graph::AbstractGraph{V},
    ) where {W, V, E, HV}
    @assert nov(hgraph) <= length(hproject0)
    @assert nov(hgraph) <= length(hproject1)
    @assert nv(hgraph) <= length(part)
    @assert nv(hgraph) == nv(graph)

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
    for v in vertices(graph)
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

    child0, child1, label2 = twinfreepartition!(work00, work01, work02, work03, work04,
        work05, work06, work07, work08, work09, work10, work11, work12, work13,
        part, weights, graph)

    graph0, weights0, label0, clique0 = child0
    graph1, weights1, label1, clique1 = child1

    tag = one(V)

    hgraph0, tag = hcompresspart(h0, tag, hgraph, hproject0, hpart, label0, clique0)
    hgraph1, tag = hcompresspart(h1, tag, hgraph, hproject1, hpart, label1, clique1)

    hchild0 = (hgraph0, graph0, weights0, label0, clique0)
    hchild1 = (hgraph1, graph1, weights1, label1, clique1)

    return hchild0, hchild1, label2
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
function twinfreepartition!(
        work0::AbstractScalar{V},
        work1::AbstractVector{V},
        work2::AbstractVector{V},
        work3::AbstractVector{V},
        work4::AbstractVector{V},
        work5::AbstractVector{V},
        label0::AbstractVector{V},
        label1::AbstractVector{V},
        pointer0::AbstractVector{E},
        pointer1::AbstractVector{E},
        target0::AbstractVector{V},
        target1::AbstractVector{V},
        project0::AbstractVector{V},
        project1::AbstractVector{V},
        part::AbstractVector{V},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
    ) where {W, V, E}
    @assert nv(graph) <= length(work1)
    @assert nv(graph) <= length(work2)
    @assert nv(graph) <= length(work3)
    @assert nv(graph) <= length(work4)
    @assert nv(graph) <= length(work5)
    @assert nv(graph) <= length(label0)
    @assert nv(graph) <= length(label1)
    @assert nv(graph) <= length(part)
    @assert nv(graph) <= length(project0)
    @assert nv(graph) <= length(project1)
    @assert nv(graph) <= length(weights)

    # S = W ∩ B
    n2 = zero(V)

    @inbounds for v in vertices(graph)
        if istwo(part[v])
            n2 += one(V)
        end
    end

    label2 = FVector{V}(undef, n2); t2 = zero(V)

    @inbounds for v in vertices(graph)
        if istwo(part[v])
            t2 += one(V); label2[t2] = v
        end
    end

    # count[v] = | N(v) ∩ S |
    count = project1

    @inbounds for v in vertices(graph)
        count[v] = zero(V)
    end

    # keys of S in both children: sizes and checksums
    degree0 = work2; checksum0 = FVector{UInt64}(undef, n2)
    degree1 = label1; checksum1 = FVector{UInt64}(undef, n2)

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

    child0 = twinfreechild!(work1, work3, work4, work5, label0, project0,
        degree0, checksum0, count, label2, zero(V), part, weights, graph)

    child1 = twinfreechild!(work1, work3, work4, work5, label0, project0,
        degree1, checksum1, count, label2, one(V), part, weights, graph)

    # the caller relies on `target0` and `target1`
    # being long enough to hold the arcs of either child
    m01 = max(ne(first(child0)), ne(first(child1)))

    if m01 > length(target0)
        resize!(target0, m01)
    end

    if m01 > length(target1)
        resize!(target1, m01)
    end

    return child0, child1, label2
end

# splitmix64 finalizer
@inline function twinhash(v::Integer)
    x = convert(UInt64, v) * 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    return x ⊻ (x >> 31)
end

function twinfreechild!(
        mark::AbstractVector{V},
        class::AbstractVector{V},
        size::AbstractVector{V},
        xrep::AbstractVector{V},
        marker::AbstractVector{V},
        project::AbstractVector{V},
        sdegree::AbstractVector{V},
        schecksum::AbstractVector{UInt64},
        count::AbstractVector{V},
        label2::AbstractVector{V},
        side::V,
        part::AbstractVector{V},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
    ) where {W, V}
    n = nv(graph); n2 = convert(V, length(label2))
    E = etype(graph)

    # v ∈ S ∪ A*
    @inline function inx(v::V)
        @inbounds pv = part[v]
        return istwo(pv) || (pv == side && ispositive(n2) && @inbounds count[v] == n2)
    end

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

    nncmp = ncmp + one(V)
    prjptr = FVector{V}(undef, nncmp)
    prjtgt = FVector{V}(undef, nsub)
    cmpweights = FVector{W}(undef, ncmp)
    cursor = marker

    @inbounds prjptr[begin] = q = one(V)

    @inbounds for u in oneto(ncmp)
        r = xrep[u]
        s = inx(r) ? size[class[r]] : one(V)
        cursor[u] = q
        prjptr[u + one(V)] = q += s
        cmpweights[u] = zero(W)
    end

    @inbounds for v in vertices(graph)
        pv = part[v]

        if pv == side || istwo(pv)
            u = project[v]
            prjtgt[cursor[u]] = v; cursor[u] += one(V)
            cmpweights[u] += weights[v]
        end
    end

    cmplabel = BipartiteGraph(convert(V, n), ncmp, nsub, prjptr, prjtgt)

    # classes containing a separator vertex
    flag = marker; kcmp = zero(V)

    @inbounds for u in oneto(ncmp)
        flag[u] = zero(V)
    end

    @inbounds for x in label2
        u = project[x]

        if iszero(flag[u])
            flag[u] = one(V); kcmp += one(V)
        end
    end

    cmpclique = FVector{V}(undef, kcmp); kcmp = zero(V)

    @inbounds for u in oneto(ncmp)
        if isone(flag[u])
            kcmp += one(V); cmpclique[kcmp] = u
        end
    end

    ##################################
    # 4. build the quotient directly #
    ##################################

    mcmp = zero(E)

    @inbounds for u in oneto(ncmp)
        r = xrep[u]

        if istwo(part[r]) # N₀[r] = (N(r) ∩ A) ∪ S
            mcmp += convert(E, outdegree(graph, r) - count[r] + kcmp)
        else              # N₀[r] = N[r]
            mcmp += convert(E, outdegree(graph, r))
        end
    end

    cmpptr = FVector{E}(undef, nncmp)
    cmptgt = FVector{V}(undef, mcmp)

    @inbounds for u in oneto(ncmp)
        marker[u] = zero(V)
    end

    @inbounds cmpptr[begin] = p = one(E)

    @inbounds for u in oneto(ncmp)
        r = xrep[u]; marker[u] = u

        if istwo(part[r])
            for w in neighbors(graph, r)
                if part[w] == side
                    t = project[w]

                    if marker[t] != u
                        marker[t] = u; cmptgt[p] = t; p += one(E)
                    end
                end
            end

            for t in cmpclique
                if marker[t] != u
                    marker[t] = u; cmptgt[p] = t; p += one(E)
                end
            end
        else
            for w in neighbors(graph, r)
                t = project[w]

                if marker[t] != u
                    marker[t] = u; cmptgt[p] = t; p += one(E)
                end
            end
        end

        cmpptr[u + one(V)] = p
    end

    cmpgraph = BipartiteGraph(ncmp, ncmp, p - one(E), cmpptr, cmptgt)
    return (cmpgraph, cmpweights, cmplabel, cmpclique)
end

function hcompresspart(
        hh::V,
        tag::V,
        hgraph::AbstractGraph{HV},
        hproject::AbstractVector{V},
        mark::AbstractVector{V},
        label::AbstractGraph{V},
        clique::AbstractVector{V},
    ) where {HV, V}
    @assert nov(hgraph) <= length(mark)
    @assert nv(hgraph) == nov(label)

    HE = etype(hgraph); hn = convert(HV, nv(label)); hm = convert(HE, length(clique))

    for v in vertices(label)
        tag += one(V)

        for w in neighbors(label, v), hw in neighbors(hgraph, w)
            hx = hproject[hw]

            if ispositive(hx) && mark[hw] < tag
                mark[hw] = tag
                hm += one(HE)
            end
        end
    end

    hchild = BipartiteGraph{HV, HE}(hh, hn, hm)
    pointers(hchild)[begin] = p = one(HE)

    for v in vertices(label)
        tag += one(V); flag = false

        for w in neighbors(label, v), hw in neighbors(hgraph, w)
            hx = hproject[hw]

            if ispositive(hx) && mark[hw] < tag
                mark[hw] = tag
                targets(hchild)[p] = hx; p += one(HE)
            end

            flag = flag || iszero(hx)
        end

        if flag
            targets(hchild)[p] = one(HV); p += one(HE)
        end

        pointers(hchild)[v + one(V)] = p
    end

    return hchild, tag
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
