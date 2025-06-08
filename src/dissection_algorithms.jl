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

function partition!(part::AbstractVector{V}, project0::AbstractVector{V}, project1::AbstractVector{V}, weights::AbstractVector{W}, graph::AbstractGraph{V}) where {W, V}
    E = etype(graph)

    # V = W ∪ B
    n0 = zero(V); m0 = zero(E)
    n1 = zero(V); m1 = zero(E)
    n2 = zero(V)

    @inbounds for v in vertices(graph)
        vv = part[v]

        if iszero(vv)    # v ∈ W - B
            n0 += one(V); m0 += convert(E, outdegree(graph, v))
        elseif isone(vv) # v ∈ B - W
            n1 += one(V); m1 += convert(E, outdegree(graph, v))
        else             # v ∈ W ∩ B
            n0 += one(V); m0 += convert(E, twice(n2))
            n1 += one(V); m1 += convert(E, twice(n2))
            n2 += one(V)

            for w in outneighbors(graph, v)
                ww = part[w]

                if iszero(ww)    # w ∈ W - B
                    m0 += one(E)
                elseif isone(ww) # w ∈ B - W
                    m1 += one(E)
                end
            end
        end
    end

    t0 = zero(V); label0 = Vector{V}(undef, n0)
    t1 = zero(V); label1 = Vector{V}(undef, n1)
    t2 = zero(V); label2 = Vector{V}(undef, n2)
    clique0 = Vector{V}(undef, n2)
    clique1 = Vector{V}(undef, n2)

    @inbounds for v in vertices(graph)
        vv = part[v]

        if iszero(vv)    # v ∈ W - B
            project0[v] = t0 += one(V); label0[t0] = v
        elseif isone(vv) # v ∈ B - W
            project1[v] = t1 += one(V); label1[t1] = v
        else             # v ∈ W ∩ B
            t2 += one(V); label2[t2] = v
            clique0[t2] = project0[v] = t0 += one(V); label0[t0] = v
            clique1[t2] = project1[v] = t1 += one(V); label1[t1] = v
        end
    end

    weights0 = Vector{V}(undef, n0); graph0 = BipartiteGraph{V, E}(n0, n0, m0)
    weights1 = Vector{V}(undef, n1); graph1 = BipartiteGraph{V, E}(n1, n1, m1)
    t0 = one(V); pointers(graph0)[t0] = p0 = one(E)
    t1 = one(V); pointers(graph1)[t1] = p1 = one(E)
    width0 = zero(W)
    width1 = zero(W)

    @inbounds for v in vertices(graph)
        vv = part[v]
        wt = weights[v]

        if iszero(vv)    # v ∈ W - B
            for w in neighbors(graph, v)                     # w ∈ W
                targets(graph0)[p0] = project0[w]; p0 += one(E)
            end

            width0 += wt; weights0[t0] = wt; t0 += one(V); pointers(graph0)[t0] = p0
        elseif isone(vv) # v ∈ B - W
            for w in neighbors(graph, v)                     # w ∈ B
                targets(graph1)[p1] = project1[w]; p1 += one(E)
            end

            width1 += wt; weights1[t1] = wt; t1 += one(V); pointers(graph1)[t1] = p1
        else             # v ∈ W ∩ B
            for w in neighbors(graph, v)
                ww = part[w]

                if iszero(ww)                                # w ∈ W - B
                    targets(graph0)[p0] = project0[w]; p0 += one(E)
                elseif isone(ww)                             # w ∈ B - W
                    targets(graph1)[p1] = project1[w]; p1 += one(E)
                end
            end

            for (w, w0, w1) in zip(label2, clique0, clique1) # w ∈ W ∩ B
                v == w && continue
                targets(graph0)[p0] = w0; p0 += one(E)
                targets(graph1)[p1] = w1; p1 += one(E)
            end

            width0 += wt; weights0[t0] = wt; t0 += one(V); pointers(graph0)[t0] = p0
            width1 += wt; weights1[t1] = wt; t1 += one(V); pointers(graph1)[t1] = p1
        end
    end

    child0 = (graph0, weights0, label0, clique0, width0)
    child1 = (graph1, weights1, label1, clique1, width1)
    return child0, child1, label2
end

function hpartition!(hpart::AbstractVector, part::AbstractVector, project0::AbstractVector, project1::AbstractVector, graph::AbstractGraph{V}) where {V}
    E = etype(graph)

    t0 = one(V); h0 = one(V)
    t1 = one(V); h1 = one(V)

    @inbounds for w in outvertices(graph)
        ww = hpart[w]

        if iszero(ww)
            h0 += one(V); project0[w] = t0 += one(V)
        else
            h1 += one(V); project1[w] = t1 += one(V)
        end
    end

    # V = W ∪ B
    n0 = zero(V); m0 = zero(E)
    n1 = zero(V); m1 = zero(E)

    @inbounds for v in vertices(graph)
        vv = three(V)

        for w in neighbors(graph, v)
            ww = hpart[w]

            if iszero(ww) # v ∈ W
                m0 += one(E)

                if isthree(vv)
                    n0 += one(V); vv = zero(V)
                elseif isone(vv)  # v ∈ W ∩ B
                    n0 += one(V); vv = two(V)
                    m0 += one(E)
                    m1 += one(E)
                end
            else          # v ∈ B
                m1 += one(E)

                if isthree(vv)
                    n1 += one(V); vv = one(V)
                elseif iszero(vv) # v ∈ W ∩ B
                    n1 += one(V); vv = two(V)
                    m0 += one(E)
                    m1 += one(E)
                end
            end
        end

        part[v] = vv
    end

    graph0 = BipartiteGraph{V, E}(h0, n0, m0)
    graph1 = BipartiteGraph{V, E}(h1, n1, m1)
    t0 = one(V); pointers(graph0)[t0] = p0 = one(E)
    t1 = one(V); pointers(graph1)[t1] = p1 = one(E)

    @inbounds for v in vertices(graph)
        vv = part[v]

        if iszero(vv)    # v ∈ W - B
            for w in neighbors(graph, v)
                targets(graph0)[p0] = project0[w]; p0 += one(E)
            end

            t0 += one(V); pointers(graph0)[t0] = p0
        elseif isone(vv) # v ∈ B - W
            for w in neighbors(graph, v)
                targets(graph1)[p1] = project1[w]; p1 += one(E)
            end

            t1 += one(V); pointers(graph1)[t1] = p1
        else             # v ∈ W ∩ B
            for w in neighbors(graph, v)
                ww = hpart[w]

                if iszero(ww)
                    targets(graph0)[p0] = project0[w]; p0 += one(E)
                elseif isone(ww)
                    targets(graph1)[p1] = project1[w]; p1 += one(E)
                end
            end

            targets(graph0)[p0] = one(V); p0 += one(E)
            targets(graph1)[p1] = one(V); p1 += one(E)
            t0 += one(V); pointers(graph0)[t0] = p0
            t1 += one(V); pointers(graph1)[t1] = p1
        end
    end

    return graph0, graph1
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
    n = nv(graph); m = ne(graph); mm = m + one(E)
    marker = zeros(V, n)

    #### bucket queue (degree) ########################
    degree = Vector{V}(undef, n)
    deghead = zeros(V, n)
    degprev = Vector{V}(undef, n)
    degnext = Vector{V}(undef, n)

    function degset(deg::V)
        @inbounds head = view(deghead, deg + one(V))
        return DoublyLinkedList(head, degprev, degnext)
    end
    ###################################################

    #### bucket queue (score) #########################
    score = Vector{E}(undef, n)
    scrhead = zeros(V, mm)
    scrprev = Vector{V}(undef, n)
    scrnext = Vector{V}(undef, n)

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
    ptrC = Vector{E}(undef, mm)
    tgtC = Vector{V}(undef, twice(m))
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
