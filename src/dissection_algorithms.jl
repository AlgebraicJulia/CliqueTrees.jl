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

function partition!(
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
    @argcheck nv(graph) <= length(label0)
    @argcheck nv(graph) <= length(label1)
    @argcheck nv(graph) < length(pointer0)
    @argcheck nv(graph) < length(pointer1)
    @argcheck de(graph) <= length(target0)
    @argcheck de(graph) <= length(target1)
    @argcheck nv(graph) <= length(part)
    @argcheck nv(graph) <= length(project0)
    @argcheck nv(graph) <= length(project1)
    @argcheck nv(graph) <= length(weights)

    n = nv(graph)

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

    t0 = zero(V)
    t1 = zero(V)
    t2 = zero(V); label2 = FVector{V}(undef, n2)

    @inbounds for v in vertices(graph)
        vv = part[v]

        if iszero(vv)    # v ∈ W - B
            project0[v] = t0 += one(V); label0[t0] = v
        elseif isone(vv) # v ∈ B - W
            project1[v] = t1 += one(V); label1[t1] = v
        else             # v ∈ W ∩ B
            t2 += one(V); label2[t2] = v
            project0[v] = t0 += one(V); label0[t0] = v
            project1[v] = t1 += one(V); label1[t1] = v
        end
    end

    if max(m0, m1) > length(target0)
        resize!(target0, max(m0, m1))
        resize!(target1, max(m0, m1))
    end

    graph0 = BipartiteGraph{V, E}(n0, n0, m0, pointer0, target0)
    graph1 = BipartiteGraph{V, E}(n1, n1, m1, pointer1, target1)
    t0 = one(V); pointers(graph0)[t0] = p0 = one(E)
    t1 = one(V); pointers(graph1)[t1] = p1 = one(E)

    @inbounds for v in vertices(graph)
        vv = part[v]

        if iszero(vv)    # v ∈ W - B
            for w in neighbors(graph, v)                     # w ∈ W
                targets(graph0)[p0] = project0[w]; p0 += one(E)
            end

            t0 += one(V); pointers(graph0)[t0] = p0
        elseif isone(vv) # v ∈ B - W
            for w in neighbors(graph, v)                     # w ∈ B
                targets(graph1)[p1] = project1[w]; p1 += one(E)
            end

            t1 += one(V); pointers(graph1)[t1] = p1
        else             # v ∈ W ∩ B
            for w in neighbors(graph, v)
                ww = part[w]

                if iszero(ww)                                # w ∈ W - B
                    targets(graph0)[p0] = project0[w]; p0 += one(E)
                elseif isone(ww)                             # w ∈ B - W
                    targets(graph1)[p1] = project1[w]; p1 += one(E)
                end
            end

            for w in label2                                  # w ∈ W ∩ B
                v == w && continue
                targets(graph0)[p0] = project0[w]; p0 += one(E)
                targets(graph1)[p1] = project1[w]; p1 += one(E)
            end

            t0 += one(V); pointers(graph0)[t0] = p0
            t1 += one(V); pointers(graph1)[t1] = p1
        end
    end

    child0 = compresspart(work0, work1, work2, work3,
        work4, work5, part, weights, n, graph0, label0)

    child1 = compresspart(work0, work1, work2, work3,
        work4, work5, part, weights, n, graph1, label1)

    return child0, child1, label2
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
    @argcheck nov(hgraph) <= length(hproject0)
    @argcheck nov(hgraph) <= length(hproject1)
    @argcheck nv(hgraph) <= length(part)
    @argcheck nv(hgraph) == nv(graph)

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

    child0, child1, label2 = partition!(work00, work01, work02, work03, work04,
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

function compresspart(
        work0::AbstractScalar{V},
        work1::AbstractVector{V},
        work2::AbstractVector{V},
        work3::AbstractVector{V},
        work4::AbstractVector{V},
        work5::AbstractVector{V},
        part::AbstractVector{V},
        supweights::AbstractVector{W},
        nsup::V,
        subgraph::AbstractGraph{V},
        sublabel::AbstractVector{V},
    ) where {W, V}
    @argcheck nsup <= length(supweights)
    @argcheck nv(subgraph) <= length(sublabel)

    E = etype(subgraph); nsub = nv(subgraph); msub = de(subgraph); nnsub = nsub + one(V)
    #
    #     + ------------------------- +
    #     |          subgraph         |
    #     |  project ↙     ↘ suplabel |
    #     |    cmpgraph ⇷ supgraph    |
    #     |          cmplabel         |
    #     + ------------------------- +
    #
    prjpointer = FVector{V}(undef, nnsub)
    prjtarget = FVector{V}(undef, nsub)
    cmppointer = FVector{E}(undef, nnsub)
    cmptarget = FVector{V}(undef, msub)
    cmptype = Val(true) # true twins
    
    cmpgraph, project = compress_impl!(work0, prjpointer, work1, prjtarget,
        work2, work3, work4, work5, cmppointer, cmptarget, subgraph, cmptype)

    ncmp = nv(cmpgraph); kcmp = zero(V)
    cmpweights = FVector{V}(undef, ncmp)
    cmplabel = BipartiteGraph(nsup, ncmp, nsub, prjpointer, prjtarget)

    @inbounds for v in vertices(subgraph)
        prjtarget[v] = sublabel[prjtarget[v]]
    end

    @inbounds for v in vertices(cmpgraph)
        w = zero(W); f = false
        
        for u in neighbors(cmplabel, v)
            w += supweights[u]; f = f || istwo(part[u])
        end

        cmpweights[v] = w

        if f
            kcmp += one(V); sublabel[kcmp] = v
        end
    end
    #
    #     + -------------------------------- +
    #     |              cmplabel    part    |
    #     |        cmpgraph ⇷ supgraph → 3   |
    #     | cmpclique ↑                  ↑ 3 |
    #     |          kcmp        →       1   |
    #     + -------------------------------- +
    #
    cmpclique = FVector{V}(undef, kcmp)

    @inbounds for i in oneto(kcmp)
        cmpclique[i] = sublabel[i]
    end

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
    @argcheck nov(hgraph) <= length(mark)
    @argcheck nv(hgraph) == nov(label)

    HE = etype(hgraph); hn = convert(HV, nv(label)); hm = convert(HE, length(clique))

    for v in vertices(label)
        tag += one(V)

        for w in neighbors(label, v), hw in neighbors(hgraph, w)
            hx = hproject[hw]

            if ispositive(hx) && mark[hx] < tag
                mark[hx] = tag
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

            if ispositive(hx) && mark[hx] < tag
                mark[hx] = tag
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
    @argcheck zero(W) < beta <= one(W)
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
