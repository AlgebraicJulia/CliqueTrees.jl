"""
    DissectionAlgorithm

An algorithm for computing a vertex separator of a graph.
"""
abstract type DissectionAlgorithm end

"""
    METISND <: DissectionAlgorithm

Compute a vertex separator using `METIS_computeVertexSeparator`.
"""
@kwdef struct METISND <: DissectionAlgorithm
    nseps::Int = -1
    ufactor::Int = -1
    seed::Int = -1
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

function Base.show(io::IO, ::MIME"text/plain", alg::METISND)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "METISND:")
    println(io, " "^indent * "    seed: $(alg.seed)")
    println(io, " "^indent * "    ufactor: $(alg.ufactor)")
    return
end

"""
    DEFAULT_DISSECTION_ALGORITHM = METISND()

The default dissection algorithm.
"""
const DEFAULT_DISSECTION_ALGORITHM = METISND()
