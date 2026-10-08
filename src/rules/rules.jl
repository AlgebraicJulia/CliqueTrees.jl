function compressreduce(rules::Integer, weights::AbstractVector{W}, graph::AbstractGraph{V}, width::W) where {W <: Number, V <: Integer}
    if rules == 3
        return compressreduce(pr3, weights, graph, width)
    elseif rules == 4
        return compressreduce(pr4, weights, graph, width)
    else
        return compressreduce(pr5, weights, graph, width)
    end
end

function compressreduce(reduce::F, weights::AbstractVector{W}, graph::AbstractGraph{V}, width::W) where {F <: Function, W <: Number, V <: Integer}
    weights00 = weights; graph00 = graph; width00 = width; n00 = nv(graph00)
    inject03 = Vector{V}(undef, n00); n03 = zero(V)
    # V01
    #  ↓ inject01
    # V00
    #  ↑ inject02
    # V02
    #  ↓ project10
    # V10
    graph02, inject01, inject02, width10 = reduce(weights00, graph00, width00)
    graph10, project10 = compress(graph02, Val(true))
    n02 = nv(graph02); n10 = nv(graph10)

    @inbounds for v01 in oneto(n00 - n02)
        v00 = inject01[v01]
        n03 += one(V); inject03[n03] = v00
    end
    #   inject02 V02
    #        ↙    ↓ project10
    #   V00  →   V10
    #     project11
    weights10 = Vector{W}(undef, n10)
    project11 = BipartiteGraph{V, V}(n00, n10, n00 - n03)
    @inbounds pointers(project11)[begin] = p = one(V)

    @inbounds for v10 in vertices(graph10)
        w10 = zero(W); vv10 = v10 + one(V)

        for v02 in neighbors(project10, v10)
            v00 = inject02[v02]
            w10 += weights00[v00]
            targets(project11)[p] = v00; p += one(V)
        end

        weights10[v10] = w10
        pointers(project11)[vv10] = p
    end

    lo = n10; hi = n00

    @inbounds while lo < hi
        hi = lo
        # V11
        #  ↓ inject11
        # V10
        #  ↑ inject12
        # V12
        #  ↓ project20
        # V20
        graph12, inject11, inject12, width20 = reduce(weights10, graph10, width10)
        graph20, project20 = compress(graph12, Val(true))
        n12 = nv(graph12); n20 = nv(graph20)

        for v11 in oneto(n10 - n12)
            v10 = inject11[v11]

            for v00 in neighbors(project11, v10)
                n03 += one(V); inject03[n03] = v00
            end
        end
        #            inject12
        #          V10  ←   V12
        # project11 ↑        ↓ project20
        #          V00  →   V20
        #           project21

        weights20 = Vector{W}(undef, n20)
        project21 = BipartiteGraph{V, V}(n00, n20, n00 - n03)
        pointers(project21)[begin] = p = one(V)

        for v20 in vertices(graph20)
            w20 = zero(W); vv20 = v20 + one(V)

            for v12 in neighbors(project20, v20)
                v10 = inject12[v12]
                w20 += weights10[v10]

                for v00 in neighbors(project11, v10)
                    targets(project21)[p] = v00; p += one(V)
                end
            end

            weights20[v20] = w20
            pointers(project21)[vv20] = p
        end

        lo = n10 = n20; weights10 = weights20; graph10 = graph20; width10 = width20; project11 = project21
    end

    return weights10, graph10, view(inject03, oneto(n03)), project11, width10
end

include("pr3.jl")
include("pr4.jl")
include("pr5.jl")

