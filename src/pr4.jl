function compressreduce(reduce::F, weights::AbstractVector{W}, graph::AbstractGraph{V}, width::W, tao::Number) where {F <: Function, W <: Number, V <: Integer}
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
    graph10, project10 = compress(graph02, Val(true), tao)
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
        graph20, project20 = compress(graph12, Val(true), tao)
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

# Pre-processing for Triangulation of Probabilistic Networks
# Bodlaender, Koster, Eijkhof, and van der Gaag
#
# Preprocessing Rules for Triangulation of Probabilistic Networks
# Bodlaender, Koster, Eijkhof, and van der Gaag
#
#  PR-4 (PR-3 + Simplicial)
function pr4(weights::AbstractVector{W}, graph::AbstractGraph, width::Number) where {W}
    weights0 = weights; graph0 = graph; width0 = width
    n0 = nv(graph0); weights1 = Vector{W}(undef, n0)

    graph1, stack1, inject1, width1 = pr3(weights0, graph0, width0)
    n1 = nv(graph1); m1 = n0 - n1

    @inbounds for i in oneto(n1)
        weights1[i] = weights0[inject1[i]]
    end

    graph2, stack2, inject2, width2 = sr(weights1, graph1, width1)
    n2 = nv(graph2); m2 = n1 - n2

    @inbounds for i in oneto(m2)
        stack1[i + m1] = inject1[stack2[i]]
    end

    @inbounds for i in oneto(n2)
        inject2[i] = inject1[inject2[i]]
    end

    return (graph2, stack1, inject2, width2)
end

function sr(weights::AbstractVector{W}, graph::AbstractGraph, width::Number) where {W <: Number}
    return sr(weights, graph, convert(W, width))
end

function sr(weights::AbstractVector{W}, graph::AbstractGraph{V}, width::W) where {W <: Number, V <: Integer}
    @assert nv(graph) <= length(weights)

    E = etype(graph); n = nv(graph); m = de(graph); nn = n + one(V)

    # `totdeg` is the total weight of the
    # vertices in the graph
    totdeg = zero(W)

    @inbounds for v in oneto(n)
        totdeg += weights[v]
    end

    marker = FVector{V}(undef, n)
    stack0 = FVector{V}(undef, n)
    stack1 = FVector{V}(undef, n)
    tmpptr = FVector{E}(undef, nn)

    fillin = FVector{Int}(undef, n)
    degree = FVector{W}(undef, n)
    number = FVector{V}(undef, n)
    source = FVector{V}(undef, m)
    target = FVector{V}(undef, m)
    begptr = FVector{E}(undef, nn)
    endptr = FVector{E}(undef, n)
    invptr = FVector{E}(undef, m)

    kernel, stack, inject, width = sr_impl!(marker, stack0, stack1,
        tmpptr, fillin, degree, number, source,
        target, begptr, endptr, invptr, totdeg, weights, graph, width)

    return kernel, stack, inject, width
end

function sr_impl!(
        marker::AbstractVector{V},
        stack0::AbstractVector{V},
        stack1::AbstractVector{V},
        tmpptr::AbstractVector{E},
        fillin::AbstractVector{F},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        totdeg::W,
        weight::AbstractVector{W},
        graph::AbstractGraph{V},
        width::W,
    ) where {W, V, E, F <: Integer}
    @assert nv(graph) <= length(marker)
    @assert nv(graph) <= length(stack0)
    @assert nv(graph) <= length(stack1)
    @assert nv(graph) < length(tmpptr)
    @assert nv(graph) <= length(fillin)
    @assert nv(graph) <= length(degree)
    @assert nv(graph) <= length(number)
    @assert de(graph) <= length(source)
    @assert de(graph) <= length(target)
    @assert nv(graph) < length(begptr)
    @assert nv(graph) <= length(endptr)
    @assert de(graph) <= length(invptr)

    # `n` is the number of vertices in the input graph
    n = nv(graph)

    # `hi0` is the number of simplicial vertices
    # `mindeg` is the minimum weighted degree
    hi0, mindeg = sr_init!(marker, stack0, tmpptr, fillin, degree, number,
        source, target, begptr, endptr, invptr, totdeg, weight, graph)

    # the weighted treewidth is at least the minimum
    # weighted degree
    width = max(width, mindeg)

    # `hi` is the number of eliminated vertices
    hi1 = zero(V)

    # while there exists a simplicial vertex...
    @inbounds while ispositive(hi0)
        # `v` is a simplicial vertex
        hi0, v = pr3_stack_pop!(stack0, hi0)

        # add `v` to the stack of eliminated
        # vertices
        hi1 = pr3_stack_add!(stack1, hi1, v)

        # `deg` is the weighted degree of `v`
        deg = degree[v]

        # `num` is the degree of `v`
        num = number[v]

        # `wgt` is the weight of `v`
        wgt = weight[v]

        # `v` is simplicial: update the lower bound
        width = max(width, deg)

        # `v` is incident to the arcs
        # {`p`, ..., `pend` - 1}
        p = begptr[v]; pend = endptr[v]

        while p < pend
            # `p` is the arc (`v`, `w`)
            w = target[p]

            # `wfil` is the degeneracy of `w`
            wfil = fillin[w]

            # `wdeg` is the weighted degree of `w`
            wdeg = degree[w]

            # `wnum` is the degree of `w`
            wnum = number[w]

            # `q` is the arc (`w`, `v`)
            q = invptr[p]

            # `qend` is the last arc incident to `w`
            qend = endptr[w] -= one(E)

            # replace `v` with a vertex `x` in the
            # neighborhood of `w`
            if q < qend
                # `qend` is the arc (`w`, `x`)
                x = target[q] = target[qend]

                # `qinv` is the arc (`x`, `w`)
                qinv = invptr[q] = invptr[qend]
                invptr[qinv] = q
            end

            # increase the degeneracy of `w` by the
            # degree of `v` and decrease it by the
            # degree of `w`
            fillin[w] = wfil - convert(F, wnum - num)

            # decrease the weighted degree of `w` by
            # the weight of `v`
            degree[w] = wdeg - wgt

            # decrement the degree of `w`
            number[w] = wnum - one(V)

            # if `w` is simplicial...
            if iszero(fillin[w]) && ispositive(wfil)
                # add `w` to the stack of simplicial
                # vertices
                hi0 = pr3_stack_add!(stack0, hi0, w)
            end

            # increment `p`
            p += one(E)
        end
    end

    # construct the reduced graph
    m, n = sr_make!(stack1, target, begptr, endptr,
        stack0, number, tmpptr, source, hi1, n)

    # `kernel` is the reduced graph
    kernel = BipartiteGraph(n, n, m, tmpptr, source)
    return kernel, stack1, stack0, width
end

function sr_make!(
        stack1::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        inj::AbstractVector{V},
        prj::AbstractVector{V},
        ptr::AbstractVector{E},
        tgt::AbstractVector{V},
        hi1::V,
        n::V,
    ) where {V, E}

    # mark eliminated vertices with -1
    @inbounds for i in oneto(hi1)
        w = stack1[i]; prj[w] = -one(V)
    end

    # `v` is a vertex in the reduced graph
    v = one(V)

    # for a vertices `w`...
    @inbounds for w in oneto(n)
        # if `w` is not eliminated...
        if !isnegative(prj[w])
            # associate `v` and `w`
            prj[w] = v
            inj[v] = w

            # increment `v`
            v += one(V)
        end
    end

    # `v` is a vertex in the reduced graph
    @inbounds v = one(V)

    # `p` is an edge incident to `v`
    @inbounds ptr[v] = p = one(E)

    @inbounds while v + hi1 <= n
        # `w` is the vertex associated
        # to `v`
        w = inj[v]

        # `w` is incident to the arcs
        #     {`q`, ..., `qend` - 1}
        q = begptr[w]; qend = endptr[w]

        while q < qend
            # `q` is the arc (`w`, `x`)
            x = target[q]; q += one(E)
            tgt[p] = prj[x]; p += one(E)
        end

        v += one(V); ptr[v] = p
    end

    # `m` is the number of arcs in the
    # reduced graph
    m = p - one(E)

    # `n` is the number of vertices in the
    # reduced graph
    n = v - one(V)
    return m, n
end

function sr_init!(
        marker::AbstractVector{V},
        stack0::AbstractVector{V},
        tmpptr::AbstractVector{E},
        fillin::AbstractVector{F},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        totdeg::W,
        weight::AbstractVector{W},
        graph::AbstractGraph{V},
    ) where {W, V, E, F}

    # `n` is the number of vertices in the
    # input graph
    n = nv(graph); nn = n + one(V)

    # `hi0` is the number of simplicial
    # vertices
    hi0 = zero(V)

    # `mindeg` is the minimum weighted degree
    mindeg = totdeg

    # `sorted` is true if every neighborhood is sorted
    sorted = true

    # `p` is the current arc
    p = one(E)

    # copy the graph into the array `target`
    @inbounds for v in vertices(graph)
        begptr[v] = p

        # `deg` is the weighted degree of `v`
        deg = weight[v]

        # `num` is the degree of `v`
        num = zero(V)

        # `prv` is the previous neighbor of `v`
        prv = zero(V)

        # for all neighbors `w` of `v`...
        for w in neighbors(graph, v)
            # ignore self loops
            if v != w
                # `p` is the arc (`v`, `w`)
                target[p] = w; p += one(E)

                # check if the neighborhood of `v` is sorted
                sorted = sorted && prv < w; prv = w

                # increase `deg` by the weight of `w`
                deg += weight[w]

                # increment `num`
                num += one(V)
            end
        end

        endptr[v] = p

        # update the minimum weighted degree
        mindeg = min(mindeg, deg)
        degree[v] = deg
        number[v] = num
    end

    @inbounds begptr[nn] = p

    # if the neighborhoods are not sorted, sort them by
    # transposing the graph
    @inbounds if !sorted
        for v in vertices(graph)
            tmpptr[v] = begptr[v]
        end

        for v in vertices(graph)
            p = begptr[v]; pend = endptr[v]

            while p < pend
                # `p` is the arc (`v`, `w`)
                w = target[p]; p += one(E)

                # `q` is the arc (`w`, `v`)
                q = tmpptr[w]; invptr[q] = v; tmpptr[w] = q + one(E)
            end
        end

        copyto!(target, begptr[begin], invptr, begptr[begin], begptr[nn] - one(E))
    end

    # compute the reverse of every arc: since the neighborhoods
    # are sorted, the reverse of the arc (`v`, `w`) is the first
    # unvisited arc incident to `w`
    @inbounds for v in vertices(graph)
        tmpptr[v] = begptr[v]
    end

    @inbounds for v in vertices(graph)
        p = begptr[v]; pend = endptr[v]

        while p < pend
            # `p` is the arc (`v`, `w`)
            w = target[p]

            # `q` is the arc (`w`, `v`)
            q = tmpptr[w]; invptr[p] = q; tmpptr[w] = q + one(E)
            p += one(E)
        end
    end

    # count the triangles containing each vertex
    sr_triangles!(marker, tmpptr, fillin, number, source, target, begptr, endptr, n)

    @inbounds for v in vertices(graph)
        # `num` is the degree of `v`
        num = Int(number[v])

        # the fill-in of `v` is the number of missing
        # edges in its neighborhood
        fillin[v] = convert(F, (num * (num - 1)) ÷ 2) - fillin[v]

        # if `v` is simplicial...
        if iszero(fillin[v])
            # add `v` to the stack of simplicial vertices
            hi0 = pr3_stack_add!(stack0, hi0, v)
        end
    end

    return hi0, mindeg
end

"""
    sr_triangles!(marker, fwdptr, count, number, fwdtgt, target,
        begptr, endptr, n)

Count the triangles containing each vertex, writing the counts
to `count`. Each edge is oriented toward the endpoint of larger
(degree, label); every triangle {u, v, w} with u < v < w in this
order is found once, by intersecting the out-neighborhoods of
u and v. An out-neighborhood has at most √(2m) vertices, so this
takes O(m√m) time, instead of the O(Σ deg(v)²) time taken by
scanning N(w) for every neighbor w of every vertex.

The neighbors of `v` are `target[begptr[v]:endptr[v] - 1]`, in any
order. Every edge must appear in the lists of both endpoints, and
there must be no self loops. `number[v]` should be the degree of `v`
(any values give correct counts; degrees give the time bound).

working arrays:
  - `marker`: marker array (length ≥ n; must hold values up to n)
  - `fwdptr`: the first out-arc of a vertex (length ≥ n + 1)
  - `fwdtgt`: the target of an out-arc (length ≥ number of edges)
"""
function sr_triangles!(
        marker::AbstractVector{M},
        fwdptr::AbstractVector{E},
        count::AbstractVector{F},
        number::AbstractVector{V},
        fwdtgt::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        n::V,
    ) where {V, E, F, M}

    # construct the oriented graph
    @inbounds fwdptr[begin] = q = one(E)

    @inbounds for v in oneto(n)
        vnum = number[v]
        count[v] = zero(F)
        marker[v] = zero(M)

        for p in begptr[v]:endptr[v] - one(E)
            w = target[p]; wnum = number[w]

            if vnum < wnum || (vnum == wnum && v < w)
                fwdtgt[q] = w; q += one(E)
            end
        end

        fwdptr[v + one(V)] = q
    end

    # for each vertex `u`...
    @inbounds for u in oneto(n)
        tag = convert(M, u)

        # mark the out-neighbors of `u`
        for p in fwdptr[u]:fwdptr[u + one(V)] - one(E)
            marker[fwdtgt[p]] = tag
        end

        # for each out-neighbor `v` of `u`...
        for p in fwdptr[u]:fwdptr[u + one(V)] - one(E)
            v = fwdtgt[p]; c = zero(F)

            # for each out-neighbor `w` of `v`...
            for q in fwdptr[v]:fwdptr[v + one(V)] - one(E)
                w = fwdtgt[q]

                # if `w` is an out-neighbor of `u`, then
                # {`u`, `v`, `w`} is a triangle
                if marker[w] == tag
                    c += one(F); count[w] += one(F)
                end
            end

            count[u] += c; count[v] += c
        end
    end

    return
end
