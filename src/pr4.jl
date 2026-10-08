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
#  PR-4 (PR-3 + Simplicial + Almost Simplicial)
function pr4(weights::AbstractVector{W}, graph::AbstractGraph, width::Number) where {W}
    weights0 = weights; graph0 = graph; width0 = width
    n0 = nv(graph0); weights1 = Vector{W}(undef, n0)

    graph1, stack1, inject1, width1 = pr3(weights0, graph0, width0)
    n1 = nv(graph1); m1 = n0 - n1

    @inbounds for i in oneto(n1)
        weights1[i] = weights0[inject1[i]]
    end

    graph2, stack2, inject2, width2 = asr(weights1, graph1, width1)
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

    # compute the fill-in of each vertex
    sr_fillin!(fillin, marker, stack0, tmpptr, source, number, target, begptr, endptr, n)

    @inbounds for v in vertices(graph)
        # if `v` is simplicial...
        if iszero(fillin[v])
            # add `v` to the stack of simplicial vertices
            hi0 = pr3_stack_add!(stack0, hi0, v)
        end
    end

    return hi0, mindeg
end

"""
    sr_fillin!(fillin, marker, order, tail, adjtgt, number,
        target, begptr, endptr, n)

Compute the fill-in of each vertex: the number of missing edges
in its neighborhood. Starting from the empty graph, add the edges
back one at a time, keeping the fill-in of every vertex up to
date with Wing-Huang updates. When the edge {`u`, `w`} is added,

  - every common neighbor of `u` and `w` loses one unit of fill-in
  - `u` gains one unit for each neighbor of `u` that is not a
    neighbor of `w`, and vice versa.

The vertices are processed in order of decreasing degree; when a
vertex `u` is processed, the edges joining `u` to unprocessed
vertices are added. The common neighbors of `u` and `w` are found
by scanning the processed neighbors of `w`, each of which has
degree at least that of `w`, so there are at most √(2m) of them,
and this takes O(m√m) time.

The neighbors of `v` are `target[begptr[v]:endptr[v] - 1]`, in any
order, and `number[v]` is the degree of `v`. The graph must be
symmetric, with no self loops or repeated neighbors.

working arrays:
  - `marker`: marker array (length ≥ n)
  - `order`: the vertices in order of decreasing degree (length ≥ n)
  - `tail`: the end of the processed neighbors of a vertex (length ≥ n)
  - `adjtgt`: the processed neighbors of a vertex, stored at the
    same positions as its neighbors in `target`
"""
function sr_fillin!(
        fillin::AbstractVector{F},
        marker::AbstractVector{V},
        order::AbstractVector{V},
        tail::AbstractVector{E},
        adjtgt::AbstractVector{V},
        number::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        n::V,
    ) where {V, E, F}

    iszero(n) && return

    # `maxnum` is the maximum degree
    maxnum = zero(V)

    @inbounds for v in oneto(n)
        maxnum = max(maxnum, number[v])
    end

    # the counting sort below uses `marker[1:maxnum + 1]`
    @assert maxnum < n

    # sort the vertices by decreasing degree, using
    # `marker` to count the vertices of each degree
    @inbounds for j in oneto(maxnum + one(V))
        marker[j] = zero(V)
    end

    @inbounds for v in oneto(n)
        j = number[v] + one(V); marker[j] += one(V)
    end

    # `pos` is the first position in `order` of the
    # vertices of degree `num`
    pos = one(V)

    @inbounds for num in maxnum:-one(V):zero(V)
        j = num + one(V); k = marker[j]; marker[j] = pos; pos += k
    end

    @inbounds for v in oneto(n)
        j = number[v] + one(V); pos = marker[j]; order[pos] = v; marker[j] = pos + one(V)
    end

    # start from the empty graph
    @inbounds for v in oneto(n)
        marker[v] = zero(V)
        tail[v] = begptr[v]
        fillin[v] = zero(F)
    end

    # for each vertex `u`, in order of decreasing degree...
    @inbounds for i in oneto(n)
        u = order[i]

        # mark the processed neighbors of `u`: these are
        # its neighbors in the growing graph
        for p in begptr[u]:tail[u] - one(E)
            marker[adjtgt[p]] = i
        end

        # `unum` is the degree of `u` in the growing graph
        unum = convert(F, tail[u] - begptr[u])

        # `ufil` is the fill-in gained by `u`
        ufil = zero(F)

        # for all neighbors `w` of `u`...
        for p in begptr[u]:endptr[u] - one(E)
            w = target[p]

            # if `w` is not processed, then it is not marked:
            # add the edge {`u`, `w`} to the growing graph
            if marker[w] != i
                # the neighbors of `w` in the growing graph
                # are the arcs {`wbeg`, ..., `wtail` - 1}
                wbeg = begptr[w]; wtail = tail[w]

                # `cnt` is the number of common neighbors
                # of `u` and `w`
                cnt = zero(F)

                for q in wbeg:wtail - one(E)
                    x = adjtgt[q]

                    # if `x` is a common neighbor of `u` and `w`,
                    # then {`u`, `w`} is no longer missing from
                    # the neighborhood of `x`
                    if marker[x] == i
                        cnt += one(F); fillin[x] -= one(F)
                    end
                end

                # `u` and `w` gain one unit of fill-in for each
                # of their neighbors that are not common
                ufil += unum - cnt
                fillin[w] += convert(F, wtail - wbeg) - cnt

                # add `u` to the neighbors of `w`
                adjtgt[wtail] = u; tail[w] = wtail + one(E)

                # increment the degree of `u`
                unum += one(F)
            end
        end

        fillin[u] += ufil
    end

    return
end
