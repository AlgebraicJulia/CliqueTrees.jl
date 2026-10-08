function pr4(weights::AbstractVector{W}, graph::AbstractGraph{V}, width::Number) where {W, V}
    E = etype(graph); n = nv(graph); m = de(graph); nn = n + one(V)

    weight1 = FVector{W}(undef, n)

    degree = FVector{W}(undef, n)
    number = FVector{V}(undef, n)
    status = FVector{UInt8}(undef, n)
    marker = FVector{V}(undef, n)
    source = FVector{V}(undef, m)
    target = FVector{V}(undef, m)
    begptr = FVector{E}(undef, nn)
    endptr = FVector{E}(undef, n)
    invptr = FVector{E}(undef, m)
    stack1 = FVector{V}(undef, n)
    stack2 = FVector{V}(undef, n)
    stack3 = FVector{V}(undef, n)
    stack4 = FVector{V}(undef, n)
    stack5 = FVector{V}(undef, n)
    stack6 = FVector{V}(undef, n)
    stack7 = FVector{V}(undef, n)
    stack8 = FVector{V}(undef, n)
    stack0 = FVector{V}(undef, n)
    tmpptr = FVector{E}(undef, nn)

    sfillin = FVector{Int}(undef, n)
    ssource = FVector{V}(undef, m)
    starget = FVector{V}(undef, m)
    sbegptr = FVector{E}(undef, nn)
    stmpptr = FVector{E}(undef, nn)

    return pr4_impl!(
        degree, number, status, marker, source, target, begptr, endptr, invptr,
        stack1, stack2, stack3, stack4, stack5, stack6, stack7, stack8, stack0, tmpptr,
        sfillin, ssource, starget, sbegptr, stmpptr, weight1, weights, graph, convert(W, width))
end

function pr4_impl!(
        degree::AbstractVector{W}, number::AbstractVector{V}, status::AbstractVector{UInt8},
        marker::AbstractVector{V}, source::AbstractVector{V}, target::AbstractVector{V},
        begptr::AbstractVector{E}, endptr::AbstractVector{E}, invptr::AbstractVector{E},
        stack1::AbstractVector{V}, stack2::AbstractVector{V}, stack3::AbstractVector{V},
        stack4::AbstractVector{V}, stack5::AbstractVector{V}, stack6::AbstractVector{V},
        stack7::AbstractVector{V}, stack8::AbstractVector{V}, stack0::AbstractVector{V},
        tmpptr::AbstractVector{E},
        sfillin::AbstractVector{F}, ssource::AbstractVector{V}, starget::AbstractVector{V},
        sbegptr::AbstractVector{E}, stmpptr::AbstractVector{E},
        weight1::AbstractVector{W}, weight::AbstractVector{W}, graph::AbstractGraph{V}, width::W,
    ) where {W, V, E, F <: Integer}
    n0 = nv(graph)

    totdeg = zero(W)

    @inbounds for v in oneto(n0)
        totdeg += weight[v]
    end

    graph1, stack, inject1, width1 = pr3_impl!(
        weight, degree, number, status, marker, source, target, begptr,
        endptr, invptr, stack1, stack2, stack3, stack4, stack5, stack6,
        stack7, stack8, stack0, tmpptr, totdeg, width, graph)

    n1 = nv(graph1); m1 = n0 - n1

    totdeg1 = zero(W)

    @inbounds for i in oneto(n1)
        w = weight[inject1[i]]
        weight1[i] = w; totdeg1 += w
    end

    graph2, sstack, inject2, width2 = sr_impl!(
        marker, stack1, stack2, stmpptr, sfillin, degree, number,
        ssource, starget, sbegptr, endptr, invptr, totdeg1, weight1, graph1, width1)

    n2 = nv(graph2); m2 = n1 - n2

    @inbounds for i in oneto(m2)
        stack[i + m1] = inject1[sstack[i]]
    end

    @inbounds for i in oneto(n2)
        inject2[i] = inject1[inject2[i]]
    end

    return (graph2, stack, inject2, width2)
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
    fillin!(fillin, marker, stack0, tmpptr, source, number, target, begptr, endptr, n)

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
    fillin!(fillin, marker, order, tail, adjtgt, number,
        target, begptr, endptr, n)

    fillin!(fillin, weights, wdegree, marker, order, tail, adjtgt,
        number, target, begptr, endptr, n)

Compute the fill-in of each vertex: the weight of the missing edges
in its neighborhood, where a missing edge {`a`, `b`} has weight
`weights[a] * weights[b]`. Without `weights`, every vertex has
weight one, and the fill-in is the number of missing edges.

Starting from the empty graph, add the edges back one at a time,
keeping the fill-in of every vertex up to date with Wing-Huang
updates. When the edge {`u`, `w`} is added,

  - every common neighbor of `u` and `w` loses `weights[u] * weights[w]`
  - `u` gains `weights[w] * weights[x]` for each neighbor `x` of `u`
    that is not a neighbor of `w`, and vice versa.

The vertices are processed in order of decreasing degree; when a
vertex `u` is processed, the edges joining `u` to unprocessed
vertices are added. The common neighbors of `u` and `w` are found
by scanning the processed neighbors of `w`, each of which has
degree at least that of `w`, so there are at most √(2m) of them,
and this takes O(m√m) time. The weights do not affect the order.

The neighbors of `v` are `target[begptr[v]:endptr[v] - 1]`, in any
order, and `number[v]` is the degree of `v`. The graph must be
symmetric, with no self loops or repeated neighbors.

working arrays:
  - `marker`: marker array (length ≥ n)
  - `order`: the vertices in order of decreasing degree (length ≥ n)
  - `tail`: the end of the processed neighbors of a vertex (length ≥ n)
  - `adjtgt`: the processed neighbors of a vertex, stored at the
    same positions as its neighbors in `target`
  - `wdegree`: the weighted degree of a vertex in the growing graph
    (length ≥ n); unused if `weights` is a `Ones`
"""
function fillin!(
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
    weights = Ones{F}(n)
    return fillin!(fillin, weights, fillin, marker, order, tail, adjtgt, number, target, begptr, endptr, n)
end

function fillin!(
        fillin::AbstractVector{F},
        weights::AbstractVector,
        wdegree::AbstractVector{F},
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

    # `maxnum` is the maximum degree
    maxnum = -one(V)

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

    fillin_init!(wdegree, weights, n)

    # for each vertex `u`, in order of decreasing degree...
    @inbounds for i in oneto(n)
        u = order[i]

        # mark the processed neighbors of `u`: these are
        # its neighbors in the growing graph
        for p in begptr[u]:tail[u] - one(E)
            marker[adjtgt[p]] = i
        end

        # `unum` is the weighted degree of `u` in the growing graph
        unum = fillin_degree(wdegree, weights, begptr, tail, u)
        uwgt = convert(F, weights[u])

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
                wwgt = convert(F, weights[w]); uwwgt = uwgt * wwgt

                # `cnt` is the weight of the common
                # neighbors of `u` and `w`
                cnt = zero(F)

                for q in wbeg:wtail - one(E)
                    x = adjtgt[q]

                    # if `x` is a common neighbor of `u` and `w`,
                    # then {`u`, `w`} is no longer missing from
                    # the neighborhood of `x`
                    if marker[x] == i
                        cnt += convert(F, weights[x]); fillin[x] -= uwwgt
                    end
                end

                # `u` and `w` gain fill-in for each of
                # their neighbors that are not common
                ufil += wwgt * (unum - cnt)
                fillin[w] += uwgt * (fillin_degree(wdegree, weights, begptr, tail, w) - cnt)

                # add `u` to the neighbors of `w`
                adjtgt[wtail] = u; tail[w] = wtail + one(E)
                fillin_addarc!(wdegree, weights, w, uwgt)

                # increment the weighted degree of `u`
                unum += wwgt
            end
        end

        fillin[u] += ufil
    end

    return
end

# With unit weights, the weighted degree of a vertex in the growing graph
# is its number of processed neighbors, and `wdegree` is not used.
@inline function fillin_init!(wdegree::AbstractVector{F}, weights::Ones, n::Integer) where {F}
    return
end

@inline function fillin_init!(wdegree::AbstractVector{F}, weights::AbstractVector, n::Integer) where {F}
    @inbounds for v in oneto(n)
        wdegree[v] = zero(F)
    end

    return
end

@inline function fillin_degree(wdegree::AbstractVector{F}, weights::Ones, begptr::AbstractVector, tail::AbstractVector, v::Integer) where {F}
    return @inbounds convert(F, tail[v] - begptr[v])
end

@inline function fillin_degree(wdegree::AbstractVector{F}, weights::AbstractVector, begptr::AbstractVector, tail::AbstractVector, v::Integer) where {F}
    return @inbounds wdegree[v]
end

@inline function fillin_addarc!(wdegree::AbstractVector{F}, weights::Ones, w::Integer, uwgt::F) where {F}
    return
end

@inline function fillin_addarc!(wdegree::AbstractVector{F}, weights::AbstractVector, w::Integer, uwgt::F) where {F}
    @inbounds wdegree[w] += uwgt
    return
end
