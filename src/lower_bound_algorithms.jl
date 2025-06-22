"""
    LowerBoundAlgorithm

An algorithm for computing a lower bound to the treewidth of a graph. The options are

| type          | name            | time | space    |
|:--------------|:----------------|:-----|:---------|
| [`MMW`](@ref) | minor-min-width |      | O(m + n) |
"""
abstract type LowerBoundAlgorithm end

"""
    WidthOrAlgorithm = Union{Number, LowerBoundAlgorithm}
"""
const WidthOrAlgorithm = Union{Number, LowerBoundAlgorithm}

"""
    MMW{S} <: LowerBoundAlgorithm

    MMW{S}()

The minor-min-width heuristic.

```jldoctest
julia> using CliqueTrees

julia> graph = [
           0 1 0 0 0 0 0 0
           1 0 1 0 0 1 0 0
           0 1 0 1 0 1 1 1
           0 0 1 0 0 0 0 0
           0 0 0 0 0 1 1 0
           0 1 1 0 1 0 0 0
           0 0 1 0 1 0 0 1
           0 0 1 0 0 0 1 0
       ];

julia> alg = MMW{1}()
MMW{1}

julia> lowerbound(graph; alg)
2
```

### Parameters

  - `S`: strategy
    - `1`: min-d (fast)
    - `2`: max-d (fast)
    - `3`: least-c (slow)

### References

  - Gogate, Vibhav, and Rina Dechter. "A complete anytime algorithm for treewidth." *Proceedings of the 20th conference on Uncertainty in artificial intelligence.* 2004.
  - Bodlaender, Hans, Thomas Wolle, and Arie Koster. "Contraction and treewidth lower bounds." *Journal of Graph Algorithms and Applications* 10.1 (2006): 5-49.
"""
struct MMW{S} <: LowerBoundAlgorithm end

function MMW()
    return MMW{3}()
end

"""
    lowerbound([weights, ]graph;
        alg::WidthOrAlgorithm=DEFAULT_LOWER_BOUND_ALGORITHM)

Compute a lower bound to the treewidth of a graph.

```jldoctest
julia> using CliqueTrees

julia> graph = [
           0 1 0 0 0 0 0 0
           1 0 1 0 0 1 0 0
           0 1 0 1 0 1 1 1
           0 0 1 0 0 0 0 0
           0 0 0 0 0 1 1 0
           0 1 1 0 1 0 0 0
           0 0 1 0 1 0 0 1
           0 0 1 0 0 0 1 0
       ];

julia> lowerbound(graph)
2
```
"""
function lowerbound(graph; alg::WidthOrAlgorithm = DEFAULT_LOWER_BOUND_ALGORITHM)
    return lowerbound(graph, alg)
end

function lowerbound(weights::AbstractVector, graph; alg::WidthOrAlgorithm = DEFAULT_LOWER_BOUND_ALGORITHM)
    return lowerbound(weights, graph, alg)
end

function lowerbound(graph, alg::Number)
    return lowerbound(BipartiteGraph(graph), alg)
end

function lowerbound(graph::AbstractGraph{V}, alg::Number) where {V}
    width = convert(V, alg)
    return width
end

function lowerbound(weights::AbstractVector{W}, graph, alg::Number) where {W}
    width = convert(W, alg)
    return width
end

function lowerbound(graph, ::MMW{S}) where {S}
    strategy = Val(S)
    return mmw(graph, strategy)
end

function lowerbound(weights::AbstractVector, graph, ::MMW{S}) where {S}
    strategy = Val(S)
    return mmw(weights, graph, strategy)
end

function mmw(graph, strategy::Val)
    return mmw(BipartiteGraph(graph), strategy)
end

function mmw(graph::AbstractGraph{V}, strategy::Val) where {V}
    weights = Ones{V}(nv(graph))
    width = mmw(weights, graph, strategy)
    return width - one(V)
end

function mmw(weights::AbstractVector, graph, strategy::Val)
    return mmw(weights, BipartiteGraph(graph), strategy)
end

function mmw(weights::AbstractVector{W}, graph::AbstractGraph{V}, strategy::Val) where {W, V}
    width = mmw(trunc.(V, weights), graph, strategy)
    return convert(W, width)
end

function mmw(weights::AbstractVector{V}, graph::AbstractGraph{V}, strategy::Val) where {V}
    @argcheck nv(graph) <= length(weights)

    E = etype(graph); n = nv(graph); m = de(graph); nn = n + one(V)

    # `totdeg` is the total weight of the
    # vertices in the graph
    totdeg = zero(V)

    @inbounds for v in oneto(n)
        totdeg += weights[v]
    end
    
    marker = FixedSizeVector{V}(undef, n)
    vstack = FixedSizeVector{V}(undef, n)
    tmpptr = FixedSizeVector{E}(undef, n)

    degree = FixedSizeVector{V}(undef, n)
    source = FixedSizeVector{V}(undef, m)
    target = FixedSizeVector{V}(undef, m)
    begptr = FixedSizeVector{E}(undef, nn)
    endptr = FixedSizeVector{E}(undef, n)
    invptr = FixedSizeVector{E}(undef, m)

    head = FixedSizeVector{V}(undef, totdeg)
    prev = FixedSizeVector{V}(undef, n)
    next = FixedSizeVector{V}(undef, n)
    
    width = mmw_impl!(marker, vstack, tmpptr, degree, source, target, begptr,
        endptr, invptr, head, prev, next, totdeg, weights, graph, strategy)
    
    return width
end

"""
  mmw_impl!(marker, vstack, tmpptr, degree, source, target, begptr,
    endptr, invptr, head, prev, next, totdeg, weights, graph, strategy)  

Contraction and Treewidth Lower Bounds
Bodlaender, Koster, and Wolle
MMD+ heuristic

A Complete Anytime Algorithm for Treewidth
Gogate and Dechter
minor-min-width

Find a lower bound to the weighted treewidth of a graph
by constructing a sequence of graph minors. The treewidth
of the graph is lower-bounded by the treewidth of each
minor, and the treewidth of each minor is lower-bounded by
its minimum degree.

The algorithm employs a data structure called a *quotient graph.*
It is a directed graph with two types of vertices: elements and
supernodes. The arcs in the graph obey the following invariant
  - every supernode w has at most one predecessor v, which must
    be an element
We say that an element w is *reachable* by another element v if
there exists a path (v, x₁, ..., xₙ, w) from v to w through
supernodes {x₁, ..., xₙ}. Reachability is a symmetric, irreflexive
relation on the set of elements.

input parameters:
  - `totdeg`: total vertex weight
  - `weights`: vertex weights
  - `graph`: input graph
  - `S`: strategy
    - `1`: min-d (fast)
    - `2`: max-d (fast)
    - `3`: least-c (slow)

output parameters:
  - `maxmindeg`: maximum minimum degree

working arrays:
  - miscellaneous:
    - `marker`: marker array
    - `vstack`: vertex stack
    - `tmpptr`: temporary pointer array 
  - quotient graph:
    - `source`: the source vertex of an edge
    - `target`: the target vertex of an edge
      - positive vertices are elements
      - negative vertices are supernodes
    - `begptr`: the first edge incident to a vertex
    - `endptr`: the final edge incident to a vertex
    - `invptr`: the reverse of an edge
 - vertex-degree bucket queue:
    - `head`: the first vertex in a bucket
    - `prev`: the predecessor of a vertex
    - `next`: the successor of a vertex
"""
function mmw_impl!(
        marker::AbstractVector{V},
        vstack::AbstractVector{V},
        tmpptr::AbstractVector{E},
        degree::AbstractVector{V},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        head::AbstractVector{V},
        prev::AbstractVector{V},
        next::AbstractVector{V},
        totdeg::V,
        weights::AbstractVector{V},
        graph::AbstractGraph{V},
        strategy::Val,
    ) where {V <: Signed, E}
    
    @argcheck nv(graph) <= length(marker)
    @argcheck nv(graph) <= length(degree)
    @argcheck nv(graph) <= length(vstack)
    @argcheck ne(graph) <= length(source)
    @argcheck ne(graph) <= length(target)
    @argcheck nv(graph) <= length(tmpptr)
    @argcheck nv(graph) <  length(begptr)
    @argcheck nv(graph) <= length(endptr)
    @argcheck ne(graph) <= length(invptr)
    @argcheck totdeg    <= length(head)
    @argcheck nv(graph) <= length(prev)
    @argcheck nv(graph) <= length(next)

    # `set(i)` constructs the bucket for weighted degree `i`
    function set(i::V)
        @inbounds h = view(head, i)
        return DoublyLinkedList(h, prev, next)
    end

    # `n` is the number of vertices in the graph    
    n = nv(graph)

    # `mindeg` is the minimum weighted degree
    # `maxdeg` is the maximum weighted degree
    mindeg, maxdeg = mmw_init!(marker, tmpptr, degree, source, target,
        begptr, endptr, invptr, head, prev, next, totdeg, weights, graph)

    # `maxmindeg` is the largest value of `mindeg`
    # encountered during the algorithm
    maxmindeg = zero(V)

    @inbounds for tag in oneto(n)
        # find the new minimum degree        
        while isempty(set(mindeg))
            mindeg += one(V)
        end

        # find the new maximum degree
        while isempty(set(maxdeg))
            maxdeg -= one(V)
        end

        # update `maxmindeg`
        maxmindeg = max(maxmindeg, mindeg)

        # select a vertex of minimum degree
        v = popfirst!(set(mindeg))

        # find a neighbor according to strategy `S`
        w = mmw_search!(marker, vstack, degree, target,
            begptr, endptr, invptr, weights, tag, v, n, strategy)

        # if a neighbor was found, contract the edge
        # {`v`, `w`}, turning `v` into a supernode
        if ispositive(w)
            mmw_update_1!(marker, vstack, degree, source, target,
                begptr, endptr, invptr, head, prev, next, weights, tag, v, w)

            # the weighted degree of `w` may have increased
            maxdeg = max(maxdeg, degree[w])

        # otherwise, remove `v` from the graph
        else
            mmw_update_2!(vstack, degree, source, target,
                begptr, endptr, invptr, head, prev, next, weights, v)
        end

        # the weighted degree of every element reachable by
        # `v` has decreased by the weight of `v`
        mindeg = max(mindeg - weights[v], one(V))
    end

    return maxmindeg
end

"""
    mmw_init!(marker, tmpptr, degree, source, target, begptr,
        endptr, invptr, head, prev, next, totdeg, weights, graph)

Initialize quotient graph and degree bucket queue.

input parameters:
  - `weights`: vertex weights
  - `graph`: input graph
  - `totdeg`: total vertex weight

output parameters:
 - `mindeg`: minimum weighted degree
 - `maxdeg`: maximum weighted degree
"""
function mmw_init!(
        marker::AbstractVector{V},
        tmpptr::AbstractVector{E},
        degree::AbstractVector{V},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        head::AbstractVector{V},
        prev::AbstractVector{V},
        next::AbstractVector{V},
        totdeg::V,
        weights::AbstractVector{V},
        graph::AbstractGraph{V},
    ) where {V, E}
 
    n = nv(graph); nn = n + one(V)

    # `set(i)` constructs the bucket for weighted degree `i`    
    function set(i::V)
        @inbounds h = view(head, i)
        return DoublyLinkedList(h, prev, next)
    end
   
    # empty the bucket queue
    @inbounds for v in oneto(totdeg)
        head[v] = zero(V)
    end
 
    # `mindeg` is the minimum weighted degree
    # `maxdeg` is the maximum weighted degree
    mindeg = totdeg; maxdeg = zero(V)

    # `p` is the current arc
    p = one(E)

    @inbounds for v in vertices(graph)
        tmpptr[v] = begptr[v] = endptr[v] = p
        marker[v] = zero(V)

        # `deg` is the weighted degree of `v`
        deg = weights[v]
        
        for w in neighbors(graph, v)
            if v != w
                # `p` is the arc (`v`, `w`)
                source[p] = v; p += one(E)
                deg += weights[w]
            end
        end
        
        mindeg = min(mindeg, deg)
        maxdeg = max(maxdeg, deg)
        degree[v] = deg; pushfirst!(set(deg), v)
    end
    
    @inbounds for v in vertices(graph), w in neighbors(graph, v)
        if v != w
            # `q` is the arc (`w`, `v`)
            q = endptr[w]; target[q] = v; endptr[w] = q + one(E)
        end
    end
    
    @inbounds for v in vertices(graph)
        # the arcs {`p`, ..., `pend` - 1} are incident
        # to `v`
        p = begptr[v]; pend = endptr[v]
        
        while p < pend
            # `p` is the arc (`v`, `w`)
            w = target[p]

            # `q` is the arc (`w`, `v`)
            q = tmpptr[w]; invptr[p] = q; tmpptr[w] = q + one(E)
            p += one(E)
        end
    end

    if ispositive(n)
        @inbounds begptr[nn] = endptr[n]
    end
    
    return mindeg, maxdeg
end

"""
    mmw_search!(marker, vstack, degree, target,
        begptr, endptr, invptr, weights, tag, v, n, strategy)

Find an element reachable by `v` using the min-d or
max-d heuristics. The min-d heuristic selects an element
with the least weighted degree. The max-d heuristic
selects an element with the greatest weighted degree.

input parameters:
 - `tag`: tag for marking vertices
 - `v`: minimum degree vertex
 - `n`: number of vertices
 - `S`: strategy
   - `1`: min-d
   - `2`: max-d

output parameters:
 - `w`: chosen element
"""
function mmw_search!(
        marker::AbstractVector{V},
        vstack::AbstractVector{V},
        degree::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        weights::AbstractVector{V},
        tag::V,
        v::V,
        n::V,
        strategy::Val{S},
    ) where {V, E, S}
   
    # `wgt` is the weight of `v` 
    @inbounds wgt = weights[v]

    # `w` is the chosen element
    w = deg = zero(V)

    w, deg = mmw_reach!((w, deg), vstack,
            target, begptr, endptr, invptr, v) do (w, deg), pp, ww
        # `ww` is reachable by `v`; `wwgt` is its weight
        @inbounds wwgt = weights[ww]

        if wwgt <= wgt
            # `ddeg` is the weighted degree of `ww`
            @inbounds ddeg = degree[ww]

            if iszero(w) || (isone(S) && ddeg < deg) || (istwo(S) && ddeg > deg)
                w, deg = ww, ddeg
            end
        end

        return (w, deg)
    end

    return w
end

"""
    mmw_search!(marker, vstack, degree, target,
        begptr, endptr, invptr, weights, tag, v, n, strategy)

Find an element reachable by `v` using the least-c
heuristic. The least-c heuristic selects a neighbor
`w` that minimizes the sum
   Σ weights(x)
 x ∈ N(v) ∩ N(w)

input parameters:
 - `tag`: tag for marking vertices
 - `v`: minimum degree vertex
 - `n`: number of vertices

output parameters:
 - `w`: chosen element
"""
function mmw_search!(
        marker::AbstractVector{V},
        vstack::AbstractVector{V},
        degree::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        weights::AbstractVector{V},
        tag::V,
        v::V,
        n::V,
        strategy::Val{3},
    ) where {V, E}

    # mark the elements reachable by `v` and write them
    # to `vstack`
    numend = n + one(V)

    numend = mmw_reach!(numend, vstack, 
            target, begptr, endptr, invptr, v) do numend, p, w
        # `w` is reachable by `v`; mark `w` and push it
        # to the stack
        @inbounds numend -= one(V); vstack[numend] = w
        @inbounds marker[w] = -tag
        return numend
    end
   
    # `w` is the chosen neighbor of `v`
    w = zero(V)

    # `scr` is the score of `w`
    scr = zero(V)

    # `wgt` is the weight of `v`
    @inbounds wgt = weights[v]

    # elements reachable by `v` are located at the
    # indices (`numend`, ..., `n`) in `vstack`
    @inbounds while numend <= n
        # `ww` is reachable by `v`
        ww = vstack[numend]; numend += one(V)

        # if the weight of `ww` is no greater than
        # the weight of `v`, compute its score
        if weights[ww] <= wgt
            # `sscr` is the score of `ww`
            sscr = zero(V)

            sscr = mmw_reach!(sscr, vstack,
                    target, begptr, endptr, invptr, ww) do scr, p, w
                # `ttag` indicates whether `w` is reachable by `v`
                #  - `ttag` = `-tag`: reachable
                #  - `ttag` ≠ `-tag`: not reachable
                @inbounds ttag = marker[w]

                # if `w` is reachable by `v`, increase
                # the score of `ww` by the weight of `w`
                if ttag == -tag
                    @inbounds scr += weights[w]
                end

                return scr
            end

            # store the element `w` with the smallest
            # score
            if iszero(w) || sscr < scr
                w, scr = ww, sscr
            end
        end
    end

    return w
end

"""
    mmw_update_1!(marker, vstack, degree, source, begptr,
        endptr, invptr, head, prev, next, weights, tag, v, w)

Contract the edge {`v`, `w`}, turning the
element `v` into a supernode.

input parameters:
 - `tag`: tag for marking vertices
 - `v`: minimum degree vertex
 - `w`: neighbor of `v`
"""
function mmw_update_1!(
        marker::AbstractVector{V},
        vstack::AbstractVector{V},
        degree::AbstractVector{V},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        head::AbstractVector{V},
        prev::AbstractVector{V},
        next::AbstractVector{V},
        weights::AbstractVector{V},
        tag::V,
        v::V,
        w::V,
    ) where {V, E}
    
    # `set(i)` constructs the bucket for weighted degree `i`
    function set(i::V)
        @inbounds h = view(head, i)
        return DoublyLinkedList(h, prev, next)
    end

    # mark `w`
    @inbounds marker[w] = tag

    # mark the elements reachable by `w`
    mmw_reach!(nothing, vstack,
            target, begptr, endptr, invptr, w) do _, pp, ww
        # `ww` is reachable by `w`; mark it
        @inbounds marker[ww] = tag
        return
    end

    # `wgt` is the weight of `v`
    @inbounds wgt = weights[v]

    # `del` is the difference between the weight
    # of `v` and the weight of `w`
    @inbounds del = wgt - weights[w]

    # `deg` is the weighted degree of `w`
    @inbounds deg = degree[w]

    # push `v` to the stack
    @inbounds num = one(V); vstack[num] = v

    @inbounds while ispositive(num)
        # `vv` is a supernode adjacent to `v`
        vv = vstack[num]; num -= one(V)

        # the arcs {`pp`, ..., `ppend` - 1} are incident
        # to `vv`
        pp = begptr[vv]; ppend = endptr[vv]

        while pp < ppend
            # `ww` is adjacent to `vv` and reachable by `v`
            ww = target[pp]

            # if `ww` is an element, update the graph
            if ispositive(ww)
                # `qq` is the reverse of `pp`
                qq = invptr[pp]

                # if `ww` is not equal to or reachable by `w`, add
                # it to the neighborhood of `ww`
                if marker[ww] < tag
                    target[qq] = w; pp += one(E)

                    # increase the weighted degree of `w` by the
                    # weight of `ww`
                    deg += weights[ww]

                    # increase the weighted degree of `ww` by the
                    # weight of `w` and decrease it by the weight
                    # of `v`
                    if ispositive(del)
                        delete!(set(degree[ww]), ww)
                        degree[ww] -= del; pushfirst!(set(degree[ww]), ww)
                    end

                # otherwise, `ww` is either equal to or reachable by `w`
                else
                    # replace `ww` with a vertex `xx` in the neighborhood
                    # of `vv`
                    ppend -= one(E)

                    if pp < ppend
                        # `xx` is adjacent to `vv` and reachable by `v`
                        xx = target[pp] = target[ppend]

                        if ispositive(xx)
                            # `xx` is an element; `ppinv` is the arc
                            # (`xx`, `v`)
                            ppinv = invptr[pp] = invptr[ppend]
                            invptr[ppinv] = pp
                        end
                    end

                    # if `ww` is equal to `w`, replace `v` with `-v` in
                    # the neighborhood of `ww`
                    if w == ww
                        target[qq] = -v

                        # decrease the weighted degree of `w` by the
                        # weight of `v`
                        deg -= wgt

                    # otherwise, replace `v` with a vertex `xx` in the
                    # neighborhood of `ww`
                    else
                        qqend = endptr[source[qq]] -= one(E)

                        if qq < qqend
                            # `xx` is adjacent to `ww` and reachable by `v`
                            xx = target[qq] = target[qqend]

                            if ispositive(xx)
                                # `xx` is an element; `qqinv` is the arc
                                # (`xx`, `v`)
                                qqinv = invptr[qq] = invptr[qqend]
                                invptr[qqinv] = qq
                            end
                        end

                        # increase the weighted degree of `ww` by
                        # the weight of `v`
                        delete!(set(degree[ww]), ww)
                        degree[ww] -= wgt; pushfirst!(set(degree[ww]), ww)
                    end

                end

            # otherwise, `ww` is a supernode
            else
                # push `ww` to the stack
                ww = -ww
                num += one(V); vstack[num] = ww                           
                pp += one(E)
            end
        end

        endptr[vv] = ppend
    end

    # update the weighted degree of `w`
    @inbounds delete!(set(degree[w]), w)
    @inbounds degree[w] = deg; pushfirst!(set(deg), w)
    return
end

"""
    mmw_update_2!(vstack, degree, source, target,
        begptr, endptr, invptr, head, prev, next, weights, v)

Remove the vertex `v` from the graph.

input parameters:
 - `v`: minimum degree vertex
"""
function mmw_update_2!(
        vstack::AbstractVector{V},
        degree::AbstractVector{V},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        head::AbstractVector{V},
        prev::AbstractVector{V},
        next::AbstractVector{V},
        weights::AbstractVector{V},
        v::V,
    ) where {V, E}
   
    # `wgt` is the weight of `v` 
    @inbounds wgt = weights[v]

    # `set(i)` constructs the bucket for weighted degree `i`
    function set(i::V)
        @inbounds h = view(head, i)
        return DoublyLinkedList(h, prev, next)
    end

    # remove `v` from the graph
    mmw_reach!(nothing, vstack,
            target, begptr, endptr, invptr, v) do _, p, w
        # `w` is reachable by `v` through a path
        #    (`v`, ..., `vv`, `w`),
        # and `p` is the arc (`vv`, `w`)
        #
        # `v` is reachable by `w` through a path
        #    (`w`, ..., `ww`, `v`),
        # and `q` is the arc (`ww`, `v`)
        @inbounds q = invptr[p]; ww = source[q]

        # `qend` = (`ww`, `x`) is the last arc incident
        # to `ww`
        #
        # if `w` is not equal to `x`, replace `w` with `x`
        # in the neighborhood of `ww`
        @inbounds qend = endptr[ww] -= one(E)
        if q < qend
            @inbounds x = target[q] = target[qend]

            if ispositive(x)
                # `x` is an element; `qinv` is the arc
                # (`x`, `w`)
                @inbounds qinv = invptr[q] = invptr[qend]
                @inbounds invptr[qinv] = q
            end
        end

        # decrease the weighted degree of `w` by the
        # weight of `v`
        @inbounds delete!(set(degree[w]), w)
        @inbounds degree[w] -= wgt; pushfirst!(set(degree[w]), w)
        return
    end
     
    return
end

"""
    mmw_reach!(combine, result, vstack,
        target, begptr, endptr, invptr, v)

Compute the linear fold
  combine(combine(combine(combine(result, w), x), y), ...)
where {w, x, y, ...} are the neighbors of `v`.

input parameters:
 - `v`: an element

output parameters:
 - `result`: folded value
"""
function mmw_reach!(
        combine::Function,
        result,
        vstack::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        v::V,
    ) where {V, E}
    # push `v` to the stack
    @inbounds num = one(V); vstack[num] = v

    @inbounds while ispositive(num)
        # `vv` is a supernode adjacent to `v`
        vv = vstack[num]; num -= one(V)

        # the arcs {`pp`, ..., `ppend` - 1} are incident
        # to `vv`
        pp = begptr[vv]; ppend = endptr[vv]

        while pp < ppend
            # `ww` is adjacent to `vv` and reachable by `v`
            ww = target[pp]

            # if `ww` is an element, fold it into `result`
            if ispositive(ww)
                result = combine(result, pp, ww)
                pp += one(E)

            # otherwise, `ww` is a supernode
            else
                # `ppnxt` is the largest possible value of `ppend`
                ppnxt = begptr[vv + one(V)]

                # the arcs {`qq`, ..., `qqend` - 1} are incident
                # to `ww`
                ww = -ww; qq = begptr[ww]; qqend = endptr[ww]

                # while there is space, move neighbors of `ww` 
                # into the neighborhood of `vv`
                while ppend < ppnxt && qq < qqend
                    qqend -= one(E)

                    # `xx` is adjacent to `ww` and reachable by `v`
                    xx = target[ppend] = target[qqend]

                    if ispositive(xx)
                        # `xx` is an element; `ppinv` is the arc
                        # (`xx`, `v`)
                        ppinv = invptr[ppend] = invptr[qqend]
                        invptr[ppinv] = ppend
                    end

                    ppend += one(E)
                end

                endptr[ww] = qqend

                # if `ww` has more than two neighbors, push
                # it to the stack
                if qq < qqend - one(E)
                    num += one(V); vstack[num] = ww                           
                    pp += one(E)

                # if `ww` has only one neighbor `xx`, replace
                # `ww` with `xx` in the neighborhood of `vv` 
                elseif qq == qqend - one(E)
                    xx = target[pp] = target[qq]

                    if ispositive(xx)
                        # `xx` is an element; `ppinv` is the arc
                        # (`xx`, `v`)
                        ppinv = invptr[pp] = invptr[qq]
                        invptr[ppinv] = pp
                    end

                # if `ww` has no neighbors, replace it with a vertex
                # `xx` in the neighborhood of `vv`
                else
                    ppend -= one(E)

                    if pp < ppend
                        # `xx` is adjacent to `vv` and reachable by `v`
                        xx = target[pp] = target[ppend]

                        if ispositive(xx)
                            # `xx` is an element; `ppinv` is the arc
                            # (`xx`, `v`)
                            ppinv = invptr[pp] = invptr[ppend]
                            invptr[ppinv] = pp
                        end
                    end
                end
            end
        end

        endptr[vv] = ppend
    end
    
    return result
end

function Base.convert(::Type{MMW{S}}, alg::MMW) where {S}
    return MMW{S}()
end

function Base.show(io::IO, ::MIME"text/plain", alg::MMW{S}) where {S}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "MMW{$S}")
    return nothing
end

"""
    DEFAULT_LOWER_BOUND_ALGORITHM = MMW()
"""
const DEFAULT_LOWER_BOUND_ALGORITHM = MMW()
