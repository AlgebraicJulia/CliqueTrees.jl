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

function lowerbound(graph, alg::WidthOrAlgorithm)
    return lowerbound(BipartiteGraph(graph), alg)
end

function lowerbound(graph::AbstractGraph{V}, width::Integer) where {V}
    return convert(V, width)
end

function lowerbound(graph::AbstractGraph{V}, alg::LowerBoundAlgorithm) where {V}
    n = nv(graph); weights = Ones{V}(n)
    return lowerbound(weights, graph, alg) - one(V)
end

function lowerbound(weights::AbstractVector, graph; alg::WidthOrAlgorithm = DEFAULT_LOWER_BOUND_ALGORITHM)
    return lowerbound(weights, graph, alg)
end

function lowerbound(weights::AbstractVector, graph, alg::WidthOrAlgorithm)
    return lowerbound(weights, BipartiteGraph(graph), alg)
end

function lowerbound(weights::AbstractVector{W}, graph::AbstractGraph, width::Number) where {W}
    return convert(W, width)
end

function lowerbound(weights::AbstractVector, graph::AbstractGraph, ::MMW{S}) where {S}
    return mmw(weights, graph, Val(S))
end

function mmw(weights::AbstractVector{W}, graph::AbstractGraph{V}, strategy::Val) where {W, V}
    @assert nv(graph) <= length(weights)

    n = nv(graph); m = de(graph)

    # `totdeg` is the total weight of the
    # vertices in the graph
    totdeg = 0

    @inbounds for v in oneto(n)
        totdeg += trunc(Int, weights[v])
    end

    # the working arrays use 32-bit integers
    # when they suffice
    if n < typemax(Int32) ÷ 4 && mmwcapacity(n, m) + 4n < typemax(Int32) && totdeg < typemax(Int32)
        width = mmw(Int32, Int32, totdeg, weights, graph, strategy)
    else
        width = mmw(promote_type(V, Int), promote_type(etype(graph), Int), totdeg, weights, graph, strategy)
    end

    return convert(W, width)
end

function mmw(::Type{V}, ::Type{E}, totdeg::Integer, weights::AbstractVector, graph::AbstractGraph, strategy::Val) where {V, E}
    n = convert(V, nv(graph)); m = convert(E, de(graph)); nn = n + one(V)
    cap = convert(E, mmwcapacity(n, m)); totdeg = convert(V, totdeg)
    weight = mmwweights(V, weights, n)

    marker = FVector{V}(undef, n)
    hubmark = FVector{V}(undef, n)
    stash = FVector{V}(undef, n)

    degree = FVector{V}(undef, n)
    target = FVector{V}(undef, cap)
    invptr = FVector{E}(undef, cap)
    begptr = FVector{E}(undef, nn)
    endptr = FVector{E}(undef, n)
    limptr = FVector{E}(undef, n)

    head = FVector{V}(undef, totdeg + one(V))
    prev = FVector{V}(undef, n)
    next = FVector{V}(undef, n)

    width = mmw_impl!(marker, hubmark, stash, degree, target, invptr,
        begptr, endptr, limptr, head, prev, next, cap, totdeg, weight,
        graph, strategy)

    return width
end

# the size of the arc storage: the arcs of the graph, the
# room left at the end of every list, and room for lists to
# move to between compactions
function mmwcapacity(n::Integer, m::Integer)
    return m + mmwslack(m) + max(m ÷ 2, 4n) + 1
end

# the room left at the end of a list of length `len`
function mmwslack(len::I) where {I <: Integer}
    return len ÷ two(I)
end

function mmwweights(::Type{V}, weights::Ones, n::V) where {V}
    return Ones{V}(n)
end

function mmwweights(::Type{V}, weights::AbstractVector, n::V) where {V}
    weight = FVector{V}(undef, n)

    @inbounds for v in oneto(n)
        weight[v] = trunc(V, weights[v])
    end

    return weight
end

"""
  mmw_impl!(marker, hubmark, stash, degree, target, invptr,
    begptr, endptr, limptr, head, prev, next, cap, totdeg,
    weight, graph, strategy)

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

The graph is stored as a set of adjacency lists. The
neighbors of a vertex `v` are stored in the list

    target[begptr[v]:endptr[v] - 1],

which can grow until `endptr[v] = limptr[v]`. Every arc
`p` = (`v`, `w`) has a reverse arc `invptr[p]` = (`w`, `v`).
When an edge {`v`, `w`} is contracted, the new neighbors
of `w` are appended to its list. If the list is full, it
is moved to the end of the storage and its capacity is
doubled. If the storage is full, it is compacted.

The algorithm stops as soon as the total weight of the
remaining graph is no greater than `maxmindeg`. The
weighted degree of a vertex never exceeds the total weight
of its graph, and the total weight of a minor never exceeds
the total weight of the graph, so no later minor can
increase `maxmindeg`.

input parameters:
  - `cap`: size of the arc storage
  - `totdeg`: total vertex weight
  - `weight`: vertex weight
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
    - `hubmark`: marks the neighbors of a vertex `hub`
    - `stash`: temporary vertex array
  - adjacency lists:
    - `target`: the target vertex of an arc
    - `invptr`: the reverse of an arc
    - `begptr`: the first arc incident to a vertex
    - `endptr`: one past the last arc incident to a vertex
    - `limptr`: one past the last arc that can be incident
      to a vertex
 - vertex-degree bucket queue:
    - `head`: the first vertex in a bucket; the bucket for
      weighted degree `i` is headed by `head[i + 1]`, since
      vertex weights may be zero
    - `prev`: the predecessor of a vertex
    - `next`: the successor of a vertex
"""
function mmw_impl!(
        marker::AbstractVector{V},
        hubmark::AbstractVector{V},
        stash::AbstractVector{V},
        degree::AbstractVector{V},
        target::AbstractVector{V},
        invptr::AbstractVector{E},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        limptr::AbstractVector{E},
        head::AbstractVector{V},
        prev::AbstractVector{V},
        next::AbstractVector{V},
        cap::E,
        totdeg::V,
        weight::AbstractVector{V},
        graph::AbstractGraph,
        strategy::Val,
    ) where {V <: Signed, E}

    @assert nv(graph) <= length(marker)
    @assert nv(graph) <= length(hubmark)
    @assert nv(graph) <= length(stash)
    @assert nv(graph) <= length(degree)
    @assert de(graph) <  cap
    @assert cap       <= length(target)
    @assert cap       <= length(invptr)
    @assert nv(graph) <  length(begptr)
    @assert nv(graph) <= length(endptr)
    @assert nv(graph) <= length(limptr)
    @assert totdeg    <  length(head)
    @assert nv(graph) <= length(prev)
    @assert nv(graph) <= length(next)

    # `set(i)` constructs the bucket for weighted degree `i`
    function set(i::V)
        @inbounds h = view(head, i + one(V))
        return DoublyLinkedList(h, prev, next)
    end

    # `n` is the number of vertices in the graph
    n = convert(V, nv(graph))

    # `mindeg` is the minimum weighted degree
    # `pfree` is the first free arc in the storage
    mindeg, pfree = mmw_init!(marker, hubmark, degree, target, invptr,
        begptr, endptr, limptr, head, prev, next, totdeg, weight, graph)

    # the neighbors `x` of `hub` satisfy
    #    hubmark[x] = `epoch`
    hub = epoch = zero(V)

    # `maxmindeg` is the largest value of `mindeg`
    # encountered during the algorithm
    maxmindeg = zero(V)

    # `remdeg` is the total weight of the remaining graph
    remdeg = totdeg

    @inbounds for tag in oneto(n)
        # find the new minimum degree
        while isempty(set(mindeg))
            mindeg += one(V)
        end

        # update `maxmindeg`
        maxmindeg = max(maxmindeg, mindeg)

        # if the remaining graph is too light to
        # increase `maxmindeg`, stop
        if remdeg <= maxmindeg
            break
        end

        # select a vertex of minimum degree
        v = popfirst!(set(mindeg))
        remdeg -= weight[v]

        # find a neighbor according to strategy `S`
        w = mmw_search!(marker, hubmark, degree, target, begptr,
            endptr, weight, hub, epoch, tag, v, strategy)

        # if a neighbor was found, contract the edge
        # {`v`, `w`}
        if ispositive(w)
            pfree, hub, epoch = mmw_contract!(hubmark, stash, degree,
                target, invptr, begptr, endptr, limptr, head, prev,
                next, weight, pfree, cap, hub, epoch, n, v, w)

        # otherwise, remove `v` from the graph
        else
            mmw_remove!(degree, target, invptr, begptr,
                endptr, head, prev, next, weight, v)
        end

        # the weighted degree of every neighbor of `v`
        # has decreased by at most the weight of `v`
        mindeg = max(mindeg - weight[v], zero(V))
    end

    return maxmindeg
end

"""
    mmw_init!(marker, hubmark, degree, target, invptr, begptr,
        endptr, limptr, head, prev, next, totdeg, weight, graph)

Initialize adjacency lists and degree bucket queue.

input parameters:
  - `weight`: vertex weight
  - `graph`: input graph
  - `totdeg`: total vertex weight

output parameters:
 - `mindeg`: minimum weighted degree
 - `pfree`: first free arc
"""
function mmw_init!(
        marker::AbstractVector{V},
        hubmark::AbstractVector{V},
        degree::AbstractVector{V},
        target::AbstractVector{V},
        invptr::AbstractVector{E},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        limptr::AbstractVector{E},
        head::AbstractVector{V},
        prev::AbstractVector{V},
        next::AbstractVector{V},
        totdeg::V,
        weight::AbstractVector{V},
        graph::AbstractGraph,
    ) where {V, E}

    n = convert(V, nv(graph)); nn = n + one(V)

    # `set(i)` constructs the bucket for weighted degree `i`
    function set(i::V)
        @inbounds h = view(head, i + one(V))
        return DoublyLinkedList(h, prev, next)
    end

    # empty the bucket queue
    @inbounds for i in oneto(totdeg + one(V))
        head[i] = zero(V)
    end

    # `mindeg` is the minimum weighted degree
    mindeg = totdeg

    # `p` is the current arc
    p = one(E)

    # `limptr` is used as a temporary pointer array
    @inbounds for v in oneto(n)
        limptr[v] = begptr[v] = endptr[v] = p
        marker[v] = hubmark[v] = zero(V)

        # `deg` is the weighted degree of `v`
        deg = weight[v]

        for w in neighbors(graph, v)
            if v != w
                p += one(E)
                deg += weight[w]
            end
        end

        mindeg = min(mindeg, deg)
        degree[v] = deg; pushfirst!(set(deg), v)

        # leave room for the list to grow
        p += mmwslack(p - begptr[v])
    end

    @inbounds begptr[nn] = p

    @inbounds for v in oneto(n), w in neighbors(graph, v)
        if v != w
            # `q` is the arc (`w`, `v`)
            q = endptr[w]; target[q] = v; endptr[w] = q + one(E)
        end
    end

    @inbounds for v in oneto(n)
        for p in begptr[v]:endptr[v] - one(E)
            # `p` is the arc (`v`, `w`)
            w = target[p]

            # `q` is the arc (`w`, `v`)
            q = limptr[w]; invptr[p] = q; limptr[w] = q + one(E)
        end
    end

    # clear the room at the end of each list, so that
    # it is never mistaken for a list head during a
    # compaction
    @inbounds for v in oneto(n)
        limptr[v] = begptr[v + one(V)]

        for p in endptr[v]:limptr[v] - one(E)
            target[p] = zero(V)
        end
    end

    # on output, `p` is the first free arc
    return mindeg, p
end

"""
    mmw_search!(marker, hubmark, degree, target, begptr,
        endptr, weight, hub, epoch, tag, v, strategy)

Find a neighbor of `v` using the min-d or max-d
heuristics. The min-d heuristic selects a neighbor
with the least weighted degree. The max-d heuristic
selects a neighbor with the greatest weighted degree.
Only neighbors no heavier than `v` are considered.

input parameters:
 - `tag`: tag for marking vertices
 - `v`: minimum degree vertex
 - `S`: strategy
   - `1`: min-d
   - `2`: max-d

output parameters:
 - `w`: chosen neighbor, or zero if there is none
"""
function mmw_search!(
        marker::AbstractVector{V},
        hubmark::AbstractVector{V},
        degree::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        weight::AbstractVector{V},
        hub::V,
        epoch::V,
        tag::V,
        v::V,
        strategy::Val{S},
    ) where {V, E, S}

    # `wgt` is the weight of `v`
    @inbounds wgt = weight[v]

    # `w` is the chosen neighbor
    w = deg = zero(V)

    @inbounds for p in begptr[v]:endptr[v] - one(E)
        # `ww` is a neighbor of `v`; `wwgt` is its weight
        ww = target[p]; wwgt = weight[ww]

        if wwgt <= wgt
            # `ddeg` is the weighted degree of `ww`
            ddeg = degree[ww]

            if iszero(w) || (isone(S) && ddeg < deg) || (istwo(S) && ddeg > deg)
                w, deg = ww, ddeg
            end
        end
    end

    return w
end

"""
    mmw_search!(marker, hubmark, degree, target, begptr,
        endptr, weight, hub, epoch, tag, v, strategy)

Find a neighbor of `v` using the least-c heuristic.
The least-c heuristic selects a neighbor `w` that
minimizes the score

   Σ weight(x)
 x ∈ N(v) ∩ N(w)

Only neighbors no heavier than `v` are considered.
Ties are broken in favor of the first neighbor
examined.

The score of a candidate is computed by scanning its
neighborhood. The scan is abandoned as soon as the
partial score reaches the score of the best candidate
so far, since the candidate can no longer be chosen.
If the candidate is `hub`, whose neighbors are marked
in `hubmark`, and its neighborhood is larger than the
neighborhood of `v`, then the score is computed by
scanning the neighborhood of `v` instead.

input parameters:
 - `hub`: the neighbors `x` of `hub` satisfy
     hubmark[x] = `epoch`
 - `epoch`: tag for marking vertices
 - `tag`: tag for marking vertices
 - `v`: minimum degree vertex

output parameters:
 - `w`: chosen neighbor, or zero if there is none
"""
function mmw_search!(
        marker::AbstractVector{V},
        hubmark::AbstractVector{V},
        degree::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        weight::AbstractVector{V},
        hub::V,
        epoch::V,
        tag::V,
        v::V,
        strategy::Val{3},
    ) where {V, E}

    # the arcs {`pbeg`, ..., `pend` - 1} are
    # incident to `v`
    @inbounds pbeg = begptr[v]; pend = endptr[v]

    # mark the neighbors of `v`
    @inbounds for p in pbeg:pend - one(E)
        marker[target[p]] = tag
    end

    # `w` is the chosen neighbor of `v`
    w = zero(V)

    # `scr` is the score of `w`
    scr = typemax(V)

    # `wgt` is the weight of `v`
    @inbounds wgt = weight[v]

    # `p` is the current arc
    p = pbeg

    # if `scr` is zero, then no candidate can
    # improve on `w`
    @inbounds while p < pend && ispositive(scr)
        # `ww` is a neighbor of `v`
        ww = target[p]; p += one(E)

        # if the weight of `ww` is no greater than the
        # weight of `v`, compute its score
        if weight[ww] <= wgt
            # `sscr` is the score of `ww`, unless it
            # is no less than `scr`
            if ww == hub && pend - pbeg < endptr[ww] - begptr[ww]
                sscr = mmw_score(hubmark, target, pbeg,
                    pend, weight, epoch, scr)
            else
                sscr = mmw_score(marker, target, begptr[ww],
                    endptr[ww], weight, tag, scr)
            end

            # if the scan was completed, `ww` is the best
            # candidate so far
            if sscr < scr
                w, scr = ww, sscr
            end
        end
    end

    return w
end

"""
    mmw_score(marker, target, pbeg, pend,
        weight, tag, maxscr)

Compute the sum

       Σ weight(x)
 x ∈ target[pbeg:pend - 1]
   marker[x] = tag

which is the score of a candidate `w` if the arcs
{`pbeg`, ..., `pend` - 1} are incident to `w` and
the neighbors of `v` are marked with `tag`. The
computation is abandoned as soon as the sum reaches
`maxscr`.

input parameters:
 - `pbeg`, `pend`: the arcs {`pbeg`, ..., `pend` - 1}
 - `tag`: tag for marking vertices
 - `maxscr`: maximum score

output parameters:
 - `scr`: the sum, or a number no less than `maxscr`
"""
function mmw_score(
        marker::AbstractVector{V},
        target::AbstractVector{V},
        pbeg::E,
        pend::E,
        weight::AbstractVector{V},
        tag::V,
        maxscr::V,
    ) where {V, E}
    scr = zero(V)

    # the indices are converted to `Int`, so that
    # consecutive arcs have consecutive addresses
    p = Int(pbeg); pend = Int(pend)

    # scan the arcs in blocks of `MMW_BLOCK`; within
    # a block, the loop has no exits, so that it can
    # be vectorized
    @inbounds while p + MMW_BLOCK <= pend
        for k in 0:MMW_BLOCK - 1
            # `x` is the target of an arc
            x = target[p + k]
            scr += ifelse(marker[x] == tag, weight[x], zero(V))
        end

        # if the sum has reached `maxscr`, stop
        if scr >= maxscr
            return scr
        end

        p += MMW_BLOCK
    end

    @inbounds while p < pend
        # `x` is the target of an arc
        x = target[p]
        scr += ifelse(marker[x] == tag, weight[x], zero(V))

        # if the sum has reached `maxscr`, stop
        if scr >= maxscr
            return scr
        end

        p += 1
    end

    return scr
end

# the block size in `mmw_score`
const MMW_BLOCK = 16

"""
    mmw_contract!(hubmark, stash, degree, target, invptr,
        begptr, endptr, limptr, head, prev, next, weight,
        pfree, cap, hub, epoch, n, v, w)

Contract the edge {`v`, `w`}, merging the vertex `v`
into the vertex `w`. The neighbors of `v` that are not
neighbors of `w` are appended to the list of `w`.

The neighbors of `w` are found by marking them in
`hubmark`. The vertex `w` becomes `hub`, and its
neighbors remain marked until another vertex becomes
`hub`. Since an edge incident to `hub` disappears only
when one of its endpoints is eliminated, the marks need
only be updated when `hub` gains a neighbor. Hence, if
several edges incident to the same vertex are contracted
in a row, which is typical of the heuristic max-d, then
its neighbors are marked only once.

input parameters:
 - `cap`: size of the arc storage
 - `n`: number of vertices
 - `v`: minimum degree vertex
 - `w`: neighbor of `v`

updated parameters:
 - `pfree`: first free arc
 - `hub`: the neighbors `x` of `hub` satisfy
     hubmark[x] = `epoch`
 - `epoch`: tag for marking vertices
"""
function mmw_contract!(
        hubmark::AbstractVector{V},
        stash::AbstractVector{V},
        degree::AbstractVector{V},
        target::AbstractVector{V},
        invptr::AbstractVector{E},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        limptr::AbstractVector{E},
        head::AbstractVector{V},
        prev::AbstractVector{V},
        next::AbstractVector{V},
        weight::AbstractVector{V},
        pfree::E,
        cap::E,
        hub::V,
        epoch::V,
        n::V,
        v::V,
        w::V,
    ) where {V, E}

    # `set(i)` constructs the bucket for weighted degree `i`
    function set(i::V)
        @inbounds h = view(head, i + one(V))
        return DoublyLinkedList(h, prev, next)
    end

    # if `w` is not `hub`, make it `hub` and
    # mark its neighbors
    if w != hub
        hub = w; epoch += one(V)

        @inbounds for p in begptr[w]:endptr[w] - one(E)
            hubmark[target[p]] = epoch
        end
    end

    # the arcs {`pbeg`, ..., `pend` - 1} are
    # incident to `v`
    @inbounds pbeg = begptr[v]; pend = endptr[v]

    # make room for the new neighbors of `w`; there
    # are fewer of them than there are neighbors
    # of `v`, since `w` is one
    @inbounds if limptr[w] < endptr[w] + (pend - pbeg) - one(E)
        pfree = mmw_relocate!(stash, target, invptr, begptr, endptr,
            limptr, pfree, cap, n, w, (pend - pbeg) - one(E))

        # the list of `v` may have moved
        pbeg = begptr[v]; pend = endptr[v]
    end

    # `wgt` is the weight of `v`
    @inbounds wgt = weight[v]

    # `del` is the difference between the weight
    # of `v` and the weight of `w`
    @inbounds del = wgt - weight[w]

    # `deg` is the weighted degree of `w`
    @inbounds deg = degree[w]

    @inbounds for p in pbeg:pend - one(E)
        # `p` is the arc (`v`, `x`) and `q` is the
        # arc (`x`, `v`)
        x = target[p]; q = invptr[p]

        # if `x` is equal to `w`, remove the arc
        # (`w`, `v`)
        if x == w
            mmw_delete!(target, invptr, endptr, w, q)

            # decrease the weighted degree of `w` by the
            # weight of `v`
            deg -= wgt

        # if `x` is not adjacent to `w`, replace the
        # arc (`x`, `v`) with the arc (`x`, `w`) and
        # append the arc (`w`, `x`) to the list of `w`
        elseif hubmark[x] != epoch
            qq = endptr[w]; endptr[w] = qq + one(E)
            target[q] = w
            target[qq] = x; invptr[qq] = q; invptr[q] = qq
            hubmark[x] = epoch

            # increase the weighted degree of `w` by the
            # weight of `x`
            deg += weight[x]

            # increase the weighted degree of `x` by the
            # weight of `w` and decrease it by the weight
            # of `v`
            if ispositive(del)
                delete!(set(degree[x]), x)
                degree[x] -= del; pushfirst!(set(degree[x]), x)
            end

        # otherwise, `x` is adjacent to `w`; remove
        # the arc (`x`, `v`)
        else
            mmw_delete!(target, invptr, endptr, x, q)

            # decrease the weighted degree of `x` by
            # the weight of `v`
            delete!(set(degree[x]), x)
            degree[x] -= wgt; pushfirst!(set(degree[x]), x)
        end
    end

    # empty the list of `v`
    @inbounds endptr[v] = pbeg

    # update the weighted degree of `w`
    @inbounds delete!(set(degree[w]), w)
    @inbounds degree[w] = deg; pushfirst!(set(deg), w)
    return pfree, hub, epoch
end

"""
    mmw_remove!(degree, target, invptr, begptr,
        endptr, head, prev, next, weight, v)

Remove the vertex `v` from the graph.

input parameters:
 - `v`: minimum degree vertex
"""
function mmw_remove!(
        degree::AbstractVector{V},
        target::AbstractVector{V},
        invptr::AbstractVector{E},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        head::AbstractVector{V},
        prev::AbstractVector{V},
        next::AbstractVector{V},
        weight::AbstractVector{V},
        v::V,
    ) where {V, E}

    # `set(i)` constructs the bucket for weighted degree `i`
    function set(i::V)
        @inbounds h = view(head, i + one(V))
        return DoublyLinkedList(h, prev, next)
    end

    # `wgt` is the weight of `v`
    @inbounds wgt = weight[v]

    # the arcs {`pbeg`, ..., `pend` - 1} are
    # incident to `v`
    @inbounds pbeg = begptr[v]; pend = endptr[v]

    @inbounds for p in pbeg:pend - one(E)
        # `p` is the arc (`v`, `x`) and `q` is the
        # arc (`x`, `v`); remove `q`
        x = target[p]; q = invptr[p]
        mmw_delete!(target, invptr, endptr, x, q)

        # decrease the weighted degree of `x` by the
        # weight of `v`
        delete!(set(degree[x]), x)
        degree[x] -= wgt; pushfirst!(set(degree[x]), x)
    end

    # empty the list of `v`
    @inbounds endptr[v] = pbeg
    return
end

"""
    mmw_delete!(target, invptr, endptr, v, p)

Remove the arc `p` from the list of `v`, replacing
it with the last arc in the list.

input parameters:
 - `v`: a vertex
 - `p`: an arc incident to `v`
"""
@inline function mmw_delete!(
        target::AbstractVector{V},
        invptr::AbstractVector{E},
        endptr::AbstractVector{E},
        v::V,
        p::E,
    ) where {V, E}
    # `pend` is the last arc incident to `v`
    @inbounds pend = endptr[v] -= one(E)

    if p < pend
        # `pp` is the reverse of `pend`
        @inbounds target[p] = target[pend]
        @inbounds pp = invptr[p] = invptr[pend]
        @inbounds invptr[pp] = p
    end

    return
end

"""
    mmw_relocate!(stash, target, invptr, begptr, endptr,
        limptr, pfree, cap, n, w, need)

Move the list of `w` to the end of the storage, making
room for at least `need` more arcs. If the storage is
full, compact it first.

input parameters:
 - `cap`: size of the arc storage
 - `n`: number of vertices
 - `w`: a vertex
 - `need`: number of arcs

updated parameters:
 - `pfree`: first free arc
"""
function mmw_relocate!(
        stash::AbstractVector{V},
        target::AbstractVector{V},
        invptr::AbstractVector{E},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        limptr::AbstractVector{E},
        pfree::E,
        cap::E,
        n::V,
        w::V,
        need::E,
    ) where {V, E}
    # `len` is the length of the list of `w`
    @inbounds len = endptr[w] - begptr[w]

    # `newlen` is the capacity of the new list
    newlen = max(twice(len), len + need)

    # if the storage is full, compact it
    if cap < pfree + newlen
        pfree = mmw_compact!(stash, target, invptr,
            begptr, endptr, limptr, pfree, n)
    end

    # move the arcs (`w`, `x`) to the end of the
    # storage
    pnew = pfree

    @inbounds for p in begptr[w]:endptr[w] - one(E)
        target[pnew] = target[p]
        pp = invptr[pnew] = invptr[p]
        invptr[pp] = pnew
        pnew += one(E)
    end

    # clear the room at the end of the new list, so
    # that it is never mistaken for a list head during
    # a compaction
    @inbounds for p in pnew:pfree + newlen - one(E)
        target[p] = zero(V)
    end

    @inbounds begptr[w] = pfree
    @inbounds endptr[w] = pfree + len
    @inbounds limptr[w] = pfree + newlen
    return pfree + newlen
end

"""
    mmw_compact!(stash, target, invptr, begptr,
        endptr, limptr, pfree, n)

Compact the arc storage by moving every nonempty list
to the front. The list heads are found by replacing the
first arc (`v`, `x`) of every nonempty list with -`v`,
and then scanning the storage. The vertices in the lists
are positive, and the unused parts of the storage are
nonnegative.

input parameters:
 - `n`: number of vertices

updated parameters:
 - `pfree`: first free arc

working arrays:
 - `stash`: the first neighbor of every vertex
"""
function mmw_compact!(
        stash::AbstractVector{V},
        target::AbstractVector{V},
        invptr::AbstractVector{E},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        limptr::AbstractVector{E},
        pfree::E,
        n::V,
    ) where {V, E}
    # replace the first arc (`v`, `x`) of every
    # nonempty list with -`v`; store `x` in `stash`
    @inbounds for v in oneto(n)
        pbeg = begptr[v]

        if pbeg < endptr[v]
            stash[v] = target[pbeg]; target[pbeg] = -v
        end
    end

    # `psrc` is the current arc
    # `pdst` is the first free arc
    psrc = pdst = one(E)

    @inbounds while psrc < pfree
        v = target[psrc]

        # if `psrc` is the first arc of a list,
        # move the list to `pdst`
        if isnegative(v)
            v = -v; len = endptr[v] - begptr[v]

            target[psrc] = stash[v]
            begptr[v] = pdst; endptr[v] = limptr[v] = pdst + len

            # if the list does not move, skip it
            if psrc == pdst
                pdst += len; psrc += len
            else
                for _ in oneto(len)
                    target[pdst] = target[psrc]
                    pp = invptr[pdst] = invptr[psrc]
                    invptr[pp] = pdst
                    pdst += one(E); psrc += one(E)
                end
            end
        else
            psrc += one(E)
        end
    end

    # empty lists get no room
    @inbounds for v in oneto(n)
        if endptr[v] <= begptr[v]
            begptr[v] = endptr[v] = limptr[v] = pdst
        end
    end

    return pdst
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
