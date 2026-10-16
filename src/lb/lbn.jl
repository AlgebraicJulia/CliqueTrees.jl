"""
    LBN{S} <: LowerBoundAlgorithm

    LBN{S}()

The LBN+ heuristic, with the minor-min-width heuristic as the
inner bound.

Let K be a lower bound on the weighted treewidth (the largest bag
weight), and suppose for contradiction that the weighted treewidth is
at most K. Then any two nonadjacent vertices whose common neighbors
weigh at least K - wmin + 1, where wmin is the least vertex weight, are
adjacent in every minimal triangulation of width K: a minimal separator
of such a triangulation contains all common neighbors of the two
vertices, and it weighs at most K - wmin. Adding all such edges, until
none are left, gives the neighbor-improved graph, which has the same
weighted treewidth. If a minor of the improved graph has minimum
(closed) weighted degree greater than K, then the hypothesis is false,
and K + 1 is a lower bound. The test is repeated until it fails.

### Parameters

  - `S`: contraction strategy of the inner MMW
    - `1`: min-d (fast)
    - `2`: max-d (fast)
    - `3`: least-c (slow)

### References

  - Clautiaux, François, et al. "New lower and upper bounds for graph treewidth." *International Workshop on Experimental and Efficient Algorithms.* 2003.
  - Bodlaender, Hans, Thomas Wolle, and Arie Koster. "Contraction and treewidth lower bounds." *Journal of Graph Algorithms and Applications* 10.1 (2006): 5-49.
  - Bodlaender, Hans, and Arie Koster. "Treewidth computations II. Lower bounds." *Information and Computation* 209.7 (2011): 1103-1119.
"""
struct LBN{S} <: LowerBoundAlgorithm end

function LBN()
    return LBN{3}()
end

function lowerbound(weights::AbstractVector, graph::AbstractGraph, ::LBN{S}) where {S}
    return lbn(weights, graph, Val(S))
end

function lbn(weights::AbstractVector{W}, graph::AbstractGraph, strategy::Val) where {W}
    width = lbcall(weights, graph, true) do V, E, totdeg
        lbn(V, E, totdeg, weights, graph, strategy)
    end

    return convert(W, width)
end

function lbn(::Type{V}, ::Type{E}, totdeg::Integer, weights::AbstractVector, graph::AbstractGraph, strategy::Val) where {V, E}
    n = convert(V, nv(graph)); m = convert(E, de(graph)); nn = n + one(V)
    cap = convert(E, mmwcapacity(n, m)); totdeg = convert(V, totdeg)
    weight = mmwweights(V, weights, n)

    hubmark = FVector{V}(undef, n)

    # the arc storage grows when edges are added
    degree = FVector{V}(undef, n)
    target = Vector{V}(undef, cap)
    invptr = Vector{E}(undef, cap)
    begptr = FVector{E}(undef, nn)
    endptr = FVector{E}(undef, n)
    limptr = FVector{E}(undef, n)

    head = FVector{V}(undef, totdeg + one(V))
    prev = FVector{V}(undef, n)
    next = FVector{V}(undef, n)

    nbrmark = FVector{Int}(undef, n)
    auxmark = FVector{Int}(undef, n)
    buffer = FVector{V}(undef, n)
    work = FVector{V}(undef, n)
    inwork = FVector{Int}(undef, n)
    nbrs = FVector{V}(undef, n)
    alist = FVector{V}(undef, n)
    apos = FVector{V}(undef, n)

    fill!(auxmark, 0)
    fill!(inwork, 0)

    # `width` is the MMW bound
    width = mmw(V, E, totdeg, weights, graph, strategy)

    # `tag` is a running marker tag; it is greater than `n`,
    # since `nbrmark` stores vertices during a compaction
    tag = Int(n)

    # while the hypothesis "the weighted treewidth is at most
    # `width`" is refuted, increase `width`
    while true
        refuted, tag = lbn_impl!(hubmark, degree, target, invptr, begptr,
            endptr, limptr, head, prev, next, nbrmark, auxmark, buffer,
            work, inwork, nbrs, alist, apos, tag, totdeg, width, weight,
            graph, strategy)

        refuted || break
        width += one(V)
    end

    return width
end

# the least vertex weight, which must be positive
function lbnwmin(weight::AbstractVector{V}, n::V) where {V}
    wmin = one(V)

    if ispositive(n)
        wmin = minimum(weight)
    end

    @assert ispositive(wmin)
    return wmin
end

"""
    lbn_impl!(hubmark, degree, target, invptr, begptr, endptr, limptr,
        head, prev, next, nbrmark, auxmark, buffer, work, inwork, nbrs,
        alist, apos, tag, totdeg, width, weight, graph, strategy)

New Lower and Upper Bounds for Graph Treewidth
Clautiaux, Carlier, Moukrim, and Nègre
LBN

Treewidth Computations II. Lower Bounds
Bodlaender and Koster
LBN+

Test the hypothesis that the weighted treewidth of a graph is at
most `width`. If it is, then two nonadjacent vertices whose common
neighbors weigh at least

    `thr` = `width` - `wmin` + 1

where `wmin` is the least vertex weight, are adjacent in every minimal
triangulation of width `width`, so joining them does not change the
weighted treewidth. The test runs the minor-min-width heuristic (see
[`mmw_impl!`](@ref)) on the graph, and joins such pairs of vertices
before the first contraction and after every contraction. If a minor
has minimum weighted degree greater than `width`, then the hypothesis
is refuted.

Joining the vertices `u` and `y` changes the common neighbors of the
pairs that contain `u` or `y` only, so the pairs are tested by a work
stack of vertices: a vertex `u` is scanned by counting the weight of
its common neighbors with every vertex at distance two, or, if `u`
has more neighbors than non-neighbors, by testing each non-neighbor
directly. After the contraction of `v` into `w`, a pair {`w`, `y`}
gains common neighbors only if `y` is adjacent to a new neighbor of
`w`, and a pair {`a`, `b`} that does not contain `w` gains a common
neighbor only if `a` is a new neighbor of `w` and `b` is an old
neighbor of `w` that is not adjacent to `v` (see
[`lbn_contracted!`](@ref)).

The state of the algorithm is a collection of arrays and scalars.

  - graph (see [`mmw_impl!`](@ref)):
    - `target`, `invptr`, `begptr`, `endptr`, `limptr`: adjacency
      lists; `target` and `invptr` grow when the storage is full
    - `degree`: weighted degree
    - `head`, `prev`, `next`: vertex-degree bucket queue
    - `hubmark`: as in [`mmw_impl!`](@ref)
  - improvement:
    - `nbrmark`: marks the closed neighborhood of a vertex, and the
                 neighbors of a minimum degree vertex; it serves as
                 the arrays `marker` and `stash` of
                 [`mmw_impl!`](@ref), so the tags are greater than
                 the number of vertices
    - `auxmark`: marks the vertices whose count is valid, and the
                 neighbors of a contracted vertex
    - `buffer`: the vertices at distance two from a vertex, and the
                vertices to be joined to it
    - `work`: work stack of vertices to be scanned
    - `inwork`: membership in `work`: `inwork[v]` is -1 if and only
                if `v` is in `work`
    - `nbrs`: the neighbors of a contracted vertex, and the weight
              of the common neighbors of two vertices
    - `alist`: list of the remaining vertices
    - `apos`: positions in `alist`
  - scalars:
    - `tag`: a running marker tag
    - `pfree`: first free arc
    - `hub`, `epoch`: as in [`mmw_impl!`](@ref)
    - `nalive`: length of `alist`
    - `nwork`: height of `work`

input parameters:
  - `totdeg`: total vertex weight
  - `width`: hypothetical weighted treewidth
  - `weight`: vertex weight
  - `graph`: input graph
  - `S`: strategy (see [`mmw_impl!`](@ref))

output parameters:
  - `refuted`: true if the hypothesis is refuted
  - `tag`: a running marker tag
"""
function lbn_impl!(
        hubmark::AbstractVector{V},
        degree::AbstractVector{V},
        target::Vector{V},
        invptr::Vector{E},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        limptr::AbstractVector{E},
        head::AbstractVector{V},
        prev::AbstractVector{V},
        next::AbstractVector{V},
        nbrmark::AbstractVector{Int},
        auxmark::AbstractVector{Int},
        buffer::AbstractVector{V},
        work::AbstractVector{V},
        inwork::AbstractVector{Int},
        nbrs::AbstractVector{V},
        alist::AbstractVector{V},
        apos::AbstractVector{V},
        tag::Int,
        totdeg::V,
        width::V,
        weight::AbstractVector{V},
        graph::AbstractGraph,
        strategy::Val,
    ) where {V <: Signed, E}

    @assert nv(graph) <= length(nbrmark)
    @assert nv(graph) <= length(auxmark)
    @assert nv(graph) <= length(buffer)
    @assert nv(graph) <= length(work)
    @assert nv(graph) <= length(inwork)
    @assert nv(graph) <= length(nbrs)
    @assert nv(graph) <= length(alist)
    @assert nv(graph) <= length(apos)

    # `set(i)` constructs the bucket for weighted degree `i`
    function set(i::V)
        @inbounds h = view(head, i + one(V))
        return DoublyLinkedList(h, prev, next)
    end

    # `n` is the number of vertices in the graph
    n = convert(V, nv(graph))

    # `thr` is the least weight of the common neighbors
    # of two vertices that are joined
    thr = width - lbnwmin(weight, n) + one(V)

    # initialize the graph, and improve it
    mindeg, pfree, nalive, tag = lbn_init!(hubmark, degree, target,
        invptr, begptr, endptr, limptr, head, prev, next, nbrmark,
        auxmark, buffer, work, inwork, nbrs, alist, apos, tag, totdeg,
        thr, weight, graph)

    # the neighbors `x` of `hub` satisfy
    #    hubmark[x] = `epoch`
    hub = epoch = zero(V)

    # `remdeg` is the total weight of the remaining graph
    remdeg = totdeg

    @inbounds for i in oneto(n)
        # find the new minimum degree; the improvement only
        # increases degrees, so `mindeg` is a lower bound
        while isempty(set(mindeg))
            mindeg += one(V)
        end

        # if the minimum degree is greater than `width`, then
        # the hypothesis is refuted
        if mindeg > width
            return true, tag
        end

        # if the remaining graph is too light to refute the
        # hypothesis, stop
        if remdeg <= width
            return false, tag
        end

        # select a vertex of minimum degree
        v = popfirst!(set(mindeg))
        remdeg -= weight[v]
        nalive = lbn_remove!(alist, apos, nalive, v)

        # find a neighbor according to strategy `S`
        tag += 1

        w = mmw_search!(nbrmark, hubmark, degree, target, begptr,
            endptr, weight, hub, epoch, tag, v, strategy)

        # if a neighbor was found, contract the edge
        # {`v`, `w`} and improve the graph
        if ispositive(w)
            pfree, hub, epoch, tag = lbn_contract!(hubmark, degree,
                target, invptr, begptr, endptr, limptr, head, prev, next,
                nbrmark, auxmark, buffer, work, inwork, nbrs, alist,
                nalive, pfree, hub, epoch, tag, thr, weight, n, v, w)

        # otherwise, remove `v` from the graph
        else
            mmw_remove!(degree, target, invptr, begptr,
                endptr, head, prev, next, weight, v)
        end

        # the weighted degree of every neighbor of `v`
        # has decreased by at most the weight of `v`
        mindeg = max(mindeg - weight[v], zero(V))
    end

    return false, tag
end

"""
    lbn_init!(hubmark, degree, target, invptr, begptr, endptr, limptr,
        head, prev, next, nbrmark, auxmark, buffer, work, inwork, nbrs,
        alist, apos, tag, totdeg, thr, weight, graph)

Initialize the adjacency lists, the bucket queue, and the list of
remaining vertices, and join every two vertices whose common
neighbors weigh at least `thr`.

input parameters:
  - `totdeg`: total vertex weight
  - `thr`: threshold
  - `weight`: vertex weight
  - `graph`: input graph

output parameters:
  - `mindeg`: a lower bound on the minimum weighted degree
  - `pfree`: first free arc
  - `nalive`: length of `alist`
  - `tag`: a running marker tag
"""
function lbn_init!(
        hubmark::AbstractVector{V},
        degree::AbstractVector{V},
        target::Vector{V},
        invptr::Vector{E},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        limptr::AbstractVector{E},
        head::AbstractVector{V},
        prev::AbstractVector{V},
        next::AbstractVector{V},
        nbrmark::AbstractVector{Int},
        auxmark::AbstractVector{Int},
        buffer::AbstractVector{V},
        work::AbstractVector{V},
        inwork::AbstractVector{Int},
        nbrs::AbstractVector{V},
        alist::AbstractVector{V},
        apos::AbstractVector{V},
        tag::Int,
        totdeg::V,
        thr::V,
        weight::AbstractVector{V},
        graph::AbstractGraph,
    ) where {V, E}

    # `n` is the number of vertices in the graph
    n = convert(V, nv(graph))

    # `mindeg` is the minimum weighted degree
    # `pfree` is the first free arc
    mindeg, pfree = mmw_init!(nbrmark, hubmark, degree, target, invptr,
        begptr, endptr, limptr, head, prev, next, totdeg, weight, graph)

    # every vertex remains
    @inbounds for v in oneto(n)
        alist[v] = apos[v] = v
    end

    nalive = Int(n)

    # scan every vertex
    nwork = 0

    @inbounds for v in reverse(oneto(n))
        nwork = lbn_push!(work, inwork, nwork, v)
    end

    pfree, tag = lbn_improve!(hubmark, degree, target, invptr, begptr,
        endptr, limptr, head, prev, next, nbrmark, auxmark, nbrs, buffer,
        work, inwork, alist, nalive, nwork, pfree, zero(V), zero(V), tag,
        thr, weight, n)

    return mindeg, pfree, nalive, tag
end

"""
    lbn_contract!(hubmark, degree, target, invptr, begptr, endptr,
        limptr, head, prev, next, nbrmark, auxmark, buffer, work, inwork,
        nbrs, alist, nalive, pfree, hub, epoch, tag, thr, weight, n, v,
        w)

Contract the edge {`v`, `w`}, merging the vertex `v` into the vertex
`w` (see [`mmw_contract!`](@ref)), and join every two vertices whose
common neighbors weigh at least `thr`.

input parameters:
  - `thr`: threshold
  - `weight`: vertex weight
  - `n`: number of vertices
  - `v`: minimum degree vertex
  - `w`: neighbor of `v`

updated parameters:
  - `pfree`: first free arc
  - `hub`: the neighbors `x` of `hub` satisfy
      hubmark[x] = `epoch`
  - `epoch`: tag for marking vertices
  - `tag`: a running marker tag
"""
function lbn_contract!(
        hubmark::AbstractVector{V},
        degree::AbstractVector{V},
        target::Vector{V},
        invptr::Vector{E},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        limptr::AbstractVector{E},
        head::AbstractVector{V},
        prev::AbstractVector{V},
        next::AbstractVector{V},
        nbrmark::AbstractVector{Int},
        auxmark::AbstractVector{Int},
        buffer::AbstractVector{V},
        work::AbstractVector{V},
        inwork::AbstractVector{Int},
        nbrs::AbstractVector{V},
        alist::AbstractVector{V},
        nalive::Int,
        pfree::E,
        hub::V,
        epoch::V,
        tag::Int,
        thr::V,
        weight::AbstractVector{V},
        n::V,
        v::V,
        w::V,
    ) where {V, E}

    # mark the neighbors of `w`
    tag += 1

    @inbounds for p in begptr[w]:endptr[w] - one(E)
        auxmark[target[p]] = tag
    end

    # record the neighbors of `v`: first the ones that
    # will be new neighbors of `w`, and then the others
    nnew = 0

    @inbounds for p in begptr[v]:endptr[v] - one(E)
        x = target[p]

        if x != w && auxmark[x] != tag
            nnew += 1; nbrs[nnew] = x
        end
    end

    nnbr = nnew

    @inbounds for p in begptr[v]:endptr[v] - one(E)
        x = target[p]

        if x == w || auxmark[x] == tag
            nnbr += 1; nbrs[nnbr] = x
        end
    end

    # make room for the new neighbors of `w`, so that the
    # contraction does not run out of storage
    @inbounds lenv = endptr[v] - begptr[v]
    @inbounds lenw = endptr[w] - begptr[w]

    @inbounds if limptr[w] < endptr[w] + lenv - one(E)
        pfree = lbn_reserve!(nbrmark, target, invptr, begptr, endptr,
            limptr, pfree, n, max(twice(lenw), lenw + lenv - one(E)))
    end

    cap = convert(E, length(target))

    pfree, hub, epoch = mmw_contract!(hubmark, nbrmark, degree, target,
        invptr, begptr, endptr, limptr, head, prev, next, weight, pfree,
        cap, hub, epoch, n, v, w)

    # test the pairs that may have gained common neighbors
    nwork = 0

    pfree, nwork, tag = lbn_contracted!(hubmark, degree, target, invptr,
        begptr, endptr, limptr, head, prev, next, nbrmark, auxmark,
        buffer, work, inwork, nbrs, nnew, nnbr, nwork, pfree, hub, epoch,
        tag, thr, weight, n, w)

    # the neighbors of `v` are no longer needed, so `nbrs`
    # holds the weight of the common neighbors
    pfree, tag = lbn_improve!(hubmark, degree, target, invptr, begptr,
        endptr, limptr, head, prev, next, nbrmark, auxmark, nbrs, buffer,
        work, inwork, alist, nalive, nwork, pfree, hub, epoch, tag, thr,
        weight, n)

    return pfree, hub, epoch, tag
end

"""
    lbn_contracted!(hubmark, degree, target, invptr, begptr, endptr,
        limptr, head, prev, next, nbrmark, auxmark, buffer, work, inwork,
        nbrs, nnew, nnbr, nwork, pfree, hub, epoch, tag, thr, weight, n,
        w)

After the contraction of a vertex `v` into the vertex `w`, test the
pairs of vertices that may have gained common neighbors:

  - the pairs {`w`, `y`}, where `y` is adjacent to a new neighbor of `w`
  - the pairs {`a`, `b`}, where `a` is a new neighbor of `w` and `b` is
    an old neighbor of `w` that is not adjacent to `v`; the new common
    neighbor is `w`

The neighbors of `v` are `nbrs[1:nnbr]`, and the new neighbors of `w`
are `nbrs[1:nnew]`. A pair is joined if the common neighbors weigh at
least `thr`; then both of its vertices are pushed onto the work stack.

input parameters:
  - `nnew`: number of new neighbors of `w`
  - `nnbr`: number of neighbors of `v`
  - `thr`: threshold
  - `weight`: vertex weight
  - `n`: number of vertices
  - `w`: contracted vertex

updated parameters:
  - `nwork`: height of `work`
  - `pfree`: first free arc
  - `tag`: a running marker tag
"""
function lbn_contracted!(
        hubmark::AbstractVector{V},
        degree::AbstractVector{V},
        target::Vector{V},
        invptr::Vector{E},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        limptr::AbstractVector{E},
        head::AbstractVector{V},
        prev::AbstractVector{V},
        next::AbstractVector{V},
        nbrmark::AbstractVector{Int},
        auxmark::AbstractVector{Int},
        buffer::AbstractVector{V},
        work::AbstractVector{V},
        inwork::AbstractVector{Int},
        nbrs::AbstractVector{V},
        nnew::Int,
        nnbr::Int,
        nwork::Int,
        pfree::E,
        hub::V,
        epoch::V,
        tag::Int,
        thr::V,
        weight::AbstractVector{V},
        n::V,
        w::V,
    ) where {V, E}

    # test the pairs {`w`, `y`}, where `y` is adjacent to a
    # new neighbor of `w`
    @inbounds if degree[w] - weight[w] >= thr
        tag += 1

        # mark the closed neighborhood of `w`
        nbrmark[w] = tag

        for p in begptr[w]:endptr[w] - one(E)
            nbrmark[target[p]] = tag
        end

        # `nf` is the number of vertices to be joined to `w`
        nf = 0

        for i in oneto(nnew)
            # `a` is a new neighbor of `w`
            a = nbrs[i]

            for p in begptr[a]:endptr[a] - one(E)
                y = target[p]

                # `slack` is the greatest weight that the
                # neighbors of `y` outside of N(`w`) may have
                slack = degree[y] - weight[y] - thr

                if nbrmark[y] != tag && auxmark[y] != tag && !isnegative(slack)
                    auxmark[y] = tag

                    if lbn_test(degree, target, begptr, endptr, nbrmark, weight, tag, slack, y)
                        nf += 1; buffer[nf] = y
                    end
                end
            end
        end

        for i in oneto(nf)
            y = buffer[i]

            pfree = lbn_addedge!(hubmark, nbrmark, degree, target,
                invptr, begptr, endptr, limptr, head, prev, next, weight,
                pfree, hub, epoch, n, w, y)

            nwork = lbn_push!(work, inwork, nwork, y)
        end

        if ispositive(nf)
            nwork = lbn_push!(work, inwork, nwork, w)
        end
    end

    # mark the old neighbors of `v`
    tag += 1; vtag = tag

    @inbounds for i in oneto(nnbr)
        auxmark[nbrs[i]] = vtag
    end

    # test the pairs {`a`, `b`}, where `a` is a new neighbor
    # of `w` and `b` is an old neighbor of `w` that is not
    # adjacent to `v`
    @inbounds for i in oneto(nnew)
        # `a` is a new neighbor of `w`
        a = nbrs[i]

        if degree[a] - weight[a] >= thr
            tag += 1

            # mark the closed neighborhood of `a`
            nbrmark[a] = tag

            for p in begptr[a]:endptr[a] - one(E)
                nbrmark[target[p]] = tag
            end

            # `nf` is the number of vertices to be joined to `a`
            nf = 0

            for p in begptr[w]:endptr[w] - one(E)
                b = target[p]

                # `slack` is the greatest weight that the
                # neighbors of `b` outside of N(`a`) may have
                slack = degree[b] - weight[b] - thr

                if auxmark[b] != vtag && nbrmark[b] != tag && !isnegative(slack)
                    if lbn_test(degree, target, begptr, endptr, nbrmark, weight, tag, slack, b)
                        nf += 1; buffer[nf] = b
                    end
                end
            end

            for j in oneto(nf)
                b = buffer[j]

                pfree = lbn_addedge!(hubmark, nbrmark, degree, target,
                    invptr, begptr, endptr, limptr, head, prev, next,
                    weight, pfree, hub, epoch, n, a, b)

                nwork = lbn_push!(work, inwork, nwork, b)
            end

            if ispositive(nf)
                nwork = lbn_push!(work, inwork, nwork, a)
            end
        end
    end

    return pfree, nwork, tag
end

"""
    lbn_test(degree, target, begptr, endptr, nbrmark, weight, tag,
        slack, y)

Determine whether the common neighbors of a vertex `u` and the vertex
`y` weigh at least `thr`, where the closed neighborhood of `u` is
marked with `tag` in `nbrmark` and

    `slack` = degree[`y`] - weight[`y`] - `thr`.

This is so if and only if the neighbors of `y` outside of the closed
neighborhood of `u` weigh at most `slack`. The scan of the neighbors
of `y` is abandoned as soon as they weigh more.

input parameters:
  - `tag`: tag for marking vertices
  - `slack`: the greatest weight of the neighbors of `y` outside of
             the closed neighborhood of `u`
  - `y`: a vertex that is not adjacent to `u`
"""
function lbn_test(
        degree::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        nbrmark::AbstractVector{Int},
        weight::AbstractVector{V},
        tag::Int,
        slack::V,
        y::V,
    ) where {V, E}

    # `miss` is the weight of the neighbors of `y` outside
    # of the closed neighborhood of `u`
    miss = zero(V)

    @inbounds for p in begptr[y]:endptr[y] - one(E)
        x = target[p]
        miss += ifelse(nbrmark[x] == tag, zero(V), weight[x])

        if miss > slack
            return false
        end
    end

    return true
end

"""
    lbn_improve!(hubmark, degree, target, invptr, begptr, endptr, limptr,
        head, prev, next, nbrmark, auxmark, common, buffer, work, inwork,
        alist, nalive, nwork, pfree, hub, epoch, tag, thr, weight, n)

Scan the vertices on the work stack, joining every vertex `u` to every
nonadjacent vertex `y` such that the common neighbors of `u` and `y`
weigh at least `thr`. Joining `u` and `y` changes the common
neighbors of the pairs that contain `u` or `y` only, so both are
scanned again.

A vertex `u` whose neighbors weigh less than `thr` is skipped. If `u`
has fewer non-neighbors than neighbors, then each non-neighbor is
tested directly (see [`lbn_test`](@ref)); otherwise, the weight of the
common neighbors is counted for every vertex at distance two from `u`.

input parameters:
  - `nalive`: length of `alist`
  - `nwork`: height of `work`
  - `hub`: the neighbors `x` of `hub` satisfy
      hubmark[x] = `epoch`
  - `epoch`: tag for marking vertices
  - `thr`: threshold
  - `weight`: vertex weight
  - `n`: number of vertices

updated parameters:
  - `pfree`: first free arc
  - `tag`: a running marker tag
"""
function lbn_improve!(
        hubmark::AbstractVector{V},
        degree::AbstractVector{V},
        target::Vector{V},
        invptr::Vector{E},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        limptr::AbstractVector{E},
        head::AbstractVector{V},
        prev::AbstractVector{V},
        next::AbstractVector{V},
        nbrmark::AbstractVector{Int},
        auxmark::AbstractVector{Int},
        common::AbstractVector{V},
        buffer::AbstractVector{V},
        work::AbstractVector{V},
        inwork::AbstractVector{Int},
        alist::AbstractVector{V},
        nalive::Int,
        nwork::Int,
        pfree::E,
        hub::V,
        epoch::V,
        tag::Int,
        thr::V,
        weight::AbstractVector{V},
        n::V,
    ) where {V, E}

    @inbounds while ispositive(nwork)
        # pop a vertex `u` from the work stack
        u = work[nwork]; nwork -= 1; inwork[u] = 0

        # the neighbors of `u` must weigh at least `thr`
        if degree[u] - weight[u] >= thr
            tag += 1

            # mark the closed neighborhood of `u`
            nbrmark[u] = tag

            for p in begptr[u]:endptr[u] - one(E)
                nbrmark[target[p]] = tag
            end

            # `nf` is the number of vertices to be joined to `u`
            nf = 0

            # `lenu` is the number of neighbors of `u`
            lenu = Int(endptr[u] - begptr[u])

            # if `u` has fewer non-neighbors than neighbors,
            # then test its non-neighbors directly
            if nalive - 1 - lenu < lenu
                for i in oneto(nalive)
                    y = alist[i]

                    # `slack` is the greatest weight that the
                    # neighbors of `y` outside of N(`u`) may have
                    slack = degree[y] - weight[y] - thr

                    if nbrmark[y] != tag && !isnegative(slack)
                        if lbn_test(degree, target, begptr, endptr, nbrmark, weight, tag, slack, y)
                            nf += 1; buffer[nf] = y
                        end
                    end
                end

            # otherwise, count the weight of the common neighbors
            # of `u` and each vertex `y` at distance two
            else
                nt = 0

                for p in begptr[u]:endptr[u] - one(E)
                    x = target[p]; wx = weight[x]

                    for q in begptr[x]:endptr[x] - one(E)
                        y = target[q]

                        if nbrmark[y] != tag && degree[y] - weight[y] >= thr
                            if auxmark[y] != tag
                                auxmark[y] = tag; common[y] = zero(V)
                                nt += 1; buffer[nt] = y
                            end

                            common[y] += wx
                        end
                    end
                end

                for i in oneto(nt)
                    y = buffer[i]

                    if common[y] >= thr
                        nf += 1; buffer[nf] = y
                    end
                end
            end

            # join `u` to `buffer[1:nf]`
            for i in oneto(nf)
                y = buffer[i]

                pfree = lbn_addedge!(hubmark, nbrmark, degree, target,
                    invptr, begptr, endptr, limptr, head, prev, next,
                    weight, pfree, hub, epoch, n, u, y)

                nwork = lbn_push!(work, inwork, nwork, y)
            end

            if ispositive(nf)
                nwork = lbn_push!(work, inwork, nwork, u)
            end
        end
    end

    return pfree, tag
end

"""
    lbn_addedge!(hubmark, stash, degree, target, invptr, begptr,
        endptr, limptr, head, prev, next, weight, pfree, hub, epoch,
        n, u, y)

Add the edge {`u`, `y`}. If the list of `u` or `y` is full, it is
moved to the end of the storage (see [`mmw_relocate!`](@ref)); if the
storage is full, it is compacted, and if it is still full, it grows.

input parameters:
  - `weight`: vertex weight
  - `hub`: the neighbors `x` of `hub` satisfy
      hubmark[x] = `epoch`
  - `epoch`: tag for marking vertices
  - `n`: number of vertices
  - `u`, `y`: nonadjacent vertices

updated parameters:
  - `pfree`: first free arc
"""
function lbn_addedge!(
        hubmark::AbstractVector{V},
        stash::AbstractVector{T},
        degree::AbstractVector{V},
        target::Vector{V},
        invptr::Vector{E},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        limptr::AbstractVector{E},
        head::AbstractVector{V},
        prev::AbstractVector{V},
        next::AbstractVector{V},
        weight::AbstractVector{V},
        pfree::E,
        hub::V,
        epoch::V,
        n::V,
        u::V,
        y::V,
    ) where {T, V, E}

    # `set(i)` constructs the bucket for weighted degree `i`
    function set(i::V)
        @inbounds h = view(head, i + one(V))
        return DoublyLinkedList(h, prev, next)
    end

    # `fullu` and `fully` are true if the lists of `u`
    # and `y` are full
    @inbounds fullu = endptr[u] >= limptr[u]
    @inbounds fully = endptr[y] >= limptr[y]

    if fullu || fully
        # `lenu` and `leny` are the lengths of the lists
        @inbounds lenu = endptr[u] - begptr[u]
        @inbounds leny = endptr[y] - begptr[y]

        # `need` is the room needed by the moved lists
        need = zero(E)
        fullu && (need += lbnnewlen(lenu))
        fully && (need += lbnnewlen(leny))

        # if the storage is full, compact it; then both lists
        # have lost their room
        if length(target) < pfree + need
            pfree = mmw_compact!(stash, target, invptr, begptr,
                endptr, limptr, pfree, n)

            pfree = lbn_reserve!(stash, target, invptr, begptr, endptr,
                limptr, pfree, n, lbnnewlen(lenu) + lbnnewlen(leny))
        end

        cap = convert(E, length(target))

        @inbounds if endptr[u] >= limptr[u]
            pfree = mmw_relocate!(stash, target, invptr, begptr,
                endptr, limptr, pfree, cap, n, u, one(E))
        end

        @inbounds if endptr[y] >= limptr[y]
            pfree = mmw_relocate!(stash, target, invptr, begptr,
                endptr, limptr, pfree, cap, n, y, one(E))
        end
    end

    @inbounds begin
        # `pu` is the arc (`u`, `y`) and `py` is the arc (`y`, `u`)
        pu = endptr[u]; endptr[u] = pu + one(E)
        py = endptr[y]; endptr[y] = py + one(E)
        target[pu] = y; target[py] = u
        invptr[pu] = py; invptr[py] = pu

        # increase the weighted degrees of `u` and `y`
        delete!(set(degree[u]), u)
        degree[u] += weight[y]; pushfirst!(set(degree[u]), u)

        delete!(set(degree[y]), y)
        degree[y] += weight[u]; pushfirst!(set(degree[y]), y)

        # keep the neighbors of `hub` marked
        if u == hub
            hubmark[y] = epoch
        elseif y == hub
            hubmark[u] = epoch
        end
    end

    return pfree
end

# the capacity of a moved list of length `len`, with
# room for one more arc
function lbnnewlen(len::I) where {I <: Integer}
    return max(twice(len), len + one(I))
end

"""
    lbn_reserve!(stash, target, invptr, begptr, endptr, limptr,
        pfree, n, need)

Make room for at least `need` arcs at the end of the storage. If the
storage is full, compact it (see [`mmw_compact!`](@ref)); if it is
still full, grow it.

input parameters:
  - `n`: number of vertices
  - `need`: number of arcs

updated parameters:
  - `pfree`: first free arc
"""
function lbn_reserve!(
        stash::AbstractVector{T},
        target::Vector{V},
        invptr::Vector{E},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        limptr::AbstractVector{E},
        pfree::E,
        n::V,
        need::E,
    ) where {T, V, E}

    # `cap` is the size of the arc storage
    cap = length(target)

    if cap < pfree + need
        pfree = mmw_compact!(stash, target, invptr, begptr,
            endptr, limptr, pfree, n)
    end

    if cap < pfree + need
        newcap = max(twice(cap), Int(pfree + need) + 1)
        resize!(target, newcap)
        resize!(invptr, newcap)
    end

    return pfree
end

# push the vertex `v` onto the work stack, unless it is there
# already; returns the new height of the stack
function lbn_push!(work::AbstractVector{V}, inwork::AbstractVector{Int}, nwork::Int, v::V) where {V}
    @inbounds if inwork[v] != -1
        inwork[v] = -1
        nwork += 1; work[nwork] = v
    end

    return nwork
end

# remove the vertex `v` from the list of remaining vertices,
# replacing it with the last vertex; returns the new length
# of the list
function lbn_remove!(alist::AbstractVector{V}, apos::AbstractVector{V}, nalive::Int, v::V) where {V}
    @inbounds begin
        i = apos[v]; x = alist[nalive]
        alist[i] = x; apos[x] = i
    end

    return nalive - 1
end

function Base.show(io::IO, ::MIME"text/plain", alg::LBN{S}) where {S}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "LBN{$S}")
    return nothing
end
