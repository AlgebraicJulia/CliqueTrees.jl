"""
    LBP{S} <: LowerBoundAlgorithm

    LBP{S}()

The LBP+ heuristic, with the minor-min-width heuristic as the
inner bound.

LBP is LBN with a stronger improvement rule. Let K be a lower bound on
the weighted treewidth (the largest bag weight), and suppose that the
weighted treewidth is at most K. Then any two nonadjacent vertices that
cannot be separated by a set of weight at most K - wmin, where wmin is
the least vertex weight, are adjacent in every minimal triangulation
of width K. Two vertices that have common neighbors of weight at least
K - wmin + 1 cannot be separated by such a set, so the path rule
subsumes the neighbor rule of [`LBN`](@ref).

The bound is at least the LBN+ bound: the tests start from it.

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
struct LBP{S} <: LowerBoundAlgorithm end

function LBP()
    return LBP{3}()
end

function lowerbound(weights::AbstractVector, graph::AbstractGraph, ::LBP{S}) where {S}
    return lbp(weights, graph, Val(S))
end

function lbp(weights::AbstractVector{W}, graph::AbstractGraph, strategy::Val) where {W}
    width = lbcall(weights, graph, true) do V, E, totdeg
        lbp(V, E, totdeg, weights, graph, strategy)
    end

    return convert(W, width)
end

function lbp(::Type{V}, ::Type{E}, totdeg::Integer, weights::AbstractVector, graph::AbstractGraph, strategy::Val) where {V, E}
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
    nbrs = FVector{V}(undef, n)
    alist = FVector{V}(undef, n)
    apos = FVector{V}(undef, n)

    # the flows grow with the arc storage
    flow = Vector{V}(undef, n + cap)
    fstamp = Vector{Int}(undef, n + cap)

    nodemark = FVector{Int}(undef, twice(n))
    from = FVector{Int}(undef, twice(n))
    step = FVector{E}(undef, twice(n))
    queue = FVector{Int}(undef, twice(n))

    parent = FVector{V}(undef, n)
    arena = Vector{V}(undef, 0)
    pocket = FVector{Int}(undef, n)

    fill!(auxmark, 0)
    fill!(fstamp, 0)
    fill!(nodemark, 0)

    # `width` is the MMW bound
    width = mmw(V, E, totdeg, weights, graph, strategy)

    # `tag` is a running marker tag; it is greater than `n`,
    # since `nbrmark` stores vertices during a compaction
    tag = Int(n)

    # `width` is the LBN+ bound; `nodemark` serves as the array
    # `inwork` of the test
    while true
        refuted, tag = lbn_impl!(hubmark, degree, target, invptr, begptr,
            endptr, limptr, head, prev, next, nbrmark, auxmark, buffer,
            work, nodemark, nbrs, alist, apos, tag, totdeg, width,
            weight, graph, strategy)

        refuted || break
        width += one(V)
    end

    # `width` is the LBP+ bound
    while true
        refuted, tag = lbp_impl!(hubmark, degree, target, invptr, begptr,
            endptr, limptr, head, prev, next, nbrmark, auxmark, buffer,
            work, nbrs, alist, apos, flow, fstamp, nodemark, from, step,
            queue, parent, arena, pocket, tag, totdeg, width, weight,
            graph, strategy)

        refuted || break
        width += one(V)
    end

    return width
end

"""
    lbp_impl!(hubmark, degree, target, invptr, begptr, endptr,
        limptr, head, prev, next, nbrmark, auxmark, buffer, work, nbrs,
        alist, apos, flow, fstamp, nodemark, from, step, queue, parent,
        arena, pocket, tag, totdeg, width, weight, graph, strategy)

New Lower and Upper Bounds for Graph Treewidth
Clautiaux, Carlier, Moukrim, and Nègre
LBP

Treewidth Computations II. Lower Bounds
Bodlaender and Koster
LBP+

Test the hypothesis that the weighted treewidth of a graph is at
most `width`. If it is, then two nonadjacent vertices that cannot be
separated by a set of weight less than

    `thr` = `width` - `wmin` + 1,

where `wmin` is the least vertex weight, are adjacent in every minimal
triangulation of width `width`, so joining them does not change the
weighted treewidth. The test is
the test of [`lbn_impl!`](@ref), which joins pairs of vertices whose
common neighbors weigh at least `thr`, with this stronger rule.

Joining two such vertices does not change the weight of any separator
of weight less than `thr`, so the graph is closed under the rule in
one round: every vertex is queried for the vertices that it cannot be
separated from (see [`lbp_query!`](@ref)). After the contraction of
`v` into a vertex `w` that is no heavier, only the pairs that contain
`w` can become inseparable, so only `w` is queried.

A query finds a minimum cut whenever a flow falls short of `thr`. The
cut separates a set P of vertices from the rest of the graph, and its
vertices form a set S of weight less than `thr`: the pair (P, S) is a
*pocket*. A pocket remains a pocket when the rule joins two vertices,
since no such vertices are separated by S. It also remains a pocket
when `v` is contracted into `w`, if the contracted vertex joins S
whenever `v` is in S. So the pockets are stored, in terms of the
vertices of the graph at the start of the test, and a contracted
vertex is represented by the vertex that it was contracted into. Each
vertex keeps the smallest pocket that contains it in P.

The state of the algorithm is a collection of arrays and scalars.

  - graph and improvement (see [`lbn_impl!`](@ref)):
    - `target`, `invptr`, `begptr`, `endptr`, `limptr`: adjacency
      lists; `target` and `invptr` grow when the storage is full
    - `degree`: weighted degree
    - `head`, `prev`, `next`: vertex-degree bucket queue
    - `hubmark`: as in [`mmw_impl!`](@ref)
    - `nbrmark`: marks the closed neighborhood of a vertex, and the
                 neighbors of a minimum degree vertex; it serves as
                 the arrays `marker` and `stash` of
                 [`mmw_impl!`](@ref)
    - `auxmark`: marks the vertices whose count is valid, the
                 neighbors of a contracted vertex, the neighbors of
                 a sink, and the vertices of a pocket
    - `buffer`: the vertices at distance two from a vertex, the
                candidate sinks of a query, and the vertices to be
                joined to a vertex
    - `work`: work stack of vertices to be scanned
    - `nbrs`: the neighbors of a contracted vertex, and the weight
              of the common neighbors of two vertices
    - `alist`, `apos`: list of the remaining vertices
  - flows (see [`lbp_flow!`](@ref)):
    - `flow`, `fstamp`: the flow through the vertex `x` is stored in
      `flow[x]`, and the flow on the arc `p` in `flow[n + p]`; `flow`
      and `fstamp` grow with the arc storage
    - `nodemark`, `from`, `step`, `queue`: searches for augmenting
      paths (see [`lbp_augment!`](@ref)); before the searches,
      `nodemark` marks vertices and `step` stores arcs (see
      [`lbp_paths!`](@ref) and [`lbp_pockets!`](@ref))

  The arrays of the searches are free outside of the queries, when
  the work stack is used: `nodemark` serves as the array `inwork` (see
  [`lbn_impl!`](@ref)).
  - pockets (see [`lbp_pockets!`](@ref)):
    - `parent`: the vertex that a vertex was contracted into
    - `arena`: storage for the pockets
    - `pocket`: the smallest pocket of a vertex
  - scalars:
    - `tag`: a running marker tag
    - `pcap`: capacity of `arena`
    - `alen`: used length of `arena`
    - `pfree`, `hub`, `epoch`, `nalive`: see [`lbn_impl!`](@ref)

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
function lbp_impl!(
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
        nbrs::AbstractVector{V},
        alist::AbstractVector{V},
        apos::AbstractVector{V},
        flow::Vector{V},
        fstamp::Vector{Int},
        nodemark::AbstractVector{Int},
        from::AbstractVector{Int},
        step::AbstractVector{E},
        queue::AbstractVector{Int},
        parent::AbstractVector{V},
        arena::Vector{V},
        pocket::AbstractVector{Int},
        tag::Int,
        totdeg::V,
        width::V,
        weight::AbstractVector{V},
        graph::AbstractGraph,
        strategy::Val,
    ) where {V <: Signed, E}

    @assert twice(nv(graph)) <= length(nodemark)
    @assert twice(nv(graph)) <= length(from)
    @assert twice(nv(graph)) <= length(step)
    @assert twice(nv(graph)) <= length(queue)
    @assert nv(graph) <= length(parent)
    @assert nv(graph) <= length(pocket)

    # `set(i)` constructs the bucket for weighted degree `i`
    function set(i::V)
        @inbounds h = view(head, i + one(V))
        return DoublyLinkedList(h, prev, next)
    end

    # `n` is the number of vertices in the graph
    n = convert(V, nv(graph))

    # `thr` is the least weight of a set of vertices that
    # separates two vertices that are not joined
    thr = width - lbnwmin(weight, n) + one(V)

    # `pcap` is the capacity of `arena`
    pcap = 4mmwcapacity(Int(n), Int(de(graph))) + 16Int(n)

    # initialize the graph, and improve it by the neighbor rule
    mindeg, pfree, nalive, tag = lbn_init!(hubmark, degree, target,
        invptr, begptr, endptr, limptr, head, prev, next, nbrmark,
        auxmark, buffer, work, nodemark, nbrs, alist, apos, tag, totdeg,
        thr, weight, graph)

    # the neighbors `x` of `hub` satisfy
    #    hubmark[x] = `epoch`
    hub = epoch = zero(V)

    # forget the pockets of the last test
    @inbounds for x in oneto(n)
        parent[x] = x; pocket[x] = 0
    end

    alen = 0

    # improve the graph by the path rule: query every vertex,
    # skipping the sinks that were queried as sources
    @inbounds for i in oneto(nalive)
        s = alist[i]

        nf, alen, tag = lbp_query!(degree, target, invptr, begptr,
            endptr, nbrmark, auxmark, buffer, alist, flow, fstamp,
            nodemark, from, step, queue, parent, arena, pocket, nalive,
            alen, pcap, tag, thr, weight, n, s, i + 1)

        pfree = lbp_join!(hubmark, nbrmark, degree, target, invptr,
            begptr, endptr, limptr, head, prev, next, buffer, apos, nf,
            pfree, hub, epoch, weight, n, s)
    end

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
                nbrmark, auxmark, buffer, work, nodemark, nbrs, alist,
                nalive, pfree, hub, epoch, tag, thr, weight, n, v, w)

            # `v` is represented by `w`; only the pairs that
            # contain `w` can have become inseparable
            parent[v] = w

            nf, alen, tag = lbp_query!(degree, target, invptr, begptr,
                endptr, nbrmark, auxmark, buffer, alist, flow, fstamp,
                nodemark, from, step, queue, parent, arena, pocket,
                nalive, alen, pcap, tag, thr, weight, n, w, 1)

            pfree = lbp_join!(hubmark, nbrmark, degree, target, invptr,
                begptr, endptr, limptr, head, prev, next, buffer, apos,
                nf, pfree, hub, epoch, weight, n, w)

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
    lbp_query!(degree, target, invptr, begptr, endptr, nbrmark,
        auxmark, buffer, alist, flow, fstamp, nodemark, from, step,
        queue, parent, arena, pocket, nalive, alen, pcap, tag, thr,
        weight, n, s, lo)

Find the vertices `y` in `alist[lo:nalive]` that are not adjacent to
the vertex `s` and cannot be separated from `s` by a set of weight
less than `thr`. They are stored in `buffer[1:nf]`.

The candidates are the vertices `y` whose neighbors weigh at least
`thr`, since the neighbors of `y` separate it from `s`. A candidate
is discarded if a stored pocket separates it from `s` (see
[`lbp_pockets!`](@ref)). Otherwise, a flow from `s` to `y` is computed
(see [`lbp_flow!`](@ref)). If it falls short of `thr`, then the search
that failed to augment it found a minimum cut: if the search from `s`
was exhausted, then the vertices that it did not reach are separated
from `s` by the cut; otherwise, the vertices that reach `y` are. These
candidates are discarded, and the pocket is stored (see
[`lbp_store!`](@ref)).

input parameters:
  - `nalive`: length of `alist`
  - `pcap`: capacity of `arena`
  - `thr`: threshold
  - `weight`: vertex weight
  - `n`: number of vertices
  - `s`: source
  - `lo`: first index in `alist`

output parameters:
  - `nf`: number of vertices found

updated parameters:
  - `alen`: used length of `arena`
  - `tag`: a running marker tag
"""
function lbp_query!(
        degree::AbstractVector{V},
        target::Vector{V},
        invptr::Vector{E},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        nbrmark::AbstractVector{Int},
        auxmark::AbstractVector{Int},
        buffer::AbstractVector{V},
        alist::AbstractVector{V},
        flow::Vector{V},
        fstamp::Vector{Int},
        nodemark::AbstractVector{Int},
        from::AbstractVector{Int},
        step::AbstractVector{E},
        queue::AbstractVector{Int},
        parent::AbstractVector{V},
        arena::Vector{V},
        pocket::AbstractVector{Int},
        nalive::Int,
        alen::Int,
        pcap::Int,
        tag::Int,
        thr::V,
        weight::AbstractVector{V},
        n::V,
        s::V,
        lo::Int,
    ) where {V, E}

    # the neighbors of `s` separate it from every other vertex
    @inbounds if degree[s] - weight[s] < thr
        return 0, alen, tag
    end

    # mark the closed neighborhood of `s`
    tag += 1; stag = tag

    @inbounds begin
        nbrmark[s] = stag

        for p in begptr[s]:endptr[s] - one(E)
            nbrmark[target[p]] = stag
        end
    end

    # `nc` is the number of candidates
    nc = 0

    @inbounds for i in lo:nalive
        y = alist[i]

        if nbrmark[y] != stag && degree[y] - weight[y] >= thr
            nc += 1; buffer[nc] = y
        end
    end

    # discard the candidates that a stored pocket
    # separates from `s`
    nc, tag = lbp_pockets!(buffer, parent, arena, pocket, auxmark,
        nodemark, nc, alen, tag, s)

    # `nf` is the number of vertices found
    nf = 0

    @inbounds for i in oneto(nc)
        y = buffer[i]

        if ispositive(y)
            value, tag, ctag, fwd, qlo, qhi = lbp_flow!(weight, target,
                invptr, begptr, endptr, nbrmark, auxmark, flow, fstamp,
                nodemark, from, step, queue, tag, stag, thr, n, s, y)

            if value >= thr
                nf += 1; buffer[nf] = y
            else
                # store the pocket
                alen = lbp_store!(nodemark, queue, arena, pocket, alen,
                    pcap, ctag, fwd, qlo, qhi, n)

                # discard the candidates that the cut separates
                # from `s`
                if fwd
                    for j in i + 1:nc
                        z = buffer[j]

                        if ispositive(z) && nodemark[twice(Int(z)) - 1] != ctag
                            buffer[j] = zero(V)
                        end
                    end
                else
                    for j in i + 1:nc
                        z = buffer[j]

                        if ispositive(z) && nodemark[twice(Int(z)) - 1] == ctag
                            buffer[j] = zero(V)
                        end
                    end
                end
            end
        end
    end

    return nf, alen, tag
end

"""
    lbp_join!(hubmark, stash, degree, target, invptr, begptr, endptr,
        limptr, head, prev, next, buffer, apos, nf, pfree, hub, epoch,
        weight, n, s)

Join the vertex `s` to the vertices `buffer[1:nf]`, in the order of the
list of remaining vertices, so that the order does not depend on the
stored pockets.

input parameters:
  - `nf`: number of vertices
  - `hub`: the neighbors `x` of `hub` satisfy
      hubmark[x] = `epoch`
  - `epoch`: tag for marking vertices
  - `weight`: vertex weight
  - `n`: number of vertices
  - `s`: a vertex

updated parameters:
  - `pfree`: first free arc
"""
function lbp_join!(
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
        buffer::AbstractVector{V},
        apos::AbstractVector{V},
        nf::Int,
        pfree::E,
        hub::V,
        epoch::V,
        weight::AbstractVector{V},
        n::V,
        s::V,
    ) where {T, V, E}

    @inbounds sort!(view(buffer, oneto(nf)); by = y -> apos[y])

    @inbounds for i in oneto(nf)
        y = buffer[i]

        pfree = lbn_addedge!(hubmark, stash, degree, target, invptr,
            begptr, endptr, limptr, head, prev, next, weight, pfree, hub,
            epoch, n, s, y)
    end

    return pfree
end

"""
    lbp_pockets!(buffer, parent, arena, pocket, auxmark, nodemark,
        nc, alen, tag, s)

Discard the candidates `buffer[1:nc]` that a stored pocket separates
from the vertex `s`. The pocket of `s` separates it from every vertex
outside of the pocket. The pocket of a candidate `y` separates `y`
from `s` if `s` is outside of it. The candidates are sorted by pocket,
so that each pocket is read once.

A pocket is stored in `arena` as its length, followed by the vertices
of P (positive) and S (negative). A vertex `x` stands for the vertex
that represents it (see [`lbp_find!`](@ref)). If an S-vertex stands
for a vertex of P, then that vertex has joined S, and the pocket no
longer contains it in P. The vertices of a pocket are marked in
`auxmark`, and the vertices of S in `nodemark`.

input parameters:
  - `alen`: used length of `arena`
  - `s`: source

updated parameters:
  - `nc`: number of candidates
  - `tag`: a running marker tag
"""
function lbp_pockets!(
        buffer::AbstractVector{V},
        parent::AbstractVector{V},
        arena::AbstractVector{V},
        pocket::AbstractVector{Int},
        auxmark::AbstractVector{Int},
        nodemark::AbstractVector{Int},
        nc::Int,
        alen::Int,
        tag::Int,
        s::V,
    ) where {V}

    # if no pocket is stored, stop
    if iszero(alen)
        return nc, tag
    end

    # `q` is the pocket of `s`
    @inbounds q = pocket[s]

    @inbounds if ispositive(q)
        tag += 1; valid = true

        # mark the vertices of the pocket
        for i in q + 1:q + Int(arena[q])
            e = arena[i]; r = lbp_find!(parent, abs(e))

            # if `s` is in S, then the pocket does
            # not contain `s` in P
            if r == s && isnegative(e)
                valid = false; break
            end

            auxmark[r] = tag
        end

        # discard the candidates outside of the pocket
        if valid
            k = 0

            for i in oneto(nc)
                y = buffer[i]

                if auxmark[y] == tag
                    k += 1; buffer[k] = y
                end
            end

            nc = k
        else
            pocket[s] = 0
        end
    end

    # sort the candidates by pocket
    @inbounds sort!(view(buffer, oneto(nc)); by = y -> pocket[y])

    # `k` is the number of candidates left
    k = 0; i = 1

    @inbounds while i <= nc
        # `buffer[i:j - 1]` are the candidates in the pocket `q`
        q = pocket[buffer[i]]; j = i + 1

        while j <= nc && pocket[buffer[j]] == q
            j += 1
        end

        if iszero(q)
            for l in i:j - 1
                k += 1; buffer[k] = buffer[l]
            end
        else
            tag += 1; inside = false

            # mark the vertices of S
            for l in q + 1:q + Int(arena[q])
                e = arena[l]; r = lbp_find!(parent, abs(e))

                if r == s
                    inside = true; break
                end

                if isnegative(e)
                    nodemark[r] = tag
                end
            end

            for l in i:j - 1
                y = buffer[l]

                # if `s` is in the pocket, then keep `y`
                if inside
                    k += 1; buffer[k] = y

                # if `y` is in S, then the pocket no longer
                # contains `y` in P
                elseif nodemark[y] == tag
                    pocket[y] = 0; k += 1; buffer[k] = y
                end
            end
        end

        i = j
    end

    return k, tag
end

"""
    lbp_store!(nodemark, queue, arena, pocket, alen, pcap, ctag,
        fwd, qlo, qhi, n)

Store the pocket found by an exhausted search, if it is smaller than
the pocket of one of the vertices of P. The search marked the nodes
`queue[qlo:qhi]` with `ctag`. If it was the search from the source,
then P consists of the vertices whose out-nodes it reached, and S of
the vertices whose in-nodes it reached and whose out-nodes it did not;
otherwise, the roles of the in-nodes and the out-nodes are reversed.
If the storage is full, every pocket is forgotten.

input parameters:
  - `pcap`: capacity of `arena`
  - `ctag`: tag of the search
  - `fwd`: true if the search was from the source
  - `qlo`, `qhi`: range of `queue`
  - `n`: number of vertices

updated parameters:
  - `alen`: used length of `arena`
"""
function lbp_store!(
        nodemark::AbstractVector{Int},
        queue::AbstractVector{Int},
        arena::Vector{V},
        pocket::AbstractVector{Int},
        alen::Int,
        pcap::Int,
        ctag::Int,
        fwd::Bool,
        qlo::Int,
        qhi::Int,
        n::V,
    ) where {V}

    # `len` is the size of the pocket, and `useful` is true if
    # it is smaller than the pocket of a vertex of P
    len = 0; useful = false

    @inbounds for i in qlo:qhi
        nd = queue[i]

        if iseven(nd) == fwd
            len += 1
        elseif nodemark[lbpother(nd)] != ctag
            len += 1
        end
    end

    @inbounds for i in qlo:qhi
        nd = queue[i]

        if iseven(nd) == fwd
            q = pocket[(nd + 1) >> 1]

            if iszero(q) || arena[q] > len
                useful = true; break
            end
        end
    end

    if !useful
        return alen
    end

    # if the storage is full, forget every pocket
    if alen + len + 1 > pcap
        @inbounds for x in oneto(n)
            pocket[x] = 0
        end

        alen = 0
    end

    if length(arena) < alen + len + 1
        resize!(arena, max(twice(length(arena)), alen + len + 1))
    end

    # `q` is the new pocket
    q = alen + 1
    @inbounds arena[q] = len
    alen = q

    @inbounds for i in qlo:qhi
        nd = queue[i]; x = convert(V, (nd + 1) >> 1)

        if iseven(nd) == fwd
            alen += 1; arena[alen] = x
            r = pocket[x]

            if iszero(r) || arena[r] > len
                pocket[x] = q
            end
        elseif nodemark[lbpother(nd)] != ctag
            alen += 1; arena[alen] = -x
        end
    end

    return alen
end

# the other node of the vertex of the node `nd`
function lbpother(nd::Int)
    return isodd(nd) ? nd + 1 : nd - 1
end

# the vertex that represents the vertex `x`: the vertex that
# it was contracted into, if any
function lbp_find!(parent::AbstractVector{V}, x::V) where {V}
    @inbounds while parent[x] != x
        y = parent[x]; parent[x] = parent[y]; x = y
    end

    return x
end

# the flow on the arc `p`, which is stored in `flow[n + p]`;
# it is zero unless `fstamp[n + p]` is the tag `ftag` of the
# flow
@inline function lbpflow(flow::AbstractVector{V}, fstamp::AbstractVector{Int}, ftag::Int, n, p) where {V}
    i = n + p
    @inbounds return ifelse(fstamp[i] == ftag, flow[i], zero(V))
end

# set the flow on the arc `p`
@inline function lbpflow!(flow::AbstractVector{V}, fstamp::AbstractVector{Int}, ftag::Int, n, p, f) where {V}
    i = n + p
    @inbounds fstamp[i] = ftag; flow[i] = f
    return
end

# the flow through the vertex `x`, which is stored in `flow[x]`
@inline function lbpthrough(flow::AbstractVector{V}, fstamp::AbstractVector{Int}, ftag::Int, x) where {V}
    @inbounds return ifelse(fstamp[x] == ftag, flow[x], zero(V))
end

# set the flow through the vertex `x`
@inline function lbpthrough!(flow::AbstractVector{V}, fstamp::AbstractVector{Int}, ftag::Int, x, f) where {V}
    @inbounds fstamp[x] = ftag; flow[x] = f
    return
end

"""
    lbp_flow!(weight, target, invptr, begptr, endptr, nbrmark,
        auxmark, flow, fstamp, nodemark, from, step, queue, tag, stag,
        thr, n, s, t)

Compute a flow of value `thr` from the vertex `s` to the vertex `t`,
or else a maximum flow, where every vertex other than `s` and `t` has
capacity equal to its weight. The closed neighborhood of `s` is marked
with `stag` in `nbrmark`.

The flow through a vertex `x` is `flow[x]`, and the flow on an arc
`p` is `flow[n + p]`, if the corresponding entry of `fstamp` is the
tag of the flow; otherwise, it is zero. The flow starts with paths of
length two, three, and four, which are sent greedily (see
[`lbp_paths!`](@ref)), and then it is augmented (see
[`lbp_augment!`](@ref)).

input parameters:
  - `stag`: tag of the closed neighborhood of `s`
  - `thr`: threshold
  - `n`: number of vertices
  - `s`: source
  - `t`: sink

output parameters:
  - `value`: value of the flow
  - `ctag`: tag of the exhausted search, if the flow falls short
  - `fwd`: true if the exhausted search is from `s`
  - `qlo`, `qhi`: the nodes of the exhausted search are
                  `queue[qlo:qhi]`

updated parameters:
  - `tag`: a running marker tag
"""
function lbp_flow!(
        weight::AbstractVector{V},
        target::Vector{V},
        invptr::Vector{E},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        nbrmark::AbstractVector{Int},
        auxmark::AbstractVector{Int},
        flow::Vector{V},
        fstamp::Vector{Int},
        nodemark::AbstractVector{Int},
        from::AbstractVector{Int},
        step::AbstractVector{E},
        queue::AbstractVector{Int},
        tag::Int,
        stag::Int,
        thr::V,
        n::V,
        s::V,
        t::V,
    ) where {V, E}

    # the arc storage may have grown
    if length(flow) < n + length(target)
        len = n + length(target); old = length(fstamp)
        resize!(flow, len); resize!(fstamp, len)

        @inbounds for i in old + 1:len
            fstamp[i] = 0
        end
    end

    # `ftag` is the tag of the flow
    tag += 1; ftag = tag

    # send flow along the paths of length two, three, and four
    value, tag = lbp_paths!(weight, target, invptr, begptr, endptr,
        nbrmark, auxmark, flow, fstamp, nodemark, step, ftag, tag, stag,
        thr, n, s, t)

    # augment the flow
    while value < thr
        del, tag, ctag, fwd, qlo, qhi = lbp_augment!(weight, target,
            invptr, begptr, endptr, flow, fstamp, nodemark, from, step,
            queue, ftag, tag, thr - value, n, s, t)

        # if there is no augmenting path, stop
        if iszero(del)
            return value, tag, ctag, fwd, qlo, qhi
        end

        value += del
    end

    return value, tag, 0, false, 1, 0
end

"""
    lbp_paths!(weight, target, invptr, begptr, endptr, nbrmark,
        auxmark, flow, fstamp, nodemark, step, ftag, tag, stag, thr, n,
        s, t)

Send flow greedily from the vertex `s` to the vertex `t` along paths

  - (`s`, `x`, `t`), where `x` is a common neighbor of `s` and `t`
  - (`s`, `a`, `b`, `t`), where `a` is a neighbor of `s` but not `t`
    and `b` is a neighbor of `t` but not `s`
  - (`s`, `a`, `b`, `c`, `t`), where `a` is a neighbor of `s` and `c`
    is a neighbor of `t` but not `s`

until its value is `thr`. On sparse graphs, nearly every augmenting
path has one of these forms. The paths of length four are found by
recording, for each vertex `b` with an unsaturated neighbor `c` of `t`,
one such `c`.

The searches for augmenting paths have not started, so their arrays
are free: `step[2x - 1]` is the arc (`s`, `x`) for each neighbor `x` of
`s`, and the arc (`x`, `t`) for each other neighbor `x` of `t`; the
vertices `b` are marked in `nodemark`, and `step[2b]` is the arc
(`b`, `c`).

input parameters:
  - `ftag`: tag of the flow
  - `stag`: tag of the closed neighborhood of `s`
  - `thr`: threshold
  - `n`: number of vertices
  - `s`: source
  - `t`: sink

output parameters:
  - `value`: value of the flow

updated parameters:
  - `tag`: a running marker tag
"""
function lbp_paths!(
        weight::AbstractVector{V},
        target::AbstractVector{V},
        invptr::AbstractVector{E},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        nbrmark::AbstractVector{Int},
        auxmark::AbstractVector{Int},
        flow::AbstractVector{V},
        fstamp::AbstractVector{Int},
        nodemark::AbstractVector{Int},
        step::AbstractVector{E},
        ftag::Int,
        tag::Int,
        stag::Int,
        thr::V,
        n::V,
        s::V,
        t::V,
    ) where {V, E}

    # `value` is the value of the flow
    value = zero(V)

    # record the arcs (`s`, `x`)
    @inbounds for p in begptr[s]:endptr[s] - one(E)
        step[twice(Int(target[p])) - 1] = p
    end

    # send flow along the paths (`s`, `x`, `t`), and mark the other
    # neighbors of `t`; record the arcs (`x`, `t`)
    tag += 1; ttag = tag

    @inbounds for q in begptr[t]:endptr[t] - one(E)
        x = target[q]; i = twice(Int(x)) - 1

        if nbrmark[x] == stag
            del = min(weight[x], thr - value)
            lbpflow!(flow, fstamp, ftag, n, step[i], del)
            lbpflow!(flow, fstamp, ftag, n, invptr[q], del)
            lbpthrough!(flow, fstamp, ftag, x, del)
            value += del

            if value >= thr
                return value, tag
            end
        else
            auxmark[x] = ttag; step[i] = invptr[q]
        end
    end

    # send flow along the paths (`s`, `a`, `b`, `t`)
    @inbounds for p in begptr[s]:endptr[s] - one(E)
        a = target[p]

        if auxmark[a] != ttag && iszero(lbpthrough(flow, fstamp, ftag, a))
            # `tha` is the flow through `a`
            tha = zero(V)

            for pp in begptr[a]:endptr[a] - one(E)
                b = target[pp]

                if auxmark[b] == ttag
                    # `bt` is the arc (`b`, `t`)
                    bt = step[twice(Int(b)) - 1]
                    thb = lbpthrough(flow, fstamp, ftag, b)
                    del = min(weight[a] - tha, weight[b] - thb, thr - value)

                    if ispositive(del)
                        lbpflow!(flow, fstamp, ftag, n, p, lbpflow(flow, fstamp, ftag, n, p) + del)
                        lbpflow!(flow, fstamp, ftag, n, pp, del)
                        lbpflow!(flow, fstamp, ftag, n, bt, lbpflow(flow, fstamp, ftag, n, bt) + del)
                        lbpthrough!(flow, fstamp, ftag, b, thb + del)
                        tha += del; value += del

                        if value >= thr
                            lbpthrough!(flow, fstamp, ftag, a, tha)
                            return value, tag
                        end

                        if tha >= weight[a]
                            break
                        end
                    end
                end
            end

            if ispositive(tha)
                lbpthrough!(flow, fstamp, ftag, a, tha)
            end
        end
    end

    # for every vertex `b` with an unsaturated neighbor `c` of
    # `t`, record the arc (`b`, `c`) for one such `c`
    tag += 1; btag = tag

    @inbounds for q in begptr[t]:endptr[t] - one(E)
        c = target[q]

        if auxmark[c] == ttag && lbpthrough(flow, fstamp, ftag, c) < weight[c]
            for qq in begptr[c]:endptr[c] - one(E)
                b = target[qq]

                if b != s && b != t && lbpthrough(flow, fstamp, ftag, b) < weight[b]
                    nodemark[b] = btag; step[twice(Int(b))] = invptr[qq]
                end
            end
        end
    end

    # send flow along the paths (`s`, `a`, `b`, `c`, `t`)
    @inbounds for p in begptr[s]:endptr[s] - one(E)
        a = target[p]

        # `tha` is the flow through `a`
        tha = lbpthrough(flow, fstamp, ftag, a)

        if auxmark[a] != ttag && tha < weight[a]
            for pp in begptr[a]:endptr[a] - one(E)
                b = target[pp]

                if nodemark[b] == btag && b != a
                    # `thb` is the flow through `b`
                    thb = lbpthrough(flow, fstamp, ftag, b)

                    if thb < weight[b]
                        # `bc` is the arc (`b`, `c`)
                        bc = step[twice(Int(b))]; c = target[bc]

                        # if `c` is saturated, then look for another
                        if lbpthrough(flow, fstamp, ftag, c) >= weight[c]
                            c = zero(V)

                            for qq in begptr[b]:endptr[b] - one(E)
                                cc = target[qq]

                                if auxmark[cc] == ttag && cc != a && lbpthrough(flow, fstamp, ftag, cc) < weight[cc]
                                    c = cc; bc = qq; step[twice(Int(b))] = qq
                                    break
                                end
                            end

                            if iszero(c)
                                nodemark[b] = 0
                            end
                        end

                        if ispositive(c) && c != a
                            # `ct` is the arc (`c`, `t`), and `thc` is
                            # the flow through `c`
                            ct = step[twice(Int(c)) - 1]
                            thc = lbpthrough(flow, fstamp, ftag, c)
                            del = min(weight[a] - tha, weight[b] - thb, weight[c] - thc, thr - value)

                            lbpflow!(flow, fstamp, ftag, n, p, lbpflow(flow, fstamp, ftag, n, p) + del)
                            lbpflow!(flow, fstamp, ftag, n, pp, lbpflow(flow, fstamp, ftag, n, pp) + del)
                            lbpflow!(flow, fstamp, ftag, n, bc, lbpflow(flow, fstamp, ftag, n, bc) + del)
                            lbpflow!(flow, fstamp, ftag, n, ct, lbpflow(flow, fstamp, ftag, n, ct) + del)
                            tha += del; lbpthrough!(flow, fstamp, ftag, a, tha)
                            lbpthrough!(flow, fstamp, ftag, b, thb + del)
                            lbpthrough!(flow, fstamp, ftag, c, thc + del)
                            value += del

                            if value >= thr
                                return value, tag
                            end

                            if tha >= weight[a]
                                break
                            end
                        end
                    end
                end
            end
        end
    end

    return value, tag
end

"""
    lbp_augment!(weight, target, invptr, begptr, endptr, flow,
        fstamp, nodemark, from, step, queue, ftag, tag, need, n, s, t)

Find an augmenting path from the vertex `s` to the vertex `t`, and send
at most `need` units of flow along it.

The flow is a flow in the split graph, where each vertex `x` other than
`s` and `t` is replaced by an in-node 2`x` - 1 and an out-node 2`x`,
joined by an arc of capacity weight(`x`), and each arc (`a`, `b`) of
the graph runs from the out-node of `a` to the in-node of `b`, with
infinite capacity. A step of a path is labeled by an arc: positive if
it follows the arc forward, negative if it follows it backward, and
zero if it follows the internal arc of a vertex.

The path is found by two breadth-first searches, one from `s` and one
backward from `t`, that take turns, so that each has done about the
same amount of work. The search from `s` uses the queue
`queue[1:tail]`, and the search from `t` uses the queue
`queue[btail:2n]`; a node reached by a search is marked with its tag
in `nodemark`, and `from` and `step` give the node and the step that
it was reached from. The searches stop when they meet. If either is
exhausted, then there is no augmenting path, and the nodes that it
reached form one side of a minimum cut. So the cost of a failed search
is proportional to the smaller side of the cut.

input parameters:
  - `ftag`: tag of the flow
  - `need`: maximum amount of flow
  - `n`: number of vertices
  - `s`: source
  - `t`: sink

output parameters:
  - `del`: the amount of flow sent
  - `ctag`: tag of the exhausted search, if no flow was sent
  - `fwd`: true if the exhausted search is from `s`
  - `qlo`, `qhi`: the nodes of the exhausted search are
                  `queue[qlo:qhi]`

updated parameters:
  - `tag`: a running marker tag
"""
function lbp_augment!(
        weight::AbstractVector{V},
        target::AbstractVector{V},
        invptr::AbstractVector{E},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        flow::AbstractVector{V},
        fstamp::AbstractVector{Int},
        nodemark::AbstractVector{Int},
        from::AbstractVector{Int},
        step::AbstractVector{E},
        queue::AbstractVector{Int},
        ftag::Int,
        tag::Int,
        need::V,
        n::V,
        s::V,
        t::V,
    ) where {V, E}

    # `stag` marks the nodes reached from `s`, and `ttag`
    # marks the nodes that reach `t`
    tag += 1; stag = tag
    tag += 1; ttag = tag

    # `sout` is the out-node of `s`, and `tin` is the
    # in-node of `t`
    sout = twice(Int(s)); tin = twice(Int(t)) - 1; nn = twice(Int(n))

    @inbounds nodemark[sout] = stag; queue[1] = sout
    @inbounds nodemark[tin] = ttag; queue[nn] = tin
    head = tail = 1; bhead = btail = nn

    # `fwork` and `bwork` are the work done by each search
    fwork = bwork = 0

    # `meet` is a node reached by both searches. The search
    # that reached it second reached it from the node `mnode`
    # by the step `mstep`; `mfwd` is true if that was the
    # search from `s`.
    meet = mnode = 0; mstep = zero(E); mfwd = false

    @inbounds while iszero(meet)
        # if the search from `s` is exhausted, stop
        if head > tail
            return zero(V), tag, stag, true, 1, tail

        # if the search from `t` is exhausted, stop
        elseif bhead < btail
            return zero(V), tag, ttag, false, btail, nn

        # search from `s`
        elseif fwork <= bwork
            nd = queue[head]; head += 1
            x = convert(V, (nd + 1) >> 1)
            fwork += 1

            if iseven(nd)
                # `nd` is the out-node of `x`: follow the arcs (`x`, `b`)
                pbeg = begptr[x]; pend = endptr[x]
                fwork += Int(pend - pbeg)

                for p in pbeg:pend - one(E)
                    b = target[p]

                    if b != s
                        bin = twice(Int(b)) - 1

                        if nodemark[bin] == ttag
                            meet = bin; mnode = nd; mstep = p; mfwd = true
                            break
                        elseif nodemark[bin] != stag
                            nodemark[bin] = stag; from[bin] = nd; step[bin] = p
                            tail += 1; queue[tail] = bin
                        end
                    end
                end

                # undo flow through `x`
                if iszero(meet) && x != s && ispositive(lbpthrough(flow, fstamp, ftag, x))
                    xin = nd - 1

                    if nodemark[xin] == ttag
                        meet = xin; mnode = nd; mstep = zero(E); mfwd = true
                    elseif nodemark[xin] != stag
                        nodemark[xin] = stag; from[xin] = nd; step[xin] = zero(E)
                        tail += 1; queue[tail] = xin
                    end
                end
            else
                # `nd` is the in-node of `x`
                thx = lbpthrough(flow, fstamp, ftag, x)

                # send flow through `x`
                if thx < weight[x]
                    xout = nd + 1

                    if nodemark[xout] == ttag
                        meet = xout; mnode = nd; mstep = zero(E); mfwd = true
                    elseif nodemark[xout] != stag
                        nodemark[xout] = stag; from[xout] = nd; step[xout] = zero(E)
                        tail += 1; queue[tail] = xout
                    end
                end

                # undo flow on the arcs (`a`, `x`)
                if iszero(meet) && ispositive(thx)
                    pbeg = begptr[x]; pend = endptr[x]
                    fwork += Int(pend - pbeg)

                    for q in pbeg:pend - one(E)
                        p = invptr[q]

                        if ispositive(lbpflow(flow, fstamp, ftag, n, p))
                            aout = twice(Int(target[q]))

                            if nodemark[aout] == ttag
                                meet = aout; mnode = nd; mstep = -p; mfwd = true
                                break
                            elseif nodemark[aout] != stag
                                nodemark[aout] = stag; from[aout] = nd; step[aout] = -p
                                tail += 1; queue[tail] = aout
                            end
                        end
                    end
                end
            end

        # search from `t`
        else
            nd = queue[bhead]; bhead -= 1
            y = convert(V, (nd + 1) >> 1)
            bwork += 1

            if isodd(nd)
                # `nd` is the in-node of `y`: follow the arcs (`x`, `y`)
                pbeg = begptr[y]; pend = endptr[y]
                bwork += Int(pend - pbeg)

                for q in pbeg:pend - one(E)
                    x = target[q]

                    if x != t
                        xout = twice(Int(x)); p = invptr[q]

                        if nodemark[xout] == stag
                            meet = xout; mnode = nd; mstep = p; mfwd = false
                            break
                        elseif nodemark[xout] != ttag
                            nodemark[xout] = ttag; from[xout] = nd; step[xout] = p
                            btail -= 1; queue[btail] = xout
                        end
                    end
                end

                # undo flow through `y`
                if iszero(meet) && y != t && ispositive(lbpthrough(flow, fstamp, ftag, y))
                    yout = nd + 1

                    if nodemark[yout] == stag
                        meet = yout; mnode = nd; mstep = zero(E); mfwd = false
                    elseif nodemark[yout] != ttag
                        nodemark[yout] = ttag; from[yout] = nd; step[yout] = zero(E)
                        btail -= 1; queue[btail] = yout
                    end
                end
            else
                # `nd` is the out-node of `y`
                thy = lbpthrough(flow, fstamp, ftag, y)

                # send flow through `y`
                if thy < weight[y]
                    yin = nd - 1

                    if nodemark[yin] == stag
                        meet = yin; mnode = nd; mstep = zero(E); mfwd = false
                    elseif nodemark[yin] != ttag
                        nodemark[yin] = ttag; from[yin] = nd; step[yin] = zero(E)
                        btail -= 1; queue[btail] = yin
                    end
                end

                # undo flow on the arcs (`y`, `b`)
                if iszero(meet) && ispositive(thy)
                    pbeg = begptr[y]; pend = endptr[y]
                    bwork += Int(pend - pbeg)

                    for p in pbeg:pend - one(E)
                        if ispositive(lbpflow(flow, fstamp, ftag, n, p))
                            bin = twice(Int(target[p])) - 1

                            if nodemark[bin] == stag
                                meet = bin; mnode = nd; mstep = -p; mfwd = false
                                break
                            elseif nodemark[bin] != ttag
                                nodemark[bin] = ttag; from[bin] = nd; step[bin] = -p
                                btail -= 1; queue[btail] = bin
                            end
                        end
                    end
                end
            end
        end
    end

    # the path runs from `sout` to `meet` by the steps of the
    # search from `s`, and from `meet` to `tin` by the steps of
    # the search from `t`, except that the step to or from `meet`
    # of the search that reached it second is (`mnode`, `mstep`)
    del = need

    # find the bottleneck
    if mfwd
        del = min(del, lbpresidual(weight, flow, fstamp, ftag, n, meet, mstep)); nd = mnode
    else
        nd = meet
    end

    @inbounds while nd != sout
        del = min(del, lbpresidual(weight, flow, fstamp, ftag, n, nd, step[nd])); nd = from[nd]
    end

    if mfwd
        nd = meet
    else
        del = min(del, lbpresidual(weight, flow, fstamp, ftag, n, mnode, mstep)); nd = mnode
    end

    @inbounds while nd != tin
        del = min(del, lbpresidual(weight, flow, fstamp, ftag, n, from[nd], step[nd])); nd = from[nd]
    end

    # augment
    if mfwd
        lbp_push!(flow, fstamp, ftag, n, meet, mstep, del); nd = mnode
    else
        nd = meet
    end

    @inbounds while nd != sout
        lbp_push!(flow, fstamp, ftag, n, nd, step[nd], del); nd = from[nd]
    end

    if mfwd
        nd = meet
    else
        lbp_push!(flow, fstamp, ftag, n, mnode, mstep, del); nd = mnode
    end

    @inbounds while nd != tin
        lbp_push!(flow, fstamp, ftag, n, from[nd], step[nd], del); nd = from[nd]
    end

    return del, tag, 0, false, 1, 0
end

# the residual capacity of the step `a` to the node `nd`
function lbpresidual(
        weight::AbstractVector{V},
        flow::AbstractVector{V},
        fstamp::AbstractVector{Int},
        ftag::Int,
        n::V,
        nd::Int,
        a::E,
    ) where {V, E}

    if iszero(a)
        x = convert(V, (nd + 1) >> 1)
        thx = lbpthrough(flow, fstamp, ftag, x)
        @inbounds res = iseven(nd) ? weight[x] - thx : thx
    elseif isnegative(a)
        res = lbpflow(flow, fstamp, ftag, n, -a)
    else
        res = typemax(V)
    end

    return res
end

# send `del` units of flow along the step `a` to the node `nd`
function lbp_push!(
        flow::AbstractVector{V},
        fstamp::AbstractVector{Int},
        ftag::Int,
        n::V,
        nd::Int,
        a::E,
        del::V,
    ) where {V, E}

    if iszero(a)
        x = (nd + 1) >> 1
        thx = lbpthrough(flow, fstamp, ftag, x)
        lbpthrough!(flow, fstamp, ftag, x, iseven(nd) ? thx + del : thx - del)
    elseif ispositive(a)
        lbpflow!(flow, fstamp, ftag, n, a, lbpflow(flow, fstamp, ftag, n, a) + del)
    else
        lbpflow!(flow, fstamp, ftag, n, -a, lbpflow(flow, fstamp, ftag, n, -a) - del)
    end

    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::LBP{S}) where {S}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "LBP{$S}")
    return nothing
end
