# status flags used by `asr`
const ASR_QUEUED = 0x01 # the vertex is in the work queue
const ASR_PARKED = 0x02 # the vertex failed a test because of `width`
const ASR_DELETE = 0x04 # the vertex has been eliminated

# the connector search (see `asr_connector!`) for a vertex of
# degree d scans at most ASR_CONNECTOR_BASE + ASR_CONNECTOR_DEGREE × d
# arcs before giving up
const ASR_CONNECTOR_BASE = 1 << 16
const ASR_CONNECTOR_DEGREE = 1 << 12

function asr(weights::AbstractVector{W}, graph::AbstractGraph, width::Number) where {W <: Number}
    return asr(weights, graph, convert(W, width))
end

function asr(weights::AbstractVector{W}, graph::AbstractGraph{V}, width::W) where {W <: Number, V <: Integer}
    @assert nv(graph) <= length(weights)

    E = etype(graph); n = nv(graph); m = de(graph); nn = n + one(V)

    # `totdeg` is the total weight of the
    # vertices in the graph
    totdeg = zero(W)

    @inbounds for v in oneto(n)
        totdeg += weights[v]
    end

    weight = weights
    degree = FVector{W}(undef, n)
    number = FVector{V}(undef, n)
    fillin = FVector{Int}(undef, n)
    status = FVector{UInt8}(undef, n)
    marker = FVector{Int}(undef, n)
    marker2 = FVector{Int}(undef, n)
    source = FVector{V}(undef, m)
    target = FVector{V}(undef, m)
    begptr = FVector{E}(undef, nn)
    endptr = FVector{E}(undef, n)
    invptr = FVector{E}(undef, m)
    stack1 = FVector{V}(undef, n)
    stack4 = FVector{V}(undef, n)
    stack5 = FVector{V}(undef, n)
    stack7 = FVector{V}(undef, n)
    stack8 = FVector{V}(undef, n)
    stack2 = FVector{V}(undef, n)
    tmpptr = FVector{E}(undef, nn)
    tgt = FVector{V}(undef, m)

    return asr_impl!(
        weight, degree, number, fillin, status, marker, marker2, source,
        target, begptr, endptr, invptr, stack1, stack4, stack5, stack7,
        stack8, stack2, tmpptr, tgt, totdeg, width, graph)
end

"""
    asr_impl!(...)

Safe Reduction Rules for Weighted Treewidth
Eijkhof, Bodlaender, and Koster

Preprocess a graph by applying two *safe* reduction rules to
vertices of any degree.
  - simplicial: N(v) is a clique
  - almost simplicial: N(v) \\ {u} is a clique for a neighbor u
    of v, with weight(u) ≤ weight(v) and weight(N[v]) ≤ `width`.
    Then v is contracted into u.

If weight(N[v]) > `width`, the almost simplicial rule may still
apply: a *connector* is a connected set Q of vertices outside N[v],
each at least as heavy as u, that is adjacent to u and to every
vertex of N(v) \\ N[u]. Contracting Q into u makes N[v] a clique
in a weighted minor of the graph, so weight(N[v]) is a lower bound
for the treewidth: `width` is raised to it, and v is contracted
into u (see `asr_connector!`).

Each vertex v stores its fill-in: the number of missing edges in
N(v), computed by [`sr_fillin!`](@ref). A vertex is simplicial if
its fill-in is zero, and it is almost simplicial only if its
fill-in is less than its degree, since every missing edge has u as
an endpoint. When a vertex is contracted, the fill-in of the
remaining vertices is updated using the method of Wing and Huang,
as in `sr_fillin!`: every new edge {u, x} decreases the fill-in of
each common neighbor of u and x.

The algorithm is driven by a work queue: a vertex is tested
again only when its fill-in, degree, or neighborhood changes, or
when the lower bound `width` increases. The almost simplicial test
searches for u directly: u must be an endpoint of every missing
edge in N(v), so it suffices to find one missing edge {a, b} and
count the missing edges at a and b.

The graph is stored as a quotient graph, as in [`pr3_impl!`](@ref).
Contracting an almost simplicial vertex v into u turns v into a
supernode of u, so no new storage is needed.

The state of the algorithm is a collection of arrays and scalars.

  - quotient graph (see [`pr3_impl!`](@ref)):
    - `source`: the owner of an arc slot
    - `target`: the target vertex of an arc
    - `begptr`: the first arc slot owned by a vertex
    - `endptr`: one past the last arc incident to a vertex
    - `invptr`: the reverse of an arc
  - vertex data:
    - `weight`: vertex weight
    - `degree`: weighted degree (weight of the closed neighborhood)
    - `number`: degree
    - `fillin`: number of missing edges in the neighborhood
    - `status`: queue membership and elimination flags
    - `marker`, `marker2`: marker arrays
  - work queues:
    - `stack1`: vertices whose fill-in, degree, or neighborhood
                has changed
    - `stack7`: vertices whose last test failed because of `width`
  - miscellaneous:
    - `stack2`: common neighbors buffered during contraction, and
                the seeds of a connector search
    - `stack4`: eliminated vertices
    - `stack5`: traversal stack
    - `stack8`: the neighbors of a vertex, and the queue of a
                connector search
    - `arcs`: the arcs of a vertex
  - scalars:
    - `width`: treewidth lower bound
    - `parked`: the value of `width` when the first vertex was parked
    - `tag`: a running marker tag
    - `hi1`, `hi4`, `hi7`: the heights of `stack1`, `stack4`, `stack7`

input parameters:
  - `totdeg`: total vertex weight
  - `graph`: input graph

output parameters:
  - `kernel`: reduced graph
  - `stack4`: stack of eliminated vertices
  - `inject`: mapping from the vertices of `kernel` to the vertices
              of `graph`
  - `width`: treewidth lower bound
"""
function asr_impl!(
        weight::AbstractVector{W},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        fillin::AbstractVector{Int},
        status::AbstractVector{UInt8},
        marker::AbstractVector{Int},
        marker2::AbstractVector{Int},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        stack1::AbstractVector{V},
        stack4::AbstractVector{V},
        stack5::AbstractVector{V},
        stack7::AbstractVector{V},
        stack8::AbstractVector{V},
        stack2::AbstractVector{V},
        tmpptr::AbstractVector{E},
        tgt::AbstractVector{V},
        totdeg::W,
        width::W,
        graph::AbstractGraph{V},
    ) where {W, V, E}
    n = nv(graph)

    @assert n <= length(weight)
    @assert n <= length(degree)
    @assert n <= length(number)
    @assert n <= length(fillin)
    @assert n <= length(status)
    @assert n <= length(marker)
    @assert n <= length(marker2)
    @assert de(graph) <= length(source)
    @assert de(graph) <= length(target)
    @assert n < length(begptr)
    @assert n <= length(endptr)
    @assert de(graph) <= length(invptr)
    @assert n <= length(stack1)
    @assert n <= length(stack4)
    @assert n <= length(stack5)
    @assert n <= length(stack7)
    @assert n <= length(stack8)
    @assert n <= length(stack2)
    @assert n < length(tmpptr)
    @assert de(graph) <= length(tgt)

    # initialize the quotient graph, the fill-in, and the work queue
    width, parked, hi1 = asr_init!(weight, degree, number, fillin, status,
        marker, marker2, source, target, begptr, endptr, invptr, stack1,
        stack4, stack5, tmpptr, tgt, totdeg, width, graph)

    # apply reduction rules until no more apply
    width, hi4 = asr_loop!(weight, degree, number, fillin, status, marker,
        marker2, source, target, begptr, endptr, invptr, stack1, stack4,
        stack5, stack7, stack8, tmpptr, stack2, width, parked, 0, hi1,
        zero(V), zero(V))

    # construct the reduced graph
    m, n = pr3_make!(stack4, stack5, target, begptr, endptr, invptr, stack1,
        number, tmpptr, tgt, hi4, n)

    # `kernel` is the reduced graph
    kernel = BipartiteGraph(n, n, m, tmpptr, tgt)
    return kernel, stack4, stack1, width
end

function asr_init!(
        weight::AbstractVector{W},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        fillin::AbstractVector{Int},
        status::AbstractVector{UInt8},
        marker::AbstractVector{Int},
        marker2::AbstractVector{Int},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        stack1::AbstractVector{V},
        stack4::AbstractVector{V},
        stack5::AbstractVector{V},
        tmpptr::AbstractVector{E},
        tgt::AbstractVector{V},
        totdeg::W,
        width::W,
        graph::AbstractGraph{V},
    ) where {W, V, E}
    # `n` is the number of vertices in the graph
    n = nv(graph); nn = n + one(V)

    # `mindeg` is the minimum weighted degree
    mindeg = totdeg

    # `sorted` is true if every neighborhood is sorted
    sorted = true

    # `p` is the current arc
    p = one(E)

    # copy the graph into the arrays `source` and `target`
    @inbounds for v in vertices(graph)
        status[v] = zero(UInt8)
        marker[v] = marker2[v] = 0
        begptr[v] = p

        # `deg` is the weighted degree of `v`
        deg = weight[v]

        # `num` is the degree of `v`
        num = zero(V)

        # `prv` is the previous neighbor of `v`
        prv = zero(V)

        for w in neighbors(graph, v)
            # ignore self loops
            if v != w
                # `p` is the arc (`v`, `w`)
                source[p] = v; target[p] = w; p += one(E)

                # check if the neighborhood of `v` is sorted
                sorted = sorted && prv < w; prv = w

                deg += weight[w]
                num += one(V)
            end
        end

        endptr[v] = p
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
                w = target[p]; p += one(E)
                q = tmpptr[w]; invptr[q] = v; tmpptr[w] = q + one(E)
            end
        end

        copyto!(target, begptr[begin], invptr, begptr[begin], begptr[nn] - one(E))
    end

    # compute the reverse of every arc
    @inbounds for v in vertices(graph)
        tmpptr[v] = begptr[v]
    end

    @inbounds for v in vertices(graph)
        p = begptr[v]; pend = endptr[v]

        while p < pend
            w = target[p]
            q = tmpptr[w]; invptr[p] = q; tmpptr[w] = q + one(E)
            p += one(E)
        end
    end

    # compute the fill-in of each vertex, using `stack5`, `stack4`,
    # `tmpptr`, and `tgt` as working storage
    sr_fillin!(fillin, stack5, stack4, tmpptr, tgt,
        number, target, begptr, endptr, n)

    # the weighted treewidth of the input graph is no less
    # than its minimum weighted degree
    width = max(width, mindeg); parked = width

    # `hi1` is the height of the work queue
    hi1 = zero(V)

    # add vertices to the work queue in reverse order, so
    # that they are tested in increasing order
    @inbounds for v in reverse(vertices(graph))
        hi1 = asr_touch!(status, stack1, hi1, v)
    end

    return width, parked, hi1
end

function asr_loop!(
        weight::AbstractVector{W},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        fillin::AbstractVector{Int},
        status::AbstractVector{UInt8},
        marker::AbstractVector{Int},
        marker2::AbstractVector{Int},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        stack1::AbstractVector{V},
        stack4::AbstractVector{V},
        stack5::AbstractVector{V},
        stack7::AbstractVector{V},
        stack8::AbstractVector{V},
        arcs::AbstractVector{E},
        stack2::AbstractVector{V},
        width::W,
        parked::W,
        tag::Int,
        hi1::V,
        hi4::V,
        hi7::V,
    ) where {W, V, E}
    tol = tolerance(W)

    @inbounds while true
        if ispositive(hi1)
            hi1, v = pr3_stack_pop!(stack1, hi1)
            flag = status[v]; status[v] = flag & ~ASR_QUEUED

            if iszero(flag & ASR_DELETE)
                fil = fillin[v]

                if iszero(fil)
                    # `v` is simplicial
                    width, hi4, hi1 = asr_simplicial!(weight, degree, number,
                        fillin, status, source, target, begptr, endptr, invptr,
                        stack1, stack4, stack5, stack8, arcs, width, hi4, hi1, v)
                elseif fil < Int(number[v])
                    # `v` may be almost simplicial
                    if degree[v] < width + tol
                        u, tag = asr_friend!(weight, number, fillin, marker,
                            marker2, stack5, stack8, arcs, target, begptr,
                            endptr, invptr, tag, v)

                        if ispositive(u)
                            hi4, hi1, tag = asr_contract!(weight, degree,
                                number, fillin, status, marker, marker2, source,
                                target, begptr, endptr, invptr, stack1, stack4,
                                stack5, stack8, arcs, stack2, hi4, hi1, tag, v, u)
                        end
                    else
                        # the lower bound is too small to certify `v`;
                        # search for a connector instead
                        u, tag = asr_friend!(weight, number, fillin, marker,
                            marker2, stack5, stack8, arcs, target, begptr,
                            endptr, invptr, tag, v)

                        found = false

                        if ispositive(u)
                            found, tag = asr_connector!(weight, number, marker,
                                marker2, stack2, stack5, stack8, target, begptr,
                                endptr, invptr, tag, v, u)
                        end

                        if found
                            # the clique N[`v`] is a minor: update the
                            # lower bound
                            width = max(width, degree[v])

                            hi4, hi1, tag = asr_contract!(weight, degree,
                                number, fillin, status, marker, marker2, source,
                                target, begptr, endptr, invptr, stack1, stack4,
                                stack5, stack8, arcs, stack2, hi4, hi1, tag, v, u)
                        else
                            parked, hi7 = asr_park!(status, stack7, parked, hi7, width, v)
                        end
                    end
                end
            end
        elseif ispositive(hi7) && parked < width
            # the lower bound has increased: re-test every
            # vertex that failed a test because of it
            for i in oneto(hi7)
                v = stack7[i]
                status[v] &= ~ASR_PARKED
                hi1 = asr_touch!(status, stack1, hi1, v)
            end

            hi7 = zero(V)
        elseif ispositive(hi7)
            # the remaining graph is a minor of the input graph, so
            # its minimum weighted degree is a lower bound
            mindeg = typemax(W); alive = false

            for v in eachindex(status)
                if iszero(status[v] & ASR_DELETE)
                    mindeg = min(mindeg, degree[v]); alive = true
                end
            end

            if alive && width < mindeg
                width = mindeg
            else
                break
            end
        else
            break
        end
    end

    return width, hi4
end

# add `v` to the work queue
function asr_touch!(status::AbstractVector{UInt8}, stack1::AbstractVector{V}, hi1::V, v::V) where {V}
    @inbounds flag = status[v]

    @inbounds if iszero(flag & (ASR_DELETE | ASR_QUEUED))
        status[v] = flag | ASR_QUEUED
        hi1 = pr3_stack_add!(stack1, hi1, v)
    end

    return hi1
end

# a test of `v` failed because of the lower bound
function asr_park!(status::AbstractVector{UInt8}, stack7::AbstractVector{V}, parked::W, hi7::V, width::W, v::V) where {W, V}
    @inbounds flag = status[v]

    @inbounds if iszero(flag & ASR_PARKED)
        if iszero(hi7)
            parked = width
        end

        status[v] = flag | ASR_PARKED
        hi7 = pr3_stack_add!(stack7, hi7, v)
    end

    return parked, hi7
end

# write the neighbors of `v` to `stack8` and the corresponding
# arcs to `arcs`; returns the number of neighbors
function asr_neighbors!(
        stack8::AbstractVector{V},
        arcs::AbstractVector{E},
        stack5::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        v::V,
    ) where {V, E}
    k = pr3_reach!(zero(V), stack5, target, begptr, endptr, invptr, v) do k, p, w
        k += one(V)
        @inbounds stack8[k] = w; arcs[k] = p
        return k
    end

    return k
end

# returns the number of vertices `w` reachable by `x`
# with `marker[w] == tag`
function asr_count(
        marker::AbstractVector{Int},
        stack5::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        x::V,
        tag::Int,
    ) where {V, E}
    c = pr3_reach!(0, stack5, target, begptr, endptr, invptr, x) do c, _, w
        @inbounds if marker[w] == tag
            c += 1
        end

        return c
    end

    return c
end

# returns the number of vertices `w` reachable by `x` with
# `marker[w] == tag`, and sets `marker2[w] = tag2` for each one
function asr_count_mark(
        marker::AbstractVector{Int},
        marker2::AbstractVector{Int},
        stack5::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        x::V,
        tag::Int,
        tag2::Int,
    ) where {V, E}
    c = pr3_reach!(0, stack5, target, begptr, endptr, invptr, x) do c, _, w
        @inbounds if marker[w] == tag
            c += 1; marker2[w] = tag2
        end

        return c
    end

    return c
end

# eliminate a simplicial vertex `v`
function asr_simplicial!(
        weight::AbstractVector{W},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        fillin::AbstractVector{Int},
        status::AbstractVector{UInt8},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        stack1::AbstractVector{V},
        stack4::AbstractVector{V},
        stack5::AbstractVector{V},
        stack8::AbstractVector{V},
        arcs::AbstractVector{E},
        width::W,
        hi4::V,
        hi1::V,
        v::V,
    ) where {W, V, E}
    # add `v` to the stack of eliminated vertices
    @inbounds status[v] |= ASR_DELETE
    hi4 = pr3_stack_add!(stack4, hi4, v)

    # `v` is simplicial: update the lower bound
    @inbounds width = max(width, degree[v])

    # `d` is the degree of `v`
    @inbounds d = number[v]; wgt = weight[v]

    k = asr_neighbors!(stack8, arcs, stack5, target, begptr, endptr, invptr, v)

    @inbounds for i in oneto(k)
        w = stack8[i]; p = arcs[i]

        # remove `v` from the neighborhood of `w`
        pr3_reach_del!(source, target, endptr, invptr, invptr[p])

        # the missing edges in N(`w`) incident to `v` join
        # `v` to the neighbors of `w` outside N[`v`]
        fillin[w] -= Int(number[w] - d)
        number[w] -= one(V)
        degree[w] -= wgt

        hi1 = asr_touch!(status, stack1, hi1, w)
    end

    return width, hi4, hi1
end

# find a neighbor `u` of `v` such that every missing edge in N(`v`) is
# incident to `u` and weight(`u`) ≤ weight(`v`); returns zero if there
# is none. The fill-in of `v` is positive.
function asr_friend!(
        weight::AbstractVector{W},
        number::AbstractVector{V},
        fillin::AbstractVector{Int},
        marker::AbstractVector{Int},
        marker2::AbstractVector{Int},
        stack5::AbstractVector{V},
        stack8::AbstractVector{V},
        arcs::AbstractVector{E},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        tag::Int,
        v::V,
    ) where {W, V, E}
    tol = tolerance(W)

    @inbounds fil = fillin[v]; d = Int(number[v]); wgt = weight[v]

    # mark the neighbors of `v` with `ntag`
    tag += 1; ntag = tag
    k = asr_neighbors!(stack8, arcs, stack5, target, begptr, endptr, invptr, v)

    @inbounds for i in oneto(k)
        marker[stack8[i]] = ntag
    end

    # search for a missing edge {`x`, `y`}, scanning
    # small neighborhoods first
    @inbounds for pass in 1:2, i in oneto(k)
        x = stack8[i]

        # in the first pass, skip vertices of large degree; in
        # the second pass, skip the vertices scanned in the first
        (Int(number[x]) <= twice(d)) == isone(pass) || continue

        # `miss` is the number of missing edges incident to `x`
        tag += 1; tag2 = tag
        miss = d - 1 - asr_count_mark(marker, marker2, stack5, target,
            begptr, endptr, invptr, x, ntag, tag2)

        iszero(miss) && continue

        if miss == fil
            # `x` is incident to every missing edge
            weight[x] < wgt + tol && return x, tag

            # if there is one missing edge {`x`, `y`},
            # then `y` is also incident to it
            if isone(fil)
                y = asr_unmarked(marker2, stack8, x, k, tag2)
                weight[y] < wgt + tol && return y, tag
            end

            return zero(V), tag
        elseif isone(miss)
            # `x` is not incident to every missing edge, so the
            # other endpoint `y` of its missing edge must be
            y = asr_unmarked(marker2, stack8, x, k, tag2)

            if weight[y] < wgt + tol && d - 1 - asr_count(marker, stack5,
                    target, begptr, endptr, invptr, y, ntag) == fil
                return y, tag
            end

            return zero(V), tag
        else
            # `x` is incident to two missing edges but not
            # to every missing edge
            return zero(V), tag
        end
    end

    return zero(V), tag
end

# returns the neighbor `y` != `x` of `v` (stored in `stack8`) with
# `marker2[y] != tag2`
function asr_unmarked(marker2::AbstractVector{Int}, stack8::AbstractVector{V}, x::V, k::V, tag2::Int) where {V}
    @inbounds for i in oneto(k)
        y = stack8[i]

        if y != x && marker2[y] != tag2
            return y
        end
    end

    return zero(V)
end

# returns true if there is a connector for the almost simplicial
# vertex `v` with friend `u`: a connected set Q of vertices outside
# N[`v`], each at least as heavy as `u`, such that some vertex of Q
# is adjacent to `u` and every vertex of N(`v`) not adjacent to `u`
# has a neighbor in Q. Contracting Q into `u` makes N[`v`] a clique
# in a weighted minor of the graph, so weight(N[`v`]) is a lower
# bound. The search gives up after a fixed number of steps.
#
# On entry, `stack8[1:number[v]]` holds N(`v`). On exit, `stack8`,
# `stack2`, `marker`, and `marker2` are overwritten.
function asr_connector!(
        weight::AbstractVector{W},
        number::AbstractVector{V},
        marker::AbstractVector{Int},
        marker2::AbstractVector{Int},
        stack2::AbstractVector{V},
        stack5::AbstractVector{V},
        stack8::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        tag::Int,
        v::V,
        u::V,
    ) where {W, V, E}
    tol = tolerance(W)

    # `d` is the degree of `v`
    @inbounds d = Int(number[v])

    # `uwgt` is the weight of `u`: every vertex of Q
    # must be at least this heavy
    @inbounds uwgt = weight[u]

    # mark N[`v`] with `ntag`
    tag += 1; ntag = tag
    @inbounds marker[v] = ntag

    @inbounds for i in oneto(d)
        marker[stack8[i]] = ntag
    end

    # mark N(`u`) with `utag`
    tag += 1; utag = tag

    pr3_reach!(nothing, stack5, target, begptr, endptr, invptr, u) do _, _, y
        @inbounds marker2[y] = utag
        return
    end

    # mark X = N(`v`) minus N[`u`] with `xtag`; later in the search,
    # a vertex `y` is in X if and only if `marker2[y] >= xtag`
    tag += 1; xtag = tag

    # `nx` is the size of X
    nx = 0

    @inbounds for i in oneto(d)
        y = stack8[i]

        if y != u && marker2[y] != utag
            marker2[y] = xtag; nx += 1
        end
    end

    iszero(nx) && return false, tag

    # the seeds are the neighbors of `u` outside N[`v`] that
    # are at least as heavy as `u`; store them in `stack2`
    ns = pr3_reach!(0, stack5, target, begptr, endptr, invptr, u) do ns, _, z
        @inbounds if marker[z] != ntag && weight[z] > uwgt - tol
            ns += 1; stack2[ns] = z
        end

        return ns
    end

    # `budget` is the number of arcs the search may scan
    budget = ASR_CONNECTOR_BASE + ASR_CONNECTOR_DEGREE * d

    # `visits` is the number of arcs scanned so far
    visits = 0

    # vertices visited by the search are marked with `vtag`
    tag += 1; vtag = tag

    # for each seed `z`...
    @inbounds for s in oneto(ns)
        z = stack2[s]

        # if `z` was reached from an earlier seed, its
        # region has already been searched
        marker[z] == vtag && continue

        # search the region of `z`: the heavy vertices outside
        # N[`v`] connected to `z` through heavy vertices. The
        # vertices of X that it covers are marked with `rtag`.
        tag += 1; rtag = tag

        # `covered` is the number of vertices of X covered
        # by the region
        covered = 0

        # `stack8[hd:tl]` is the breadth-first search queue
        hd = tl = 1; stack8[1] = z; marker[z] = vtag

        while hd <= tl
            q = stack8[hd]; hd += 1

            tl, covered, visits = pr3_reach!((tl, covered, visits), stack5,
                    target, begptr, endptr, invptr, q) do (tl, covered, visits), _, y
                visits += 1

                @inbounds if marker2[y] >= xtag
                    # `y` is in X: it is covered by the region
                    if marker2[y] != rtag
                        marker2[y] = rtag; covered += 1
                    end
                elseif marker[y] != ntag && marker[y] != vtag && weight[y] > uwgt - tol
                    # `y` is a heavy vertex outside N[`v`]
                    marker[y] = vtag; tl += 1; stack8[tl] = y
                end

                return (tl, covered, visits)
            end

            # the region covers X: it is a connector
            covered == nx && return true, tag

            # the search has run out of steps
            visits > budget && return false, tag
        end
    end

    return false, tag
end

# contract an almost simplicial vertex `v` into its neighbor `u`
function asr_contract!(
        weight::AbstractVector{W},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        fillin::AbstractVector{Int},
        status::AbstractVector{UInt8},
        marker::AbstractVector{Int},
        marker2::AbstractVector{Int},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        stack1::AbstractVector{V},
        stack4::AbstractVector{V},
        stack5::AbstractVector{V},
        stack8::AbstractVector{V},
        arcs::AbstractVector{E},
        stack2::AbstractVector{V},
        hi4::V,
        hi1::V,
        tag::Int,
        v::V,
        u::V,
    ) where {W, V, E}
    # add `v` to the stack of eliminated vertices
    @inbounds status[v] |= ASR_DELETE
    hi4 = pr3_stack_add!(stack4, hi4, v)

    @inbounds d = number[v]; wgt = weight[v]; uwgt = weight[u]

    # `stack8[1:k]` are the neighbors of `v`, and
    # `arcs[1:k]` are the corresponding arcs
    k = asr_neighbors!(stack8, arcs, stack5, target, begptr, endptr, invptr, v)

    # mark the neighbors of `u` with `tag`
    tag += 1

    pr3_reach!(nothing, stack5, target, begptr, endptr, invptr, u) do _, _, w
        @inbounds marker[w] = tag
        return
    end

    # Wing-Huang updates: add the edges {`u`, `x`}, where `x` is a
    # neighbor of `v` not adjacent to `u`, one at a time. Each edge
    # decreases the fill-in of every common neighbor of `u` and `x`.
    # The vertices `x` are moved to the front of `stack8`.
    @inbounds unum = Int(number[u]); uwgtadd = zero(W); nx = zero(V)

    @inbounds for i in oneto(k)
        x = stack8[i]

        if x != u && marker[x] != tag
            # `c` is the number of common neighbors of `u` and `x`;
            # they are buffered in `stack2[1:c]` to be re-queued
            # after the traversal (so the traversal reads nothing
            # that `asr_touch!` would change)
            c = pr3_reach!(0, stack5, target, begptr, endptr, invptr, x) do c, _, t
                @inbounds if marker[t] == tag
                    c += 1; fillin[t] -= 1; stack2[c] = t
                end

                return c
            end

            for j in oneto(c)
                hi1 = asr_touch!(status, stack1, hi1, stack2[j])
            end

            # the neighbors of `u` not adjacent to `x`, and the
            # neighbors of `x` not adjacent to `u`, are new missing
            # edges at `u` and `x`
            fillin[u] += unum - c
            fillin[x] += Int(number[x]) - c

            unum += 1; number[x] += one(V); marker[x] = tag
            uwgtadd += weight[x]

            # move `x` to the front
            nx += one(V)
            stack8[i], stack8[nx] = stack8[nx], stack8[i]
            arcs[i], arcs[nx] = arcs[nx], arcs[i]
        end
    end

    @inbounds number[u] = convert(V, unum)

    # N[`v`] is now a clique: remove `v` from the graph
    @inbounds for i in oneto(k)
        z = stack8[i]
        fillin[z] -= Int(number[z] - d)
        number[z] -= one(V)
        hi1 = asr_touch!(status, stack1, hi1, z)
    end

    # update the weighted degrees
    @inbounds degree[u] += uwgtadd - wgt

    @inbounds for i in oneto(k)
        z = stack8[i]

        if i <= nx
            degree[z] += uwgt - wgt
        elseif z != u
            degree[z] -= wgt
        end
    end

    # update the quotient graph: `v` becomes a supernode of `u`
    # whose neighbors are the vertices `x`
    @inbounds for i in oneto(k)
        z = stack8[i]; zinv = invptr[arcs[i]]

        if i <= nx
            # replace `v` with `u` in the neighborhood of `x`
            target[zinv] = u
        elseif z == u
            # replace `v` with the supernode `-v` in the
            # neighborhood of `u`
            target[zinv] = -v
        else
            # remove `v` from the neighborhood of a common
            # neighbor `z` of `u` and `v`
            pr3_reach_del!(source, target, endptr, invptr, zinv)
        end
    end

    # remove `u` and its common neighbors with `v` from the
    # neighborhood of `v`. Deleting an arc moves the last arc of
    # its list into its slot, so delete in decreasing order.
    if nx < k
        sort!(view(arcs, nx + one(V):k); rev = true, alg = QuickSort)

        @inbounds for i in nx + one(V):k
            pr3_reach_del!(source, target, endptr, invptr, arcs[i])
        end
    end

    return hi4, hi1, tag
end
