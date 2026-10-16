# status flags used by `asr`
const ASR_QUEUED = 0x01 # the vertex is in the work queue
const ASR_PARKED = 0x02 # the vertex failed a test because of `width`
const ASR_DELETE = 0x04 # the vertex has been eliminated

# the connector search (see `asr_connector!`) for a vertex of
# degree d scans at most ASR_CONNECTOR_BASE + ASR_CONNECTOR_DEGREE × d
# arcs before giving up
const ASR_CONNECTOR_BASE = 1 << 16
const ASR_CONNECTOR_DEGREE = 1 << 12

function pr5(weights::AbstractVector{W}, graph::AbstractGraph{V}, width::Number) where {W, V}
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

    afillin = FVector{Int}(undef, n)
    amarker = FVector{Int}(undef, n)
    amarker2 = FVector{Int}(undef, n)
    asource = FVector{V}(undef, m)
    atarget = FVector{V}(undef, m)
    abegptr = FVector{E}(undef, nn)
    atmpptr = FVector{E}(undef, nn)
    atgt = FVector{V}(undef, m)
    ws = ASRWork{V, E}(n)

    return pr5_impl!(
        degree, number, status, marker, source, target, begptr, endptr, invptr,
        stack1, stack2, stack3, stack4, stack5, stack6, stack7, stack8, stack0, tmpptr,
        afillin, amarker, amarker2, asource, atarget, abegptr, atmpptr, atgt,
        weight1, weights, graph, convert(W, width), ASR_EXT_DEFAULT, ws)
end

function pr5_impl!(
        degree::AbstractVector{W}, number::AbstractVector{V}, status::AbstractVector{UInt8},
        marker::AbstractVector{V}, source::AbstractVector{V}, target::AbstractVector{V},
        begptr::AbstractVector{E}, endptr::AbstractVector{E}, invptr::AbstractVector{E},
        stack1::AbstractVector{V}, stack2::AbstractVector{V}, stack3::AbstractVector{V},
        stack4::AbstractVector{V}, stack5::AbstractVector{V}, stack6::AbstractVector{V},
        stack7::AbstractVector{V}, stack8::AbstractVector{V}, stack0::AbstractVector{V},
        tmpptr::AbstractVector{E},
        afillin::AbstractVector{Int}, amarker::AbstractVector{Int}, amarker2::AbstractVector{Int},
        asource::AbstractVector{V}, atarget::AbstractVector{V}, abegptr::AbstractVector{E},
        atmpptr::AbstractVector{E}, atgt::AbstractVector{V},
        weight1::AbstractVector{W}, weight::AbstractVector{W}, graph::AbstractGraph{V}, width::W,
        ext::UInt8 = 0x00, ws = nothing,
    ) where {W, V, E}
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

    graph2, astack, inject2, width2 = asr_impl!(
        weight1, degree, number, afillin, status, amarker, amarker2, asource,
        atarget, abegptr, endptr, invptr, stack1, stack2, stack3, stack5,
        stack6, stack7, atmpptr, atgt, totdeg1, width1, graph1, ext, ws)

    n2 = nv(graph2); m2 = n1 - n2

    @inbounds for i in oneto(m2)
        stack[i + m1] = inject1[astack[i]]
    end

    @inbounds for i in oneto(n2)
        inject2[i] = inject1[inject2[i]]
    end

    return (graph2, stack, inject2, width2)
end

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
    ws = ASRWork{V, E}(n)

    return asr_impl!(
        weight, degree, number, fillin, status, marker, marker2, source,
        target, begptr, endptr, invptr, stack1, stack4, stack5, stack7,
        stack8, stack2, tmpptr, tgt, totdeg, width, graph, ASR_EXT_DEFAULT, ws)
end

"""
    asr_impl!(...)

Safe Reduction Rules for Weighted Treewidth
Eijkhof, Bodlaender, and Koster

Preprocess a graph by applying *safe* reduction rules to
vertices of any degree.
  - simplicial: N(v) is a clique
  - almost simplicial: N(v) \\ {u} is a clique for a neighbor u
    of v, with weight(u) ≤ weight(v) and weight(N[v]) ≤ `width`.
    Then v is contracted into u.
  - star forest: the missing edges of N(v) form vertex-disjoint
    stars, with centers no heavier than v, and weight(N[v]) ≤ `width`.
    Then v is eliminated.
  - twins: v and the vertices with the same neighborhood S = N(v)
    can pay for a vertex cover of the missing edges of S (see
    `asr_twins!`). Then they are eliminated.
The last two rules are selected by the bit mask `ext` (both by
default); see the comments above `ASR_EXT_STAR`.

If weight(N[v]) > `width`, the almost simplicial rule may still
apply: a *connector* is a connected set Q of vertices outside N[v],
each at least as heavy as u, that is adjacent to u and to every
vertex of N(v) \\ N[u]. Contracting Q into u makes N[v] a clique
in a weighted minor of the graph, so weight(N[v]) is a lower bound
for the treewidth: `width` is raised to it, and v is contracted
into u (see `asr_connector!`).

Each vertex v stores its fill-in: the number of missing edges in
N(v), computed by [`fillin!`](@ref). A vertex is simplicial if
its fill-in is zero, and it is almost simplicial only if its
fill-in is less than its degree, since every missing edge has u as
an endpoint. When a vertex is contracted, the fill-in of the
remaining vertices is updated using the method of Wing and Huang,
as in `fillin!`: every new edge {u, x} decreases the fill-in of
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
        ext::UInt8 = 0x00,
        ws = nothing,
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

    # the twin rule needs the false-twin classes
    if !isnothing(ws) && !iszero(ext & ASR_EXT_TWIN)
        asr_twinclasses!(ws, number, graph)
    end

    # apply reduction rules until no more apply
    width, hi4 = asr_loop!(weight, degree, number, fillin, status, marker,
        marker2, source, target, begptr, endptr, invptr, stack1, stack4,
        stack5, stack7, stack8, tmpptr, stack2, width, parked, 0, hi1,
        zero(V), zero(V), n, ext, ws)

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
    fillin!(fillin, stack5, stack4, tmpptr, tgt,
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
        n::V,
        ext::UInt8 = 0x00,
        ws = nothing,
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
                elseif degree[v] < width + tol && (fil < Int(number[v]) || asr_twin_candidate(ws, ext, fil, number[v], n, v))
                    # `v` may be almost simplicial
                    u = zero(V)

                    if fil < Int(number[v])
                        u, tag = asr_friend!(weight, number, fillin, marker,
                            marker2, stack5, stack8, arcs, target, begptr,
                            endptr, invptr, tag, v)
                    end

                    if ispositive(u)
                        hi4, hi1, tag = asr_contract!(weight, degree,
                            number, fillin, status, marker, marker2, source,
                            target, begptr, endptr, invptr, stack1, stack4,
                            stack5, stack8, arcs, stack2, hi4, hi1, tag, v, u)
                    elseif !isnothing(ws) && !iszero(ext & (ASR_EXT_STAR | ASR_EXT_TWIN))
                        # the star-forest and twin rules (see below)
                        width, hi4, hi1, tag = asr_extend!(ws, ext, weight,
                            degree, number, fillin, status, marker, marker2,
                            source, target, begptr, endptr, invptr, stack1,
                            stack4, stack5, stack8, arcs, stack2, width, hi4,
                            hi1, tag, v)
                    end
                elseif fil < Int(number[v])
                    # `v` may be almost simplicial
                    begin
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

                            # the unit connector (see below)
                            if !found && !isnothing(ws) && !iszero(ext & ASR_EXT_UNIT)
                                found, tag = asr_connector_units!(ws, weight, number,
                                    marker, marker2, stack5, target, begptr, endptr,
                                    invptr, tag, v, u)
                            end
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
                elseif asr_twin_candidate(ws, ext, fil, number[v], n, v)
                    # the twin rule may apply once the lower bound is larger
                    parked, hi7 = asr_park!(status, stack7, parked, hi7, width, v)
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

            # (only the first `n` entries belong to this graph: the work
            # arrays are shared with `pr3` and with earlier rounds)
            for v in oneto(n)
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

    # quick rejection: a vertex of X with no heavy neighbor outside
    # N[`v`] can never be covered, and without this check the search
    # below would exhaust its budget before giving up
    @inbounds for i in oneto(d)
        y = stack8[i]

        if marker2[y] == xtag
            ok = pr3_reach!(false, stack5, target, begptr, endptr, invptr, y) do ok, _, z
                @inbounds return ok || (marker[z] != ntag && weight[z] > uwgt - tol)
            end

            ok || return false, tag
        end
    end

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

# contract an almost simplicial vertex `v` into its neighbor `u`. If `rem` is
# given, N[`v`] need not become a clique: `rem[z]` is the number of vertices of
# N(`v`) that are not adjacent to `z` once the edges of `u` are added (used by
# the twin rule)
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
        rem::Union{Nothing, AbstractVector{Int}} = nothing,
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

    # N[`v`] is now a clique (unless `rem` is given): remove `v` from the graph
    @inbounds for i in oneto(k)
        z = stack8[i]
        fillin[z] -= Int(number[z] - d) + (isnothing(rem) ? 0 : rem[z])
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

# ---------------------------------------------------------------------------
# The star-forest and twin rules of `asr`
# ---------------------------------------------------------------------------
#
# Both rules are safe separator rules for the separator S = N(v), and both
# extend the almost simplicial rule. They are tried on a vertex v that is
# neither simplicial nor almost simplicial, with w(N[v]) ≤ `width`.
#
#   - Star forest: if the missing edges of N(v) form a star forest
#     (vertex-disjoint stars) whose centers are no heavier than v, then v is
#     eliminated. A single star is the almost simplicial rule.
#
#     Why it is safe. Let S = N(v), and for each component D of G - N[v], let
#     S_D = N(D). Then S_D is a minimal separator (D and the component
#     containing v are full), and G + K(S) has the clique separator S, so it
#     suffices that tw(G[S_D ∪ D] + K(S_D)) ≤ tw(G) for each D. This follows
#     from Theorem 16 of Medema and Lazovik, "A safeness condition for minimal
#     separators based on vertex connectivity" (2025), applied to S_D and the
#     component D only: the proof of their Lemma 37 bounds the cliques of each
#     block using the condition for that block alone. Take K = the vertices of
#     S_D that are not the center of a star with a leaf in S_D (a clique). A
#     missing edge {c, ℓ} of S_D is then joined outside D ∪ K by the path
#     through v and the paths through the other centers, which is |S_D| - |K|
#     paths. With weights, think of a vertex of weight w as w unit-weight true
#     twins (the equivalence behind twin compression; rational weights can be
#     scaled to integers), and take K = the units of the non-centers: a missing
#     edge between a unit of a center c and a unit of a leaf is joined by the
#     w(v) paths through the units of v and the w(c') paths through the units of
#     each other center c', and the theorem asks for the sum of w(c) over all
#     centers, which holds if no center is heavier than v.
#
#     The fill edges are stored in the quotient graph without allocating: the
#     arc of each neighbor of v to v is reused. This is possible if at most one
#     star has two leaves or more; that star is handled as in the almost
#     simplicial rule (v becomes a supernode of its center). Star forests with
#     two such stars are not reduced.
#
#   - Twins (a generalization of the buddy rule of Bodlaender et al. and the
#     buddies rule of van den Eijkhof et al.): the absorbers are v and its
#     false twins, the vertices y ∉ N[v] with N(y) = S. If the missing edges of
#     S have a vertex cover C whose weight is at most the total weight of the
#     usable absorbers (w(N[a]) ≤ `width`), then the usable absorbers are
#     eliminated. See `asr_twins!`.
#
#     Searching every vertex for twins is too slow on large sparse graphs, so
#     the false-twin classes are computed at the start of each round, in linear
#     time, and only vertices with a false twin, or whose degree has changed
#     since, are searched. Twins created by a change that keeps the degree are
#     found at the start of the next round.

const ASR_EXT_STAR = 0x01 # star-forest rule
const ASR_EXT_TWIN = 0x02 # twin rule
const ASR_EXT_UNIT = 0x04 # unit connector for the almost simplicial rule

# the rules used by default (`ext = 0x00` gives the rules of Eijkhof et al. only)
const ASR_EXT_DEFAULT = ASR_EXT_STAR | ASR_EXT_TWIN | ASR_EXT_UNIT

# the unit connector looks for at most this many connectors
const ASR_UNIT_ROUNDS = 64

# the twin rule uses at most this many twins of a vertex
const ASR_TWIN_MAX = 64

# the twin rule searches for vertex covers with at most this many vertices
# (larger covers: only the Buddies cover, see `asr_twins!`)
const ASR_COVER_MAX = 8

# work arrays of the extensions
struct ASRWork{V, E}
    ipos::Vector{Int}   # (vertex) index of a neighbor of `v` in `stack8`
    mdeg::Vector{Int}   # (index) number of missing edges at a neighbor of `v`
    rem::Vector{Int}    # (vertex) missing edges left at a vertex of S (twin rule)
    ea::Vector{Int}     # (edge) missing edges {ea[e], eb[e]} of N(`v`), as indices
    eb::Vector{Int}
    pz::Vector{E}       # (index) the arc of a neighbor of `v` to `v`
    dpos::Vector{E}     # arcs to delete
    twin::Vector{V}     # twins of `v`, and `v`
    cen::Vector{V}      # the vertex cover (twin rule)
    sv::Vector{V}       # a copy of N(`v`) (twin rule)
    cover::Vector{Int}  # (index) membership in the vertex cover
    bcov::Vector{Int}   # (index) membership in the best vertex cover found
    stats::Vector{Int}  # counters, see `ASR_STAT_*`
    # false-twin classes of the graph at the start of a round (twin rule)
    tnew::Array{V, 0}
    tvar::Vector{V}
    tsvar::Vector{V}
    tflag::Vector{V}
    tsize::Vector{V}
    elig::Vector{Bool}  # (vertex) in a false-twin class with two vertices or more
    deg0::Vector{V}     # (vertex) degree at the start of the round
    # unit connector (see `asr_connector_units!`)
    cap::Vector{Float64}  # (vertex) units left
    ctag::Vector{Int}     # (vertex) `cap` is valid if `ctag` is the current search
    ccov::Vector{Int}     # (vertex) a leaf touched in the current round
    chit::Vector{V}       # (vertex) the vertex that touched a leaf
    cpar::Vector{V}       # (vertex) parent in the search forest
    cqueue::Vector{V}     # search queue
    cseed::Vector{V}      # the neighbors of `u` outside N[`v`]
    clist::Vector{V}      # the leaves, then the vertices of a connector
end

const ASR_STAT_STAR = 1           # star forests reduced (two stars or more)
const ASR_STAT_STAR_MULTI = 2     # star forests rejected: two stars with two leaves or more
const ASR_STAT_TWIN = 3           # twin groups reduced
const ASR_STAT_TESTS = 4          # missing-edge scans
const ASR_STAT_TWINSEARCH = 5     # twin searches
const ASR_STAT_TWINSCAN = 6       # arcs scanned by twin searches
const ASR_STAT_UNIT = 7           # lower bounds certified by the unit connector
const ASR_NSTATS = 7

function ASRWork{V, E}(n::Integer) where {V, E}
    return ASRWork{V, E}(zeros(Int, n), zeros(Int, n), zeros(Int, n), zeros(Int, n),
        zeros(Int, n), zeros(E, n), zeros(E, n), zeros(V, n), zeros(V, n), zeros(V, n),
        zeros(Int, n), zeros(Int, n), zeros(Int, ASR_NSTATS), fill(zero(V)), zeros(V, n + 1), zeros(V, n),
        zeros(V, n), zeros(V, n), falses(n) |> collect, zeros(V, n),
        zeros(Float64, n), zeros(Int, n), zeros(Int, n), zeros(V, n), zeros(V, n),
        zeros(V, n), zeros(V, n), zeros(V, 2n))
end

# List the missing edges of N(`v`). On entry, `stack8[1:k]` holds N(`v`), each
# marked with `ntag` in `marker`, and `fil` = fillin[`v`]. The missing edges
# are written to `ws.ea[1:ne]` and `ws.eb[1:ne]` (indices into `stack8`) and
# `ws.mdeg[i]` is the number of missing edges at `stack8[i]`. Returns `ne`, or
# -1 if the scan was abandoned.
#
# Neighbors of degree at most 2k are scanned first. The others are scanned only
# if some missing edge is still unaccounted for, since a missing edge with an
# endpoint of small degree has already been found.
#
# If `star` is true, the scan is abandoned as soon as two vertices with two
# missing edges or more are found to be joined by a missing edge: the missing
# edges then do not form a star forest. To find such a pair early, the
# partners of a vertex with two missing edges or more are scanned next.
function asr_missing!(
        ws::ASRWork,
        number::AbstractVector{V},
        marker::AbstractVector{Int},
        marker2::AbstractVector{Int},
        stack5::AbstractVector{V},
        stack8::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        tag::Int,
        ntag::Int,
        fil::Int,
        k::Int,
        star::Bool = false,
    ) where {V, E}
    mdeg = ws.mdeg; ea = ws.ea; eb = ws.eb; queue = ws.cover; ws.stats[ASR_STAT_TESTS] += 1

    @inbounds for i in oneto(k)
        ws.ipos[stack8[i]] = i; mdeg[i] = -1
    end

    ne = 0

    @inbounds for pass in 1:2
        (pass == 2 && ne == fil) && break

        for i0 in oneto(k)
            # `queue[1:nq]` holds the vertices to scan next: `i0`, then
            # (in star mode) the partners of vertices with two missing edges
            nq = 1; queue[1] = i0; iq = 0

            while iq < nq
                iq += 1; i = queue[iq]
                x = stack8[i]
                mdeg[i] < 0 || continue
                (Int(number[x]) <= twice(k)) == isone(pass) || continue

                tag += 1; tag2 = tag
                cnt = asr_count_mark(marker, marker2, stack5, target, begptr,
                    endptr, invptr, x, ntag, tag2)

                miss = k - 1 - cnt; mdeg[i] = miss

                if ispositive(miss)
                    for j in oneto(k)
                        y = stack8[j]

                        if j != i && marker2[y] != tag2
                            if mdeg[j] < 0
                                # record the missing edges to neighbors not scanned yet
                                ne += 1
                                ne > fil && return -1, tag
                                ea[ne] = i; eb[ne] = j

                                if star && miss >= 2 && nq < k
                                    nq += 1; queue[nq] = j
                                end
                            elseif star && miss >= 2 && mdeg[j] >= 2
                                return -1, tag
                            end
                        end
                    end
                end
            end
        end
    end

    ne == fil || return -1, tag

    @inbounds for i in oneto(k)
        mdeg[i] = 0
    end

    @inbounds for e in oneto(ne)
        mdeg[ea[e]] += 1; mdeg[eb[e]] += 1
    end

    return ne, tag
end

# Decide whether the missing edges form a star forest whose centers are no
# heavier than `v`. Orients every edge as (center, leaf). Returns the index of
# the center of the star with two leaves or more (0 if none), the number of
# stars, and whether the rule applies.
function asr_starplan!(ws::ASRWork, weight::AbstractVector{W}, stack8::AbstractVector{V}, ne::Int, k::Int, wv::W) where {W, V}
    tol = tolerance(W); mdeg = ws.mdeg; ea = ws.ea; eb = ws.eb
    absorber = 0; nstars = 0

    @inbounds for e in oneto(ne)
        a = ea[e]; b = eb[e]
        (mdeg[a] >= 2 && mdeg[b] >= 2) && return 0, 0, false

        # orient (center, leaf); a single edge is centered at its lighter end
        if mdeg[b] >= 2 || (isone(mdeg[a]) && isone(mdeg[b]) && weight[stack8[b]] < weight[stack8[a]])
            ea[e] = b; eb[e] = a
        end

        weight[stack8[ea[e]]] < wv + tol || return 0, 0, false
        isone(mdeg[ea[e]]) && (nstars += 1)
    end

    # a vertex with two missing edges or more is a center
    @inbounds for i in oneto(k)
        if mdeg[i] >= 2
            nstars += 1

            if iszero(absorber)
                absorber = i
            else
                ws.stats[ASR_STAT_STAR_MULTI] += 1
                return 0, nstars, false
            end
        end
    end

    return absorber, nstars, true
end

function findfirst_center(ea::AbstractVector{Int}, ne::Int, i::Int)
    @inbounds for e in oneto(ne)
        ea[e] == i && return e
    end

    return 1
end

# Eliminate `v`, whose missing edges form a star forest with two stars or more
# (planned by `asr_starplan!`). On entry, `stack8[1:k]` and `arcs[1:k]` are the
# neighbors and arcs of `v`, and `ws.ea[1:ne]`, `ws.eb[1:ne]` the missing edges,
# as (center, leaf) indices.
function asr_starforest!(
        ws::ASRWork,
        weight::AbstractVector{W},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        fillin::AbstractVector{Int},
        status::AbstractVector{UInt8},
        marker::AbstractVector{Int},
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
        k::Int,
        ne::Int,
        absorber::Int,
    ) where {W, V, E}
    ea = ws.ea; eb = ws.eb; mdeg = ws.mdeg; pz = ws.pz
    ws.stats[ASR_STAT_STAR] += 1

    # add `v` to the stack of eliminated vertices
    @inbounds status[v] |= ASR_DELETE
    hi4 = pr3_stack_add!(stack4, hi4, v)
    @inbounds d = Int(number[v]); wgt = weight[v]

    # Wing-Huang updates, one star at a time: the stars are vertex-disjoint,
    # so the edges of one star create no common neighbors for the edges of
    # another (every vertex of N(`v`) outside a star is adjacent to all of it)
    @inbounds for e in oneto(ne)
        ci = ea[e]

        # each center is processed once, at its first edge
        findfirst_center(ea, ne, ci) == e || continue

        c = stack8[ci]; tag += 1; ctag = tag

        pr3_reach!(nothing, stack5, target, begptr, endptr, invptr, c) do _, _, w
            @inbounds marker[w] = ctag
            return
        end

        for f in e:ne
            ea[f] == ci || continue
            l = stack8[eb[f]]

            cnt = pr3_reach!(0, stack5, target, begptr, endptr, invptr, l) do cnt, _, t
                @inbounds if marker[t] == ctag
                    cnt += 1; fillin[t] -= 1; stack2[cnt] = t
                end

                return cnt
            end

            for j in oneto(cnt)
                hi1 = asr_touch!(status, stack1, hi1, stack2[j])
            end

            fillin[c] += Int(number[c]) - cnt
            fillin[l] += Int(number[l]) - cnt
            number[c] += one(V); number[l] += one(V); marker[l] = ctag
            degree[c] += weight[l]; degree[l] += weight[c]
        end
    end

    # N[`v`] is now a clique: remove `v` from the graph
    @inbounds for i in oneto(k)
        z = stack8[i]
        fillin[z] -= Int(number[z]) - d
        number[z] -= one(V)
        degree[z] -= wgt
        hi1 = asr_touch!(status, stack1, hi1, z)
    end

    # update the quotient graph. `pz[i]` is the arc of `stack8[i]` to `v`
    @inbounds for i in oneto(k)
        pz[i] = invptr[arcs[i]]
    end

    # role of each neighbor: 0 = outside every star, otherwise the index of
    # its partner (single edges), or -1 (the star with two leaves or more)
    @inbounds for i in oneto(k)
        mdeg[i] = 0
    end

    @inbounds for e in oneto(ne)
        c = ea[e]; l = eb[e]

        if c == absorber
            mdeg[c] = -1; mdeg[l] = -1
        else
            mdeg[c] = l; mdeg[l] = c
        end
    end

    nd = 0
    cvert = iszero(absorber) ? zero(V) : stack8[absorber]

    @inbounds for i in oneto(k)
        r = mdeg[i]; p = pz[i]

        if r == -1
            if i == absorber
                # `v` becomes a supernode of the center
                target[p] = -v
            else
                # a leaf of that star: replace `v` with the center
                target[p] = cvert
            end
        elseif ispositive(r)
            # a single missing edge {`stack8[i]`, `stack8[r]`}: reuse the
            # arcs of both endpoints to `v`
            target[p] = stack8[r]; invptr[p] = pz[r]
        else
            # outside every star: remove `v` from its neighborhood
            pr3_reach_del!(source, target, endptr, invptr, p)
        end

        # the arcs of `v` to keep are those to the leaves of the big star
        if !iszero(absorber) && (r != -1 || i == absorber)
            nd += 1; ws.dpos[nd] = arcs[i]
        end
    end

    # remove the other arcs of `v` (if it became a supernode); deleting an arc
    # moves the last arc of its list into its slot, so delete in decreasing order
    if ispositive(nd)
        sort!(view(ws.dpos, 1:nd); rev = true)

        @inbounds for j in 1:nd
            pr3_reach_del!(source, target, endptr, invptr, ws.dpos[j])
        end
    end

    return hi4, hi1, tag
end

# Try the rules of `ext` on `v`, which is neither simplicial nor almost
# simplicial, and has w(N[`v`]) ≤ `width`. Returns the lower bound `width`
# (unchanged for now: a rule that certifies its own lower bound would raise it).
function asr_extend!(
        ws::ASRWork,
        ext::UInt8,
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
        width::W,
        hi4::V,
        hi1::V,
        tag::Int,
        v::V,
    ) where {W, V, E}
    @inbounds fil = fillin[v]

    # mark the neighbors of `v` with `ntag`
    tag += 1; ntag = tag
    k = Int(asr_neighbors!(stack8, arcs, stack5, target, begptr, endptr, invptr, v))

    @inbounds for i in oneto(k)
        marker[stack8[i]] = ntag
    end

    ne = -1

    # a star forest with two stars or more has at most k - 2 missing edges
    if !iszero(ext & ASR_EXT_STAR) && fil <= k - 2
        ne, tag = asr_missing!(ws, number, marker, marker2, stack5, stack8,
            target, begptr, endptr, invptr, tag, ntag, fil, k, true)

        absorber, nstars, ok = 0, 0, false

        if ne >= 0
            @inbounds absorber, nstars, ok = asr_starplan!(ws, weight, stack8, ne, k, weight[v])
        end

        if ok && nstars >= 2
            hi4, hi1, tag = asr_starforest!(ws, weight, degree, number, fillin,
                status, marker, source, target, begptr, endptr, invptr, stack1,
                stack4, stack5, stack8, arcs, stack2, hi4, hi1, tag, v, k, ne,
                absorber)

            return width, hi4, hi1, tag
        end
    end

    if !iszero(ext & ASR_EXT_TWIN) && asr_twin_eligible(ws, k, v)
        # search for twins first: it is cheaper than listing the missing edges
        nt = asr_findtwins!(ws, number, marker, stack2, stack5, stack8,
            target, begptr, endptr, invptr, v, k, ntag)

        if ispositive(nt)
            if ne < 0
                ne, tag = asr_missing!(ws, number, marker, marker2, stack5, stack8,
                    target, begptr, endptr, invptr, tag, ntag, fil, k)

                ne < 0 && return width, hi4, hi1, tag
            end

            width, hi4, hi1, tag = asr_twins!(ws, weight, degree, number, fillin,
                status, marker, source, target, begptr, endptr, invptr, stack1,
                stack4, stack5, stack8, arcs, stack2, width, hi4, hi1, tag, v, k,
                ne, nt)
        end
    end

    return width, hi4, hi1, tag
end

# Search for a vertex cover of the edges {ea[e], eb[e]} (indices into `stack8`)
# with at most `budget` more vertices than `cover`, of weight less than `wbest`,
# by a bounded search tree. The best one is copied to `bcov`. Returns the
# weight of the best cover found (`wbest` if none).
function asr_mincover!(
        cover::AbstractVector{Int},
        bcov::AbstractVector{Int},
        ea::AbstractVector{Int},
        eb::AbstractVector{Int},
        weight::AbstractVector{W},
        stack8::AbstractVector{V},
        ne::Int,
        k::Int,
        budget::Int,
        wcur::W,
        wbest::W,
    ) where {W, V}
    @inbounds for e in oneto(ne)
        a = ea[e]; b = eb[e]

        if iszero(cover[a]) && iszero(cover[b])
            iszero(budget) && return wbest

            for x in (a, b)
                wx = wcur + weight[stack8[x]]

                if wx < wbest
                    cover[x] = 1
                    wbest = asr_mincover!(cover, bcov, ea, eb, weight, stack8, ne, k, budget - 1, wx, wbest)
                    cover[x] = 0
                end
            end

            return wbest
        end
    end

    # every edge is covered
    @inbounds for i in oneto(k)
        bcov[i] = cover[i]
    end

    return wcur
end

# can `v` be reduced by the twin rule although its fill-in is at least its degree?
function asr_twin_candidate(ext::UInt8, fil::Int, d::Integer, n::Integer)
    return !iszero(ext & ASR_EXT_TWIN) && fil <= Int(n)
end

function asr_twin_candidate(ws::ASRWork, ext::UInt8, fil::Int, d::Integer, n::Integer, v::Integer)
    return asr_twin_candidate(ext, fil, d, n) && asr_twin_eligible(ws, d, v)
end

asr_twin_candidate(::Nothing, ext::UInt8, fil::Int, d::Integer, n::Integer, v::Integer) = false

# A twin search is needed only if `v` had a false twin at the start of the
# round, or if its neighborhood has changed since (detected by its degree).
# This misses twins created by a change that keeps the degree; they are found
# at the start of the next round, which takes place whenever a round reduces
# the graph.
function asr_twin_eligible(ws::ASRWork, d::Integer, v::Integer)
    @inbounds return ws.elig[v] || d != ws.deg0[v]
end

# compute the false-twin classes of `graph` at the start of a round
function asr_twinclasses!(ws::ASRWork{V}, number::AbstractVector{V}, graph::AbstractGraph{V}) where {V}
    n = nv(graph)
    partition = twins_impl!(ws.tnew, ws.tvar, ws.tsvar, ws.tflag, ws.tsize, graph, Val(:F))

    @inbounds for v in oneto(n)
        ws.elig[v] = eltypedegree(partition, ws.tsvar[v]) >= 2
        ws.deg0[v] = number[v]
    end

    return
end

# Find the twins of `v`: the vertices y ∉ N[`v`] with N(y) = N(`v`). On entry,
# `stack8[1:k]` is N(`v`), marked with `ntag`. At most `ASR_TWIN_MAX - 1` twins
# are written to `ws.twin`; returns their number.
function asr_findtwins!(
        ws::ASRWork,
        number::AbstractVector{V},
        marker::AbstractVector{Int},
        stack2::AbstractVector{V},
        stack5::AbstractVector{V},
        stack8::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        v::V,
        k::Int,
        ntag::Int,
    ) where {V, E}
    # a twin of `v` is adjacent to every neighbor of `v`: search the
    # neighborhood of the neighbor `x` of least degree
    xi = 1

    @inbounds for i in 2:k
        number[stack8[i]] < number[stack8[xi]] && (xi = i)
    end

    @inbounds x = stack8[xi]
    ws.stats[ASR_STAT_TWINSEARCH] += 1; ws.stats[ASR_STAT_TWINSCAN] += Int(number[x])

    nc = pr3_reach!(0, stack5, target, begptr, endptr, invptr, x) do nc, _, y
        @inbounds if y != v && marker[y] != ntag && Int(number[y]) == k
            nc += 1; stack2[nc] = y
        end

        return nc
    end

    nt = 0

    @inbounds for c in oneto(nc)
        y = stack2[c]
        asr_count(marker, stack5, target, begptr, endptr, invptr, y, ntag) == k || continue
        nt += 1; ws.twin[nt] = y
        nt == ASR_TWIN_MAX - 1 && break
    end

    return nt
end

# The twin rule. On entry, `stack8[1:k]` is N(`v`), `ws.twin[1:nt]` are the
# twins of `v`, and the missing edges of S = N(`v`) are `ws.ea[1:ne]`,
# `ws.eb[1:ne]` (indices), with `ws.mdeg` their number at each vertex.
# The vertex `v` satisfies w(N[`v`]) ≤ `width`.
#
# The absorbers are `v` and its twins; an absorber a is usable if
# w(N[a]) = w(S) + w(a) ≤ `width`. Think of a vertex of weight w as w unit-weight
# true twins (the equivalence behind twin compression; rational weights can be
# scaled to integers). Contracting distinct absorber units into the units of a
# vertex cover C of the missing edges turns S into a clique, so if
# w(C) ≤ w(usable absorbers), the graph G - (usable absorbers) + K(S) is a
# minor of G, and its width is at most that of G. The usable absorbers are
# eliminated with bags S ∪ {a} of weight at most `width`. The other twins are
# then simplicial.
#
# In the quotient graph, the fill edges are stored by contracting one absorber
# into each vertex of C, so |C| is at most the number of usable absorbers. The
# usable absorbers that are not contracted become simplicial too.
function asr_twins!(
        ws::ASRWork,
        weight::AbstractVector{W},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        fillin::AbstractVector{Int},
        status::AbstractVector{UInt8},
        marker::AbstractVector{Int},
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
        width::W,
        hi4::V,
        hi1::V,
        tag::Int,
        v::V,
        k::Int,
        ne::Int,
        nt::Int,
    ) where {W, V, E}
    tol = tolerance(W); ea = ws.ea; eb = ws.eb; absorber = ws.twin

    # `v` and its twins are the absorbers, heaviest first
    @inbounds absorber[nt + 1] = v; t = nt + 1
    sort!(view(absorber, 1:t); by = y -> weight[y], rev = true)

    # the usable absorbers are the lightest ones: `absorber[jstart:t]`
    @inbounds wS = degree[v] - weight[v]; capacity = zero(W); jstart = t + 1

    @inbounds for j in t:-1:1
        degree[absorber[j]] < width + tol || break
        capacity += weight[absorber[j]]; jstart = j
    end

    budget = t - jstart + 1
    budget < 2 && return width, hi4, hi1, tag

    # the lightest vertex cover with at most `budget` vertices (searched up to
    # `ASR_COVER_MAX` vertices) and weight at most `capacity`
    @inbounds for i in oneto(k)
        ws.cover[i] = 0; ws.bcov[i] = 0
    end

    wbest = asr_mincover!(ws.cover, ws.bcov, ea, eb, weight, stack8, ne, k,
        min(budget, ASR_COVER_MAX), zero(W), capacity + tol)

    # the Buddies cover: every vertex of S but the heaviest
    if budget >= k - 1
        imax = 1

        @inbounds for i in 2:k
            weight[stack8[i]] > weight[stack8[imax]] && (imax = i)
        end

        @inbounds wbuddies = wS - weight[stack8[imax]]

        if wbuddies < wbest
            wbest = wbuddies

            @inbounds for i in oneto(k)
                ws.bcov[i] = Int(i != imax)
            end
        end
    end

    wbest < capacity + tol || return width, hi4, hi1, tag

    nr = 0

    @inbounds for i in oneto(k)
        if isone(ws.bcov[i])
            nr += 1; ws.cen[nr] = stack8[i]
        end
    end

    ws.stats[ASR_STAT_TWIN] += 1

    # copy N(`v`) and the missing edges (as vertices): `stack8` is reused
    rem = ws.rem

    @inbounds for i in oneto(k)
        ws.sv[i] = stack8[i]; rem[stack8[i]] = ws.mdeg[i]
    end

    @inbounds for e in oneto(ne)
        ea[e] = Int(stack8[ea[e]]); eb[e] = Int(stack8[eb[e]])
    end

    # contract each absorber into its center. `rem[z]` is the number of missing
    # edges of S at z once the edges of the current center are added
    @inbounds for j in oneto(nr)
        c = ws.cen[j]; a = absorber[jstart + j - 1]

        for e in oneto(ne)
            if ispositive(ea[e]) && (ea[e] == c || eb[e] == c)
                rem[ea[e]] -= 1; rem[eb[e]] -= 1; ea[e] = -ea[e]
            end
        end

        hi4, hi1, tag = asr_contract!(weight, degree, number, fillin, status,
            marker, marker, source, target, begptr, endptr, invptr, stack1,
            stack4, stack5, stack8, arcs, stack2, hi4, hi1, tag, a, c, rem)
    end

    @inbounds for i in oneto(k)
        rem[ws.sv[i]] = 0
    end

    return width, hi4, hi1, tag
end

# The unit connector. `v` is almost simplicial with friend `u`, w(N[`v`]) is
# larger than `width`, and `asr_connector!` found no connector. Think of a
# vertex of weight w as w unit-weight true twins (the equivalence behind twin
# compression; rational weights can be scaled to integers). Then N[`v`] is a
# clique minor if each of the w(`u`) units of `u` has its own connector: a
# connected set of units outside N[`v`] that is adjacent to `u` and touches
# every leaf (every vertex of N(`v`) not adjacent to `u`). A vertex q can lend
# its units to at most w(q) of them. The connector of `asr_connector!` is the
# case where one set of vertices, each at least as heavy as `u`, serves every
# unit.
#
# The connectors are found greedily, one per round. A round searches the
# vertices outside N[`v`] with units left, starting from one neighbor of `u`
# (a root), until every leaf is touched, and keeps the paths of the search tree
# that reach the leaves. The roots are tried in turn; if none reaches every
# leaf, the round starts from all of them at once (each tree of the forest
# contains a neighbor of `u`, so the paths and `u` still form a connected set).
# Every vertex of the connector lends α units, where α is the least number of
# units left on it, or the number still needed. The search gives up after
# `ASR_UNIT_ROUNDS` rounds, or after scanning as many arcs as `asr_connector!`
# may.
function asr_connector_units!(
        ws::ASRWork{V},
        weight::AbstractVector{W},
        number::AbstractVector{V},
        marker::AbstractVector{Int},
        marker2::AbstractVector{Int},
        stack5::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        tag::Int,
        v::V,
        u::V,
    ) where {W, V, E}
    tol = Float64(tolerance(W)); cap = ws.cap; ctag = ws.ctag
    @inbounds d = Int(number[v])

    # mark N[`v`] with `ntag`
    tag += 1; ntag = tag
    @inbounds marker[v] = ntag

    pr3_reach!(nothing, stack5, target, begptr, endptr, invptr, v) do _, _, y
        @inbounds marker[y] = ntag
        return
    end

    # mark N(`u`) with `utag`, then the leaves with `xtag`; list them in `clist`
    tag += 1; utag = tag

    pr3_reach!(nothing, stack5, target, begptr, endptr, invptr, u) do _, _, y
        @inbounds marker2[y] = utag
        return
    end

    tag += 1; xtag = tag

    nx = pr3_reach!(0, stack5, target, begptr, endptr, invptr, v) do nx, _, y
        @inbounds if y != u && marker2[y] != utag
            marker2[y] = xtag; nx += 1; ws.clist[nx] = y
        end

        return nx
    end

    iszero(nx) && return false, tag

    # the roots: the neighbors of `u` outside N[`v`]
    ns = pr3_reach!(0, stack5, target, begptr, endptr, invptr, u) do ns, _, z
        @inbounds if marker[z] != ntag
            ns += 1; ws.cseed[ns] = z
        end

        return ns
    end

    iszero(ns) && return false, tag

    # `need` is the number of units of `u` without a connector; `cap` is
    # initialized lazily (`ctag[q] == stag`)
    @inbounds need = Float64(weight[u])
    tag += 1; stag = tag
    budget = ASR_CONNECTOR_BASE + ASR_CONNECTOR_DEGREE * d; visits = 0

    for _ in 1:ASR_UNIT_ROUNDS
        ok = false

        # one root at a time, then all of them
        for s in 1:ns + 1
            lo, hi = s <= ns ? (s, s) : (1, ns)
            s <= ns && asr_units(ws, weight, stag, ws.cseed[s]) < tol && continue

            ok, tag, visits = asr_unit_search!(ws, weight, marker, marker2, stack5,
                target, begptr, endptr, invptr, tag, lo, hi, ntag, xtag, stag, nx,
                visits)

            visits > budget && return false, tag
            ok && break
        end

        ok || return false, tag

        # the connector: the paths from the leaves to the roots, listed after
        # the leaves in `clist`, marked with `ptag`
        tag += 1; ptag = tag; np = nx; alpha = need

        @inbounds for i in oneto(nx)
            q = ws.chit[ws.clist[i]]

            while ispositive(q) && marker2[q] != ptag
                marker2[q] = ptag; np += 1; ws.clist[np] = q
                alpha = min(alpha, asr_units(ws, weight, stag, q))
                q = ws.cpar[q]
            end
        end

        # every vertex of the connector lends `alpha` units
        @inbounds for i in nx + 1:np
            q = ws.clist[i]; cap[q] = asr_units(ws, weight, stag, q) - alpha; ctag[q] = stag
        end

        need -= alpha

        if need < tol
            ws.stats[ASR_STAT_UNIT] += 1
            return true, tag
        end
    end

    return false, tag
end

# One round of the unit-connector search: a breadth-first search from the
# roots `ws.cseed[lo:hi]` through the vertices outside N[v] (marked `ntag`)
# with units left. Returns true if every leaf (marked `xtag`) is touched; then
# `ws.chit[x]` is the vertex that touched the leaf x, and `ws.cpar` the parents.
function asr_unit_search!(
        ws::ASRWork{V},
        weight::AbstractVector{W},
        marker::AbstractVector{Int},
        marker2::AbstractVector{Int},
        stack5::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        tag::Int,
        lo::Int,
        hi::Int,
        ntag::Int,
        xtag::Int,
        stag::Int,
        nx::Int,
        visits::Int,
    ) where {W, V, E}
    tol = Float64(tolerance(W))

    # visited vertices are marked with `vtag`, touched leaves with `rtag`
    tag += 1; vtag = tag; tag += 1; rtag = tag; tl = 0

    @inbounds for s in lo:hi
        z = ws.cseed[s]

        if marker[z] != vtag && asr_units(ws, weight, stag, z) >= tol
            marker[z] = vtag; ws.cpar[z] = zero(V); tl += 1; ws.cqueue[tl] = z
        end
    end

    hd = 1; covered = 0

    @inbounds while hd <= tl && covered < nx
        q = ws.cqueue[hd]; hd += 1

        tl, covered, visits = pr3_reach!((tl, covered, visits), stack5, target,
                begptr, endptr, invptr, q) do (tl, covered, visits), _, y
            visits += 1

            @inbounds if marker2[y] == xtag
                # `y` is a leaf, touched by `q`
                if ws.ccov[y] != rtag
                    ws.ccov[y] = rtag; ws.chit[y] = q; covered += 1
                end
            elseif marker[y] != ntag && marker[y] != vtag && asr_units(ws, weight, stag, y) >= tol
                marker[y] = vtag; ws.cpar[y] = q; tl += 1; ws.cqueue[tl] = y
            end

            return (tl, covered, visits)
        end
    end

    return covered == nx, tag, visits
end

# the units of `q` left in the unit-connector search `stag`
@inline function asr_units(ws::ASRWork, weight::AbstractVector, stag::Int, q::Integer)
    @inbounds return ws.ctag[q] == stag ? ws.cap[q] : Float64(weight[q])
end
