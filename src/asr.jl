# status flags used by `asr`
const ASR_QUEUED = 0x01 # the vertex is in the work queue
const ASR_PARKED = 0x02 # the vertex failed a test because of `width`
const ASR_DELETE = 0x04 # the vertex has been eliminated

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

    stack0 = FVector{V}(undef, n)
    tmpptr = FVector{E}(undef, nn)
    tgt = FVector{V}(undef, m)

    work = ASRWorkspace(
        weights,
        FVector{W}(undef, n),       # degree
        FVector{V}(undef, n),       # number
        FVector{Int}(undef, n),     # fillin
        FVector{UInt8}(undef, n),   # status
        FVector{Int}(undef, n),     # marker
        FVector{Int}(undef, n),     # marker2
        FVector{V}(undef, m),       # source
        FVector{V}(undef, m),       # target
        FVector{E}(undef, nn),      # begptr
        FVector{E}(undef, n),       # endptr
        FVector{E}(undef, m),       # invptr
        FVector{V}(undef, n),       # stack1
        FVector{V}(undef, n),       # stack4
        FVector{V}(undef, n),       # stack5
        FVector{V}(undef, n),       # stack7
        FVector{V}(undef, n),       # stack8
        FVector{E}(undef, n),       # arcs
        width,
    )

    kernel, stack, inject, width = asr_impl!(work, stack0, tmpptr, tgt, totdeg, graph)
    return kernel, stack, inject, width
end

"""
    ASRWorkspace

Working storage for [`asr`](@ref).

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
    - `stack4`: eliminated vertices
    - `stack5`: traversal stack
    - `stack8`: the neighbors of a vertex
    - `arcs`: the arcs of a vertex
"""
mutable struct ASRWorkspace{
        W, V, E,
        Wgt <: AbstractVector{W},
        WVec <: AbstractVector{W},
        VVec <: AbstractVector{V},
        EVec <: AbstractVector{E},
        IVec <: AbstractVector{Int},
        SVec <: AbstractVector{UInt8},
    }
    const weight::Wgt
    const degree::WVec
    const number::VVec
    const fillin::IVec
    const status::SVec
    const marker::IVec
    const marker2::IVec
    const source::VVec
    const target::VVec
    const begptr::EVec
    const endptr::EVec
    const invptr::EVec
    const stack1::VVec
    const stack4::VVec
    const stack5::VVec
    const stack7::VVec
    const stack8::VVec
    const arcs::EVec
    width::W
    parked::W
    tag::Int
    hi1::V
    hi4::V
    hi7::V
end

function ASRWorkspace(
        weight::Wgt,
        degree::WVec,
        number::VVec,
        fillin::IVec,
        status::SVec,
        marker::IVec,
        marker2::IVec,
        source::VVec,
        target::VVec,
        begptr::EVec,
        endptr::EVec,
        invptr::EVec,
        stack1::VVec,
        stack4::VVec,
        stack5::VVec,
        stack7::VVec,
        stack8::VVec,
        arcs::EVec,
        width::W,
    ) where {
        W, V, E,
        Wgt <: AbstractVector{W},
        WVec <: AbstractVector{W},
        VVec <: AbstractVector{V},
        EVec <: AbstractVector{E},
        IVec <: AbstractVector{Int},
        SVec <: AbstractVector{UInt8},
    }
    return ASRWorkspace{W, V, E, Wgt, WVec, VVec, EVec, IVec, SVec}(
        weight, degree, number, fillin, status, marker, marker2, source,
        target, begptr, endptr, invptr, stack1, stack4, stack5, stack7,
        stack8, arcs, width, width, 0, zero(V), zero(V), zero(V))
end

"""
    asr_impl!(work, stack0, tmpptr, tgt, totdeg, graph)

Safe Reduction Rules for Weighted Treewidth
Eijkhof, Bodlaender, and Koster

Preprocess a graph by applying two *safe* reduction rules to
vertices of any degree.
  - simplicial: N(v) is a clique
  - almost simplicial: N(v) \\ {u} is a clique for a neighbor u
    of v, with weight(u) ≤ weight(v) and weight(N[v]) ≤ `width`.
    Then v is contracted into u.

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
        work::ASRWorkspace{W, V, E},
        stack0::AbstractVector{V},
        tmpptr::AbstractVector{E},
        tgt::AbstractVector{V},
        totdeg::W,
        graph::AbstractGraph{V},
    ) where {W, V, E}
    n = nv(graph)

    @assert n <= length(work.weight)
    @assert n <= length(work.degree)
    @assert n <= length(work.number)
    @assert n <= length(work.fillin)
    @assert n <= length(work.status)
    @assert n <= length(work.marker)
    @assert n <= length(work.marker2)
    @assert de(graph) <= length(work.source)
    @assert de(graph) <= length(work.target)
    @assert n < length(work.begptr)
    @assert n <= length(work.endptr)
    @assert de(graph) <= length(work.invptr)
    @assert n <= length(work.stack1)
    @assert n <= length(work.stack4)
    @assert n <= length(work.stack5)
    @assert n <= length(work.stack7)
    @assert n <= length(work.stack8)
    @assert n <= length(work.arcs)
    @assert n <= length(stack0)
    @assert n < length(tmpptr)
    @assert de(graph) <= length(tgt)

    # initialize the quotient graph, the fill-in, and the work queue
    asr_init!(work, tmpptr, tgt, totdeg, graph)

    # apply reduction rules until no more apply
    asr_loop!(work)

    # construct the reduced graph
    m, n = pr3_make!(work.stack4, work.stack5, work.target, work.begptr,
        work.endptr, work.invptr, stack0, work.number, tmpptr, tgt,
        work.hi4, n)

    # `kernel` is the reduced graph
    kernel = BipartiteGraph(n, n, m, tmpptr, tgt)
    return kernel, work.stack4, stack0, work.width
end

function asr_init!(
        work::ASRWorkspace{W, V, E},
        tmpptr::AbstractVector{E},
        tgt::AbstractVector{V},
        totdeg::W,
        graph::AbstractGraph{V},
    ) where {W, V, E}
    weight = work.weight
    degree = work.degree
    number = work.number
    fillin = work.fillin
    status = work.status
    marker = work.marker
    marker2 = work.marker2
    source = work.source
    target = work.target
    begptr = work.begptr
    endptr = work.endptr
    invptr = work.invptr

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
    # `arcs`, and `tgt` as working storage
    sr_fillin!(fillin, work.stack5, work.stack4, work.arcs, tgt,
        number, target, begptr, endptr, n)

    # the weighted treewidth of the input graph is no less
    # than its minimum weighted degree
    work.width = work.parked = max(work.width, mindeg)

    # add vertices to the work queue in reverse order, so
    # that they are tested in increasing order
    @inbounds for v in reverse(vertices(graph))
        asr_touch!(work, v)
    end

    return
end

function asr_loop!(work::ASRWorkspace{W, V, E}) where {W, V, E}
    tol = tolerance(W)
    weight = work.weight
    degree = work.degree
    number = work.number
    fillin = work.fillin
    status = work.status

    @inbounds while true
        if ispositive(work.hi1)
            work.hi1, v = pr3_stack_pop!(work.stack1, work.hi1)
            flag = status[v]; status[v] = flag & ~ASR_QUEUED

            if iszero(flag & ASR_DELETE)
                fil = fillin[v]

                if iszero(fil)
                    # `v` is simplicial
                    asr_simplicial!(work, v)
                elseif fil < Int(number[v])
                    # `v` may be almost simplicial
                    if degree[v] < work.width + tol
                        u = asr_friend!(work, v)

                        if ispositive(u)
                            asr_contract!(work, v, u)
                        end
                    else
                        asr_park!(work, v)
                    end
                end
            end
        elseif ispositive(work.hi7) && work.parked < work.width
            # the lower bound has increased: re-test every
            # vertex that failed a test because of it
            for i in oneto(work.hi7)
                v = work.stack7[i]
                status[v] &= ~ASR_PARKED
                asr_touch!(work, v)
            end

            work.hi7 = zero(V)
        elseif ispositive(work.hi7)
            # the remaining graph is a minor of the input graph, so
            # its minimum weighted degree is a lower bound
            mindeg = typemax(W); alive = false

            for v in eachindex(status)
                if iszero(status[v] & ASR_DELETE)
                    mindeg = min(mindeg, degree[v]); alive = true
                end
            end

            if alive && work.width < mindeg
                work.width = mindeg
            else
                break
            end
        else
            break
        end
    end

    return
end

# add `v` to the work queue
function asr_touch!(work::ASRWorkspace{W, V, E}, v::V) where {W, V, E}
    @inbounds flag = work.status[v]

    @inbounds if iszero(flag & (ASR_DELETE | ASR_QUEUED))
        work.status[v] = flag | ASR_QUEUED
        work.hi1 = pr3_stack_add!(work.stack1, work.hi1, v)
    end

    return
end

# a test of `v` failed because of the lower bound
function asr_park!(work::ASRWorkspace{W, V, E}, v::V) where {W, V, E}
    @inbounds flag = work.status[v]

    @inbounds if iszero(flag & ASR_PARKED)
        if iszero(work.hi7)
            work.parked = work.width
        end

        work.status[v] = flag | ASR_PARKED
        work.hi7 = pr3_stack_add!(work.stack7, work.hi7, v)
    end

    return
end

# returns a fresh tag
function asr_tag!(work::ASRWorkspace)
    work.tag += 1
    return work.tag
end

# write the neighbors of `v` to `stack8` and the corresponding
# arcs to `arcs`; returns the number of neighbors
function asr_neighbors!(work::ASRWorkspace{W, V, E}, v::V) where {W, V, E}
    stack8 = work.stack8
    arcs = work.arcs

    k = pr3_reach!(zero(V), work.stack5, work.target, work.begptr,
            work.endptr, work.invptr, v) do k, p, w
        k += one(V)
        @inbounds stack8[k] = w; arcs[k] = p
        return k
    end

    return k
end

# returns the number of vertices `w` reachable by `x`
# with `marker[w] == tag`
function asr_count(work::ASRWorkspace{W, V, E}, x::V, tag::Int) where {W, V, E}
    marker = work.marker

    c = pr3_reach!(0, work.stack5, work.target, work.begptr,
            work.endptr, work.invptr, x) do c, _, w
        @inbounds if marker[w] == tag
            c += 1
        end

        return c
    end

    return c
end

# returns the number of vertices `w` reachable by `x` with
# `marker[w] == tag`, and sets `marker2[w] = tag2` for each one
function asr_count_mark(work::ASRWorkspace{W, V, E}, x::V, tag::Int, tag2::Int) where {W, V, E}
    marker = work.marker
    marker2 = work.marker2

    c = pr3_reach!(0, work.stack5, work.target, work.begptr,
            work.endptr, work.invptr, x) do c, _, w
        @inbounds if marker[w] == tag
            c += 1; marker2[w] = tag2
        end

        return c
    end

    return c
end

# eliminate a simplicial vertex `v`
function asr_simplicial!(work::ASRWorkspace{W, V, E}, v::V) where {W, V, E}
    weight = work.weight
    degree = work.degree
    number = work.number
    fillin = work.fillin
    invptr = work.invptr
    stack8 = work.stack8
    arcs = work.arcs

    # add `v` to the stack of eliminated vertices
    @inbounds work.status[v] |= ASR_DELETE
    work.hi4 = pr3_stack_add!(work.stack4, work.hi4, v)

    # `v` is simplicial: update the lower bound
    @inbounds work.width = max(work.width, degree[v])

    # `d` is the degree of `v`
    @inbounds d = number[v]; wgt = weight[v]

    k = asr_neighbors!(work, v)

    @inbounds for i in oneto(k)
        w = stack8[i]; p = arcs[i]

        # remove `v` from the neighborhood of `w`
        pr3_reach_del!(work.source, work.target, work.endptr, invptr, invptr[p])

        # the missing edges in N(`w`) incident to `v` join
        # `v` to the neighbors of `w` outside N[`v`]
        fillin[w] -= Int(number[w] - d)
        number[w] -= one(V)
        degree[w] -= wgt

        asr_touch!(work, w)
    end

    return
end

# find a neighbor `u` of `v` such that every missing edge in N(`v`) is
# incident to `u` and weight(`u`) ≤ weight(`v`); returns zero if there
# is none. The fill-in of `v` is positive.
function asr_friend!(work::ASRWorkspace{W, V, E}, v::V) where {W, V, E}
    tol = tolerance(W)
    weight = work.weight
    number = work.number
    fillin = work.fillin
    marker = work.marker
    marker2 = work.marker2
    stack8 = work.stack8

    @inbounds fil = fillin[v]; d = Int(number[v]); wgt = weight[v]

    # mark the neighbors of `v` with `tag`
    tag = asr_tag!(work)
    k = asr_neighbors!(work, v)

    @inbounds for i in oneto(k)
        marker[stack8[i]] = tag
    end

    # search for a missing edge {`x`, `y`}, scanning
    # small neighborhoods first
    @inbounds for pass in 1:2, i in oneto(k)
        x = stack8[i]

        # in the first pass, skip vertices of large degree; in
        # the second pass, skip the vertices scanned in the first
        (Int(number[x]) <= twice(d)) == isone(pass) || continue

        # `miss` is the number of missing edges incident to `x`
        tag2 = asr_tag!(work)
        miss = d - 1 - asr_count_mark(work, x, tag, tag2)

        iszero(miss) && continue

        if miss == fil
            # `x` is incident to every missing edge
            weight[x] < wgt + tol && return x

            # if there is one missing edge {`x`, `y`},
            # then `y` is also incident to it
            if isone(fil)
                y = asr_unmarked(work, x, k, tag2)
                weight[y] < wgt + tol && return y
            end

            return zero(V)
        elseif isone(miss)
            # `x` is not incident to every missing edge, so the
            # other endpoint `y` of its missing edge must be
            y = asr_unmarked(work, x, k, tag2)

            if weight[y] < wgt + tol && d - 1 - asr_count(work, y, tag) == fil
                return y
            end

            return zero(V)
        else
            # `x` is incident to two missing edges but not
            # to every missing edge
            return zero(V)
        end
    end

    return zero(V)
end

# returns the neighbor `y` != `x` of `v` (stored in `stack8`) with
# `marker2[y] != tag2`
function asr_unmarked(work::ASRWorkspace{W, V, E}, x::V, k::V, tag2::Int) where {W, V, E}
    marker2 = work.marker2
    stack8 = work.stack8

    @inbounds for i in oneto(k)
        y = stack8[i]

        if y != x && marker2[y] != tag2
            return y
        end
    end

    return zero(V)
end

# contract an almost simplicial vertex `v` into its neighbor `u`
function asr_contract!(work::ASRWorkspace{W, V, E}, v::V, u::V) where {W, V, E}
    weight = work.weight
    degree = work.degree
    number = work.number
    fillin = work.fillin
    marker = work.marker
    source = work.source
    target = work.target
    endptr = work.endptr
    invptr = work.invptr
    stack8 = work.stack8
    arcs = work.arcs

    # add `v` to the stack of eliminated vertices
    @inbounds work.status[v] |= ASR_DELETE
    work.hi4 = pr3_stack_add!(work.stack4, work.hi4, v)

    @inbounds d = number[v]; wgt = weight[v]; uwgt = weight[u]

    # `stack8[1:k]` are the neighbors of `v`, and
    # `arcs[1:k]` are the corresponding arcs
    k = asr_neighbors!(work, v)

    # mark the neighbors of `u` with `tag`
    tag = asr_tag!(work)

    pr3_reach!(nothing, work.stack5, target, work.begptr,
            endptr, invptr, u) do _, _, w
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
            # `c` is the number of common neighbors of `u` and `x`
            c = pr3_reach!(0, work.stack5, target, work.begptr,
                    endptr, invptr, x) do c, _, t
                @inbounds if marker[t] == tag
                    c += 1; fillin[t] -= 1
                    asr_touch!(work, t)
                end

                return c
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
        asr_touch!(work, z)
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

    return
end
