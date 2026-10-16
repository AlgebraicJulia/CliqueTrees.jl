# status flags used by `pr3`
const PR3_STACK1 = 0x01 # the vertex is in the queue of degree 0 and 1 vertices
const PR3_STACK2 = 0x02 # the vertex is in the queue of degree 2 vertices
const PR3_STACK3 = 0x04 # the vertex is in the queue of degree 3 vertices
const PR3_PARKED = 0x08 # the vertex failed a test because of `width`
const PR3_DELETE = 0x10 # the vertex has been eliminated

function pr3(weights::AbstractVector{W}, graph::AbstractGraph, width::Number) where {W <: Number}
    return pr3(weights, graph, convert(W, width))
end

function pr3(weights::AbstractVector{W}, graph::AbstractGraph{V}, width::W) where {W <: Number, V}
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

    return pr3_impl!(
        weight, degree, number, status, marker, source, target, begptr,
        endptr, invptr, stack1, stack2, stack3, stack4, stack5, stack6,
        stack7, stack8, stack0, tmpptr, totdeg, width, graph)
end

"""
    pr3_impl!(...)

Pre-processing for Triangulation of Probabilistic Networks
Bodlaender, Koster, Eijkhof, and van der Gaag

Preprocessing Rules for Triangulation of Probabilistic Networks
Bodlaender, Koster, Eijkhof, and van der Gaag

Safe Reduction Rules for Weighted Treewidth
Eijkhof, Bodlaender, and Koster

Preprocess a graph by applying a set of *safe* reduction
rules to vertices of degree at most three.
  - islet (degree 0, simplicial)
  - twig (degree 1, simplicial)
  - series (degree 2, simplicial or almost simplicial)
  - triangle (degree 3, simplicial or almost simplicial)
  - buddy
  - cube

The algorithm is driven by work queues: a vertex is (re-)tested
only when something that its test depends on has changed. These
dependencies are
  - the neighborhood of the vertex,
  - the adjacency between its neighbors,
  - the neighborhoods of its degree-3 neighbors (cube rule), and
  - the lower bound `width`.
When the queues are empty, no rule applies to any vertex.

The output is a reduced graph R and a sequence (v₁, ..., vₙ) of
eliminated vertices. Any minimum-treewidth elimination ordering
(w₁, ..., wₘ) of R can be appended to the sequence to create
a minimum-treewidth elimination ordering of the input graph.
   (v₁, ..., vₙ, w₁, ..., wₘ).

The algorithm employs a data structure called a *quotient graph.*
It is a directed graph with two types of vertices: elements and
supernodes. The arcs in the graph obey the following invariant
  - every supernode w has at most one predecessor v, which must
    be an element
We say that an element w is *reachable* by another element v if
there exists a path (v, x₁, ..., xₙ, w) from v to w through
supernodes {x₁, ..., xₙ}. Reachability is a symmetric, irreflexive
relation on the set of elements.

The state of the algorithm is a collection of arrays and scalars.

  - quotient graph:
    - `source`: the owner of an arc slot
    - `target`: the target vertex of an arc
      - positive vertices are elements
      - negative vertices are supernodes
    - `begptr`: the first arc slot owned by a vertex
    - `endptr`: one past the last arc incident to a vertex
    - `invptr`: the reverse of an arc
  - vertex data:
    - `weight`: vertex weight
    - `degree`: weighted degree (weight of the closed neighborhood)
    - `number`: degree
    - `status`: queue membership and elimination flags
    - `marker`: marker array
  - work queues:
    - `stack1`: vertices of degree at most 1
    - `stack2`: degree 2 vertices whose neighborhood has changed
    - `stack3`: degree 3 vertices whose neighborhood has changed
    - `stack7`: vertices whose last test failed because of `width`
  - miscellaneous:
    - `stack4`: eliminated vertices
    - `stack5`: traversal stack
    - `stack6`: traversal stack (nested traversals)
    - `stack8`: scratch space
  - scalars:
    - `width`: treewidth lower bound
    - `parked`: the value of `width` when the first vertex was parked
    - `tag`: a running marker tag
    - `hi1`, `hi2`, `hi3`, `hi4`, `hi7`: the heights of `stack1`,
      `stack2`, `stack3`, `stack4`, `stack7`

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
function pr3_impl!(
        weight::AbstractVector{W},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        status::AbstractVector{UInt8},
        marker::AbstractVector{V},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        stack3::AbstractVector{V},
        stack4::AbstractVector{V},
        stack5::AbstractVector{V},
        stack6::AbstractVector{V},
        stack7::AbstractVector{V},
        stack8::AbstractVector{V},
        stack0::AbstractVector{V},
        tmpptr::AbstractVector{E},
        totdeg::W,
        width::W,
        graph::AbstractGraph{V},
    ) where {W, V, E}
    n = nv(graph)

    @assert n <= length(weight)
    @assert n <= length(degree)
    @assert n <= length(number)
    @assert n <= length(status)
    @assert n <= length(marker)
    @assert de(graph) <= length(source)
    @assert de(graph) <= length(target)
    @assert n < length(begptr)
    @assert n <= length(endptr)
    @assert de(graph) <= length(invptr)
    @assert n <= length(stack1)
    @assert n <= length(stack2)
    @assert n <= length(stack3)
    @assert n <= length(stack4)
    @assert n <= length(stack5)
    @assert n <= length(stack6)
    @assert n <= length(stack7)
    @assert n <= length(stack8)
    @assert n <= length(stack0)
    @assert n < length(tmpptr)

    # initialize the quotient graph and the work queues
    width, parked, hi1, hi2, hi3 = pr3_init!(weight, degree, number, status,
        marker, source, target, begptr, endptr, invptr, stack1, stack2,
        stack3, tmpptr, totdeg, width, graph)

    # apply reduction rules until no more apply
    width, hi4 = pr3_loop!(weight, degree, number, status, marker, source,
        target, begptr, endptr, invptr, stack1, stack2, stack3, stack4,
        stack5, stack6, stack7, stack8, width, parked, one(V), hi1,
        hi2, hi3, zero(V), zero(V))

    # if no vertex was eliminated, then the quotient graph is
    # the input graph, stored in `begptr` and `target`
    if iszero(hi4)
        @inbounds for v in oneto(n)
            stack0[v] = v
        end

        @inbounds m = begptr[n + one(V)] - one(E)
        kernel = BipartiteGraph(n, n, m, begptr, target)
        return kernel, stack4, stack0, width
    end

    # construct the reduced graph R = (V, E):
    #  - V is the set of elements in the quotient graph
    #  - E contains an arc (v, w) if w is reachable by v in the quotient graph
    m, n = pr3_make!(stack4, stack5, target, begptr, endptr, invptr, stack0,
        number, tmpptr, source, hi4, n)

    # `kernel` is the reduced graph
    kernel = BipartiteGraph(n, n, m, tmpptr, source)
    return kernel, stack4, stack0, width
end

function pr3_init!(
        weight::AbstractVector{W},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        status::AbstractVector{UInt8},
        marker::AbstractVector{V},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        stack3::AbstractVector{V},
        tmpptr::AbstractVector{E},
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
        marker[v] = zero(V)
        begptr[v] = p

        # `deg` is the weighted degree of `v`
        deg = weight[v]

        # `num` is the unweighted degree of `v`
        num = zero(V)

        # `prv` is the previous neighbor of `v`
        prv = zero(V)

        # for all neighbors `w` of `v`...
        for w in neighbors(graph, v)
            # ignore self loops
            if v != w
                # `p` is the arc (`v`, `w`)
                source[p] = v; target[p] = w; p += one(E)

                # check if the neighborhood of `v` is sorted
                sorted = sorted && prv < w; prv = w

                # increase the weighted degree of `v` by
                # the weight of `w`
                deg += weight[w]

                # increment the degree of `v`
                num += one(V)
            end
        end

        endptr[v] = p

        # update the minimum weighted degree
        mindeg = min(mindeg, deg)

        # store the weighted degree of `v`
        degree[v] = deg

        # store the degree of `v`
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

    # the weighted treewidth of the input graph is no less
    # than its minimum weighted degree
    width = max(width, mindeg); parked = width

    # the heights of the degree queues
    hi1 = hi2 = hi3 = zero(V)

    # add vertices to the work queues in reverse order, so
    # that they are tested in increasing order
    @inbounds for v in reverse(vertices(graph))
        hi1, hi2, hi3 = pr3_touch!(number, status, stack1, stack2, stack3, hi1, hi2, hi3, v)
    end

    return width, parked, hi1, hi2, hi3
end

function pr3_loop!(
        weight::AbstractVector{W},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        status::AbstractVector{UInt8},
        marker::AbstractVector{V},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        stack3::AbstractVector{V},
        stack4::AbstractVector{V},
        stack5::AbstractVector{V},
        stack6::AbstractVector{V},
        stack7::AbstractVector{V},
        stack8::AbstractVector{V},
        width::W,
        parked::W,
        tag::V,
        hi1::V,
        hi2::V,
        hi3::V,
        hi4::V,
        hi7::V,
    ) where {W, V, E}
    @inbounds while true
        if ispositive(hi1)
            # `v` is a vertex with degree 0 or 1
            hi1, v = pr3_stack_pop!(stack1, hi1)
            flag = status[v]; status[v] = flag & ~PR3_STACK1

            if iszero(flag & PR3_DELETE)
                if iszero(number[v])
                    width, hi4 = pr3_islet!(degree, status, stack4, width, hi4, v)
                elseif isone(number[v])
                    width, hi1, hi2, hi3, hi4 = pr3_twig!(weight, degree, number,
                        status, source, target, begptr, endptr, invptr, stack1,
                        stack2, stack3, stack4, stack5, stack8, width, hi1, hi2,
                        hi3, hi4, v)
                end
            end
        elseif ispositive(hi2)
            # `v` is a vertex with degree 2 (probably)
            hi2, v = pr3_stack_pop!(stack2, hi2)
            flag = status[v]; status[v] = flag & ~PR3_STACK2

            if iszero(flag & PR3_DELETE) && istwo(number[v])
                width, parked, hi1, hi2, hi3, hi4, hi7 = pr3_series!(weight,
                    degree, number, status, source, target, begptr, endptr,
                    invptr, stack1, stack2, stack3, stack4, stack5, stack7,
                    stack8, width, parked, hi1, hi2, hi3, hi4, hi7, v)
            end
        elseif ispositive(hi3)
            # `v` is a vertex with degree 3 (probably)
            hi3, v = pr3_stack_pop!(stack3, hi3)
            flag = status[v]; status[v] = flag & ~PR3_STACK3

            if iszero(flag & PR3_DELETE) && isthree(number[v])
                width, parked, tag, hi1, hi2, hi3, hi4, hi7 = pr3_triangle!(
                    weight, degree, number, status, marker, source, target,
                    begptr, endptr, invptr, stack1, stack2, stack3, stack4,
                    stack5, stack6, stack7, stack8, width, parked, tag,
                    hi1, hi2, hi3, hi4, hi7, v)
            end
        elseif ispositive(hi7) && parked < width
            # the lower bound has increased: re-test every
            # vertex that failed a test because of it
            hi1, hi2, hi3, hi7 = pr3_unpark!(number, status, stack1, stack2,
                stack3, stack7, hi1, hi2, hi3, hi7)
        else
            break
        end
    end

    return width, hi4
end

# eliminate a vertex `v` with degree 0
function pr3_islet!(
        degree::AbstractVector{W},
        status::AbstractVector{UInt8},
        stack4::AbstractVector{V},
        width::W,
        hi4::V,
        v::V,
    ) where {W, V}
    # add `v` to the stack of eliminated vertices
    hi4 = pr3_delete!(status, stack4, hi4, v)

    # `v` is simplicial: update the lower bound
    @inbounds width = max(width, degree[v])
    return width, hi4
end

# eliminate a vertex `v` with degree 1
function pr3_twig!(
        weight::AbstractVector{W},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        status::AbstractVector{UInt8},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        stack3::AbstractVector{V},
        stack4::AbstractVector{V},
        stack5::AbstractVector{V},
        stack8::AbstractVector{V},
        width::W,
        hi1::V,
        hi2::V,
        hi3::V,
        hi4::V,
        v::V,
    ) where {W, V, E}
    # add `v` to the stack of eliminated vertices
    hi4 = pr3_delete!(status, stack4, hi4, v)

    # `v` is simplicial: update the lower bound
    @inbounds width = max(width, degree[v])

    # `w` is the unique element reachable by `v`
    p, w = pr3_reach1!(stack5, target, begptr, endptr, invptr, v)

    # remove `v` from the reachable set of `w`
    @inbounds pr3_reach_del!(source, target, endptr, invptr, invptr[p])

    # decrement the degree of `w`
    @inbounds number[w] -= one(V)

    # decrease the weighted degree of `w` by the weight
    # of `v`
    @inbounds degree[w] -= weight[v]

    # the neighborhood of `w` has changed
    hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
        stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, w)
    return width, hi1, hi2, hi3, hi4
end

# test a vertex `v` with degree 2
function pr3_series!(
        weight::AbstractVector{W},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        status::AbstractVector{UInt8},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        stack3::AbstractVector{V},
        stack4::AbstractVector{V},
        stack5::AbstractVector{V},
        stack7::AbstractVector{V},
        stack8::AbstractVector{V},
        width::W,
        parked::W,
        hi1::V,
        hi2::V,
        hi3::V,
        hi4::V,
        hi7::V,
        v::V,
    ) where {W, V, E}
    tol = tolerance(W)

    # `w` and `ww` are the elements reachable by `v`
    p, w, pp, ww = pr3_reach2!(stack5, target, begptr, endptr, invptr, v)

    # sort `w` and `ww` by degree
    @inbounds if number[ww] < number[w]
        p, w, pp, ww = pp, ww, p, w
    end

    @inbounds if pr3_adjacent!(number, stack5, target, begptr, endptr, invptr, w, ww)
        # w ─── ww
        # │  ╱
        # v

        # add `v` to the stack of eliminated vertices
        hi4 = pr3_delete!(status, stack4, hi4, v)

        # `v` is simplicial: update the lower bound
        width = max(width, degree[v])

        # remove `v` from the reachable sets of `w` and `ww`
        pr3_reach_del!(source, target, endptr, invptr, invptr[p])
        pr3_reach_del!(source, target, endptr, invptr, invptr[pp])

        # decrement the degree of `w` and `ww`
        number[w] -= one(V)
        number[ww] -= one(V)

        # decrease the weighted degree of `w` and `ww` by the
        # weight of `v`
        degree[w] -= weight[v]
        degree[ww] -= weight[v]

        # the neighborhoods of `w` and `ww` have changed
        hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
            stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, w)
        hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
            stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, ww)
    elseif min(weight[w], weight[ww]) < weight[v] + tol
        if degree[v] < width + tol
            # w     ww
            # │  ╱
            # v

            # add `v` to the stack of eliminated vertices
            hi4 = pr3_delete!(status, stack4, hi4, v)

            pinv = invptr[p]
            ppinv = invptr[pp]

            # replace `v` with `ww` in the reachable set of `w`
            target[pinv] = ww; invptr[pinv] = ppinv

            # replace `v` with `w` in the reachable set of `ww`
            target[ppinv] = w; invptr[ppinv] = pinv

            # increase the weighted degree of `w` by the weight
            # of `ww` and decrease it by the weight of `v`
            degree[w] -= (weight[v] - weight[ww])

            # increase the weighted degree of `ww` by the weight
            # of `w` and decrease it by the weight of `v`
            degree[ww] -= (weight[v] - weight[w])

            # the neighborhoods of `w` and `ww` have changed
            hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
                stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, w)
            hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
                stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, ww)

            # the edge {`w`, `ww`} was created
            hi1, hi2, hi3 = pr3_edge!(number, status, stack1, stack2, stack3,
                stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, w, ww)
        else
            # the test failed because of the lower bound
            parked, hi7 = pr3_park!(status, stack7, parked, hi7, width, v)
        end
    end

    return width, parked, hi1, hi2, hi3, hi4, hi7
end

# test a vertex `v` with degree 3
function pr3_triangle!(
        weight::AbstractVector{W},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        status::AbstractVector{UInt8},
        marker::AbstractVector{V},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        stack3::AbstractVector{V},
        stack4::AbstractVector{V},
        stack5::AbstractVector{V},
        stack6::AbstractVector{V},
        stack7::AbstractVector{V},
        stack8::AbstractVector{V},
        width::W,
        parked::W,
        tag::V,
        hi1::V,
        hi2::V,
        hi3::V,
        hi4::V,
        hi7::V,
        v::V,
    ) where {W, V, E}
    tol = tolerance(W)

    # `w`, `ww`, and `www` are the elements reachable by `v`
    p, w, pp, ww, ppp, www = pr3_reach3!(stack5, target, begptr, endptr, invptr, v)

    # sort `w`, `ww`, and `www` by degree
    (p, w), (pp, ww), (ppp, www) = sortthree((p, w), (pp, ww), (ppp, www)) do (p, w)
        @inbounds wnum = number[w]
        return wnum
    end

    # `f` is true if `ww` is reachable by `w`
    # `ff` is true if `www` is reachable by `w`
    f, ff = pr3_adjacent2!(stack5, target, begptr, endptr, invptr, w, ww, www)

    # `fff` is true if `www` is reachable by `ww`
    fff = pr3_adjacent!(number, stack5, target, begptr, endptr, invptr, ww, www)

    # sort `f`, `ff`, and `fff` by true value
    if f
        pp, ppp = ppp, pp
        ww, www = www, ww
        f, ff = ff, f
    end

    if ff
        p, pp = pp, p
        w, ww = ww, w
        ff, fff = fff, ff
    end

    if f
        pp, ppp = ppp, pp
        ww, www = www, ww
        f, ff = ff, f
    end

    @inbounds if f
        # w ─── ww
        # │  ╳  │
        # v ─── www

        # add `v` to the stack of eliminated vertices
        hi4 = pr3_delete!(status, stack4, hi4, v)

        # `v` is simplicial: update the lower bound
        width = max(width, degree[v])

        # remove `v` from the reachable sets of `w`, `ww`, and `www`
        pr3_reach_del!(source, target, endptr, invptr, invptr[p])
        pr3_reach_del!(source, target, endptr, invptr, invptr[pp])
        pr3_reach_del!(source, target, endptr, invptr, invptr[ppp])

        # decrement the degrees of `w`, `ww`, and `www`
        number[w] -= one(V)
        number[ww] -= one(V)
        number[www] -= one(V)

        # decrease the weighted degrees of `w`, `ww`, and `www`
        # by the weight of `v`
        degree[w] -= weight[v]
        degree[ww] -= weight[v]
        degree[www] -= weight[v]

        # the neighborhoods of `w`, `ww`, and `www` have changed
        hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
            stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, w)
        hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
            stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, ww)
        hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
            stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, www)
    elseif ff
        if min(weight[w], weight[ww]) < weight[v] + tol
            if degree[v] < width + tol
                # w     ww
                # │  ╳  │
                # v ─── www

                # add `v` to the stack of eliminated vertices
                hi4 = pr3_delete!(status, stack4, hi4, v)

                # remove `v` from the reachable set of `www`
                pr3_reach_del!(source, target, endptr, invptr, invptr[ppp])

                # decrement the degree of `www`
                number[www] -= one(V)

                # decrease the weighted degree of `www` by the
                # weight of `v`
                degree[www] -= weight[v]

                pinv = invptr[p]
                ppinv = invptr[pp]

                # replace `v` with `ww` in the reachable set of `w`
                target[pinv] = ww; invptr[pinv] = ppinv

                # replace `v` with `w` in the reachable set of `ww`
                target[ppinv] = w; invptr[ppinv] = pinv

                # increase the weighted degree of `w` by the weight of
                # `ww` and decrease it by the weight of `v`
                degree[w] -= (weight[v] - weight[ww])

                # increase the weighted degree of `ww` by the weight of
                # `w` and decrease it by the weight of `v`
                degree[ww] -= (weight[v] - weight[w])

                # the neighborhoods of `w`, `ww`, and `www` have changed
                hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
                    stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, w)
                hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
                    stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, ww)
                hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
                    stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, www)

                # the edge {`w`, `ww`} was created
                hi1, hi2, hi3 = pr3_edge!(number, status, stack1, stack2, stack3,
                    stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, w, ww)
            else
                # the test failed because of the lower bound
                parked, hi7 = pr3_park!(status, stack7, parked, hi7, width, v)
            end
        end
    elseif fff
        if weight[w] < weight[v] + tol
            if degree[v] < width + tol
                # w     ww
                # │  ╱  │
                # v ─── www

                # add `v` to the stack of eliminated vertices
                hi4 = pr3_delete!(status, stack4, hi4, v)

                # increment the degree of `w`
                number[w] += one(V)

                pinv = invptr[p]
                ppinv = invptr[pp]
                pppinv = invptr[ppp]

                # turn `v` into a supernode
                target[pinv] = -v

                # replace `v` with `w` in the reachable sets of `ww` and `www`
                target[ppinv] = w
                target[pppinv] = w

                # increase the weighted degree of `w` by the weights of
                # `ww` and `www` and decrease it by the weight of `v`
                degree[w] -= (weight[v] - weight[ww] - weight[www])

                # increase the weighted degree of `ww` by the weight
                # of `w` and decrease it by the weight of `v`
                degree[ww] -= (weight[v] - weight[w])

                # increase the weighted degree of `www` by the weight
                # of `w` and decrease it by the weight of `v`
                degree[www] -= (weight[v] - weight[w])

                # remove `w` from the reachable set of `v`
                pr3_reach_del!(source, target, endptr, invptr, p)

                # the neighborhoods of `w`, `ww`, and `www` have changed
                hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
                    stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, w)
                hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
                    stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, ww)
                hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
                    stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, www)

                # the edges {`w`, `ww`} and {`w`, `www`} were created
                hi1, hi2, hi3 = pr3_edge!(number, status, stack1, stack2, stack3,
                    stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, w, ww)
                hi1, hi2, hi3 = pr3_edge!(number, status, stack1, stack2, stack3,
                    stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, w, www)
            else
                # the test failed because of the lower bound
                parked, hi7 = pr3_park!(status, stack7, parked, hi7, width, v)
            end
        end
    else
        # w     ww
        # │  ╱
        # v ─── www

        # `flag` is true if a rule was applied
        flag = false

        # `park` is true if a test failed because
        # of the lower bound
        park = false

        if min(weight[w], weight[ww], weight[www]) >= weight[v] + tol
            # `v` cannot be contracted to any of its neighbors,
            # so it has no buddy
        elseif degree[v] < width + tol
            # search for a buddy `vv`
            vv, park = pr3_buddy_search!(weight, degree, number, stack5, stack6,
                target, begptr, endptr, invptr, width, v, w, ww, www)

            # if a buddy was found ...
            if ispositive(vv)
                # w ────────── vv
                # │         ╱  │
                # │     ww     │
                # │  ╱         │
                # v ────────── www

                # add `v` and `vv` to the stack of eliminated vertices
                hi4 = pr3_delete!(status, stack4, hi4, v)
                hi4 = pr3_delete!(status, stack4, hi4, vv)

                # eliminate `v` and `vv`
                pr3_buddy!(stack5, degree, source, target, begptr, endptr,
                    invptr, weight, v, vv, p, w, pp, ww, ppp, www)

                # the neighborhoods of `w`, `ww`, and `www` have changed
                hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
                    stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, w)
                hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
                    stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, ww)
                hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
                    stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, www)

                # the edges {`w`, `ww`}, {`w`, `www`}, and
                # {`ww`, `www`} were created
                hi1, hi2, hi3 = pr3_edge!(number, status, stack1, stack2, stack3,
                    stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, w, ww)
                hi1, hi2, hi3 = pr3_edge!(number, status, stack1, stack2, stack3,
                    stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, w, www)
                hi1, hi2, hi3 = pr3_edge!(number, status, stack1, stack2, stack3,
                    stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, ww, www)

                flag = true
            end
        else
            park = true
        end

        if !flag && isthree(number[w]) && isthree(number[ww]) && isthree(number[www])
            if max(degree[w], degree[ww], degree[www]) < width + tol
                # search for a cube `x`, `y`, and `z`
                q, qq, r, rr, s, ss, x, y, z = pr3_cube_reach!(weight, stack5,
                    target, begptr, endptr, invptr, v, w, ww, www)

                # if a cube was found...
                if ispositive(q)
                    #       x
                    #    ╱     ╲
                    # w           ww
                    # │  ╲     ╱  │
                    # │     v     │
                    # z     │     y
                    #    ╲  │  ╱
                    #      www

                    # `v` will be simplicial after eliminating `w`, `ww`, and
                    # `www`, with neighborhood {`x`, `y`, `z`}: update the lower
                    # bound
                    @inbounds width = max(width, weight[v] + weight[x] + weight[y] + weight[z])

                    # add `w`, `ww`, `www`, and `v` to the stack of
                    # eliminated vertices
                    hi4 = pr3_delete!(status, stack4, hi4, w)
                    hi4 = pr3_delete!(status, stack4, hi4, ww)
                    hi4 = pr3_delete!(status, stack4, hi4, www)
                    hi4 = pr3_delete!(status, stack4, hi4, v)

                    # eliminate `w`, `ww`, `www`, and `v`
                    tag = pr3_cube!(weight, degree, number, marker, source,
                        target, begptr, endptr, invptr, stack5, tag, w, ww,
                        www, q, qq, r, rr, s, ss, x, y, z)

                    # the neighborhoods of `x`, `y`, and `z` have changed
                    hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
                        stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, x)
                    hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
                        stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, y)
                    hi1, hi2, hi3 = pr3_touch_nbr!(number, status, stack1, stack2, stack3,
                        stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, z)

                    # the edges {`x`, `y`}, {`y`, `z`}, and {`z`, `x`}
                    # may have been created
                    hi1, hi2, hi3 = pr3_edge!(number, status, stack1, stack2, stack3,
                        stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, x, y)
                    hi1, hi2, hi3 = pr3_edge!(number, status, stack1, stack2, stack3,
                        stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, y, z)
                    hi1, hi2, hi3 = pr3_edge!(number, status, stack1, stack2, stack3,
                        stack5, stack8, target, begptr, endptr, invptr, hi1, hi2, hi3, z, x)

                    flag = true
                end
            else
                park = true
            end
        end

        if !flag && park
            # the test failed because of the lower bound
            parked, hi7 = pr3_park!(status, stack7, parked, hi7, width, v)
        end
    end

    return width, parked, tag, hi1, hi2, hi3, hi4, hi7
end

# search for a buddy of `v`: a vertex `vv` != `v` whose
# neighborhood is {`w`, `ww`, `www`}
function pr3_buddy_search!(
        weight::AbstractVector{W},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        stack5::AbstractVector{V},
        stack6::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        width::W,
        v::V,
        w::V,
        ww::V,
        www::V,
    ) where {W, V, E}
    tol = tolerance(W)

    # `vwgt` is the weight of `v`
    @inbounds vwgt = weight[v]

    # sort the weights of `w`, `ww`, and `www`
    @inbounds minwgt, medwgt, maxwgt = sortthree(weight[w], weight[ww], weight[www])

    # `vv` is a buddy of `v`
    # `park` is nonzero if a buddy was rejected because of `width`
    #
    # the accumulator is a homogeneous tuple: a loop-carried
    # tuple of type (Int, Bool) is not kept in registers
    (vv, park), _ = pr3_reach_until!((zero(V), zero(V)), stack5, target,
            begptr, endptr, invptr, w) do (vv, park), _, x

        if x != v && isthree(number[x])
            @inbounds xwgt = weight[x]

            if minwgt < min(vwgt, xwgt) + tol && medwgt < max(vwgt, xwgt) + tol
                # `x` is a buddy if `ww` and `www` are reachable by `x`
                if pr3_contains2!(stack6, target, begptr, endptr, invptr, x, ww, www)
                    @inbounds xdeg = degree[x]

                    if xdeg < width + tol
                        return (x, park), true
                    else
                        park = one(V)
                    end
                end
            end
        end

        return (vv, park), false
    end

    return vv, ispositive(park)
end

# remove a vertex from the graph
function pr3_delete!(
        status::AbstractVector{UInt8},
        stack4::AbstractVector{V},
        hi4::V,
        v::V,
    ) where {V}
    @inbounds status[v] |= PR3_DELETE
    hi4 = pr3_stack_add!(stack4, hi4, v)
    return hi4
end

# the neighborhood of `v` has changed: add it to the
# appropriate work queue
function pr3_touch!(
        number::AbstractVector{V},
        status::AbstractVector{UInt8},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        stack3::AbstractVector{V},
        hi1::V,
        hi2::V,
        hi3::V,
        v::V,
    ) where {V}
    @inbounds flag = status[v]

    @inbounds if iszero(flag & PR3_DELETE)
        num = number[v]

        if num <= one(V)
            if iszero(flag & PR3_STACK1)
                status[v] = flag | PR3_STACK1
                hi1 = pr3_stack_add!(stack1, hi1, v)
            end
        elseif istwo(num)
            if iszero(flag & PR3_STACK2)
                status[v] = flag | PR3_STACK2
                hi2 = pr3_stack_add!(stack2, hi2, v)
            end
        elseif isthree(num)
            if iszero(flag & PR3_STACK3)
                status[v] = flag | PR3_STACK3
                hi3 = pr3_stack_add!(stack3, hi3, v)
            end
        end
    end

    return hi1, hi2, hi3
end

# the neighborhood of `v` has changed: add `v` to a work queue.
# if `v` has degree 3, it may belong to a cube centered at
# one of its degree 3 neighbors, so add them to a queue too.
function pr3_touch_nbr!(
        number::AbstractVector{V},
        status::AbstractVector{UInt8},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        stack3::AbstractVector{V},
        stack5::AbstractVector{V},
        stack8::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        hi1::V,
        hi2::V,
        hi3::V,
        v::V,
    ) where {V, E}
    hi1, hi2, hi3 = pr3_touch!(number, status, stack1, stack2, stack3, hi1, hi2, hi3, v)

    @inbounds if isthree(number[v])
        # collect the degree-3 neighbors not yet in the queue, then
        # add them (the traversal reads nothing that `pr3_touch!`
        # would change, so collecting first keeps the result the same)
        k = pr3_reach!(zero(V), stack5, target, begptr, endptr, invptr, v) do k, _, w
            @inbounds if isthree(number[w]) && iszero(status[w] & PR3_STACK3)
                k += one(V); stack8[k] = w
            end

            return k
        end

        for i in oneto(k)
            hi1, hi2, hi3 = pr3_touch!(number, status, stack1, stack2, stack3, hi1, hi2, hi3, stack8[i])
        end
    end

    return hi1, hi2, hi3
end

# the edge {`v`, `w`} was created: every vertex of degree 2 or 3
# adjacent to both `v` and `w` must be tested again
function pr3_edge!(
        number::AbstractVector{V},
        status::AbstractVector{UInt8},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        stack3::AbstractVector{V},
        stack5::AbstractVector{V},
        stack8::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        hi1::V,
        hi2::V,
        hi3::V,
        v::V,
        w::V,
    ) where {V, E}
    # search the smaller reachable set: `a` is the endpoint
    # with smaller degree and `b` is the other endpoint
    @inbounds a, b = number[w] < number[v] ? (w, v) : (v, w)

    # collect the untested vertices of degree 2 and 3 adjacent to `a`
    num = pr3_reach!(zero(V), stack5, target, begptr, endptr, invptr, a) do num, _, x

        if x != b
            @inbounds xnum = number[x]
            @inbounds xflag = status[x]

            if (istwo(xnum) && iszero(xflag & PR3_STACK2)) || (isthree(xnum) && iszero(xflag & PR3_STACK3))
                num += one(V); @inbounds stack8[num] = x
            end
        end

        return num
    end

    # add the ones adjacent to `b` to a work queue
    @inbounds for i in oneto(num)
        x = stack8[i]

        if pr3_adjacent!(number, stack5, target, begptr, endptr, invptr, x, b)
            hi1, hi2, hi3 = pr3_touch!(number, status, stack1, stack2, stack3, hi1, hi2, hi3, x)
        end
    end

    return hi1, hi2, hi3
end

# the lower bound has increased: re-test the parked vertices
function pr3_unpark!(
        number::AbstractVector{V},
        status::AbstractVector{UInt8},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        stack3::AbstractVector{V},
        stack7::AbstractVector{V},
        hi1::V,
        hi2::V,
        hi3::V,
        hi7::V,
    ) where {V}
    @inbounds for i in oneto(hi7)
        v = stack7[i]
        status[v] &= ~PR3_PARKED
        hi1, hi2, hi3 = pr3_touch!(number, status, stack1, stack2, stack3, hi1, hi2, hi3, v)
    end

    return hi1, hi2, hi3, zero(V)
end

# a test of `v` failed because of the lower bound
function pr3_park!(
        status::AbstractVector{UInt8},
        stack7::AbstractVector{V},
        parked::W,
        hi7::V,
        width::W,
        v::V,
    ) where {W, V}
    @inbounds flag = status[v]

    @inbounds if iszero(flag & PR3_PARKED)
        if iszero(hi7)
            parked = width
        end

        status[v] = flag | PR3_PARKED
        hi7 = pr3_stack_add!(stack7, hi7, v)
    end

    return parked, hi7
end

# returns true if `w` is reachable by `v`
function pr3_adjacent!(
        number::AbstractVector{V},
        stack5::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        v::V,
        w::V,
    ) where {V, E}
    # search the smaller reachable set
    @inbounds a, b = number[w] < number[v] ? (w, v) : (v, w)

    flag, _ = pr3_reach_until!(false, stack5, target,
            begptr, endptr, invptr, a) do flag, _, x
        flag = x == b
        return flag, flag
    end

    return flag
end

# returns (`w` reachable by `v`, `ww` reachable by `v`)
function pr3_adjacent2!(
        stack5::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        v::V,
        w::V,
        ww::V,
    ) where {V, E}
    # bit 1 of `flag` is set if `w` is reachable by `v`
    # bit 2 of `flag` is set if `ww` is reachable by `v`
    flag, _ = pr3_reach_until!(zero(V), stack5, target,
            begptr, endptr, invptr, v) do flag, _, x
        flag |= ifelse(x == w, one(V), zero(V)) | ifelse(x == ww, two(V), zero(V))
        return flag, flag == three(V)
    end

    return isodd(flag), flag >= two(V)
end

# returns true if `w` and `ww` are both reachable by `v`
# uses the nested traversal stack
@noinline function pr3_contains2!(
        stack6::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        v::V,
        w::V,
        ww::V,
    ) where {V, E}
    flag, _ = pr3_reach_until!(zero(V), stack6, target,
            begptr, endptr, invptr, v) do flag, _, x
        flag |= ifelse(x == w, one(V), zero(V)) | ifelse(x == ww, two(V), zero(V))
        return flag, flag == three(V)
    end

    return flag == three(V)
end

function pr3_reach1!(
        stack5::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        v::V,
    ) where {V, E}
    p = zero(E); w = zero(V)

    p, w = pr3_reach!((p, w), stack5, target,
            begptr, endptr, invptr, v) do _, p, w
        return (p, w)
    end

    return p, w
end

function pr3_reach2!(
        stack5::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        v::V,
    ) where {V, E}
    p = pp = zero(E); w = ww = zero(V)

    p, w, pp, ww = pr3_reach!((p, w, pp, ww), stack5, target,
            begptr, endptr, invptr, v) do (p, w, pp, ww), ppp, www

        if iszero(w)
            p, w = ppp, www
        else
            pp, ww = ppp, www
        end

        return (p, w, pp, ww)
    end

    return p, w, pp, ww
end

function pr3_reach3!(
        stack5::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        v::V,
    ) where {V, E}
    return pr3_3_reach!(stack5, target, begptr, endptr, invptr, v)
end

function pr3_make!(
        stack4::AbstractVector{V},
        stack5::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        inj::AbstractVector{V},
        prj::AbstractVector{V},
        ptr::AbstractVector{E},
        tgt::AbstractVector{V},
        hi4::V,
        n::V,
    ) where {V, E}

    # mark eliminated vertices with -1
    @inbounds for i in oneto(hi4)
        w = stack4[i]; prj[w] = -one(V)
    end

    # `v` is a vertex in the reduced graph
    v = one(V)

    # for every vertex `w`...
    @inbounds for w in oneto(n)

        # if `w` was not eliminated
        if !isnegative(prj[w])
            # associate `v` and `w`
            prj[w] = v
            inj[v] = w

            # increment `v`
            v += one(V)
        end
    end

    # `v` is a vertex in the reduced graph
    v = one(V)

    # `p` is the first arc incident to `v`
    @inbounds ptr[v] = p = one(E)

    # for all vertices `v` in the reduced graph...
    @inbounds while v + hi4 <= n
        # `w` is the corresponding element in the
        # quotient graph
        w = inj[v]

        p =  pr3_reach!(p, stack5,
            target, begptr, endptr, invptr, w) do p, _, x

            # `x` is an element reachable by `w`
            @inbounds tgt[p] = prj[x]; p += one(E)
            return p
        end

        # update `v` and `p`
        v += one(V); ptr[v] = p
    end

    # `m` is the number of arcs in the reduced graph
    m = p - one(E)

    # `n` is the number of vertices in the reduced graph
    n = v - one(V)
    return m, n
end

function pr3_buddy!(
        stack5::AbstractVector{V},
        degree::AbstractVector{W},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        weight::AbstractVector{W},
        v::V,
        vv::V,
        p::E,
        w::V,
        pp::E,
        ww::V,
        ppp::E,
        www::V,
    ) where {W, V, E}
    # w ────────── vv
    # │         ╱  │
    # │     ww     │
    # │  ╱         │
    # v ────────── www
    q = qq = qqq = zero(E)

    q, qq, qqq = pr3_reach!((q, qq, qqq),
        stack5, target, begptr, endptr, invptr, vv) do (q, qq, qqq), qnxt, wnxt

        if w == wnxt
            q = qnxt
        elseif ww == wnxt
            qq = qnxt
        elseif www == wnxt
            qqq = qnxt
        end

        return (q, qq, qqq)
    end

    @inbounds pinv = invptr[p]
    @inbounds ppinv = invptr[pp]
    @inbounds pppinv = invptr[ppp]

    @inbounds qinv = invptr[q]
    @inbounds qqinv = invptr[qq]
    @inbounds qqqinv = invptr[qqq]

    # replace `v` with `ww` in the reachable set of `w`
    @inbounds target[pinv] = ww
    @inbounds invptr[pinv] = qqinv

    # replace `v` with `www` in the reachable set of `ww`
    @inbounds target[ppinv] = www
    @inbounds invptr[ppinv] = qqqinv

    # replace `v` with `w` in the reachable set of `www`
    @inbounds target[pppinv] = w
    @inbounds invptr[pppinv] = qinv

    # replace `vv` with `www` in the reachable set of `w`
    @inbounds target[qinv] = www
    @inbounds invptr[qinv] = pppinv

    # replace `vv` with `w` in the reachable set of `ww`
    @inbounds target[qqinv] = w
    @inbounds invptr[qqinv] = pinv

    # replace `vv` with `ww` in the reachable set of `www`
    @inbounds target[qqqinv] = ww
    @inbounds invptr[qqqinv] = ppinv

    # decrease the weighted degree of `w` by the weights of `v`
    # and `vv`, and increase it by the weights of `ww` and `www`
    @inbounds degree[w] -= (weight[v] + weight[vv] - weight[ww] - weight[www])

    # decrease the weighted degree of `ww` by the weights of `v`
    # and `vv`, and increase it by the weights of `w` and `www`
    @inbounds degree[ww] -= (weight[v] + weight[vv] - weight[w] - weight[www])

    # decrease the weighted degree of `www` by the weights of `v`
    # and `vv`, and increase it by the weights of `w` and `ww`
    @inbounds degree[www] -= (weight[v] + weight[vv] - weight[w] - weight[ww])
    return
end

function pr3_3_reach!(
        stack5::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        v::V,
    ) where {V, E}
    p = pp = ppp = zero(E)
    w = ww = www = zero(V)

    p, w, pp, ww, ppp, www = pr3_reach!((p, w, pp, ww, ppp, www),
        stack5, target, begptr, endptr, invptr, v) do (p, w, pp, ww, ppp, www), pnxt, wnxt

        if iszero(w)
            p, w = pnxt, wnxt
        elseif iszero(ww)
            pp, ww = pnxt, wnxt
        elseif iszero(www)
            ppp, www = pnxt, wnxt
        end

        return (p, w, pp, ww, ppp, www)
    end

    return (p, w, pp, ww, ppp, www)
end

function pr3_3_mark!(
        marker::AbstractVector{V},
        stack5::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        tag::V,
        w::V,
        ww::V,
    ) where {V, E}
    # mark elements reachable by `w` with `tag`
    pr3_reach!(nothing, stack5,
        target, begptr, endptr, invptr, w) do _, _, x
        @inbounds marker[x] = tag
        return
    end

    # mark elements reachable by `ww` and `w` with `tag` + 1
    # mark elements reachable by `ww` and not `w` with `tag` + 2
    pr3_reach!(nothing, stack5,
        target, begptr, endptr, invptr, ww) do _, _, xx
        @inbounds xxtag = marker[xx]

        if xxtag == tag
            @inbounds marker[xx] = tag + one(V)
        else
            @inbounds marker[xx] = tag + two(V)
        end

        return
    end

    return
end

function pr3_cube_reach!(
        weight::AbstractVector{W},
        stack5::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        v::V,
        w::V,
        ww::V,
        www::V,
    ) where {W, V, E}
    tol = tolerance(W)

    # `x`, `xx` and `xxx` are elements reachable by `w`
    q, x, qq, xx, qqq, xxx = pr3_reach3!(stack5, target, begptr, endptr, invptr, w)

    # `y`, `yy` and `yyy` are elements reachable by `ww`
    r, y, rr, yy, rrr, yyy = pr3_reach3!(stack5, target, begptr, endptr, invptr, ww)

    # `z`, `zz`, and `zzz` are elements reachable by `www`
    s, z, ss, zz, sss, zzz = pr3_reach3!(stack5, target, begptr, endptr, invptr, www)

    # ensure that `v` = `xxx`
    (q, x), (qq, xx), (qqq, xxx) = sortthree((q, x), (qq, xx), (qqq, xxx)) do (_, x)
        return x == v
    end

    # ensure that `v` = `yyy`
    (r, y), (rr, yy), (rrr, yyy) = sortthree((r, y), (rr, yy), (rrr, yyy)) do (_, y)
        return y == v
    end

    # ensure that `v` = `zzz`
    (s, z), (ss, zz), (sss, zzz) = sortthree((s, z), (ss, zz), (sss, zzz)) do (_, z)
        return z == v
    end

    # ensure that `z` is reachable by `w`
    if zz == x || zz == xx
        s, z, ss, zz = ss, zz, s, z
    end

    # copy `z` to avoid boxing
    zc = z

    # ensure that `z` = `xx`
    (q, x), (qq, xx) = sorttwo((q, x), (qq, xx)) do (_, x)
        x == zc
    end

    # copy `x` to avoid boxing
    xc = x

    # ensure that `x` = `yy`
    (r, y), (rr, yy) = sorttwo((r, y), (rr, yy)) do (_, y)
        y == xc
    end

    # `wwgt` is the weight of `w`
    @inbounds wwgt = weight[w]

    # `wwwgt` is the weight of `ww`
    @inbounds wwwgt = weight[ww]

    # `wwwwgt` is the weight of `www`
    @inbounds wwwwgt = weight[www]

    # `xwgt` is the weight of `x`
    @inbounds xwgt = weight[x]

    # `ywgt` is the weight of `y`
    @inbounds ywgt = weight[y]

    # `zwgt` is the weight of `z`
    @inbounds zwgt = weight[z]

    if x != yy || y != zz || z != xx || (
            (wwgt + tol <= xwgt || wwwgt + tol <= ywgt || wwwwgt + tol <= zwgt) &&
            (wwgt + tol <= zwgt || wwwgt + tol <= xwgt || wwwwgt + tol <= ywgt)
        )

        q = qq = r = rr = s = ss = zero(E)
        x = y = z = zero(V)
    end

    return (q, qq, r, rr, s, ss, x, y, z)
end

function pr3_cube!(
        weight::AbstractVector{W},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        marker::AbstractVector{V},
        source::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        stack5::AbstractVector{V},
        tag::V,
        w::V, ww::V, www::V,
        q::E, qq::E, r::E, rr::E,
        s::E, ss::E, x::V, y::V, z::V,
    ) where {W, V, E}
    #       x
    #    ╱     ╲
    # w           ww
    # │  ╲     ╱  │
    # │     v     │
    # z     │     y
    #    ╲  │  ╱
    #      www

    # `usetag` is used to mark vertices
    usetag, tag = pr3_tag!(marker, tag)

    # mark elements reachable by `x` and not `y` with `usetag`
    # mark elements reachable by `y` and `x` with `usetag` + 1
    # mark elements reachable by `y` and not `x` with `usetag` + 2
    pr3_3_mark!(marker, stack5, target, begptr,
        endptr, invptr, usetag, x, y)

    @inbounds xtag = marker[x]
    @inbounds ztag = marker[z]

    @inbounds xnum = number[x]
    @inbounds ynum = number[y]
    @inbounds znum = number[z]

    @inbounds xdeg = degree[x]
    @inbounds ydeg = degree[y]
    @inbounds zdeg = degree[z]

    @inbounds xwgt = weight[x]
    @inbounds ywgt = weight[y]
    @inbounds zwgt = weight[z]

    @inbounds wwgt = weight[w]
    @inbounds wwwgt = weight[ww]
    @inbounds wwwwgt = weight[www]

    @inbounds qinv = invptr[q]
    @inbounds qqinv = invptr[qq]

    if usetag <= ztag <= usetag + one(V)
        # w ─── x
        # │  ╱
        # z

        # remove `w` from the reachable sets of `x` and `z`
        pr3_reach_del!(source, target, endptr, invptr, qinv)
        pr3_reach_del!(source, target, endptr, invptr, qqinv)

        # decrement the degree of `x` and `z`
        xnum -= one(V)
        znum -= one(V)

        # decrease the weighted degree of `x` and `z` by the
        # weight of `w`
        xdeg -= wwgt
        zdeg -= wwgt
    else
        # w ─── x
        # │
        # z

        # replace `w` with `z` in the reachable set of `x`
        @inbounds target[qinv] = z; invptr[qinv] = qqinv

        # replace `w` with `x` in the reachable set of `z`
        @inbounds target[qqinv] = x; invptr[qqinv] = qinv

        # increase the weighted degree of of `x` by the weight
        # of `z` and decrease it by the weight of `w`
        xdeg -= (wwgt - zwgt)

        # increase the weighted degree of `z` by the weight
        # of `x` and decrease it by the weight of `w`
        zdeg -= (wwgt - xwgt)
    end

    @inbounds rinv = invptr[r]
    @inbounds rrinv = invptr[rr]

    if usetag + one(V) <= xtag
        # ww ─── y
        #  │  ╱
        #  x

        # remove `ww` from the reachable sets of `y` and `x`
        pr3_reach_del!(source, target, endptr, invptr, rinv)
        pr3_reach_del!(source, target, endptr, invptr, rrinv)

        # decrement the degree of `y` and `x`
        ynum -= one(V)
        xnum -= one(V)

        # decrease the weighted degree of `y` and `x` by the
        # weight of `ww`
        ydeg -= wwwgt
        xdeg -= wwwgt
    else
        # ww ─── y
        #  │
        #  x

        # replace `ww` with `x` in the reachable set of `y`
        @inbounds target[rinv] = x; invptr[rinv] = rrinv

        # replace `ww` with `y` in the reachable set of `x`
        @inbounds target[rrinv] = y; invptr[rrinv] = rinv

        # increase the weighted degree of of `y` by the weight
        # of `x` and decrease it by the weight of `ww`
        ydeg -= (wwwgt - xwgt)

        # increase the weighted degree of `x` by the weight
        # of `y` and decrease it by the weight of `ww`
        xdeg -= (wwwgt - ywgt)
    end

    @inbounds sinv = invptr[s]
    @inbounds ssinv = invptr[ss]

    if usetag + one(V) <= ztag
        # www ─── z
        #   │  ╱
        #   y

        # remove `www` from the reachable sets of `z` and `y`
        pr3_reach_del!(source, target, endptr, invptr, sinv)
        pr3_reach_del!(source, target, endptr, invptr, ssinv)

        # decrement the degree of `z` and `y`
        znum -= one(V)
        ynum -= one(V)

        # decrease the weighted degree of `z` and `y` by the
        # weight of `www`
        zdeg -= wwwwgt
        ydeg -= wwwwgt
    else
        # www ─── z
        #   │
        #   y

        # replace `www` with `y` in the reachable set of `z`
        @inbounds target[sinv] = y; invptr[sinv] = ssinv

        # replace `www` with `z` in the reachable set of `y`
        @inbounds target[ssinv] = z; invptr[ssinv] = sinv

        # increase the weighted degree of of `z` by the weight
        # of `y` and decrease it by the weight of `www`
        zdeg -= (wwwwgt - ywgt)

        # increase the weighted degree of `y` by the weight
        # of `z` and decrease it by the weight of `www`
        ydeg -= (wwwwgt - zwgt)
    end

    @inbounds number[x] = xnum
    @inbounds number[y] = ynum
    @inbounds number[z] = znum

    @inbounds degree[x] = xdeg
    @inbounds degree[y] = ydeg
    @inbounds degree[z] = zdeg

    return tag
end

# returns a fresh tag `usetag`: the values `usetag`, `usetag` + 1,
# and `usetag` + 2 do not appear in `marker`. Returns the updated
# running tag as the second value.
function pr3_tag!(marker::AbstractVector{V}, tag::V) where {V}
    if tag > typemax(V) - four(V)
        fill!(marker, zero(V)); tag = one(V)
    end

    return tag, tag + three(V)
end

function pr3_stack_add!(stack::AbstractVector{V}, hi::V, v::V) where {V}
    @inbounds hi += one(V); stack[hi] = v
    return hi
end

function pr3_stack_pop!(stack::AbstractVector{V}, hi::V) where {V}
    @inbounds v = stack[hi]; hi -= one(V)
    return hi, v
end

function pr3_reach_del!(
        source::AbstractVector{V},
        target::AbstractVector{V},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        p::E
    ) where {V, E}
    @inbounds pend = endptr[source[p]] -= one(E)

    if p < pend
        @inbounds x = target[p] = target[pend]

        if ispositive(x)
            @inbounds pinv = invptr[p] = invptr[pend]
            @inbounds invptr[pinv] = p
        end
    end

    return
end

function sorttwo(x, y)
    return sorttwo(identity, x, y)
end

function sorttwo(f::Function, x, y)
    if f(y) < f(x)
        x, y = y, x
    end

    return x, y
end

function sortthree(x, y, z)
    return sortthree(identity, x, y, z)
end

function sortthree(f::Function, x, y, z)
    x, y = sorttwo(f, x, y)
    y, z = sorttwo(f, y, z)
    x, y = sorttwo(f, x, y)
    return (x, y, z)
end

function pr3_reach!(
        combine::Function,
        result,
        vstack::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        invptr::AbstractVector{E},
        v::V,
    ) where {V, E}
    result, _ = pr3_reach_until!(result, vstack, target,
            begptr, endptr, invptr, v) do result, p, w
        return (@inline combine(result, p, w)), false
    end

    return result
end

# Fold `combine` over the elements reachable by `v`. The
# function `combine` returns a pair (`result`, `stop`); the
# traversal terminates early if `stop` is true. As a side
# effect, supernodes are compressed.
function pr3_reach_until!(
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
                result, stop = @inline combine(result, pp, ww)

                if stop
                    endptr[vv] = ppend
                    return result, true
                end

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

    return result, false
end
