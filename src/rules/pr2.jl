function pr2(weights::AbstractVector{W}, graph::AbstractGraph, width::Number) where {W <: Number}
    return pr2(weights, graph, convert(W, width))
end

function pr2(weights::AbstractVector{W}, graph::AbstractGraph{V}, width::W) where {W <: Number, V}
    @assert nv(graph) <= length(weights)

    E = etype(graph); n = nv(graph); m = de(graph); nn = n + one(V)

    weight = weights
    degree = FVector{W}(undef, n)
    number = FVector{V}(undef, n)
    status = FVector{UInt8}(undef, n)
    target = FVector{V}(undef, m)
    begptr = FVector{E}(undef, nn)
    stack1 = FVector{V}(undef, n)
    stack2 = FVector{V}(undef, n)
    stack7 = FVector{V}(undef, n)
    stack4 = FVector{V}(undef, n)
    inject = FVector{V}(undef, n)
    ptr = FVector{E}(undef, nn)
    tgt = FVector{V}(undef, m)

    return pr2_impl!(weight, degree, number, status, target, begptr, stack1,
        stack2, stack7, stack4, inject, ptr, tgt, width, graph)
end

function pr2_impl!(
        weight::AbstractVector{W},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        status::AbstractVector{UInt8},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        stack7::AbstractVector{V},
        stack4::AbstractVector{V},
        inject::AbstractVector{V},
        ptr::AbstractVector{E},
        tgt::AbstractVector{V},
        width::W,
        graph::AbstractGraph{V},
    ) where {W, V, E}
    n = nv(graph)

    @assert n <= length(weight)
    @assert n <= length(degree)
    @assert n <= length(number)
    @assert n <= length(status)
    @assert de(graph) <= length(target)
    @assert n < length(begptr)
    @assert n <= length(stack1)
    @assert n <= length(stack2)
    @assert n <= length(stack7)
    @assert n <= length(stack4)
    @assert n <= length(inject)
    @assert n < length(ptr)
    @assert de(graph) <= length(tgt)

    # copy the adjacency into `target`, compute the degree `number`
    # and the closed weighted degree `degree`, and queue the vertices
    # of degree 0 and 1 and the vertices of degree 2
    hi1 = zero(V); hi2 = zero(V)
    @inbounds begptr[one(V)] = p = one(E)

    @inbounds for v in oneto(n)
        d = weight[v]; status[v] = zero(UInt8)

        for w in neighbors(graph, v)
            target[p] = w; p += one(E); d += weight[w]
        end

        begptr[v + one(V)] = p
        c = convert(V, p - begptr[v])
        number[v] = c; degree[v] = d

        if c <= one(V)
            status[v] = PR3_STACK1; hi1 = pr3_stack_add!(stack1, hi1, v)
        elseif istwo(c)
            status[v] = PR3_STACK2; hi2 = pr3_stack_add!(stack2, hi2, v)
        end
    end

    # apply the islet (degree 0), twig (degree 1), and series
    # (degree 2) rules until no more apply
    parked = width; hi4 = zero(V); hi7 = zero(V)

    @inbounds while true
        if ispositive(hi1)
            # `v` is a vertex with degree 0 or 1
            hi1, v = pr3_stack_pop!(stack1, hi1)
            status[v] &= ~PR3_STACK1

            if iszero(status[v] & PR3_DELETE) && number[v] <= one(V)
                # `v` is simplicial: eliminate it and update the lower bound
                hi4 = pr3_delete!(status, stack4, hi4, v)
                width = max(width, degree[v])

                if isone(number[v])
                    # `w` is the unique surviving neighbor of `v`
                    w = pr2_reach1!(status, target, begptr, v)
                    number[w] -= one(V)
                    degree[w] -= weight[v]
                    hi1, hi2 = pr2_touch!(number, status, stack1, stack2, hi1, hi2, w)
                end
            end
        elseif ispositive(hi2)
            # `v` is a vertex with degree 2 (probably)
            hi2, v = pr3_stack_pop!(stack2, hi2)
            status[v] &= ~PR3_STACK2

            if iszero(status[v] & PR3_DELETE) && istwo(number[v])
                width, parked, hi1, hi2, hi4, hi7 = pr2_series!(weight, degree,
                    number, status, target, begptr, stack1, stack2, stack4,
                    stack7, width, parked, hi1, hi2, hi4, hi7, v)
            end
        elseif ispositive(hi7) && parked < width
            # the lower bound has increased: re-test every vertex
            # that failed a test because of it
            hi1, hi2, hi7 = pr2_unpark!(number, status, stack1, stack2, stack7,
                hi1, hi2, hi7)
        else
            break
        end
    end

    # construct the reduced graph on the surviving vertices; reuse
    # `number` to map an original vertex to its index in the kernel
    k = zero(V)

    @inbounds for v in oneto(n)
        if iszero(status[v] & PR3_DELETE)
            k += one(V); number[v] = k; inject[k] = v
        else
            number[v] = zero(V)
        end
    end

    @inbounds ptr[one(V)] = q = one(E)

    @inbounds for kk in oneto(k)
        v = inject[kk]

        for a in begptr[v]:(begptr[v + one(V)] - one(E))
            w = number[target[a]]

            if ispositive(w)
                tgt[q] = w; q += one(E)
            end
        end

        ptr[kk + one(V)] = q
    end

    # `kernel` is the reduced graph
    kernel = BipartiteGraph(k, k, q - one(E), ptr, tgt)
    return kernel, stack4, inject, width
end

# test a vertex `v` with degree 2
function pr2_series!(
        weight::AbstractVector{W},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        status::AbstractVector{UInt8},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        stack4::AbstractVector{V},
        stack7::AbstractVector{V},
        width::W,
        parked::W,
        hi1::V,
        hi2::V,
        hi4::V,
        hi7::V,
        v::V,
    ) where {W, V, E}
    tol = tolerance(W)

    # `w` and `ww` are the surviving neighbors of `v`
    w, ww = pr2_reach2!(status, target, begptr, v)

    # sort `w` and `ww` by degree
    @inbounds if number[ww] < number[w]
        w, ww = ww, w
    end

    if pr2_adjacent!(status, target, begptr, w, ww)
        # w ─── ww
        # │  ╱
        # v

        # `v` is simplicial: eliminate it and update the lower bound
        hi4 = pr3_delete!(status, stack4, hi4, v)
        @inbounds width = max(width, degree[v])

        # `w` and `ww` lose `v`
        @inbounds number[w] -= one(V); number[ww] -= one(V)
        @inbounds degree[w] -= weight[v]; degree[ww] -= weight[v]

        hi1, hi2 = pr2_touch!(number, status, stack1, stack2, hi1, hi2, w)
        hi1, hi2 = pr2_touch!(number, status, stack1, stack2, hi1, hi2, ww)
    elseif @inbounds min(weight[w], weight[ww]) < weight[v] + tol
        if @inbounds degree[v] < width + tol
            # w     ww
            # │  ╱
            # v

            # eliminate `v` and replace it with the edge {`w`, `ww`}
            hi4 = pr3_delete!(status, stack4, hi4, v)
            pr2_replace!(target, begptr, w, v, ww)
            pr2_replace!(target, begptr, ww, v, w)

            @inbounds degree[w] -= (weight[v] - weight[ww])
            @inbounds degree[ww] -= (weight[v] - weight[w])

            hi1, hi2 = pr2_touch!(number, status, stack1, stack2, hi1, hi2, w)
            hi1, hi2 = pr2_touch!(number, status, stack1, stack2, hi1, hi2, ww)
        else
            # the test failed because of the lower bound
            parked, hi7 = pr3_park!(status, stack7, parked, hi7, width, v)
        end
    end

    return width, parked, hi1, hi2, hi4, hi7
end

# the neighborhood of `v` has changed: add it to the appropriate
# work queue
function pr2_touch!(
        number::AbstractVector{V},
        status::AbstractVector{UInt8},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        hi1::V,
        hi2::V,
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
        end
    end

    return hi1, hi2
end

# re-test every vertex that failed a test because of the lower bound
function pr2_unpark!(
        number::AbstractVector{V},
        status::AbstractVector{UInt8},
        stack1::AbstractVector{V},
        stack2::AbstractVector{V},
        stack7::AbstractVector{V},
        hi1::V,
        hi2::V,
        hi7::V,
    ) where {V}
    @inbounds for i in oneto(hi7)
        v = stack7[i]
        status[v] &= ~PR3_PARKED
        hi1, hi2 = pr2_touch!(number, status, stack1, stack2, hi1, hi2, v)
    end

    return hi1, hi2, zero(V)
end

# the unique surviving neighbor of the degree 1 vertex `v`
function pr2_reach1!(
        status::AbstractVector{UInt8},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        v::V,
    ) where {V, E}
    w = v

    @inbounds for a in begptr[v]:(begptr[v + one(V)] - one(E))
        x = target[a]

        if iszero(status[x] & PR3_DELETE)
            w = x
        end
    end

    return w
end

# the two surviving neighbors of the degree 2 vertex `v`
function pr2_reach2!(
        status::AbstractVector{UInt8},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        v::V,
    ) where {V, E}
    w = v; ww = v

    @inbounds for a in begptr[v]:(begptr[v + one(V)] - one(E))
        x = target[a]

        if iszero(status[x] & PR3_DELETE)
            if w == v
                w = x
            else
                ww = x
            end
        end
    end

    return w, ww
end

# returns true if `ww` is a surviving neighbor of `w`
function pr2_adjacent!(
        status::AbstractVector{UInt8},
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        w::V,
        ww::V,
    ) where {V, E}
    @inbounds for a in begptr[w]:(begptr[w + one(V)] - one(E))
        if target[a] == ww
            return true
        end
    end

    return false
end

# replace the neighbor `v` of `w` with `ww`
function pr2_replace!(
        target::AbstractVector{V},
        begptr::AbstractVector{E},
        w::V,
        v::V,
        ww::V,
    ) where {V, E}
    @inbounds for a in begptr[w]:(begptr[w + one(V)] - one(E))
        if target[a] == v
            target[a] = ww
            return
        end
    end

    return
end
