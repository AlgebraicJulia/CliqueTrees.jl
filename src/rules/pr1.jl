function pr1(weights::AbstractVector{W}, graph::AbstractGraph, width::Number) where {W <: Number}
    return pr1(weights, graph, convert(W, width))
end

function pr1(weights::AbstractVector{W}, graph::AbstractGraph{V}, width::W) where {W <: Number, V}
    @assert nv(graph) <= length(weights)

    E = etype(graph); n = nv(graph); m = de(graph); nn = n + one(V)

    weight = weights
    degree = FVector{W}(undef, n)
    number = FVector{V}(undef, n)
    status = FVector{UInt8}(undef, n)
    stack1 = FVector{V}(undef, n)
    stack4 = FVector{V}(undef, n)
    inject = FVector{V}(undef, n)
    ptr = FVector{E}(undef, nn)
    tgt = FVector{V}(undef, m)

    return pr1_impl!(weight, degree, number, status, stack1, stack4, inject,
        ptr, tgt, width, graph)
end

function pr1_impl!(
        weight::AbstractVector{W},
        degree::AbstractVector{W},
        number::AbstractVector{V},
        status::AbstractVector{UInt8},
        stack1::AbstractVector{V},
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
    @assert n <= length(stack1)
    @assert n <= length(stack4)
    @assert n <= length(inject)
    @assert n < length(ptr)
    @assert de(graph) <= length(tgt)

    # `number` is the degree and `degree` is the closed weighted
    # degree; queue the vertices of degree 0 and 1
    hi1 = zero(V)

    @inbounds for v in oneto(n)
        d = weight[v]; c = zero(V)

        for w in neighbors(graph, v)
            d += weight[w]; c += one(V)
        end

        degree[v] = d; number[v] = c; status[v] = zero(UInt8)

        if c <= one(V)
            hi1 += one(V); stack1[hi1] = v; status[v] = PR3_STACK1
        end
    end

    # apply the islet (degree 0) and twig (degree 1) rules until
    # no more apply
    hi4 = zero(V)

    @inbounds while ispositive(hi1)
        hi1, v = pr3_stack_pop!(stack1, hi1)
        status[v] &= ~PR3_STACK1

        if iszero(status[v] & PR3_DELETE) && number[v] <= one(V)
            # `v` is simplicial: eliminate it and update the lower bound
            hi4 += one(V); stack4[hi4] = v; status[v] |= PR3_DELETE
            width = max(width, degree[v])

            if isone(number[v])
                # `w` is the unique surviving element reachable by `v`
                w = v

                for x in neighbors(graph, v)
                    if iszero(status[x] & PR3_DELETE)
                        w = x
                    end
                end

                # `w` loses `v`
                number[w] -= one(V)
                degree[w] -= weight[v]

                # queue `w` if its degree dropped to 0 or 1
                if number[w] <= one(V) && iszero(status[w] & PR3_STACK1)
                    hi1 += one(V); stack1[hi1] = w; status[w] |= PR3_STACK1
                end
            end
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

    @inbounds ptr[one(V)] = p = one(E)

    @inbounds for kk in oneto(k)
        v = inject[kk]

        for w in neighbors(graph, v)
            ww = number[w]

            if ispositive(ww)
                tgt[p] = ww; p += one(E)
            end
        end

        ptr[kk + one(V)] = p
    end

    # `kernel` is the reduced graph
    kernel = BipartiteGraph(k, k, p - one(E), ptr, tgt)
    return kernel, stack4, inject, width
end
