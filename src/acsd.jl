function acsd_find!(
        head::AbstractVector{V},
        next::AbstractVector{V},
        mark::AbstractVector{V},
        wperm::AbstractVector{V},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        order::AbstractVector{V},
        tree::CliqueTree{V, E},
        uniform::Bool,
    ) where {W, V, E}
    @assert nv(graph) <= length(head)
    @assert nv(graph) <= length(next)
    @assert nv(graph) <= length(mark)
    @assert nv(graph) <= length(weights)
    @assert nv(graph) <= length(order)

    function set(v::V)
        h = view(head, v)
        return SinglyLinkedList(h, next)
    end

    minwgt = typemax(W)
    tol = tolerance(W)
    node = zero(V)
    totcnt = zero(E)

    for v in vertices(graph)
        head[v] = zero(V)
        mark[v] = zero(V)
        minwgt = min(minwgt, weights[v])
    end

    for bag in tree
        node += one(V)
        sep = separator(bag)
        maxcnt = convert(V, length(sep))

        # pre-filter: a vertex of degree less than |S| - 2 is missing at
        # least two edges in S; two such vertices rule out an almost clique
        numlow = zero(V)

        for i in sep
            if eltypedegree(graph, order[i]) + two(V) < maxcnt
                numlow += one(V)
                numlow > one(V) && break
            end
        end

        if numlow <= one(V)
            for i in sep
                mark[order[i]] = node
            end

            poscnt = negcnt = vert = alt1 = alt2 = zero(V)

            for i in sep
                v = order[i]; cnt = maxcnt - one(V)

                for w in neighbors(graph, v)
                    if w != v && mark[w] == node
                        cnt -= one(V)
                    end
                end

                if isone(cnt)
                    negcnt += one(V)

                    if iszero(alt1)
                        alt1 = v
                    else
                        alt2 = v
                    end
                elseif ispositive(cnt)
                    if iszero(vert)
                        poscnt = cnt; vert = v
                    else
                        poscnt = -one(V)
                        break
                    end
                end
            end

            if !isnegative(poscnt) && !(iszero(vert) && iszero(negcnt))
                # the minimum weight of a vertex outside of S
                minout = minwgt

                if !uniform
                    for v in wperm
                        if mark[v] != node
                            minout = weights[v]
                            break
                        end
                    end
                end

                center = zero(V); k = zero(V)

                if ispositive(vert)
                    if poscnt == negcnt && weights[vert] < minout + tol
                        center = vert; k = poscnt
                    end
                elseif negcnt == two(V)
                    if weights[alt1] < minout + tol
                        center = alt1; k = one(V)
                    elseif weights[alt2] < minout + tol
                        center = alt2; k = one(V)
                    end
                end

                if ispositive(center)
                    totcnt += convert(E, k)
                    pushfirst!(set(center), node)
                end
            end
        end
    end

    return de(graph) + twice(totcnt)
end

function acsd_complete!(
        pointer::AbstractVector{E},
        target::AbstractVector{V},
        head::AbstractVector{V},
        next::AbstractVector{V},
        mark::AbstractVector{V},
        graph::AbstractGraph{V},
        order::AbstractVector{V},
        tree::CliqueTree{V, E},
    ) where {V, E}   
    @assert nv(graph) < length(pointer)
    @assert de(graph) <= length(target)
    @assert nv(graph) <= length(head)
    @assert nv(graph) <= length(next)
    @assert nv(graph) <= length(mark)
    @assert nv(graph) <= length(order)

    n = nv(graph)
    
    function set(v::V)
        h = view(head, v)
        return SinglyLinkedList(h, next)
    end

    for v in vertices(graph)
        mark[v] = zero(V)
        pointer[v + one(V)] = zero(V)
    end

    for v in oneto(n - one(V))
        nodes = set(v); ndeg = pointer[v + two(V)] + eltypedegree(graph, v)
        
        if !isempty(nodes)         
            for w in neighbors(graph, v)
                mark[w] = v
            end

            for node in nodes, i in separator(tree, node)
                w = order[i]

                if v != w && mark[w] < v
                    mark[w] = v; ndeg += one(V)

                    if w < n
                        pointer[w + two(V)] += one(V)
                    end
                end
            end
        end

        pointer[v + two(V)] = ndeg
    end

    nodes = set(n)

    if !isempty(nodes)
        for w in neighbors(graph, n)
            mark[w] = n
        end

        for node in nodes, i in separator(tree, node)
            w = order[i]

            if n != w && mark[w] < n
                mark[w] = n

                if w < n
                    pointer[w + two(V)] += one(V)
                end
            end
        end
    end

    pointer[one(V)] = p = one(E) 
    
   for v in vertices(graph)
        mark[v] = zero(V)
        pointer[v + one(V)] = p += pointer[v + one(V)]
    end
    
    for v in vertices(graph)
        nodes = set(v); p = pointer[v + one(V)]
        
        if !isempty(nodes)            
            for w in neighbors(graph, v)
                target[p] = w; p += one(E)
                mark[w] = v
            end

            for node in nodes, i in separator(tree, node)
                w = order[i]

                if v != w && mark[w] < v
                    mark[w] = v
                    q = pointer[w + one(V)]
                    target[p] = w; p += one(E)
                    target[q] = v; q += one(E)
                    pointer[w + one(V)] = q
                end
            end
        else
            for w in neighbors(graph, v)
                target[p] = w; p += one(E)
            end
        end

        pointer[v + one(V)] = p
    end

    p = qstop = one(E)
    
    for v in vertices(graph)
        qstrt = qstop
        qstop = pointer[v + one(V)]

        for q in qstrt:qstop - one(E)
            w = target[q]

            if mark[w] < v + n
                mark[w] = v + n
                target[p] = w; p += one(V)
            end
        end

        pointer[v + one(V)] = p  
    end

    m = p - one(E)
    return BipartiteGraph(n, n, m, pointer, target)
end

function acsd(weights::AbstractVector{W}, graph::AbstractGraph{V}, alg::MinimalAlgorithm) where {W, V}
    E = etype(graph); n = nv(graph)

    head = FVector{V}(undef, n)
    next = FVector{V}(undef, n)
    mark = FVector{V}(undef, n)
    pointer = FVector{E}(undef, n + one(V))

    order, tree = cliquetree(weights, graph, alg)

    minwgt, maxwgt = extrema(weights)
    uniform = maxwgt < minwgt + tolerance(W)

    if uniform
        wperm = Vector{V}(undef, 0)
    else
        wperm = Vector{V}(sortperm(weights))
    end

    m = acsd_find!(head, next, mark, wperm,
        weights, graph, order, tree, uniform)

    target = FVector{V}(undef, m)
    return acsd_complete!(pointer, target, head, next, mark, graph, order, tree)
end

function acsd(weights::AbstractVector, graph::AbstractGraph, alg::MinimalAlgorithm, rest::MinimalAlgorithm...) 
    return acsd(weights, acsd(weights, graph, alg), rest...)
end

function safeseparators(weights::AbstractVector, graph::AbstractGraph{V}, alg::EliminationAlgorithm, mins::Tuple) where {V}
    n = nv(graph)

    if n > one(V)
        cmpgraph = acsd(weights, graph, mins...)
        order, tree = atomtree(cmpgraph, MinimalChordal())
        index = invperm(order)

        if length(tree) > 1
            pmtweights = weights[order]
            pmtgraph = permute(cmpgraph, order, index)
            pmtindex = safeseparators(pmtweights, pmtgraph, tree, alg, mins)

            for v in vertices(graph)
                i = index[v] = pmtindex[index[v]]
                order[i] = v
            end

            return order, index
        end
    end

    return permutation(weights, graph, alg)
end

function safeseparators(weights::AbstractVector{W}, graph::AbstractGraph{V}, tree::CliqueTree{V, E}, alg::EliminationAlgorithm, mins::Tuple) where {W, V, E}
    n = nv(graph); m = de(graph)
    
    index = FVector{V}(undef, n)
    work1 = FVector{V}(undef, n)
    work2 = FVector{V}(undef, n)
    work3 = FVector{V}(undef, n)
    
    subwgt = FVector{W}(undef, n)
    submsk = FVector{V}(undef, n)
    subinv = FVector{V}(undef, n)
    
    subptr = FVector{E}(undef, n + 1)
    uppptr = FVector{E}(undef, n + 1)
    
    subtgt = FVector{V}(undef, m)
    upptgt = FVector{V}(undef, m)

    for i in vertices(graph)
        submsk[i] = zero(V)
    end

    for bag in tree
        res = residual(bag)
        sep = separator(bag)

        strt = first(res)
        stop = last(res)
        ii = stop - strt + one(V)
        
        for i in sep
            submsk[i] = strt
            subinv[i] = ii += one(V)
        end

        ii = zero(V); subptr[ii + one(V)] = pp = one(E)

        for i in bag
            ii += one(V); subwgt[ii] = weights[i]
            
            for j in neighbors(graph, i)
                jj = zero(V)
                
                if strt <= j
                    if j <= stop
                        jj = j - strt + one(V)
                    elseif submsk[j] == strt
                        jj = subinv[j]
                    end
                end

                if ispositive(jj)
                    subtgt[pp] = jj; pp += one(E)
                end
            end

            subptr[ii + one(V)] = pp
        end

        nn = ii; mm = pp - one(E)
        subgraph = BipartiteGraph(nn, nn, mm, subptr, subtgt)
        suborder, subindex = permutation(subwgt, subgraph, Compression(SafeSeparators(alg, mins)))
        subupper = sympermute!_impl!(uppptr, upptgt, subgraph, subindex, Forward)
        sublower = BipartiteGraph(nn, nn, de(subupper), subptr, subtgt)
        subclique = view(subindex, stop - strt + two(V):nn)
        subtree = Parent(nn, suborder)
        
        compositerotations_impl!(work1, work2, work3,
            subinv, sublower, subtree, subupper, subclique)

        for i in strt:stop
            ii = i - strt + one(V)
            index[i] = work1[subindex[ii]] + strt - one(V)
        end
    end

    return index
end
