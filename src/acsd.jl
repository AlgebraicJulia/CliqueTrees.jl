function acsd_find!(
        head::AbstractVector{V},
        next::AbstractVector{V},
        mark::AbstractVector{V},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        order::AbstractVector{V},
        tree::CliqueTree{V, E},
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
        node += one(V); maxcnt = poscnt = negcnt = vert = zero(V)

        for i in separator(bag)
            v = order[i]
            mark[v] = node; maxcnt += one(V)
        end

        for i in separator(bag)
            v = order[i]; cnt = maxcnt - one(V)
            
            if weights[v] < minwgt + tol
                flag = true
            else
                flag = false
            end
            
            for w in neighbors(graph, v)
                if w != v && mark[w] == node
                    cnt -= one(V)
                end
            end

            if isone(cnt)
                negcnt += cnt
            elseif ispositive(cnt)
                if flag && !ispositive(vert) && cnt >= negcnt
                    poscnt = cnt; vert = v
                else
                    poscnt = -one(V)
                    break
                end
            end
        end

        if poscnt == negcnt && ispositive(vert)
            totcnt += convert(E, poscnt)
            pushfirst!(set(vert), node)
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

function acsd(weights::AbstractVector, graph::AbstractGraph{V}, alg::MinimalAlgorithm) where {V}
    E = etype(graph); n = nv(graph)
    
    head = FVector{V}(undef, n)
    next = FVector{V}(undef, n)
    mark = FVector{V}(undef, n)
    pointer = FVector{E}(undef, n + one(V))

    order, tree = cliquetree(weights, graph, alg)
   
    m = acsd_find!(head, next, mark,
        weights, graph, order, tree)

    target = FVector{V}(undef, m)
    return acsd_complete!(pointer, target, head, next, mark, graph, order, tree)
end

function safeseparators(weights::AbstractVector, graph::AbstractGraph{V}, alg::EliminationAlgorithm, min::MinimalAlgorithm) where {V}
    n = nv(graph)

    if n < two(V)
        order = collect(oneto(n))
        index = collect(oneto(n))
    else
        cmpgraph = acsd(weights, graph, min)
        order, tree = atomtree(cmpgraph, min); index = invperm(order)

        if length(tree) < 2
            order, index = permutation(weights, cmpgraph, alg)
        else
            pmtweights = weights[order]
            pmtgraph = permute(cmpgraph, order, index)
            pmtindex = safeseparators(pmtweights, pmtgraph, tree, alg, min)

            for v in vertices(graph)
                i = index[v] = pmtindex[index[v]]
                order[i] = v
            end
        end
    end

    return order, index
end

function safeseparators(weights::AbstractVector{W}, graph::AbstractGraph{V}, tree::CliqueTree{V, E}, alg::EliminationAlgorithm, min::MinimalAlgorithm) where {W, V, E}
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
        suborder, subindex = permutation(subwgt, subgraph, Compression(SafeSeparators(alg, min)))
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
