"""
    CliqueTree{V, E} <: AbstractVector{Clique{V, E}}

A rooted forest T = (V, E) and functions

       clique: V → 2ᵁ
    separator: V → 2ᵁ

This type implements the [indexed tree interface](https://juliacollections.github.io/AbstractTrees.jl/stable/#The-Indexed-Tree-Interface).
"""
struct CliqueTree{V, E} <: AbstractVector{Clique{V, E}}
    """
    The rooted forest T = (V, E).
    """
    tree::SupernodeTree{V}

    """
    A directed bipartite graph G = (U, V, E') with an arc

        (v, u) ∈ E'

    for all vertices v ∈ V and u ∈ separator(v).
    """
    graph::BipartiteGraph{V, E, FVector{E}, FVector{V}}

    function CliqueTree{V, E}(tree::SupernodeTree{V}, graph::BipartiteGraph{V, E}) where {V, E}
        @assert nv(residuals(tree)) == nv(graph)
        @assert nov(residuals(tree)) == nov(graph)
        return new{V, E}(tree, graph)
    end
end

function CliqueTree(tree::SupernodeTree{V}, graph::BipartiteGraph{V, E}) where {V, E}
    return CliqueTree{V, E}(tree, graph)
end

function Tree(tree::CliqueTree)
    return Tree(tree.tree)
end

"""
    cliquetree([weights, ]graph;
        alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM,
        snd::SupernodeType=DEFAULT_SUPERNODE_TYPE)

Construct a tree decomposition of a simple graph.
The vertices of the graph are first ordered by a fill-reducing permutation computed by the algorithm `alg`.
The size of the resulting decomposition is determined by the supernode partition `snd`.

```julia-repl
julia> using CliqueTrees

julia> graph = [
           0 1 0 0 0 0 0 0
           1 0 1 0 0 1 0 0
           0 1 0 1 0 1 1 1
           0 0 1 0 0 0 0 0
           0 0 0 0 0 1 1 0
           0 1 1 0 1 0 0 0
           0 0 1 0 1 0 0 1
           0 0 1 0 0 0 1 0
       ];

julia> label, tree = cliquetree(graph);

julia> tree
6-element CliqueTree{Int64, Int64}:
 [6, 7, 8]
 └─ [5, 7, 8]
    ├─ [1, 5]
    ├─ [3, 5, 7]
    │  └─ [2, 3]
    └─ [4, 5, 8]
```
"""
function cliquetree(
        graph;
        alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM,
        snd::SupernodeType = DEFAULT_SUPERNODE_TYPE,
    )
    return cliquetree(graph, alg, snd)
end

function cliquetree(
        weights::AbstractVector,
        graph;
        alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM,
        snd::SupernodeType = DEFAULT_SUPERNODE_TYPE,
    )
    return cliquetree(weights, graph, alg, snd)
end

function cliquetree(graph, alg::PermutationOrAlgorithm, snd::SupernodeType)
    return cliquetree(BipartiteGraph(graph), alg, snd)
end

function cliquetree(graph::AbstractGraph{V}, alg::PermutationOrAlgorithm, snd::SupernodeType = Maximal()) where {V}
    n = nv(graph); weights = Ones{V}(n)
    return cliquetree(weights, graph, alg, snd)
end

function cliquetree(weights::AbstractVector, graph, alg::PermutationOrAlgorithm, snd::SupernodeType = Maximal())
    return cliquetree(weights, BipartiteGraph(graph), alg, snd)
end

function cliquetree(weights::AbstractVector, graph::AbstractGraph{V}, alg::PermutationOrAlgorithm, snd::SupernodeType = Maximal()) where {V}
    E = etype(graph); n = nv(graph); m = half(de(graph))
    nn = n + one(V); nnn = nn + one(V)
    
    target1 = FVector{V}(undef, max(m, nn))
    target2 = FVector{V}(undef, m)
    target3 = FVector{V}(undef, n)

    pointer1 = FVector{E}(undef, nn)
    pointer2 = FVector{E}(undef, nn)
    pointer3 = FVector{V}(undef, nnn)

    colcount = FVector{V}(undef, n)
    elmorder = FVector{V}(undef, n)
    elmindex = FVector{V}(undef, n)
    sndptr = FVector{V}(undef, nn)
    sepptr = FVector{E}(undef, nn)
    new = FVector{V}(undef, n)
    parent = FVector{V}(undef, n)
    elmtree = Parent{V}(n)

    order, index = permutation(weights, graph, alg)

    sndtree, upper, lower = supernodetree_impl!(target1, target2,
        target3, pointer1, pointer2, pointer3, colcount,
        elmorder, elmindex, sndptr, sepptr, new, parent, elmtree, graph,
        order, index, snd)

    h = last(sndtree.tree)
    k = sepptr[h + one(V)] - one(E)

    septgt = FVector{V}(undef, k)
    separator = BipartiteGraph(n, h, k, sepptr, septgt)

    clqtree = cliquetree_impl!(elmindex,
        upper, lower, separator, sndtree)

    return order, clqtree
end

function cliquetree_impl!(
        index::AbstractVector{V},
        target::BipartiteGraph{V, E},
        source::BipartiteGraph{V, E},
        separator::BipartiteGraph{V, E},
        sndtree::SupernodeTree{V},
    ) where {V, E}
    n = last(sndtree.tree)
    lower = sympermute!_impl!(target, source, index, Reverse)
    residual = residuals(sndtree); cache = index

    for j in oneto(n)
        pstrt = pointers(separator)[j]
        pstop = pointers(separator)[j + one(V)] 

        qstrt = pointers(residual)[j]
        qstop = pointers(residual)[j + one(V)]

        rstrt = pointers(lower)[qstrt]
        rstop = pointers(lower)[qstrt + one(V)]

        p = pstrt; r = rstrt
    
        while r < rstop && targets(lower)[r] < qstop
            r += one(E)
        end
    
        while r < rstop
            targets(separator)[p] = targets(lower)[r]; p += one(E)
            r += one(E)
        end 

        p1 = p 
 
        for i in childindices(sndtree, j)
            pstop1 = p1
            pstrt1 = pstrt

            pstop2 = pointers(separator)[i + one(V)]
            pstrt2 = pointers(separator)[i]
        
            p1 = pstrt1; p2 = pstrt2; t = one(V)

            while p2 < pstop2 && targets(separator)[p2] < qstop
                p2 += one(E)
            end

            while p1 < pstop1 && p2 < pstop2
                v1 = targets(separator)[p1]
                v2 = targets(separator)[p2]

                if v1 == v2
                    cache[t] = v1
                    p1 += one(E)
                    p2 += one(E)
                elseif v1 < v2
                    cache[t] = v1
                    p1 += one(E)
                else
                    cache[t] = v2
                    p2 += one(E)
                end

                t += one(V)
            end

            while p1 < pstop1
                cache[t] = targets(separator)[p1]
                p1 += one(E)
                t += one(V)
            end

            while p2 < pstop2
                cache[t] = targets(separator)[p2]
                p2 += one(E)
                t += one(V)
            end

            p1 = pstrt; tstop = t

            for t in oneto(tstop - one(V))
                targets(separator)[p1] = cache[t]
                p1 += one(E)
            end
        end
    end        

    return CliqueTree(sndtree, separator)
end

function cliquetree(tree::CliqueTree{V, E}, root::Integer) where {V, E}
    graph = separators(tree); h = nov(graph); n = nv(graph); m = ne(graph)
    perm = FVector{V}(undef, h)
    invp = FVector{V}(undef, h)

    separator = BipartiteGraph{V, E, FVector{E}, FVector{V}}(h, n, m)
    p = m + one(E); pointers(separator)[n + one(V)] = p

    residual = BipartiteGraph{V, E, FVector{E}, OneTo{V}}(h, n, h)
    q = h + one(V); pointers(residual)[n + one(V)] = q

    sndtree = copy(tree.tree.tree.tree)
    sndinvp = postorder!(setrootindex!(sndtree, root))
    sndperm = invperm(sndinvp)

    for v in oneto(h)
        invp[v] = zero(V)
    end

    for i in reverse(sndtree)
        ii = sndperm[i]; clique = tree[ii]

        for vv in Iterators.reverse(clique)
            v = invp[vv]

            if ispositive(v)
                p -= one(E); targets(separator)[p] = v
            else
                q -= one(V); perm[q] = vv; invp[vv] = q
            end
        end

        pointers(separator)[i] = p
        pointers(residual)[i] = q
    end

    converse = BipartiteGraph{V, E, FVector{E}, FVector{V}}(n, h, m)
    reverse!(converse, separator)    
    reverse!(separator, converse)

    tree = CliqueTree(SupernodeTree(Tree(sndtree), residual), separator)
    return collect(perm), tree
end

function cliquetree!(tree::CliqueTree, root::Integer)
    perm, source = cliquetree(tree, root)
    copy!(tree, source)
    return perm
end

function atomtree(graph::AbstractGraph; alg::MinimalAlgorithm=DEFAULT_MINIMAL_ALGORITHM)
    return atomtree(graph, alg)
end

function atomtree(graph::AbstractGraph, alg::MinimalAlgorithm)
    return atomtree!(cliquetree(graph, alg)..., graph)
end

function atomtree!(order::AbstractVector{V}, tree::CliqueTree{V, E}, graph::AbstractGraph{V}) where {V, E}
    res = residuals(tree)
    sep = separators(tree)
    
    h = nov(sep); m = ne(sep); n = nv(sep)

    work1 = FVector{V}(undef, h)
    work2 = FVector{V}(undef, n)
    work3 = FVector{V}(undef, n)

    mark = FVector{V}(undef, h)
    prnt = FVector{V}(undef, n)
    proj = FVector{V}(undef, n)

    resptr = FVector{V}(undef, n + one(V))
    sepptr = FVector{E}(undef, n + one(V))
    septgt = FVector{V}(undef, m)

    for v in vertices(graph)
        mark[v] = n + one(V)
    end

    qode = zero(V)
    
    for node in reverse(oneto(n))
        flag = true; maxcnt = zero(V)
        
        for i in neighbors(sep, node)
            v = order[i]
            mark[v] = node; maxcnt += one(V)
        end

        for i in neighbors(sep, node)
            v = order[i]; cnt = maxcnt - one(V)

            for w in neighbors(graph, v)
                if w != v && mark[w] == node
                    cnt -= one(V)
                end
            end

            if ispositive(cnt)
                flag = false
                break
            end
        end

        npnt = parentindex(tree, node)

        if isnothing(npnt)
            spnt = zero(V)
        else
            spnt = abs(proj[npnt])
        end

        if flag
            proj[node] = qode += one(V)
            prnt[qode] = spnt
        else
            proj[node] = -spnt
        end
    end

    q = qode; qree = Parent(q, prnt)
    postorder!_impl!(work1, work2, mark, work3, qree)

    for node in oneto(n)
        qode = proj[node]

        if ispositive(qode)
            qode = mark[qode]
        else
            qode = -mark[-qode]
        end
            
        proj[node] = qode
    end

    for qode in oneto(q + one(V))
        resptr[qode] = zero(V)
    end

    sepptr[q + one(V)] = zero(E)

    for node in oneto(n)
        qode = proj[node]
        
        if ispositive(qode)
            sepptr[qode] = eltypedegree(sep, node)
        else
            qode = -qode
        end
        
        resptr[qode] += eltypedegree(res, node)
    end
    
    r = one(V)
    s = one(E)

    for qode in oneto(q + one(V))
        resptr[qode] = r += resptr[qode]
        sepptr[qode] = s += sepptr[qode]
    end

    for node in reverse(oneto(n))
        qode = proj[node]

        if ispositive(qode)
            s = sepptr[qode]

            for i in Iterators.reverse(neighbors(sep, node))
                s -= one(E); septgt[s] = mark[i]
            end

            sepptr[qode] = s
        else
            qode = -qode
        end

        r = resptr[qode]

        for i in reverse(neighbors(res, node))
            mark[i] = r -= one(E)
        end

        resptr[qode] = r
    end

    for i in oneto(h)
        work1[mark[i]] = order[i]
    end

    for i in oneto(h)
        order[i] = work1[i]
    end

    m = sepptr[q + one(V)] - one(E)
    qres = BipartiteGraph(h, q, h, resptr, oneto(h))
    qsep = BipartiteGraph(h, q, m, sepptr, septgt)
    atomtree = CliqueTree(SupernodeTree(Tree(qree), qres), qsep)
    return order, atomtree
end

"""
    treewidth([weights, ]tree::CliqueTree)

Compute the [width](https://en.wikipedia.org/wiki/Treewidth) of a clique tree.

```julia-repl
julia> using CliqueTrees

julia> graph = [
           0 1 0 0 0 0 0 0
           1 0 1 0 0 1 0 0
           0 1 0 1 0 1 1 1
           0 0 1 0 0 0 0 0
           0 0 0 0 0 1 1 0
           0 1 1 0 1 0 0 0
           0 0 1 0 1 0 0 1
           0 0 1 0 0 0 1 0
       ];

julia> label, tree = cliquetree(graph);

julia> treewidth(tree)
2
```

"""
function treewidth(tree::CliqueTree{V}) where {V}
    n = nov(residuals(tree)); weights = Ones{V}(n)
    width = treewidth(weights, tree)
    return width - one(V)
end

function treewidth(weights::AbstractVector{W}, tree::CliqueTree{V}) where {W, V}
    maxwidth = zero(W)

    for clique in tree
        width = zero(W)

        for v in clique
            width += weights[v]
        end

        maxwidth = max(maxwidth, width)
    end

    return maxwidth
end

function treewidth(weights::Ones{W}, tree::CliqueTree) where {W}
    residual = residuals(tree)
    separator = separators(tree)
    width = zero(W)

    @inbounds for i in vertices(residual)
        nn = convert(W, outdegree(residual, i))
        na = convert(W, outdegree(separator, i))
        width = max(width, nn + na)
    end

    return width
end

"""
    treewidth([weights, ]graph;
        alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM)

Compute the [width](https://en.wikipedia.org/wiki/Treewidth) induced by an elimination
algorithm.

```julia-repl
julia> using CliqueTrees, TreeWidthSolver

julia> graph = [
           0 1 0 0 0 0 0 0
           1 0 1 0 0 1 0 0
           0 1 0 1 0 1 1 1
           0 0 1 0 0 0 0 0
           0 0 0 0 0 1 1 0
           0 1 1 0 1 0 0 0
           0 0 1 0 1 0 0 1
           0 0 1 0 0 0 1 0
       ];

julia> treewidth(graph; alg=MCS())
3

julia> treewidth(graph; alg=BT()) # exact treewidth
2
```
"""
function treewidth(graph; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM)
    return treewidth(graph, alg)
end

function treewidth(graph, alg::PermutationOrAlgorithm)
    return treewidth(BipartiteGraph(graph), alg)
end

function treewidth(graph::AbstractGraph{V}, alg::PermutationOrAlgorithm) where {V}
    n = nv(graph); weights = Ones{V}(n)
    width = treewidth(weights, graph, alg)
    return width - one(V)
end

function treewidth(weights::AbstractVector, graph; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM)
    return treewidth(weights, graph, alg)
end

function treewidth(weights::AbstractVector, graph, alg::PermutationOrAlgorithm)
    return treewidth(weights, BipartiteGraph(graph), alg)
end

function treewidth(weights::AbstractVector{W}, graph::AbstractGraph{V}, alg::PermutationOrAlgorithm) where {W, V}
    E = etype(graph); m = half(de(graph)); n = nv(graph)
    order, index = permutation(weights, graph, alg)

    lower = BipartiteGraph{V, E}(n, n, m)
    upper = BipartiteGraph{V, E}(n, n, m)
    tree = Parent{V}(n)
    sets = UnionFind{V}(n)
    
    wwork0 = FVector{W}(undef, n)
    wwork1 = FVector{W}(undef, n)
    vwork0 = FVector{V}(undef, n)
    vwork1 = FVector{V}(undef, n)
    vwork2 = FVector{V}(undef, n)
    vwork3 = FVector{V}(undef, n)
    vwork4 = FVector{V}(undef, n)
    vwork5 = FVector{V}(undef, n)
    
    width = treewidth_impl!(lower, upper, tree, sets, wwork0, wwork1, vwork0,
        vwork1, vwork2, vwork3, vwork4, vwork5, weights, graph, index)

    return width
end

function treewidth_impl!(
        lower::BipartiteGraph{V, E},
        upper::BipartiteGraph{V, E},
        tree::Parent{V},
        sets::UnionFind{V},
        wwork0::AbstractVector{W},
        wwork1::AbstractVector{W},
        vwork0::AbstractVector{V},
        vwork1::AbstractVector{V},
        vwork2::AbstractVector{V},
        vwork3::AbstractVector{V},
        vwork4::AbstractVector{V},
        vwork5::AbstractVector{V},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        index::AbstractVector{V},
    ) where {W, V, E}
    @assert nv(graph) <= length(weights)
    @assert nv(graph) <= length(wwork1)
    
    @inbounds for v in vertices(graph)
        wwork1[index[v]] = weights[v]
    end
    
    sympermute!_impl!(upper, graph, index, Forward)
    reverse!_impl!(lower, upper)
    etree_impl!(tree, vwork0, upper)
    
    supcnt_impl!(wwork0, vwork0, vwork1, vwork2, vwork3,
        vwork4, vwork5, sets, wwork1, lower, tree)

    width = zero(W)

    @inbounds for v in vertices(lower)
        width = max(width, wwork0[v])
    end

    return width
end

function bestwidth(weights::AbstractVector{W}, graph::AbstractGraph{V}, algs::NTuple{<:Any, PermutationOrAlgorithm}) where {W, V}
    E = etype(graph); m = de(graph); n = nv(graph)
    
    pairs = map(algs) do alg
        return permutation(weights, graph, alg)
    end

    indices = map(pairs) do (order, index)
        return index
    end

    lower = BipartiteGraph{V, E}(n, n, m)
    upper = BipartiteGraph{V, E}(n, n, m)
    tree = Parent{V}(n)
    sets = UnionFind{V}(n)

    wwork0 = FVector{W}(undef, n)
    wwork1 = FVector{W}(undef, n)
    vwork0 = FVector{V}(undef, n)
    vwork1 = FVector{V}(undef, n)
    vwork2 = FVector{V}(undef, n)
    vwork3 = FVector{V}(undef, n)
    vwork4 = FVector{V}(undef, n)
    vwork5 = FVector{V}(undef, n)

    index = bestwidth_impl!(lower, upper, tree, sets, wwork0, wwork1, vwork0,
        vwork1, vwork2, vwork3, vwork4, vwork5, weights, graph, indices)

    return pairs[index]
end

function bestwidth_impl!(
        lower::BipartiteGraph{V, E},
        upper::BipartiteGraph{V, E},
        tree::Parent{V},
        sets::UnionFind{V},
        wwork0::AbstractVector{W},
        wwork1::AbstractVector{W},
        vwork0::AbstractVector{V},
        vwork1::AbstractVector{V},
        vwork2::AbstractVector{V},
        vwork3::AbstractVector{V},
        vwork4::AbstractVector{V},
        vwork5::AbstractVector{V},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        indices::NTuple{N, AbstractVector{V}},
    ) where {W, V, E, N}

    minindex = zero(N); minwidth = typemax(W)

    for index in oneto(N)
        width = treewidth_impl!(lower, upper, tree, sets, wwork0, wwork1, vwork0,
            vwork1, vwork2, vwork3, vwork4, vwork5, weights, graph, indices[index])

        if width < minwidth
            minindex, minwidth = index, width
        end
    end

    return minindex
end

function treefill(tree::CliqueTree{<:Any, E}) where {E}
    n = nov(residuals(tree)); weights = Ones{E}(n)
    fill = treefill(weights, tree)
    return fill - convert(E, n)
end

function treefill(weights::AbstractVector{W}, tree::CliqueTree) where {W}
    fill = zero(W)

    @inbounds for bag in tree
        res = residual(bag)

        for vv in bag
            ww = weights[vv]

            for v in res
                v > vv && break
                w = weights[v]
                fill += ww * w
            end
        end
    end

    return fill
end

function treefill(weights::Ones{W}, tree::CliqueTree) where {W}
    residual = residuals(tree)
    separator = separators(tree)
    fill = zero(W)

    @inbounds for i in vertices(residual)
        nn = convert(W, outdegree(residual, i))
        na = convert(W, outdegree(separator, i))
        fill += half(nn * (nn + one(I))) + nn * na
    end

    return fill
end

function treefill(graph; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM)
    return treefill(graph, alg)
end

function treefill(graph, alg::PermutationOrAlgorithm)
    return treefill(BipartiteGraph(graph), alg)
end

function treefill(graph::AbstractGraph, alg::PermutationOrAlgorithm)
    E = etype(graph); n = nv(graph); weights = Ones{E}(n)
    fill = treefill(weights, graph, alg)
    return fill - convert(E, n)
end

function treefill(weights::AbstractVector, graph; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM)
    return treefill(weights, graph, alg)
end

function treefill(weights::AbstractVector, graph, alg::PermutationOrAlgorithm)
    return treefill(weights, BipartiteGraph(graph), alg)
end

function treefill(weights::AbstractVector{W}, graph::AbstractGraph{V}, alg::PermutationOrAlgorithm) where {W, V}
    E = etype(graph); m = de(graph); n = nv(graph)
    order, index = permutation(weights, graph, alg)
    
    lower = BipartiteGraph{V, E}(n, n, m)
    upper = BipartiteGraph{V, E}(n, n, m)
    tree = Parent{V}(n)
    sets = UnionFind{V}(n)
    
    wwork0 = FVector{W}(undef, n)
    wwork1 = FVector{W}(undef, n)
    vwork0 = FVector{V}(undef, n)
    vwork1 = FVector{V}(undef, n)
    vwork2 = FVector{V}(undef, n)
    vwork3 = FVector{V}(undef, n)
    vwork4 = FVector{V}(undef, n)
    vwork5 = FVector{V}(undef, n)
    
    fill = treefill_impl!(lower, upper, tree, sets, wwork0, wwork1, vwork0,
        vwork1, vwork2, vwork3, vwork4, vwork5, weights, graph, index)

    return fill
end

function treefill_impl!(
        lower::BipartiteGraph{V, E},
        upper::BipartiteGraph{V, E},
        tree::Parent{V},
        sets::UnionFind{V},
        wwork0::AbstractVector{W},
        wwork1::AbstractVector{W},
        vwork0::AbstractVector{V},
        vwork1::AbstractVector{V},
        vwork2::AbstractVector{V},
        vwork3::AbstractVector{V},
        vwork4::AbstractVector{V},
        vwork5::AbstractVector{V},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        index::AbstractVector{V},
    ) where {W, V, E}
    @assert nv(graph) <= length(weights)
    @assert nv(graph) <= length(wwork1)
    
    @inbounds for v in vertices(graph)
        wwork1[index[v]] = weights[v]
    end
    
    sympermute!_impl!(upper, graph, index, Forward)
    reverse!_impl!(lower, upper)
    etree_impl!(tree, vwork0, upper)
    
    supcnt_impl!(wwork0, vwork0, vwork1, vwork2, vwork3,
        vwork4, vwork5, sets, wwork1, lower, tree)
    
    fill = zero(W)

    @inbounds for v in vertices(lower)
        fill += wwork1[v] * wwork0[v]
    end

    return fill
end

function bestfill(weights::AbstractVector{W}, graph::AbstractGraph{V}, algs::NTuple{<:Any, PermutationOrAlgorithm}) where {W, V}
    E = etype(graph); m = de(graph); n = nv(graph)
    
    pairs = map(algs) do alg
        return permutation(weights, graph, alg)
    end

    indices = map(pairs) do (order, index)
        return index
    end

    lower = BipartiteGraph{V, E}(n, n, m)
    upper = BipartiteGraph{V, E}(n, n, m)
    tree = Parent{V}(n)
    sets = UnionFind{V}(n)

    wwork0 = FVector{W}(undef, n)
    wwork1 = FVector{W}(undef, n)
    vwork0 = FVector{V}(undef, n)
    vwork1 = FVector{V}(undef, n)
    vwork2 = FVector{V}(undef, n)
    vwork3 = FVector{V}(undef, n)
    vwork4 = FVector{V}(undef, n)
    vwork5 = FVector{V}(undef, n)

    index = bestfill_impl!(lower, upper, tree, sets, wwork0, wwork1, vwork0,
        vwork1, vwork2, vwork3, vwork4, vwork5, weights, graph, indices)

    return pairs[index]
end

function bestfill_impl!(
        lower::BipartiteGraph{V, E},
        upper::BipartiteGraph{V, E},
        tree::Parent{V},
        sets::UnionFind{V},
        wwork0::AbstractVector{W},
        wwork1::AbstractVector{W},
        vwork0::AbstractVector{V},
        vwork1::AbstractVector{V},
        vwork2::AbstractVector{V},
        vwork3::AbstractVector{V},
        vwork4::AbstractVector{V},
        vwork5::AbstractVector{V},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        indices::NTuple{N, AbstractVector{V}},
    ) where {W, V, E, N}

    minindex = zero(N); minfill = typemax(W)

    for index in oneto(N)
        fill = treefill_impl!(lower, upper, tree, sets, wwork0, wwork1, vwork0,
            vwork1, vwork2, vwork3, vwork4, vwork5, weights, graph, indices[index])

        if fill < minfill
            minindex, minfill = index, fill
        end
    end

    return minindex
end

function separatorwidth(tree::CliqueTree{V}) where {V}
    n = nov(residuals(tree)); weights = Ones{V}(n)
    return separatorwidth(weights, tree)
end

function separatorwidth(weights::AbstractVector{W}, tree::CliqueTree) where {W}
    maxwidth = zero(W)

    @inbounds for bag in tree
        width = zero(W)

        for v in separator(bag)
            width += weights[v]
        end

        maxwidth = max(maxwidth, width)
    end

    return maxwidth
end

function separatorwidth(weights::Ones{W}, tree::CliqueTree) where {W}
    maxwidth = zero(W)

    @inbounds for bag in tree
        width = convert(W, length(separator(bag)))
        maxwidth = max(maxwidth, width)
    end

    return maxwidth
end

function separatorwidth(graph; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM, snd::SupernodeType = DEFAULT_SUPERNODE_TYPE)
    return separatorwidth(graph, alg)
end

function separatorwidth(graph, alg::PermutationOrAlgorithm, snd::SupernodeType)
    perm, tree = cliquetree(graph, alg, snd)
    return separatorwidth(tree)
end

function separatorwidth(weights::AbstractVector, graph; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM, snd::SupernodeType = DEFAULT_SUPERNODE_TYPE)
    return separatorwidth(weights, graph, alg, snd)
end

function separatorwidth(weights::AbstractVector, graph, alg::PermutationOrAlgorithm, snd::SupernodeType)
    perm, tree = cliquetree(graph, alg, snd)
    return separatorwidth(weights[perm], tree)
end

"""
    residual(tree::CliqueTree, i::Integer)

Get the residual at node `i`.
"""
function residual(tree::CliqueTree, i::Integer)
    return tree.tree[i]
end

"""
    separator(tree::CliqueTree, i::Integer)

Get the separator at node `i`.
"""
function separator(tree::CliqueTree, i::Integer)
    return neighbors(separators(tree), i)
end

"""
    residuals(tree::CliqueTree)

Get the residuals of a clique tree.
"""
function residuals(tree::CliqueTree)
    return residuals(tree.tree)
end

"""
    separators(tree::CliqueTree)

Get the separators of a clique tree.
"""
function separators(tree::CliqueTree)
    return tree.graph
end

function Base.copy(tree::CliqueTree)
    return CliqueTree(copy(tree.tree), copy(tree.graph))
end

function Base.copy!(dst::CliqueTree, src::CliqueTree)
    copy!(dst.tree, src.tree)
    copy!(dst.graph, src.graph)
    return dst
end

function Base.:(==)(left::CliqueTree, right::CliqueTree)
    return left.tree == right.tree && left.graph == right.graph
end

##########################
# Indexed Tree Interface #
##########################

function AbstractTrees.rootindex(tree::CliqueTree)
    return rootindex(tree.tree)
end

function AbstractTrees.parentindex(tree::CliqueTree, i::Integer)
    return parentindex(tree.tree, i)
end

function rootindices(tree::CliqueTree)
    return rootindices(tree.tree)
end

function AbstractTrees.childindices(tree::CliqueTree, i::Integer)
    return childindices(tree.tree, i)
end

function ancestorindices(tree::CliqueTree, i::Integer)
    return ancestorindices(tree.tree, i)
end

#############################
# Abstract Vector Interface #
#############################

function Base.getindex(tree::CliqueTree, i::Integer)
    return Clique(residual(tree, i), separator(tree, i))
end

function Base.IndexStyle(::Type{<:CliqueTree})
    return IndexLinear()
end

function Base.size(tree::CliqueTree)
    return size(tree.tree)
end
