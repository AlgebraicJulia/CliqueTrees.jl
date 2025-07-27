"""
    CliqueTree{V, E} <: AbstractVector{Clique{V, E}}

A tree T = (V, E) and functions
    clique: V → 2ᵁ
    separator: V → 2ᵁ
This type implements the [indexed tree interface](https://juliacollections.github.io/AbstractTrees.jl/stable/#The-Indexed-Tree-Interface).
"""
struct CliqueTree{V, E} <: AbstractVector{Clique{V, E}}
    """
    The tree T = (V, E).
    """
    tree::SupernodeTree{V}

    """
    A directed bipartite graph G = (U, V, E') with an arc

        (v, u) ∈ E'

    for all vertices v ∈ V and u ∈ separator(v).
    """
    graph::BipartiteGraph{V, E, FVector{E}, FVector{V}}

    function CliqueTree{V, E}(tree::SupernodeTree{V}, graph::BipartiteGraph{V, E}) where {V, E}
        @argcheck nv(residuals(tree)) == nv(graph)
        @argcheck nov(residuals(tree)) == nov(graph)
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

```julia
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

function cliquetree(graph::AbstractGraph{V}, alg::PermutationOrAlgorithm, snd::SupernodeType) where {V}
    n = nv(graph); weights = Ones{V}(n)
    return cliquetree(weights, graph, alg, snd)
end

function cliquetree(weights::AbstractVector, graph, alg::PermutationOrAlgorithm, snd::SupernodeType)
    return cliquetree(weights, BipartiteGraph(graph), alg, snd)
end

function cliquetree(weights::AbstractVector, graph::AbstractGraph{V}, alg::PermutationOrAlgorithm, snd::SupernodeType) where {V}
    E = etype(graph); n = nv(graph); m = half(de(graph))
    nn = n + one(V); nnn = nn + one(V)
    
    target1 = FVector{V}(undef, max(m, nn))
    target2 = FVector{V}(undef, m)
    target3 = FVector{V}(undef, n)

    ework1 = FVector{E}(undef, n)
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
        target3, ework1, pointer1, pointer2, pointer3, colcount,
        elmorder, elmindex, sndptr, sepptr, new, parent, elmtree, graph,
        order, index, snd)

    h = last(sndtree.tree); k = sepptr[h + one(V)] - one(E)
    septgt = FVector{V}(undef, k)
    separator = BipartiteGraph(n, h, k, sepptr, septgt)

    clqtree = cliquetree_impl!(ework1, elmindex,
        upper, lower, separator, sndtree)

    return order, clqtree
end

function cliquetree_impl!(
        count::AbstractVector{E},
        index::AbstractVector{V},
        target::BipartiteGraph{V, E},
        source::BipartiteGraph{V, E},
        separator::BipartiteGraph{V, E},
        sndtree::SupernodeTree{V},
    ) where {V, E}
    n = last(sndtree.tree)
    lower = sympermute!_impl!(count, target, source, index, Reverse)
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

"""
    treewidth([weights, ]tree::CliqueTree)

Compute the [width](https://en.wikipedia.org/wiki/Treewidth) of a clique tree.

```julia
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
    n::V = maximum(length, tree; init = 0) - 1
    return n
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
    E = etype(graph); m = half(de(graph)); n = nv(graph); nn = n + one(V)
    order, index = permutation(weights, graph, alg)
    
    lower = BipartiteGraph(
        n,
        n,
        m,
        FVector{E}(undef, nn),
        FVector{V}(undef, m),
    )

    upper = BipartiteGraph(
        n,
        n,
        m,
        FVector{E}(undef, nn),
        FVector{V}(undef, m),
    )
    
    tree = Parent(
        n,
        FVector{V}(undef, n),    
    )

    sets = UnionFind(
        FVector{V}(undef, n),
        FVector{V}(undef, n),
        FVector{V}(undef, n),
    )
    
    wwork0 = FVector{W}(undef, n)
    wwork1 = FVector{W}(undef, n)
    vwork0 = FVector{V}(undef, n)
    vwork1 = FVector{V}(undef, n)
    vwork2 = FVector{V}(undef, n)
    vwork3 = FVector{V}(undef, n)
    vwork4 = FVector{V}(undef, n)
    vwork5 = FVector{V}(undef, n)
    ework0 = FVector{E}(undef, n)
    
    width = treewidth_impl!(lower, upper, tree, sets, wwork0, wwork1, vwork0,
        vwork1, vwork2, vwork3, vwork4, vwork5, ework0, weights, graph, index)

    return width
end

function treewidth_impl!(
        lower::BipartiteGraph{V},
        upper::BipartiteGraph{V},
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
        ework0::AbstractVector{E},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        index::AbstractVector{V},
    ) where {W, V, E}
    @argcheck nv(graph) <= length(weights)
    @argcheck nv(graph) <= length(wwork1)
    
    @inbounds for v in vertices(graph)
        wwork1[index[v]] = weights[v]
    end
    
    sympermute!_impl!(ework0, upper, graph, index, Forward)
    reverse!_impl!(ework0, lower, upper)
    etree_impl!(tree, vwork0, upper)
    
    supcnt_impl!(wwork0, vwork0, vwork1, vwork2, vwork3,
        vwork4, vwork5, sets, wwork1, lower, tree)

    width = zero(W)

    @inbounds for v in vertices(lower)
        width = max(width, wwork0[v])
    end

    return width
end

function bestwidth(graph, algs::NTuple{<:Any, PermutationOrAlgorithm})
    return bestwidth(BipartiteGraph(graph), algs)
end

function bestwidth(graph::AbstractGraph{V}, algs::NTuple{<:Any, PermutationOrAlgorithm}) where {V}
    n = nv(graph); weights = Ones{V}(n)
    return bestwidth(weights, graph, algs)
end

function bestwidth(weights::AbstractVector, graph, algs::NTuple{<:Any, PermutationOrAlgorithm})
    return bestwidth(weights, BipartiteGraph(graph), algs)
end

function bestwidth(weights::AbstractVector{W}, graph::AbstractGraph{V}, algs::NTuple{<:Any, PermutationOrAlgorithm}) where {W, V}
    E = etype(graph); m = de(graph); n = nv(graph); nn = n + one(V)
    
    pairs = map(algs) do alg
        return permutation(weights, graph, alg)
    end

    indices = map(pairs) do (order, index)
        return index
    end

    lower = BipartiteGraph(
        n,
        n,
        m,
        FVector{E}(undef, nn),
        FVector{V}(undef, m),
    )

    upper = BipartiteGraph(
        n,
        n,
        m,
        FVector{E}(undef, nn),
        FVector{V}(undef, m),
    )
    
    tree = Parent(
        n,
        FVector{V}(undef, n),    
    )

    sets = UnionFind(
        FVector{V}(undef, n),
        FVector{V}(undef, n),
        FVector{V}(undef, n),
    )

    wwork0 = FVector{W}(undef, n)
    wwork1 = FVector{W}(undef, n)
    vwork0 = FVector{V}(undef, n)
    vwork1 = FVector{V}(undef, n)
    vwork2 = FVector{V}(undef, n)
    vwork3 = FVector{V}(undef, n)
    vwork4 = FVector{V}(undef, n)
    vwork5 = FVector{V}(undef, n)
    ework0 = FVector{E}(undef, n)

    index = bestwidth_impl!(lower, upper, tree, sets, wwork0, wwork1, vwork0,
        vwork1, vwork2, vwork3, vwork4, vwork5, ework0, weights, graph, indices)

    return pairs[index]
end

function bestwidth_impl!(
        lower::BipartiteGraph{I},
        upper::BipartiteGraph{I},
        tree::Parent{I},
        sets::UnionFind{I},
        wwork0::AbstractVector{W},
        wwork1::AbstractVector{W},
        vwork0::AbstractVector{I},
        vwork1::AbstractVector{I},
        vwork2::AbstractVector{I},
        vwork3::AbstractVector{I},
        vwork4::AbstractVector{I},
        vwork5::AbstractVector{I},
        weights::AbstractVector{W},
        graph::AbstractGraph{I},
        indices::NTuple{N, AbstractVector{I}},
    ) where {W, I, N}
    ework0 = vwork0

    index = bestwidth_impl!(lower, upper, tree, sets, wwork0, wwork1, vwork0,
        vwork1, vwork2, vwork3, vwork4, vwork5, ework0, weights, graph, indices)

    return index
end

function bestwidth_impl!(
        lower::BipartiteGraph{V},
        upper::BipartiteGraph{V},
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
        ework0::AbstractVector{E},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        indices::NTuple{N, AbstractVector{V}},
    ) where {W, V, E, N}

    minindex = zero(N); minwidth = typemax(W)

    for index in oneto(N)
        width = treewidth_impl!(lower, upper, tree, sets, wwork0, wwork1, vwork0,
            vwork1, vwork2, vwork3, vwork4, vwork5, ework0, weights, graph, indices[index])

        if width < minwidth
            minindex, minwidth = index, width
        end
    end

    return minindex
end

function treefill(tree::CliqueTree{<:Any, E}) where {E}
    fill = zero(E)

    @inbounds for bag in tree
        res = residual(bag)

        for vv in bag
            for v in res
                v == vv && break
                fill += one(E)
            end
        end
    end

    return fill
end

function treefill(weights::AbstractVector{W}, tree::CliqueTree) where {W}
    fill = zero(W)

    @inbounds for bag in tree
        res = residual(bag)

        for vv in bag
            wvv = weights[vv]

            for v in res
                v > vv && break
                wv = weights[v]
                fill += wvv * wv
            end
        end
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
    E = etype(graph); m = de(graph); n = nv(graph); nn = n + one(V)
    order, index = permutation(weights, graph, alg)
    
    lower = BipartiteGraph(
        n,
        n,
        m,
        FVector{E}(undef, nn),
        FVector{V}(undef, m),
    )

    upper = BipartiteGraph(
        n,
        n,
        m,
        FVector{E}(undef, nn),
        FVector{V}(undef, m),
    )
    
    tree = Parent(
        n,
        FVector{V}(undef, n),    
    )

    sets = UnionFind(
        FVector{V}(undef, n),
        FVector{V}(undef, n),
        FVector{V}(undef, n),
    )
    
    wwork0 = FVector{W}(undef, n)
    wwork1 = FVector{W}(undef, n)
    vwork0 = FVector{V}(undef, n)
    vwork1 = FVector{V}(undef, n)
    vwork2 = FVector{V}(undef, n)
    vwork3 = FVector{V}(undef, n)
    vwork4 = FVector{V}(undef, n)
    vwork5 = FVector{V}(undef, n)
    ework0 = FVector{E}(undef, n)
    
    fill = treefill_impl!(lower, upper, tree, sets, wwork0, wwork1, vwork0,
        vwork1, vwork2, vwork3, vwork4, vwork5, ework0, weights, graph, index)

    return fill
end

function treefill_impl!(
        lower::BipartiteGraph{V},
        upper::BipartiteGraph{V},
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
        ework0::AbstractVector{E},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        index::AbstractVector{V},
    ) where {W, V, E}
    @argcheck nv(graph) <= length(weights)
    @argcheck nv(graph) <= length(wwork1)
    
    @inbounds for v in vertices(graph)
        wwork1[index[v]] = weights[v]
    end
    
    sympermute!_impl!(ework0, upper, graph, index, Forward)
    reverse!_impl!(ework0, lower, upper)
    etree_impl!(tree, vwork0, upper)
    
    supcnt_impl!(wwork0, vwork0, vwork1, vwork2, vwork3,
        vwork4, vwork5, sets, wwork1, lower, tree)
    
    fill = zero(W)

    @inbounds for v in vertices(lower)
        fill += wwork1[v] * wwork0[v]
    end

    return fill
end

function bestfill(graph, algs::NTuple{<:Any, PermutationOrAlgorithm})
    return bestfill(BipartiteGraph(graph), algs)
end

function bestfill(graph::AbstractGraph, algs::NTuple{<:Any, PermutationOrAlgorithm})
    E = etype(graph); n = nv(graph); weights = Ones{E}(n)
    return bestfill(weights, graph, algs)
end

function bestfill(weights::AbstractVector, graph, algs::NTuple{<:Any, PermutationOrAlgorithm})
    return bestfill(weights, BipartiteGraph(graph), algs)
end

function bestfill(weights::AbstractVector{W}, graph::AbstractGraph{V}, algs::NTuple{<:Any, PermutationOrAlgorithm}) where {W, V}
    E = etype(graph); m = de(graph); n = nv(graph); nn = n + one(V)
    
    pairs = map(algs) do alg
        return permutation(weights, graph, alg)
    end

    indices = map(pairs) do (order, index)
        return index
    end

    lower = BipartiteGraph(
        n,
        n,
        m,
        FVector{E}(undef, nn),
        FVector{V}(undef, m),
    )

    upper = BipartiteGraph(
        n,
        n,
        m,
        FVector{E}(undef, nn),
        FVector{V}(undef, m),
    )
    
    tree = Parent(
        n,
        FVector{V}(undef, n),    
    )

    sets = UnionFind(
        FVector{V}(undef, n),
        FVector{V}(undef, n),
        FVector{V}(undef, n),
    )

    wwork0 = FVector{W}(undef, n)
    wwork1 = FVector{W}(undef, n)
    vwork0 = FVector{V}(undef, n)
    vwork1 = FVector{V}(undef, n)
    vwork2 = FVector{V}(undef, n)
    vwork3 = FVector{V}(undef, n)
    vwork4 = FVector{V}(undef, n)
    vwork5 = FVector{V}(undef, n)
    ework0 = FVector{E}(undef, n)

    index = bestfill_impl!(lower, upper, tree, sets, wwork0, wwork1, vwork0,
        vwork1, vwork2, vwork3, vwork4, vwork5, ework0, weights, graph, indices)

    return pairs[index]
end

function bestfill_impl!(
        lower::BipartiteGraph{I},
        upper::BipartiteGraph{I},
        tree::Parent{I},
        sets::UnionFind{I},
        wwork0::AbstractVector{W},
        wwork1::AbstractVector{W},
        vwork0::AbstractVector{I},
        vwork1::AbstractVector{I},
        vwork2::AbstractVector{I},
        vwork3::AbstractVector{I},
        vwork4::AbstractVector{I},
        vwork5::AbstractVector{I},
        weights::AbstractVector{W},
        graph::AbstractGraph{I},
        indices::NTuple{N, AbstractVector{I}},
    ) where {W, I, N}
    ework0 = vwork0

    index = bestfill_impl!(lower, upper, tree, sets, wwork0, wwork1, vwork0,
        vwork1, vwork2, vwork3, vwork4, vwork5, ework0, weights, graph, indices)

    return index
end

function bestfill_impl!(
        lower::BipartiteGraph{V},
        upper::BipartiteGraph{V},
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
        ework0::AbstractVector{E},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        indices::NTuple{N, AbstractVector{V}},
    ) where {W, V, E, N}

    minindex = zero(N); minfill = typemax(W)

    for index in oneto(N)
        fill = treefill_impl!(lower, upper, tree, sets, wwork0, wwork1, vwork0,
            vwork1, vwork2, vwork3, vwork4, vwork5, ework0, weights, graph, indices[index])

        if fill < minfill
            minindex, minfill = index, fill
        end
    end

    return minindex
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
