"""
    CliqueTree{V, E} <: AbstractVector{Clique{V, E}}

A [clique tree](https://en.wikipedia.org/wiki/Tree_decomposition) with vertices of type `V` and edges of type `E`.
This type implements the [indexed tree interface](https://juliacollections.github.io/AbstractTrees.jl/stable/#The-Indexed-Tree-Interface).
"""
struct CliqueTree{V, E} <: AbstractVector{Clique{V, E}}
    tree::SupernodeTree{V}
    sep::BipartiteGraph{V, E, Vector{E}, Vector{V}}
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

```jldoctest
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
    return cliquetree(supernodetree(graph, alg, snd)...)
end

function cliquetree(weights::AbstractVector, graph, alg::PermutationOrAlgorithm, snd::SupernodeType)
    return cliquetree(supernodetree(weights, graph, alg, snd)...)
end

@views function cliquetree(label, tree, index, ptr, lower, upper)
    lower = sympermute!(upper, lower, index, Reverse)

    # compute separators
    function diff(col, res)
        i = 1

        while i in eachindex(col) && col[i] in res
            i += 1
        end

        return col[i:end]
    end

    V = eltype(lower)
    E = etype(lower)
    h = nv(lower); n = last(tree.tree); nn = n + one(V); m = ptr[nn] - one(E)
    sep = BipartiteGraph(h, n, m, ptr, Vector{V}(undef, m))
    cache = Vector{V}(undef, Δout(sep))

    for (j, res) in enumerate(tree)
        # get representative vertex
        vertex = res[begin]

        # subtract residual from higher neighbors
        column = diff(neighbors(lower, vertex), res)

        # initialize separator
        state = neighbors(sep, j)[eachindex(column)] .= column

        # iterate over children
        i = firstchildindex(tree, j)

        while !isnothing(i) && length(state) < outdegree(sep, j)
            # subtract residual from child separator
            column = diff(neighbors(sep, i), res)

            # update separator
            union = mergesorted!(cache, state, column)
            state = neighbors(sep, j)[eachindex(union)] .= union

            # update child
            i = nextsiblingindex(tree, i)
        end
    end

    # construct clique tree
    return label, CliqueTree(tree, sep)
end

function cliquetree(tree::CliqueTree{V, E}, root::Integer) where {V <: Signed, E <: Signed}
    return cliquetree(tree, V(root))
end

function cliquetree(tree::CliqueTree{V, E}, root::V) where {V <: Signed, E <: Signed}
    n = last(Tree(tree)); h = root; nn = n + one(V)
    parent = Vector{V}(undef, n)

    for i in Tree(tree)
        j = parentindex(tree, i)

        if isnothing(j)
            j = zero(V)

            if h == i
                i = root
            end
        else
            if h == i
                h = j; j = i; i = h
            end
        end

        parent[i] = j
    end

    root = h
    sv = nov(tree.sep); rv = nov(tree.tree.res)
    se = ne(tree.sep); re = ne(tree.tree.res)
    etree = Tree(parent)
    eindex = postorder!(etree)
    eorder = invperm(eindex)
    sepval = Vector{V}(undef, se)
    index = Vector{V}(undef, re)
    sepptr = Vector{E}(undef, nn)
    sndptr = Vector{V}(undef, nn)
    sepptr[begin] = p = one(E)
    sndptr[begin] = q = one(V)

    for ii in etree
        i = j = eorder[ii]

        if h == i
            jj = parentindex(etree, ii)

            if isnothing(jj)
                j = root
            else
                h = j = eorder[jj]
            end
        end

        sep = separator(tree, j)
        len = V(length(sep)); s = one(V)

        for v in tree[i]
            if s <= len && v == sep[s]
                sepval[p] = v
                p += one(E)
                s += one(V)
            else
                index[v] = q
                q += one(V)
            end
        end

        sepptr[ii + one(V)] = p
        sndptr[ii + one(V)] = q
    end

    sepval = index[sepval]; sndval = oneto(rv)
    stree = SupernodeTree(etree, BipartiteGraph(rv, n, q - one(V), sndptr, sndval))
    ctree = CliqueTree(stree, BipartiteGraph(sv, n, p - one(E), sepptr, sepval))
    return invperm(index), ctree
end

function cliquetree!(tree::CliqueTree, root::Integer)
    label, _tree = cliquetree(tree, root)
    copy!(tree, _tree)
    return label
end

"""
    treewidth([weights, ]tree::CliqueTree)

Compute the [width](https://en.wikipedia.org/wiki/Treewidth) of a clique tree.

```jldoctest
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
    return tree.sep
end

"""
    relatives(tree::CliqueTree)

Compute the relative indices of a clique tree.
"""
function relatives(tree::CliqueTree{V}) where {V}
    sep = separators(tree); h = nov(sep); n = nv(sep); m = ne(sep)
    tgt = Vector{V}(undef, m)
    graph = BipartiteGraph(h, n, m, pointers(sep), tgt)

    for i in Tree(tree)
        j = parentindex(tree, i)

        if !isnothing(j)
            indexinsorted!(neighbors(graph, i), separator(tree, i), tree[j])
        end
    end

    return graph
end

function Base.copy(tree::CliqueTree)
    return CliqueTree(copy(tree.tree), copy(tree.sep))
end

function Base.copy!(dst::CliqueTree, src::CliqueTree)
    copy!(dst.tree, src.tree)
    copy!(dst.sep, src.sep)
    return dst
end

function Base.:(==)(left::CliqueTree, right::CliqueTree)
    return left.tree == right.tree && left.sep == right.sep
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

function firstchildindex(tree::CliqueTree, i::Integer)
    return firstchildindex(tree.tree, i)
end

function AbstractTrees.nextsiblingindex(tree::CliqueTree, i::Integer)
    return nextsiblingindex(tree.tree, i)
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
