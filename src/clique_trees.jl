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

function cliquedissect(graph::AbstractGraph{V}, tree::CliqueTree{V}, alg::EliminationAlgorithm) where {V}
    E = etype(graph); n = nv(graph); m = half(de(graph)); nn = n + one(V)
    orders = Vector{Vector{V}}(undef, length(tree))
    index = Vector{V}(undef, n)

    vwork1 = Vector{V}(undef, n)
    vwork2 = Vector{V}(undef, n)
    vwork3 = Vector{V}(undef, n)
    vwork4 = Vector{V}(undef, n)
    ework1 = Vector{E}(undef, nn)
    lwork1 = BipartiteGraph{V, E}(n, n, m)
    twork1 = Tree{V}(n)

    for (j, bag) in enumerate(tree)
        subgraph, order = induced_subgraph(graph, bag)
        perm, invp = permutation(subgraph, alg)
        upper = sympermute(subgraph, invp, Forward)
        invpermute!(order, invp); index[order] = vertices(upper)

        for i in childindices(tree, j)
            n = nv(upper); m = ne(upper); nn = n + one(V)
            alpha = view(vwork1, oneto(n))
            clique = view(index, separator(tree, i))

            compositerotations_impl!(
                alpha,
                vwork2,
                vwork3,
                vwork4,
                ework1,
                BipartiteGraph(
                    n,
                    n,
                    m,
                    lwork1.ptr,
                    lwork1.tgt,
                ),
                Tree(
                    n,
                    twork1.parent,
                    twork1.root,
                    twork1.child,
                    twork1.brother,
                ),
                upper,
                clique,
            )

            invpermute!(order, alpha)
            prepend!(order, orders[i])
            upper = isu!(graph, order, index)
        end

        n = nv(upper); m = ne(upper); nn = n + one(V)
        alpha = view(vwork1, oneto(n))
        clique = view(index, separator(tree, j))

        compositerotations_impl!(
            alpha,
            vwork2,
            vwork3,
            vwork4,
            ework1,
            BipartiteGraph(
                n,
                n,
                m,
                lwork1.ptr,
                lwork1.tgt,
            ),
            Tree(
                n,
                twork1.parent,
                twork1.root,
                twork1.child,
                twork1.brother,
            ),
            upper,
            clique,
        )

        invpermute!(order, alpha)

        for _ in separator(tree, j)
            pop!(order)
        end

        orders[j] = order
    end

    order = V[]

    for j in rootindices(tree)
        append!(order, orders[j])
    end

    return order
end

function cliquedissect(weights::AbstractVector, graph::AbstractGraph{V}, tree::CliqueTree{V}, alg::EliminationAlgorithm) where {V}
    E = etype(graph); n = nv(graph); m = half(de(graph)); nn = n + one(V)
    orders = Vector{Vector{V}}(undef, length(tree))
    index = Vector{V}(undef, n)

    vwork1 = Vector{V}(undef, n)
    vwork2 = Vector{V}(undef, n)
    vwork3 = Vector{V}(undef, n)
    vwork4 = Vector{V}(undef, n)
    ework1 = Vector{E}(undef, nn)
    lwork1 = BipartiteGraph{V, E}(n, n, m)
    twork1 = Tree{V}(n)

    for (j, bag) in enumerate(tree)
        subgraph, order = induced_subgraph(graph, bag)
        perm, invp = permutation(view(weights, order), subgraph, alg)
        upper = sympermute(subgraph, invp, Forward)
        invpermute!(order, invp); index[order] = vertices(upper)

        for i in childindices(tree, j)
            n = nv(upper); m = ne(upper); nn = n + one(V)
            alpha = view(vwork1, oneto(n))
            clique = view(index, separator(tree, i))

            compositerotations_impl!(
                alpha,
                vwork2,
                vwork3,
                vwork4,
                ework1,
                BipartiteGraph(
                    n,
                    n,
                    m,
                    lwork1.ptr,
                    lwork1.tgt,
                ),
                Tree(
                    n,
                    twork1.parent,
                    twork1.root,
                    twork1.child,
                    twork1.brother,
                ),
                upper,
                clique,
            )

            invpermute!(order, alpha)
            prepend!(order, orders[i])
            upper = isu!(graph, order, index)
        end

        n = nv(upper); m = ne(upper); nn = n + one(V)
        alpha = view(vwork1, oneto(n))
        clique = view(index, separator(tree, j))

        compositerotations_impl!(
            alpha,
            vwork2,
            vwork3,
            vwork4,
            ework1,
            BipartiteGraph(
                n,
                n,
                m,
                lwork1.ptr,
                lwork1.tgt,
            ),
            Tree(
                n,
                twork1.parent,
                twork1.root,
                twork1.child,
                twork1.brother,
            ),
            upper,
            clique,
        )
        invpermute!(order, alpha)

        for _ in separator(tree, j)
            pop!(order)
        end

        orders[j] = order
    end

    order = V[]

    for j in rootindices(tree)
        append!(order, orders[j])
    end

    return order
end

function isu!(graph::AbstractGraph{V}, order::AbstractVector{V}, index::AbstractVector{V}) where {V}
    index .= n = zero(V)

    for v in order
        index[v] = n += one(V)
    end

    E = etype(graph)
    subgraph = BipartiteGraph{V, E}(n, n, zero(E))
    pointers(subgraph)[begin] = p = one(E)

    for v in order
        i = index[v]

        for w in neighbors(graph, v)
            j = index[w]

            if i > j > zero(V)
                p += one(E)
                push!(targets(subgraph), j)
            end
        end

        pointers(subgraph)[i + one(V)] = p
    end

    return subgraph
end

function safetree(graph; alg::PermutationOrAlgorithm = MinimalChordal())
    return safetree(graph, alg)
end

function safetree(graph, alg::PermutationOrAlgorithm)
    return safetree(BipartiteGraph(graph), alg)
end

function safetree(graph::AbstractGraph, alg::PermutationOrAlgorithm)
    graph = Graph(graph)
    return graph, safetree!(graph, alg)...
end

# A Heuristic for Listing Almost-Clique Minimal Separators of a Graph
# Tamaki
function safetree!(graph::Graph{V}, alg::PermutationOrAlgorithm) where {V}
    n = nv(graph)
    marker = zeros(Int, n); tag = 0; flag = true

    while flag
        flag = false
        label, tree = cliquetree(graph, alg, Maximal())

        for bag in tree
            sep = separator(bag)
            num = convert(V, length(sep))
            ww = zero(V); tag += 1

            for v in sep
                marker[label[v]] = tag
            end

            for v in sep
                count = num - one(V)

                for w in neighbors(graph, label[v])
                    if marker[w] == tag
                        count -= one(V)
                    end
                end

                if ispositive(count)
                    if ispositive(ww)
                        ww = zero(V)
                        break
                    else
                        ww = label[v]
                    end
                end
            end

            if ispositive(ww)
                flag = true

                for v in sep
                    w = label[v]

                    if w != ww
                        add_edge!(graph, w, ww)
                    end
                end
            end
        end
    end

    return atomtree(graph, MCSM())
end

function atomtree(graph; alg::PermutationOrAlgorithm = MCSM())
    return atomtree(graph, alg)
end

function atomtree(graph, alg::PermutationOrAlgorithm)
    return atomtree(BipartiteGraph(graph), alg)
end

function atomtree(graph::AbstractGraph{V}, alg::PermutationOrAlgorithm) where {V}
    return atomtree(graph, cliquetree(graph, alg, Maximal())...)
end

# Organizing the Atoms of the Clique Separator Decomposition into an Atom Tree
# Berry, Pogorelcnik, and Simonet
# Algorithm Atom-Tree
function atomtree(graph::AbstractGraph{V}, label::AbstractVector{V}, tree::CliqueTree{V, E}) where {V, E}
    n = nv(graph); m = last(Tree(tree))
    marker = zeros(Int, n); tag = 0
    atom = Vector{V}(undef, m)
    a = zero(V)
    b = zero(E)

    # identify clique separators
    for i in reverse(oneto(m))
        sep = separator(tree, i)
        flag = true
        s = zero(V)

        while flag && s < length(sep)
            ss = s += one(V)
            v = label[sep[s]]
            marker[neighbors(graph, v)] .= tag += 1

            while flag && one(V) < ss
                ss -= one(V)
                vv = label[sep[ss]]

                if marker[vv] < tag
                    flag = false
                end
            end
        end

        if flag
            atom[i] = a += one(V)
            b += convert(E, length(sep))
        else
            j = parentindex(tree, i)
            atom[i] = atom[j]
        end
    end

    # construct postordered atom tree
    parent = Vector{V}(undef, a)

    for i in oneto(m)
        j = parentindex(tree, i)

        if isnothing(j)
            parent[atom[i]] = zero(V)
        elseif atom[i] != atom[j]
            parent[atom[i]] = atom[j]
        end
    end

    atree = Tree(parent)
    aindex = postorder!(atree)

    for i in oneto(m)
        atom[i] = aindex[atom[i]]
    end

    # invert quotient map
    ainv = Vector{Vector{V}}(undef, a)

    for aa in oneto(a)
        ainv[aa] = V[]
    end

    for i in oneto(m)
        push!(ainv[atom[i]], i)
    end

    # construct residuals
    index = Vector{V}(undef, n)
    sndptr = Vector{V}(undef, a + one(V))
    sndptr[begin] = p = one(V)

    for aa in oneto(a)
        set = ainv[aa]

        for i in set, v in residual(tree, i)
            index[v] = p
            p += one(V)
        end

        sndptr[aa + one(V)] = p
    end

    # construct separators
    sepval = Vector{V}(undef, b)
    sepptr = Vector{E}(undef, a + one(V))
    sepptr[begin] = q = one(E)

    for aa in oneto(a)
        set = ainv[aa]

        for v in separator(tree, last(set))
            sepval[q] = index[v]
            q += one(V)
        end

        sepptr[aa + one(V)] = q
    end

    # return atom tree
    invpermute!(label, index)
    stree = SupernodeTree(atree, BipartiteGraph(n, m, n, sndptr, oneto(n)))
    ctree = CliqueTree(stree, BipartiteGraph(n, a, b, sepptr, sepval))
    return label, ctree
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

@views function cliquetree(label, tree, count, index, ptr, lower, upper)
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
    n::V = maximum(length, tree; init = 1) - 1
    return n
end

function treewidth(weights::AbstractVector, tree::CliqueTree{V}) where {V}
    W = eltype(weights)
    treewidth = zero(W)

    for clique in tree
        width = zero(W)

        for v in clique
            width += weights[v]
        end

        treewidth = max(treewidth, width)
    end

    return treewidth
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

# Simple Linear-Time Algorithms to Test Chordality of BipartiteGraphs, Test Acyclicity of Hypergraphs, and Selectively Reduce Acyclic Hypergraphs
# Tarjan and Yannakakis
# Fill-In computation
#
# Compute the width of an elimination ordering.
# The complexity is O(m' + n), where m' = |E'| and n = |V|.
function treewidth(graph::AbstractGraph{V}, alg::PermutationOrAlgorithm) where {V}
    n = nv(graph)
    f = Vector{V}(undef, n)
    findex = Vector{V}(undef, n)
    counts = Vector{V}(undef, n)
    return treewidth_impl!(f, findex, counts, graph, permutation(graph, alg)...)
end

function treewidth_impl!(
        f::AbstractVector{V},
        findex::AbstractVector{V},
        counts::AbstractVector{V},
        graph::AbstractGraph{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
    ) where {V}
    @argcheck nv(graph) <= length(f)
    @argcheck nv(graph) <= length(findex)
    @argcheck nv(graph) <= length(counts)
    @argcheck nv(graph) <= length(order)
    @argcheck nv(graph) <= length(index)
    width = zero(V)

    @inbounds for i in vertices(graph)
        w = order[i]
        f[w] = w
        findex[w] = i

        counts[w] = zero(V)

        for v in neighbors(graph, w)
            if index[v] < i
                x = v

                while findex[x] < i
                    findex[x] = i
                    width = max(width, counts[x] += one(V))
                    x = f[x]
                end

                if f[x] == x
                    f[x] = w
                end
            end
        end
    end

    return width
end

function treewidth(weights::AbstractVector, graph; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM)
    return treewidth(weights, graph, alg)
end

function treewidth(weights::AbstractVector, graph, alg::PermutationOrAlgorithm)
    return treewidth(weights, BipartiteGraph(graph), alg)
end

# Simple Linear-Time Algorithms to Test Chordality of BipartiteGraphs, Test Acyclicity of Hypergraphs, and Selectively Reduce Acyclic Hypergraphs
# Tarjan and Yannakakis
# Fill-In computation
#
# Compute the weighted width of an elimination ordering.
# The complexity is O(m' + n), where m' = |E'| and n = |V|.
function treewidth(weights::AbstractVector{W}, graph::AbstractGraph{V}, alg::PermutationOrAlgorithm) where {W, V}
    n = nv(graph)
    f = Vector{V}(undef, n)
    findex = Vector{V}(undef, n)
    counts = Vector{W}(undef, n)
    return treewidth_impl!(f, findex, counts, weights, graph, permutation(weights, graph, alg)...)
end

function treewidth_impl!(
        f::AbstractVector{V},
        findex::AbstractVector{V},
        counts::AbstractVector{W},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
    ) where {W, V}
    @argcheck nv(graph) <= length(f)
    @argcheck nv(graph) <= length(findex)
    @argcheck nv(graph) <= length(counts)
    @argcheck nv(graph) <= length(weights)
    @argcheck nv(graph) <= length(order)
    @argcheck nv(graph) <= length(index)
    width = zero(W)

    @inbounds for i in vertices(graph)
        w = order[i]
        f[w] = w
        findex[w] = i

        weight = counts[w] = weights[w]
        width = max(width, weight)

        for v in neighbors(graph, w)
            if index[v] < i
                x = v

                while findex[x] < i
                    findex[x] = i
                    width = max(width, counts[x] += weight)
                    x = f[x]
                end

                if f[x] == x
                    f[x] = w
                end
            end
        end
    end

    return width
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

# Simple Linear-Time Algorithms to Test Chordality of BipartiteGraphs, Test Acyclicity of Hypergraphs, and Selectively Reduce Acyclic Hypergraphs
# Tarjan and Yannakakis
# Fill-In computation
#
# Compute the fill-in of an elimination ordering.
# The complexity is O(m' + n), where m' = |E'| and n = |V|.
function treefill(graph::AbstractGraph{V}, alg::PermutationOrAlgorithm) where {V}
    n = nv(graph)
    f = Vector{V}(undef, n)
    findex = Vector{V}(undef, n)
    return treefill_impl!(f, findex, graph, permutation(graph, alg)...)
end

function treefill_impl!(
        f::AbstractVector{V},
        findex::AbstractVector{V},
        graph::AbstractGraph{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
    ) where {V}
    @argcheck nv(graph) <= length(f)
    @argcheck nv(graph) <= length(findex)
    @argcheck nv(graph) <= length(order)
    @argcheck nv(graph) <= length(index)
    E = etype(graph); fill = zero(E)

    @inbounds for i in vertices(graph)
        w = order[i]
        f[w] = w
        findex[w] = i

        for v in neighbors(graph, w)
            if index[v] < i
                x = v

                while findex[x] < i
                    findex[x] = i
                    fill += one(E)
                    x = f[x]
                end

                if f[x] == x
                    f[x] = w
                end
            end
        end
    end

    return fill
end

function treefill(weights::AbstractVector, graph; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM)
    return treefill(weights, graph, alg)
end

function treefill(weights::AbstractVector, graph, alg::PermutationOrAlgorithm)
    return treefill(weights, BipartiteGraph(graph), alg)
end

# Simple Linear-Time Algorithms to Test Chordality of BipartiteGraphs, Test Acyclicity of Hypergraphs, and Selectively Reduce Acyclic Hypergraphs
# Tarjan and Yannakakis
# Fill-In computation
#
# Compute the weighted fill-in of an elimination ordering.
# The complexity is O(m' + n), where m' = |E'| and n = |V|.
function treefill(weights::AbstractVector{W}, graph::AbstractGraph{V}, alg::PermutationOrAlgorithm) where {W, V}
    n = nv(graph)
    f = Vector{V}(undef, n)
    findex = Vector{V}(undef, n)
    return treefill_impl!(f, findex, weights, graph, permutation(weights, graph, alg)...)
end

function treefill_impl!(
        f::AbstractVector{V},
        findex::AbstractVector{V},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
    ) where {W, V}
    @argcheck nv(graph) <= length(f)
    @argcheck nv(graph) <= length(findex)
    @argcheck nv(graph) <= length(weights)
    @argcheck nv(graph) <= length(order)
    @argcheck nv(graph) <= length(index)
    fill = zero(W)

    @inbounds for i in vertices(graph)
        w = order[i]
        f[w] = w
        findex[w] = i

        weight = weights[w]; fill += weight * weight

        for v in neighbors(graph, w)
            if index[v] < i
                x = v

                while findex[x] < i
                    findex[x] = i
                    fill += weights[x] * weight
                    x = f[x]
                end

                if f[x] == x
                    f[x] = w
                end
            end
        end
    end

    return fill
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
