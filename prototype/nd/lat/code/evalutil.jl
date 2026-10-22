using CliqueTrees, Graphs, Random
using CliqueTrees: BipartiteGraph, bestwork, sympermute!_impl!, reverse!_impl!, etree_impl!, supcnt_impl!, Forward, half, de, etype

# column weights w(v) + sum of higher filled neighbours, indexed by elimination position
function colweights(weights::AbstractVector{W}, graph0, order::AbstractVector) where {W}
    graph = BipartiteGraph(graph0)
    V = eltype(graph); E = etype(graph); n = nv(graph); m = half(de(graph))
    index = invperm(order)
    lower, upper, tree, sets, wwork0, wwork1, vwork0, vwork1, vwork2, vwork3, vwork4, vwork5 = bestwork(W, V, E, n, m)
    for v in 1:n
        wwork1[index[v]] = weights[v]
    end
    sympermute!_impl!(upper, graph, convert(Vector{V}, index), Forward)
    reverse!_impl!(lower, upper)
    etree_impl!(tree, vwork0, upper)
    supcnt_impl!(wwork0, vwork0, vwork1, vwork2, vwork3, vwork4, vwork5, sets, wwork1, lower, tree)
    return W[wwork0[i] for i in 1:n]
end

isperm_of(order, n) = length(order) == n && isperm(order)

# returns (width, log2 total cost) with total cost = sum_v 2^col(v)
function quality(weights, graph, order)
    n = nv(graph)
    @assert isperm_of(order, n) "not a permutation"
    w = treewidth(weights, graph, order)
    cw = colweights(weights, graph, order)
    @assert maximum(cw) ≈ w
    mx = maximum(cw)
    lc = mx + log2(sum(exp2(c - mx) for c in cw))
    return w, lc
end
