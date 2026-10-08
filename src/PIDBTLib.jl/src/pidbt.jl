# Returns an elimination order, or `nothing` if the deadline (in the sense of
# `time()`) passes first.
function pidbt(weights::AbstractVector{Int}, g::Graphs.AbstractGraph, min_k::Int=0; deadline::Float64=Inf)
    n = convert(Int, Graphs.nv(g))
    S = settype(n)
    return _pidbt(S, weights, g, min_k, deadline)
end

function _pidbt(::Type{PSet}, weights::AbstractVector{Int}, g::Graphs.AbstractGraph{V}, min_k::Int, deadline::Float64=Inf) where {V, PSet <: AbstractPackedSet}
    n = convert(Int, Graphs.nv(g))
    @assert all(ispositive, view(weights, 1:n))

    # Renumber the vertices in Cuthill-McKee order.
    # perm[new_index] = old_index
    #
    # The vertex order matters twice over. A PTD is only built if it is not
    # incoming, i.e. if its inlet avoids the smallest vertex outside it, so the
    # order decides which side of each separator gets built. And the sieve
    # tests vertices in increasing order. On the PACE 2017 instances,
    # Cuthill-McKee was about 1.5x faster (geometric mean) than the input
    # order. Reverse Cuthill-McKee was as fast on average, but it puts
    # high-degree vertices first on dense graphs, and was over 20x slower on
    # one instance.
    perm = cuthill_mckee(g, n)
    old_to_new = invperm(perm)

    # Build graph and permuted weights
    mg = Graph{PSet}(n)
    new_weights = Vector{Int}(undef, n)

    for new_v in 1:n
        old_v = perm[new_v]
        new_weights[new_v] = weights[old_v]

        s = PSet()
        for old_u in Graphs.neighbors(g, old_v)
            old_u == old_v && continue
            s = s ∪ old_to_new[old_u]
        end
        mg.neighbors[new_v] = s
    end

    result = treewidth(Weights{PSet}(new_weights), mg; min_k = max(0, min_k - 1), deadline)
    isnothing(result) && return nothing
    (tw, (pool, root)) = result

    # Map elimination ordering back to original indices
    ordering = _elimination_ordering(pool, root)
    return convert(Vector{V}, perm[ordering])
end

function _elimination_ordering(pool::PTDPool{PSet}, root::Int) where {PSet}
    ordering = Int[]
    _postorder!(ordering, pool, root, PSet())
    return ordering
end

function _postorder!(ordering::Vector{Int}, pool::PTDPool{PSet}, node::Int, parent_bag::PSet) where {PSet}
    B = bag(pool[node])
    for p in incident(pool, node)
        _postorder!(ordering, pool, target(pool, p), B)
    end
    for v in setdiff(B, parent_bag)
        push!(ordering, v)
    end
end

# ---------------------------------------------------------------------------
# Cuthill-McKee
# ---------------------------------------------------------------------------

# Return a permutation of 1:n (perm[new] = old). Each connected component is
# searched breadth-first from a pseudo-peripheral vertex, visiting neighbors in
# order of increasing degree.
function cuthill_mckee(g::Graphs.AbstractGraph, n::Int)
    adj = Vector{Int}[sort!(Int[u for u in Graphs.neighbors(g, v) if u != v]) for v in 1:n]
    deg = length.(adj)

    for list in adj
        sort!(list; by = u -> (deg[u], u))
    end

    order = Int[]
    seen = falses(n)
    mark = falses(n)

    for v in sortperm(deg)
        seen[v] && continue
        root = _peripheral_vertex!(mark, adj, v)
        _bfs!(order, seen, adj, root)
    end

    return order
end

# Breadth-first search from `root`, appending newly seen vertices to `order`.
function _bfs!(order::Vector{Int}, seen::BitVector, adj::Vector{Vector{Int}}, root::Int)
    i = length(order) + 1
    seen[root] = true
    push!(order, root)

    while i <= length(order)
        v = order[i]; i += 1

        for u in adj[v]
            seen[u] && continue
            seen[u] = true
            push!(order, u)
        end
    end

    return order
end

# Find a vertex far from `v` by repeated breadth-first search.
function _peripheral_vertex!(mark::BitVector, adj::Vector{Vector{Int}}, v::Int)
    order = Int[]

    for _ in 1:4
        fill!(mark, false); empty!(order)
        _bfs!(order, mark, adj, v)
        u = last(order)
        u == v && break
        v = u
    end

    return v
end
