function pidbt(weights::AbstractVector{Int}, g::Graphs.AbstractGraph, min_k::Int=0)
    n = convert(Int, Graphs.nv(g))
    S = settype(n)
    return _pidbt(S, weights, g, min_k)
end

function _pidbt(::Type{PSet}, weights::AbstractVector{Int}, g::Graphs.AbstractGraph{V}, min_k::Int) where {V, PSet <: AbstractPackedSet}
    @assert all(weights .> 0)
    n = convert(Int, Graphs.nv(g))

    # Sort vertices by weight (increasing order)
    # perm[new_index] = old_index
    perm = sortperm(weights)
    old_to_new = invperm(perm)

    # Build graph and sorted weights
    mg = Graph{PSet}(n)
    sorted_weights = Vector{Int}(undef, domain(PSet))

    for new_v in 1:n
        old_v = perm[new_v]
        sorted_weights[new_v] = weights[old_v]

        s = PSet()
        for old_u in Graphs.neighbors(g, old_v)
            old_u == old_v && continue
            s = s ∪ old_to_new[old_u]
        end
        mg.neighbors[new_v] = s
    end

    (tw, (pool, root)) = treewidth(sorted_weights, mg; min_k = max(0, min_k - 1))

    # Map elimination ordering back to original indices
    sorted_ordering = _elimination_ordering(pool, root)
    return convert(Vector{V}, perm[sorted_ordering])
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
