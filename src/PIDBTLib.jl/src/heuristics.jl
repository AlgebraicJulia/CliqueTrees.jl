# Heuristics for finding small tree decompositions.
# Based on Tamaki_Tree_Decomp/Heuristics.cs, using 1-based vertex indexing
# and AbstractPackedSet bitsets.
#
# Only the min-degree elimination ordering is retained (min_fill and min_defect
# were only used by the initial safe separator preprocessing, now removed).

# ==================== HeuristicBagsAndNeighbors ====================

"""
    heuristic_bags_and_neighbors(graph, weights)

Compute an elimination ordering using the min-degree heuristic.

Operates on the provided (mutable) graph, eliminating vertices one at a time.
Each step yields a `(bag, parent)` pair where:
- `bag`    = N[v] (closed neighborhood of v) at the time of elimination
- `parent` = N(v) (open neighborhood of v) at the time of elimination

Returns `(pairs, remaining)` where `pairs` is a `Vector{Tuple{PSet, PSet}}` of
(bag, parent) pairs and `remaining` is the set of vertices still present
(a clique) when the heuristic stops early.
"""
function heuristic_bags_and_neighbors(weights::Vector{Int}, graph::Graph{PSet}) where {PSet <: AbstractPackedSet}
    total_weight = wt(weights, vertices(graph))
    mindegree = total_weight
    result = Tuple{PSet, PSet}[]

    closed_nbrs = Vector{PSet}(undef, domain(PSet))

    for v in vertices(graph)
        closed_nbrs[v] = neighbors(graph, v) ∪ v
    end

    # Bucket Queue #############
    head = zeros(Int, total_weight)
    prev = Vector{Int}(undef, domain(PSet))
    next = Vector{Int}(undef, domain(PSet))

    function set(degree::Int)
        return DoublyLinkedList(view(head, degree), prev, next)
    end

    for v in vertices(graph)
        degree = wt(weights, closed_nbrs[v])
        pushfirst!(set(degree), v)
        mindegree = min(mindegree, degree)
    end
    ############################

    R = vertices(graph)
    eliminated_weight = 0

    while mindegree + eliminated_weight < total_weight
        v = first(set(mindegree))

        W = closed_nbrs[v]; V = PSet()

        for w in W
            if closed_nbrs[w] == W
                U = setdiff(W, V)
                V = V ∪ w
                eliminated_weight += weights[w]
                push!(result, (U, setdiff(U, w)))
                delete!(set(mindegree), w)
            end
        end

        R = setdiff(R, V)

        for w in setdiff(W, V)
            degree = wt(weights, closed_nbrs[w])
            delete!(set(degree), w)
            closed_nbrs[w] = setdiff(closed_nbrs[w] ∪ W, V)

            degree = wt(weights, closed_nbrs[w])
            pushfirst!(set(degree), w)
            mindegree = min(mindegree, degree)
        end

        while isempty(set(mindegree))
            mindegree += 1
        end
    end

    return (result, R)
end
