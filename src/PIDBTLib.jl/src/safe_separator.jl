# Safe separator detection and recombination.
# All vertex indices are 1-based.

# ==================== Apply Safe Separator ====================

"""
    apply_externally_found_safe_separator!(weights, graph, S, min_k, already_calc_component)

Separate the graph at a separator found externally (e.g. during HasTreeWidth).
Returns `(subgraphs, already_calculated_C_index, min_k)`.
"""
function apply_externally_found_safe_separator!(
    weights::AbstractVector{Int}, graph::Graph{PSet}, S::PSet,
    min_k::Int, already_calc_component::PSet
) where {PSet}
    make_into_clique!(graph, S)

    # Separate, finding which subgraph contains the already-calculated component
    subgraphs = Graph{PSet}[]
    already_calc_idx = -1

    for (idx, (C, _)) in enumerate(components(graph, S))
        V = C ∪ S

        # Create subgraph with filtered neighborhoods
        subgraph = Graph(V)

        for v in V
            subgraph.neighbors[v] = neighbors(graph, v) ∩ V
        end
        make_into_clique!(subgraph, S)

        push!(subgraphs, subgraph)

        # Check if this component contains the already-calculated component
        if !isempty(already_calc_component) && C == already_calc_component
            already_calc_idx = idx
        end
    end

    min_k = max(min_k, wt(weights, S))

    return (subgraphs, already_calc_idx, min_k)
end

# ==================== RecombineTreeDecompositions ====================

"""
    recombine_tree_decompositions(pool, S, roots)

Recombine tree decompositions from separated subgraphs into a single PTD.
`roots` is a vector/view of PTD root indices.
"""
function recombine_tree_decompositions(pool::PTDPool{PSet}, S::PSet, roots::AbstractVector{Int}) where {PSet}
    first_root = roots[1]
    rest = @view roots[2:end]

    # Find node whose bag contains the separator
    root = 0
    stack = Int[first_root]

    while !isempty(stack)
        current = pop!(stack)

        if bag(pool[current]) ⊇ S
            root = current
            empty!(stack)
            break
        end

        for p in incident(pool, current)
            push!(stack, target(pool, p))
        end
    end

    @assert root != 0 "No node found with bag ⊇ S"

    # Reroot the other tree decompositions and attach them at root
    for y_root in rest
        y_root = reroot!(pool, y_root, S)
        add_edge!(pool, root, y_root)
    end

    return first_root
end

# ==================== Heuristic Safe Separator Test ====================

const MAX_MISSINGS = 100
const MAX_STEPS = 1000000

# ---------- Is Safe Separator (Heuristic) ----------

"""
    is_safe_separator_heuristic(graph, weights, S)

Test heuristically if a candidate separator is a safe separator.
If this method returns true, the separator is guaranteed safe.
False negatives are possible.
"""
# Reusable scratch space for `find_clique_minor`.
struct MinorWork{PSet <: AbstractPackedSet}
    edges::Vector{Tuple{Int, Int, Bool}}
    nodes::Vector{Tuple{PSet, PSet, Int, Int}}   # (D, N(D), label, minweight(D))
    layers::Vector{PSet}
    count::Vector{Int}
    rem::Vector{PSet}       # rem[v]: the x with {v, x} a remaining missing edge
    val::Vector{Int}        # (phase 3) cover counts of the edges without a node
    order::Vector{Int}      # (phase 3) the edges, sorted by `val`
    bucket::Vector{Int}
end

function MinorWork{PSet}() where {PSet <: AbstractPackedSet}
    return MinorWork{PSet}(Tuple{Int, Int, Bool}[], Tuple{PSet, PSet, Int, Int}[], PSet[], Int[],
        Vector{PSet}(undef, domain(PSet)), Int[], Int[], Int[])
end

function is_safe_separator_heuristic(weights::AbstractVector{Int}, graph::Graph{PSet}, S::PSet, work::MinorWork{PSet}=MinorWork{PSet}()) where {PSet}

    # count missing edges
    n = 0

    for v in S
        n += length(setdiff(S, neighbors(graph, v))) - 1
    end

    n > 2MAX_MISSINGS && return false

    isfirst = true

    for (C, N) in components(graph, S)
        isfirst && S ∪ C == vertices(graph) && return false
        isfirst = false
        find_clique_minor(work, weights, graph, N, setdiff(vertices(graph), N ∪ C)) || return false
    end

    return true
end

# ---------- Find Clique Minor ----------

# Try to determine if S is a labelled clique minor of the graph G[V].
# The following definition is taken from Bodlaender and Koster, "Safe
# Separators for Treewidth".
#
# Definition 9: A graph H is a *labelled minor* of G if H can be obtained
# from G by a sequence of zero or more of the following operations:
#
#   - deletion of edges
#   - deletion of vertices (and all adjacent edges)
#   - edge contraction that keeps the label of one endpoint: when contracting
#     the edge {v, w}, the resulting vertex will be labelled either v or w
#
# The function works by constructing a mapping D → S, where D ⊆ V, indicating
# which edges need to be contracted in order to make S into a clique. If such
# a mapping is found, the function returns `true`; otherwise, `false`.
function find_clique_minor(work::MinorWork{PSet}, weights::AbstractVector{Int}, graph::Graph{PSet}, S::PSet, V::PSet) where {PSet}
    # The vector `edges` contains all ordered nonadjacent vertices in S.
    # Each of these "missing edges" needs to be covered by the contraction
    # mapping D → S.
    edges = empty!(work.edges)

    # The vector `nodes` contains the contraction mapping D → S. Each element
    # is a triple (Dᵢ, Nᵢ, vᵢ), where vᵢ ∈ S is a vertex in the image of the
    # mapping, Dᵢ ⊆ D is its pre-image, and Nᵢ := N(Dᵢ) is the open neighborhood
    # of Dᵢ.
    nodes = empty!(work.nodes)

    # Find every ordered pair v < w of nonadjacent vertices in S and append
    # it to `edges`.
    R = S; E = PSet()

    while !isempty(R)
        v, R = popfirst_nonempty(R)

        for w in setdiff(R, neighbors(graph, v))
            push!(edges, (v, w, false))
            E = E ∪ v
            E = E ∪ w
        end
    end

    # Find every vertex v ∈ V with
    #
    #  - two or more neighbors, and
    #  - one or more neighbors which is an endpoint of a missing edge
    #
    # and append it to `nodes`.
    for w in E
        for v in neighbors(graph, w) ∩ V
            N = neighbors(graph, v)

            if length(N) > 1 && weights[v] >= weights[w]
                push!(nodes, (packedset(PSet, v), N, 0, weights[v]))
                V = setdiff(V, v)
            end
        end
    end

    # Halt early if `steps` exceeds `MAX_STEPS`.
    steps = 0

    # -- PHASE 1 ------------------------------------------------------
    #
    # A missing edge {w₁, w₂} is "zero-covered" if
    #
    #     {w₁, w₂} ⊈ N(D)
    #
    # for all vertex subsets D in `nodes`. For all such edges we search
    # for a pair of vertex subsets D₁ and D₂ in `nodes` such that
    #
    #   - w₁ ∈ N(D₁) and w₂ ∈ N(D₂),
    #   - w₂ ∉ N(D₁) and w₁ ∉ N(D₂), and
    #   - there exists a path P from D₁ to D₂ outside S and D.
    #
    # When such a pair is found, we remove D₁ and D₂ from `nodes`
    # and replace them with the union D₁ ∪ D₂ ∪ P. Note that D
    # also changes to D ∪ P. This new set now "covers" the missing
    # edge: if the set is contracted into either endpoint, the
    # missing edge will be introduced.
    i = find_zero_covered_edge(edges, nodes, weights)

    while !iszero(i)
        steps += 1
        steps < MAX_STEPS || return false

        edge = edges[i]
        pair = find_covering_pair(edge, nodes, V, weights, graph)

        if !isnothing(pair)
            V = merge_nodes!(pair..., edge, nodes, V, weights, graph, work.layers)
        else
            return false
        end

        i = find_zero_covered_edge(edges, nodes, weights)
    end

    # -- PHASE 2 ------------------------------------------------------
    #
    # At this point, every missing edge is "covered" by a vertex subset
    # in `nodes`. However, the number of these subsets may be very
    # large, impacting the performance of PHASE 3.
    i = find_least_covered_edge(edges, nodes, weights)

    while 2length(nodes) > length(S) && !iszero(i)
        steps += 1
        steps < MAX_STEPS || return false

        edge = edges[i]
        pair = find_covering_pair(edge, nodes, V, weights, graph)

        if !isnothing(pair)
            V = merge_nodes!(pair..., edge, nodes, V, weights, graph, work.layers)
        else
            w₁, w₂, _ = edge; edges[i] = (w₁, w₂, true)
            break
        end

        i = find_least_covered_edge(edges, nodes, weights)
    end

    # Remove ...
    i = 1

    while i ≤ length(nodes)
        _, N, _, _ = nodes[i]

        iscovered = false

        for (w₁, w₂, _) in edges
            iscovered && break
            iscovered = (w₁ ∈ N) & (w₂ ∈ N)
        end

        if iscovered
            i += 1
        elseif i < length(nodes)
            nodes[i] = pop!(nodes)
        else
            pop!(nodes)
        end
    end

    # -- PHASE 3 ------------------------------------------------------
    #
    # Assign labels greedily. Each round assigns the label v ∈ S to the
    # unassigned node (D, N) that maximizes (n, c), where c is the number of
    # missing edges that the assignment covers ({v, x} with x ∈ N), and n is
    # the least number of unassigned nodes that potentially cover a remaining
    # edge that it does not cover (typemax if there is none), not counting
    # (D, N) itself. Ties go to the smallest v, then to the first node.
    #
    # For a node, n depends on v only through the edges the assignment covers,
    # all of which are incident to v. So the edges are sorted once per node by
    # their cover count without the node, and n is the count of the first edge
    # in that order that the assignment does not cover. This takes O(c + 1)
    # steps per label, rather than one step per edge.
    count = work.count; rem = work.rem; val = work.val; order = work.order; bucket = work.bucket

    while !isempty(edges)
        vmax = 0
        imax = 0
        nmax = 0
        cmax = 0

        # No remaining edge is covered by an assigned node (covered edges are
        # removed below), so assigning one more node changes the cover counts
        # only through that node. Count once, then update.
        potential_cover_counts!(count, edges, nodes, weights)

        @inbounds for v in S
            rem[v] = PSet()
        end

        @inbounds for (w₁, w₂, _) in edges
            rem[w₁] = rem[w₁] ∪ w₂
            rem[w₂] = rem[w₂] ∪ w₁
        end

        m = length(edges); resize!(val, m); resize!(order, m)

        @inbounds for (i, (_, N, w, mw)) in enumerate(nodes)
            iszero(w) || continue
            L = S ∩ N
            isempty(L) && continue

            # the cover counts of the edges without this node, and the
            # edges sorted by them (counting sort; the counts are at most
            # the number of nodes)
            top = 0

            for (e, (w₁, w₂, _)) in enumerate(edges)
                x = val[e] = count[e] - ((w₁ ∈ N) & (w₂ ∈ N) & (mw >= min(weights[w₁], weights[w₂])))
                top = max(top, x)
            end

            resize!(bucket, top + 2); fill!(bucket, 0); bucket[1] = 1

            for e in 1:m
                bucket[val[e] + 2] += 1
            end

            for x in 2:top + 2
                bucket[x] += bucket[x - 1]
            end

            for e in 1:m
                x = val[e] + 1; order[bucket[x]] = e; bucket[x] += 1
            end

            for v in L
                mw >= weights[v] || continue
                steps += 1
                steps < MAX_STEPS || return false

                c = length(rem[v] ∩ N)
                n = typemax(Int)

                for k in 1:m
                    e = order[k]; w₁, w₂, _ = edges[e]

                    if !(((v == w₁) & (w₂ ∈ N)) | ((v == w₂) & (w₁ ∈ N)))
                        n = val[e]
                        break
                    end
                end

                # (the nodes are visited in order, so on a tie, (v, i)
                # comes first iff v < vmax)
                if (n, c) > (nmax, cmax) || ((n, c) == (nmax, cmax) && v < vmax)
                    vmax = v
                    imax = i
                    nmax = n
                    cmax = c
                end
            end
        end

        iszero(nmax) && return false
        D, N, _, mw = nodes[imax]; nodes[imax] = (D, N, vmax, mw)

        # Remove...
        i = 1

        while i ≤ length(edges)
            w₁, w₂, _ = edges[i]

            iscovered = false

            for (_, N, v, _) in nodes
                iscovered && break
                iscovered = ((v == w₁) & (w₂ in N)) | ((v == w₂) & (w₁ in N))
            end

            if !iscovered
                i += 1
            elseif i < length(edges)
                edges[i] = pop!(edges)
            else
                pop!(edges)
            end
        end
    end

    return true
end

# ---------- Clique Minor Helper Functions ----------

# Get the index of a missing edge {w, x} in `edges` such that,
# for all unassigned vertex subsets D in `nodes`,
#
#    {w, x} ⊈ N(D)  or  wmineight(D) < min(weight(w), weight(x))
#
# If no such edge exists, return 0.
function find_zero_covered_edge(edges::Vector{Tuple{Int,Int,Bool}}, nodes::Vector{Tuple{PSet,PSet,Int,Int}}, weights::AbstractVector{Int}) where {PSet}
    for (i, (w, x, _)) in enumerate(edges)
        iscovered = false
        wmin = min(weights[w], weights[x])

        for (D, N, v, mw) in nodes
            iscovered && break
            iscovered = iszero(v) & (w in N) & (x in N) & (mw >= wmin)
        end

        iscovered || return i
    end

    return 0
end

"""
    find_least_covered_edge(edges, nodes, weights)

Find the augmentable missing edge potentially covered by the fewest right nodes.
Returns the index into `edges`, or `0` if no augmentable edge exists.
"""
function find_least_covered_edge(edges::Vector{Tuple{Int,Int,Bool}}, nodes::Vector{Tuple{PSet,PSet,Int,Int}}, weights::AbstractVector{Int}) where {PSet}
    nmin = imin = 0

    for (i, (w, x, flag)) in enumerate(edges)
        flag && continue

        n = 0
        wmin = min(weights[w], weights[x])

        for (D, N, v, mw) in nodes
            if iszero(v) & (w in N) & (x in N) & (mw >= wmin)
                n += 1
            end
        end

        if iszero(imin) || n < nmin
            nmin = n
            imin = i
        end
    end

    return imin
end

# Given a missing edge {w₁, w₂}, find a pair (V₁, V₂) of
# vertex subsets in `nodes` such that
#
#   - w₁ ∈ N(V₁) and w₂ ∉ N(V₁)
#   - w₂ ∈ N(V₂) and w₁ ∉ N(V₂)
#   - wmineight(V₁) >= min(weight(w₁), weight(w₂))
#   - wmineight(V₂) >= min(weight(w₁), weight(w₂))
#
# and there is a path from V₁ to V₂ in V using only vertices with
# weight >= min(weight(w₁), weight(w₂)).
function find_covering_pair((w₁, w₂, _)::Tuple{Int, Int, Bool}, nodes::Vector{Tuple{PSet, PSet, Int, Int}}, V::PSet, weights::AbstractVector{Int}, graph::Graph{PSet}) where {PSet}
    wmin = min(weights[w₁], weights[w₂])
    V = atleast(weights, V, wmin)

    for (i₁, (D₁, N₁, _, mw₁)) in enumerate(nodes)
        (w₁ ∈ N₁ && w₂ ∉ N₁ && mw₁ >= wmin) || continue

        for (i₂, (D₂, N₂, _, mw₂)) in enumerate(nodes)
            (w₁ ∉ N₂ && w₂ ∈ N₂ && mw₂ >= wmin) || continue

            U = D₁ # visited
            M = N₁ # frontier

            while isdisjoint(M, D₂) && !isdisjoint(M, V)
                U = U ∪ (M ∩ V)
                M = setdiff(neighbors(graph, M ∩ V), U)
            end

            !isdisjoint(M, D₂) && return (i₁, i₂)
        end
    end

    return
end

function merge_nodes!(i₁::Int, i₂::Int, (w₁, w₂, _)::Tuple{Int,Int,Bool}, nodes::Vector{Tuple{PSet,PSet,Int,Int}}, V::PSet, weights::AbstractVector{Int}, graph::Graph{PSet}, layers::Vector{PSet}) where {PSet}
    D₁, N₁, _, _ = nodes[i₁]
    D₂, N₂, _, _ = nodes[i₂]

    # Restrict path search to vertices with weight >= min(w₁, w₂)
    wmin = min(weights[w₁], weights[w₂])
    U, M = merge_nodes(graph, atleast(weights, V, wmin), D₁, D₂, N₁, N₂, layers)

    nodes[i₁] = (U, M, 0, minweight(weights, U))

    if i₂ != length(nodes)
        nodes[i₂] = nodes[end]
    end

    pop!(nodes)

    return setdiff(V, U)
end

# We are given disjoint vertex sets V, D₁, and D₂, as
# well as the neighborhoods
#
#   N₁ := N(D₁)
#   N₂ := N(D₂).
#
# This function finds a path P from D₁ to D₂ through V.
# It returns the union
#
#    U := D₁ ∪ D₂ ∪ P
#
# as well as its neighborhood N(U).
function merge_nodes(graph::Graph{PSet}, V::PSet, D₁::PSet, D₂::PSet, N₁::PSet, N₂::PSet, layers::Vector{PSet}=PSet[]) where {PSet}
    empty!(layers)

    U = D₁ # visited
    M = N₁ # frontier

    while isdisjoint(M, D₂)
        push!(layers, M ∩ V)
        U = U ∪ (M ∩ V)
        M = setdiff(neighbors(graph, M ∩ V), U)
    end

    U = D₁ ∪ D₂
    M = N₁ ∪ N₂
    B = N₂

    for L in Iterators.reverse(layers)
        v = first(L ∩ B)
        B = neighbors(graph, v)

        U = U ∪ v
        M = M ∪ B
    end

    return (U, setdiff(M, U))
end

# For every missing edge e = {w₁, w₂}, count[e] is the number of unassigned
# nodes (D, N) that potentially cover e: {w₁, w₂} ⊆ N and minweight(D) ≥
# min(weight(w₁), weight(w₂)).
function potential_cover_counts!(count::Vector{Int}, edges::Vector{Tuple{Int,Int,Bool}}, nodes::Vector{Tuple{PSet,PSet,Int,Int}}, weights::AbstractVector{Int}) where {PSet}
    resize!(count, length(edges))

    @inbounds for (e, (w₁, w₂, _)) in enumerate(edges)
        n = 0
        wmin = min(weights[w₁], weights[w₂])

        for (_, N, v, mw) in nodes
            n += iszero(v) & (w₁ ∈ N) & (w₂ ∈ N) & (mw >= wmin)
        end

        count[e] = n
    end

    return count
end
