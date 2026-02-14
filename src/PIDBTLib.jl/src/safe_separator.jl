# Safe separator detection and recombination.
# All vertex indices are 1-based.

# ==================== Apply Safe Separator ====================

"""
    apply_externally_found_safe_separator!(weights, graph, S, min_k, already_calc_component)

Separate the graph at a separator found externally (e.g. during HasTreeWidth).
Returns `(subgraphs, already_calculated_C_index, min_k)`.
"""
function apply_externally_found_safe_separator!(
    weights::Vector{Int}, graph::Graph{PSet}, S::PSet,
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
        if !isempty(already_calc_component) && !isdisjoint(C, already_calc_component)
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

# ==================== Articulation Points ====================

"""
    articulation_points(graph, V)

List all articulation points of the graph restricted to `V`.
Uses an iterative version of Tarjan's algorithm, faithfully transpiled from C#.

This method can also be used to test for separators of size n by passing a set
of n-1 vertices as ignored (removed from V), and combining the
result with the ignored vertices.
"""
function articulation_points(graph::Graph{PSet}, V::PSet=vertices(graph)) where {PSet}
    A = PSet()

    stack = Tuple{Int, Int, Int}[]
    count = Vector{Int}(undef, domain(V))
    reach = Vector{Int}(undef, domain(V))
    queue = Vector{PSet}(undef, domain(V))

    for v in V
        count[v] = typemax(Int)
    end

    u = first(V); count[u] = n = 0

    for v in neighbors(graph, u) ∩ V
        if count[v] == typemax(Int)
            push!(stack, (v, 1, u))        

            while !isempty(stack)
                (x, timer, w) = last(stack)

                if count[x] == typemax(Int)
                    count[x] = timer
                    reach[x] = timer
                    queue[x] = setdiff(neighbors(graph, x) ∩ V, w)
                elseif !isempty(queue[x])
                    y, queue[x] = popfirst_nonempty(queue[x])

                    if count[y] < typemax(Int)
                        reach[x] = min(reach[x], count[y])
                    else
                        push!(stack, (y, timer + 1, x))
                    end
                else
                    if x != v
                        reach[w] = min(reach[w], reach[x])

                        if reach[x] ≥ count[w]
                            A = A ∪ w
                        end
                    end

                    pop!(stack)
                end
            end

            n += 1
        end
    end

    if n > 1
        A = A ∪ u
    end

    return A
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
function is_safe_separator_heuristic(weights::Vector{Int}, graph::Graph{PSet}, S::PSet) where {PSet}

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
        find_clique_minor(weights, graph, N, setdiff(vertices(graph), N ∪ C)) || return false
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
function find_clique_minor(weights::Vector{Int}, graph::Graph{PSet}, S::PSet, V::PSet) where {PSet}
    # The vector `edges` contains all ordered nonadjacent vertices in S.
    # Each of these "missing edges" needs to be covered by the contraction
    # mapping D → S.
    edges = Tuple{Int, Int, Bool}[]

    # The vector `nodes` contains the contraction mapping D → S. Each element
    # is a triple (Dᵢ, Nᵢ, vᵢ), where vᵢ ∈ S is a vertex in the image of the
    # mapping, Dᵢ ⊆ D is its pre-image, and Nᵢ := N(Dᵢ) is the open neighborhood
    # of Dᵢ.
    nodes = Tuple{PSet, PSet, Int}[]

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
                push!(nodes, (packedset(PSet, v), N, 0))
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
            V = merge_nodes!(pair..., edge, nodes, V, weights, graph)
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
            V = merge_nodes!(pair..., edge, nodes, V, weights, graph)
        else
            w₁, w₂, _ = edge; edges[i] = (w₁, w₂, true)
            break
        end

        i = find_least_covered_edge(edges, nodes, weights)
    end

    # Remove ...
    i = 1

    while i ≤ length(nodes)
        _, N, _ = nodes[i]

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
    while !isempty(edges)
        vmax = 0
        imax = 0
        nmax = 0
        cmax = 0

        for v in S
            for (i, (D, N, w)) in enumerate(nodes)
                # Can only assign if: unassigned, adjacent, and weight constraint satisfied
                if iszero(w) && v ∈ N && weights[first(D)] >= weights[v]
                    steps += 1
                    steps < MAX_STEPS || return false

                    nodes[i] = (D, N, v)
                    n, c = min_cover(edges, nodes, weights)
                    nodes[i] = (D, N, 0)

                    if (n, c) > (nmax, cmax)
                        vmax = v
                        imax = i
                        nmax = n
                        cmax = c
                    end
                end
            end
        end

        iszero(nmax) && return false
        D, N, _ = nodes[imax]; nodes[imax] = (D, N, vmax)

        # Remove...
        i = 1

        while i ≤ length(edges)
            w₁, w₂, _ = edges[i]

            iscovered = false

            for (_, N, v) in nodes
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
function find_zero_covered_edge(edges::Vector{Tuple{Int,Int,Bool}}, nodes::Vector{Tuple{PSet,PSet,Int}}, weights::Vector{Int}) where {PSet}
    for (i, (w, x, _)) in enumerate(edges)
        iscovered = false
        wmin = min(weights[w], weights[x])

        for (D, N, v) in nodes
            iscovered && break
            iscovered = iszero(v) & (w in N) & (x in N) & (weights[first(D)] >= wmin)
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
function find_least_covered_edge(edges::Vector{Tuple{Int,Int,Bool}}, nodes::Vector{Tuple{PSet,PSet,Int}}, weights::Vector{Int}) where {PSet}
    nmin = imin = 0

    for (i, (w, x, flag)) in enumerate(edges)
        flag && continue

        n = 0
        wmin = min(weights[w], weights[x])

        for (D, N, v) in nodes
            if iszero(v) & (w in N) & (x in N) & (weights[first(D)] >= wmin)
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
function find_covering_pair((w₁, w₂, _)::Tuple{Int, Int, Bool}, nodes::Vector{Tuple{PSet, PSet, Int}}, V::PSet, weights::Vector{Int}, graph::Graph{PSet}) where {PSet}
    wmin = min(weights[w₁], weights[w₂])
    k = searchsortedfirst(weights, wmin)
    V = V ∩ upset(PSet, k)

    for (i₁, (D₁, N₁, _)) in enumerate(nodes)
        (w₁ ∈ N₁ && w₂ ∉ N₁ && weights[first(D₁)] >= wmin) || continue

        for (i₂, (D₂, N₂, _)) in enumerate(nodes)
            (w₁ ∉ N₂ && w₂ ∈ N₂ && weights[first(D₂)] >= wmin) || continue

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

function merge_nodes!(i₁::Int, i₂::Int, (w₁, w₂, _)::Tuple{Int,Int,Bool}, nodes::Vector{Tuple{PSet,PSet,Int}}, V::PSet, weights::Vector{Int}, graph::Graph{PSet}) where {PSet}
    D₁, N₁, _ = nodes[i₁]
    D₂, N₂, _ = nodes[i₂]

    # Restrict path search to vertices with weight >= min(w₁, w₂)
    wmin = min(weights[w₁], weights[w₂])
    k = searchsortedfirst(weights, wmin)
    U, M = merge_nodes(graph, V ∩ upset(PSet, k), D₁, D₂, N₁, N₂)

    nodes[i₁] = (U, M, 0)

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
function merge_nodes(graph::Graph{PSet}, V::PSet, D₁::PSet, D₂::PSet, N₁::PSet, N₂::PSet) where {PSet}
    layers = PSet[]

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

"""
    min_cover(edges, nodes, weights)

Determine the minimum number of right nodes that potentially cover
any non-finally-covered missing edge.
"""
function min_cover(edges::Vector{Tuple{Int,Int,Bool}}, nodes::Vector{Tuple{PSet,PSet,Int}}, weights::Vector{Int}) where {PSet}
    nmin = typemax(Int); c = 0

    for (w₁, w₂, _) in edges
        n = 0
        wmin = min(weights[w₁], weights[w₂])

        for (D, N, v) in nodes
            b₁ = w₁ ∈ N
            b₂ = w₂ ∈ N

            if iszero(v) & b₁ & b₂ & (weights[first(D)] >= wmin)
                n += 1
            elseif ((v == w₁) & b₂) | ((v == w₂) & b₁)
                n = typemax(Int)
                c += 1
                break
            end
        end

        nmin = min(nmin, n)
    end

    return (nmin, c)
end
