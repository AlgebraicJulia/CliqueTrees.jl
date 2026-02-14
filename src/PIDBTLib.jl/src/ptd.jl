# PTD (Partial Tree Decomposition) — pool-based DAG implementation.
# A PTD is represented as (pool, root_index) where pool is a DAG{PTD{PSet}}.
# All vertex indices are 1-based.

# ---------------------------------------------------------------------------
const TEST_IF_ADDING_ONE_VERTEX_TO_BAG_FORMS_PMC = true
const NEIGHBORS_FIRST = true

# ---------------------------------------------------------------------------
# PTD: value type for node data (stored in DAG pool)
# ---------------------------------------------------------------------------

struct PTD{PSet <: AbstractPackedSet}
    bag::PSet
    vertices::PSet
    outlet::PSet
end

@inline bag(data::PTD) = data.bag
@inline vertices(data::PTD) = data.vertices
@inline outlet(data::PTD) = data.outlet
@inline inlet(data::PTD) = setdiff(data.vertices, data.outlet)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

const PTDPool{PSet} = DAG{PTD{PSet}}

# ---------------------------------------------------------------------------
# Constructors (return root index into pool)
# ---------------------------------------------------------------------------

# Bag-only constructor (for import/heuristic)
function make_ptd(pool::PTDPool{PSet}, B::PSet) where {PSet <: AbstractPackedSet}
    return add_vertex!(pool, PTD{PSet}(B, PSet(), PSet()))
end

# Bag+outlet constructor
function make_ptd(pool::PTDPool{PSet}, B::PSet, outlet::PSet) where {PSet <: AbstractPackedSet}
    return add_vertex!(pool, PTD{PSet}(B, B, outlet))
end

# Copy: new node with same data and same children edges
function copy_ptd(pool::PTDPool{PSet}, root::Int) where {PSet}
    data = pool[root]
    new_root = add_vertex!(pool, data)
    for p in incident(pool, root)
        add_edge!(pool, new_root, target(pool, p))
    end
    return new_root
end

# ---------------------------------------------------------------------------
# Combination Rules
# ---------------------------------------------------------------------------

# C# CreatePTDURFromPTD: new bag = old outlet, old PTD becomes only child.
function create_ptdur_from_ptd(pool::PTDPool{PSet}, tau_root::Int) where {PSet}
    tau_data = pool[tau_root]
    data = PTD{PSet}(outlet(tau_data), vertices(tau_data), outlet(tau_data))
    root = add_vertex!(pool, data)
    add_edge!(pool, root, tau_root)
    return root
end

# C# AddPTDToPTDUR_CheckBagSize_CheckPossiblyUsable_CheckCliquish
# Returns (success::Bool, result_root::Int) where result_root=0 on failure.
function add_ptd_to_ptdur_check(work::Vector{PSet}, pool::PTDPool{PSet}, tp_root::Int, tau_root::Int,
                                 weights::Vector{Int}, graph::CachedGraph{PSet}, k::Int, mutable_graph::Graph{PSet}) where {PSet}
    tp_data = pool[tp_root]
    tau_data = pool[tau_root]

    # Check bag size (weighted)
    future_bag_size = wt(weights, bag(tp_data) ∪ outlet(tau_data))
    if future_bag_size > k + 1
        return (false, 0)
    end

    B = bag(tp_data) ∪ outlet(tau_data)

    # Check possibly usable: only need to check new child against existing ones
    if !is_possibly_usable(pool, tp_root, tau_root, graph)
        return (false, 0)
    end

    # If bag is at max size and not PMC, reject
    if future_bag_size == k + 1 && !is_pmc!(work, graph, B)
        return (false, 0)
    end

    # Check cliquish
    if !is_csh!(work, graph, B)
        return (false, 0)
    end

    # One-vertex-to-PMC test
    if TEST_IF_ADDING_ONE_VERTEX_TO_BAG_FORMS_PMC && future_bag_size == k
        if !is_pmc!(work, graph, B)
            useless = true

            for (component, nbrs) in components(graph, B)
                if nbrs == B && isdisjoint(component, vertices(tp_data)) && isdisjoint(component, vertices(tau_data))
                    if NEIGHBORS_FIRST
                        for v in B
                            v_nbrs = neighbors(graph, v) ∩ component

                            if length(v_nbrs) == 1
                                candidate = first(v_nbrs)
                                test_bag = B ∪ candidate

                                if is_csh!(work, graph, test_bag)
                                    useless = false
                                    break
                                end
                            end
                        end
                        if !useless
                            break
                        end
                    end

                    for ap in articulation_points(mutable_graph, component)
                        test_bag = B ∪ ap

                        if is_pmc!(work, graph, test_bag)
                            useless = false
                            break
                        end
                    end
                    if !useless
                        break
                    end

                    if !NEIGHBORS_FIRST
                        for v in B
                            v_nbrs = neighbors(graph, v) ∩ component

                            if length(v_nbrs) == 1
                                candidate = first(v_nbrs)
                                test_bag = B ∪ candidate

                                if is_csh!(work, graph, test_bag)
                                    useless = false
                                    break
                                end
                            end
                        end
                        if !useless
                            break
                        end
                    end
                end
            end

            if useless
                return (false, 0)
            end
        end
    end

    # Build result
    V = vertices(tp_data) ∪ vertices(tau_data)
    S = outlet(graph, B, V)

    new_root = add_vertex!(pool, PTD{PSet}(B, V, S))

    # Copy existing children from tp
    for p in incident(pool, tp_root)
        add_edge!(pool, new_root, target(pool, p))
    end
    # Add tau as new child
    add_edge!(pool, new_root, tau_root)

    return (true, new_root)
end

# C# IsPossiblyUsable: Check only new child against existing children (O(n)).
function is_possibly_usable(pool::PTDPool{PSet}, parent_root::Int, tau_root::Int, graph::CachedGraph{PSet}) where {PSet}
    tau_data = pool[tau_root]

    for p in incident(pool, parent_root)
        child_data = pool[target(pool, p)]

        if !isdisjoint(inlet(child_data), inlet(tau_data))
            return false
        end
        vi = vertices(child_data) ∩ vertices(tau_data)

        if !(outlet(child_data) ⊇ vi) || !(outlet(tau_data) ⊇ vi)
            return false
        end
    end
    return true
end


# ---------------------------------------------------------------------------
# Extend-to-PMC rules
# ---------------------------------------------------------------------------

# C# ExtendToPMC_Rule2
function extend_to_pmc_rule2(pool::PTDPool{PSet}, tw_root::Int, v_neighbors::PSet, graph::CachedGraph{PSet}) where {PSet}
    tw_data = pool[tw_root]
    B = v_neighbors
    V = vertices(tw_data) ∪ v_neighbors
    S = outlet(graph, B, V)
    new_root = add_vertex!(pool, PTD{PSet}(B, V, S))

    for p in incident(pool, tw_root)
        add_edge!(pool, new_root, target(pool, p))
    end
    return new_root
end

# C# ExtendToPMC_Rule3
function extend_to_pmc_rule3(pool::PTDPool{PSet}, tw_root::Int, new_root_bag::PSet, graph::CachedGraph{PSet}) where {PSet}
    tw_data = pool[tw_root]
    @assert new_root_bag ⊇ bag(tw_data)
    B = new_root_bag
    V = vertices(tw_data) ∪ new_root_bag
    S = outlet(graph, B, V)
    new_root = add_vertex!(pool, PTD{PSet}(B, V, S))

    for p in incident(pool, tw_root)
        add_edge!(pool, new_root, target(pool, p))
    end
    return new_root
end

# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

# C# IsIncoming: inlet.First() < complement(vertices).First()
function is_incoming(pool::PTDPool{PSet}, root::Int, graph::CachedGraph{PSet}) where {PSet}
    data = pool[root]
    rest = setdiff(vertices(graph), vertices(data))

    if isempty(rest)
        return true  # all vertices covered
    end
    return first(inlet(data)) < first(rest)
end

# C# IsNormalized
function is_normalized(pool::PTDPool{PSet}, root::Int) where {PSet}
    data = pool[root]

    for p in incident(pool, root)
        child_data = pool[target(pool, p)]

        if outlet(data) ⊇ outlet(child_data)
            return false
        end
    end
    return true
end

# ---------------------------------------------------------------------------
# Tree restructuring
# ---------------------------------------------------------------------------

# C# Reroot: restructure the tree so a node whose bag ⊇ root_set becomes root.
# Returns the new root index.
function reroot!(pool::PTDPool{PSet}, root::Int, S::PSet) where {PSet}
    path = Vector{Int}(undef, domain(PSet))
    stack = Tuple{Int, Int}[]

    n = 0
    node = root
    data = pool[node]

    while S ⊈ bag(data)
        for p in incident(pool, node)
            push!(stack, (n + 1, p))
        end

        n, p = pop!(stack)
        node = target(pool, p)
        data = pool[node]
        path[n] = p
    end

    for p in view(path, oneto(n))
        node = target(pool, p)
        rem_edge!(pool, root, p)
        add_edge!(pool, node, root)
        root = node
    end

    return node
end

