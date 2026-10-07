# PTD (Partial Tree Decomposition) — pool-based DAG implementation.
# A PTD is represented as (pool, root_index) where pool is a DAG{PTD{PSet}}.
# All vertex indices are 1-based.

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
                                 weights::AbstractVector{Int}, graph::CachedGraph{PSet}, k::Int) where {PSet}
    tp_data = pool[tp_root]
    tau_data = pool[tau_root]

    # Check bag size (weighted)
    future_bag_size = wt(weights, bag(tp_data) ∪ outlet(tau_data))
    if future_bag_size > k + 1
        return (false, 0)
    end

    B = bag(tp_data) ∪ outlet(tau_data)

    # Check that τ is possibly usable together with every child of tp:
    #
    #   - no vertex of tp lies in the inlet of τ, and
    #   - no inlet vertex of a child of tp lies in V(τ).
    #
    # The union of the children's inlets is V(tp) - bag(tp), since the children
    # of a PTDUR are pairwise possibly usable (no child's outlet meets another
    # child's inlet). The first condition is also the sieve query condition,
    # but checking it here keeps this function correct on its own.
    if !isdisjoint(vertices(tp_data), inlet(tau_data)) ||
       !isdisjoint(setdiff(vertices(tp_data), bag(tp_data)), vertices(tau_data))
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


# Copy the tree rooted at `root` in `src` into `dst`. Returns the new root.
function copy_tree!(dst::PTDPool{PSet}, src::PTDPool{PSet}, root::Int) where {PSet}
    new_root = add_vertex!(dst, src[root])
    stack = Tuple{Int, Int}[(root, new_root)]

    while !isempty(stack)
        u, v = pop!(stack)

        for p in incident(src, u)
            x = target(src, p)
            y = add_vertex!(dst, src[x])
            add_edge!(dst, v, y)
            push!(stack, (x, y))
        end
    end

    return new_root
end
