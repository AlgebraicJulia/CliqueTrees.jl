# Exact (weighted) treewidth by positive-instance driven dynamic programming.
#
# This is the algorithm of Tamaki, as reformulated by Althaus, Schnurbusch,
# Wüschner, and Ziegler ("On Tamaki's Algorithm to Compute Treewidths",
# SEA 2021). It started out as a port of their C# implementation
# (Tamaki_Tree_Decomp/Treewidth.cs, "blocksieve" path) and deviates from it
# in the following ways.
#
#   - When the outlet of a new PTD is found to be a safe separator, the graph
#     is split at *that* outlet. The C# splits at the outlet of the PTD
#     currently being processed, which was never tested.
#   - PTDs that are delayed by the ">2 components" rule (Section 4.6) are
#     released when P runs dry, instead of being dropped. Dropping them is
#     unsound: two delayed PTDs can each wait on a component whose PTD can
#     only be built from the other.
#   - Equivalent PTDURs (same inlet) are not pruned (Section 4.2): only exact
#     duplicates are dropped. Pruning by bag size, or even by bag inclusion,
#     is unsound for weighted graphs.
#   - Heuristic completion (min-degree) is tried only when a PTD's inlet is
#     heavier than any tried before, instead of on every other PTD.
#   - The "adding one vertex to the bag forms a PMC" test (Section 4.5) and
#     the cliquish/PMC cache are gone: neither paid for itself.
#   - Graph reduction is skipped (the input is assumed to be pre-reduced).
#   - The caller (`pidbt`) renumbers the vertices in Cuthill-McKee order,
#     which the algorithm is sensitive to (see pidbt.jl). Vertices need not be
#     sorted by weight.
#
# All vertex indices are 1-based.

const COMPLETE_HEURISTICALLY = true
const TEST_OUTLET_IS_CLIQUE_MINOR = true
const MORE_THAN_2_COMPONENTS_OPTIMIZATION = true

@enum State Continue Divide Halt

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

"""
    treewidth(weights, graph::Graph{PSet}; min_k::Int=0) where {PSet}

Compute the exact treewidth of `graph` and return `(treewidth, ptd)` where
`ptd` is a `(pool, root_index)` tuple.

`min_k` is a lower bound on the treewidth. The DP search begins at this value,
so supplying a tight lower bound (e.g. from MMD+) avoids wasted iterations.

The graph must be connected. (The search builds only PTDs that are not
incoming. If the smallest vertex were isolated, the only PTD containing it
would be incoming, so no tree decomposition of the whole graph would ever be
found.) Safe separators found during the search split a connected graph into
connected pieces, so this needs to be checked only here.
"""
function treewidth(weights::AbstractVector{Int}, graph::Graph{PSet}; min_k::Int=0) where {PSet}
    if count(_ -> true, components(graph, PSet())) > 1
        throw(ArgumentError("PIDBT requires a connected graph. Wrap it with `ConnectedComponents`."))
    end

    weights isa Weights{PSet} || (weights = Weights{PSet}(collect(weights)))
    if nv(graph) == 0
        pool = PTDPool{PSet}()
        root = make_ptd(pool, PSet())
        return (0, (pool, root))
    elseif nv(graph) == 1
        pool = PTDPool{PSet}()
        v = first(vertices(graph))
        only_bag = packedset(PSet, v)
        root = add_vertex!(pool, PTD{PSet}(only_bag, only_bag, PSet()))
        return (weights[v] - 1, (pool, root))
    end

    return treewidth_computation(weights, graph, min_k)
end

# ---------------------------------------------------------------------------
# Achievable subset sums of vertex weights
# ---------------------------------------------------------------------------

function achievable_sums(weights::AbstractVector{Int}, graph::Graph{PSet}) where {PSet}
    W = wt(weights, vertices(graph))
    achievable = falses(W + 1)
    achievable[1] = true  # 0 is achievable (1-indexed: index 1 = sum 0)

    for v in vertices(graph)
        w = weights[v]

        for j in W:-1:w
            achievable[j + 1] |= achievable[j + 1 - w]
        end
    end

    sums = Int[]

    for j in 0:W
        if achievable[j + 1]
            push!(sums, j - 1)
        end
    end

    return sums
end

# ---------------------------------------------------------------------------
# Orchestrator  (C# TreeWidth_Computation)
# ---------------------------------------------------------------------------

function treewidth_computation(weights::AbstractVector{Int}, graph::Graph{PSet}, lower_bound::Int) where {PSet}
    outlets_already_checked = Set{PSet}()

    min_k = lower_bound

    pool = PTDPool{PSet}()
    sums = achievable_sums(weights, graph)

    sub_graphs = Graph{PSet}[graph]
    separators = PSet[]                          # index j -> separator
    separator_subgraph_indices = Int[]           # index j -> subgraph index i
    child_stop = Int[1]                          # children for sep j = child_stop[j]+1 : child_stop[j+1]
    ptd_roots = Int[]                            # index i -> root index (0 = pending)
    precomputed = Dict{Int, Int}()               # subgraph index -> already computed root

    for (i, graph_i) in enumerate(sub_graphs)
        if haskey(precomputed, i)
            push!(ptd_roots, precomputed[i])
            continue
        end

        first_k = true

        while min_k < wt(weights, vertices(graph_i)) - 1
            if nv(graph_i) == 0
                push!(ptd_roots, make_ptd(pool, PSet()))
                break
            end

            if first_k
                empty!(outlets_already_checked)
            end

            state, tree_decomp_root, outlet_safe_sep = has_treewidth(
                pool, weights, CachedGraph(graph_i), min_k, graph_i, outlets_already_checked)

            if state == Halt
                push!(ptd_roots, tree_decomp_root)
                break
            elseif state == Divide
                separated_graphs, already_calc_idx, min_k = apply_externally_found_safe_separator!(
                    weights, graph_i, outlet_safe_sep, min_k, inlet(pool[tree_decomp_root]))

                # The PTD whose outlet is the separator is already a tree
                # decomposition of one of the pieces.
                if ispositive(already_calc_idx)
                    precomputed[length(sub_graphs) + already_calc_idx] = tree_decomp_root
                end

                append!(sub_graphs, separated_graphs)

                push!(separators, outlet_safe_sep)
                push!(separator_subgraph_indices, i)
                push!(child_stop, length(sub_graphs))
                push!(ptd_roots, 0)
                break
            end

            j = searchsortedfirst(sums, min_k + 1)
            min_k = sums[j]
            first_k = false
        end

        # If graph is smaller than min bound (all vertices form a single bag)
        if length(ptd_roots) < i
            push!(ptd_roots, make_ptd(pool, vertices(graph_i)))
        end
    end

    # Recombine subgraphs that have been safe separated
    for j in length(separators):-1:1
        parent_idx = separator_subgraph_indices[j]
        children = child_stop[j] + 1 : child_stop[j + 1]
        ptd_roots[parent_idx] = recombine_tree_decompositions(pool, separators[j], view(ptd_roots, children))
    end

    return (min_k, (pool, ptd_roots[1]))
end

# ---------------------------------------------------------------------------
# Search state for a single call to `has_treewidth`
# ---------------------------------------------------------------------------

mutable struct Search{PSet <: AbstractPackedSet}
    const pool::PTDPool{PSet}             # scratch pool for this search
    const weights::AbstractVector{Int}
    const graph::CachedGraph{PSet}
    const mutable_graph::Graph{PSet}
    const k::Int
    const work::Vector{PSet}
    const is_small_pmc::Vector{Bool}

    # P: the PTDs still to be processed, and the inlets of every PTD added so far
    const P::Vector{Int}
    const P_inlets::Set{PSet}

    # >2 components optimization (Section 4.6)
    const waiting::Dict{PSet, Vector{Int}}   # component -> PTDs waiting on it
    const nmissing::Dict{Int, Int}           # PTD -> number of components it waits on

    # U: the PTDURs, indexed by vertex set, and their (inlet, bag) pairs
    const sieve::LayeredSieve{PSet}
    const U::Set{Tuple{PSet, PSet}}
    const results::Vector{Int}               # query results
    const cmps::Vector{PSet}                 # components of an outlet (reused buffer; see is_minimal_separator)

    const outlets_checked::Set{PSet}
    const minor::MinorWork{PSet}
    heuristic_best::Int     # heaviest inlet tried by heuristic completion
end

function Search(weights::Weights{PSet}, graph::CachedGraph{PSet}, k::Int, mutable_graph::Graph{PSet},
                outlets_checked::Set{PSet}) where {PSet}
    return Search{PSet}(
        PTDPool{PSet}(), weights, graph, mutable_graph, k,
        Vector{PSet}(undef, domain(PSet)), Vector{Bool}(undef, domain(PSet)),
        Int[], Set{PSet}(),
        Dict{PSet, Vector{Int}}(), Dict{Int, Int}(),
        LayeredSieve{PSet}(k, weights), Set{Tuple{PSet, PSet}}(), Int[], PSet[],
        outlets_checked, MinorWork{PSet}(), 0)
end

# ---------------------------------------------------------------------------
# Core DP algorithm  (C# HasTreeWidth)
#
# Returns (state, root, separator). On `Halt`, `root` is a tree decomposition of
# the graph of width ≤ k. On `Divide`, `root` is a PTD whose outlet `separator`
# is a safe separator. The returned trees are copied into `pool`.
# ---------------------------------------------------------------------------

function has_treewidth(pool::PTDPool{PSet}, weights::AbstractVector{Int}, graph::CachedGraph{PSet}, k::Int,
                       mutable_graph::Graph{PSet}, outlets_already_checked::Set{PSet}) where {PSet}
    if nv(graph) == 0
        return (Halt, make_ptd(pool, PSet()), PSet())
    end

    search = Search(weights, graph, k, mutable_graph, outlets_already_checked)
    state, root, sep = _search!(search)

    if state != Continue
        root = copy_tree!(pool, search.pool, root)
    end

    return (state, root, sep)
end

function _search!(s::Search{PSet}) where {PSet}
    graph = s.graph; pool = s.pool; weights = s.weights; k = s.k

    # --------- lines 1-4: leaves N[v] ----------
    for v in vertices(graph)
        N = neighbors(graph, v) ∪ v
        s.is_small_pmc[v] = flag = wt(weights, N) <= k + 1 && is_pmc!(s.work, graph, N)

        if flag
            root = make_ptd(pool, N, neighbors(graph, setdiff(vertices(graph), N)))
            data = pool[root]

            if inlet(data) ∉ s.P_inlets && !is_incoming(pool, root, graph)
                is_ms, cmps = is_minimal_separator(graph, outlet(data), s.cmps)
                is_ms && _add_to_p!(s, root, cmps)
            end
        end
    end

    # --------- lines 5-27: main loop ----------
    while true
        if isempty(s.P)
            _release_waiting!(s) || break
        end

        tau = pop!(s.P)

        # line 6-7: the PTDUR with τ as its only child
        rho = create_ptdur_from_ptd(pool, tau)
        _add_to_u!(s, rho) || continue
        _flush!(s)

        R = inlet(pool[tau])
        S = outlet(pool[tau])

        # line 9-10: Tb = Te
        state, root = _extend!(s, rho, tau)
        state == Continue || return (state, root, outlet(pool[root]))

        # lines 8, 11-16: Tb = T' + τ
        for rho2 in query!(s.results, s.sieve, R, S)
            success, rho3 = add_ptd_to_ptdur_check(s.work, pool, rho2, tau, weights, graph, k)
            success || continue
            _add_to_u!(s, rho3) || continue

            state, root = _extend!(s, rho3, tau)
            state == Continue || return (state, root, outlet(pool[root]))
        end

        _flush!(s)
    end

    return (Continue, 0, PSet())
end

# lines 17-27: try to turn the PTDUR `rho` into PTDs
function _extend!(s::Search{PSet}, rho::Int, tau::Int) where {PSet}
    pool = s.pool; graph = s.graph; weights = s.weights; k = s.k
    data = pool[rho]
    B = bag(data)

    # lines 17-18: the root bag is already a PMC. No proper superset of a PMC
    # is a PMC, so there is nothing else to try.
    if wt(weights, B) == k + 1 || is_pmc!(s.work, graph, B)
        return _offer_ptd!(s, copy_ptd(pool, rho))
    end

    V = vertices(data)
    O = outlet(data)

    # lines 19-22: X = N[v] for v ∉ V with O ⊆ N[v]
    if !isempty(O)
        candidates = neighbors(graph, first(O))

        for u in O
            candidates = candidates ∩ neighbors(graph, u)
        end

        for v in setdiff(candidates, V)
            N = neighbors(graph, v) ∪ v

            if s.is_small_pmc[v] && N ⊇ B
                state, root = _offer_ptd!(s, extend_to_pmc_rule2(pool, rho, N, graph))
                state == Continue || return (state, root)
            end
        end
    end

    # lines 23-27: X = B ∪ (N(v) - inlet) for v ∈ B
    R = inlet(data)

    for v in B
        X = setdiff(neighbors(graph, v), R) ∪ B

        if wt(weights, X) <= k + 1 && is_pmc!(s.work, graph, X)
            state, root = _offer_ptd!(s, extend_to_pmc_rule3(pool, rho, X, graph))
            state == Continue || return (state, root)
        end
    end

    return (Continue, 0)
end

# Decide what to do with a freshly built PTD: finish, divide, add to P, or discard.
function _offer_ptd!(s::Search{PSet}, root::Int) where {PSet}
    pool = s.pool; graph = s.graph
    data = pool[root]

    if vertices(data) == vertices(graph)
        return (Halt, root)
    end

    if inlet(data) ∉ s.P_inlets
        is_ms, cmps = is_minimal_separator(graph, outlet(data), s.cmps)

        if is_ms && !is_incoming(pool, root, graph) && is_normalized(pool, root)
            # Try to complete the PTD heuristically whenever its inlet is
            # heavier than that of every PTD tried before. This bounds the
            # number of attempts by the weight of the graph.
            if COMPLETE_HEURISTICALLY
                w = wt(s.weights, inlet(data))

                if w > s.heuristic_best
                    s.heuristic_best = w
                    _try_heuristic_completion!(s, root) && return (Halt, root)
                end
            end

            if _outlet_is_safe_separator(s, root)
                return (Divide, root)
            end

            _add_to_p!(s, root, cmps)
            return (Continue, 0)
        end
    end

    rem_vertex!(pool, root)
    return (Continue, 0)
end

# ---------------------------------------------------------------------------
# U: add a PTDUR unless one with the same inlet and bag exists. Returns `false`
# (and frees `rho`) if it was discarded.
#
# Section 4.2 of the paper also discards a PTDUR if U contains one with the
# same inlet and a smaller bag. That is unsound for weighted graphs, and
# dropping it costs nothing on unweighted ones, so we do not do it.
# ---------------------------------------------------------------------------

function _add_to_u!(s::Search{PSet}, rho::Int) where {PSet}
    data = s.pool[rho]
    key = (inlet(data), bag(data))

    if key in s.U
        rem_vertex!(s.pool, rho)
        return false
    end

    push!(s.U, key)
    s.sieve[rho] = (bag(data), vertices(data))
    return true
end

function _flush!(s::Search)
    flush!(s.sieve)
    return
end

# ---------------------------------------------------------------------------
# P: add a PTD, possibly delaying it until PTDs for its other components exist
# (Section 4.6, C# AddToP).
# ---------------------------------------------------------------------------

function _add_to_p!(s::Search{PSet}, root::Int, cmps) where {PSet}
    R = inlet(s.pool[root])
    push!(s.P_inlets, R)

    if MORE_THAN_2_COMPONENTS_OPTIMIZATION && cmps !== nothing && length(cmps) > 2
        nmissing = 0

        for i in 2:length(cmps)   # 1 is always the incoming component
            C = cmps[i]

            if isdisjoint(C, R) && C ∉ s.P_inlets
                push!(get!(() -> Int[], s.waiting, C), root)
                nmissing += 1
            end
        end

        if ispositive(nmissing)
            s.nmissing[root] = nmissing
            return
        end
    end

    _push_p!(s, root)
    return
end

function _push_p!(s::Search{PSet}, root::Int) where {PSet}
    push!(s.P, root)

    if MORE_THAN_2_COMPONENTS_OPTIMIZATION
        R = inlet(s.pool[root])
        dependents = pop!(s.waiting, R, nothing)

        if dependents !== nothing
            for dep in dependents
                n = s.nmissing[dep] -= 1

                if iszero(n)
                    delete!(s.nmissing, dep)
                    _push_p!(s, dep)
                end
            end
        end
    end

    return
end

# When P runs dry, release every delayed PTD. Returns `true` if there were any.
function _release_waiting!(s::Search)
    isempty(s.nmissing) && return false

    for root in keys(s.nmissing)
        push!(s.P, root)
    end

    empty!(s.nmissing)
    empty!(s.waiting)
    return true
end

# ---------------------------------------------------------------------------
# OutletIsSafeSeparator
# ---------------------------------------------------------------------------

function _outlet_is_safe_separator(s::Search{PSet}, root::Int) where {PSet}
    S = outlet(s.pool[root])

    if TEST_OUTLET_IS_CLIQUE_MINOR && S ∉ s.outlets_checked
        push!(s.outlets_checked, S)
        return is_safe_separator_heuristic(s.weights, s.mutable_graph, S, s.minor)
    end

    return false
end

# ---------------------------------------------------------------------------
# TryHeuristicCompletion: complete a PTD to a tree decomposition of the whole
# graph using the min-degree heuristic. On success the completion is attached
# below `root`, which then covers the whole graph.
# ---------------------------------------------------------------------------

function _try_heuristic_completion!(s::Search{PSet}, root::Int) where {PSet}
    pool = s.pool; weights = s.weights; k = s.k; immutable_graph = s.graph
    data = pool[root]
    R = inlet(data)

    # Remove the inlet and make the outlet a clique.
    graph = Graph(setdiff(vertices(immutable_graph), R))

    for u in vertices(immutable_graph)
        if u in R
            graph.neighbors[u] = PSet()
        else
            graph.neighbors[u] = setdiff(neighbors(immutable_graph, u), R)
        end
    end

    make_into_clique!(graph, outlet(data))

    pairs, remaining = heuristic_bags_and_neighbors(weights, graph)

    for (B, _) in pairs
        wt(weights, B) > k + 1 && return false
    end

    wt(weights, remaining) > k + 1 && return false

    # Build the decomposition from the bags.
    other = make_ptd(pool, remaining)
    subtrees = Int[other]

    for idx in length(pairs):-1:1
        B, parent = pairs[idx]
        node = make_ptd(pool, B)

        for i in length(subtrees):-1:1
            if bag(pool[subtrees[i]]) ⊇ parent
                add_edge!(pool, subtrees[i], node)
                break
            end
        end

        push!(subtrees, node)
    end

    other = reroot!(pool, other, outlet(data))
    add_edge!(pool, root, other)
    return true
end
