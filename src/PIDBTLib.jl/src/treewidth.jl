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
#   - The sieve is keyed by bags rather than vertex sets, and tests the bag
#     size condition of a combination exactly (see sieve.jl).
#   - On weighted graphs, the widths are not tried one by one: after a few
#     failures, the search skips ahead and then bisects (see "Galloping"
#     in `treewidth_computation`).
#
# All vertex indices are 1-based.

const COMPLETE_HEURISTICALLY = true
const TEST_OUTLET_IS_CLIQUE_MINOR = true
const MORE_THAN_2_COMPONENTS_OPTIMIZATION = true
const GALLOP = true
const GALLOP_GROWTH = 1.7       # gallop if failed rounds grow by at most this factor per width
const GALLOP_MIN_EFFORT = 10000 # ignore cheaper rounds when estimating the growth
const GALLOP_BUDGET = 3.0       # budget of a speculative round, relative to its predicted cost

@enum State Continue Divide Halt Abort Timeout

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
function treewidth(weights::AbstractVector{Int}, graph::Graph{PSet}; min_k::Int=0, deadline::Float64=Inf) where {PSet}
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

    return treewidth_computation(weights, graph, min_k, deadline)
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

# Returns `nothing` if the deadline passes.
function treewidth_computation(weights::AbstractVector{Int}, graph::Graph{PSet}, lower_bound::Int, deadline::Float64=Inf) where {PSet}
    outlets_already_checked = Set{PSet}()

    min_k = lower_bound

    pool = PTDPool{PSet}()
    sums = achievable_sums(weights, graph)

    # Galloping (see below) only pays for weighted graphs, where the widths
    # to try are finely spaced. On unweighted graphs, failed rounds usually
    # double in cost from one width to the next, and it is a wash.
    gallop = GALLOP && weights isa Weights && length(weights.values) > 1

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

        if nv(graph_i) == 0
            push!(ptd_roots, make_ptd(pool, PSet()))
            continue
        end

        empty!(outlets_already_checked)

        # Find the least width k = sums[ihi] ≥ min_k at which graph_i has a
        # tree decomposition. Invariants: graph_i has no decomposition of width
        # sums[ilo] (or sums[ilo] < min_k), and has one of width sums[ihi]:
        # `root` (or a single bag, if root == 0).
        ilo = searchsortedfirst(sums, min_k) - 1
        ihi = searchsortedfirst(sums, wt(weights, vertices(graph_i)) - 1)
        root = 0
        divided = false

        # Galloping. Failed rounds get more expensive as k grows, by a factor
        # `growth` per step (estimated from the last two failed rounds). If it
        # is small, there are many rounds of similar cost below tw, and it pays
        # to skip ahead: probe ilo + step, doubling `step` after every failure,
        # and binary search once a decomposition is found. Every failed probe
        # is a round that k-by-k search would run as well, so the only risk is
        # a successful probe far above tw, which can be expensive. So a probe
        # beyond ilo + 1 runs with a budget of a few times the predicted cost
        # of a failure; if it runs out, we go back to k-by-k search.
        step = 1
        effort = 0               # effort of the last failed round
        growth = Inf

        while ilo + 1 < ihi
            if !gallop || ilo + 1 >= ihi - 1 || growth > GALLOP_GROWTH
                p = ilo + 1
            elseif iszero(root)
                p = min(ilo + step, ihi - 1)
            else
                p = (ilo + ihi) ÷ 2
            end

            if p == ilo + 1
                budget = typemax(Int)
            else
                budget = round(Int, min(GALLOP_BUDGET * effort * growth^(p - ilo), 1e15))
            end

            state, tree_decomp_root, outlet_safe_sep, n = has_treewidth(
                pool, weights, CachedGraph(graph_i), sums[p], graph_i, outlets_already_checked, budget, deadline)

            state == Timeout && return nothing

            if state == Continue
                # (cheap rounds say little about the growth)
                growth = p == ilo + 1 && n >= GALLOP_MIN_EFFORT && ispositive(effort) ? n / effort : growth
                ilo = p; effort = n
                step = growth > GALLOP_GROWTH ? 1 : 2step
            elseif state == Halt
                ihi = searchsortedfirst(sums, treewidth(pool, tree_decomp_root, weights))
                root = tree_decomp_root
            elseif state == Abort
                step = 1
                growth = Inf   # stop galloping
            else # state == Divide
                # Safe separators do not depend on k. But the PTD for one of
                # the pieces has width ≤ sums[p], which is only known to be
                # optimal if p == ilo + 1.
                separated_graphs, already_calc_idx, min_k = apply_externally_found_safe_separator!(
                    weights, graph_i, outlet_safe_sep, sums[ilo + 1], inlet(pool[tree_decomp_root]))

                if ispositive(already_calc_idx) && p == ilo + 1
                    precomputed[length(sub_graphs) + already_calc_idx] = tree_decomp_root
                end

                append!(sub_graphs, separated_graphs)

                push!(separators, outlet_safe_sep)
                push!(separator_subgraph_indices, i)
                push!(child_stop, length(sub_graphs))
                push!(ptd_roots, 0)
                divided = true
                break
            end
        end

        if !divided
            min_k = max(min_k, sums[ihi])
            # If no round succeeded, all vertices form a single bag.
            push!(ptd_roots, iszero(root) ? make_ptd(pool, vertices(graph_i)) : root)
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

# The width (largest bag weight, minus one) of the tree decomposition at `root`.
function treewidth(pool::PTDPool{PSet}, root::Int, weights::AbstractVector{Int}) where {PSet}
    width = -1
    stack = Int[root]

    while !isempty(stack)
        node = pop!(stack)
        width = max(width, wt(weights, bag(pool[node])) - 1)

        for p in incident(pool, node)
            push!(stack, target(pool, p))
        end
    end

    return width
end

# ---------------------------------------------------------------------------
# Search state for a single call to `has_treewidth`
# ---------------------------------------------------------------------------

mutable struct Search{PSet <: AbstractPackedSet}
    const pool::PTDPool{PSet}             # scratch pool for this search
    const weights::Weights{PSet}
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
    const tried::Vector{PSet}                # rule-3 bags tried (see _extend!)
    const cmps::Vector{PSet}                 # components of an outlet (reused buffer; see is_minimal_separator)

    const outlets_checked::Set{PSet}
    const minor::MinorWork{PSet}
    const deadline::Float64
    heuristic_best::Int     # heaviest inlet tried by heuristic completion
    const budget::Int       # give up (Abort) once `effort` exceeds this
    effort::Int             # PTDs processed plus PTDURs they were combined with
end

function Search(weights::Weights{PSet}, graph::CachedGraph{PSet}, k::Int, mutable_graph::Graph{PSet},
                outlets_checked::Set{PSet}, budget::Int=typemax(Int), deadline::Float64=Inf) where {PSet}
    return Search{PSet}(
        PTDPool{PSet}(), weights, graph, mutable_graph, k,
        Vector{PSet}(undef, domain(PSet)), Vector{Bool}(undef, domain(PSet)),
        Int[], Set{PSet}(),
        Dict{PSet, Vector{Int}}(), Dict{Int, Int}(),
        LayeredSieve{PSet}(k, weights), Set{Tuple{PSet, PSet}}(), Int[], PSet[], PSet[],
        outlets_checked, MinorWork{PSet}(), deadline, 0, budget, 0)
end

# ---------------------------------------------------------------------------
# Core DP algorithm  (C# HasTreeWidth)
#
# Returns (state, root, separator, effort). On `Halt`, `root` is a tree
# decomposition of the graph of width ≤ k. On `Divide`, `root` is a PTD whose
# outlet `separator` is a safe separator. The returned trees are copied into
# `pool`. `effort` is the number of PTDs processed plus the number of PTDURs
# they were combined with, a deterministic measure of running time. On `Abort`,
# the effort exceeded `budget` before a decision was reached. On `Timeout`, the
# deadline has passed.
# ---------------------------------------------------------------------------

function has_treewidth(pool::PTDPool{PSet}, weights::AbstractVector{Int}, graph::CachedGraph{PSet}, k::Int,
                       mutable_graph::Graph{PSet}, outlets_already_checked::Set{PSet}, budget::Int=typemax(Int), deadline::Float64=Inf) where {PSet}
    if nv(graph) == 0
        return (Halt, make_ptd(pool, PSet()), PSet(), 0)
    end

    search = Search(weights, graph, k, mutable_graph, outlets_already_checked, budget, deadline)
    state, root, sep = _search!(search)

    if state == Halt || state == Divide
        root = copy_tree!(pool, search.pool, root)
    end

    return (state, root, sep, search.effort)
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
    # The deadline is checked on every iteration, and every 64 combinations
    # within one: a single iteration can combine τ with very many PTDURs, and
    # on dense graphs 64 iterations can take long enough to exhaust memory.
    while true
        if isempty(s.P)
            _release_waiting!(s) || break
        end

        if isfinite(s.deadline) && time() > s.deadline
            return (Timeout, 0, PSet())
        end

        tau = pop!(s.P)
        (s.effort += 1) > s.budget && return (Abort, 0, PSet())

        # line 6-7: the PTDUR with τ as its only child
        rho = create_ptdur_from_ptd(pool, tau)
        _add_to_u!(s, rho) || continue
        _flush!(s)

        R = inlet(pool[tau])
        S = outlet(pool[tau])

        # line 9-10: Tb = Te. The root bag of rho is the outlet of tau, a
        # minimal separator, which has full components and is not a PMC.
        state, root = _extend!(s, rho, tau, false)
        state == Continue || return (state, root, outlet(pool[root]))

        # lines 8, 11-16: Tb = T' + τ
        query!(s.results, s.sieve, R, S)
        s.effort += length(s.results)

        for (i, rho2) in enumerate(s.results)
            if iszero(i & 63) && isfinite(s.deadline) && time() > s.deadline
                return (Timeout, 0, PSet())
            end

            success, rho3, ispmc = add_ptd_to_ptdur_check(s.work, pool, rho2, tau, weights, graph, k)
            success || continue
            _add_to_u!(s, rho3) || continue

            state, root = _extend!(s, rho3, tau, ispmc)
            state == Continue || return (state, root, outlet(pool[root]))
        end

        _flush!(s)
    end

    return (Continue, 0, PSet())
end

# lines 17-27: try to turn the PTDUR `rho` into PTDs. `ispmc` tells whether
# the root bag of `rho` is a PMC.
function _extend!(s::Search{PSet}, rho::Int, tau::Int, ispmc::Bool) where {PSet}
    pool = s.pool; graph = s.graph; weights = s.weights; k = s.k
    data = pool[rho]
    B = bag(data)

    # lines 17-18: the root bag is already a PMC. No proper superset of a PMC
    # is a PMC, so there is nothing else to try.
    if ispmc || wt(weights, B) == k + 1
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

    # lines 23-27: X = B ∪ (N(v) - inlet) for v ∈ B. Different v often give
    # the same X, so remember the ones tried. (X = B is not a PMC; see above.)
    R = inlet(data)
    tried = empty!(s.tried)

    budget = k + 1 - wt(weights, B)

    for v in B
        D = setdiff(neighbors(graph, v), R ∪ B)
        (isempty(D) || !wtatmost(weights, D, budget)) && continue
        X = B ∪ D
        X in tried && continue
        push!(tried, X)

        if is_pmc!(s.work, graph, X)
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
