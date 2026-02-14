# Treewidth.jl — faithful transpilation of Tamaki_Tree_Decomp/Treewidth.cs
# Contains the core DP algorithm for computing exact treewidth.
# Uses the blocksieve path (C# #define blocksieve).
# Graph reduction is SKIPPED (input is assumed pre-reduced).
# All vertex indices are 1-based.

const COMPLETE_HEURISTICALLY = true
const HEURISTIC_COMPLETION_FREQUENCY = 1
const TEST_OUTLET_IS_CLIQUE_MINOR = true
const MORE_THAN_2_COMPONENTS_OPTIMIZATION = true
const HEURISTIC_INLET_MIN = 0.0f0
const HEURISTIC_INLET_MAX = 1.0f0
const MAX_TESTS_PER_GRAPH_AND_K = typemax(Int)

@enum State Continue Divide Halt

# ---------------------------------------------------------------------------
# Main entry point  (C# TreeWidth, lines 27-51)
# ---------------------------------------------------------------------------

"""
    treewidth(graph::Graph{PSet}, weights::Vector{Int}; min_k::Int=0) where {PSet}

Compute the exact treewidth of `graph` and return `(treewidth, ptd)` where
`ptd` is a `(pool, root_index)` tuple.

`min_k` is a lower bound on the treewidth. The DP search begins at this value,
so supplying a tight lower bound (e.g. from MMD+) avoids wasted iterations.
"""
function treewidth(weights::Vector{Int}, graph::Graph{PSet}; min_k::Int=0) where {PSet}
    # Edge cases
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

function achievable_sums(weights::Vector{Int}, graph::Graph{PSet}) where {PSet}
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
# Orchestrator  (C# TreeWidth_Computation, lines 63-297)
# ---------------------------------------------------------------------------

function treewidth_computation(weights::Vector{Int}, graph::Graph{PSet}, lower_bound::Int) where {PSet}
    outlets_already_checked = Set{PSet}()

    min_k = lower_bound

    pool = PTDPool{PSet}()
    sums = achievable_sums(weights, graph)

    sub_graphs = Graph{PSet}[graph]
    separators = PSet[]                          # index j -> separator
    separator_subgraph_indices = Int[]           # index j -> subgraph index i
    child_stop = Int[1]                      # children for sep j = child_stop[j]+1 : child_stop[j+1]
    ptd_roots = Int[]                            # index i -> root index (0 = pending)

    for (i, graph_i) in enumerate(sub_graphs)
        # Loop over possible treewidths for the current subgraph (C# line 141)
        first_k = true

        while min_k < wt(weights, vertices(graph_i)) - 1
            # Break early if no vertices remain (C# lines 152-161)
            if nv(graph_i) == 0
                push!(ptd_roots, make_ptd(pool, PSet()))
                break
            end

            if first_k
                empty!(outlets_already_checked)
            end

            immutable_graph = CachedGraph(graph_i)

            state, tree_decomp_root, outlet_safe_sep = has_treewidth(
                pool, weights, immutable_graph, min_k, graph_i, outlets_already_checked)

            if state == Halt
                push!(ptd_roots, tree_decomp_root)
                break
            elseif state == Divide
                tree_decomp_data = pool[tree_decomp_root]
                separated_graphs, already_calc_idx, min_k = apply_externally_found_safe_separator!(
                    weights, graph_i, outlet_safe_sep, min_k, inlet(tree_decomp_data))

                append!(sub_graphs, separated_graphs)

                push!(separators, outlet_safe_sep)
                push!(separator_subgraph_indices, i)
                push!(child_stop, length(sub_graphs))
                push!(ptd_roots, 0)

                # Continue with the next graph
                break
            end

            j = searchsortedfirst(sums, min_k + 1)
            min_k = sums[j]
            first_k = false
        end

        # If graph is smaller than min bound (all vertices form a single bag) (C# lines 268-271)
        if length(ptd_roots) < i
            push!(ptd_roots, make_ptd(pool, vertices(graph_i)))
        end

    end

    # Recombine subgraphs that have been safe separated (C# lines 277-293)
    for j in length(separators):-1:1
        parent_idx = separator_subgraph_indices[j]
        children = child_stop[j] + 1 : child_stop[j + 1]
        ptd_roots[parent_idx] = recombine_tree_decompositions(pool, separators[j], view(ptd_roots, children))
    end

    return (min_k, (pool, ptd_roots[1]))
end

# ---------------------------------------------------------------------------
# Core DP algorithm  (C# HasTreeWidth, lines 317-736)
#
# Uses the blocksieve path.
# Returns (found::Bool, root_or_nothing::Union{Int, Nothing}, outlet_safe_sep::Union{PSet, Nothing}).
# ---------------------------------------------------------------------------

function has_treewidth(pool::PTDPool{PSet}, weights::Vector{Int}, graph::CachedGraph{PSet}, k::Int, mutable_graph::Graph{PSet},
                       outlets_already_checked::Set{PSet}) where {PSet}
    if nv(graph) == 0
        return (Halt, make_ptd(pool, PSet()), PSet())
    end

    ptd_roots = Int[]              # Stack of root indices (push!/pop!)
    ptd_inlets = Set{PSet}()

    sieve = LayeredSieve{PSet}(k, weights)
    ptdur_inlets = Dict{PSet, Int}()    # inlet -> ptdur root index

    is_small_pmc = Vector{Bool}(undef, domain(PSet))
    work = Vector{PSet}(undef, domain(PSet))

    # moreThan2ComponentsOptimization data structures (C# lines 345-346)
    if MORE_THAN_2_COMPONENTS_OPTIMIZATION
        cmp_to_ptd_roots = Dict{PSet, Vector{Int}}()
        ptd_root_to_cmps = Dict{Int, Int}()
    else
        cmp_to_ptd_roots = nothing
        ptd_root_to_cmps = nothing
    end

    heuristic_completion_in = 1
    current_tests_per_gk = 0

    # --------- lines 2 to 6: Initialize PTDs from vertex neighborhoods ----------
    # (C# lines 357-380)

    for v in vertices(graph)
        N = neighbors(graph, v) ∪ v
        is_small_pmc[v] = flag = wt(weights, N) <= k + 1 && is_pmc!(work, graph, N)

        if flag
            ptd_root = make_ptd(pool, N, neighbors(graph, setdiff(vertices(graph), N)))
            ptd_data = pool[ptd_root]
            S = outlet(ptd_data)
            R = inlet(ptd_data)

            if R ∉ ptd_inlets && !is_incoming(pool, ptd_root, graph)
                is_min_sep, cmps = is_minimal_separator(graph, S)
                is_min_sep && _add_to_p!(pool, ptd_root, cmps, graph, ptd_roots, ptd_inlets, cmp_to_ptd_roots, ptd_root_to_cmps)
            end
        end
    end

    # --------- lines 7 to 32: Main DP loop ----------
    # (C# lines 384-721)

    while !isempty(ptd_roots)
        ptd_root = pop!(ptd_roots)

        # --------- line 9 ----------
        ptdur_root_1 = create_ptdur_from_ptd(pool, ptd_root)

        # --------- line 10: add PTDUR to U ----------
        # (C# lines 400-421, blocksieve path)
        ptdur_data_1 = pool[ptdur_root_1]
        V_1 = vertices(ptdur_data_1)
        B_1 = bag(ptdur_data_1)
        R_1 = inlet(ptdur_data_1)

        if haskey(ptdur_inlets, R_1)
            ptdur_root_0 = ptdur_inlets[R_1]
            ptdur_data_0 = pool[ptdur_root_0]
            B_0 = bag(ptdur_data_0)

            if wt(weights, B_1) < wt(weights, B_0)
                replace_subset!(sieve, ptdur_root_0, B_0, ptdur_root_1, B_1, V_1)
                ptdur_inlets[R_1] = ptdur_root_1
                rem_vertex!(pool, ptdur_root_0)
            else
                rem_vertex!(pool, ptdur_root_1)
                continue
            end
        else
            ptdur_inlets[R_1] = ptdur_root_1
            sieve[ptdur_root_1] = (B_1, V_1)
            flush!(sieve)
        end

        # --------- lines 11-32: iterate over eligible PTDURs ----------
        # (C# lines 448-718)
        ptd_data = pool[ptd_root]
        R = inlet(ptd_data)
        S = outlet(ptd_data)

        for ptdur_root_2 in query(sieve, R, S, ptdur_root_1)
            ptdur_data_2 = pool[ptdur_root_2]
            R_2 = inlet(ptdur_data_2)

            # --------- lines 12-15 ----------
            # (C# lines 460-520)
            if R_1 != R_2
                success, new_root = add_ptd_to_ptdur_check(work, pool, ptdur_root_2, ptd_root, weights, graph, k, mutable_graph)
                success || continue
   
                ptdur_root_3 = new_root

                # Add to U (blocksieve path, C# lines 470-492)
                ptdur_data_3 = pool[ptdur_root_3]
                V_3 = vertices(ptdur_data_3)
                B_3 = bag(ptdur_data_3)
                R_3 = inlet(ptdur_data_3)

                if haskey(ptdur_inlets, R_3)
                    ptdur_root_0 = ptdur_inlets[R_3]
                    ptdur_data_0 = pool[ptdur_root_0]
                    B_0 = bag(ptdur_data_0)

                    if wt(weights, B_3) < wt(weights, B_0)
                        replace_subset!(sieve, ptdur_root_0, B_0, ptdur_root_3, B_3, V_3)
                        ptdur_inlets[R_3] = ptdur_root_3
                        rem_vertex!(pool, ptdur_root_0)
                    else
                        rem_vertex!(pool, ptdur_root_3)
                        continue
                    end
                else
                    sieve[ptdur_root_3] = (B_3, V_3)
                    ptdur_inlets[R_3] = ptdur_root_3
                end
            else
                ptdur_root_3 = ptdur_root_1
            end

            ptdur_data_3 = pool[ptdur_root_3]
            B_3 = bag(ptdur_data_3)
            ptdur_size = wt(weights, B_3)

            # --------- lines 16-20: Rule 1 - Check if completion ----------
            if ptdur_size == k + 1 || is_pmc!(work, graph, B_3)
                state, root, sep, heuristic_completion_in, current_tests_per_gk =
                    _rule1!(pool, ptdur_root_3, ptd_root, weights, graph, k, mutable_graph,
                            ptd_roots, ptd_inlets, cmp_to_ptd_roots, ptd_root_to_cmps, outlets_already_checked,
                            heuristic_completion_in, current_tests_per_gk)

                state == Continue || return (state, root, sep)
            else
                # --------- lines 21-26: Rule 2 ----------
                state, root, sep, heuristic_completion_in, current_tests_per_gk =
                    _rule2!(pool, ptdur_root_3, ptdur_data_3, ptd_root, weights, graph, k, mutable_graph,
                            is_small_pmc, ptd_roots, ptd_inlets, cmp_to_ptd_roots, ptd_root_to_cmps, outlets_already_checked,
                            heuristic_completion_in, current_tests_per_gk)

                state == Continue || return (state, root, sep)

                # --------- lines 27-32: Rule 3 ----------
                state, root, sep, heuristic_completion_in, current_tests_per_gk =
                    _rule3!(work, pool, ptdur_root_3, ptdur_data_3, ptd_root, weights, graph, k, mutable_graph,
                            ptd_roots, ptd_inlets, cmp_to_ptd_roots, ptd_root_to_cmps, outlets_already_checked,
                            heuristic_completion_in, current_tests_per_gk)

                state == Continue || return (state, root, sep)
            end
        end

        # Flush deferred additions (C# lines 719-721, blocksieve path)
        flush!(sieve)
    end

    return (Continue, 0, PSet())
end

# ---------------------------------------------------------------------------
# Rule 1: Check if completion (C# lines 530-580)
# ---------------------------------------------------------------------------

function _rule1!(pool::PTDPool{PSet}, ptdur_root::Int, ptd_root::Int,
                 weights::Vector{Int}, graph::CachedGraph{PSet}, k::Int, mutable_graph::Graph{PSet},
                 ptd_roots::Vector{Int}, ptd_inlets::Set{PSet}, cmp_to_ptd_roots, ptd_root_to_cmps,
                 outlets_already_checked::Set{PSet},
                 heuristic_completion_in::Int, current_tests_per_gk::Int) where {PSet}
    new_ptd_root = copy_ptd(pool, ptdur_root)
    consumed = false
    new_ptd_data = pool[new_ptd_root]
    V_new = vertices(new_ptd_data)
    R_new = inlet(new_ptd_data)
    S_new = outlet(new_ptd_data)

    if V_new == vertices(graph)
        return (Halt, new_ptd_root, PSet(), heuristic_completion_in, current_tests_per_gk)
    end

    if !(R_new in ptd_inlets)
        is_ms, cmps = is_minimal_separator(graph, S_new)

        if is_ms && !is_incoming(pool, new_ptd_root, graph) && is_normalized(pool, new_ptd_root)
            heuristic_completion_in -= 1

            if COMPLETE_HEURISTICALLY && heuristic_completion_in == 0
                success, current_tests_per_gk =
                    _try_heuristic_completion!(pool, weights, graph, new_ptd_root, k, mutable_graph,
                        current_tests_per_gk)

                if success
                    return (Halt, new_ptd_root, PSet(), heuristic_completion_in, current_tests_per_gk)
                end
            end
            if heuristic_completion_in < 0
                heuristic_completion_in = HEURISTIC_COMPLETION_FREQUENCY
            end

            if _outlet_is_safe_separator(pool, new_ptd_root, weights, graph, mutable_graph, outlets_already_checked)
                return (Divide, new_ptd_root, outlet(pool[ptd_root]), heuristic_completion_in, current_tests_per_gk)
            end

            _add_to_p!(pool, new_ptd_root, cmps, graph, ptd_roots, ptd_inlets, cmp_to_ptd_roots, ptd_root_to_cmps)
            consumed = true
        end
    end

    consumed || rem_vertex!(pool, new_ptd_root)

    return (Continue, 0, PSet(), heuristic_completion_in, current_tests_per_gk)
end

# ---------------------------------------------------------------------------
# Rule 2: Extend via vertex neighborhood (C# lines 582-648)
# ---------------------------------------------------------------------------

function _rule2!(pool::PTDPool{PSet}, ptdur_root::Int, ptdur_data::PTD{PSet}, ptd_root::Int,
                 weights::Vector{Int}, graph::CachedGraph{PSet}, k::Int, mutable_graph::Graph{PSet},
                 is_small_pmc::Vector{Bool},
                 ptd_roots::Vector{Int}, ptd_inlets::Set{PSet}, cmp_to_ptd_roots, ptd_root_to_cmps,
                 outlets_already_checked::Set{PSet},
                 heuristic_completion_in::Int, current_tests_per_gk::Int) where {PSet}
    V = vertices(ptdur_data)
    B = bag(ptdur_data)
    S = outlet(ptdur_data)

    # Compute candidates: common neighbors of all outlet vertices, minus vertices already covered
    candidates = vertices(graph)
    first_outlet = true
    remaining = S

    while !isempty(remaining)
        v, remaining = popfirst_nonempty(remaining)
        if first_outlet
            candidates = neighbors(graph, v)
            first_outlet = false
        else
            candidates = candidates ∩ neighbors(graph, v)
        end
    end
    if first_outlet
        candidates = PSet()  # empty outlet => no candidates
    end

    candidates = setdiff(candidates, V)

    # Process each candidate
    remaining = candidates

    while !isempty(remaining)
        v, remaining = popfirst_nonempty(remaining)

        N = neighbors(graph, v) ∪ v

        if is_small_pmc[v] && N ⊇ B
            new_ptd_root = extend_to_pmc_rule2(pool, ptdur_root, N, graph)
            consumed = false
            new_ptd_data = pool[new_ptd_root]
            V_new = vertices(new_ptd_data)
            R_new = inlet(new_ptd_data)
            S_new = outlet(new_ptd_data)

            if V_new == vertices(graph)
                return (Halt, new_ptd_root, PSet(), heuristic_completion_in, current_tests_per_gk)
            end

            heuristic_completion_in -= 1

            if !(R_new in ptd_inlets)
                is_ms, cmps = is_minimal_separator(graph, S_new)

                if is_ms && !is_incoming(pool, new_ptd_root, graph) && is_normalized(pool, new_ptd_root)
                    if COMPLETE_HEURISTICALLY && heuristic_completion_in == 0
                        success, current_tests_per_gk =
                            _try_heuristic_completion!(pool, weights, graph, new_ptd_root, k, mutable_graph,
                                current_tests_per_gk)

                        if success
                            return (Halt, new_ptd_root, PSet(), heuristic_completion_in, current_tests_per_gk)
                        end
                    end
                    if heuristic_completion_in < 0
                        heuristic_completion_in = HEURISTIC_COMPLETION_FREQUENCY
                    end

                    if _outlet_is_safe_separator(pool, new_ptd_root, weights, graph, mutable_graph, outlets_already_checked)
                        return (Divide, new_ptd_root, outlet(pool[ptd_root]), heuristic_completion_in, current_tests_per_gk)
                    end

                    _add_to_p!(pool, new_ptd_root, cmps, graph, ptd_roots, ptd_inlets, cmp_to_ptd_roots, ptd_root_to_cmps)
                    consumed = true
                end
            end

            consumed || rem_vertex!(pool, new_ptd_root)
        end
    end

    return (Continue, 0, PSet(), heuristic_completion_in, current_tests_per_gk)
end

# ---------------------------------------------------------------------------
# Rule 3: Extend by adding neighbors (C# lines 650-713)
# ---------------------------------------------------------------------------

function _rule3!(work::Vector{PSet}, pool::PTDPool{PSet}, ptdur_root::Int, ptdur_data::PTD{PSet}, ptd_root::Int,
                 weights::Vector{Int}, graph::CachedGraph{PSet}, k::Int, mutable_graph::Graph{PSet},
                 ptd_roots::Vector{Int}, ptd_inlets::Set{PSet}, cmp_to_ptd_roots, ptd_root_to_cmps,
                 outlets_already_checked::Set{PSet},
                 heuristic_completion_in::Int, current_tests_per_gk::Int) where {PSet}
    B = bag(ptdur_data)
    R = inlet(ptdur_data)

    for v in B
        pot_new_bag = setdiff(neighbors(graph, v), R) ∪ B

        if wt(weights, pot_new_bag) <= k + 1 && is_pmc!(work, graph, pot_new_bag)
            new_ptd_root = extend_to_pmc_rule3(pool, ptdur_root, pot_new_bag, graph)
            consumed = false
            new_ptd_data = pool[new_ptd_root]
            V_new = vertices(new_ptd_data)
            R_new = inlet(new_ptd_data)
            S_new = outlet(new_ptd_data)

            if V_new == vertices(graph)
                return (Halt, new_ptd_root, PSet(), heuristic_completion_in, current_tests_per_gk)
            end

            if !(R_new in ptd_inlets)
                is_ms, cmps = is_minimal_separator(graph, S_new)

                if is_ms && !is_incoming(pool, new_ptd_root, graph) && is_normalized(pool, new_ptd_root)
                    heuristic_completion_in -= 1

                    if COMPLETE_HEURISTICALLY && heuristic_completion_in == 0
                        success, current_tests_per_gk =
                            _try_heuristic_completion!(pool, weights, graph, new_ptd_root, k, mutable_graph,
                                current_tests_per_gk)

                        if success
                            return (Halt, new_ptd_root, PSet(), heuristic_completion_in, current_tests_per_gk)
                        end
                    end
                    if heuristic_completion_in < 0
                        heuristic_completion_in = HEURISTIC_COMPLETION_FREQUENCY
                    end

                    if _outlet_is_safe_separator(pool, new_ptd_root, weights, graph, mutable_graph, outlets_already_checked)
                        return (Divide, new_ptd_root, outlet(pool[ptd_root]), heuristic_completion_in, current_tests_per_gk)
                    end

                    _add_to_p!(pool, new_ptd_root, cmps, graph, ptd_roots, ptd_inlets, cmp_to_ptd_roots, ptd_root_to_cmps)
                    consumed = true
                end
            end

            consumed || rem_vertex!(pool, new_ptd_root)
        end
    end

    return (Continue, 0, PSet(), heuristic_completion_in, current_tests_per_gk)
end

# ---------------------------------------------------------------------------
# AddToP  (C# AddToP, lines 896-963)
# ---------------------------------------------------------------------------

function _add_to_p!(pool::PTDPool{PSet}, ptd_root::Int, cmps, graph::CachedGraph{PSet}, ptd_roots::Vector{Int},
                    ptd_inlets::Set{PSet}, cmp_to_ptd_roots, ptd_root_to_cmps;
                    is_confirmed_no_missing::Bool=false) where {PSet}
    ptd_data = pool[ptd_root]
    ptd_inlet = inlet(ptd_data)
    # Mark this PTD as accounted for (C# line 899)
    push!(ptd_inlets, ptd_inlet)

    if MORE_THAN_2_COMPONENTS_OPTIMIZATION
        # (C# lines 904-933)
        if !is_confirmed_no_missing && cmps !== nothing && length(cmps) > 2
            # Determine missing cmps
            missing_cmps = PSet[]

            for ci in 2:length(cmps)   # start from index 2 (1 is always the incoming cmp)
                cmp = cmps[ci]
                if isdisjoint(cmp, ptd_inlet)
                    if !(cmp in ptd_inlets)
                        push!(missing_cmps, cmp)
                        if haskey(cmp_to_ptd_roots, cmp)
                            push!(cmp_to_ptd_roots[cmp], ptd_root)
                        else
                            cmp_to_ptd_roots[cmp] = Int[ptd_root]
                        end
                    end
                end
            end

            # If cmps are missing, store and return (C# lines 928-932)
            if !isempty(missing_cmps)
                ptd_root_to_cmps[ptd_root] = length(missing_cmps)
                return
            end
        end

        # No cmps missing: add to P (C# line 936)
        push!(ptd_roots, ptd_root)

        # Update dependent PTDs (C# lines 940-956)
        if haskey(cmp_to_ptd_roots, ptd_inlet)
            dependents = cmp_to_ptd_roots[ptd_inlet]

            for dep_root in dependents
                ptd_root_to_cmps[dep_root] -= 1
                if ptd_root_to_cmps[dep_root] == 0
                    delete!(ptd_root_to_cmps, dep_root)
                    _add_to_p!(pool, dep_root, nothing, graph, ptd_roots, ptd_inlets, cmp_to_ptd_roots, ptd_root_to_cmps;
                               is_confirmed_no_missing=true)
                end
            end
            delete!(cmp_to_ptd_roots, ptd_inlet)
        end
    else
        # (C# lines 958-962)
        push!(ptd_roots, ptd_root)
    end
end

# ---------------------------------------------------------------------------
# OutletIsSafeSeparator  (C# OutletIsSafeSeparator, lines 971-983)
# ---------------------------------------------------------------------------

function _outlet_is_safe_separator(pool::PTDPool{PSet}, ptd_root::Int, weights::Vector{Int}, graph::CachedGraph{PSet}, mutable_graph::Graph{PSet}, outlets_checked::Set{PSet}) where {PSet}
    ptd_data = pool[ptd_root]

    if TEST_OUTLET_IS_CLIQUE_MINOR && !(outlet(ptd_data) in outlets_checked)
        push!(outlets_checked, outlet(ptd_data))
        return is_safe_separator_heuristic(weights, mutable_graph, outlet(ptd_data))
    end
    return false
end

# ---------------------------------------------------------------------------
# TryHeuristicCompletion  (C# TryHeuristicCompletion, lines 1007-1129)
# ---------------------------------------------------------------------------

function _try_heuristic_completion!(pool::PTDPool{PSet}, weights::Vector{Int}, immutable_graph::CachedGraph{PSet}, ptd_root::Int, k::Int,
                                    mutable_graph_template::Graph{PSet},
                                    tests_per_gk::Int) where {PSet}
    ptd_data = pool[ptd_root]
    ptd_inlet = inlet(ptd_data)
    # Inlet ratio check (C# lines 1022-1026)
    inlet_ratio = Float32(wt(weights, ptd_inlet)) / wt(weights, vertices(immutable_graph))

    if inlet_ratio < HEURISTIC_INLET_MIN || inlet_ratio > HEURISTIC_INLET_MAX
        return (false, tests_per_gk)
    end

    tests_per_gk += 1

    if tests_per_gk > MAX_TESTS_PER_GRAPH_AND_K
        return (false, tests_per_gk)
    end

    # Build subgraph: remove inlet vertices, make outlet a clique (C# lines 1038-1061)
    not_removed = setdiff(vertices(immutable_graph), ptd_inlet)
    graph = Graph(not_removed)

    for u in vertices(immutable_graph)
        if u in ptd_inlet
            graph.neighbors[u] = PSet()
        else
            graph.neighbors[u] = setdiff(neighbors(immutable_graph, u), ptd_inlet)
        end
    end
    make_into_clique!(graph, outlet(ptd_data))

    # Calculate bags using heuristic (C# lines 1064-1077)
    bags_and_parents_stack = Tuple{PSet, PSet}[]
    heuristic_pairs, remaining_clique = heuristic_bags_and_neighbors(weights, graph)

    for (bag, parent) in heuristic_pairs
        if wt(weights, bag) > k + 1
            return (false, tests_per_gk)
        end
        push!(bags_and_parents_stack, (bag, parent))
    end

    # Return also if the remaining clique is too large (C# lines 1074-1077)
    if wt(weights, remaining_clique) > k + 1
        return (false, tests_per_gk)
    end

    # Build the PTD from those bags (C# lines 1080-1108)
    other_root = make_ptd(pool, remaining_clique)
    subtree_list = Int[other_root]

    for idx in length(bags_and_parents_stack):-1:1
        current_bag, current_parent = bags_and_parents_stack[idx]
        current_root = make_ptd(pool, current_bag)

        # Iterate from most recent to least recent (C# lines 1092-1103)
        for i in length(subtree_list):-1:1
            if bag(pool[subtree_list[i]]) ⊇ current_parent
                add_edge!(pool, subtree_list[i], current_root)
                break
            end
        end
        push!(subtree_list, current_root)
    end

    # Reroot and attach (C# lines 1110-1112)
    other_root = reroot!(pool, other_root, outlet(ptd_data))
    add_edge!(pool, ptd_root, other_root)

    return (true, tests_per_gk)
end
