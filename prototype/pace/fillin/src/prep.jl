# Preprocessing for minimum fill-in (all rules exact):
#   1. simplicial vertices: minfill(G) = minfill(G - v) when N(v) is a clique.
#   2. connected components.
#   3. clique separators (any clique separator S, minimal or not): minfill(G) = sum over the
#      pieces G[C ∪ S]. Candidates are the sets madj_H(v) of a minimal triangulation H (MCS-M),
#      as in the atom decomposition of Berry, Pogorelcnik and Simonet (2010); each candidate is
#      checked to be a clique separator of the current graph, so correctness does not depend on H.
#   4. a minimal separator N(C), C a component of G - N[u], with exactly one missing edge:
#      the edge is added (forced fill) and the separator becomes a clique separator. This is the
#      rule of the PACE 2017 Track B winner (Kobayashi-Tamaki, Solver.safeSeparator).
# The almost-clique separator rule for treewidth (S minus one vertex a clique) is NOT used: it is
# not safe for fill-in as stated. (The winner has a restricted version with a disjoint-paths test.)

using CliqueTrees, Graphs

isclique(G, S) = all(v -> length(setdiff(S, G[v])) <= 1, S)   # every u in S \ {v} adjacent to v

function missing_pairs(G, S; limit = typemax(Int))
    out = Tuple{Int, Int}[]
    for v in S, u in S
        u > v && !(u in G[v]) && (push!(out, (v, u)); length(out) > limit && return out)
    end
    return out
end

# Components of G[S \ X] (X ⊆ S) as BitSets.
function components(G, S::BitSet, X::BitSet = BitSet())
    rest = setdiff(S, X); out = BitSet[]
    while !isempty(rest)
        v = first(rest); C = BitSet(v); q = [v]; delete!(rest, v)
        while !isempty(q)
            x = pop!(q)
            for u in G[x]
                if u in rest
                    delete!(rest, u); push!(C, u); push!(q, u)
                end
            end
        end
        push!(out, C)
    end
    return out
end
nbhd(G, C::BitSet, S::BitSet) = (N = BitSet(); for v in C; union!(N, G[v]); end; intersect!(N, S); setdiff!(N, C); N)

# Split S by clique separators found from a minimal triangulation of G[S].
function cliquesep_pieces(G, S::BitSet)
    vs = collect(S); loc = Dict(v => i for (i, v) in enumerate(vs)); k = length(vs)
    g = SimpleGraph(k)
    for (i, v) in enumerate(vs), u in G[v]
        j = get(loc, u, 0); j > i && add_edge!(g, i, j)
    end
    order, _ = permutation(g; alg = MCSM())
    # elimination game: madj of each vertex in H
    H = [BitSet(neighbors(g, i)) for i in 1:k]; done = falses(k); madj = Vector{BitSet}(undef, k)
    for v in order
        N = BitSet(u for u in H[v] if !done[u]); madj[v] = N
        for u in N
            union!(H[u], N); delete!(H[u], u)
        end
        done[v] = true
    end
    alive = copy(S); pieces = BitSet[]
    for v in order
        vv = vs[v]; vv in alive || continue
        Sv = BitSet(vs[u] for u in madj[v]); intersect!(Sv, alive)
        isempty(Sv) && continue
        isclique(G, Sv) || continue
        cs = components(G, alive, Sv)
        length(cs) < 2 && continue
        C = cs[findfirst(c -> vv in c, cs)]
        push!(pieces, union(C, Sv)); setdiff!(alive, C)
    end
    push!(pieces, alive)
    return pieces
end

# Rule 4: a minimal separator with exactly one missing edge; returns (sep, (a, b)) or nothing.
function one_missing_sep(G, S::BitSet)
    for u in S
        Nu = intersect(G[u], S); push!(Nu, u)
        for C in components(G, S, Nu)
            sep = nbhd(G, C, S)
            mp = missing_pairs(G, sep; limit = 1)
            length(mp) == 1 && return sep, mp[1]
        end
    end
    return nothing
end

# Returns (atoms, forced, G): atoms as sorted vertex lists of the graph G (input + forced edges).
function preprocess(adj::Vector{BitSet}; rule4 = true, verbose = false, maxrule4 = 4000)
    G = deepcopy(adj); n = length(G); forced = Tuple{Int, Int}[]
    queue = [BitSet(1:n)]; atoms = Vector{Int}[]
    while !isempty(queue)
        S = pop!(queue)
        changed = true
        while changed
            changed = false
            for v in collect(S)
                if isclique(G, intersect(G[v], S))
                    delete!(S, v); changed = true
                end
            end
        end
        length(S) <= 3 && continue
        cs = components(G, S)
        length(cs) > 1 && (append!(queue, cs); continue)
        ps = cliquesep_pieces(G, S)
        length(ps) > 1 && (append!(queue, ps); continue)
        if rule4 && length(S) <= maxrule4
            r = one_missing_sep(G, S)
            if r !== nothing
                sep, (a, b) = r
                push!(G[a], b); push!(G[b], a); push!(forced, minmax(a, b))
                for C in components(G, S, sep)
                    push!(queue, union(C, sep))
                end
                continue
            end
        end
        push!(atoms, sort!(collect(S)))
    end
    verbose && println("prep: n=$n atoms=$(length(atoms)) sizes=$(sort(length.(atoms); rev = true)[1:min(end, 10)]) forced=$(length(forced))")
    return atoms, forced, G
end

# Local graph of an atom: adjacency lists over 1..k.
function atom_graph(G, verts::Vector{Int})
    loc = Dict(v => i for (i, v) in enumerate(verts))
    return [sort!([loc[u] for u in G[v] if haskey(loc, u)]) for v in verts]
end

# Fill edges of an elimination order (local adjacency lists).
function order_fill(adjl::Vector{Vector{Int}}, order)
    k = length(adjl); H = [BitSet(a) for a in adjl]; done = falses(k); fill = Tuple{Int, Int}[]
    for v in order
        N = [u for u in H[v] if !done[u]]
        for i in eachindex(N), j in (i + 1):length(N)
            a, b = N[i], N[j]
            if !(b in H[a])
                push!(H[a], b); push!(H[b], a); push!(fill, minmax(a, b))
            end
        end
        done[v] = true
    end
    return fill
end

# The best of a few minimal-triangulation heuristics: (fill edges, order).
function heuristic_ub(adjl::Vector{Vector{Int}})
    k = length(adjl); g = SimpleGraph(k)
    for v in 1:k, u in adjl[v]
        u > v && add_edge!(g, v, u)
    end
    best = nothing
    for alg in (MinimalChordal(MF()), MinimalChordal(MMD()), MinimalChordal(AMF()), MCSM())
        o, _ = permutation(g; alg)
        f = order_fill(adjl, o)
        (best === nothing || length(f) < length(best[1])) && (best = (f, collect(o)))
    end
    return best
end
