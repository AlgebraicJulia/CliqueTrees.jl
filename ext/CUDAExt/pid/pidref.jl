# CPU reference of weighted PID-BT (Tamaki, "Positive-instance driven dynamic programming
# for treewidth", ESA 2017 / J. Comb. Optim. 2019), in the level-synchronous (semi-naive)
# form that the GPU version runs. Decision: is there a tree decomposition all of whose bags
# weigh at most K? Unit weights: K = tw + 1.
#
# Objects (Tamaki's notation, sizes replaced by weights):
#   I-block: inbound full component C of N(C), feasible.       key: C
#   O-block: outbound full component A of N(A), with N(A) the union of N(C) over feasible
#            I-blocks C (and w(N(A)) + wmin(A) <= K).            key: A
#   PMC:     potential maximal clique Ω, w(Ω) <= K, buildable;  feasible when every component
#            in support(Ω) is a feasible I-block.                key: Ω
# A set C is outbound if it is the full component of N(C) with the smallest vertex.
#
# Rules (all monotone, so any application order reaches the same closure):
#   R0  N[v] is a PMC with w <= K                          -> PMC
#   R1  I-block C                                          -> O-block: outbound full comp of N(C)
#   R2  I-block C ⊆ O-block B, X = N(C) ∪ N(B), w(X) <= K   -> PMC X, or O-block (outbound full comp of X)
#   R3  O-block A, v ∈ N(A), X = N(A) ∪ (N(v) ∩ A), w <= K  -> PMC X
#   R4  PMC Ω whose support ⊆ I                             -> I-block crib(outlet(Ω), Ω), or YES if outlet = ∅
# Semi-naive rounds: round r joins ΔI × (O_old ∪ ΔO) and I_old × ΔO, so every (I, O) pair
# is joined exactly once.

include("bitsets.jl")

struct PGraph{W}
    n::Int
    adj::Vector{BS{W}}
    all::BS{W}
    masks::Vector{BS{W}}     # weight classes
    wcls::Vector{Float64}
    weights::Vector{Float64}
end

function PGraph(adjl::Vector{Vector{Int}}, weights::Vector{Float64})
    n = length(adjl); W = max(1, cld(n, 64)); V = Val(W)
    adj = [reduce(bs_or, (bs_bit(V, u) for u in adjl[v] if u != v); init = bs_zero(V)) for v in 1:n]
    all = reduce(bs_or, (bs_bit(V, v) for v in 1:n); init = bs_zero(V))
    wcls = sort!(unique(weights))
    masks = [reduce(bs_or, (bs_bit(V, v) for v in 1:n if weights[v] == c); init = bs_zero(V)) for c in wcls]
    return PGraph{W}(n, adj, all, masks, wcls, weights)
end

wt(g::PGraph, s) = bs_weight(s, g.masks, g.wcls, length(g.wcls))
wmn(g::PGraph, s) = bs_wmin(s, g.masks, g.wcls, length(g.wcls))

# The components of G - X, each with its neighborhood (⊆ X).
function comps(g::PGraph{W}, X::BS{W}) where {W}
    rest = bs_andnot(g.all, X); out = Tuple{BS{W}, BS{W}}[]
    while !bs_isempty(rest)
        v = bs_min(rest)
        D = bs_comp(g.adj, rest, v)
        rest = bs_andnot(rest, D)
        push!(out, (D, bs_open(g.adj, D)))
    end
    return out
end

# The outbound full component of X (the full component with the smallest vertex), or nothing.
function outbound_full(cs, X)
    for (D, N) in cs   # comps come in increasing order of their smallest vertex
        N === X && return D
    end
    return nothing
end

# Is X (with no full component) cliquish?
function cliquish(g::PGraph{W}, X::BS{W}, cs) where {W}
    for v in 1:g.n
        bs_has(X, v) || continue
        cover = bs_or(g.adj[v], bs_bit(Val(W), v))
        for (D, N) in cs
            bs_has(N, v) && (cover = bs_or(cover, N))
        end
        bs_subset(X, cover) || return false
    end
    return true
end

is_pmc(g, X, cs) = isnothing(outbound_full(cs, X)) && cliquish(g, X, cs)

# outlet(Ω) and support(Ω) for a PMC Ω with components cs.
function outlet_support(g::PGraph{W}, Ω::BS{W}, cs) where {W}
    outlet = nothing; best = -1
    for (j, (D, N)) in enumerate(cs)
        # the other full components of N: comps with the same neighborhood, and crib(N, Ω)
        m = bs_min(D); ob = true
        cm = bs_min(bs_andnot(Ω, N))
        cm < m && (ob = false)
        for (i, (D2, N2)) in enumerate(cs)
            i == j && continue
            if N2 === N || !bs_subset(N2, N)
                bs_min(D2) < m && (ob = false; break)
            end
        end
        if ob && bs_count(N) > best
            best = bs_count(N); outlet = N
        end
    end
    supp = outlet === nothing ? [D for (D, N) in cs] : [D for (D, N) in cs if !bs_subset(N, outlet)]
    return outlet, supp
end

mutable struct PIDResult{W}
    yes::Bool
    nextK::Float64          # if no: every width below nextK is impossible
    root::Int               # index of the root PMC
    I::Vector{BS{W}}; Ibp::Vector{Int}
    O::Vector{BS{W}}
    P::Vector{BS{W}}
    stats::Dict{Symbol, Int}
end

mutable struct PIDState{W}
    g::PGraph{W}
    K::Float64; tol::Float64; nextK::Float64
    I::Vector{BS{W}}; Ibp::Vector{Int}; Iset::Dict{BS{W}, Int}
    O::Vector{BS{W}}; Oset::Dict{BS{W}, Int}
    P::Vector{BS{W}}; Pset::Dict{BS{W}, Int}
    Pout::Vector{Union{Nothing, BS{W}}}; Psupp::Vector{Vector{BS{W}}}
    newI::Vector{Int}; newO::Vector{Int}; newP::Vector{Int}
    st::Dict{Symbol, Int}
end

@inline okw(s::PIDState, x) = x <= s.K + s.tol
@inline rej!(s::PIDState, x) = (s.nextK = min(s.nextK, x); nothing)

function addO!(s::PIDState, A)
    haskey(s.Oset, A) && return
    push!(s.O, A); s.Oset[A] = length(s.O); push!(s.newO, length(s.O)); return
end
function addP!(s::PIDState, X, cs)
    haskey(s.Pset, X) && return
    push!(s.P, X); s.Pset[X] = length(s.P); push!(s.newP, length(s.P))
    o, sp = outlet_support(s.g, X, cs); push!(s.Pout, o); push!(s.Psupp, sp); return
end
function tryO!(s::PIDState, X, A)
    x = wt(s.g, X) + wmn(s.g, A)
    okw(s, x) ? addO!(s, A) : rej!(s, x)
end
# R2/R3 outcome for a candidate X with w(X) <= K: a PMC, or an O-block.
function tryX!(s::PIDState, X; oblock = true)
    s.st[:pmc_tests] += 1
    cs = comps(s.g, X)
    A = outbound_full(cs, X)
    if A === nothing
        cliquish(s.g, X, cs) && addP!(s, X, cs)
    elseif oblock
        tryO!(s, X, A)
    end
    return
end
function join!(s::PIDState, c::Int, b::Int)
    s.st[:joins] += 1
    C = s.I[c]; B = s.O[b]
    bs_subset(C, B) || return
    s.st[:subset] += 1
    X = bs_or(bs_open(s.g.adj, C), bs_open(s.g.adj, B)); x = wt(s.g, X)
    okw(s, x) || return rej!(s, x)
    tryX!(s, X)
end

# The decision at bag weight K. tol absorbs float noise in comparisons.
function pid_decide(g::PGraph{W}, K::Float64; tol = 1e-9, maxrounds = typemax(Int), verbose = false) where {W}
    V = Val(W)
    s = PIDState{W}(g, K, tol, Inf, BS{W}[], Int[], Dict{BS{W}, Int}(), BS{W}[], Dict{BS{W}, Int}(),
        BS{W}[], Dict{BS{W}, Int}(), Union{Nothing, BS{W}}[], Vector{BS{W}}[], Int[], Int[], Int[],
        Dict(:joins => 0, :subset => 0, :rounds => 0, :pmc_tests => 0))
    pending = Int[]; root = 0; st = s.st
    # R0
    for v in 1:g.n
        X = bs_or(g.adj[v], bs_bit(V, v)); x = wt(g, X)
        if okw(s, x)
            cs = comps(g, X)
            is_pmc(g, X, cs) && addP!(s, X, cs)
        else
            rej!(s, x)
        end
    end
    nIold = 0; nOold = 0   # I[1:nIold], O[1:nOold] are fully joined
    dI = Int[]; dO = Int[]
    while true
        # R4 on the new and pending PMCs
        append!(pending, s.newP); empty!(s.newP)
        still = Int[]
        for p in pending
            if all(D -> haskey(s.Iset, D), s.Psupp[p])
                o = s.Pout[p]
                if o === nothing
                    root = p; break
                end
                C = bs_andnot(s.P[p], o)
                for D in s.Psupp[p]
                    C = bs_or(C, D)
                end
                if !haskey(s.Iset, C)
                    push!(s.I, C); push!(s.Ibp, p); s.Iset[C] = length(s.I); push!(s.newI, length(s.I))
                end
            else
                push!(still, p)
            end
        end
        pending = still
        root > 0 && break
        empty!(dI); append!(dI, s.newI); empty!(s.newI)
        empty!(dO); append!(dO, s.newO); empty!(s.newO)
        (isempty(dI) && isempty(dO)) && break
        st[:rounds] += 1
        st[:rounds] > maxrounds && break
        # R1
        for c in dI
            S = bs_open(g.adj, s.I[c]); cs = comps(g, S)
            A = outbound_full(cs, S)
            A === nothing || tryO!(s, S, A)
        end
        # R2: ΔI × O[1:end of ΔO], I_old × ΔO
        nOcur = isempty(dO) ? nOold : last(dO)
        for c in dI, b in 1:nOcur
            join!(s, c, b)
        end
        for c in 1:nIold, b in dO
            join!(s, c, b)
        end
        # R3 on ΔO
        for b in dO
            A = s.O[b]; S = bs_open(g.adj, A)
            for v in 1:g.n
                bs_has(S, v) || continue
                X = bs_or(S, bs_and(g.adj[v], A)); x = wt(g, X)
                okw(s, x) ? tryX!(s, X; oblock = false) : rej!(s, x)
            end
        end
        nIold = isempty(dI) ? nIold : last(dI); nOold = nOcur
        verbose && println("round $(st[:rounds]): |I|=$(length(s.I)) |O|=$(length(s.O)) |P|=$(length(s.P)) pending=$(length(pending)) joins=$(st[:joins])")
    end
    st[:I] = length(s.I); st[:O] = length(s.O); st[:P] = length(s.P)
    return PIDResult{W}(root > 0, s.nextK, root, s.I, s.Ibp, s.O, s.P, st)
end

# An elimination ordering from a YES result: children before parents, root bag last.
function pid_order(g::PGraph{W}, res::PIDResult{W}) where {W}
    Iset = Dict(C => i for (i, C) in enumerate(res.I))
    order = Int[]; placed = falses(g.n)
    place!(S) = for v in 1:g.n
        if bs_has(S, v) && !placed[v]
            placed[v] = true; push!(order, v)
        end
    end
    # iterative post-order over (PMC, parent separator) pairs
    function visit(Ω, outlet, comps_)
        for D in comps_
            c = Iset[D]; Ω2 = res.P[res.Ibp[c]]
            cs = comps(g, Ω2); o, s = outlet_support(g, Ω2, cs)
            visit(Ω2, o, s)
            # the vertices of this block's bag that are not in its separator
            place!(bs_andnot(Ω2, bs_open(g.adj, D)))
        end
    end
    Ω = res.P[res.root]; cs = comps(g, Ω)
    visit(Ω, nothing, [D for (D, N) in cs])
    place!(Ω)
    @assert length(order) == g.n
    return order
end

# The weighted treewidth by iterating K upward from lo (valid lower bound).
function pid_tw(g::PGraph, lo::Float64; hi = Inf, verbose = false, tlimit = Inf)
    K = lo; t0 = time(); runs = 0
    while K <= hi
        runs += 1
        res = pid_decide(g, K)
        verbose && println("K=$K yes=$(res.yes) next=$(res.nextK) stats=$(res.stats)")
        res.yes && return (; width = K, order = pid_order(g, res), runs, res)
        isfinite(res.nextK) || return (; width = Inf, order = Int[], runs, res)
        K = res.nextK
        time() - t0 > tlimit && return (; width = NaN, order = Int[], runs, res)
    end
    return (; width = Inf, order = Int[], runs, res = nothing)
end

# Vertex orders for the solver. The PID orientation (outbound = holds the smallest vertex) and the
# tile coherence both depend on the numbering. Returns perm with new vertex i = old vertex perm[i].
function vorder(adjl::Vector{Vector{Int}}, kind::Symbol)
    n = length(adjl)
    kind == :id && return collect(1:n)
    deg(v) = length(adjl[v])
    function bfs(s)
        order = [s]; seen = falses(n); seen[s] = true; h = 1
        while h <= length(order)
            u = order[h]; h += 1
            for x in sort(adjl[u]; by = deg)
                seen[x] || (seen[x] = true; push!(order, x))
            end
        end
        for v in 1:n   # disconnected leftovers
            seen[v] || (seen[v] = true; push!(order, v))
        end
        return order
    end
    far(s) = last(bfs(s))
    o = bfs(far(far(1)))
    kind == :bfs && return o
    kind == :rbfs && return reverse(o)
    error("unknown order $kind")
end
function relabel(adjl::Vector{Vector{Int}}, weights::Vector{Float64}, perm::Vector{Int})
    inv = invperm(perm)
    adj = [sort([inv[u] for u in adjl[perm[i]]]) for i in eachindex(perm)]
    return PGraph(adj, weights[perm])
end
