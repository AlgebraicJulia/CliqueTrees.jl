# Exact minimum fill-in by positive-instance-driven Bouchitté–Todinca (PID-BT), CPU reference.
# Needs tn-work/gpupid/src/{bitsets,pidref,devfuns}.jl loaded (PGraph, comps, cliquish, dv_outlet ...).
#
# The search is gpupid's (Tamaki ESA 2017 rules R0-R4, level-synchronous, semi-naive rounds) with
# the bag weight w(X) replaced by fill(X) = number of non-adjacent pairs in X, which is also
# monotone under inclusion, so the generation side is unchanged:
#   R0/R2/R3 candidate X kept iff fill(X) <= K;
#   O-block A kept iff fill(N(A)) + min_{a in A} |N(A) \ N(a)| <= K (the PMC built on A contains
#     N(A) and some a in A).
# R4 is where the sum objective enters. A PMC Ω (outlet S, support D_1..D_m, all I-blocks) gives
# its target I-block C = (Ω \ S) ∪ D_1 ∪ ... ∪ D_m the cost
#   c(Ω) = fill(Ω) - fill(S) + sum_i cost(D_i),
# and a root PMC (no outlet) the total fill(Ω) + sum_i cost(D_i). Tamaki's Track B solver orders this
# Dijkstra-like: in phase t only targets with c <= t are admitted, each with its final optimal cost,
# and t moves to the least waiting c once the rounds reach a fixpoint. (A PMC whose support was
# admitted before phase t is generated before phase t: its generation chain uses only its support
# blocks and set-monotone tests. A violation would be counted in st[:late].)
# Pruning: fill(Ω) + sum_i cost(D_i) > K drops the PMC (every completion costs at least that).
# Decision: some root with total <= K? If yes, the first root found is optimal (cost order).
# If no: OPT > K and OPT >= the least rejected value (nextK).

@inline function bs_fill(adj, X::BS{W}) where {W}
    s = 0
    for i in 1:W
        word = X[i]
        while !iszero(word)
            v = 64 * (i - 1) + trailing_zeros(word) + 1; word &= word - one(UInt64)
            @inbounds s += bs_count(bs_andnot(X, adj[v]))
        end
    end
    return (s - bs_count(X)) >> 1
end
# min over a in A of |S \ N(a)| (S ∩ A empty, A nonempty)
@inline function bs_minmiss(adj, S::BS{W}, A::BS{W}) where {W}
    best = typemax(Int)
    for i in 1:W
        word = A[i]
        while !iszero(word)
            v = 64 * (i - 1) + trailing_zeros(word) + 1; word &= word - one(UInt64)
            @inbounds best = min(best, bs_count(bs_andnot(S, adj[v])))
        end
    end
    return best
end
@inline ofill(adj, S, A) = bs_fill(adj, S) + bs_minmiss(adj, S, A)

# Lower bound on the total of any solution through O-block A (S = N(A)), Tamaki's T-block
# relevance: the components D of G - (S ∪ A) are components of G - Ω for every PMC Ω built on A,
# so each adds its block cost: cost(D) if D is an admitted I-block; t (the phase) if D is of
# I-block type (a full component of N(D) that is not the outbound one) but not admitted yet
# (its cost is >= t, or no solution of total <= K uses it); 0 otherwise.
# look(D) = I-block index or 0. Returns (lb, lb0): lb0 omits the t terms and is what a rejection
# reports for nextK (a lower bound on every solution through A, not only on those of total <= K).
@inline function olb(adj, all::BS{W}, S::BS{W}, A::BS{W}, look, Icost, t) where {W}
    lb = ofill(adj, S, A); lb0 = lb
    rest = bs_andnot(all, bs_or(S, A))
    while !bs_isempty(rest)
        v = bs_min(rest)
        D = bs_comp(adj, rest, v)
        rest = bs_andnot(rest, D)
        j = look(D)
        if j > 0
            @inbounds lb += Icost[j]; @inbounds lb0 += Icost[j]
        else
            F = dv_full(adj, all, bs_open(adj, D))
            F === D || (lb += t)
        end
    end
    return lb, lb0
end

const JLB = Ref(get(ENV, "FILL_JLB", "1") == "1")
const OLB = Ref(get(ENV, "FILL_OLB", "1") == "1")

mutable struct FState{W}
    g::PGraph{W}
    K::Int; t::Int; minrej::Int
    I::Vector{BS{W}}; IN::Vector{BS{W}}; Icost::Vector{Int}; Ibp::Vector{Int}; Iset::Dict{BS{W}, Int}
    O::Vector{BS{W}}; ON::Vector{BS{W}}; Oset::Dict{BS{W}, Int}; Ooff::Vector{Int}; Ooff0::Vector{Int}
    P::Vector{BS{W}}; Pset::Dict{BS{W}, Int}
    Pfill::Vector{Int}; Pout::Vector{BS{W}}; Phas::Vector{Bool}; Psupp::Vector{Vector{BS{W}}}
    Pc::Vector{Int}        # -1: support not all admitted; else c (or root total)
    newO::Vector{Int}; newP::Vector{Int}
    st::Dict{Symbol, Any}
end

rej!(s::FState, x) = (s.minrej = min(s.minrej, x); nothing)

function addO!(s::FState, X, A)
    x, x0 = OLB[] ? olb(s.g.adj, s.g.all, X, A, D -> get(s.Iset, D, 0), s.Icost, s.t) : (f = ofill(s.g.adj, X, A); (f, f))
    x <= s.K || return rej!(s, x0)
    haskey(s.Oset, A) && return
    f = ofill(s.g.adj, X, A)
    push!(s.O, A); push!(s.ON, X); s.Oset[A] = length(s.O); push!(s.newO, length(s.O)); push!(s.Ooff, x - f); push!(s.Ooff0, x0 - f); return
end
function addP!(s::FState{W}, X, cs) where {W}
    haskey(s.Pset, X) && return
    push!(s.P, X); s.Pset[X] = length(s.P); push!(s.newP, length(s.P))
    o, has = dv_outlet(s.g.adj, s.g.all, X)
    push!(s.Pfill, bs_fill(s.g.adj, X)); push!(s.Pout, o); push!(s.Phas, has)
    push!(s.Psupp, [D for (D, N) in cs if !has || !bs_subset(N, o)]); push!(s.Pc, -1)
    return
end
function tryX!(s::FState, X; oblock = true)
    s.st[:pmc_tests] += 1
    cs = comps(s.g, X)
    A = outbound_full(cs, X)
    if A === nothing
        cliquish(s.g, X, cs) && addP!(s, X, cs)
    elseif oblock
        addO!(s, X, A)
    end
    return
end
function candX!(s::FState, X; oblock = true)
    x = bs_fill(s.g.adj, X)
    x <= s.K ? tryX!(s, X; oblock) : rej!(s, x)
end

struct FillResult{W}
    yes::Bool
    opt::Int               # if yes
    nextK::Int             # if no: OPT >= nextK (> K)
    root::Int
    s::FState{W}
end

function fill_decide(g::PGraph{W}, K::Int; deadline = Inf, verbose = false) where {W}
    V = Val(W); adj = g.adj
    s = FState{W}(g, K, 0, typemax(Int), BS{W}[], BS{W}[], Int[], Int[], Dict{BS{W}, Int}(), BS{W}[], BS{W}[], Dict{BS{W}, Int}(), Int[], Int[],
        BS{W}[], Dict{BS{W}, Int}(), Int[], BS{W}[], Bool[], Vector{BS{W}}[], Int[], Int[], Int[],
        Dict{Symbol, Any}(:joins => 0, :subset => 0, :rounds => 0, :pmc_tests => 0, :phases => 0, :late => 0, :log => Tuple{Int, Int, Int, Int, Int}[]))
    st = s.st
    # R0
    for v in 1:g.n
        candX!(s, bs_or(adj[v], bs_bit(V, v)); oblock = false)
    end
    pending = Int[]; root = 0; best = typemax(Int)
    nIold = 0; nOold = 0; nIidx = 0; nOidx = 0
    Imin = [Int[] for _ in 1:g.n]; Overt = [Int[] for _ in 1:g.n]
    while true
        time() > deadline && (st[:timeout] = true; break)
        # R4: evaluate pending PMCs; admit targets with c <= t (min cost per target)
        append!(pending, s.newP); empty!(s.newP)
        still = Int[]; cand = Dict{BS{W}, Tuple{Int, Int, BS{W}}}()
        for p in pending
            if s.Pc[p] < 0
                tot = s.Pfill[p]; ok = true
                for D in s.Psupp[p]
                    i = get(s.Iset, D, 0)
                    i == 0 && (ok = false; break)
                    tot += s.Icost[i]
                end
                if !ok
                    push!(still, p); continue
                end
                if tot > K
                    rej!(s, tot); continue           # dropped for good
                end
                s.Pc[p] = s.Phas[p] ? tot - bs_fill(adj, s.Pout[p]) : tot
            end
            c = s.Pc[p]
            if c > s.t
                push!(still, p); continue
            end
            if !s.Phas[p]
                c < best && (best = c; root = p)
                continue
            end
            C = bs_andnot(s.P[p], s.Pout[p])
            for D in s.Psupp[p]
                C = bs_or(C, D)
            end
            haskey(s.Iset, C) && continue
            old = get(cand, C, nothing)
            (old === nothing || (c, p) < (old[1], old[2])) && (cand[C] = (c, p, s.Pout[p]))
        end
        pending = still
        root > 0 && break
        for C in sort!(collect(keys(cand)); lt = bs_less)
            c, p, N = cand[C]
            c < s.t && (st[:late] += 1)
            push!(s.I, C); push!(s.IN, N); push!(s.Icost, c); push!(s.Ibp, p); s.Iset[C] = length(s.I)
        end
        dIlo = nIold + 1; dIhi = length(s.I); dOlo = nOold + 1; dOhi = length(s.O); empty!(s.newO)
        for c in nIidx+1:dIhi; push!(Imin[bs_min(s.I[c])], c); end; nIidx = dIhi
        for b in nOidx+1:dOhi, v in 1:g.n; bs_has(s.O[b], v) && push!(Overt[v], b); end; nOidx = dOhi
        if dIhi < dIlo && dOhi < dOlo
            # fixpoint at phase t: move to the least waiting cost
            tn = minimum((s.Pc[p] for p in pending if s.Pc[p] >= 0); init = typemax(Int))
            tn == typemax(Int) && break
            s.t = tn; st[:phases] += 1
            continue
        end
        st[:rounds] += 1
        # R1
        for c in dIlo:dIhi
            S = s.IN[c]; cs = comps(g, S)
            A = outbound_full(cs, S)
            A === nothing || addO!(s, S, A)
        end
        # R2: ΔI × O[1:dOhi], I_old × ΔO
        # (indexed: C ⊆ A needs every vertex of C in A, so scan the O-blocks containing the vertex
        # of C with the fewest, and the I-blocks whose least vertex is in A)
        for c in dIlo:dIhi
            C = s.I[c]; vb = 0; bl = typemax(Int)
            for v in 1:g.n
                bs_has(C, v) && length(Overt[v]) < bl && (bl = length(Overt[v]); vb = v)
            end
            for b in Overt[vb]
                b > dOhi && break
                join!(s, c, b)
            end
        end
        for b in dOlo:dOhi
            A = s.O[b]
            for v in 1:g.n
                bs_has(A, v) || continue
                for c in Imin[v]
                    c > nIold && break
                    join!(s, c, b)
                end
            end
        end
        # R3 on ΔO
        for b in dOlo:dOhi
            A = s.O[b]; S = s.ON[b]
            for v in 1:g.n
                bs_has(S, v) || continue
                candX!(s, bs_or(S, bs_and(adj[v], A)); oblock = false)
            end
        end
        nIold = dIhi; nOold = dOhi
        push!(st[:log], (s.t, length(s.I), length(s.O), length(s.P), length(pending)))
        verbose && println("round $(st[:rounds]) t=$(s.t): |I|=$(length(s.I)) |O|=$(length(s.O)) |P|=$(length(s.P)) pending=$(length(pending))")
    end
    st[:I] = length(s.I); st[:O] = length(s.O); st[:P] = length(s.P)
    root > 0 && return FillResult{W}(true, best, 0, root, s)
    return FillResult{W}(false, -1, max(K + 1, s.minrej), 0, s)
end

function join!(s::FState, c::Int, b::Int)
    s.st[:joins] += 1
    C = s.I[c]; B = s.O[b]
    bs_subset(C, B) || return
    s.st[:subset] += 1
    X = bs_or(s.IN[c], s.ON[b])
    f = bs_fill(s.g.adj, X)
    if JLB[]   # C and the components outside A ∪ N(A) are components of G - Ω for the PMC Ω ⊇ X
        f + s.Icost[c] + s.Ooff[b] <= s.K ? tryX!(s, X) : rej!(s, f + s.Icost[c] + s.Ooff0[b])
    else
        f <= s.K ? tryX!(s, X) : rej!(s, f)
    end
end

# The bags of the decomposition found (root PMC and, recursively, the PMC of each support block).
function fill_bags(g::PGraph{W}, P::Vector{BS{W}}, root::Int, Iset::Dict{BS{W}, Int}, Ibp::Vector{Int}) where {W}
    bags = BS{W}[]; stack = [root]
    while !isempty(stack)
        p = pop!(stack); Ω = P[p]; push!(bags, Ω)
        o, has = dv_outlet(g.adj, g.all, Ω)
        for (D, N) in comps(g, Ω)
            (!has || !bs_subset(N, o)) && push!(stack, Ibp[Iset[D]])
        end
    end
    return bags
end
function bags_fill(g::PGraph, bags)
    out = Set{Tuple{Int, Int}}()
    for Ω in bags
        vs = [v for v in 1:g.n if bs_has(Ω, v)]
        for i in eachindex(vs), j in (i + 1):length(vs)
            bs_has(g.adj[vs[i]], vs[j]) || push!(out, (vs[i], vs[j]))
        end
    end
    return sort!(collect(out))
end

# Exact minimum fill-in of a connected graph without clique separators (local adjacency lists).
# ub: (fill edges, order) of a heuristic. K schedule: K = lb, then K grows by a factor `grow`
# (at least to nextK), capped at UB - 1; a NO at UB - 1 proves UB optimal.
# decide(pg, K; deadline) -> (yes, opt, nextK, fill::Vector{Tuple{Int,Int}} in pg labels, stats)
function solve_atom(adjl::Vector{Vector{Int}}, decide; deadline = Inf, grow = 1.5, lb0 = 1, verbose = false, order = :rbfs, schedule = :up)
    ubf, _ = heuristic_ub(adjl); ub = length(ubf)
    ub == 0 && return (opt = 0, fill = ubf, status = :found, runs = 0, lb = 0)
    perm = vorder(adjl, order)
    pg = relabel(adjl, ones(length(adjl)), perm)
    lb = lb0; runs = 0; Kprev = 0
    while true
        lb >= ub && return (opt = ub, fill = ubf, status = :found, runs, lb)
        K = schedule == :ub ? ub - 1 : min(ub - 1, max(lb, ceil(Int, grow * Kprev)))
        runs += 1
        r = decide(pg, K; deadline)
        verbose && println("  K=$K yes=$(r.yes) opt=$(r.opt) next=$(r.nextK) ub=$ub stats=$(r.stats)")
        r.timeout && return (opt = -1, fill = ubf, status = :timeout, runs, lb, ub)
        if r.yes
            f = [minmax(perm[a], perm[b]) for (a, b) in r.fill]
            return (opt = r.opt, fill = f, status = :found, runs, lb = r.opt)
        end
        lb = max(lb, r.nextK)
        Kprev = K
    end
end

function cpu_decide(pg::PGraph, K::Int; deadline = Inf, verbose = false)
    r = fill_decide(pg, K; deadline, verbose)
    s = r.s; to = get(s.st, :timeout, false)
    stats = Dict(k => v for (k, v) in s.st if k != :log)
    if r.yes
        bags = fill_bags(pg, s.P, r.root, s.Iset, s.Ibp)
        f = bags_fill(pg, bags)
        length(f) == r.opt || error("reconstruction: $(length(f)) fill edges, cost $(r.opt)")
        return (yes = true, opt = r.opt, nextK = 0, fill = f, stats, timeout = false, log = s.st[:log])
    end
    return (yes = false, opt = -1, nextK = r.nextK, fill = Tuple{Int, Int}[], stats, timeout = to, log = s.st[:log])
end
