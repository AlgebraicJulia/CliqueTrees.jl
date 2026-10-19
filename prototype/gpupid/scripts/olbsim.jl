using CliqueTrees, Graphs, Serialization, Random
include(joinpath(@__DIR__, "..", "src", "pidref.jl"))
D = deserialize(ARGS[1]); a = D.a; K = D.K; d = D.d
pg = PGraph(a.adj, a.weights); w(s) = wt(pg, s); n = a.n
nO = length(d.O); rng = Xoshiro(5); samp = rand(rng, 1:nO, min(nO, parse(Int, get(ARGS, 2, "3000"))))
vl(s) = [v for v in 1:n if bs_has(s, v)]
# G<A>: G[S ∪ A] with S a clique
function gA(A, S)
    vs = vcat(vl(S), vl(A)); idx = Dict(v => i for (i, v) in enumerate(vs))
    h = SimpleGraph(length(vs))
    for v in vs, u in a.adj[v]
        haskey(idx, u) && idx[u] > idx[v] && add_edge!(h, idx[v], idx[u])
    end
    ns = bs_count(S)
    for i in 1:ns, j in i+1:ns; add_edge!(h, i, j); end
    return h, a.weights[vs]
end
# greedy: connected P ⊆ R covering all of S, grown by BFS preferring vertices that cover new S
function carve(R, S)
    need = S; best = 0; bestcov = -1
    for v in vl(R)
        c = bs_count(bs_and(pg.adj[v], need)); c > bestcov && (bestcov = c; best = v)
    end
    best == 0 && return nothing
    P = bs_bit(Val(length(S)), best); need = bs_andnot(need, pg.adj[best])
    while !bs_isempty(need)
        fr = bs_andnot(bs_and(bs_nbrs(pg.adj, P), R), P)
        bs_isempty(fr) && return nothing
        # the frontier vertex covering most of need, ties to the smallest
        bv = 0; bc = -1
        for v in vl(fr)
            c = bs_count(bs_and(pg.adj[v], need)); c > bc && (bc = c; bv = v)
        end
        P = bs_or(P, bs_bit(Val(length(S)), bv)); need = bs_andnot(need, pg.adj[bv])
    end
    return P
end
function twoparts(A, S)
    P = carve(A, S); P === nothing && return false
    rest = bs_andnot(A, P)
    for D in [x[1] for x in comps_in(rest)]
        bs_subset(S, bs_open(pg.adj, D)) && return true
    end
    return false
end
function comps_in(R)
    out = Tuple{BS{length(R)}, Int}[]; rest = R
    while !bs_isempty(rest)
        v = bs_min(rest); C = bs_comp(pg.adj, rest, v); rest = bs_andnot(rest, C); push!(out, (C, 0))
    end
    out
end
cnt = Dict{Float64, Vector{Int}}()   # margin -> [total, mmw prunable, twoparts prunable]
for b in samp
    A = d.O[b]; S = d.ON[b]; m = round(K - d.Ow[b], digits = 4)
    c = get!(cnt, m, [0, 0, 0]); c[1] += 1
    h, wh = gA(A, S)
    lb = lowerbound(wh, h, MMW())
    lb > K + 1e-9 && (c[2] += 1)
    (w(S) + 2 * minimum(a.weights[vl(A)]) > K + 1e-9 && twoparts(A, S)) && (c[3] += 1)
end
for m in sort(collect(keys(cnt)))
    c = cnt[m]; println("margin $m: $(c[1]) sampled, MMW(G<A>) > K: $(round(100c[2]/c[1], digits=1))%, two-parts minor: $(round(100c[3]/c[1], digits=1))%")
end
tot = reduce(+, values(cnt)); println("all: MMW $(round(100tot[2]/tot[1], digits=1))%, two-parts $(round(100tot[3]/tot[1], digits=1))%")
