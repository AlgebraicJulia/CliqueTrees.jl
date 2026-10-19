using CliqueTrees, Graphs, Serialization, Random
include(joinpath(@__DIR__, "..", "src", "pidref.jl"))
D = deserialize(ARGS[1]); a = D.a; K = D.K; d = D.d
pg = PGraph(a.adj, a.weights); W = length(d.I[1])
w(s) = wt(pg, s)
rng = Xoshiro(1)
nI = length(d.I); nO = length(d.O)
samp = rand(rng, 1:nI, 3000)
# exact pair survivors for the sample
function exact(samp)
    c = 0
    for i in samp, b in 1:nO
        C = d.I[i]; A = d.O[b]
        bs_subset(C, A) || continue
        d.Ow[b] + w(bs_and(d.IN[i], A)) <= K + 1e-9 && (c += 1)
    end
    c
end
ex = exact(samp)
println("nI=$nI nO=$nO exact survivors per sampled I-block: $(ex / length(samp))  (pair density $(ex / length(samp) / nO))")
# vertex lists for lexicographic order of a set
vl(s) = [v for v in 1:a.n if bs_has(s, v)]
for (oname, perm) in (("NA-lex", sortperm(d.ON; by = vl)), ("NA-words", sortperm(d.ON; lt = bs_less)), ("A-lex", sortperm(d.O; by = vl)), ("NA-minmax", sortperm(1:nO; by = b -> (bs_min(d.ON[b]), bs_count(d.ON[b]), vl(d.ON[b])))))
    for T in (32, 256)
        tiles = [perm[i:min(i + T - 1, nO)] for i in 1:T:nO]
        NAor = [reduce(bs_or, d.ON[t]) for t in tiles]
        Aor = [reduce(bs_or, d.O[t]) for t in tiles]
        mmax = [maximum(K .- d.Ow[t]) for t in tiles]
        surv = 0
        for i in samp
            C = d.I[i]; NC = d.IN[i]
            for j in eachindex(tiles)
                bs_subset(C, Aor[j]) || continue
                # N(C) ∩ A = N(C) \ N(A); lower bound over the tile: N(C) \ NAor
                w(bs_andnot(NC, NAor[j])) <= mmax[j] + 1e-9 || continue
                surv += 1
            end
        end
        frac = surv / (length(samp) * length(tiles))
        # work: tile tests + T pair tests per surviving tile
        work = length(tiles) + surv / length(samp) * T
        println("$oname T=$T: surviving tiles $(round(100frac, digits=2))%  work/I-block $(round(work)) vs $nO  speedup $(round(nO / work, digits=1))x")
    end
end
