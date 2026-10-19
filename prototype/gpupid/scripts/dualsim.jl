using CliqueTrees, Graphs, Serialization, Random
include(joinpath(@__DIR__, "..", "src", "pidref.jl"))
D = deserialize(ARGS[1]); a = D.a; K = D.K; d = D.d
pg = PGraph(a.adj, a.weights); w(s) = wt(pg, s)
nI = length(d.I); nO = length(d.O)
rng = Xoshiro(2)
po = sortperm(d.ON; lt = bs_less)
for (iname, pi) in (("I-by-NC", sortperm(d.IN; lt = bs_less)), ("I-by-C", sortperm(d.I; lt = bs_less)), ("I-random", randperm(rng, nI)))
    for (TI, TO) in ((32, 32), (64, 32), (32, 8))
        It = [pi[i:min(i + TI - 1, nI)] for i in 1:TI:nI]; Ot = [po[i:min(i + TO - 1, nO)] for i in 1:TO:nO]
        Ot_or = [reduce(bs_or, d.O[t]) for t in Ot]; Ot_and = [reduce(bs_and, d.O[t]) for t in Ot]
        Ot_nor = [reduce(bs_or, d.ON[t]) for t in Ot]; Ot_m = [maximum(K .- d.Ow[t]) for t in Ot]
        sampI = rand(rng, 1:length(It), 60)
        tp = 0; it = 0; pr = 0
        for ii in sampI
            t = It[ii]
            Cand = reduce(bs_and, d.I[t]); NCand = reduce(bs_and, d.IN[t])
            for j in eachindex(Ot)
                tp += 1
                bs_subset(Cand, Ot_or[j]) || continue
                max(w(bs_andnot(NCand, Ot_nor[j])), w(bs_and(NCand, Ot_and[j]))) <= Ot_m[j] + 1e-9 || continue
                for i in t
                    it += 1
                    bs_subset(d.I[i], Ot_or[j]) || continue
                    max(w(bs_andnot(d.IN[i], Ot_nor[j])), w(bs_and(d.IN[i], Ot_and[j]))) <= Ot_m[j] + 1e-9 || continue
                    pr += length(Ot[j])
                end
            end
        end
        per = (tp + it + pr) / (length(sampI) * TI)
        println("$iname TI=$TI TO=$TO: work per I-block $(round(per)) (tile-pairs $(round(tp/(length(sampI)*TI))), I-vs-tile $(round(it/(length(sampI)*TI))), pairs $(round(pr/(length(sampI)*TI)))) speedup $(round(nO/per, digits=1))x")
    end
end
