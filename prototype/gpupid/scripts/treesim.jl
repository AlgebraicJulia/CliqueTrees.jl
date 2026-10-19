using CliqueTrees, Graphs, Serialization, Random
include(joinpath(@__DIR__, "..", "src", "pidref.jl"))
D = deserialize(ARGS[1]); a = D.a; K = D.K; d = D.d
pg = PGraph(a.adj, a.weights)
w(s) = wt(pg, s)
rng = Xoshiro(1); nI = length(d.I); nO = length(d.O)
samp = rand(rng, 1:nI, 1000)
vl(s) = [v for v in 1:a.n if bs_has(s, v)]
for (oname, perm) in (("A-lex", sortperm(d.O; by = vl)), ("NA-lex", sortperm(d.ON; by = vl)), ("NA-words", sortperm(d.ON; lt = bs_less)))
    for B in (2, 4, 8, 32)
        # levels of a B-ary tree over the sorted array: level l has nodes of span B^l
        levels = Any[]; span = 1
        while span < nO
            span *= B
            nodes = [perm[i:min(i + span - 1, nO)] for i in 1:span:nO]
            push!(levels, (span, [reduce(bs_or, d.O[t]) for t in nodes], [reduce(bs_and, d.O[t]) for t in nodes],
                [reduce(bs_or, d.ON[t]) for t in nodes], [maximum(K .- d.Ow[t]) for t in nodes]))
        end
        visits = 0; found = 0
        for i in samp
            C = d.I[i]; NC = d.IN[i]
            front = [1]  # node ids at the top level
            for l in length(levels):-1:1
                span, Aor, Aand, NAor, mmax = levels[l]
                next = Int[]
                for j in front
                    visits += 1
                    bs_subset(C, Aor[j]) || continue
                    lbw = max(w(bs_andnot(NC, NAor[j])), w(bs_and(NC, Aand[j])))
                    lbw <= mmax[j] + 1e-9 || continue
                    if l == 1
                        # leaves: exact tests
                        lo = (j - 1) * span + 1
                        for q in lo:min(lo + span - 1, nO)
                            b = perm[q]; visits += 1
                            bs_subset(C, d.O[b]) && d.Ow[b] + w(bs_and(NC, d.O[b])) <= K + 1e-9 && (found += 1)
                        end
                    else
                        for c in ((j - 1) * B + 1):min(j * B, length(levels[l - 1][2]))
                            push!(next, c)
                        end
                    end
                end
                front = next
            end
        end
        println("$oname B=$B: visits/I-block $(round(visits / length(samp))) (all-pairs $nO, speedup $(round(nO / (visits / length(samp)), digits = 1))x), found/I $(found / length(samp))")
    end
end
