using CliqueTrees, Graphs, Serialization, Random
include(joinpath(@__DIR__, "..", "src", "pidref.jl"))
D = deserialize(ARGS[1]); a = D.a; K = D.K; d = D.d
pg = PGraph(a.adj, a.weights); w(s) = wt(pg, s)
nI = length(d.I); nO = length(d.O); rng = Xoshiro(7)
pi = sortperm(d.I; lt = bs_less); po = sortperm(d.ON; lt = bs_less)
It = [pi[i:min(i + 31, nI)] for i in 1:32:nI]
sampT = rand(rng, 1:length(It), 40)
summ(idx) = (reduce(bs_or, d.O[idx]), reduce(bs_and, d.O[idx]), reduce(bs_or, d.ON[idx]), minimum(d.Ow[idx]))
tilepass(tC, tN, s) = bs_subset(tC, s[1]) && max(w(bs_andnot(tN, s[3])), w(bs_and(tN, s[2]))) + s[4] <= K + 1e-9
brute = length(It) * nO / 32
for TO in (4, 8, 16), S in (0, 64, 256)
    S != 0 && S <= TO && continue
    tiles = [po[i:min(i + TO - 1, nO)] for i in 1:TO:nO]; ts = [summ(t) for t in tiles]
    sup = S == 0 ? nothing : [((i - 1) * (S ÷ TO) + 1):min(i * (S ÷ TO), length(tiles)) for i in 1:cld(length(tiles), S ÷ TO)]
    ss = S == 0 ? nothing : [summ(vcat(tiles[r]...)) for r in sup]
    W1 = W2 = W3 = W4 = 0.0
    for t in sampT
        idx = It[t]; tC = reduce(bs_and, d.I[idx]); tN = reduce(bs_and, d.IN[idx])
        cand = if S == 0
            W2 += length(tiles) / 32; 1:length(tiles)
        else
            W1 += length(sup) / 32
            c = Int[]
            for (k, r) in enumerate(sup)
                tilepass(tC, tN, ss[k]) || continue
                W2 += length(r) / 32; append!(c, r)
            end
            c
        end
        for j in cand
            tilepass(tC, tN, ts[j]) || continue
            W3 += 1
            any(i -> bs_subset(d.I[i], ts[j][1]) && max(w(bs_andnot(d.IN[i], ts[j][3])), w(bs_and(d.IN[i], ts[j][2]))) + ts[j][4] <= K + 1e-9, idx) && (W4 += length(tiles[j]))
        end
    end
    sc = length(It) / length(sampT)
    tot = (W1 + W2 + W3 + W4) * sc
    println("TO=$TO S=$S: W1=$(round(W1*sc, sigdigits=3)) W2=$(round(W2*sc, sigdigits=3)) W3=$(round(W3*sc, sigdigits=3)) W4=$(round(W4*sc, sigdigits=3)) total=$(round(tot, sigdigits=3)) vs brute $(round(brute, sigdigits=3)): $(round(brute/tot, digits=1))x")
end
