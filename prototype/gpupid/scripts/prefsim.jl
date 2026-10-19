using CliqueTrees, Graphs, Serialization, Random
include(joinpath(@__DIR__, "..", "src", "pidref.jl"))
D = deserialize(ARGS[1]); a = D.a; K = D.K; d = D.d
pg = PGraph(a.adj, a.weights); w(s) = wt(pg, s)
nI = length(d.I); nO = length(d.O); n = a.n
marg = K .- d.Ow
mvals = sort(unique(round.(marg, digits = 6)))
println("margins: ", [(m, sum(abs.(marg .- m) .< 1e-6)) for m in mvals])
# per margin class: frequency of each vertex in N(A)
vl(s) = [v for v in 1:n if bs_has(s, v)]
NAl = [vl(x) for x in d.ON]
rng = Xoshiro(3); samp = rand(rng, 1:nI, 2000)
total = 0.0; totalU = 0.0
for m in mvals
    cls = findall(abs.(marg .- m) .< 1e-6)
    freq = zeros(Int, n); for b in cls, v in NAl[b]; freq[v] += 1; end
    rank = invperm(sortperm(freq))         # rarest first
    lists = [Int[] for _ in 1:n]; for b in cls, v in NAl[b]; push!(lists[v], b); end
    cand = 0; candU = 0
    for i in samp
        NC = sort(vl(d.IN[i]); by = v -> rank[v])
        # prefix: N(C) vertices in rank order until their weight exceeds m (need at least one hit)
        pre = Int[]; acc = 0.0
        for v in NC
            push!(pre, v); acc += a.weights[v]
            acc > m + 1e-9 && break
        end
        acc > m + 1e-9 || (cand += length(cls); candU += length(cls); continue)
        cand += sum(length(lists[v]) for v in pre)
        candU += length(union((lists[v] for v in pre)...))
    end
    println("margin $m: $(length(cls)) O-blocks, candidates/I-block $(round(cand/length(samp))) (distinct $(round(candU/length(samp))))")
    global total += cand / length(samp); global totalU += candU / length(samp)
end
println("total candidates/I-block $(round(total)) (distinct $(round(totalU))) vs all-pairs $nO: $(round(nO/total, digits=1))x")
