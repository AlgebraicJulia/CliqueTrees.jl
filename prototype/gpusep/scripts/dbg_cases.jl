using CliqueTrees, CUDA, Graphs, Random
using CliqueTrees: CPUSeptw, GPUSeptw, septw, septwbackend!
ext = Base.get_extension(CliqueTrees, :CUDAExt)
function graphs(rng, n, p, m)
    out = BipartiteGraph{Int, Int}[]
    while length(out) < m
        g = erdos_renyi(n, p; rng); is_connected(g) && push!(out, BipartiteGraph(g))
    end
    return out
end
weights(rng, kind, n) = kind === :unit ? ones(Int, n) : kind === :int ? rand(rng, 1:4, n) : log2.(rand(rng, 2:6, n))
run(b, w, g, lo, hi) = (old = septwbackend!(b); try septw(w, g, lo, hi, Inf, 1) finally septwbackend!(old) end)
rng = MersenneTwister(7)
for (n, p) in ((12, 0.4), (24, 0.25), (40, 0.12), (90, 0.035)), kind in (:unit, :int, :float), g in graphs(rng, n, p, 2)
    w = weights(rng, kind, n)
    tc = @elapsed c = run(CPUSeptw(), w, g, 0, typemax(eltype(w)))
    tg = @elapsed r = run(GPUSeptw(; budget = 0.0), w, g, 0, typemax(eltype(w)))
    tp = @elapsed q = run(GPUSeptw(; engine = :pid, budget = 0.0), w, g, 0, typemax(eltype(w)))
    println("n=$n $kind ne=$(ne(g)) cpu $(c[3]) $(round(tc;digits=2)) | septw $(r == c) $(round(tg;digits=2)) | pid $(q[3]) $(round(tp;digits=2))"); flush(stdout)
end
