using CliqueTrees, CUDA, Graphs, Random
using CliqueTrees: CPUSeptw, GPUSeptw, septw, septwbackend!
run(b, w, g, lo, hi) = (old = septwbackend!(b); try septw(w, g, lo, hi, Inf, 1) finally septwbackend!(old) end)
rng = MersenneTwister(7)
for kind in (:unit, :int, :float), r in 1:2
    g = Graphs.grid([5, 14]); for _ in 1:3; add_edge!(g, rand(rng, 1:70), rand(rng, 1:70)); end; g = BipartiteGraph(g)
    n = 70; w = kind === :unit ? ones(Int, n) : kind === :int ? rand(rng, 1:4, n) : log2.(rand(rng, 2:6, n))
    tc = @elapsed c = run(CPUSeptw(), w, g, 0, typemax(eltype(w)))
    tg = @elapsed r = run(GPUSeptw(; budget = 0.0), w, g, 0, typemax(eltype(w)))
    tp = @elapsed q = run(GPUSeptw(; engine = :pid, budget = 0.0), w, g, 0, typemax(eltype(w)))
    println("grid $kind cpu $(c[3]) $(round(tc;digits=2)) | septw $(r == c) $(round(tg;digits=2)) | pid $(q[3]) $(round(tp;digits=2))"); flush(stdout)
end
