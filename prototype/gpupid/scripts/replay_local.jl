# Replay captured HBT exact local calls on the GPU with the same (lo, hi) and time budget.
#   julia --project=env replay_local.jl <out.tsv> <calls.jls>...
using CliqueTrees, Graphs, Serialization, Printf
include(joinpath(@__DIR__, "..", "src", "gpupid.jl"))
out = ARGS[1]
for W in 1:2
    n = 64 * (W - 1) + 10; g = cycle_graph(n); add_edge!(g, 1, n ÷ 2)
    gpu_tw(PGraph([sort(neighbors(g, v)) for v in 1:n], [isodd(v) ? 1.0 : 2.0 for v in 1:n]), 2.0)
end
io = open(out, "a")
for f in ARGS[2:end]
    calls = deserialize(f); tag = splitext(basename(f))[1]
    for (i, c) in enumerate(calls)
        (c.n > 128 || c.n < 3) && continue
        adjl = Vector{Vector{Int}}(c.adj)
        perm = vorder(adjl, :rbfs); pg = relabel(adjl, c.weights, perm)
        t0 = time()
        r = gpu_tw(pg, max(c.lo, maximum(c.weights)); hi = c.hi, tlimit = max(c.budget, 0.01), minstep = length(pg.wcls) > 2 ? 0.25 : 0.0)
        t = time() - t0
        @printf(io, "%s\t%d\t%d\t%d\t%s\t%.5f\t%.12g\t%s\t%.5f\t%.12g\n", tag, i, c.n, length(pg.wcls), c.status, c.t, c.k, r.status, t, r.status == :found ? r.width : NaN)
    end
    flush(io)
end
close(io)
