# CPU baselines on one atom, single thread.
#   julia -t 1 cpu_base.jl <atomdir> <name> <mode: septw|hbt> <seconds> <out.tsv>
using CliqueTrees, Graphs, Serialization, Printf
const CT = CliqueTrees
dir, name, mode, T, out = ARGS[1], ARGS[2], ARGS[3], parse(Float64, ARGS[4]), ARGS[5]
a = deserialize(joinpath(dir, name * ".jls"))
g = SimpleGraph(a.n); for v in 1:a.n, u in a.adj[v]; u > v && add_edge!(g, v, u); end
w = a.weights
# warm-up on a small graph so compile time is not charged
let h = cycle_graph(10); CT.septw(ones(10), h, 2.0, Inf; nt = 1); permutation(ones(10), h, HBT(; time = 0.2, threads = 1)); end
if mode == "septw"
    t0 = time(); r = CT.septw(w, g, a.lb, Inf; mt = T, nt = 1); t = time() - t0
    k = r[4] == CT.BT_FOUND ? treewidth(w, g, r[1]) : NaN
    line = @sprintf("%s\t%d\t%d\tseptw\t%s\t%.4f\t%.10g\t%.10g", name, a.n, ne(g), string(r[4]), t, k, r[3])
else
    t0 = time(); o, _ = permutation(w, g, HBT(; time = T, threads = 1)); t = time() - t0
    line = @sprintf("%s\t%d\t%d\thbt\tDONE\t%.4f\t%.10g\t%.10g", name, a.n, ne(g), t, treewidth(w, g, o), a.lb)
end
open(out, "a") do io; println(io, line); end
