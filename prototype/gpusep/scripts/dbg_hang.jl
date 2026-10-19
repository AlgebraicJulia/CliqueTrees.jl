using CliqueTrees, CUDA, Graphs, Serialization
const CT = CliqueTrees
ext = Base.get_extension(CliqueTrees, :CUDAExt)
a = deserialize(ARGS[1]); g = SimpleGraph(a.n); for v in 1:a.n, u in a.adj[v]; u > v && add_edge!(g, v, u); end; g = BipartiteGraph(g); w = a.weights
CT.septwwarmup!(CT.GPUSeptw(); nmax = a.n)
# the CPU rounds, to find the widths
k = CT.bt_bas(w, g); hk = treewidth(w, g; alg = MF())
for r in 1:40
    t = @elapsed (sep, _) = CT.sepenum(w, g, k, Inf, 8)
    t2 = @elapsed (perm, _, mink, st) = CT.sepdp(w, g, sep, k, Inf, 8)
    println("round $r k=$k nsep=$(nv(sep)) cpu enum $(round(t;digits=2)) dp $(round(t2;digits=2)) $st"); flush(stdout)
    if t + t2 > 0.5
        ts = time(); dl = ts + 5.0
        r1 = ext.septwround(w, g, k, dl, nothing)
        println("  gpu round (deadline 5 s): $(round(time() - ts; digits=2)) s -> ", r1 isa Tuple ? r1[2:3] : r1); flush(stdout)
        ts = time(); r2 = ext.septwround(w, g, k, ts + 5.0, sep)
        println("  gpu sepdp on cpu seps (deadline 5 s): $(round(time() - ts; digits=2)) s -> ", r2 isa Tuple ? r2[2:3] : r2); flush(stdout)
    end
    st == CT.BT_FOUND && break
    global k = CT.bt_next(w, g, k, mink)
end
