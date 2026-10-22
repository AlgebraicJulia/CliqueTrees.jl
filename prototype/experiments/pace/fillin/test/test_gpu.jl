# GPU vs CPU reference: same answer, nextK and per-round log (t, |I|, |O|, |P|, pending) at
# K in (OPT - 1, OPT, OPT + 5) on random graphs (brute force for OPT), then on PACE atoms
# (args: seed ntrials nmax [instance files...]).
using Random, Graphs
include(joinpath(@__DIR__, "..", "src", "gpufill.jl"))
include(joinpath(@__DIR__, "..", "src", "solve.jl"))
rng = Xoshiro(parse(Int, get(ARGS, 1, "1"))); ntr = parse(Int, get(ARGS, 2, "100")); nmax = parse(Int, get(ARGS, 3, "13"))
bad = 0; nt = 0
function cmp(pg, K, tag)
    a = cpu_decide(pg, K); b = gpu_decide_fill(pg, K)
    ok = a.yes == b.yes && a.opt == b.opt && a.nextK == b.nextK && a.log == b.log && a.stats[:late] == 0 && b.stats[:late] == 0
    if !ok
        println("MISMATCH $tag K=$K cpu=($(a.yes),$(a.opt),$(a.nextK)) gpu=($(b.yes),$(b.opt),$(b.nextK)) late=$(a.stats[:late]),$(b.stats[:late])")
        println("  cpu log: ", a.log[1:min(end, 8)]); println("  gpu log: ", b.log[1:min(end, 8)])
    end
    return ok, a
end
for trial in 1:ntr
    n = rand(rng, 5:nmax); p = rand(rng, [0.25, 0.35, 0.5, 0.7])
    g = erdos_renyi(n, p; rng); is_connected(g) || continue
    adjl = [sort(neighbors(g, v)) for v in 1:n]
    opt = brute_minfill(adjl); opt == 0 && continue
    perm = vorder(adjl, :rbfs); pg = relabel(adjl, ones(n), perm)
    global nt += 1
    for K in (opt - 1, opt, opt + 5)
        ok, a = cmp(pg, K, "rand trial=$trial n=$n opt=$opt")
        ok &= (K >= opt) == a.yes && (!a.yes || a.opt == opt)
        ok || (global bad += 1)
    end
end
println("random: tests=$nt bad=$bad"); flush(stdout)
# PACE atoms: decisions at K = heuristic UB - 1 and a smaller K, rounds compared
for f in ARGS[4:end]
    labels, adj = read_graph(f)
    atoms, forced, G = preprocess(adj)
    for verts in atoms
        length(verts) > 128 && continue
        adjl = atom_graph(G, verts); ub = length(heuristic_ub(adjl)[1])
        perm = vorder(adjl, :rbfs); pg = relabel(adjl, ones(length(adjl)), perm)
        for K in unique((max(1, ub ÷ 2), ub - 1))
            t0 = time(); ok, a = cmp(pg, K, "$(basename(f)) atom n=$(length(verts)) ub=$ub")
            global nt += 1; ok || (global bad += 1)
            println("$(basename(f)) atom=$(length(verts)) K=$K yes=$(a.yes) opt=$(a.opt) rounds=$(length(a.log)) ok=$ok ($(round(time() - t0; digits = 1)) s)"); flush(stdout)
        end
    end
end
println("total: tests=$nt bad=$bad")
