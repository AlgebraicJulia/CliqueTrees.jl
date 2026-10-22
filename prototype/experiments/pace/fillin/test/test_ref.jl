# CPU reference vs brute force on random graphs: (a) the PID decision on whole graphs (no
# preprocessing), at K = OPT (yes, opt) and K = OPT - 1 (no); (b) the whole pipeline with preprocessing.
using Random, Graphs
include(joinpath(@__DIR__, "..", "src", "fillref.jl"))
include(joinpath(@__DIR__, "..", "src", "solve.jl"))
rng = Xoshiro(parse(Int, get(ARGS, 1, "1"))); ntr = parse(Int, get(ARGS, 2, "300")); nmax = parse(Int, get(ARGS, 3, "13"))
bad = 0; nt = 0
for trial in 1:ntr
    n = rand(rng, 5:nmax); p = rand(rng, [0.25, 0.35, 0.5, 0.7])
    g = erdos_renyi(n, p; rng); is_connected(g) || continue
    adjl = [sort(neighbors(g, v)) for v in 1:n]
    opt = brute_minfill(adjl)
    global nt += 1
    # (a) decisions on the whole graph
    perm = vorder(adjl, :rbfs); pg = relabel(adjl, ones(n), perm)
    if opt > 0
        a = cpu_decide(pg, opt); b = cpu_decide(pg, opt - 1)
        oka = a.yes && a.opt == opt && !b.yes && b.nextK >= opt && a.stats[:late] == 0 && b.stats[:late] == 0
        # also with a loose bound: still optimal (cost order)
        c = cpu_decide(pg, opt + 5); okc = c.yes && c.opt == opt
        if !(oka && okc)
            println("DECIDE MISMATCH trial=$trial n=$n opt=$opt a=$(a.yes),$(a.opt) b=$(b.yes),$(b.nextK) c=$(c.yes),$(c.opt) late=$(a.stats[:late])")
            global bad += 1
        end
    end
    # (b) pipeline (preprocessing + atoms + K schedule), checked solution
    path = tempname() * ".graph"
    open(path, "w") do io
        for e in edges(g); println(io, src(e), " ", dst(e)); end
    end
    r = solve_instance(path, cpu_decide; tlimit = 60.0)
    if !(r.status == :found && r.k == opt && r.check.ok)
        println("PIPELINE MISMATCH trial=$trial n=$n opt=$opt got=$(r.k) status=$(r.status) check=$(r.check)")
        global bad += 1
    end
    rm(path)
end
println("tests=$nt bad=$bad")
