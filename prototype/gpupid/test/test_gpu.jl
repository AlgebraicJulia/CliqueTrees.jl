using CliqueTrees, Graphs, Random, Serialization
const CT = CliqueTrees
include(joinpath(@__DIR__, "..", "src", "gpupid.jl"))
function randconn(rng, n, p)
    while true
        g = erdos_renyi(n, p; rng); is_connected(g) && return g
    end
end
rng = Xoshiro(parse(Int, get(ARGS, 1, "7"))); nfail = 0; ntest = 0
ntr = parse(Int, get(ARGS, 2, "300")); nmax = parse(Int, get(ARGS, 3, "30")); nmin = parse(Int, get(ARGS, 4, "5"))
for trial in 1:ntr
    n = rand(rng, nmin:nmax); p = min(0.9, rand(rng, [2.0, 3.0, 4.0, 5.0]) / (n - 1))
    g = randconn(rng, n, p)
    adjl = [sort(neighbors(g, v)) for v in 1:n]
    w = trial % 2 == 0 ? ones(n) : Float64.(rand(rng, [1.0, 1.0, log2(3), 2.0, log2(5)], n))
    pg = PGraph(adjl, w)
    lo = CT.bt_bas(w, g)
    println("trial $trial n=$n p=$p"); flush(stdout)
    rg = trial % 3 == 0 ? gpu_tw(pg, lo; tlimit = 30) : trial % 3 == 1 ? gpu_tw(pg, lo; tlimit = 30, minstep = 0.25) : gpu_tw(pg, lo; tlimit = 30, search = :bisect, init = permutation(w, g, MF())[1])
    rg.status == :timeout && (println("TIMEOUT trial=$trial"); serialize("hang_$trial.jls", (adjl, w)); continue)
    _, _, kct, st = CT.septw(w, g, lo, Inf)
    tw = treewidth(w, g, rg.order)
    global ntest += 1
    # decision-level agreement with the reference at the optimum and just below
    agree = true
    if trial % 10 == 0
        for K in (kct, kct - 1e-6)
            a = pid_decide(pg, K); b = gpu_decide(GPUGraph(pg), K); c = gpu_decide(GPUGraph(pg), K; join = :brute)
            agree &= (b.yes == c.yes) && (b.yes || c.nextK >= b.nextK - 1e-9)
            agree &= (a.yes == b.yes) && (a.yes || abs(a.nextK - b.nextK) < 1e-9)
        end
    end
    if abs(tw - kct) > 1e-9 || abs(rg.width - kct) > 1e-9 || !agree
        global nfail += 1
        println("FAIL trial=$trial n=$n p=$p gpu=$(rg.width) order=$tw septw=$kct agree=$agree")
    end
end
println("tests=$ntest failures=$nfail")
