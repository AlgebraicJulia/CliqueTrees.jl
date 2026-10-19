# GPU PID decisions on saved atoms: dedup per join chunk vs per round. Same blocks (I, O in order,
# IN, ON, Ow), answer and next width; peak bytes of the search and of the device.
#   julia --project=<env> pid_mem.jl <atomdir> <seconds> <modes> <name[:K]> ...
# K: the width to decide (default: just below the atom's upper bound, the certificate question;
# mf: just below the width of MF, the first question of the PID rounds).
# modes: comma list of chunk,round (each run in a fresh process state: GC + reclaim before).
using CliqueTrees, CUDA, Graphs, Serialization, Printf
const EXT = Base.get_extension(CliqueTrees, :CUDAExt)
const P = EXT.PIDGPU

dir, T, modes = ARGS[1], parse(Float64, ARGS[2]), Symbol.(split(ARGS[3], ','))

function pgraph(a)
    adj = [Int[x for x in a.adj[v] if x != v] for v in 1:a.n]
    perm = P.vorder(adj, :rbfs)
    return P.GPUGraph(P.relabel(adj, Float64.(a.weights), perm))
end

# compile
let a = (; n = 10, adj = [[mod1(v - 1, 10), mod1(v + 1, 10)] for v in 1:10], weights = ones(10))
    gg = pgraph(a)
    for m in modes
        P.gpu_decide(gg, 2.0; tol = 0.0, dedup = m); P.gpu_decide(gg, 1.0; tol = 0.0, dedup = m)
    end
end

for arg in ARGS[4:end]
    name, ks = occursin(':', arg) ? split(arg, ':') : (arg, "")
    a = deserialize(joinpath(dir, name * ".jls")); gg = pgraph(a)
    K = if isempty(ks)
        Float64(CliqueTrees.rtw_below(a.ub))
    elseif ks == "mf"   # the first question of the PID rounds: below the width of MF
        g = SimpleGraph(a.n); foreach(v -> foreach(u -> u > v && add_edge!(g, v, u), a.adj[v]), 1:a.n); bg = BipartiteGraph(g)
        Float64(CliqueTrees.rtw_below(treewidth(a.weights, bg; alg = MF())))
    else
        parse(Float64, ks)
    end
    res = Dict{Symbol, Any}()
    for m in modes
        GC.gc(true); CUDA.reclaim()
        println("== $name n=$(a.n) W=$(cld(a.n, 64)) K=$K dedup=$m"); flush(stdout)
        t = @elapsed r = try
            P.gpu_decide(gg, K; tol = 0.0, dedup = m, deadline = time() + T, verbose = true, dump = true)
        catch err
            err isa CUDA.OutOfGPUMemoryError || rethrow()
            println("OOM: ", sprint(showerror, err)[1:min(end, 200)]); nothing
        end
        if r === nothing
            @printf("%s %s OOM t=%.2f\n", name, m, t)
        else
            s = r.stats
            @printf("%s %s yes=%s next=%.10g timeout=%s rounds=%d I=%d O=%d P=%d t=%.2f peak=%.3f GiB device=%.3f GiB\n", name, m, r.yes, r.nextK,
                get(s, :timeout, false), s[:rounds], s[:I], s[:O], s[:P], t, s[:peak] / 2^30, s[:peakdev] / 2^30)
            res[m] = r
        end
        flush(stdout)
    end
    if haskey(res, :chunk) && haskey(res, :round) && !get(res[:round].stats, :timeout, false) && !get(res[:chunk].stats, :timeout, false)
        x, y = res[:chunk], res[:round]
        same = x.yes == y.yes && x.nextK == y.nextK && x.stats[:dump] == y.stats[:dump] && x.stats[:rounds] == y.stats[:rounds] &&
            x.stats[:P] == y.stats[:P] && sort(x.P) == sort(y.P) && x.I == y.I
        println("$name identical: $same"); flush(stdout)
    end
end
