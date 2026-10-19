# Exact width of saved atoms: CPU septw, GPU-escalated septw (engines septw and pid), and RTW.
#   julia --project=<env> -t <nt> bench_atoms.jl <atomdir> <names file> <seconds> <out.tsv> [modes]
# modes: comma list of cpu,esc-septw,esc-pid,esc-<e>:<seconds>,gpu-septw,gpu-pid,rtw (default cpu,esc-septw,esc-pid,rtw)
#   cpu        CliqueTrees.septw, CPUSeptw backend, nt tasks
#   esc-<e>    GPUSeptw(engine = e), budget calibrated on the device (delta x GPU round cost)
#   gpu-<e>    GPUSeptw(engine = e, budget = 0): every round on the GPU
#   rtw        CliqueTrees.rtw (CPU backend), nt tasks
# Row: name n m mode status seconds width bound certified path rounds(cpu/gpu) tcpu tgpu budget
# certified = the width is proved optimal (septw/rtw BT_FOUND from a lower bound; rtw with certificate).
using CliqueTrees, CUDA, Graphs, Serialization, Printf
const CT = CliqueTrees
const EXT = Base.get_extension(CliqueTrees, :CUDAExt)

dir, list, T, out = ARGS[1], ARGS[2], parse(Float64, ARGS[3]), ARGS[4]
modes = length(ARGS) >= 5 ? split(ARGS[5], ',') : ["cpu", "esc-septw", "esc-pid", "rtw"]
names = filter(!isempty, strip.(readlines(list)))
NT = Threads.nthreads()

function loadatom(name)
    a = deserialize(joinpath(dir, name * ".jls"))
    g = SimpleGraph(a.n)
    for v in 1:a.n, u in a.adj[v]
        u > v && add_edge!(g, v, u)
    end
    return a, BipartiteGraph(g)
end

# esc-<e>:<s> = GPUSeptw(engine = e, budget = s seconds)
function backend(mode)
    (mode == "cpu" || mode == "rtw") && return CT.CPUSeptw()
    e, b = occursin(':', mode) ? split(mode[5:end], ':') : (mode[5:end], "")
    budget = startswith(mode, "gpu-") ? 0.0 : isempty(b) ? NaN : parse(Float64, b)
    return CT.GPUSeptw(; engine = Symbol(e), budget)
end

# warm-up: compile the CPU code and the device code for every set type that the atoms need
let h = BipartiteGraph(cycle_graph(10)), nmax = maximum(n -> loadatom(n)[1].n, names; init = 10)
    CT.septw(ones(10), h, 2.0, Inf; nt = NT); CT.rtw(ones(10), h, 2.0, Inf; mt = 1.0, nt = NT)
    for mode in modes
        b = backend(mode)
        b isa CT.GPUSeptw && CT.septwwarmup!(CT.GPUSeptw(; engine = b.engine); nmax)
    end
end
budget(b) = b isa CT.GPUSeptw ? (isnan(b.budget) ? b.delta * get(EXT.COST, b.engine, NaN) : b.budget) : NaN

io = open(out, "a")
for name in names
    a, g = loadatom(name); w = a.weights; lo = a.lb
    for mode in modes
        b = backend(mode); old = CT.septwbackend!(b); EXT.resetstats!(); GC.gc()
        local status, k, perm, cert
        t = @elapsed try
            if mode == "rtw"
                perm, _, k, status, c = CT.rtw(w, g, lo, Inf; mt = T, nt = NT)
                cert = status == CT.BT_FOUND && (k <= lo || !isempty(c))
            else
                perm, _, k, status = CT.septw(w, g, lo, Inf; mt = T, nt = NT)
                cert = status == CT.BT_FOUND
            end
        catch err
            status = "ERROR:" * first(replace(sprint(showerror, err), '\n' => ' '), 80); k = NaN; perm = Int[]; cert = false
        finally
            CT.septwbackend!(old)
        end
        width = status == CT.BT_FOUND || (mode == "rtw" && !isempty(perm) && status != CT.BT_ABOVE) ? treewidth(w, g; alg = perm) : NaN
        bound = status == CT.BT_FOUND ? (cert ? width : lo) : k
        S = EXT.STATS; ngpu = get(S, :gpufound, 0) + get(S, :gpuabove, 0) + get(S, :gputimeout, 0)
        path = get(S, :escalated, 0) == 0 ? "cpu" : get(S, :gpufailed, 0) > 0 ? "gpu-failed>cpu" : get(S, :busy, 0) > 0 ? "gpu+busy" : "gpu"
        mode == "rtw" && (path = "cpu")
        @printf(io, "%s\t%d\t%d\t%s\t%s\t%.3f\t%.10g\t%.10g\t%s\t%s\t%d/%d\t%.3f\t%.3f\t%.4g\n", name, a.n, ne(g), mode, string(status), t, width, bound,
            cert ? "y" : "n", path, get(S, :cpu, 0), ngpu, get(S, :tcpu, 0.0), get(S, :tgpu, 0.0), budget(b))
        flush(io)
        println(name, " ", mode, " ", status, " ", round(t; digits = 2), " s width=", width, " path=", path); flush(stdout)
    end
end
close(io)
