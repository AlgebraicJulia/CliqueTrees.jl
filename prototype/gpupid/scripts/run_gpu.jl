# GPU PID-BT on corpus atoms.
#   julia --project=env run_gpu.jl <out.tsv> <tlimit s> <atom names...>   (or @file with names)
using CliqueTrees, Graphs, Serialization, Printf
const CT = CliqueTrees
include(joinpath(@__DIR__, "..", "src", "gpupid.jl"))
ATOMS = joinpath(@__DIR__, "..", "atoms")
function loadatom(name)
    a = deserialize(joinpath(ATOMS, name * ".jls"))
    g = SimpleGraph(a.n); for v in 1:a.n, u in a.adj[v]; u > v && add_edge!(g, v, u); end
    return a, g
end
out = ARGS[1]; tl = parse(Float64, ARGS[2]); SEARCH = Symbol(get(ENV, "GPUPID_SEARCH", "auto")); ORDER = Symbol(get(ENV, "GPUPID_ORDER", "rbfs"))
names = startswith(ARGS[3], "@") ? readlines(ARGS[3][2:end]) : ARGS[3:end]
done = isfile(out) ? Set(first(split(l, '\t')) for l in eachline(out) if !isempty(l)) : Set{String}()
names = filter(k -> !(k in done), names)
sticky(e) = (m = sprint(showerror, e); occursin("ILLEGAL", m) || occursin("KernelException", m) || occursin("launch failure", m) || occursin("ERROR_", m) && !occursin("OUT_OF_MEMORY", m))
# warm up compilation for every set width the list needs (compile time is not charged)
for W in sort!(unique([cld(deserialize(joinpath(ATOMS, nm * ".jls")).n, 64) for nm in names]))
    n = 64 * (W - 1) + 10
    g = cycle_graph(n); add_edge!(g, 1, n ÷ 2)
    pg = PGraph([sort(neighbors(g, v)) for v in 1:n], [isodd(v) ? 1.0 : 2.0 for v in 1:n]); gpu_tw(pg, 2.0)
    println("warm W=$W"); flush(stdout)
end
io = open(out, "a")
for name in names
    a, g = loadatom(name)
    perm = vorder(a.adj, ORDER); pg = relabel(a.adj, a.weights, perm)
    lo = max(a.lb, CT.bt_bas(a.weights, g))
    t0 = time()
    r = try
        if SEARCH == :bisect
            gpu_tw(pg, lo; tlimit = tl, search = :bisect, init = invperm(perm)[permutation(a.weights, g, MF())[1]])
        else
            gpu_tw(pg, lo; tlimit = tl, minstep = (SEARCH == :auto && length(pg.wcls) > 2) ? 0.25 : 0.0)
        end
    catch e
        println(io, join((name, a.n, "ERROR", replace(sprint(showerror, e)[1:min(end, 200)], '\n' => ' ')), '\t')); flush(io); sticky(e) && exit(3); GC.gc(true); continue
    end
    t = time() - t0
    ok = isempty(r.order) ? NaN : treewidth(a.weights, g, perm[r.order])
    s = isempty(r.stats) ? Dict(:I => 0, :O => 0, :P => 0, :pairs => 0, :rounds => 0, :t => Dict()) : r.stats[end][3]
    @printf(io, "%s\t%d\t%d\t%s\t%.4f\t%s\t%s\t%.6g\t%d\tI=%d;O=%d;P=%d;pairs=%d;rounds=%d\n", name, a.n, ne(g), r.status, t,
        string(r.width), string(ok), r.lb, r.runs, s[:I], s[:O], s[:P], s[:pairs], s[:rounds])
    flush(io)
    println(name, " ", r.status, " ", round(t, digits = 2), " width=", r.width, " check=", ok, " runs=", r.runs); flush(stdout)
    r = nothing; GC.gc(true); CUDA.reclaim()
end
close(io)
