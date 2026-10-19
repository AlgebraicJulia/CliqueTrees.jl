# Certify HBT widths with one GPU PID-BT decision just below them.
#   julia --project=env run_certify.jl <out.tsv> <tlimit> <cpu.tsv with hbt rows> @names | names...
using CliqueTrees, Graphs, Serialization, Printf
const CT = CliqueTrees
include(joinpath(@__DIR__, "..", "src", "gpupid.jl"))
ATOMS = joinpath(@__DIR__, "..", "atoms")
out = ARGS[1]; tl = parse(Float64, ARGS[2]); ORDER = Symbol(get(ENV, "GPUPID_ORDER", "rbfs"))
ub = Dict{String, Float64}()
for l in eachline(ARGS[3])
    f = split(l, '\t'); length(f) >= 7 && f[4] == "hbt" && (ub[f[1]] = parse(Float64, f[7]))
end
names = startswith(ARGS[4], "@") ? readlines(ARGS[4][2:end]) : ARGS[4:end]
names = filter(k -> haskey(ub, k), names)
done = isfile(out) ? Set(first(split(l, '\t')) for l in eachline(out) if !isempty(l)) : Set{String}()
names = filter(k -> !(k in done), names)
sticky(e) = (m = sprint(showerror, e); occursin("ILLEGAL", m) || occursin("KernelException", m) || occursin("launch failure", m) || occursin("ERROR_", m) && !occursin("OUT_OF_MEMORY", m))
for W in sort!(unique([cld(deserialize(joinpath(ATOMS, nm * ".jls")).n, 64) for nm in names]))
    n = 64 * (W - 1) + 10; g = cycle_graph(n); add_edge!(g, 1, n ÷ 2)
    gpu_certify(PGraph([sort(neighbors(g, v)) for v in 1:n], [isodd(v) ? 1.0 : 2.0 for v in 1:n]), 3.0)
end
io = open(out, "a")
for name in names
    a = deserialize(joinpath(ATOMS, name * ".jls"))
    g = SimpleGraph(a.n); for v in 1:a.n, u in a.adj[v]; u > v && add_edge!(g, v, u); end
    perm = vorder(a.adj, ORDER); pg = relabel(a.adj, a.weights, perm); u = ub[name]
    t0 = time()
    r = try
        gpu_certify(pg, u; tlimit = tl)
    catch e
        println(io, join((name, a.n, "ERROR", replace(sprint(showerror, e)[1:min(end, 150)], '\n' => ' ')), '\t')); flush(io); sticky(e) && exit(3); GC.gc(true); continue
    end
    t = time() - t0
    s = r.stats; w2 = isempty(r.order) ? NaN : treewidth(a.weights, g, perm[r.order])
    @printf(io, "%s\t%d\t%d\t%s\t%.4f\t%.10g\t%.10g\t%.10g\tI=%d;O=%d;P=%d;pairs=%d;rounds=%d\n", name, a.n, ne(g), r.status, t, u, r.lb, w2,
        s[:I], s[:O], s[:P], s[:pairs], s[:rounds])
    flush(io)
    println(name, " ", r.status, " ", round(t, digits = 2)); flush(stdout)
    r = nothing; GC.gc(true); CUDA.reclaim()
end
close(io)
