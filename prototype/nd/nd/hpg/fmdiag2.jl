# per-launch FM statistics on one graph (B200): n, parts, max moves of a part, time since previous FM launch end
const NDD = joinpath(@__DIR__, "..")
using CUDA, CliqueTrees, Graphs, SparseArrays, Serialization
include(joinpath(NDD, "ndutil.jl"))
include(joinpath(NDD, "GND.jl")); using .GND
let (p, a) = graph_csr(lat2(40)); for kw in ((;), (pregreedy = 3,)); gnd_perm(p, a; opts = GNDOpts(; Lb = 64, kw...), leafsolve = natural_leafsolve); end; end
for name in split(ENV["NAMES"], ",")
    ptr, adj = load_csr(String(name))
    for kw in ((;), (pregreedy = 3, pgmin = 20000))
        GND.FMDIAG[] = true; empty!(GND.FMLOG)
        tm = Dict{Symbol, Any}()
        t0 = time()
        gnd_perm(ptr, adj; opts = GNDOpts(; Lb = 4096, kw...), leafsolve = natural_leafsolve, timing = tm)
        L = GND.FMLOG
        println(name, " ", kw, " nd=", round(tm[:nd], digits = 3), " refine=", round(tm[:refine], digits = 3), " launches=", length(L))
        prev = t0 + tm[:upload] + tm[:twins]
        for x in L
            println("   n=", x[1], " parts=", x[2], " maxmoves=", x[3], " dt=", round((x[4] - prev) * 1000, digits = 2), " ms")
            prev = x[4]
        end
        flush(stdout)
    end
end
