# GPU job: GPU-MF / GPU-MD timing + permutations. usage: julia gpu_fill.jl inst1 inst2 ...
using CUDA, Random
include(joinpath(@__DIR__, "fcommon.jl"))
include(joinpath(CODE, "Rounds.jl")); using .Rounds
include(joinpath(CODE, "GRounds.jl")); using .GRounds
const OUTT = joinpath(RES, "gpu_times.tsv")
let g = lat2(12)
    for sc in (:md, :mf, :mf1); GRounds.grounds(ones(nv(g)), g, RoundsAlg(; score = sc, relax = 0.1)); end
end
println(CUDA.name(CUDA.device())); flush(stdout)
for name in ARGS
    try
        g = loadg(name); n = nv(g); u = ones(n); reps = n > 4_000_000 ? 2 : 3
        for (m, sc) in ("GPU-MF" => :mf, "GPU-MF1" => :mf1, "GPU-MD" => :md)
            isfile(permfile(name, m)) && continue
            best = Inf; o = nothing
            for _ in 1:reps
                t = @elapsed(o = GRounds.grounds(u, g, RoundsAlg(; score = sc, relax = 0.1))[1]); best = min(best, t)
            end
            saveperm(name, m, o)
            appendline(OUTT, join((name, n, ne(g), m, best), '\t'))
            println(rpad(name, 14), rpad(m, 8), " n=", n, " t=", round(best, digits = 3)); flush(stdout)
            GC.gc(); CUDA.reclaim()
        end
    catch err
        println("FAILED $name: ", sprint(showerror, err)); flush(stdout)
        appendline(joinpath(RES, "status.tsv"), join((name, "GPU", "error:" * first(sprint(showerror, err), 80)), '\t'))
        GC.gc(); CUDA.reclaim()
    end
end
println("GPU DONE")
