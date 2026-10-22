# julia --project=env scripts/run_gpu.jl <out.tsv> <tlimit> <instance paths...>
include(joinpath(@__DIR__, "..", "src", "gpufill.jl"))
include(joinpath(@__DIR__, "..", "src", "solve.jl"))
out = ARGS[1]; tl = parse(Float64, ARGS[2]); files = ARGS[3:end]
# warm-up (compile time not charged)
let p = tempname() * ".graph"
    open(p, "w") do io; for (a, b) in [(1,2),(2,3),(3,4),(4,5),(5,6),(6,1),(1,4),(2,7),(7,8),(8,5)]; println(io, a, " ", b); end; end
    solve_instance(p, gpu_decide_fill; tlimit = 10.0); rm(p)
end
warmup(gpu_decide_fill)
println("warm"); flush(stdout)
io = open(out, "a")
for f in files
    name = basename(f)
    r = try
        solve_instance(f, gpu_decide_fill; tlimit = tl, outpath = joinpath(dirname(out), "sol_gpu", name * ".fill"))
    catch e
        println(io, join((name, "ERROR", replace(sprint(showerror, e)[1:min(end, 200)], '\n' => ' ')), '\t')); flush(io); continue
    end
    println(io, join((name, r.n, r.m, r.atoms, r.maxatom, r.forced, r.status, r.k, round(r.t; digits = 2), round(r.tprep; digits = 2), r.runs, r.check.ok), '\t')); flush(io)
    println(name, " ", r.status, " k=", r.k, " t=", round(r.t; digits = 2), " atoms=", r.atoms, " max=", r.maxatom); flush(stdout)
end
