# CPU-only: references and timings of every captured exact round (sepenum + sepdp), 1 and N threads.
# usage: julia -t N cpu_ref.jl datadir out.jls
include(joinpath(@__DIR__, "exact_common.jl"))
rs = loadrounds(ARGS[1]); NT = Threads.nthreads()
cpu_round(rs[1], 1); cpu_round(rs[1], NT)
t1 = zeros(length(rs)); tN = zeros(length(rs)); refs = Vector{Any}(undef, length(rs))
for (i, r) in enumerate(rs)
    t1[i] = @elapsed refs[i] = cpu_round(r, 1)
end
for (i, r) in enumerate(rs)
    tN[i] = @elapsed cpu_round(r, NT)
end
serialize(ARGS[2], (; t1, tN, refs, NT, host = gethostname(), cpu = Sys.cpu_info()[1].model))
@printf("%d rounds: CPU 1 thread %.2f s, %d threads %.2f s (%s)\n", length(rs), sum(t1), NT, sum(tN), Sys.cpu_info()[1].model)
