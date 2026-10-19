include(joinpath(@__DIR__, "test_sepenum.jl"))
bgs = [BipartiteGraph(c.g) for c in calls]
tc = zeros(length(calls)); refs = Vector{Vector{UInt64}}(undef, length(calls))
for (j, c) in enumerate(calls)
    tc[j] = @elapsed refs[j] = cpu_sepenum_lists(bgs[j], c.w, c.k)
end
@printf("CPU sepenum (1 thread): %d rounds, %.3f s, %d separators\n", length(calls), sum(tc), sum(length, refs))
heavy = sortperm(tc; rev = true)[1:cld(end, 10)]
for (lab, idx) in (("all", collect(eachindex(calls))), ("heaviest 10%", heavy))
    sub = calls[idx]
    b = prepare([c.g for c in sub], [c.w for c in sub], [c.k for c in sub])
    for split in (0, 2, 3)
        gpu_sepenum(b; split)
        best = Inf; local lists, ovf, info
        for rep in 1:2
            te = @elapsed (lists, ovf, tk, info) = gpu_sepenum(b; split)
            best = min(best, tk)
        end
        bad = count(j -> !ovf[j] && lists[j] != refs[idx[j]], eachindex(sub))
        @printf("%-13s split=%d: GPU kernels %.3f s (phase1 %.3f, tasks %d -> %.3f) vs CPU %.3f s = %.1fx | mismatches %d overflows %d\n",
            lab, split, best, info.t1, info.ntask, info.t2, sum(tc[idx]), sum(tc[idx]) / best, bad, count(ovf))
        flush(stdout)
    end
end
