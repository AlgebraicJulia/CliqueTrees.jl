include(joinpath(@__DIR__, "test_sepenum.jl"))   # loads calls, warms up, checks a sample
# CPU: re-time every round offline (single thread), and verify all
bgs = [BipartiteGraph(c.g) for c in calls]
CliqueTrees.sepenum(calls[1].w, bgs[1], calls[1].k, Inf, 1)
tc = zeros(length(calls))
refs = Vector{Vector{UInt64}}(undef, length(calls))
for (j, c) in enumerate(calls)
    tc[j] = @elapsed refs[j] = cpu_sepenum_lists(bgs[j], c.w, c.k)
end
@printf("CPU sepenum (1 thread): %d rounds, %.3f s total, %.1f us median, %d separators total\n", length(calls), sum(tc), 1e6 * sort(tc)[cld(end, 2)], sum(length, refs))
for (lab, idx) in (("all", collect(eachindex(calls))), ("heaviest 10% (CPU time)", sortperm(tc; rev = true)[1:cld(end, 10)]), ("light 90%", sortperm(tc; rev = true)[cld(end, 10)+1:end]))
    sub = calls[idx]
    local b = prepare([c.g for c in sub], [c.w for c in sub], [c.k for c in sub])
    gpu_sepenum(b)   # warm
    ts = Float64[]; local lists, ovf
    for rep in 1:3
        tprep = @elapsed b = prepare([c.g for c in sub], [c.w for c in sub], [c.k for c in sub])
        te = @elapsed (lists, ovf, tk) = gpu_sepenum(b)
        push!(ts, tk)
        rep == 3 && @printf("%-26s %5d rounds: CPU %.3f s | GPU kernel %.3f s (end-to-end incl. prep %.3f + sort/download %.3f s) | kernel speedup %.1fx\n",
            lab, length(sub), sum(tc[idx]), minimum(ts), tprep, te - tk, sum(tc[idx]) / minimum(ts))
    end
    bad = count(j -> !ovf[j] && lists[j] != refs[idx[j]], eachindex(sub))
    println("   mismatches $bad, overflows $(count(ovf))")
end
