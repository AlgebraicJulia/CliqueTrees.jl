# Full exact round (sepenum + sepdp at width k) on the captured HBT rounds:
# CPU CliqueTrees (1 and N threads) vs GPU (sepenum split tasks + sepdp block values).
using CUDA, CliqueTrees, Graphs, Serialization, Printf, Random
include(joinpath(@__DIR__, "SepEnumGPU.jl")); using .SepEnumGPU
include(joinpath(@__DIR__, "SepDPGPU.jl")); using .SepDPGPU
function loadatoms(files; nmax = 64)
    out = []
    for f in files, (nm, v) in deserialize(f), a in v.atoms
        (2 <= a.n <= nmax) || continue
        g = SimpleGraph(a.n); for (x, y) in a.edges; add_edge!(g, x, y); end
        for r in a.rounds
            r[2] >= 0 && push!(out, (; inst = nm, g, w = a.weights, k = r[1], nsep = r[2], tcpu = r[3]))
        end
    end
    return out
end
calls = loadatoms(filter(f -> startswith(basename(f), "atoms_") && endswith(f, ".jls"), readdir(@__DIR__; join = true)))
NT = Threads.nthreads()
gs = [BipartiteGraph(c.g) for c in calls]; ws = [c.w for c in calls]; ks = [c.k for c in calls]
function cpu_round(i, nt)
    sep, _ = CliqueTrees.sepenum(ws[i], gs[i], ks[i], Inf, nt)
    perm, invp, mink, st = CliqueTrees.sepdp(ws[i], gs[i], sep, ks[i], Inf, nt)
    return collect(perm), mink, st == CliqueTrees.BT_FOUND ? :found : :above
end
cachef = joinpath(@__DIR__, "cpu_ref_t$(NT).jls")
if isfile(cachef)
    t1, refs, tN = deserialize(cachef)
else
    cpu_round(1, 1); cpu_round(1, NT)
    refs = Vector{Any}(undef, length(calls)); t1 = zeros(length(calls))
    for i in eachindex(calls)
        t1[i] = @elapsed refs[i] = cpu_round(i, 1)
    end
    tN = [@elapsed(cpu_round(i, NT)) for i in eachindex(calls)]
    serialize(cachef, (t1, refs, tN))
end
@printf("CPU baselines: 1 thread %.2f s, %d threads %.2f s\n", sum(t1), NT, sum(tN)); flush(stdout)
heavy = sortperm(t1; rev = true)[1:cld(end, 10)]
function gpu_rounds(idx)
    b = prepare(gs[idx], ws[idx], ks[idx])
    te = @elapsed (lists, ovf, tk, info) = gpu_sepenum(b; split = 3)
    seps = [CliqueTrees.bt_out(lists[j], nv(gs[i]), 1) for (j, i) in enumerate(idx)]
    res, dinfo, _ = gpu_sepdp(ws[idx], gs[idx], seps, ks[idx])
    return res, (; tse_kernel = tk, tse_total = te, dinfo..., ovf = count(ovf))
end
gpu_rounds(heavy[1:20])
for (lab, idx) in (("all", collect(eachindex(calls))), ("heaviest 10%", heavy))
    tt = @elapsed (res, I) = gpu_rounds(idx)
    bad = count(j -> begin r = res[j]; c = refs[idx[j]]; (r[3], r[4]) != (c[2], c[3]) || (c[3] == :found && r[1] != c[1]) end, eachindex(idx))
    @printf("%-13s %5d rounds | CPU 1 thread %.2f s, %d threads %.2f s | GPU path %.2f s wall: sepenum kernels %.2f (+%.2f sort/glue), sepdp prep(CPU) %.2f, upload %.2f, GPU %.2f, order(CPU) %.2f | mismatches %d, overflows %d+%d\n",
        lab, length(idx), sum(t1[idx]), NT, sum(tN[idx]), tt, I.tse_kernel, I.tse_total - I.tse_kernel, I.tprep, I.tupload, I.tgpu, I.tfinish, bad, I.ovf, I.novf)
    @printf("    GPU-only work (sepenum + sepdp kernels) %.2f s: %.1fx vs CPU 1 thread, %.1fx vs CPU %d threads; whole GPU path: %.1fx / %.1fx\n",
        I.tse_kernel + I.tgpu, sum(t1[idx]) / (I.tse_kernel + I.tgpu), sum(tN[idx]) / (I.tse_kernel + I.tgpu), NT, sum(t1[idx]) / tt, sum(tN[idx]) / tt)
    flush(stdout)
end
