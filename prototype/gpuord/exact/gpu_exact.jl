# GPU: every captured exact round, batched by set width, verified against the CPU references.
# usage: julia gpu_exact.jl datadir cpuref.jls
using CUDA
include(joinpath(@__DIR__, "exact_common.jl"))
include(joinpath(@__DIR__, "EnumW.jl")); using .EnumW
include(joinpath(@__DIR__, "DPW.jl")); using .DPW
rs = loadrounds(ARGS[1]); C = deserialize(ARGS[2])
println(CUDA.name(CUDA.device()), "  rounds ", length(rs), "  CPU ref host ", C.host, " (", C.cpu, ")")
bucket(n) = n <= 64 ? 1 : n <= 128 ? 2 : n <= 256 ? 4 : 8
function run(idx; split = 3)
    W = bucket(maximum(nv(rs[i].g) for i in idx))
    TE = W == 1 ? UInt64 : EnumW.Bits{W}; TD = W == 1 ? UInt64 : DPW.Bits{W}
    sub = rs[idx]
    t0 = time()
    bt = EnumW.prepare(TE, [r.g for r in sub], [r.w for r in sub], [r.k for r in sub])
    lists, ovf, tk, info = EnumW.gpu_sepenum(bt; split)
    seps = [EnumW.sepgraph(lists[j], nv(sub[j].g)) for j in eachindex(sub)]
    tse = time() - t0
    res, dinfo = DPW.gpu_sepdp(TD, [r.w for r in sub], [r.g for r in sub], seps, [r.k for r in sub])
    twall = time() - t0
    bad = 0
    for (j, i) in enumerate(idx)
        c = C.refs[i]; g = res[j]
        ((g[3], g[4]) == (c[2], c[3]) && (c[3] != :found || g[1] == c[1])) || (bad += 1)
    end
    return (; W, n = length(idx), tk, tse, tdp = dinfo.tgpu, tprep = dinfo.tprep, tup = dinfo.tupload, tfin = dinfo.tfinish, twall, bad, ovf = count(ovf) + dinfo.novf)
end
for W in (1, 2, 4, 8)
    idx = [i for i in eachindex(rs) if bucket(nv(rs[i].g)) == W]
    isempty(idx) && continue
    run(idx[1:min(20, end)])     # compile
    heavy = sort(idx; by = i -> -C.t1[i])[1:cld(length(idx), 10)]
    for (lab, sel) in (("all", idx), ("heaviest 10%", heavy))
        r = run(sel)
        c1 = sum(C.t1[sel]); cN = sum(C.tN[sel])
        @printf("W=%d %-13s %5d rounds | CPU 1t %.2f s, %dt %.2f s | GPU kernels: sepenum %.2f + sepdp %.2f = %.2f s (%.1fx vs 1t, %.1fx vs %dt) | wall incl. CPU prep/glue %.2f s (%.1fx / %.1fx) | mismatches %d, overflows %d\n",
            W, lab, r.n, c1, C.NT, cN, r.tk, r.tdp, r.tk + r.tdp, c1 / (r.tk + r.tdp), cN / (r.tk + r.tdp), C.NT, r.twall, c1 / r.twall, cN / r.twall, r.bad, r.ovf)
        flush(stdout)
    end
end
# single-round latency (HBT calls the exact solver one local graph at a time)
println("\nsingle-round latency, heaviest 20 rounds (one at a time):")
idx1 = [i for i in eachindex(rs) if bucket(nv(rs[i].g)) == 1]
top = sort(idx1; by = i -> -C.t1[i])[1:min(20, end)]
tot1 = 0.0; totN = 0.0; totg = 0.0; totk = 0.0
for i in top
    r = run([i])
    global tot1 += C.t1[i]; global totN += C.tN[i]; global totg += r.twall; global totk += r.tk + r.tdp
    @printf("  n=%d  CPU 1t %.3f s  %dt %.3f s | GPU wall %.3f s (kernels %.3f) | ok=%s\n", nv(rs[i].g), C.t1[i], C.NT, C.tN[i], r.twall, r.tk + r.tdp, r.bad == 0)
end
@printf("  total: CPU 1t %.2f s, %dt %.2f s, GPU wall %.2f s (kernels %.2f s): %.1fx / %.1fx (wall), %.1fx / %.1fx (kernels)\n",
    tot1, C.NT, totN, totg, totk, tot1 / totg, totN / totg, tot1 / totk, totN / totk)
