using CUDA, CliqueTrees, Graphs, Serialization, Printf, Random
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
NS = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 300
rng = Xoshiro(2); sample = calls[randperm(rng, length(calls))[1:min(NS, end)]]
gs = [BipartiteGraph(c.g) for c in sample]; ws = [c.w for c in sample]; ks = [c.k for c in sample]
seps = [first(CliqueTrees.sepenum(ws[i], gs[i], ks[i], Inf, 1)) for i in eachindex(sample)]
tc = @elapsed refs = [cpu_sepdp(ws[i], gs[i], seps[i], ks[i]) for i in eachindex(sample)]
res, info, _ = gpu_sepdp(ws, gs, seps, ks)   # warm (compile)
res, info, _ = gpu_sepdp(ws, gs, seps, ks)
bad = 0
for i in eachindex(sample)
    r = res[i]; c = refs[i]
    if (r[3], r[4]) != (c[3], c[4]) || (c[4] == :found && r[1] != c[1])
        global bad += 1
        bad <= 5 && println("mismatch i=$i n=$(nv(gs[i])): cpu (mink $(c[3]), $(c[4])) gpu (mink $(r[3]), $(r[4])) perm equal $(r[1] == c[1])")
    end
end
println("sepdp: $(length(sample)) rounds, mismatches $bad, GPU overflows $(info.novf), blocks $(info.nblk)")
@printf("CPU sepdp %.3f s | GPU path: prep(CPU) %.3f + upload %.3f + GPU %.3f + order(CPU) %.3f s\n", tc, info.tprep, info.tupload, info.tgpu, info.tfinish)
