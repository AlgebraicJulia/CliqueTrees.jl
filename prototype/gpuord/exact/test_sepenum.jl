using CUDA, CliqueTrees, Graphs, Serialization, Printf, Random
include(joinpath(@__DIR__, "SepEnumGPU.jl")); using .SepEnumGPU
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
files = filter(f -> startswith(basename(f), "atoms_") && endswith(f, ".jls"), readdir(@__DIR__; join = true))
calls = loadatoms(files)
println("septw rounds captured: ", length(calls), "  (n <= 64)")
# correctness on a sample first
rng = Xoshiro(1)
sample = calls[randperm(rng, length(calls))[1:min(400, end)]]
b = prepare([c.g for c in sample], [c.w for c in sample], [c.k for c in sample])
lists, ovf, t = gpu_sepenum(b)
nbad = 0; nov = count(ovf)
for (j, c) in enumerate(sample)
    ovf[j] && continue
    ref = cpu_sepenum_lists(BipartiteGraph(c.g), c.w, c.k)
    ref == lists[j] || (global nbad += 1; nbad <= 3 && println("mismatch: n=$(nv(c.g)) k=$(c.k) cpu $(length(ref)) gpu $(length(lists[j]))  sets equal: ", Set(ref) == Set(lists[j])))
end
println("sample: $(length(sample)) rounds, mismatches $nbad, overflows $nov, kernel $(round(t*1000, digits=1)) ms")
