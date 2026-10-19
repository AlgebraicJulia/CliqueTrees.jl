# Verify the W-generic exact kernels (EnumW, DPW) against CliqueTrees on
# captured septw rounds, for set types UInt64, Bits{2}, Bits{4} (forced) and
# the natural type of each atom. usage: julia test_w.jl [nsample] [nmin]
using CUDA, CliqueTrees, Graphs, Serialization, Printf, Random
include(joinpath(@__DIR__, "EnumW.jl")); using .EnumW
include(joinpath(@__DIR__, "DPW.jl")); using .DPW
const B = EnumW.Bits

function loadrounds(files; nmin = 2, nmax = 512)
    out = []
    for f in files, (nm, v) in deserialize(f), a in v.atoms
        (nmin <= a.n <= nmax) || continue
        g = SimpleGraph(a.n); for (x, y) in a.edges; add_edge!(g, x, y); end
        for r in a.rounds
            r[2] >= 0 && push!(out, (; inst = nm, g = BipartiteGraph(g), w = a.weights, k = r[1]))
        end
    end
    return out
end

function check(T, rs; split = 3)
    isempty(rs) && return (0, 0, 0)
    bt = EnumW.prepare(T, [r.g for r in rs], [r.w for r in rs], [r.k for r in rs])
    lists, ovf, tk, info = EnumW.gpu_sepenum(bt; split)
    bad_e = 0; bad_d = 0
    seps = []
    for (j, r) in enumerate(rs)
        ref = EnumW.cpu_sepenum_sets(T, r.g, r.w, r.k)
        ovf[j] || ref == lists[j] || (bad_e += 1)
        push!(seps, EnumW.sepgraph(lists[j], nv(r.g)))
    end
    TD = T === UInt64 ? UInt64 : DPW.Bits{EnumW.nwords(T)}
    res, dinfo = DPW.gpu_sepdp(TD, [r.w for r in rs], [r.g for r in rs], seps, [r.k for r in rs])
    for (j, r) in enumerate(rs)
        c = DPW.cpu_sepdp(r.w, r.g, seps[j], r.k)
        g_ = res[j]
        ((g_[3], g_[4]) == (c[3], c[4]) && (c[4] != :found || g_[1] == c[1])) || (bad_d += 1)
    end
    return bad_e, bad_d, count(ovf) + dinfo.novf
end

files = filter(f -> startswith(basename(f), "atoms_") && endswith(f, ".jls"), readdir(@__DIR__; join = true))
NS = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 200
rng = Xoshiro(3)
small = loadrounds(files; nmax = 64); small = small[randperm(rng, length(small))[1:min(NS, end)]]
for T in (UInt64, B{2}, B{4})
    e, d, o = check(T, small)
    println("n <= 64 as $T: $(length(small)) rounds, sepenum mismatches $e, sepdp mismatches $d, overflows $o")
end
big = loadrounds(files; nmin = 65)
println("captured rounds with n > 64: ", length(big), "  sizes ", isempty(big) ? "-" : extrema(nv(r.g) for r in big))
for W in (2, 4, 8)
    bucket(n) = n <= 64 ? 1 : n <= 128 ? 2 : n <= 256 ? 4 : 8
    rs = [r for r in big if bucket(nv(r.g)) == W]
    isempty(rs) && continue
    rs = rs[1:min(NS, end)]
    e, d, o = check(B{W}, rs)
    println("n in bucket W=$W: $(length(rs)) rounds, sepenum mismatches $e, sepdp mismatches $d, overflows $o")
end
