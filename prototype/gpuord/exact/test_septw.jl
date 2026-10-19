using CUDA, CliqueTrees, Graphs, Serialization, Printf, Random
include(joinpath(@__DIR__, "SeptwGPU.jl"))
files = filter(f -> startswith(basename(f), "atoms_") && endswith(f, ".jls"), readdir(@__DIR__; join = true))
atoms = []
for f in files, (nm, v) in deserialize(f), a in v.atoms
    (3 <= a.n <= 512 && !isempty(a.rounds)) || continue
    g = SimpleGraph(a.n); for (x, y) in a.edges; add_edge!(g, x, y); end
    push!(atoms, (; nm, g = BipartiteGraph(g), w = a.weights, lo = a.lo, hi = a.hi))
end
rng = Xoshiro(5); NS = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 60
sel = atoms[randperm(rng, length(atoms))[1:min(NS, end)]]
bad = 0
for a in sel
    pc, _, kc, sc = CliqueTrees.septw(a.w, a.g, a.lo, a.hi, Inf, 1)
    pg, _, kg, sg = septw_gpu(a.w, a.g, a.lo, a.hi, Inf)
    ok = kc == kg && sc == sg && (sc != CliqueTrees.BT_FOUND || pc == pg)
    ok || (global bad += 1; println("mismatch n=$(nv(a.g)): cpu ($kc, $sc) gpu ($kg, $sg) perm eq $(pc == pg)"))
end
println("septw: $(length(sel)) atoms (n $(extrema(nv(a.g) for a in sel))), mismatches $bad, GPU rounds $(GPUSTATS[:rounds])")
