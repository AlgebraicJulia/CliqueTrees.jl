# Extract the atoms (leaves of SafeRules + SafeSeparators) of corpus instances.
# julia -t 1 --project=mtenv extract_atoms.jl <outdir> <names file or "paper"> [minn]
using CliqueTrees, Graphs, Serialization
const CT = CliqueTrees
SP = "/path/to/tn-work"
include(joinpath(SP, "tncorpus", "prof", "common.jl"))
outdir = ARGS[1]; minn = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 30
names = ARGS[2] == "paper" ? [r.name for r in TNCorpus.select(; source = "einsum", paper = true)] : readlines(ARGS[2])
mkpath(outdir)
idx = open(joinpath(outdir, "index_" * basename(ARGS[2]) * ".tsv"), "a")
for name in names
    try
        inst = TNCorpus.load(name)
        w = inst.weights; g = TNCorpus.bipartite(inst)
        empty!(BATCHLOG); CAPTURE[] = true
        t = @elapsed permutation(w, g, pipeline(Tap(:atoms, MF())))
        CAPTURE[] = false
        isempty(BATCHLOG) && (println(idx, name, "\t-\t0\t0\t0\t$t"); flush(idx); continue)
        ws, gs = BATCHLOG[end]
        for (i, (wi, gi)) in enumerate(zip(ws, gs))
            n = nv(gi); n >= minn || continue
            adj = [sort!(filter(!=(v), collect(neighbors(gi, v)))) for v in 1:n]
            lb = lowerbound(wi, gi, MMW())
            ub = treewidth(wi, gi, permutation(wi, gi, MF())[1])
            aname = "$(name)__$(i)"
            serialize(joinpath(outdir, aname * ".jls"), (; name = aname, inst = name, n, weights = collect(Float64, wi), adj, lb, ub))
            m = sum(length, adj) ÷ 2
            println(idx, join((name, aname, n, m, lb, ub, round(t; digits = 3), length(unique(wi))), '\t')); flush(idx)
        end
    catch e
        println(idx, name, "\tERROR\t", sprint(showerror, e)); flush(idx)
    end
end
