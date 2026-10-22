# Greedy unslicing under a bag-width cap (a tree-decomposition view of slicing: the sliced set S
# is a vertex set whose deletion leaves a line graph of small weighted treewidth).
#   julia unslice.jl <name> <S file: one full-network vertex id per line> <bag cap B> <out S file> [<T>]
# Starting from S (e.g. cotengra's sliced set), try to restore each s in S (lowest degree first):
# keep s unsliced if an ordering of G - (S \ {s}) still has bag width <= B: the ND ordering (nd8,
# ~0.3 s) or, if that is too wide, the hbt-nd pipeline with T seconds (default 3).
include(joinpath(get(ENV, "TNCORPUS_DIR", "/path/to/tn-work/tncorpus"), "prof", "common.jl"))
name = ARGS[1]; S = parse.(Int, readlines(ARGS[2])); B = parse(Float64, ARGS[3]); outf = ARGS[4]
T = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 3.0
fdir = get(ENV, "FRONTIER_DIR", "/path/to/tn-work/frontier")
inst = TNCorpus.load(name; dir = fdir)
G = inst.graph; w = inst.weights
function bagw(Sset)
    keep = [v for v in vertices(G) if !(v in Sset)]
    sg, vmap = induced_subgraph(G, keep)
    o, _ = permutation(w[vmap], sg, nd8())
    b = treewidth(w[vmap], sg, o)
    b <= B && return b
    o, _ = permutation(w[vmap], sg, pipeline(HBT(; time = T, threads = 1, init = nd8())))
    return min(b, treewidth(w[vmap], sg, o))
end
cur = Set(S)
t0 = time()
b0 = bagw(cur)
println("start |S|=$(length(cur)) bag=$(b0) cap=$(B)")
for s in sort(S; by = v -> degree(G, v))
    delete!(cur, s)
    b = bagw(cur)
    if b <= B
        println("  restore $s (deg $(degree(G, s))): bag $b")
    else
        push!(cur, s)
    end
    flush(stdout)
end
println("end |S|=$(length(cur)) bag=$(bagw(cur)) ($(round(time() - t0; digits=1)) s)")
open(outf, "w") do io
    for v in sort(collect(cur)); println(io, v); end
end
