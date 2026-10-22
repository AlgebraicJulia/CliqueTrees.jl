# bag width (CliqueTrees treewidth) of a given ordering: julia scoreord.jl <name> <ord file>
include(joinpath(get(ENV, "TNCORPUS_DIR", "/path/to/tn-work/tncorpus"), "load.jl"))
using .TNCorpus, CliqueTrees
inst = TNCorpus.load(ARGS[1]; dir = get(ENV, "FRONTIER_DIR", "/path/to/tn-work/frontier"))
o = parse.(Int, split(readlines(ARGS[2])[2]))
println("bag width of ordering: ", treewidth(inst.weights, TNCorpus.bipartite(inst), o), "  n=", inst.n, " len=", length(o))
