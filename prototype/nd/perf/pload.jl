# shared loader for the perf scripts: mtx basename in ../fill/data, lat2_L, lat3_L, or a TN corpus name
using CliqueTrees, Graphs, SparseArrays, MatrixMarket, Random
const PDIR = @__DIR__
const RDIR = joinpath(PDIR, "..")
include(joinpath(RDIR, "Rounds.jl")); using .Rounds
include(joinpath(RDIR, "evalutil.jl")); include(joinpath(RDIR, "gens.jl"))
include(joinpath(RDIR, "..", "tncorpus", "load.jl")); using .TNCorpus
const FDATA = get(ENV, "FDATA", joinpath(RDIR, "..", "fill", "data"))
function loadg(name)
    f = joinpath(FDATA, name * ".mtx")
    if isfile(f)
        A = mmread(f); B = SparseMatrixCSC(A .!= 0); B = B + B'
        return SimpleGraph(B), "ssmc"
    elseif startswith(name, "lat2_")
        return lat2(parse(Int, name[6:end])), "tn"
    elseif startswith(name, "lat3_")
        return lat3(parse(Int, name[6:end])), "tn"
    else
        return TNCorpus.load(name).graph, "tn"
    end
end
# fill metrics with unit weights: nnz(L), flops = sum cc^2, tw
function fillq(g, o)
    n = nv(g)
    @assert length(o) == n && isperm(o) "invalid permutation"
    cw = colweights(ones(Int, n), g, o)
    return sum(cw), sum(Float64(c)^2 for c in cw), maximum(cw) - 1
end
