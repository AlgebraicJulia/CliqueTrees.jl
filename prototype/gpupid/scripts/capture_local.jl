# Record HBT's exact local calls (permutation(::HBTExact) -> septw) during an HBT run.
#   julia -t 1 --project=mtenv capture_local.jl <instance> <hbt seconds> <out.jls>
using CliqueTrees, Graphs, Serialization
const CT = CliqueTrees
SP = "/tmp/claude-1000/-home-itaykadosh-Projects-GATAS/2e762ebb-ae79-4e3b-a677-b7caf4118124/scratchpad"
include(joinpath(SP, "tncorpus", "load.jl")); using .TNCorpus
const CALLS = Any[]
function CT.permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::CT.HBTExact; verbose::Bool = false) where {V}
    t0 = time()
    perm, invp, k, status = CT.septw(weights, graph, alg.lo, alg.hi, alg.dl, 1)
    t = time() - t0
    n = nv(graph)
    adj = [sort!([u for u in neighbors(graph, v) if u != v]) for v in 1:n]
    push!(CALLS, (; n, adj, weights = collect(Float64, weights[1:n]), lo = Float64(alg.lo), hi = Float64(alg.hi), budget = alg.dl - t0, t, status = string(status), k = Float64(k)))
    alg.status[] = status
    return convert(Vector{V}, perm), convert(Vector{V}, invp)
end
name, T, out = ARGS[1], parse(Float64, ARGS[2]), ARGS[3]
inst = TNCorpus.load(name); w = inst.weights; g = TNCorpus.bipartite(inst)
minalgs() = (MinimalChordal(MF()), MinimalChordal(MF(; strategy = 1)), MinimalChordal(AMF()), MinimalChordal(MMD()))
o, _ = permutation(w, g, SafeRules(SafeSeparators(HBT(; time = T, threads = 1), minalgs(); maxmissing = 50)))
println("$name width $(treewidth(w, g, o)) exact calls $(length(CALLS)) timeouts $(count(c -> c.status == "BT_TIMEOUT", CALLS)) cpu time $(sum(c -> c.t, CALLS; init = 0.0))")
serialize(out, CALLS)
