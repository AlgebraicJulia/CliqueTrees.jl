include(joinpath(@__DIR__, "common.jl"))
g, w = synth("grid60")
alg = gpu(1.0; pmax = 256)
splits, pieces = Hybrid.splitpieces([g], [w]; pmax = 256)
porders, pw = Hybrid.gpusearch(pieces, [w]; time = 1.0)
order = Hybrid.assemble!(Int[], splits[1], splits[1].root, porders, nothing)
println("max piece width ", maximum(pw) / 1000, "  final ", treewidth(w, g; alg = order))
# which vertex has the widest bag
label, tree = cliquetree(w, g, order, Nodal())
best = 0.0; bv = 0
for (i, clique) in enumerate(tree)
    s = sum(w[label[v]] for v in clique)
    if s > best; global best = s; global bv = label[residual(clique)[1]]; end
end
inpiece = Dict(v => i for (i, pc) in enumerate(pieces) for v in pc.verts[1:pc.nint])
println("widest bag $best at vertex $bv (", haskey(inpiece, bv) ? "piece $(inpiece[bv])" : "separator", ")")
# separator sizes per level
function walk(s, node, depth)
    nd = s.nodes[node]
    if nd.piece == 0 && (nd.a > 0 || nd.b > 0)
        println("  "^depth, "sep |S\\Bd|=", length(nd.sep), " w=", round(sum(w[nd.sep]); digits=1))
        depth < 3 && (walk(s, nd.a, depth+1); walk(s, nd.b, depth+1))
    end
end
walk(splits[1], splits[1].root, 0)
println("piece sizes ", [ (nv(pc.g), pc.nint) for pc in pieces])
