# laptop: cuDSS ND partition tree (separator sizes, 2^nlevels - 1 entries): read it after the default reordering,
# compare with the order, and time factorization for user_perm with / without the tree (NOTES.md "partition tree")
using CUDA, CUDSS, SparseArrays, LinearAlgebra, Graphs, CliqueTrees, Serialization
const SRC = ENV["SRC"]
include(joinpath(SRC, "gens.jl"))
include(joinpath(SRC, "ndutil.jl"))
function spd_lower(ptr, adj)
    n = length(ptr) - 1
    rp = Int32[1]; cv = Int32[]; vv = Float64[]
    for i in 1:n
        for q in ptr[i]:(ptr[i + 1] - 1)
            adj[q] < i && (push!(cv, adj[q]); push!(vv, -1.0))
        end
        push!(cv, i); push!(vv, ptr[i + 1] - ptr[i] + 1.0); push!(rp, length(cv) + 1)
    end
    rp, cv, vv
end
L = parse(Int, get(ENV, "L", "200"))
ptr, adj = graph_csr(lat2(L)); n = length(ptr) - 1
rp, cv, vv = spd_lower(ptr, adj); d = (CuVector(rp), CuVector(cv), CuVector(vv))
x = CUDA.zeros(Float64, n); b = CUDA.ones(Float64, n)
const NT = 2^10 - 1
function run(; uperm = nothing, tree = nothing, get = false)
    s = CudssSolver(d..., "SPD", 'L')
    uperm === nothing || cudss_set(s, "user_perm", uperm)
    tree === nothing || CUDSS.cudssDataSet(s.data.handle, s.data, CUDSS.CUDSS_DATA_USER_ND_PARTITION_TREE, tree, sizeof(tree))
    CUDA.synchronize(); t0 = time()
    cudss("analysis", s, x, b); CUDA.synchronize(); t1 = time()
    cudss("factorization", s, x, b); CUDA.synchronize(); t2 = time()
    p = tr = nothing
    if get
        p = Vector{Int32}(undef, n); cudss_set(s, "perm_reorder_row", p); cudss_get(s, "perm_reorder_row")
        tr = zeros(Int32, NT); w = Ref{Csize_t}(0)
        CUDSS.cudssDataGet(s.data.handle, s.data, CUDSS.CUDSS_DATA_ND_PARTITION_TREE, tr, sizeof(tr), w)
    end
    l = cudss_get(s, "lu_nnz")
    finalize(s.data); finalize(s.matrix); finalize(s.config)
    return t1 - t0, t2 - t1, l, p, tr
end
run(); run()
ta, tf, l, p, tr = run(; get = true)
println("default: an=$ta fact=$tf lu_nnz=$l")
println("tree[1:15] = ", tr[1:15], " sum=", sum(tr), " nonzero=", count(!iszero, tr))
println("tree[end-15:end] = ", tr[end-15:end])
# where are the separators in the order? check: the last tr[1] vertices of p separate the rest?
g = csr_graph(ptr, adj)
function ncomp(g, keep)
    sg, _ = induced_subgraph(g, keep); length(connected_components(sg))
end
s1 = tr[1]
println("components after removing last tree[1]=$s1 of the order: ", ncomp(g, Int.(p[1:n - s1])), " ; removing last tree[end]=$(tr[end]): ", ncomp(g, Int.(p[1:n - tr[end]])))
serialize(joinpath(ENV["OUTD"], "tree_lat2_$L.jls"), (p, tr))
for k in 1:2
    a1, f1, l1, _, _ = run(; uperm = copy(p))
    a2, f2, l2, _, _ = run(; uperm = copy(p), tree = copy(tr))
    println("own perm: an=$a1 fact=$f1 lu=$l1 | own perm + tree: an=$a2 fact=$f2 lu=$l2")
end
