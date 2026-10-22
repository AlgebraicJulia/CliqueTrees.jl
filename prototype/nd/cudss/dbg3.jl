# laptop: which derived partition trees does cuDSS accept? (lat2_L, cuDSS's own order; NOTES.md "partition tree")
using CUDA, CUDSS, SparseArrays, LinearAlgebra, Graphs, CliqueTrees, Serialization
include(joinpath(ENV["SRC"], "gens.jl")); include(joinpath(ENV["SRC"], "ndutil.jl")); include(joinpath(@__DIR__, "ndtree.jl"))
function spd_lower(ptr, adj)
    n = length(ptr) - 1; rp = Int32[1]; cv = Int32[]; vv = Float64[]
    for i in 1:n
        for q in ptr[i]:(ptr[i + 1] - 1); adj[q] < i && (push!(cv, adj[q]); push!(vv, -1.0)); end
        push!(cv, i); push!(vv, ptr[i + 1] - ptr[i] + 1.0); push!(rp, length(cv) + 1)
    end
    rp, cv, vv
end
L = parse(Int, get(ENV, "L", "30"))
ptr, adj = graph_csr(lat2(L)); n = length(ptr) - 1
rp, cv, vv = spd_lower(ptr, adj); d = (CuVector(rp), CuVector(cv), CuVector(vv))
x = CUDA.zeros(Float64, n); b = CUDA.ones(Float64, n)
p, tr = deserialize(joinpath(ENV["OUTD"], "tree_lat2_$L.jls")); p = Int.(p)
function try1(lab, u, t)
    s = CudssSolver(d..., "SPD", 'L')
    cudss_set(s, "user_perm", Int32.(u))
    CUDSS.cudssDataSet(s.data.handle, s.data, CUDSS.CUDSS_DATA_USER_ND_PARTITION_TREE, t, sizeof(t))
    msg = try
        cudss("analysis", s, x, b); CUDA.synchronize(); t1 = time()
        cudss("factorization", s, x, b); CUDA.synchronize(); tf = time() - t1
        "ok lu_nnz=$(cudss_get(s, "lu_nnz")) fact=$(round(tf, sigdigits = 3))"
    catch err
        "FAIL " * sprint(showerror, err)[1:min(end, 60)]
    end
    println(rpad(lab, 30), msg); flush(stdout)
end
get(ENV, "MINSZ", "") == "" && try1("cuDSS tree", p, tr)
for m in [parse(Int, x) for x in split(get(ENV, "MINSZ", ""), ",") if !isempty(x)]
    no, t2 = nd_tree(ptr, adj, p; minsz = m)
    try1("derived minsz=$m zeros=$(count(iszero, t2))", no, t2)
end
