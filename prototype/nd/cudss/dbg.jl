# step-by-step cuDSS check on a tiny SPD matrix (laptop debugging of cudss_bench.jl)
using CUDA, CUDSS, SparseArrays, LinearAlgebra, Graphs, CliqueTrees, Serialization
const SRC = ENV["SRC"]
include(joinpath(SRC, "gens.jl"))
include(joinpath(SRC, "ndutil.jl"))
step(s) = (println(">> ", s); flush(stdout))
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
ptr, adj = graph_csr(lat2(30)); n = length(ptr) - 1
rp, cv, vv = spd_lower(ptr, adj)
if get(ENV, "MT", "0") == "1"
    lib = joinpath(dirname(CUDSS.libcudss), "libcudss_mtlayer_gomp.so")
    step("mt layer $lib"); CUDSS.cudssSetThreadingLayer(CUDSS.handle(), lib)
end
s = CudssSolver(CuVector(rp), CuVector(cv), CuVector(vv), "SPD", 'L')
get(ENV, "MT", "0") == "1" && (step("host_nthreads"); cudss_set(s, "host_nthreads", 4))
x = CUDA.zeros(Float64, n); b = CUDA.ones(Float64, n)
step("reordering"); cudss("reordering", s, x, b); CUDA.synchronize()
step("symbolic"); cudss("symbolic_factorization", s, x, b); CUDA.synchronize()
step("lu_nnz"); println(cudss_get(s, "lu_nnz"))
if get(ENV, "FL", "1") == "1"
    step("flops"); r = Ref{UInt64}(0); w = Ref{Csize_t}(0)
    CUDSS.cudssDataGet(s.data.handle, s.data, CUDSS.CUDSS_DATA_FLOPS, r, 8, w); println("flops raw=", r[], " as f64=", reinterpret(Float64, r[]), " written=", w[])
end
if get(ENV, "PERM", "1") == "1"
    step("perm"); p = Vector{Int32}(undef, n); cudss_set(s, "perm_reorder_row", p); cudss_get(s, "perm_reorder_row")
    println("perm min/max ", extrema(p), " isperm ", isperm(p .+ (1 - minimum(p))))
    g = csr_graph(ptr, adj)
    println("fillq(p)=", fillq(g, Int.(p) .+ (1 - minimum(p)))[1], " fillq(invp)=", fillq(g, invperm(Int.(p) .+ (1 - minimum(p))))[1])
end
step("factorization"); cudss("factorization", s, x, b); CUDA.synchronize()
step("solve"); cudss("solve", s, x, b); CUDA.synchronize()
println("info ", cudss_get(s, "info"))
step("finalize"); finalize(s.data); finalize(s.matrix); finalize(s.config)
GC.gc(); step("done")
