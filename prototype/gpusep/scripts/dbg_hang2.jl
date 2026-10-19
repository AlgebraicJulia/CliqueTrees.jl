using CliqueTrees, CUDA, Graphs, Serialization
const CT = CliqueTrees
ext = Base.get_extension(CliqueTrees, :CUDAExt)
E = ext.SepEnumGPU; D = ext.SepDPGPU
a = deserialize(ARGS[1]); g = SimpleGraph(a.n); for v in 1:a.n, u in a.adj[v]; u > v && add_edge!(g, v, u); end; g = BipartiteGraph(g); w = a.weights
k = parse(Float64, ARGS[2]); T = ext.settype(a.n)
CT.septwwarmup!(CT.GPUSeptw(); nmax = a.n)
sep, _ = CT.sepenum(w, g, k, Inf, 8); println("cpu nsep=$(nv(sep))"); flush(stdout)
ts = time(); dl = ts + 5.0
b = E.prepare(T, [g], [w], [k]); println("prepare $(round(time()-ts;digits=2))"); flush(stdout)
lists, ovf, tk, info = E.gpu_sepenum(b; split = 3, dl)
println("enum $(round(time()-ts;digits=2)) s info=$info ovf=$ovf n=$(isnothing(lists) ? -1 : length(lists[1]))"); flush(stdout)
ts = time(); dl = ts + 5.0
res, info = D.gpu_sepdp(T, [w], [g], [sep], [k]; dl)
println("dp $(round(time()-ts;digits=2)) s info=$info res=$(isnothing(res) ? nothing : res[1][3:4])"); flush(stdout)
