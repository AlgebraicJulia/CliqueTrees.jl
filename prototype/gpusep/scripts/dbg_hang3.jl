using CliqueTrees, CUDA, Graphs, Serialization
const CT = CliqueTrees
ext = Base.get_extension(CliqueTrees, :CUDAExt)
E = ext.SepEnumGPU
@eval E function timed_dl_unused(f, dl::Float64, flag, i)
    e0 = CuEvent(); e1 = CuEvent()
    CUDA.record(e0); f(); CUDA.record(e1)
    ab = false; t0 = time(); last = t0
    println("launched; stream=", CUDA.stream()); flush(stdout)
    it = 0
    while (it += 1; it <= 3 && (println("query ", it); flush(stdout)); d = CUDA.isdone(e1); it <= 3 && (println("isdone=", d); flush(stdout)); !d)
        it <= 3 && (println("body ", it); flush(stdout))
        if !ab && time() > dl
            ABORTSTREAM[] === nothing && (ABORTSTREAM[] = CuStream(; flags = CUDA.STREAM_NON_BLOCKING))
            CUDA.stream!(ABORTSTREAM[]) do
                copyto!(view(flag, i:i), Int32[1])
            end
            ab = true; println("abort set at ", round(time() - t0; digits = 2)); flush(stdout)
            v = CUDA.stream!(ABORTSTREAM[]) do
                Array(view(flag, i:i))
            end
            println("flag reads back ", v, " at ", round(time() - t0; digits = 2)); flush(stdout)
        end
        if time() - last > 10
            last = time(); println("still waiting ", round(time() - t0; digits = 1)); flush(stdout)
        end
        it <= 3 && (println("sleep ", it); flush(stdout))
        Libc.systemsleep(50e-6)
        it <= 3 && (println("slept ", it); flush(stdout))
    end
    CUDA.synchronize(e1)
    return CUDA.elapsed(e0, e1), ab
end
a = deserialize(ARGS[1]); g = SimpleGraph(a.n); for v in 1:a.n, u in a.adj[v]; u > v && add_edge!(g, v, u); end; g = BipartiteGraph(g); w = a.weights
k = parse(Float64, ARGS[2]); T = ext.settype(a.n)
ts = time(); b = E.prepare(T, [g], [w], [k])
lists, ovf, tk, info = E.gpu_sepenum(b; split = 3, dl = ts + 5.0)
println("enum $(round(time()-ts;digits=2)) s info=$info"); flush(stdout)
