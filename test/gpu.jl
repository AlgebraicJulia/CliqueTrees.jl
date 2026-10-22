using CliqueTrees
using CliqueTrees: BipartiteGraph, BT_FOUND, BT_ABOVE, BT_TIMEOUT, CPUSeptw, GPUSeptw, septw, septwbackend!, septwwarmup!
using Graphs
using Random
using Test

# The GPU rounds of the exact searches (the extension of CUDA.jl). Skipped
# without CUDA.jl or without a working device.
const HASCUDA = !isnothing(Base.find_package("CUDA"))

HASCUDA && using CUDA

if !HASCUDA || !CUDA.functional()
    @testset "GPU rounds (no GPU: skipped)" begin
        @test_skip false
    end
else
    @testset "GPU rounds" begin
        ext = Base.get_extension(CliqueTrees, :CUDAExt)

        # connected random graphs with n vertices; with p = 0, a grid of
        # five rows with three random chords (more than 64 vertices, and a
        # small width)
        function graphs(rng, n, p, m)
            out = BipartiteGraph{Int, Int}[]

            while length(out) < m
                if iszero(p)
                    g = Graphs.grid([5, n ÷ 5])

                    for _ in 1:3
                        add_edge!(g, rand(rng, 1:nv(g)), rand(rng, 1:nv(g)))
                    end
                else
                    g = erdos_renyi(n, p; rng)
                end

                is_connected(g) && push!(out, BipartiteGraph(g))
            end

            return out
        end

        function weights(rng, kind, n)
            kind === :unit && return ones(Int, n)
            kind === :int && return rand(rng, 1:4, n)
            return log2.(rand(rng, 2:6, n))
        end

        # the result of septw on the backend
        function run(backend, w, g, lo, hi)
            old = septwbackend!(backend)

            try
                return septw(w, g, lo, hi, Inf, 1)
            finally
                septwbackend!(old)
            end
        end

        below(k::Integer) = k - 1
        below(k::AbstractFloat) = k * (1 - 1.0e-12)
        same(x::Integer, y::Integer) = x == y
        same(x::AbstractFloat, y::AbstractFloat) = isapprox(x, y; rtol = 1.0e-12)

        rng = MersenneTwister(7)
        cases = [(n, p, kind) for (n, p) in ((12, 0.4), (24, 0.25), (40, 0.12), (70, 0.0)) for kind in (:unit, :int, :float)]

        for engine in (:septw, :pid)
            septwwarmup!(GPUSeptw(; engine); nmax = 128)
        end

        @testset "sepenum and sepdp on the GPU: the CPU results" begin
            gpu = GPUSeptw(; engine = :septw, budget = 0.0, race = false)

            for (n, p, kind) in cases, g in graphs(rng, n, p, 2)
                w = weights(rng, kind, n); cpu = run(CPUSeptw(), w, g, 0, typemax(eltype(w)))

                for (lo, hi) in ((zero(eltype(w)), typemax(eltype(w))), (cpu[3], typemax(eltype(w))), (cpu[3] + 1, cpu[3] + 1), (zero(eltype(w)), below(cpu[3])))
                    ext.resetstats!()
                    @test run(gpu, w, g, lo, hi) == run(CPUSeptw(), w, g, lo, hi)
                end
            end

            # sepdp alone, on the separators of the CPU
            for (n, p, kind) in cases, g in graphs(rng, n, p, 1)
                w = weights(rng, kind, n); k = run(CPUSeptw(), w, g, 0, typemax(eltype(w)))[3]

                for kk in (k - minimum(w), k)
                    sep, _ = CliqueTrees.sepenum(w, g, kk, Inf, 1)
                    perm, _, mink, status = CliqueTrees.sepdp(w, g, sep, kk, Inf, 1)
                    r = ext.septwround(w, g, kk, Inf, sep)
                    @test r[3] == status

                    if status == BT_FOUND
                        @test r[1] == perm && r[2] == mink
                    else
                        @test r[2] == CliqueTrees.bt_next(w, g, kk, mink)
                    end
                end
            end
        end

        @testset "batches of sepenum and sepdp: the rounds alone on the CPU" begin
            clk() = CliqueTrees.bt_clk(Inf)

            # graphs of different sizes with sets of the same type, each at
            # its width and below it (found and above in one batch)
            for (sizes, p) in (((16, 24, 40), 0.2), ((70, 100), 0.0)), kind in (:unit, :int, :float)
                ws = []; gs = []; ks = []

                for n in sizes, g in graphs(rng, n, p, 2)
                    w = weights(rng, kind, n); k = run(CPUSeptw(), w, g, 0, typemax(eltype(w)))[3]
                    append!(ws, [w, w]); append!(gs, [g, g]); append!(ks, [k, below(k)])
                end

                T = ext.settype(maximum(sizes)); @test all(g -> ext.settype(nv(g)) == T, gs)
                xs = ext.septwbatch(T, ws, gs, ks, _ -> false, [clk() for _ in gs])

                for i in eachindex(gs)
                    @test ext.answer(ws[i], gs[i], ks[i], xs[i]) == CliqueTrees.bt_round!(nothing, ws[i], gs[i], ks[i], Inf, 1)
                end

                # a graph that is stopped: no answer, and the same answers
                # for the others
                ys = ext.septwbatch(T, ws, gs, ks, i -> i == 2, [clk() for _ in gs])
                @test isnothing(ys[2])
                @test all(ys[i] == xs[i] for i in eachindex(gs) if i != 2)
            end

            # the worker serves the requests in the queue in batches: held
            # back while another task holds the GPU, the requests of several
            # searches go in one batch
            if ext.worker!()
                gs = [graphs(rng, 24, 0.25, 3); graphs(rng, 40, 0.12, 3)]; ws = [weights(rng, :int, nv(g)) for g in gs]
                cpu = [run(CPUSeptw(), w, g, 0, typemax(Int)) for (w, g) in zip(ws, gs)]
                gpu = GPUSeptw(; engine = :septw, budget = 0.0, race = false)
                held = Channel{Nothing}(1); done = Channel{Nothing}(1)

                task = Threads.@spawn lock(ext.LOCK) do
                    put!(held, nothing); take!(done)
                end

                take!(held); ext.resetstats!(); old = septwbackend!(gpu)

                try
                    tasks = [Threads.@spawn septw(w, g, 0, typemax(Int), Inf, 1) for (w, g) in zip(ws, gs)]
                    t = time()

                    while ext.queued(1) < length(gs) && time() < t + 60
                        sleep(0.01)
                    end

                    put!(done, nothing); wait(task)
                    @test all(fetch(t) == c for (t, c) in zip(tasks, cpu))
                finally
                    septwbackend!(old)
                end

                @test get(ext.STATS, :maxbatch, 0) == length(gs)
                @test get(ext.STATS, :batches, 0) < get(ext.STATS, :escalated, 0)
                @test iszero(get(ext.STATS, :gpufailed, 0))
            end
        end

        @testset "PID on the GPU: the CPU widths" begin
            gpu = GPUSeptw(; engine = :pid, budget = 0.0, race = false)

            for (n, p, kind) in cases, g in graphs(rng, n, p, 2)
                w = weights(rng, kind, n); hi = typemax(eltype(w))
                cpu = run(CPUSeptw(), w, g, 0, hi); res = run(gpu, w, g, 0, hi)
                @test res[4] == cpu[4] == BT_FOUND
                @test same(res[3], cpu[3])
                @test sort(res[1]) == 1:n && res[2][res[1]] == 1:n
                @test same(treewidth(w, g; alg = res[1]), res[3])

                # a target above the width: an ordering that meets it
                res = run(gpu, w, g, cpu[3] + 1, hi)
                @test res[4] == BT_FOUND && treewidth(w, g; alg = res[1]) <= cpu[3] + 1

                # a cap below the width
                res = run(gpu, w, g, 0, below(cpu[3]))
                @test res[4] == BT_ABOVE
            end
        end

        @testset "PIDBT, HBT, and RTW" begin
            gs = [graphs(rng, 24, 0.25, 3); graphs(rng, 40, 0.12, 3)]

            for engine in (:septw, :pid)
                gpu = GPUSeptw(; engine, budget = 0.0, race = false)

                for g in gs
                    w = weights(rng, :int, nv(g))
                    tw = treewidth(w, g; alg = PIDBT())
                    ext.resetstats!(); old = septwbackend!(gpu)

                    try
                        @test treewidth(w, g; alg = PIDBT()) == tw
                        @test treewidth(w, g; alg = HBT(; time = 60.0)) == tw

                        # (the time limits are long: they include compilation)
                        _, _, k, status, _ = CliqueTrees.rtw(w, g, 0, typemax(Int); mt = 60.0)
                        @test status == BT_FOUND && k == tw
                    finally
                        septwbackend!(old)
                    end

                    @test get(ext.STATS, :escalated, 0) > 0
                    @test iszero(get(ext.STATS, :gpufailed, 0))
                end
            end
        end

        @testset "PID: deduplication per join chunk and per round" begin
            P = ext.PIDGPU

            for (n, p) in ((24, 0.25), (40, 0.12), (70, 0.0)), g in graphs(rng, n, p, 2), kind in (:int, :float)
                w = weights(rng, kind, n); tw = Float64(run(CPUSeptw(), w, g, 0, typemax(eltype(w)))[3])
                adj = [Int[x for x in neighbors(g, v) if x != v] for v in 1:n]
                gg = P.GPUGraph(P.relabel(adj, Float64.(w), P.vorder(adj, :rbfs)))

                for K in (tw, prevfloat(tw), tw + 1)
                    x = P.gpu_decide(gg, K; tol = 0.0, dump = true, dedup = :chunk)
                    y = P.gpu_decide(gg, K; tol = 0.0, dump = true, dedup = :round)
                    @test x.yes == y.yes && x.nextK == y.nextK && x.stats[:dump] == y.stats[:dump] && x.I == y.I
                    # (float widths: the sums of the device may differ in the last bit)
                    kind === :int && @test x.yes == (K >= tw)
                end
            end
        end

        @testset "the race of the CPU and the GPU" begin
            gs = [graphs(rng, 24, 0.25, 2); graphs(rng, 40, 0.12, 2); graphs(rng, 70, 0.0, 1)]

            for engine in (:septw, :pid, :auto), g in gs
                w = weights(rng, :int, nv(g)); hi = typemax(Int)
                cpu = run(CPUSeptw(), w, g, 0, hi)
                res = run(GPUSeptw(; engine, budget = 0.0), w, g, 0, hi)
                @test res[3:4] == cpu[3:4]
                @test treewidth(w, g; alg = res[1]) == cpu[3]
                engine === :septw && @test res == cpu
                res = run(GPUSeptw(; engine), w, g, 0, hi)
                @test res[3:4] == cpu[3:4]
            end
        end

        @testset "the CPU, when the GPU is busy or cannot take the graph" begin
            g = only(graphs(rng, 40, 0.12, 1)); w = weights(rng, :int, 40); hi = typemax(Int)
            cpu = run(CPUSeptw(), w, g, 0, hi)

            # another task holds the GPU
            held = Channel{Nothing}(1); done = Channel{Nothing}(1)

            task = Threads.@spawn lock(ext.LOCK) do
                put!(held, nothing); take!(done)
            end

            # (with an interactive thread, the request waits for the GPU and
            # the CPU answers; without one, the round finds it busy)
            take!(held); ext.resetstats!()
            @test run(GPUSeptw(; budget = 0.0), w, g, 0, hi) == cpu
            @test iszero(get(ext.STATS, :gpuwon, 0))
            ext.worker!() || @test get(ext.STATS, :busy, 0) > 0
            put!(done, nothing); wait(task)

            # too many vertices for the sets of sepenum and sepdp
            big = BipartiteGraph(cycle_graph(ext.NMAX + 1))
            @test isnothing(CliqueTrees.bt_esc(GPUSeptw(; engine = :septw), ones(Int, ext.NMAX + 1), big, Inf))
            @test run(GPUSeptw(; budget = 0.0), ones(Int, ext.NMAX + 1), big, 0, hi)[3:4] == (3, BT_FOUND)

            # a budget that the CPU does not spend
            ext.resetstats!()
            @test run(GPUSeptw(; budget = Inf), w, g, 0, hi) == cpu
            @test iszero(get(ext.STATS, :escalated, 0))
        end
    end
end
