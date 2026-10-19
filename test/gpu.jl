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
            gpu = GPUSeptw(; engine = :septw, budget = 0.0)

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

        @testset "PID on the GPU: the CPU widths" begin
            gpu = GPUSeptw(; engine = :pid, budget = 0.0)

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
                gpu = GPUSeptw(; engine, budget = 0.0)

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

        @testset "the CPU, when the GPU is busy or cannot take the graph" begin
            g = only(graphs(rng, 40, 0.12, 1)); w = weights(rng, :int, 40); hi = typemax(Int)
            cpu = run(CPUSeptw(), w, g, 0, hi)

            # another task holds the GPU
            held = Channel{Nothing}(1); done = Channel{Nothing}(1)

            task = Threads.@spawn lock(ext.LOCK) do
                put!(held, nothing); take!(done)
            end

            take!(held); ext.resetstats!()
            @test run(GPUSeptw(; budget = 0.0), w, g, 0, hi) == cpu
            @test get(ext.STATS, :busy, 0) > 0
            put!(done, nothing); wait(task)

            # too many vertices for the sets of sepenum and sepdp
            big = BipartiteGraph(cycle_graph(ext.NMAX + 1))
            @test isnothing(CliqueTrees.bt_esc(GPUSeptw(), ones(Int, ext.NMAX + 1), big, Inf))
            @test run(GPUSeptw(; budget = 0.0), ones(Int, ext.NMAX + 1), big, 0, hi)[3:4] == (3, BT_FOUND)

            # a budget that the CPU does not spend
            ext.resetstats!()
            @test run(GPUSeptw(; budget = Inf), w, g, 0, hi) == cpu
            @test iszero(get(ext.STATS, :escalated, 0))
        end
    end
end
