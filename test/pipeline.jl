using CliqueTrees
using CliqueTrees: BipartiteGraph
using Graphs
using Random
using Test

# The hybrid pipeline (HybridTW). The CPU path always; the GPU path only
# with CUDA.jl and a working device.
const PIPE_HASCUDA = !isnothing(Base.find_package("CUDA"))

PIPE_HASCUDA && using CUDA

@testset "hybrid pipeline" begin
    # the width of a graph, exactly: the largest width of a component
    function exact(weights, graph)
        k = zero(eltype(weights))

        for c in connected_components(graph)
            sub, _ = induced_subgraph(graph, c); w = weights[c]

            if nv(sub) == 1
                k = max(k, w[1])
            else
                _, _, kc, status = CliqueTrees.septw(w, BipartiteGraph(sub), 0, typemax(eltype(w)))
                @test status == CliqueTrees.BT_FOUND
                k = max(k, kc)
            end
        end

        return k
    end

    # floats are summed in other orders
    above(k::Integer) = k
    above(k::AbstractFloat) = k * (1 + 1.0e-10)

    function check(weights, graph, alg)
        r = hybridtw(weights, graph, alg)
        k = exact(weights, graph)
        @test isperm(r.order)
        @test length(r.order) == nv(graph)
        @test r.width == treewidth(weights, graph, r.order)
        @test r.lower <= above(k) && k <= above(r.width)
        @test r.lower <= r.width
        r.certified && @test r.width ≈ k
        r.certified && @test r.lower == r.width
        @test issorted(last.(r.trace); rev = true)
        return r
    end

    function graphs(rng, m)
        out = SimpleGraph{Int}[]

        for _ in 1:m
            n = rand(rng, 8:36); p = rand(rng, (0.1, 0.2, 0.35))
            push!(out, erdos_renyi(n, p; rng))
        end

        return out
    end

    @testset "CPU" begin
        rng = Xoshiro(7); ncert = 0

        for g in graphs(rng, 25)
            r = check(ones(Float64, nv(g)), g, HybridTW(; time = 5.0, gpu = false))
            ncert += r.certified
        end

        # these graphs are small: every one is solved
        @test ncert == 25

        # integer and float weights
        for g in graphs(rng, 10)
            check(rand(rng, 1:3, nv(g)), g, HybridTW(; time = 5.0, gpu = false))
            check(1.0 .+ rand(rng, nv(g)), g, HybridTW(; time = 5.0, gpu = false))
        end

        # without exact searches, every atom goes to HBT
        for g in graphs(rng, 5)
            check(ones(Float64, nv(g)), g, HybridTW(; time = 1.0, gpu = false, nexact = 0))
        end

        # small and trivial graphs
        for g in (SimpleGraph(0), SimpleGraph(1), SimpleGraph(5), complete_graph(6), path_graph(7), cycle_graph(9), Graphs.grid([4, 5]))
            if nv(g) == 0
                r = hybridtw(Float64[], g, HybridTW(; time = 1.0, gpu = false))
                @test isempty(r.order)
            else
                check(ones(Float64, nv(g)), g, HybridTW(; time = 2.0, gpu = false))
            end
        end

        # the path of huge graphs (here: any graph with more than 10 vertices)
        for g in graphs(rng, 5)
            r = hybridtw(ones(Float64, nv(g)), g, HybridTW(; time = 1.0, gpu = false, nhuge = 10))
            @test isperm(r.order)
            @test r.width == treewidth(ones(Float64, nv(g)), g, r.order)
            @test r.lower <= exact(ones(Float64, nv(g)), g) <= r.width
        end

        # the cheapest contraction tree: never more costly than a fast ordering
        for trial in 1:5
            n = rand(rng, 12:30); m = rand(rng, 8:20)
            hidx = [sort!(unique(rand(rng, 1:n, rand(rng, 2:4)))) for _ in 1:m]
            used = sort!(unique(reduce(vcat, hidx))); new = zeros(Int, n); new[used] = eachindex(used); n = length(used)
            hidx = [new[e] for e in hidx]; push!(hidx, Int[])                           # the point: no output
            ptr = cumsum([1; length.(hidx)]); tgt = reduce(vcat, hidx)
            H = BipartiteGraph{Int, Int}(n, m + 1, length(tgt), ptr, tgt)
            w = Float64.(rand(rng, 1:2, n)); w[1] = 0.0                                 # a dimension-1 index
            r = hybridcost(w, H, m + 1, HybridTW(; time = 1.0, gpu = false))
            @test isperm(r.order)
            @test r.cost ≈ branchstats(w, H, m + 1, r.parent).cost
            @test all(c -> r.cost <= last(c) + 1.0e-9, r.candidates)
            @test length(r.candidates) == 5
        end

        # as an elimination algorithm
        g = Graphs.grid([5, 6])
        order, index = permutation(ones(Float64, nv(g)), g, HybridTW(; time = 2.0, gpu = false))
        @test isperm(order)
        @test index == invperm(order)
        @test treewidth(g; alg = HybridTW(; time = 2.0, gpu = false)) == 5
    end

    if !PIPE_HASCUDA || !CUDA.functional()
        @testset "GPU (no GPU: skipped)" begin
            @test_skip false
        end
    else
        @testset "GPU" begin
            @test hybridwarmup!(; nmax = 128)
            rng = Xoshiro(8); ncert = 0

            for g in graphs(rng, 25)
                r = check(ones(Float64, nv(g)), g, HybridTW(; time = 5.0))
                ncert += r.certified
            end

            @test ncert == 25

            # a dense graph, whose exact search makes rounds worth the GPU
            for p in (0.3, 0.5)
                g = erdos_renyi(70, p; rng)
                check(ones(Float64, nv(g)), g, HybridTW(; time = 20.0))
            end

            @test all(r -> r.n >= 1, hybridmodel())
        end
    end
end
