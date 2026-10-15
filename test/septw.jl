using CliqueTrees
using CliqueTrees: BipartiteGraph, BT_FOUND, BT_ABOVE, BT_TIMEOUT
using Graphs
using Random
using Test

@testset "widths from separators (septw, pmctw)" begin
    function bipartite(graph)
        n = nv(graph); ptr = [1]; tgt = Int[]

        for v in 1:n
            append!(tgt, sort(neighbors(graph, v))); push!(ptr, length(tgt) + 1)
        end

        return BipartiteGraph{Int, Int}(n, n, length(tgt), ptr, tgt)
    end

    # the width, from all potential maximal cliques
    function exact(weights, graph)
        sep, _ = CliqueTrees.sepenum(weights, graph, typemax(eltype(weights)))
        return CliqueTrees.pmcdp(weights, graph, CliqueTrees.pmcenum(graph, sep))[3]
    end

    # floats are summed in other orders
    above(k::Integer) = k
    above(k::AbstractFloat) = k * (1 + 1.0e-12)
    below(k::Integer) = k - 1
    below(k::AbstractFloat) = k * (1 - 1.0e-12)

    # an ordering of width k <= hi, or the identity and a bound
    function check(f, weights, graph, lo, hi, tw)
        n = nv(graph)
        perm, invp, k, status = f(weights, graph, lo, hi)
        @test invp[perm] == 1:n

        if status == BT_FOUND
            @test sort(perm) == 1:n
            @test k <= above(hi)
            @test treewidth(weights, graph; alg = perm) <= above(k)
            @test k <= above(max(lo, tw))
        else
            @test status == BT_ABOVE
            @test perm == 1:n
            @test hi < k <= above(tw)
        end

        return k, status
    end

    @testset "random graphs" begin
        rng = MersenneTwister(1)

        for _ in 1:200
            n = rand(rng, 0:16); p = rand(rng, (0.1, 0.2, 0.3, 0.5, 0.8))
            graph = bipartite(erdos_renyi(n, p; rng))

            for weights in (ones(Int, n), rand(rng, 1:5, n), rand(rng, n) .+ 0.5)
                W = eltype(weights); tw = exact(weights, graph)

                for f in (CliqueTrees.septw, CliqueTrees.pmctw)
                    # from zero: the width
                    k, status = check(f, weights, graph, zero(W), typemax(W), tw)
                    @test status == BT_FOUND && k >= below(tw)

                    # from above the width: bags no heavier than the start
                    check(f, weights, graph, tw + 2, typemax(W), tw)

                    # a cap below the width: a bound above it
                    if !iszero(n)
                        k, status = check(f, weights, graph, zero(W), below(tw), tw)
                        @test status == BT_ABOVE
                    end
                end
            end
        end
    end

    @testset "threads and time" begin
        graph = erdos_renyi(40, 0.3; seed = 1); weights = ones(Int, 40)
        one = CliqueTrees.septw(weights, graph, 0, typemax(Int); nt = 1)
        @test one[4] == BT_FOUND
        @test CliqueTrees.septw(weights, graph, 0, typemax(Int); nt = 2) == one
        @test CliqueTrees.septw(weights, graph, 0, typemax(Int), Inf, 2) == one

        for f in (CliqueTrees.septw, CliqueTrees.pmctw)
            perm, invp, k, status = f(weights, graph, 0, typemax(Int); mt = -1)
            @test status == BT_TIMEOUT && perm == 1:40 && 0 < k <= one[3]
        end
    end

    @testset "vertex types" begin
        # the lists are long; the vertices of the result have the type of the graph
        for (seed, n, p) in ((1, 20, 0.2), (2, 40, 0.1), (3, 70, 0.05))
            graph = erdos_renyi(n, p; seed); weights = rand(Xoshiro(seed), 1:3, n)
            k = CliqueTrees.septw(weights, graph, 0, typemax(Int))[3]

            for G in (BipartiteGraph{Int16, Int32}(graph), Graph{Int16}(graph))
                perm, invp, k16, status = CliqueTrees.septw(weights, G, 0, typemax(Int))
                @test eltype(perm) == Int16 && status == BT_FOUND && k16 == k
            end
        end
    end

    @testset "special cases" begin
        for f in (CliqueTrees.septw, CliqueTrees.pmctw)
            # no vertices
            @test f(Int[], Graph(0), 0, typemax(Int)) == (Int[], Int[], 0, BT_FOUND)

            # isolated vertices, a path, a complete graph, several components
            for graph in (Graph(4), path_graph(7), complete_graph(6), blockdiag(cycle_graph(5), complete_graph(4)), grid((3, 4)))
                n = nv(graph); g = bipartite(graph)
                check(f, collect(1:n), g, 0, typemax(Int), exact(collect(1:n), g))
                check(f, ones(Int, n), g, 0, typemax(Int), exact(ones(Int, n), g))
            end

            # a cap below every bag
            @test f(ones(Int, 5), bipartite(path_graph(5)), 0, 1)[3:4] == (2, BT_ABOVE)
        end

        # more than 64 vertices
        for graph in (grid((3, 30)), erdos_renyi(100, 2.2 / 100; seed = 7))
            n = nv(graph); g = bipartite(graph); weights = rand(Xoshiro(n), 1:3, n)
            k, status = check(CliqueTrees.septw, weights, g, 0, typemax(Int), CliqueTrees.septw(weights, g, 0, typemax(Int))[3])
            @test status == BT_FOUND
        end
    end
end
