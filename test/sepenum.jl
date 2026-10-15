using CliqueTrees
using CliqueTrees: BipartiteGraph, nov
using Graphs
using Random
using Test

@testset "minimal separators (sepenum)" begin
    # brute force: the components of G - X (in order of their smallest
    # vertices) and their neighborhoods, as sorted vectors
    function components(graph, X)
        n = nv(graph); inX = falses(n); inX[X] .= true; seen = falses(n); out = Tuple{Vector{Int}, Vector{Int}}[]

        for v0 in 1:n
            (inX[v0] || seen[v0]) && continue
            D = Int[]; queue = [v0]; seen[v0] = true

            while !isempty(queue)
                u = pop!(queue); push!(D, u)

                for x in neighbors(graph, u)
                    if !inX[x] && !seen[x]
                        seen[x] = true; push!(queue, x)
                    end
                end
            end

            S = sort!(unique!([x for u in D for x in neighbors(graph, u) if inX[x]]))
            push!(out, (sort!(D), S))
        end

        return out
    end

    # the minimal separators: the vertex sets with two full components
    function brute(graph)
        n = nv(graph)
        sets = [[v for v in 1:n if isodd(mask >> (v - 1))] for mask in 0:(2^n - 1)]
        return filter(S -> count(((_, T),) -> T == S, components(graph, S)) >= 2, sets)
    end

    # run the generator; check that the result is a relation from
    # separators to vertices, with sorted lists and no repeats
    function check(weights, graph, maxk)
        rel, status = CliqueTrees.sepenum(weights, graph, maxk)
        lists = [collect(neighbors(rel, s)) for s in 1:nv(rel)]
        @test status == CliqueTrees.BT_FOUND
        @test nov(rel) == nv(graph)
        @test all(issorted, lists)
        @test allunique(lists)
        return Set(lists)
    end

    @testset "small graphs against brute force" begin
        for seed in 1:120
            rng = Xoshiro(seed)
            n = rand(rng, 1:11)
            graph = erdos_renyi(n, rand(rng, (0.15, 0.25, 0.35, 0.5, 0.7)); seed)
            seps = brute(graph)

            for weights in (ones(Int, n), rand(rng, 1:4, n))
                for maxk in (typemax(Int), 1, 2, 3, 5, 8)
                    @test check(weights, graph, maxk) == Set(filter(S -> sum(weights[S]; init = 0) < maxk, seps))
                end
            end
        end
    end

    @testset "exact treewidth with sepdp" begin
        # rounds from zero, each with the separators lighter than the bound;
        # a decomposition that needs a separator of weight at least maxk has
        # a bag of weight at least maxk plus the smallest weight, so the next
        # bound is the smaller of that and the one that sepdp returns
        for seed in 1:60
            rng = Xoshiro(seed)
            n = rand(rng, 2:11)
            graph = erdos_renyi(n, rand(rng, (0.2, 0.35, 0.5, 0.7)); seed)

            for weights in (ones(Int, n), rand(rng, 1:4, n))
                maxk = 0

                while true
                    sep, _ = CliqueTrees.sepenum(weights, graph, maxk)
                    order, invp, k, _ = CliqueTrees.sepdp(weights, graph, sep, maxk)

                    if k <= maxk
                        @test sort(order) == 1:n
                        @test treewidth(weights, graph; alg = order) <= k
                        break
                    end

                    maxk = min(k, maxk + minimum(weights))
                end

                @test maxk == treewidth(weights, graph; alg = ConnectedComponents(PIDBT()))
            end
        end
    end

    @testset "special cases" begin
        # no vertices, one vertex, a complete graph: no separators
        @test isempty(check(Int[], Graph(0), typemax(Int)))
        @test isempty(check([1], Graph(1), typemax(Int)))
        @test isempty(check(ones(Int, 6), complete_graph(6), typemax(Int)))

        # a path: every inner vertex
        @test check(ones(Int, 6), path_graph(6), typemax(Int)) == Set([[v] for v in 2:5])

        # isolated vertices: ∅ only
        @test check(ones(Int, 3), Graph(3), typemax(Int)) == Set([Int[]])

        # self loops and repeated edges do not matter
        graph = erdos_renyi(10, 0.4; seed = 3); n = nv(graph)
        lists = [[collect(neighbors(graph, v)); v; collect(neighbors(graph, v))] for v in 1:n]
        multigraph = BipartiteGraph{Int, Int}(n, n, sum(length, lists), cumsum([1; length.(lists)]), reduce(vcat, lists))
        @test check(ones(Int, n), multigraph, typemax(Int)) == check(ones(Int, n), graph, typemax(Int))

        # weights of another type
        weights = rand(Xoshiro(1), 1:3, n)
        @test check(weights ./ 2, graph, 2.5) == check(weights, graph, 5)
        @test check(weights ./ 2, graph, Inf) == check(weights, graph, typemax(Int))
    end

    @testset "threads" begin
        # several tasks list the same separators, in the same order
        lists(rel) = [collect(neighbors(rel, s)) for s in 1:nv(rel)]

        for seed in 1:40
            rng = Xoshiro(seed)
            n = rand(rng, 0:24)
            graph = erdos_renyi(n, rand(rng, (0.1, 0.2, 0.35, 0.5)); seed)

            for weights in (ones(Int, n), rand(rng, 1:4, n)), maxk in (typemax(Int), 3, 6, 9)
                one = lists(first(CliqueTrees.sepenum(weights, graph, maxk; nt = 1)))

                for nt in (2, 3, 7)
                    @test lists(first(CliqueTrees.sepenum(weights, graph, maxk, Inf, nt))) == one
                end
            end
        end
    end

    @testset "deadline" begin
        # a search that runs out of time gives an empty list and
        # BT_TIMEOUT; with no time limit, the list is the same
        lists(rel) = [collect(neighbors(rel, s)) for s in 1:nv(rel)]
        graph = erdos_renyi(40, 0.3; seed = 1); weights = ones(Int, 40)
        all, status = CliqueTrees.sepenum(weights, graph, typemax(Int))
        @test status == CliqueTrees.BT_FOUND && nv(all) > 1000

        for nt in (1, 2)
            sep, status = CliqueTrees.sepenum(weights, graph, typemax(Int); nt, mt = -1)
            @test status == CliqueTrees.BT_TIMEOUT && iszero(nv(sep))
            sep, status = CliqueTrees.sepenum(weights, graph, typemax(Int), time() - 1, nt)
            @test status == CliqueTrees.BT_TIMEOUT && iszero(nv(sep))
            sep, status = CliqueTrees.sepenum(weights, graph, typemax(Int); nt, mt = Inf)
            @test status == CliqueTrees.BT_FOUND && lists(sep) == lists(all)
        end
    end
end
