using CliqueTrees
using CliqueTrees: BipartiteGraph, nov
using Graphs
using Random
using Test

@testset "potential maximal cliques (pmcenum)" begin
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

    # Bouchitté-Todinca: X is a PMC iff no component D of G - X has N(D) = X,
    # and every nonadjacent pair in X lies in some N(D)
    function ispmc(graph, X)
        cmps = components(graph, X)
        any(S == X for (_, S) in cmps) && return false

        for a in X, b in X
            (a < b && !has_edge(graph, a, b)) || continue
            any(a in S && b in S for (_, S) in cmps) || return false
        end

        return true
    end

    function brute(graph)
        n = nv(graph)
        sets = [[v for v in 1:n if isodd(mask >> (v - 1))] for mask in 1:(2^n - 1)]
        return filter(X -> ispmc(graph, X), sets)
    end

    # run the enumeration on all minimal separators; check that the result
    # is a relation from PMCs to vertices, with sorted lists and no repeats
    function check(graph)
        n = nv(graph)
        sep, _ = CliqueTrees.sepenum(ones(Int, n), graph, typemax(Int))
        rel = CliqueTrees.pmcenum(graph, sep)
        lists = [collect(neighbors(rel, p)) for p in 1:nv(rel)]
        @test nov(rel) == n
        @test all(issorted, lists)
        @test allunique(lists)
        return rel, Set(lists)
    end

    @testset "small graphs against brute force" begin
        for seed in 1:120
            rng = Xoshiro(seed)
            n = rand(rng, 1:11)
            graph = erdos_renyi(n, rand(rng, (0.15, 0.25, 0.35, 0.5, 0.7)); seed)
            _, pmcs = check(graph)
            @test pmcs == Set(brute(graph))
        end
    end

    @testset "exact treewidth with pmcdp" begin
        for seed in 1:40
            rng = Xoshiro(seed)
            n = rand(rng, 2:30)
            graph = erdos_renyi(n, rand(rng, (0.1, 0.2, 0.3)); seed)
            rel, _ = check(graph)

            for weights in (ones(Int, n), rand(rng, 1:4, n))
                order, invp, k, f = CliqueTrees.pmcdp(weights, graph, rel)
                @test sort(order) == 1:n
                @test treewidth(weights, graph; alg = order) <= k
                @test k == treewidth(weights, graph; alg = ConnectedComponents(PIDBT()))
            end
        end
    end

    @testset "special cases" begin
        # no vertices: no PMCs; one vertex: itself
        @test isempty(check(Graph(0))[2])
        @test check(Graph(1))[2] == Set([[1]])

        # a complete graph: one PMC
        @test check(complete_graph(6))[2] == Set([collect(1:6)])

        # a path: its edges
        @test check(path_graph(6))[2] == Set([[v, v + 1] for v in 1:5])

        # isolated vertices: each one
        @test check(Graph(3))[2] == Set([[v] for v in 1:3])

        # self loops and repeated edges do not matter
        graph = erdos_renyi(10, 0.4; seed = 3); n = nv(graph)
        lists = [[collect(neighbors(graph, v)); v; collect(neighbors(graph, v))] for v in 1:n]
        multigraph = BipartiteGraph{Int, Int}(n, n, sum(length, lists), cumsum([1; length.(lists)]), reduce(vcat, lists))
        @test check(multigraph)[2] == check(graph)[2]
    end
end
