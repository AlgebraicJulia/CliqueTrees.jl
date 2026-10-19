using CliqueTrees
using CliqueTrees: BipartiteGraph, BT_FOUND, BT_ABOVE, BT_TIMEOUT
using Graphs
using Random
using Test

@testset "contraction-recursive treewidth (RTW)" begin
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
    same(a::Integer, b::Integer) = a == b
    same(a, b) = isapprox(a, b; rtol = 1.0e-9)
    below(k::Integer) = k - 1
    below(k::AbstractFloat) = k * (1 - 1.0e-6)

    # the contraction of the graph by the parts of a certificate map, with
    # each vertex as light as its part; the parts must be connected
    function contraction(weights, graph, cert)
        m = maximum(cert); H = SimpleGraph(m)
        @test sort(unique(cert)) == 1:m

        for y in 1:m
            part = findall(==(y), cert)
            @test is_connected(induced_subgraph(graph, part)[1])
        end

        for e in edges(graph)
            x = cert[src(e)]; y = cert[dst(e)]
            x != y && add_edge!(H, x, y)
        end

        return [minimum(weights[cert .== y]) for y in 1:m], bipartite(H)
    end

    # an ordering of width k, as the result says
    function checkorder(weights, graph, perm, invp, k)
        n = nv(graph)
        @test sort(perm) == 1:n
        @test invp[perm] == 1:n
        @test same(treewidth(weights, graph; alg = collect(perm)), k)
        return
    end

    @testset "random graphs" begin
        rng = MersenneTwister(1)

        for _ in 1:150
            n = rand(rng, 1:14); p = rand(rng, (0.2, 0.3, 0.5, 0.8))
            graph = erdos_renyi(n, p; rng)
            is_connected(graph) || continue
            bgraph = bipartite(graph)

            # weights are at least one (in tensor networks, they are logs of
            # bond dimensions)
            for weights in (ones(Int, n), rand(rng, 1:5, n), rand(rng, n) .+ 1.0)
                W = eltype(weights); tw = exact(weights, bgraph)

                # from no target: the width, with a certificate unless the
                # width is that of the heaviest vertex or edge
                perm, invp, k, status, cert = CliqueTrees.rtw(weights, bgraph, zero(W), typemax(W))
                @test status == BT_FOUND
                @test same(k, tw)
                checkorder(weights, bgraph, perm, invp, k)

                if !isempty(cert)
                    cweights, cgraph = contraction(weights, graph, cert)
                    @test same(exact(cweights, cgraph), k)
                end

                # without the heuristic search
                perm, invp, k, status, _ = CliqueTrees.rtw(weights, bgraph, zero(W), typemax(W); hbt = nothing)
                @test status == BT_FOUND && same(k, tw)
                checkorder(weights, bgraph, perm, invp, k)

                # with fewer ticks of heuristic PID
                perm, invp, k, status, cert = CliqueTrees.rtw(weights, bgraph, zero(W), typemax(W); ticks = rand(rng, (1, 10)))
                @test status == BT_FOUND && same(k, tw)
                checkorder(weights, bgraph, perm, invp, k)

                if !isempty(cert)
                    cweights, cgraph = contraction(weights, graph, cert)
                    @test same(exact(cweights, cgraph), k)
                end

                # a target above the width: a decomposition that meets it
                perm, invp, k, status, cert = CliqueTrees.rtw(weights, bgraph, 2tw, typemax(W))
                @test status == BT_FOUND && k <= 2tw && isempty(cert)
                checkorder(weights, bgraph, perm, invp, k)

                # a cap below the width: every decomposition is heavier, with
                # a certificate unless the heaviest vertex or edge is
                if n > 1
                    hi = below(tw)
                    perm, invp, k, status, cert = CliqueTrees.rtw(weights, bgraph, zero(W), hi)
                    @test status == BT_ABOVE && k > hi
                    checkorder(weights, bgraph, perm, invp, k)

                    if isempty(cert)
                        @test CliqueTrees.bt_bas(weights, bgraph) > hi
                    else
                        cweights, cgraph = contraction(weights, graph, cert)
                        @test exact(cweights, cgraph) > hi
                    end
                end
            end
        end
    end

    @testset "the first heuristic search" begin
        # graphs above base vertices start with an HBT search, on one task or
        # several
        rng = MersenneTwister(4)

        for _ in 1:12
            n = rand(rng, 15:24); graph = erdos_renyi(n, 0.25; rng)
            is_connected(graph) || continue
            bgraph = bipartite(graph)

            for weights in (ones(Int, n), rand(rng, 1:4, n)), nt in (1, 3)
                tw = treewidth(weights, graph; alg = ConnectedComponents(PIDBT()))
                hbt = HBT(; time = 0.2, base = 8, seed = rand(rng, 1:100))
                perm, invp, k, status, _ = CliqueTrees.rtw(weights, bgraph, 0, typemax(Int); hbt, nt)
                @test status == BT_FOUND && k == tw
                checkorder(weights, bgraph, perm, invp, k)
            end
        end
    end

    @testset "the heuristic PID" begin
        # from a poor decomposition (a random order): every PMC it makes is a
        # PMC of the graph, and when its width is at most K, the PMCs it
        # hands out admit a decomposition of width at most K
        rng = MersenneTwister(7); found = 0

        for _ in 1:80
            n = rand(rng, 4:18); graph = erdos_renyi(n, rand(rng, (0.2, 0.3, 0.5)); rng)
            is_connected(graph) || continue

            for weights in (ones(Int, n), rand(rng, 1:4, n))
                bgraph = bipartite(graph); ctx = CliqueTrees.hbt_ctx(bgraph, weights)
                sep, _ = CliqueTrees.sepenum(weights, bgraph, typemax(Int))
                pmc = CliqueTrees.pmcenum(bgraph, sep); tw = CliqueTrees.pmcdp(weights, bgraph, pmc)[3]
                pmcs = Set(sort(collect(neighbors(pmc, p))) for p in vertices(pmc))

                label, tree = CliqueTrees.rtw_min(bgraph, weights, randperm(rng, n))
                hp = CliqueTrees.hpid(ctx, tw)
                CliqueTrees.hpid_import!(hp, [sort([label[v] for v in clique]) for clique in tree])
                CliqueTrees.hpid_improve!(hp, 10^6, Inf)

                for x in 1:CliqueTrees.bt_tab_len(hp.pmcs)
                    X = [v for v in 1:n if !iszero(CliqueTrees.hpid_wrd(hp.pmcs, x, cld(v, 64)) & (one(UInt64) << ((v - 1) % 64)))]
                    @test X in pmcs
                end

                if CliqueTrees.hpid_width(hp) <= tw
                    found += 1; sol = CliqueTrees.hbt_sol(Int)

                    for X in CliqueTrees.hpid_export!(hp)
                        CliqueTrees.hbt_add!(ctx, sol, X)
                    end

                    @test first(CliqueTrees.hbt_dp!(ctx, sol))[1] <= tw
                end
            end
        end

        @test found > 0
    end

    @testset "a deep recursion" begin
        # the recursion can be as deep as the graph has vertices, so it
        # runs on a task with a stack to match: a million calls do not fit
        # in the stack of the main task
        deep(k) = iszero(k) ? 0 : 1 + deep(k - 1)
        @test CliqueTrees.rtw_deep(() -> deep(10^6), 10^4) == 10^6
    end

    @testset "time limit" begin
        graph = bipartite(erdos_renyi(40, 0.3; seed = 3)); weights = ones(Int, 40)
        upper = treewidth(weights, graph; alg = MinimalChordal(MF(; strategy = 1)))

        # out of time at once: the lightest ordering so far
        perm, invp, k, status, cert = CliqueTrees.rtw(weights, graph, 0, typemax(Int); mt = 0)
        @test status == BT_TIMEOUT && isempty(cert) && k <= upper
        checkorder(weights, graph, perm, invp, k)

        elapsed = @elapsed width = treewidth(weights, graph; alg = RTW(; time = 0.5))
        @test width <= upper
        @test elapsed < 30

        # the pieces of a batch share the time limit
        rng = MersenneTwister(5)
        pieces = [erdos_renyi(70, 0.25; rng) for _ in 1:3]
        graph3 = blockdiag(blockdiag(pieces[1], pieces[2]), pieces[3])
        treewidth(graph3; alg = SafeRules(SafeSeparators(RTW(; time = 0.1))))   # compile
        elapsed = @elapsed treewidth(graph3; alg = RTW(; time = 1.0))
        @test elapsed < 2.5
        elapsed = @elapsed treewidth(graph3; alg = SafeRules(SafeSeparators(RTW(; time = 1.0))))
        @test elapsed < 2.5
    end

    @testset "disconnected graphs and pipelines" begin
        rng = MersenneTwister(2)

        for _ in 1:20
            graph = blockdiag(erdos_renyi(8, 0.4; rng), erdos_renyi(9, 0.5; rng))
            weights = rand(rng, 1:4, nv(graph))
            tw = treewidth(weights, graph; alg = ConnectedComponents(PIDBT()))
            @test treewidth(weights, graph; alg = RTW()) == tw
            @test treewidth(weights, graph; alg = RTW(; hbt = nothing)) == tw
            @test treewidth(weights, graph; alg = RTW(; ticks = 1)) == tw
            @test treewidth(weights, graph; alg = SafeRules(SafeSeparators(RTW()))) == tw

            # the components as a batch, without the exact search that each
            # graph of a batch gets first (it solves graphs this small), so
            # that the recursion shares its lower bound: the batch is as
            # wide as the graph
            subs = [induced_subgraph(graph, c) for c in connected_components(graph)]
            parts = first.(subs); pweights = [weights[last(s)] for s in subs]
            orders = CliqueTrees.rtw_batch(pweights, parts, RTW(), Inf, 0)
            @test maximum(treewidth(pweights[i], parts[i]; alg = orders[i]) for i in eachindex(parts)) == tw
        end
    end
end
