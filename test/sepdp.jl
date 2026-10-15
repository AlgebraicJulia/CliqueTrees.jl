using CliqueTrees
using CliqueTrees: BipartiteGraph, nov
using Graphs
using Random
using Test

@testset "decision DP over minimal separators (sepdp)" begin
    function relation(n, sets)
        ptr = cumsum([1; length.(sets)]); tgt = Int[x for X in sets for x in X]
        return BipartiteGraph{Int, Int}(n, length(sets), length(tgt), ptr, tgt)
    end

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
    function allseps(graph)
        n = nv(graph)
        sets = [[v for v in 1:n if isodd(mask >> (v - 1))] for mask in 0:(2^n - 1)]
        return filter(S -> count(((_, T),) -> T == S, components(graph, S)) >= 2, sets)
    end

    # the separators of the components of G - X, for X in sets
    function separators(graph, sets)
        return unique!([S for X in sets for (_, S) in components(graph, X)])
    end

    # the maximal cliques of k random minimal triangulations
    function triangulations(weights, graph, rng, k)
        n = nv(graph); sets = Vector{Int}[]

        for _ in 1:k
            order, _ = permutation(weights, graph; alg = MinimalChordal(randperm(rng, n)))
            label, tree = cliquetree(weights, graph; alg = order)

            for clique in tree
                push!(sets, sort!([label[v] for v in clique]))
            end
        end

        return sets
    end

    # run the program with the bound maxk; if it succeeds, check that the
    # order is a permutation whose width is at most the reported one
    function check(weights, graph, seps, maxk)
        n = nv(graph)
        order, invp, k, status = CliqueTrees.sepdp(weights, graph, relation(n, seps), maxk)
        @test status == (k <= maxk ? CliqueTrees.BT_FOUND : CliqueTrees.BT_ABOVE)
        @test sort(order) == 1:n
        @test invp[order] == 1:n

        if k <= maxk
            @test treewidth(weights, graph; alg = order) <= k
        else
            @test order == 1:n
        end

        return k
    end

    # rounds from zero, each with the bound returned by the last
    function ladder(weights, graph, seps)
        maxk = zero(eltype(weights))

        while true
            k = check(weights, graph, seps, maxk)
            k <= maxk && return k
            maxk = k
        end
    end

    # check the blocks against brute force
    function structure(weights, graph, seps)
        W = eltype(weights)
        n = nv(graph); S = relation(n, seps); ns = nv(S)
        work1 = Vector{Int}(undef, n)
        adj = CliqueTrees.bt_adj!(CliqueTrees.bt_mat(cld(n, 64), n), graph)

        # phase 1: the blocks, in the order of the separators
        sepwgt = Vector{W}(undef, ns)
        lstmin, lstlen, lstwgt, lstsep, lstset = CliqueTrees.sepdp_blk!(sepwgt, adj, weights, graph, S)
        nb = length(lstmin)
        ok = all(sepwgt[s] == sum(weights[seps[s]]; init = zero(W)) for s in 1:ns)

        # phase 2: in order of size, with the blocks grouped by smallest vertex
        blkmin = Vector{Int}(undef, nb); blklen = Vector{Int}(undef, nb); blkwgt = Vector{W}(undef, nb)
        blksep = Vector{Int}(undef, nb); order = Vector{Int}(undef, nb)
        fst = CliqueTrees.sepdp_srt!(blkmin, blklen, blkwgt, blksep, order, work1, lstmin, lstlen, lstwgt, lstsep, n)
        ok &= issorted(blklen) && nv(fst) == n && nov(fst) == nb
        ok &= all(neighbors(fst, v) == [b for b in 1:nb if blkmin[b] == v] for v in 1:n)

        # phase 3: the components and separators as packed sets
        blkset, sepset = CliqueTrees.sepdp_set!(adj, S, blksep, order, lstset, n)
        unpack(set, b) = [v for v in 1:n if isodd(CliqueTrees.bt_get(set, (v - 1) >> 6 + 1, b) >> ((v - 1) & 63))]
        blocks = [(unpack(sepset, b), unpack(blkset, b)) for b in 1:nb]

        for b in 1:nb
            T, D = blocks[b]
            ok &= T == sort(seps[blksep[b]]) && blkmin[b] == D[1] && blklen[b] == length(D) && blkwgt[b] == sum(weights[D])
        end

        # the blocks are the small full components of the separators
        ok &= sort(blocks) == sort([(sort(T), D) for T in seps for (D, U) in components(graph, T) if U == sort(T) && 2length(D) <= n - length(T)])
        return ok
    end

    @testset "all minimal separators of small graphs" begin
        for seed in 1:40
            rng = Xoshiro(seed)
            n = rand(rng, 3:10)
            graph = erdos_renyi(n, rand(rng, (0.2, 0.35, 0.5, 0.7)); seed)
            seps = allseps(graph)

            for weights in (ones(Int, n), rand(rng, 1:4, n))
                width = treewidth(weights, graph; alg = ConnectedComponents(PIDBT()))

                # success at the treewidth; below it, a lower bound
                @test check(weights, graph, seps, width) == width
                @test width - 1 < check(weights, graph, seps, width - 1) <= width
                @test ladder(weights, graph, seps) == width
                @test structure(weights, graph, seps)

                # the minimal separators lighter than the bound suffice
                light(maxk) = filter(T -> sum(weights[T]; init = 0) < maxk, seps)
                @test check(weights, graph, light(width), width) == width
                @test check(weights, graph, light(width - 1), width - 1) > width - 1
            end
        end
    end

    @testset "separators of minimal triangulations" begin
        for (n, p) in ((30, 0.1), (40, 0.08), (50, 0.15), (64, 0.05), (100, 0.04))
            rng = Xoshiro(n)
            graph = erdos_renyi(n, p; seed = n)

            for weights in (ones(Int, n), rand(rng, 1:3, n))
                sets = triangulations(weights, graph, rng, 10)
                seps = separators(graph, sets)
                _, _, width = CliqueTrees.pmcdp(weights, graph, relation(n, sets))

                # at least as good as the potential maximal cliques of the
                # triangulations
                @test check(weights, graph, seps, width) <= width
                @test structure(weights, graph, seps)
            end
        end
    end

    @testset "special cases" begin
        # a complete graph: no separators, one bag
        graph = complete_graph(6); weights = collect(1:6)
        @test check(weights, graph, Vector{Int}[], 21) == 21
        @test check(weights, graph, Vector{Int}[], 20) == 21

        # a path
        graph = path_graph(6)
        @test check(ones(Int, 6), graph, allseps(graph), 2) == 2
        @test check(ones(Int, 6), graph, allseps(graph), 1) == 2

        # no vertices
        @test CliqueTrees.sepdp(Int[], Graph(0), relation(0, Vector{Int}[]), 0)[3] == 0

        # disconnected graphs and isolated vertices (∅ is a separator)
        graph = blockdiag(blockdiag(cycle_graph(5), path_graph(3)), Graph(2))
        weights = [3, 1, 2, 2, 1, 1, 5, 1, 4, 1]
        width = treewidth(weights, graph; alg = ConnectedComponents(PIDBT()))
        seps = allseps(graph)
        @test [] in seps
        @test check(weights, graph, seps, width) == width
        @test check(weights, graph, seps, width - 1) > width - 1
        @test ladder(weights, graph, seps) == width
        @test structure(weights, graph, seps)

        # self loops and repeated edges do not matter
        rng = Xoshiro(1)
        graph = erdos_renyi(10, 0.4; seed = 3); n = nv(graph)
        weights = rand(rng, 1:3, n); seps = allseps(graph)
        width = treewidth(weights, graph; alg = ConnectedComponents(PIDBT()))
        lists = [[collect(neighbors(graph, v)); v; collect(neighbors(graph, v))] for v in 1:n]
        multigraph = BipartiteGraph{Int, Int}(n, n, sum(length, lists), cumsum([1; length.(lists)]), reduce(vcat, lists))
        @test CliqueTrees.sepdp(weights, multigraph, relation(n, seps), width) == CliqueTrees.sepdp(weights, graph, relation(n, seps), width)

        # unsorted separators
        @test CliqueTrees.sepdp(weights, graph, relation(n, reverse.(seps)), width)[3] == width

        # weights of another type
        order, invp, k, status = CliqueTrees.sepdp(weights, graph, relation(n, seps), width)
        @test CliqueTrees.sepdp(weights ./ 2, graph, relation(n, seps), width / 2) == (order, invp, k / 2, status)
    end

    @testset "deadline" begin
        # a search that runs out of time gives BT_TIMEOUT and the
        # identity; with no time limit, the result is the same
        graph = erdos_renyi(40, 0.3; seed = 1); weights = ones(Int, 40)
        sep, _ = CliqueTrees.sepenum(weights, graph, typemax(Int))
        @test nv(sep) > 2000
        result = CliqueTrees.sepdp(weights, graph, sep, 1000)
        @test result[4] == CliqueTrees.BT_FOUND

        for nt in (1, 2)
            @test CliqueTrees.sepdp(weights, graph, sep, 1000; nt, mt = -1) == (1:40, 1:40, 0, CliqueTrees.BT_TIMEOUT)
            @test CliqueTrees.sepdp(weights, graph, sep, 1000, time() - 1, nt) == (1:40, 1:40, 0, CliqueTrees.BT_TIMEOUT)
            @test CliqueTrees.sepdp(weights, graph, sep, 1000; nt, mt = Inf) == result
        end
    end

    @testset "threads" begin
        # several tasks give the same result
        for seed in 1:40
            rng = Xoshiro(seed)
            n = rand(rng, 0:24)
            graph = erdos_renyi(n, rand(rng, (0.1, 0.2, 0.35, 0.5)); seed)

            for weights in (ones(Int, n), rand(rng, 1:4, n)), maxk in (3, 6, 9, 30)
                sep, _ = CliqueTrees.sepenum(weights, graph, maxk)
                one = CliqueTrees.sepdp(weights, graph, sep, maxk; nt = 1)

                for nt in (2, 3, 7)
                    @test CliqueTrees.sepdp(weights, graph, sep, maxk; nt) == one
                end
            end
        end

        # large enough for the phases to use several tasks
        for (seed, graph) in enumerate((Graphs.grid([5, 5]), Graphs.grid([4, 7]), erdos_renyi(30, 0.15; seed = 1), erdos_renyi(32, 0.12; seed = 2)))
            n = nv(graph); rng = Xoshiro(seed)

            for weights in (ones(Int, n), rand(rng, 1:3, n)), maxk in (typemax(Int), 9, 12)
                sep, _ = CliqueTrees.sepenum(weights, graph, maxk)
                k = min(maxk, sum(weights))
                one = CliqueTrees.sepdp(weights, graph, sep, k; nt = 1)
                @test nv(sep) >= CliqueTrees.SEPDP_MT || maxk < typemax(Int)

                for nt in (2, 3)
                    @test CliqueTrees.sepdp(weights, graph, sep, k; nt) == one
                end
            end
        end
    end
end
