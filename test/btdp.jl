using CliqueTrees
using CliqueTrees: BipartiteGraph, nov
using Graphs
using Random
using Test

@testset "BT dynamic program over PMCs (btdp)" begin

    function relation(n, sets)
        ptr = cumsum([1; length.(sets)]); tgt = reduce(vcat, sets; init = Int[])
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

    function allpmcs(graph)
        n = nv(graph)
        sets = [[v for v in 1:n if isodd(mask >> (v - 1))] for mask in 1:(2^n - 1)]
        return filter(X -> ispmc(graph, X), sets)
    end

    # the maximal cliques of k random minimal triangulations, and the
    # smallest of their widths
    function triangulations(weights, graph, rng, k)
        n = nv(graph); sets = Vector{Int}[]; best = typemax(Int)

        for _ in 1:k
            order, _ = permutation(weights, graph; alg = MinimalChordal(randperm(rng, n)))
            best = min(best, treewidth(weights, graph; alg = order))
            label, tree = cliquetree(weights, graph; alg = order)

            for clique in tree
                push!(sets, sort!([label[v] for v in clique]))
            end
        end

        return sets, best
    end

    # the value that HBT's dynamic program gives the same PMCs
    function hbtvalue(weights, graph, sets)
        g, wgt = CliqueTrees.hbt_graph(weights, graph)
        ctx = CliqueTrees.hbt_ctx(g, wgt); sol = CliqueTrees.hbt_sol(eltype(wgt))

        for X in sets
            CliqueTrees.hbt_add!(ctx, sol, sort(X))
        end

        width, _ = CliqueTrees.hbt_dp!(ctx, sol)
        return width
    end

    # run the program; check that the order is a permutation whose width is
    # at most the reported one
    function check(weights, graph, sets)
        n = nv(graph)
        order, invp, k, f = CliqueTrees.pmcdp(weights, graph, relation(n, sets))
        @test sort(order) == 1:n
        @test invp[order] == 1:n
        @test treewidth(weights, graph; alg = order) <= k
        return (k, f)
    end

    # check every phase against brute force
    function structure(weights, graph, sets)
        W = eltype(weights)
        n = nv(graph); C = relation(n, sets); np = nv(C)

        # phase 1: the components; inpmc -> work1, insep -> work3, queue -> work4
        work1 = Vector{Int}(undef, n); work3 = Vector{Int}(undef, n); work4 = Vector{Int}(undef, n)
        cmpptr = Vector{Int}(undef, np + 1)
        cmps, owner, sep, cfirst, csize, cweight =
            CliqueTrees.pmcdp_cmp!(cmpptr, work1, work3, work4, weights, graph, C)
        nc = nv(sep); S = [sort(neighbors(sep, c)) for c in 1:nc]
        ok = nv(cmps) == length(sets) && nov(cmps) == nc

        for (p, X) in enumerate(sets)
            cmpsX = components(graph, X)
            ok &= length(neighbors(cmps, p)) == length(cmpsX)

            for (c, (D, T)) in zip(neighbors(cmps, p), cmpsX)
                ok &= owner[c] == p && cfirst[c] == D[1] && csize[c] == length(D) && cweight[c] == sum(weights[D]) && S[c] == T
            end
        end

        # phase 2: the separators; var, flag, size are reused by phase 3
        new = CliqueTrees.FScalar{Int}(undef)
        var = Vector{Int}(undef, nc + 1); svar = Vector{Int}(undef, nc)
        flag = Vector{Int}(undef, nc); size = Vector{Int}(undef, nc)
        nsep = CliqueTrees.pmcdp_sep!(new, var, svar, flag, size, sep)
        sepid = svar
        ok &= sort!(unique(sepid)) == 1:nsep
        ok &= all((sepid[c] == sepid[d]) == (S[c] == S[d]) for c in 1:nc, d in 1:nc)

        # phase 3: the children; var -> chdptr, flag -> outmin, size -> outlen, insep -> work3
        outwgt = Vector{W}(undef, nc)
        children = CliqueTrees.pmcdp_chd!(var, flag, size, outwgt, work3, weights, C, cmps, sep, cfirst, csize, cweight)
        ofirst = flag; osize = size; oweight = outwgt

        for c in 1:nc
            p = owner[c]
            ok &= neighbors(children, c) == [d for d in neighbors(cmps, p) if !issubset(S[d], S[c])]

            # the outer side is the full component of G - S_c that contains Ωₚ - S_c
            rest = setdiff(sets[p], S[c])
            D, T = only(filter(((D, T),) -> rest[1] in D, components(graph, S[c])))
            ok &= issubset(rest, D) && T == S[c]
            ok &= ofirst[c] == D[1] && osize[c] == length(D) && oweight[c] == sum(weights[D])
        end

        # phase 4: the blocks
        keymin = Vector{Int}(undef, 2nc); keylen = Vector{Int}(undef, 2nc); keysep = Vector{Int}(undef, 2nc)
        sub = Vector{Int}(undef, nc); captgt = Vector{Int}(undef, nc)
        bmin = Vector{Int}(undef, 2nc); bcmp0 = Vector{Int}(undef, max(2nc, n + 1)); bwgt = Vector{Int}(undef, 2nc)
        cptr = Vector{Int}(undef, 2nc + 1)
        caps =
            CliqueTrees.pmcdp_blk!(keymin, keylen, keysep, sub, captgt,
                bmin, bwgt, bcmp0, cptr,
                weights, sep, sepid, nsep, cfirst, csize, cweight, ofirst, osize, oweight)
        nb = nv(caps); key = [(S[bcmp0[b]], bmin[b]) for b in 1:nb]
        ok &= allunique(key)

        for c in 1:nc
            ok &= key[sub[c]] == (S[c], cfirst[c]) && bwgt[sub[c]] == sum(weights[S[c]]) + cweight[c]
        end

        # every component caps exactly one block: the one keyed by its outer side,
        # and its children's sub-blocks come before it
        ok &= sort!([c for b in 1:nb for c in neighbors(caps, b)]) == 1:nc

        for b in 1:nb, c in neighbors(caps, b)
            ok &= key[b] == (S[c], ofirst[c]) && bwgt[b] == sum(weights[S[c]]) + oweight[c]
            ok &= all(sub[d] < b for d in neighbors(children, c))
        end
        return ok
    end

    @testset "all PMCs of small graphs" begin
        for seed in 1:40
            rng = Xoshiro(seed)
            n = rand(rng, 3:10)
            graph = erdos_renyi(n, rand(rng, (0.2, 0.35, 0.5, 0.7)); seed)
            sets = allpmcs(graph)

            for weights in (ones(Int, n), rand(rng, 1:4, n))
                width = check(weights, graph, sets)
                @test width[1] == treewidth(weights, graph; alg = ConnectedComponents(PIDBT()))
                @test width == hbtvalue(weights, graph, sets)
                @test structure(weights, graph, sets)

                # subsets of the PMCs: fallback bags, possibly no caps at all
                for q in (0.1, 0.3, 0.6)
                    part = sets[randsubseq(rng, eachindex(sets), q)]
                    isempty(part) && continue
                    @test check(weights, graph, part) == hbtvalue(weights, graph, part)
                    @test structure(weights, graph, part)
                end
            end
        end
    end

    @testset "PMCs of minimal triangulations" begin
        for (n, p) in ((30, 0.1), (40, 0.08), (50, 0.15), (64, 0.05), (100, 0.04))
            rng = Xoshiro(n)
            graph = erdos_renyi(n, p; seed = n)

            for weights in (ones(Int, n), rand(rng, 1:3, n))
                sets, best = triangulations(weights, graph, rng, 10)
                width = check(weights, graph, sets)
                @test width[1] <= best
                @test width == hbtvalue(weights, graph, sets)
                @test structure(weights, graph, sets)

                # repeated PMCs change nothing
                @test check(weights, graph, [sets; sets]) == width
            end
        end
    end

    @testset "special cases" begin
        # a complete graph: one PMC, with no components
        graph = complete_graph(6); weights = collect(1:6)
        @test check(weights, graph, [collect(1:6)]) == (21, 1)

        # no PMCs: one bag
        order, invp, k, f = CliqueTrees.pmcdp(weights, graph, relation(nv(graph), Vector{Int}[]))
        @test sort(order) == 1:6 && (k, f) == (21, 1)

        # PMCs that cover part of a path: the rest is one fallback bag
        graph = path_graph(6)
        @test check(ones(Int, 6), graph, [[1, 2]]) == (5, 1)
        @test check(ones(Int, 6), graph, [[1, 2], [2, 3], [3, 4]]) == (3, 1)
        @test check(ones(Int, 6), graph, [[i, i + 1] for i in 1:5]) == (2, 5)

        # disconnected graphs and isolated vertices (∅ is a separator)
        graph = blockdiag(blockdiag(cycle_graph(5), path_graph(3)), Graph(2))
        sets = allpmcs(graph)
        weights = [3, 1, 2, 2, 1, 1, 5, 1, 4, 1]
        @test check(weights, graph, sets)[1] == treewidth(weights, graph; alg = ConnectedComponents(PIDBT()))
        @test check(weights, graph, sets) == hbtvalue(weights, graph, sets)
        @test structure(weights, graph, sets)

        rng = Xoshiro(1)
        graph = blockdiag(blockdiag(erdos_renyi(40, 0.1; seed = 1), erdos_renyi(30, 0.15; seed = 2)), Graph(3))
        weights = rand(rng, 1:3, nv(graph))
        sets, best = triangulations(weights, graph, rng, 6)
        @test check(weights, graph, sets)[1] <= best
        @test check(weights, graph, sets) == hbtvalue(weights, graph, sets)
        @test structure(weights, graph, sets)

        # PMCs of one connected component only
        part = filter(X -> all(<=(40), X), sets)
        @test check(weights, graph, part) == hbtvalue(weights, graph, part)

        # self loops and repeated edges do not matter
        graph = erdos_renyi(30, 0.12; seed = 3); n = nv(graph)
        weights = rand(rng, 1:3, n)
        sets, _ = triangulations(weights, graph, rng, 5)
        lists = [[collect(neighbors(graph, v)); v; collect(neighbors(graph, v))] for v in 1:n]
        multigraph = BipartiteGraph{Int, Int}(n, n, sum(length, lists), cumsum([1; length.(lists)]), reduce(vcat, lists))
        @test CliqueTrees.pmcdp(weights, multigraph, relation(n, sets)) == CliqueTrees.pmcdp(weights, graph, relation(n, sets))

        # unsorted PMCs
        @test CliqueTrees.pmcdp(weights, graph, relation(n, reverse.(sets)))[3:4] == CliqueTrees.pmcdp(weights, graph, relation(n, sets))[3:4]

        # weights of another type
        order, invp, k, f = CliqueTrees.pmcdp(weights, graph, relation(n, sets))
        @test CliqueTrees.pmcdp(weights ./ 2, graph, relation(n, sets)) == (order, invp, k / 2, f)
    end
end
