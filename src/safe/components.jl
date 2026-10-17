# ---------------------------------------------------------------------------
# ConnectedComponents{N}
#
# Split a graph into its N-connected components, N = 1, 2, 3, and order each
# piece with an elimination algorithm. Each N splits the pieces of N - 1:
#
#   - N = 1: the connected components (`connectedcomponents` in
#     elimination_algorithms.jl)
#   - N = 2: the blocks, which meet at cut vertices (blocks.jl)
#   - N = 3: the safe triconnected components of each block, which meet at
#     separation pairs (tutte.jl). A separation pair is filled.
#
# Every separator is a clique of the filled graph, so the treewidth of the
# graph is the largest treewidth of a piece. An ordering of the graph is put
# together from orderings of the pieces: each is rotated so that its
# separator comes last (Liu's composite rotations, which keep its fill), and
# its residual is written in the rotated order, the pieces in postorder.
#
# The pieces are ordered as one batch (see `batch_permutation`).
# ---------------------------------------------------------------------------

function connectedcomponents(weights::AbstractVector, graph::AbstractGraph, alg::EliminationAlgorithm, ::Val{1}; verbose::Bool = false)
    return connectedcomponents(weights, graph, alg; verbose)
end

function connectedcomponents(weights::AbstractVector, graph::AbstractGraph, alg::EliminationAlgorithm, ::Val{N}; verbose::Bool = false) where {N}
    return connectedcomponents_impl(weights, simplegraph(graph), alg, Val(N); verbose)
end

function connectedcomponents_impl(weights::AbstractVector{W}, graph::AbstractGraph{V}, alg::EliminationAlgorithm, ::Val{N}; verbose::Bool = false) where {W, V, N}
    E = etype(graph); n = nv(graph)

    # the blocks, with the vertices numbered in elimination order, and the
    # subgraph of each block (`nothing`: a clique)
    border, btree = blocktree(graph)
    bindex = invperm(border)
    bgraph = graphpermute(graph, border, bindex)
    blocks = cc_bags(weights[border], bgraph, btree)
    nb = length(blocks)

    # the pieces to order, as one batch
    subweights = Vector{W}[]
    subgraphs = FBipartiteGraph{V, E}[]

    # `bunit[j]`: the piece of block j, if it is not split; otherwise, the
    # Tutte decomposition of block j (N = 3): its ordering and tree, its graph
    # with the vertices numbered in elimination order and the separators
    # filled, the subgraph of each of its bags, and the piece of each bag
    bunit = zeros(Int, nb)
    torders = Vector{Union{Nothing, Vector{V}}}(nothing, nb)
    tgraphs = Vector{Union{Nothing, typeof(bgraph)}}(nothing, nb)
    ttrees = Vector{Union{Nothing, typeof(btree)}}(nothing, nb)
    tbags = Vector{Union{Nothing, typeof(blocks)}}(nothing, nb)
    tunits = Vector{Union{Nothing, Vector{Int}}}(nothing, nb)

    for j in oneto(nb)
        block = blocks[j]
        isnothing(block) && continue
        bw, bg = block
        tutte = N == 3 ? tuttetree(bw, bg) : nothing

        if isnothing(tutte)
            push!(subweights, bw); push!(subgraphs, bg)
            bunit[j] = length(subgraphs)
        else
            cmpgraph, torder, ttree = tutte
            tgraph = graphpermute(cmpgraph, torder, invperm(torder))
            bags = cc_bags(bw[torder], tgraph, ttree)
            units = zeros(Int, length(bags))

            for k in eachindex(bags)
                bag = bags[k]
                isnothing(bag) && continue
                push!(subweights, first(bag)); push!(subgraphs, last(bag))
                units[k] = length(subgraphs)
            end

            torders[j] = torder; tgraphs[j] = tgraph; ttrees[j] = ttree
            tbags[j] = bags; tunits[j] = units
        end
    end

    suborders = Vector{V}[]

    if !isempty(subgraphs)
        for order in batch_permutation(subweights, subgraphs, alg; verbose)
            push!(suborders, order)
        end
    end

    # an ordering of each block
    border1 = Vector{Union{Nothing, Vector{V}}}(nothing, nb)

    for j in oneto(nb)
        if ispositive(bunit[j])
            border1[j] = suborders[bunit[j]]
        elseif !isnothing(tunits[j])
            torder = torders[j]::Vector{V}; units = tunits[j]::Vector{Int}
            orders = Vector{Union{Nothing, Vector{V}}}(nothing, length(units))

            for k in eachindex(units)
                ispositive(units[k]) && (orders[k] = suborders[units[k]])
            end

            pindex = cc_assemble(tgraphs[j]::typeof(bgraph), ttrees[j]::typeof(btree), tbags[j]::typeof(blocks), orders)
            order = similar(torder)

            for k in eachindex(torder)
                order[pindex[k]] = torder[k]
            end

            border1[j] = order
        end
    end

    # an ordering of the graph
    pindex = cc_assemble(bgraph, btree, blocks, border1)
    order = Vector{V}(undef, n); index = Vector{V}(undef, n)

    for v in oneto(n)
        i = index[v] = pindex[bindex[v]]
        order[i] = v
    end

    return order, index
end

# The subgraph of each bag of a clique tree of `graph`, whose vertices are
# numbered in elimination order (the residual of each bag is a range): its
# residual, numbered first, then its separator. Returns, for each bag, its
# weights and graph, or `nothing` if it is a clique.
function cc_bags(weights::AbstractVector{W}, graph::AbstractGraph{V}, tree::CliqueTree{V, E}) where {W, V, E}
    n = nv(graph)
    submsk = FVector{V}(undef, n)
    subinv = FVector{V}(undef, n)
    bags = Vector{Union{Nothing, Tuple{Vector{W}, FBipartiteGraph{V, E}}}}(nothing, length(tree))

    @inbounds for i in oneto(n)
        submsk[i] = zero(V)
    end

    @inbounds for (j, bag) in enumerate(tree)
        res = residual(bag); sep = separator(bag)
        strt = first(res); stop = last(res); nres = stop - strt + one(V)
        ii = nres

        for i in sep
            submsk[i] = strt; ii += one(V); subinv[i] = ii
        end

        nn = ii; mm = zero(E); nloop = zero(E)

        for i in bag, k in neighbors(graph, i)
            mm += E(strt <= k <= stop || submsk[k] == strt)
        end

        subptr = FVector{E}(undef, nn + one(V))
        subtgt = FVector{V}(undef, mm)
        subwgt = Vector{W}(undef, nn)
        ii = zero(V); p = subptr[one(V)] = one(E)

        for i in bag
            ii += one(V); subwgt[ii] = weights[i]

            for k in neighbors(graph, i)
                kk = zero(V)

                if strt <= k <= stop
                    kk = k - strt + one(V)
                elseif submsk[k] == strt
                    kk = subinv[k]
                end

                if ispositive(kk)
                    subtgt[p] = kk; p += one(E)
                    kk == ii && (nloop += one(E))
                end
            end

            subptr[ii + one(V)] = p
        end

        # a clique: its residual can be eliminated in any order
        if mm - nloop != widemul(nn, nn - one(V))
            bags[j] = (subwgt, BipartiteGraph(nn, nn, mm, subptr, subtgt))
        end

        # (the separators of the other bags start at other positions)
        for i in sep
            submsk[i] = zero(V)
        end
    end

    return bags
end

# An ordering of `graph` from orderings of the bags of its clique tree (see
# `cc_bags`): `orders[j]` orders the subgraph of bag j, or is `nothing` if bag
# j is a clique. The separator of every bag is a clique of its subgraph.
# Returns the position of each vertex.
function cc_assemble(graph::AbstractGraph{V}, tree::CliqueTree{V, E}, bags::AbstractVector, orders::AbstractVector) where {V, E}
    n = nv(graph); index = Vector{V}(undef, n)
    nmax = zero(V); mmax = zero(E)

    for bag in bags
        isnothing(bag) && continue
        g = last(bag); nmax = max(nmax, nv(g)); mmax = max(mmax, convert(E, half(de(g))))
    end

    alpha = FVector{V}(undef, nmax)
    work1 = FVector{V}(undef, nmax)
    work2 = FVector{V}(undef, nmax)
    work3 = FVector{V}(undef, nmax)
    work4 = FVector{V}(undef, nmax)
    subindex = FVector{V}(undef, nmax)
    uppptr = FVector{E}(undef, nmax + one(V))
    lowptr = FVector{E}(undef, nmax + one(V))
    upptgt = FVector{V}(undef, mmax)
    lowtgt = FVector{V}(undef, mmax)

    @inbounds for (j, bag) in enumerate(tree)
        res = residual(bag); strt = first(res); stop = last(res)

        if isnothing(bags[j])
            for i in strt:stop
                index[i] = i
            end

            continue
        end

        subgraph = last(bags[j]); order = orders[j]::AbstractVector{V}; nn = nv(subgraph)
        nres = stop - strt + one(V)

        for k in oneto(nn)
            subindex[order[k]] = k
        end

        # move the separator (the vertices after the residual) last
        upper = sympermute!_impl!(uppptr, upptgt, subgraph, view(subindex, oneto(nn)), Forward)
        lower = BipartiteGraph(nn, nn, de(upper), lowptr, lowtgt)
        subtree = Parent(nn, work4)
        clique = view(subindex, nres + one(V):nn)
        compositerotations_impl!(alpha, work1, work2, work3, lower, subtree, upper, clique)

        for i in strt:stop
            index[i] = alpha[subindex[i - strt + one(V)]] + strt - one(V)
        end
    end

    return index
end
