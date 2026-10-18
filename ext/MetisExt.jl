module MetisExt

using Base: oneto
using Base.Order
using CliqueTrees
using CliqueTrees: EliminationAlgorithm, Parent, UnionFind, simplegraph, compresstwins, popelements!, realize!, segmentsplit!, classify!, applymerges!, rollback!, mergesorted!, writeresidual!, sympermute!_impl!, compositerotations_impl!, bestfill_impl!, bestwidth_impl!, isparallel, fetchunwrap, MINPARALLEL
using CliqueTrees.Utilities
using Graphs

import Metis

const INT = Metis.idx_t
const NOPTIONS = Metis.METIS_NOPTIONS
const OPTION_CTYPE = Metis.METIS_OPTION_CTYPE + one(INT)
const OPTION_RTYPE = Metis.METIS_OPTION_RTYPE + one(INT)
const OPTION_NSEPS = Metis.METIS_OPTION_NSEPS + one(INT)
const OPTION_NUMBERING = Metis.METIS_OPTION_NUMBERING + one(INT)
const OPTION_NITER = Metis.METIS_OPTION_NITER + one(INT)
const OPTION_SEED = Metis.METIS_OPTION_SEED + one(INT)
const OPTION_COMPRESS = Metis.METIS_OPTION_COMPRESS + one(INT)
const OPTION_CCORDER = Metis.METIS_OPTION_CCORDER + one(INT)
const OPTION_PFACTOR = Metis.METIS_OPTION_PFACTOR + one(INT)
const OPTION_UFACTOR = Metis.METIS_OPTION_UFACTOR + one(INT)

# METIS keeps its random number generator in global state, so concurrent calls
# give different results than the same calls made one at a time
const LOCK = ReentrantLock()

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::METIS; verbose::Bool = false) where {V}
    simple = simplegraph(INT, INT, graph)
    order::Vector{V}, index::Vector{V} = metis(weights, simple, alg)
    return order, index
end

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph, alg::ND{<:Any, <:EliminationAlgorithm, METISND}; verbose::Bool = false)
    order = dissect(weights, graph, alg)
    return order, invperm(order)
end

function metis(weights::AbstractVector, graph::BipartiteGraph{INT, INT}, alg::METIS)
    @assert nv(graph) <= length(weights)
    n = nv(graph); new = Vector{INT}(undef, n)

    @inbounds for v in oneto(n)
        new[v] = trunc(INT, weights[v])
    end

    return metis(new, graph, alg)
end

function metis(weights::Vector{INT}, graph::BipartiteGraph{INT, INT}, alg::METIS)
    n = nv(graph)

    # construct options
    options = Vector{INT}(undef, NOPTIONS)
    setoptions!(options, alg)

    # construct METIS graph
    xadj = pointers(graph)
    adjncy = targets(graph)
    vwght = weights

    # construct permutation
    order = Vector{INT}(undef, n)
    index = Vector{INT}(undef, n)

    @lock LOCK Metis.@check Metis.METIS_NodeND(
        Ref{INT}(n),
        xadj,
        adjncy,
        vwght,
        options,
        order,
        index,
    )

    return order, index
end

# Order the graph of a branch with `alg` on another thread. The graph and its
# weights live in arrays that the dissection reuses (`separator!` shifts them
# in place right away), so they are copied here, before the task starts. A
# branch is realized as the same graph when it is split and when its children
# have returned, so the result is the one the serial code computes at the join.
function spawngreedy(weights::AbstractVector{INT}, graph::BipartiteGraph{INT, INT}, alg::EliminationAlgorithm)
    weights = collect(view(weights, oneto(nv(graph))))
    graph = copygraph(graph)
    return spawnwork(() -> permutation(weights, graph, alg), weights, graph)
end

function copygraph(graph::BipartiteGraph{INT, INT})
    n = nv(graph); m = ne(graph)
    return BipartiteGraph(n, n, m, pointers(graph)[oneto(n + one(INT))], targets(graph)[oneto(m)])
end

function separator!(options::AbstractVector{INT}, sepsize::AbstractScalar{INT}, part::AbstractVector{INT}, weights::AbstractVector{INT}, graph::BipartiteGraph{INT, INT}, imbalance::INT, alg::METISND)
    @assert NOPTIONS <= length(options)
    @assert nv(graph) <= length(part)
    @assert nv(graph) <= length(weights)
    @assert ispositive(imbalance)
    n = nv(graph); m = ne(graph); nn = n + one(INT)

    # construct options
    setoptions!(options, imbalance, alg)

    # construct METIS graph
    xadj = pointers(graph)
    adjncy = targets(graph)
    vwght = weights

    @inbounds for i in oneto(nn)
        xadj[i] -= one(INT)
    end

    @inbounds for p in oneto(m)
        adjncy[p] -= one(INT)
    end

    # construct separator
    @lock LOCK Metis.@check Metis.METIS_ComputeVertexSeparator(
        Ref{INT}(n),
        xadj,
        adjncy,
        vwght,
        options,
        sepsize,
        part,
    )

    @inbounds for i in oneto(nn)
        xadj[i] += one(INT)
    end

    @inbounds for p in oneto(m)
        adjncy[p] += one(INT)
    end

    return
end

function dissect(weights::AbstractVector{W}, graph::AbstractGraph, alg::ND) where {W}
    n = nv(graph)
    scale = convert(W, alg.scale)
    intweights = FVector{INT}(undef, n)

    @inbounds for v in oneto(n)
        w = weights[v]

        if W <: AbstractFloat
            w *= scale
        end

        intweights[v] = trunc(INT, w)
    end

    return dissect(intweights, graph, alg)
end

function dissect(weights::FVector{INT}, graph::AbstractGraph{V}, alg::ND) where {V <: Integer}
    simple = simplegraph(INT, INT, graph)

    # merge twins up front: every graph in the dissection is then twin-free
    cmpgraph, cmpweights, project = compresstwins(weights, simple)

    # The orderings at the nodes share one workspace. Their bucket arrays are
    # largest at the root; if the orderings at the large nodes run on other
    # tasks (see `spawngreedy`), then those tasks reserve them.
    order = withworkspace() do
        if !isparallel(alg.alg, false)
            reserve!(workspace(), BUCKETS, twice(sum(Int, cmpweights; init = 0)) + 2)
        end

        return convert(Vector{V}, dissectsimple(cmpweights, cmpgraph, project, alg))
    end

    return order
end

# Nested dissection on one global quotient graph, without a node stack (see
# dissection_algorithms.jl). The state is the twin-free graph, the separators
# of the ancestors of the current node, one array of working orderings, a
# union-find holding the twin classes of the path, and a few scalars per
# level. The graph G'[W] of a node is realized, compressed, when the node is
# split and again when its children have returned; at most one realized graph
# is alive at a time.
function dissectsimple(weights::AbstractVector{INT}, graph::BipartiteGraph{INT, INT}, label::BipartiteGraph{INT, INT}, alg::ND)
    n = nv(graph)

    # the root: every vertex in its own class, below no separator
    nelm = FScalar{INT}(undef); nelm[] = zero(INT)
    elmptr = FVector{INT}(undef, convert(INT, alg.level) + two(INT)); elmptr[begin] = one(INT)
    pinvtx = Vector{INT}(undef, n)
    pinlvl = Vector{INT}(undef, n)
    pinnext = Vector{INT}(undef, n)
    pinhead = FVector{INT}(undef, n)
    ufp = FVector{INT}(undef, n)

    @inbounds for v in oneto(n)
        pinhead[v] = zero(INT); ufp[v] = v
    end

    segment, _ = dissectnode(weights, graph, ufp, nelm, elmptr, pinvtx, pinlvl,
        pinnext, pinhead, zero(INT), alg)

    # expand the vertices of the twin-free graph
    j = zero(INT); result = FVector{INT}(undef, ne(label))

    @inbounds for i in oneto(n), w in neighbors(label, segment[i])
        j += one(INT); result[j] = w
    end

    return result
end

# Children with at least this many vertices (of the twin-free graph) are
# dissected on another task: see `spawnsubtree`. Concurrent subtrees still
# call METIS one at a time (see `LOCK`), which bounds the speedup of one
# dissection by the share of METIS in its serial time.
const MINSUBTREE = 256

# Dissect the subtree of the node at level `level0` whose vertex set is every
# vertex of `graph`. The classes of the node are in `ufp`, and the separators
# of its ancestors are on the stack. Returns `segment`, whose first entries are
# the residual of the node, and the length of the residual.
function dissectnode(weights::AbstractVector{INT}, graph::BipartiteGraph{INT, INT}, ufp::AbstractVector{INT}, nelm::AbstractScalar{INT}, elmptr::AbstractVector{INT}, pinvtx::Vector{INT}, pinlvl::Vector{INT}, pinnext::Vector{INT}, pinhead::AbstractVector{INT}, level0::INT, alg::ND{S}) where {S}
    n = nv(graph); m = ne(graph); nn = n + one(INT)
    maxlevel = convert(INT, alg.level)
    minwidth = convert(INT, alg.width)
    imbalance = convert(INT, alg.imbalance)
    nl = maxlevel + one(INT)
    @assert nelm[] == level0 < nl

    # the realization of the current node
    tag = FScalar{Int}(undef); tag[] = 0
    stamp = FVector{Int}(undef, n)
    vclass = FVector{INT}(undef, n)
    marker = FVector{INT}(undef, max(n, maxlevel + two(INT)))
    mask = FVector{UInt64}(undef, n)
    clsptr = FVector{INT}(undef, maxlevel + two(INT))
    clstgt = Vector{INT}(undef, n)
    lblptr = FVector{INT}(undef, nn)
    lbltgt = FVector{INT}(undef, n)
    pointer = FVector{INT}(undef, nn)
    target = Vector{INT}(undef, max(m, one(INT)))

    # the working orderings and the classes of the path (see `segmentsplit!`)
    segment = FVector{INT}(undef, n)
    vertexset = FVector{INT}(undef, n)
    cls = FVector{INT}(undef, n)
    rmark = FVector{Int}(undef, n)
    rid = FVector{INT}(undef, n)
    mrg = Vector{INT}(undef, n)
    mlog = Vector{INT}(undef, n)
    base = FVector{INT}(undef, nl)
    na = FVector{INT}(undef, nl)
    nb = FVector{INT}(undef, nl)
    side = FVector{INT}(undef, nl)
    logstart = FVector{INT}(undef, nl)
    mrgbase = FVector{INT}(undef, nl)
    mrgstart = FVector{INT}(undef, nl)
    mrgstop = FVector{INT}(undef, nl)

    @inbounds for v in oneto(n)
        stamp[v] = 0; rmark[v] = 0; vertexset[v] = v
    end

    work00 = FScalar{INT}(undef)
    work01 = Vector{INT}(undef, half(m))
    work02 = Vector{INT}(undef, half(m))
    work03 = FVector{INT}(undef, max(n, NOPTIONS))
    work04 = FVector{INT}(undef, n)
    work05 = FVector{INT}(undef, n)
    work06 = FVector{INT}(undef, n)
    work07 = FVector{INT}(undef, n)
    work08 = FVector{INT}(undef, n)
    work09 = FVector{INT}(undef, nn)
    work10 = FVector{INT}(undef, nn)
    work11 = FVector{INT}(undef, n)
    work12 = FVector{INT}(undef, n)
    work13 = FVector{INT}(undef, n)
    work14 = FVector{INT}(undef, n)
    work15 = FVector{INT}(undef, n)
    work16 = FVector{INT}(undef, n)

    # the greedy ordering of a branch, started when the branch is split (see
    # `spawngreedy`) and compared with its dissection when its children have
    # returned
    greedy = Vector{Union{Nothing, Task}}(nothing, nl)

    # the partition and dissection ordering of the current node, and scratch
    # space for sorting
    part = FVector{INT}(undef, n)
    ndsorder = Vector{INT}(undef, n)
    ndsindex = Vector{INT}(undef, n)
    scratch = Vector{INT}(undef, n)

    # the residual of child 1 of a branch, when the child is dissected on
    # another task (see `spawnsubtree`)
    subtree = Vector{Union{Nothing, Task}}(nothing, nl)

    level = level0; nw = n; t = 0; nmrg = zero(INT); nlog = zero(INT); nres = zero(INT)
    @inbounds base[level0 + one(INT)] = one(INT)
    descend = true

    @inbounds while true
        l = level + one(INT)

        if descend
            # the node at `level`, with vertex set vertexset[1:nw]
            t += 1; nc = classify!(cls, rmark, rid, ufp, vertexset, nw, t)
            set = view(vertexset, oneto(nw)); setcls = view(cls, oneto(nw))
        else
            # the node at `level` has returned; its residual starts at base[l]
            level == level0 && break
            level -= one(INT); l = level + one(INT)
            p = base[l]; sstart = elmptr[l]; sstop = elmptr[l + one(INT)] - one(INT)

            if isone(side[l]) # child 1 has returned: start child 0 (A ∪ S)
                nlog = rollback!(ufp, mlog, nlog, logstart[l])
                nlog = applymerges!(ufp, mlog, nlog, mrg, mrgstart[l], mrgstop[l])
                nw = mergesorted!(vertexset, segment, p, p + na[l] - one(INT), pinvtx, sstart, sstop)

                # move the residual of child 1 down, to base[l], unless child 1
                # is on another task: then it goes there when it is fetched
                isnothing(subtree[l]) && copyto!(segment, p, segment, p + na[l], nb[l])

                side[l] = zero(INT); level += one(INT)
                base[level + one(INT)] = p + nb[l]
                descend = true
                continue
            end

            # child 0 has returned: join the node at `level`
            nlog = rollback!(ufp, mlog, nlog, logstart[l]); nmrg = mrgbase[l]
            nw = zero(INT)

            if !isnothing(subtree[l])
                copyto!(segment, p, fetchunwrap(subtree[l]), one(INT), nb[l])
                subtree[l] = nothing
            end

            for q in p:(p + na[l] + nb[l] - one(INT))
                nw += one(INT); vertexset[nw] = segment[q]
            end

            for q in sstart:sstop
                nw += one(INT); vertexset[nw] = pinvtx[q]
            end

            sort!(view(vertexset, oneto(nw)); scratch)
            popelements!(nelm, elmptr, pinvtx, pinnext, pinhead, level)
            t += 1; nc = classify!(cls, rmark, rid, ufp, vertexset, nw, t)
            set = view(vertexset, oneto(nw)); setcls = view(cls, oneto(nw))
        end

        cmpgraph, cmpweights, cmplabel, clique = realize!(tag, stamp, vclass, marker,
            mask, clsptr, clstgt, lblptr, lbltgt, pointer, target, nelm, elmptr,
            pinvtx, pinlvl, pinnext, pinhead, weights, set, setcls, nc, level,
            graph)

        n = nv(cmpgraph); m = ne(cmpgraph); k = convert(INT, length(clique))
        iscomplete = m == n * (n - one(INT))

        if half(m) > length(work01)
            resize!(work01, half(m))
            resize!(work02, half(m))
        end

        # a leaf is processed as soon as it is reached
        isleaf = descend

        if descend && !(n <= minwidth || level >= maxlevel || iscomplete) # branch
            if (isone(S) || istwo(S)) && m >= MINPARALLEL && isparallel(alg.alg, false)
                greedy[l] = spawngreedy(cmpweights, cmpgraph, alg.alg)
            else
                greedy[l] = nothing
            end

            separator!(work03, work00, part, cmpweights, cmpgraph, imbalance, alg.dis)

            p = base[l]; mrgbase[l] = nmrg

            nap, nbp, stop1, stop0 = segmentsplit!(work04, work05, work06, work07,
                work08, work11, work12, work13, work14, marker, segment, p, mrg, nmrg,
                ufp, lblptr, lbltgt, nelm, elmptr, pinvtx, pinlvl, pinnext, pinhead,
                set, setcls, part, nc, level, cmpgraph)

            na[l] = nap; nb[l] = nbp; side[l] = one(INT)
            mrgstart[l] = stop1 + one(INT); mrgstop[l] = nmrg = stop0

            # start child 1 (B ∪ S)
            logstart[l] = nlog + one(INT)
            nlog = applymerges!(ufp, mlog, nlog, mrg, mrgbase[l] + one(INT), stop1)
            sstart = elmptr[l]; sstop = elmptr[l + one(INT)] - one(INT)
            nw = mergesorted!(vertexset, segment, p + nap, p + nap + nbp - one(INT), pinvtx, sstart, sstop)

            level += one(INT)
            base[level + one(INT)] = p + nap

            # a large child 1 is dissected on another task; child 0 starts
            # here as if child 1 had returned
            if nw >= MINSUBTREE && isparallel(alg.alg, false)
                subtree[l] = spawnsubtree(tag, stamp, work04, weights, graph,
                    ufp, nelm, elmptr, pinvtx, pinlvl, vertexset, nw, level, alg)

                descend = false
            end

            continue
        end

        if iscomplete # complete graph
            for v in oneto(n)
                work03[v] = v
            end
        else
            tree = Parent(n, work06)
            upper = BipartiteGraph(n, n, half(m), work09, work01)
            lower = BipartiteGraph(n, n, half(m), work10, work02)

            if isleaf # leaf
                order, index = permutation(cmpweights, cmpgraph, alg.alg)
            else      # branch: child 0, then child 1, then the separator
                p = base[l]
                resize!(ndsorder, n)
                resize!(ndsindex, n)
                i = zero(INT)

                # the residuals list vertices of the quotient
                # graph: read each class at its first vertex
                seen = marker

                for c in oneto(n)
                    seen[c] = zero(INT)
                end

                for (qstart, qstop) in ((p + nb[l], p + nb[l] + na[l] - one(INT)), (p, p + nb[l] - one(INT)))
                    for q in qstart:qstop
                        c = vclass[segment[q]]

                        if iszero(seen[c])
                            seen[c] = one(INT)
                            ndsindex[c] = i += one(INT)
                            ndsorder[i] = c
                        end
                    end
                end

                # the classes of the separator, in increasing order
                for c in oneto(n)
                    if iszero(seen[c])
                        ndsindex[c] = i += one(INT)
                        ndsorder[i] = c
                    end
                end

                if isone(S) || istwo(S)
                    sets = UnionFind(n, work03, work04, work05)

                    # the type of a task's result is not inferred
                    local grdorder::Vector{INT}, grdindex::Vector{INT}

                    if isnothing(greedy[l])
                        grdorder, grdindex = permutation(cmpweights, cmpgraph, alg.alg)
                    else
                        grdorder, grdindex = fetchunwrap(greedy[l]); greedy[l] = nothing
                    end

                    if isone(S)
                        best = bestwidth_impl!(lower, upper, tree, sets, work07,
                            work08, work11, work12, work13, work14, work15,
                            work16, cmpweights, cmpgraph, (ndsindex, grdindex))
                    else
                        best = bestfill_impl!(lower, upper, tree, sets, work07,
                            work08, work11, work12, work13, work14, work15,
                            work16, cmpweights, cmpgraph, (ndsindex, grdindex))
                    end

                    order = (ndsorder, grdorder)[best]
                    index = (ndsindex, grdindex)[best]
                else
                    order, index = ndsorder, ndsindex
                end
            end

            sympermute!_impl!(upper, cmpgraph, index, Forward)

            for i in oneto(k)
                clique[i] = index[clique[i]]
            end

            compositerotations_impl!(index, work03, work04,
                work05, lower, tree, upper, clique)

            for v in oneto(n)
                i = index[v]; work03[i] = order[v]
            end
        end

        # the residual: the ordering, minus the separator of the parent
        t += 1

        if ispositive(level)
            for q in elmptr[level]:(elmptr[l] - one(INT))
                rmark[pinvtx[q]] = t
            end
        end

        nres = writeresidual!(segment, base[l], work03, n, cmplabel, rmark, t)
        descend = false
    end

    return segment, nres
end

# Dissect the subtree of the node at `level` with vertex set
# W = vertexset[1:nw] on another task, as an independent instance of
# `dissectnode`: the inputs of the subtree are G[W], the classes of W, and the
# separators of the ancestors of the node, restricted to W. They are copied
# here, before the task starts, with the vertices of W renumbered 1:nw in
# increasing order, so that every list the subtree reads (neighbors, classes,
# separators) comes in the same order as in the serial code: the subtree
# computes the same realized graphs, separators, and orderings. The classes
# of W lie in W, and so do the members of the separators that the subtree
# reads (the separators of the nodes below the node's parent). Returns a task
# whose result is the residual of the node.
function spawnsubtree(tag::AbstractScalar{Int}, stamp::AbstractVector{Int}, newid::AbstractVector{INT}, weights::AbstractVector{INT}, graph::BipartiteGraph{INT, INT}, ufp::AbstractVector{INT}, nelm::AbstractScalar{INT}, elmptr::AbstractVector{INT}, pinvtx::AbstractVector{INT}, pinlvl::AbstractVector{INT}, vertexset::AbstractVector{INT}, nw::INT, level::INT, alg::ND)
    @assert nelm[] == level
    set = vertexset[oneto(nw)]; t = tag[] += 1; m = zero(INT); npin = zero(INT)

    @inbounds for i in oneto(nw)
        v = set[i]; stamp[v] = t; newid[v] = i
    end

    @inbounds for v in set, w in neighbors(graph, v)
        if stamp[w] == t
            m += one(INT)
        end
    end

    @inbounds for p in oneto(elmptr[level + one(INT)] - one(INT))
        if stamp[pinvtx[p]] == t
            npin += one(INT)
        end
    end

    # G[W], its weights, and the classes of W
    subweights = Vector{INT}(undef, nw)
    subufp = FVector{INT}(undef, nw)
    subptr = Vector{INT}(undef, nw + one(INT))
    subtgt = Vector{INT}(undef, max(m, one(INT)))
    subptr[begin] = p = one(INT)

    @inbounds for i in oneto(nw)
        v = set[i]; u = ufp[v]; @assert stamp[u] == t
        subweights[i] = weights[v]; subufp[i] = newid[u]

        for w in neighbors(graph, v)
            if stamp[w] == t
                subtgt[p] = newid[w]; p += one(INT)
            end
        end

        subptr[i + one(INT)] = p
    end

    subgraph = BipartiteGraph(nw, nw, m, subptr, subtgt)

    # the separators of the ancestors, restricted to W
    subnelm = FScalar{INT}(undef); subnelm[] = level
    subelmptr = FVector{INT}(undef, convert(INT, alg.level) + two(INT))
    subpinvtx = Vector{INT}(undef, max(npin, nw))
    subpinlvl = Vector{INT}(undef, max(npin, nw))
    subpinnext = Vector{INT}(undef, max(npin, nw))
    subpinhead = FVector{INT}(undef, nw)

    @inbounds for i in oneto(nw)
        subpinhead[i] = zero(INT)
    end

    @inbounds subelmptr[begin] = q = one(INT)

    @inbounds for l in oneto(level)
        for p in elmptr[l]:(elmptr[l + one(INT)] - one(INT))
            v = pinvtx[p]

            if stamp[v] == t
                i = newid[v]; subpinvtx[q] = i; subpinlvl[q] = pinlvl[p]
                subpinnext[q] = subpinhead[i]; subpinhead[i] = q; q += one(INT)
            end
        end

        subelmptr[l + one(INT)] = q
    end

    return spawnwork(subweights, subgraph) do
        segment, nres = dissectnode(subweights, subgraph, subufp, subnelm,
            subelmptr, subpinvtx, subpinlvl, subpinnext, subpinhead, level, alg)

        residual = Vector{INT}(undef, nres)

        @inbounds for i in oneto(nres)
            residual[i] = set[segment[i]]
        end

        return residual
    end
end

function setoptions!(options::AbstractVector{INT}, imbalance::INT, alg::METISND)
    for i in oneto(NOPTIONS)
        options[i] = -one(INT) # null
    end

    options[OPTION_NSEPS] = convert(INT, alg.nseps)
    options[OPTION_NUMBERING] = one(INT)
    options[OPTION_SEED] = convert(INT, alg.seed)
    options[OPTION_UFACTOR] = convert(INT, imbalance)
    return
end

function setoptions!(options::AbstractVector{INT}, alg::METIS)
    for i in oneto(NOPTIONS)
        options[i] = -one(INT) # null
    end

    options[OPTION_CTYPE] = convert(INT, alg.ctype)
    options[OPTION_RTYPE] = convert(INT, alg.rtype)
    options[OPTION_NSEPS] = convert(INT, alg.nseps)
    options[OPTION_NUMBERING] = one(INT)
    options[OPTION_NITER] = convert(INT, alg.niter)
    options[OPTION_SEED] = convert(INT, alg.seed)
    options[OPTION_COMPRESS] = convert(INT, alg.compress)
    options[OPTION_CCORDER] = convert(INT, alg.ccorder)
    options[OPTION_PFACTOR] = convert(INT, alg.pfactor)
    options[OPTION_UFACTOR] = convert(INT, alg.ufactor)
    return
end

end
