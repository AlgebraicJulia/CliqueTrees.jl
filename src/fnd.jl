# Equivalent Sparse Matrix Reorderings by Elimination Tree Rotations
# Liu
# Algorithm 3.2: Composite_Rotations
#
# `compositerotations_impl!` for a graph given as a quotient graph: the
# vertices 1:n are numbered in elimination order, and the edges are the
# arcs (`gptr`, `gtgt`), which are symmetric, and the cliques 1:nel, where
# the members of the clique e are etgt[eptr[e]:eptr[e + 1] - 1]. The
# cliques are never formed.
#
#   - The elimination tree is the elimination tree of the graph in which
#     each clique is replaced by a path through its members, in increasing
#     order. If the members of a clique are k1 < k2 < ... < kr, then when
#     the vertex kj is processed by `etree_impl!`, the vertices k1, ...,
#     kj-2 are already descendants of kj-1, so their roots are the root of
#     kj-1: the edge (kj-1, kj) links the same roots to kj as the edges
#     (k1, kj), ..., (kj-1, kj).
#   - The higher neighbors of a vertex `v` are read from its arcs and from
#     its cliques. A visit from `v` while the algorithm climbs from `y`
#     numbers the neighbors greater than max(`v`, `y`) that are not yet
#     numbered, in increasing order, as `compositerotations_impl!` does:
#     the order matters. The members of each clique are sorted, and a
#     clique is scanned from its last member down, until the first member
#     that is not greater than max(`v`, `y`). The members passed are
#     numbered by the visit, so the next visit resumes where this one
#     stopped, and each clique is scanned once.
#
# On return, `alpha[v]` is the new position of the vertex `v`, and the
# members of `clique` are last. The members of each clique are sorted in
# place.
function fnd_rotations!(
        alpha::AbstractVector{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
        fdesc::AbstractVector{V},
        cand::AbstractVector{V},
        ehi::AbstractVector{E},
        uptr::AbstractVector{E},
        utgt::AbstractVector{V},
        iptr::AbstractVector{E},
        itgt::AbstractVector{V},
        tree::Parent{V},
        gptr::AbstractVector{E},
        gtgt::AbstractVector{V},
        eptr::AbstractVector{E},
        etgt::AbstractVector{V},
        clique::AbstractVector{V},
        n::V,
        nel::V,
    ) where {V, E}

    # sort the members of each clique
    @inbounds for e in oneto(nel)
        sort!(view(etgt, eptr[e]:eptr[e + one(V)] - one(E)))
    end

    ###################################################
    # 1. the lower neighbors of each vertex: its arcs #
    #    to lower vertices, and the paths of cliques  #
    ###################################################

    @inbounds for v in oneto(n + one(V))
        uptr[v] = zero(E)
    end

    @inbounds for v in oneto(n)
        for p in gptr[v]:gptr[v + one(V)] - one(E)
            if gtgt[p] < v
                uptr[v + one(V)] += one(E)
            end
        end
    end

    @inbounds for e in oneto(nel)
        for p in eptr[e] + one(E):eptr[e + one(V)] - one(E)
            uptr[etgt[p] + one(V)] += one(E)
        end
    end

    @inbounds uptr[begin] = one(E)

    @inbounds for v in oneto(n)
        uptr[v + one(V)] += uptr[v]
    end

    # `alpha[v]` is the tail of the list of `v`
    @inbounds for v in oneto(n)
        alpha[v] = convert(V, uptr[v])
    end

    @inbounds for v in oneto(n)
        for p in gptr[v]:gptr[v + one(V)] - one(E)
            w = gtgt[p]

            if w < v
                utgt[alpha[v]] = w; alpha[v] += one(V)
            end
        end
    end

    @inbounds for e in oneto(nel)
        for p in eptr[e] + one(E):eptr[e + one(V)] - one(E)
            v = etgt[p]; utgt[alpha[v]] = etgt[p - one(E)]; alpha[v] += one(V)
        end
    end

    upper = BipartiteGraph(n, n, uptr[n + one(V)] - one(E), uptr, utgt)

    ##########################################
    # 2. the cliques containing each vertex  #
    ##########################################

    @inbounds for v in oneto(n + one(V))
        iptr[v] = zero(E)
    end

    @inbounds for e in oneto(nel)
        for p in eptr[e]:eptr[e + one(V)] - one(E)
            iptr[etgt[p] + one(V)] += one(E)
        end
    end

    @inbounds iptr[begin] = one(E)

    @inbounds for v in oneto(n)
        iptr[v + one(V)] += iptr[v]
    end

    @inbounds for v in oneto(n)
        alpha[v] = convert(V, iptr[v])
    end

    @inbounds for e in oneto(nel)
        ehi[e] = eptr[e + one(V)] - one(E)

        for p in eptr[e]:eptr[e + one(V)] - one(E)
            v = etgt[p]; itgt[alpha[v]] = e; alpha[v] += one(V)
        end
    end

    ##########################
    # 3. the rotations       #
    ##########################

    etree_impl!(tree, fdesc, upper)
    postorder_impl!(alpha, fdesc, index, order, tree)
    firstdescendants_impl!(fdesc, tree, index)

    @inbounds for v in oneto(n)
        i = index[v]
        order[i] = v
        alpha[v] = zero(V)
    end

    if ispositive(n)
        y = n

        @inbounds for v in Iterators.reverse(clique)
            alpha[v] = n; n -= one(V)
            y = min(v, y)
        end

        @inbounds xstop = index[y]; xstart = xstop + one(V)

        @inbounds for z in ancestorindices(tree, y)
            ystart = fdesc[y]; ystop = index[y]

            for i in ystart:(xstart - one(V))
                n = fnd_rotate!(alpha, cand, ehi, gptr, gtgt, eptr, etgt, iptr, itgt, n, order[i], y)
            end

            for i in (xstop + one(V)):ystop
                n = fnd_rotate!(alpha, cand, ehi, gptr, gtgt, eptr, etgt, iptr, itgt, n, order[i], y)
            end

            xstart, xstop = ystart, ystop
            y = z
        end

        @inbounds for v in reverse(vertices(upper))
            if iszero(alpha[v])
                alpha[v] = n; n -= one(V)
            end
        end
    end

    return
end

# Number, from `n` down, the neighbors `w` of the vertex `v` with
# `w` > max(`v`, `y`) that are not numbered, in increasing order (see
# `fnd_rotations!`); `cand` is a work array. Returns the next number.
function fnd_rotate!(
        alpha::AbstractVector{V},
        cand::AbstractVector{V},
        ehi::AbstractVector{E},
        gptr::AbstractVector{E},
        gtgt::AbstractVector{V},
        eptr::AbstractVector{E},
        etgt::AbstractVector{V},
        iptr::AbstractVector{E},
        itgt::AbstractVector{V},
        n::V,
        v::V,
        y::V,
    ) where {V, E}
    bound = max(v, y); k = zero(V)

    # the candidates: they are marked with -1 in `alpha`
    @inbounds for p in gptr[v]:gptr[v + one(V)] - one(E)
        w = gtgt[p]

        if bound < w && iszero(alpha[w])
            alpha[w] = -one(V); k += one(V); cand[k] = w
        end
    end

    @inbounds for p in iptr[v]:iptr[v + one(V)] - one(E)
        e = itgt[p]; q = ehi[e]

        while q >= eptr[e] && etgt[q] > bound
            w = etgt[q]; q -= one(E)

            if iszero(alpha[w])
                alpha[w] = -one(V); k += one(V); cand[k] = w
            end
        end

        ehi[e] = q
    end

    # number them in increasing order
    sort!(view(cand, oneto(k)))

    @inbounds for i in oneto(k)
        alpha[cand[i]] = n; n -= one(V)
    end

    return n
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::FND; verbose::Bool = false)
    order = fnd(weights, graph, alg)
    return order, invperm(order)
end

function fnd(weights::AbstractVector{W}, graph::AbstractGraph{I}, alg::FND) where {W, I}
    @assert ispositive(alg.width)
    @assert 0 <= alg.level < 64
    @assert ispositive(alg.tries)
    @assert 0 < alg.balance < 0.5
    @assert 0 <= alg.boundary <= 1
    @assert ispositive(alg.scale)
    n = nv(graph)
    scale = convert(W, alg.scale)
    intweights = FVector{Int}(undef, n)

    @inbounds for v in oneto(n)
        w = weights[v]

        if W <: AbstractFloat
            w *= scale
        end

        intweights[v] = trunc(Int, w)
    end

    simple = simplegraph(Int, Int, graph)

    # merge twins up front: every graph in the dissection is then twin-free
    cmpgraph, cmpweights, project = compresstwins(intweights, simple)

    order = convert(Vector{I}, fnd_impl(cmpweights, cmpgraph, project, alg))
    return order
end

# Nested dissection on one global quotient graph, without a node stack and
# without realizing any graph (see dissection_algorithms.jl). The state is
# that of the dissection of `ND`: the twin-free graph G, the separators of
# the ancestors of the current node, one array of working orderings, and a
# union-find holding the twin classes of the path. The graph G'[W] of a node
# is the quotient graph whose variables are its classes, joined by the arcs
# of G (see `fnd_quotient!`), and whose elements are the separators of its
# ancestors, which are cliques of G'[W]. A node is split along a separator
# computed by a flow on this graph, with a hub for each element (see
# `fnd_separator`); a leaf is ordered by AMF on it (see `fnd_leaf`); and the
# ordering of a node is rotated, so that the separator of its parent comes
# last, by composite rotations on it (see `fnd_rotate`). So no clique K(e) is
# ever formed.
function fnd_impl(weights::AbstractVector{V}, graph::BipartiteGraph{V, V}, label::BipartiteGraph{V, V}, alg::FND) where {V}
    n = nv(graph); m = ne(graph); nn = n + one(V); ng = n
    maxlevel = convert(V, alg.level)
    minwidth = convert(V, alg.width)
    nl = maxlevel + one(V)

    # the separators of the ancestors of the current node (a stack)
    nelm = FScalar{V}(undef); nelm[] = zero(V)
    elmptr = FVector{V}(undef, maxlevel + two(V)); elmptr[begin] = one(V)
    pinvtx = Vector{V}(undef, n)
    pinlvl = Vector{V}(undef, n)
    pinnext = Vector{V}(undef, n)
    pinhead = FVector{V}(undef, n)

    # the classes of the current node
    tag = FScalar{Int}(undef); tag[] = 0
    stamp = FVector{Int}(undef, n)
    vclass = FVector{V}(undef, n)
    marker = FVector{V}(undef, max(n, maxlevel + two(V)))
    mask = FVector{UInt64}(undef, n)
    clsptr = FVector{V}(undef, maxlevel + two(V))
    clstgt = Vector{V}(undef, n)
    lblptr = FVector{V}(undef, nn)
    lbltgt = FVector{V}(undef, n)

    # the arcs of the quotient graph of the current node
    gtag = FScalar{Int}(undef); gtag[] = 0
    gmark = FVector{Int}(undef, n)
    gbuf = FVector{V}(undef, n)
    qptr = FVector{V}(undef, nn)
    qtgt = FVector{V}(undef, max(m, one(V)))

    # the working orderings and the classes of the path (see `segmentsplit!`)
    segment = FVector{V}(undef, n)
    vertexset = FVector{V}(undef, n)
    cls = FVector{V}(undef, n)
    ufp = FVector{V}(undef, n)
    rmark = FVector{Int}(undef, n)
    rid = FVector{V}(undef, n)
    mrg = Vector{V}(undef, n)
    mlog = Vector{V}(undef, n)
    base = FVector{V}(undef, nl)
    na = FVector{V}(undef, nl)
    nb = FVector{V}(undef, nl)
    side = FVector{V}(undef, nl)
    logstart = FVector{V}(undef, nl)
    mrgbase = FVector{V}(undef, nl)
    mrgstart = FVector{V}(undef, nl)
    mrgstop = FVector{V}(undef, nl)

    @inbounds for v in oneto(n)
        pinhead[v] = zero(V); stamp[v] = 0; gmark[v] = 0; rmark[v] = 0
        ufp[v] = vertexset[v] = v
    end

    work01 = FVector{V}(undef, n)
    work02 = FVector{V}(undef, n)
    work03 = FVector{V}(undef, n)
    work04 = FVector{V}(undef, n)
    work05 = FVector{V}(undef, n)
    work06 = FVector{V}(undef, n)
    work07 = FVector{V}(undef, nn)
    work08 = FVector{V}(undef, n)
    work09 = FVector{V}(undef, n)
    work10 = FVector{V}(undef, n)
    work11 = FVector{V}(undef, n)

    level = zero(V); nw = n; t = 0; nmrg = zero(V); nlog = zero(V)
    @inbounds base[begin] = one(V)
    descend = true

    @inbounds while true
        l = level + one(V)

        if descend
            # the node at `level`, with vertex set vertexset[1:nw]
            t += 1; nc = classify!(cls, rmark, rid, ufp, vertexset, nw, t)
            set = view(vertexset, oneto(nw)); setcls = view(cls, oneto(nw))
        else
            # the node at `level` has returned; its residual starts at base[l]
            iszero(level) && break
            level -= one(V); l = level + one(V)
            p = base[l]; sstart = elmptr[l]; sstop = elmptr[l + one(V)] - one(V)

            if isone(side[l]) # child 1 has returned: start child 0 (A ∪ S)
                nlog = rollback!(ufp, mlog, nlog, logstart[l])
                nlog = applymerges!(ufp, mlog, nlog, mrg, mrgstart[l], mrgstop[l])
                nw = mergesorted!(vertexset, segment, p, p + na[l] - one(V), pinvtx, sstart, sstop)

                # move the residual of child 1 down, to base[l]
                copyto!(segment, p, segment, p + na[l], nb[l])

                side[l] = zero(V); level += one(V)
                base[level + one(V)] = p + nb[l]
                descend = true
                continue
            end

            # child 0 has returned: join the node at `level`
            nlog = rollback!(ufp, mlog, nlog, logstart[l]); nmrg = mrgbase[l]
            nw = zero(V)

            for q in p:(p + na[l] + nb[l] - one(V))
                nw += one(V); vertexset[nw] = segment[q]
            end

            for q in sstart:sstop
                nw += one(V); vertexset[nw] = pinvtx[q]
            end

            sort!(view(vertexset, oneto(nw)))
            popelements!(nelm, elmptr, pinvtx, pinnext, pinhead, level)
            t += 1; nc = classify!(cls, rmark, rid, ufp, vertexset, nw, t)
            set = view(vertexset, oneto(nw)); setcls = view(cls, oneto(nw))
        end

        cmpweights, cmplabel = prepare!(tag, stamp, vclass, marker, mask, clsptr,
            clstgt, lblptr, lbltgt, nelm, elmptr, pinlvl, pinnext, pinhead,
            weights, set, setcls, nc, level, graph)

        # the number of arcs of G'[W], without forming it
        mc = implicitarcs(gtag, gmark, gbuf, tag, stamp, vclass, mask, lblptr,
            lbltgt, graph, nc)

        iscomplete = mc == nc * (nc - one(V))

        # `ni` is the number of interior classes: those whose first vertex
        # lies in no separator on the stack. The separators split the
        # interior, so the interiors of the children of a node are disjoint,
        # and the dissection stops when they are small.
        ni = zero(V)

        for c in oneto(nc)
            if iszero(mask[c])
                ni += one(V)
            end
        end

        # a leaf is processed as soon as it is reached
        isleaf = descend

        if !iscomplete
            # the arcs of the quotient graph
            fnd_quotient!(qptr, qtgt, gtag, gmark, gbuf, tag, stamp, vclass,
                mask, lblptr, lbltgt, graph, nc)
        end

        if descend && !(ni <= minwidth || level >= maxlevel || iscomplete) # branch
            part = fnd_separator(cmpweights, qptr, qtgt, clsptr, clstgt, nc, level, alg)

            p = base[l]; mrgbase[l] = nmrg

            nap, nbp, stop1, stop0 = implicitsegmentsplit!(work01, work02, work03,
                work04, work07, work08, work09, work10, work11, marker, segment, p,
                mrg, nmrg, ufp, gtag, gmark, gbuf, tag, stamp, vclass, mask, clsptr,
                clstgt, lblptr, lbltgt, nelm, elmptr, pinvtx, pinlvl, pinnext,
                pinhead, set, setcls, part, nc, level, graph)

            na[l] = nap; nb[l] = nbp; side[l] = one(V)
            mrgstart[l] = stop1 + one(V); mrgstop[l] = nmrg = stop0

            # start child 1 (B ∪ S)
            logstart[l] = nlog + one(V)
            nlog = applymerges!(ufp, mlog, nlog, mrg, mrgbase[l] + one(V), stop1)
            sstart = elmptr[l]; sstop = elmptr[l + one(V)] - one(V)
            nw = mergesorted!(vertexset, segment, p + nap, p + nap + nbp - one(V), pinvtx, sstart, sstop)

            level += one(V)
            base[level + one(V)] = p + nap
            continue
        end

        if iscomplete # complete graph
            for c in oneto(nc)
                work03[c] = c
            end
        else
            if isleaf # leaf
                order, index = fnd_leaf(cmpweights, qptr, qtgt, clsptr, clstgt, nc, level)
            else      # branch: child 0, then child 1, then the separator
                p = base[l]
                order = FVector{V}(undef, nc)
                index = FVector{V}(undef, nc)
                i = zero(V)

                # the residuals list vertices of the quotient
                # graph: read each class at its first vertex
                seen = marker

                for c in oneto(nc)
                    seen[c] = zero(V)
                end

                for (qstart, qstop) in ((p + nb[l], p + nb[l] + na[l] - one(V)), (p, p + nb[l] - one(V)))
                    for q in qstart:qstop
                        c = vclass[segment[q]]

                        if iszero(seen[c])
                            seen[c] = one(V)
                            index[c] = i += one(V)
                            order[i] = c
                        end
                    end
                end

                # the classes of the separator, in increasing order
                for c in oneto(nc)
                    if iszero(seen[c])
                        index[c] = i += one(V)
                        order[i] = c
                    end
                end
            end

            # rotate the classes meeting the separator of the parent
            # to the end
            clique = fnd_clique!(marker, vclass, elmptr, pinvtx, index, nc, level)
            alpha = fnd_rotate(qptr, qtgt, clsptr, clstgt, order, index, clique, nc, level)

            for i in oneto(nc)
                work03[alpha[i]] = order[i]
            end
        end

        # the residual: the ordering, minus the separator of the parent
        t += 1

        if ispositive(level)
            for q in elmptr[level]:(elmptr[l] - one(V))
                rmark[pinvtx[q]] = t
            end
        end

        writeresidual!(segment, base[l], work03, nc, cmplabel, rmark, t)
        descend = false
    end

    # expand the vertices of the twin-free graph
    j = zero(V); result = FVector{V}(undef, ne(label))

    @inbounds for i in oneto(ng), w in neighbors(label, segment[i])
        j += one(V); result[j] = w
    end

    return result
end

# The arcs of the quotient graph of the current node, after `prepare!`: the
# neighbors of the class c are qtgt[qptr[c]:qptr[c + 1] - 1], the classes
# of G(c) that share no separator with c. The classes that do are joined to
# c by a clique, so the arcs and the cliques of the separators cover G'[W],
# and the arcs are symmetric (see `implicitarcs`).
function fnd_quotient!(
        qptr::AbstractVector{V},
        qtgt::AbstractVector{V},
        gtag::AbstractScalar{Int},
        gmark::AbstractVector{Int},
        gbuf::AbstractVector{V},
        tag::AbstractScalar{Int},
        stamp::AbstractVector{Int},
        vclass::AbstractVector{V},
        mask::AbstractVector{UInt64},
        lblptr::AbstractVector{V},
        lbltgt::AbstractVector{V},
        graph::AbstractGraph{V},
        nc::V,
    ) where {V}
    @inbounds qptr[begin] = p = one(V)

    @inbounds for c in oneto(nc)
        μ = mask[c]
        k = gclasses!(gtag, gmark, gbuf, tag, stamp, vclass, lblptr, lbltgt, graph, c)

        for i in oneto(k)
            u = gbuf[i]

            if iszero(mask[u] & μ)
                qtgt[p] = u; p += one(V)
            end
        end

        qptr[c + one(V)] = p
    end

    return p - one(V)
end

# The classes meeting the separator of the parent, at their positions
# `index`, in increasing order of class.
function fnd_clique!(
        marker::AbstractVector{V},
        vclass::AbstractVector{V},
        elmptr::AbstractVector{V},
        pinvtx::AbstractVector{V},
        index::AbstractVector{V},
        nc::V,
        level::V,
    ) where {V}
    k = zero(V)

    @inbounds for c in oneto(nc)
        marker[c] = zero(V)
    end

    if ispositive(level)
        @inbounds for q in elmptr[level]:(elmptr[level + one(V)] - one(V))
            u = vclass[pinvtx[q]]

            if iszero(marker[u])
                marker[u] = one(V); k += one(V)
            end
        end
    end

    clique = FVector{V}(undef, k); k = zero(V)

    if ispositive(level)
        @inbounds for c in oneto(nc)
            if isone(marker[c])
                k += one(V); clique[k] = index[c]
            end
        end
    end

    return clique
end

# The star graph of the quotient graph: the classes 1:nc, joined by the
# arcs (`qptr`, `qtgt`), and a hub for each separator on the stack with two
# or more classes, joined to its classes. Its vertex separators, with the
# hubs never cut, are those of G'[W] (see `flownd!`). Returns the parts of
# the classes.
#
# A class in a separator on the stack is ordered with that separator, after
# the node, wherever the separator of the node puts it. Once the interior of
# the node is eliminated, it is adjacent to the separator of the node in
# any case, so cutting it costs little: in the flow, it weighs `boundary`
# times its weight, and so it counts less toward the balance as well.
function fnd_separator(
        weights::AbstractVector{V},
        qptr::AbstractVector{V},
        qtgt::AbstractVector{V},
        clsptr::AbstractVector{V},
        clstgt::AbstractVector{V},
        nc::V,
        level::V,
        alg::FND,
    ) where {V}
    # `nh` is the number of hubs
    nh = zero(V)

    @inbounds for l in oneto(level)
        if clsptr[l + one(V)] - clsptr[l] >= two(V)
            nh += one(V)
        end
    end

    nt = nc + nh; ptr = FVector{V}(undef, nt + one(V))

    @inbounds for x in oneto(nt + one(V))
        ptr[x] = zero(V)
    end

    @inbounds for c in oneto(nc)
        ptr[c + one(V)] += qptr[c + one(V)] - qptr[c]
    end

    h = nc

    @inbounds for l in oneto(level)
        if clsptr[l + one(V)] - clsptr[l] >= two(V)
            h += one(V)

            for r in clsptr[l]:(clsptr[l + one(V)] - one(V))
                ptr[clstgt[r] + one(V)] += one(V); ptr[h + one(V)] += one(V)
            end
        end
    end

    @inbounds ptr[begin] = one(V)

    @inbounds for x in oneto(nt)
        ptr[x + one(V)] += ptr[x]
    end

    tgt = FVector{V}(undef, ptr[nt + one(V)] - one(V))
    tail = FVector{V}(undef, nt)

    @inbounds for x in oneto(nt)
        tail[x] = ptr[x]
    end

    @inbounds for c in oneto(nc), p in qptr[c]:(qptr[c + one(V)] - one(V))
        tgt[tail[c]] = qtgt[p]; tail[c] += one(V)
    end

    h = nc

    @inbounds for l in oneto(level)
        if clsptr[l + one(V)] - clsptr[l] >= two(V)
            h += one(V)

            for r in clsptr[l]:(clsptr[l + one(V)] - one(V))
                c = clstgt[r]
                tgt[tail[c]] = h; tail[c] += one(V)
                tgt[tail[h]] = c; tail[h] += one(V)
            end
        end
    end

    # the hubs weigh nothing, and the classes in the separators on the
    # stack weigh `boundary` times their weight: they are ordered with
    # those separators, and cutting them adds little fill
    ratio = rationalize(Int, alg.boundary)
    num = convert(V, numerator(ratio)); den = convert(V, denominator(ratio))
    wght = FVector{V}(undef, nt)

    @inbounds for x in oneto(nt)
        wght[x] = x <= nc ? den * weights[x] : zero(V)
    end

    @inbounds for r in oneto(clsptr[level + one(V)] - one(V))
        c = clstgt[r]; wght[c] = num * weights[c]
    end

    part = FVector{V}(undef, nt)
    star = BipartiteGraph(nt, nt, ptr[nt + one(V)] - one(V), ptr, tgt)
    flownd!(part, wght, star, nc, alg.tries, alg.balance, alg.seed)
    return part
end

# Order the leaf with AMF, given the cliques that cover G'[W]: an edge for
# each arc of the quotient graph, and the separators on the stack (see
# `camf`). Returns the ordering of the classes and its inverse.
function fnd_leaf(
        weights::AbstractVector{V},
        qptr::AbstractVector{V},
        qtgt::AbstractVector{V},
        clsptr::AbstractVector{V},
        clstgt::AbstractVector{V},
        nc::V,
        level::V,
    ) where {V}
    # `ncl` is the number of cliques, and `nnz` is the number
    # of their members
    ncl = zero(V); nnz = zero(V)

    @inbounds for c in oneto(nc), p in qptr[c]:(qptr[c + one(V)] - one(V))
        if c < qtgt[p]
            ncl += one(V); nnz += two(V)
        end
    end

    # the separators on the stack that meet the node in fewer than
    # two classes add no edges
    @inbounds for l in oneto(level)
        if clsptr[l + one(V)] - clsptr[l] > one(V)
            ncl += one(V); nnz += clsptr[l + one(V)] - clsptr[l]
        end
    end

    cptr = FVector{V}(undef, ncl + one(V))
    ctgt = FVector{V}(undef, nnz)
    @inbounds cptr[begin] = q = one(V); k = zero(V)

    @inbounds for c in oneto(nc), p in qptr[c]:(qptr[c + one(V)] - one(V))
        u = qtgt[p]

        if c < u
            ctgt[q] = c; ctgt[q + one(V)] = u; q += two(V)
            k += one(V); cptr[k + one(V)] = q
        end
    end

    @inbounds for l in oneto(level)
        if clsptr[l + one(V)] - clsptr[l] > one(V)
            for r in clsptr[l]:(clsptr[l + one(V)] - one(V))
                ctgt[q] = clstgt[r]; q += one(V)
            end

            k += one(V); cptr[k + one(V)] = q
        end
    end

    return camf(nc, view(weights, oneto(nc)), ncl, cptr, ctgt)
end

# Rotate the ordering (`order`, `index`) of the classes so that the classes
# at the positions `clique` come last (see `fnd_rotations!`). Returns the
# new position of the class at each position.
function fnd_rotate(
        qptr::AbstractVector{V},
        qtgt::AbstractVector{V},
        clsptr::AbstractVector{V},
        clstgt::AbstractVector{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
        clique::AbstractVector{V},
        nc::V,
        level::V,
    ) where {V}
    @inbounds narc = qptr[nc + one(V)] - one(V)
    @inbounds nnz = clsptr[level + one(V)] - one(V)

    # the arcs and the cliques, at the positions of the classes
    gptr = FVector{V}(undef, nc + one(V))
    gtgt = FVector{V}(undef, narc)
    eptr = FVector{V}(undef, level + one(V))
    etgt = FVector{V}(undef, nnz)

    @inbounds gptr[begin] = p = one(V)

    @inbounds for i in oneto(nc)
        c = order[i]

        for q in qptr[c]:(qptr[c + one(V)] - one(V))
            gtgt[p] = index[qtgt[q]]; p += one(V)
        end

        gptr[i + one(V)] = p
    end

    @inbounds for l in oneto(level + one(V))
        eptr[l] = clsptr[l]
    end

    @inbounds for r in oneto(nnz)
        etgt[r] = index[clstgt[r]]
    end

    alpha = FVector{V}(undef, nc)
    rorder = FVector{V}(undef, nc)
    rindex = FVector{V}(undef, nc)
    fdesc = FVector{V}(undef, nc)
    cand = FVector{V}(undef, nc)
    ehi = FVector{V}(undef, level)
    uptr = FVector{V}(undef, nc + one(V))
    utgt = FVector{V}(undef, narc + nnz)
    iptr = FVector{V}(undef, nc + one(V))
    itgt = FVector{V}(undef, nnz)
    tree = Parent{V}(nc)

    fnd_rotations!(alpha, rorder, rindex, fdesc, cand, ehi, uptr, utgt,
        iptr, itgt, tree, gptr, gtgt, eptr, etgt, clique, nc, level)

    return alpha
end
