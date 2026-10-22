#= A collection of routines to find an exact minimum degree ordering, or an
   exact minimum external degree ordering.
   The degree of every node is kept as a lower bound. Exact degrees are
   computed only for the nodes of minimum bound, once no exact node of
   minimum bound is left, in batches that share their most recent element,
   so each element is scanned once per batch.
   With external degree, an eager pass at each elimination also recomputes
   the nodes whose bound lies within their own weight of the floor. The
   quotient graph, the detection of indistinguishable nodes, element
   absorption and the computation of external parts follow the approximate
   minimum degree algorithm.
   For information on the minimum degree algorithm see the articles:
   The evolution of the minimum degree algorithm by Alan George and
   Joseph Liu, SIAM Rev. 31 pp. 1 - 19, 1989
   An approximate minimum degree ordering algorithm by Patrick Amestoy,
   Timothy Davis, and Iain Duff, SIAM J. Matrix Anal. Appl. 17 pp. 886 -
   905, 1996
   A fast minimum degree algorithm and matching lower bound by Robert
   Cummings, Matthew Fahrbach, and Animesh Fatehpuria, SODA 2021
=#

module MDLib

using Base: oneto
using FillArrays
using ..Utilities

export md

"""
    md(neqns, xadj, adjncy; external=false, dense=10.0, mass=false, twin=false)
    md(neqns, vwght, xadj, adjncy; external=false, dense=10.0, mass=false, twin=false)

Compute an exact minimum degree ordering of the graph `(xadj, adjncy)`.
Returns `index`, where `index[node]` is the position of `node`. The vertex
weights `vwght` must be positive integers: a node of weight w stands for w
indistinguishable nodes.

  - `external`: minimize the external degree instead of the degree. Nodes
    are indistinguishable if they lie in the reachable set of a pivot and
    have the same elements and neighbors there; elements contained in a
    reachable set are absorbed into the new element (aggressive
    absorption). The ordering is exact for these supernodes.
  - `dense`: nodes with more than `max(16, dense * sqrt(neqns))` neighbors
    are kept out of the quotient graph until one of them has minimum
    degree (a negative value disables this). The ordering remains exact;
    with `external`, indistinguishable nodes must also have the same dense
    neighbors.
  - `mass`: mass elimination. Each node of the reachable set whose only
    neighbors are those of the new element (it lies in no other element,
    has no other neighbors, and has the same dense neighbors as the pivot)
    is eliminated with the pivot. With degree such a node is
    indistinguishable from the pivot, so the ordering remains exact. With
    `external` it need not be, and the ordering is no longer exact.
  - `twin`: false-twin elimination. Each node outside the reachable
    set whose neighbors are exactly the reachable set of the pivot, which
    has never been in a reachable set, and whose dense neighbors (and, with
    degree, weight) are those of the pivot, is eliminated right after the
    pivot's supernode. With degree it is then simplicial and of minimum
    degree, so the ordering remains exact. With `external` it need not be,
    and the ordering is no longer exact.
"""
function md(neqns::V, xadj::AbstractVector, adjncy::AbstractVector{V}; kwargs...) where {V}
    vwght = Ones{V}(neqns)
    return md(neqns, vwght, xadj, adjncy; kwargs...)
end

function md(neqns::V, vwght::AbstractVector, xadj::AbstractVector, adjncy::AbstractVector{V}; kwargs...) where {V}
    @assert neqns <= length(vwght)
    total = 0

    @inbounds for node in oneto(neqns)
        weight = trunc(Int, vwght[node])
        @assert ispositive(weight)
        total += weight
    end

    # the working arrays use 32-bit integers when they suffice
    if neqns < typemax(Int32) ÷ 8 && total < typemax(Int32) ÷ 4
        index = md(Int32, neqns, total, vwght, xadj, adjncy; kwargs...)
    else
        index = md(Int64, neqns, total, vwght, xadj, adjncy; kwargs...)
    end

    return convert(Vector{V}, index)
end

function md(::Type{V}, neqns::Integer, total::Integer, vwght::AbstractVector, xadj::AbstractVector{E}, adjncy::AbstractVector; external::Bool = false, dense::Real = 10.0, mass::Bool = false, twin::Bool = false, debug::Bool = false) where {V, E}
    @assert neqns < length(xadj)
    n = convert(V, neqns)
    iszero(n) && return FVector{V}(undef, 0)
    @inbounds nnz = xadj[neqns + 1] - one(E)

    # the lists never take more than `nnz` entries of `iw`, except while a
    # dense node rejoins, which adds fewer than `2 neqns` (see `mdreinsert!`).
    # The extra `max(nnz ÷ 5, neqns) + neqns` covers that, and leaves room
    # between compressions
    iwlen = nnz + max(nnz ÷ convert(E, 5), convert(E, neqns)) + convert(E, neqns)

    marker = FVector{Int}(undef, n)
    supersize = FVector{V}(undef, n)
    superwght = FVector{V}(undef, n)
    degree = FVector{V}(undef, n)
    deghead = FVector{V}(undef, 2 * (total + 1))
    degnext = FVector{V}(undef, n)
    degprev = FVector{V}(undef, n)
    hashval = FVector{UInt64}(undef, n)
    rchset = FVector{V}(undef, n)
    wext = FVector{V}(undef, n)
    extlen = FVector{V}(undef, n)
    pe = FVector{E}(undef, n)
    len = FVector{V}(undef, n)
    elen = FVector{V}(undef, n)
    iw = FVector{V}(undef, iwlen)

    # the lists of a recomputation (see `mdrecompute!`); it grows as needed
    xw = Vector{V}(undef, n ÷ convert(V, 8))

    # select the dense nodes
    dindex = FVector{V}(undef, n)
    ndense = mdselect!(dindex, n, xadj, adjncy, dense)
    nwords = cld(convert(Int, ndense), 64)
    dwght = FVector{V}(undef, ispositive(ndense) ? n : zero(V))
    dnode = FVector{V}(undef, ndense)
    ddeg = FVector{V}(undef, ndense)
    dmask = FVector{UInt64}(undef, nwords * convert(Int, n))

    md_impl!(marker, supersize, superwght, degree, deghead, degnext, degprev,
        hashval, rchset, wext, dwght, extlen, pe, len,
        elen, iw, iwlen, xw, convert(V, total), n, external, mass,
        twin, debug, vwght, xadj,
        adjncy, ndense, nwords, dindex, dnode, ddeg, dmask)

    return degnext
end

"""
    mdselect!(dindex, neqns, xadj, adjncy, dense)

This routine selects the dense nodes: the nodes with more than
`max(16, dense * sqrt(neqns))` neighbors. To keep the space linear, the
masks may use at most `nnz / 2` words, so if there are too many dense
nodes, only those of highest degree are selected. Returns the number of
dense nodes and sets `dindex`.
"""
function mdselect!(dindex::AbstractVector{V}, neqns::V, xadj::AbstractVector{E}, adjncy::AbstractVector, dense::Real) where {V, E}
    @inbounds for node in oneto(neqns)
        dindex[node] = zero(V)
    end

    isnegative(dense) && return zero(V)
    thresh = max(16.0, dense * sqrt(convert(Float64, neqns)))
    ndense = zero(V)

    @inbounds for node in oneto(neqns)
        if convert(Float64, xadj[node + one(V)] - xadj[node]) > thresh
            ndense += one(V)
        end
    end

    iszero(ndense) && return ndense

    # at most `64 * maxwords` dense nodes, where `maxwords * neqns <= nnz / 2`
    @inbounds nnz = convert(Int, xadj[neqns + one(V)] - one(E))
    maxwords = max(1, nnz ÷ (2 * convert(Int, neqns)))
    maxdense = convert(V, min(convert(Int, ndense), 64 * maxwords))
    cand = Vector{V}(undef, ndense); k = 0

    @inbounds for node in oneto(neqns)
        if convert(Float64, xadj[node + one(V)] - xadj[node]) > thresh
            k += 1; cand[k] = node
        end
    end

    if maxdense < ndense
        partialsort!(cand, 1:maxdense; by = node -> xadj[node + one(V)] - xadj[node], rev = true)
        resize!(cand, maxdense); sort!(cand)
    end

    @inbounds for j in eachindex(cand)
        dindex[cand[j]] = convert(V, j)
    end

    return maxdense
end

function md_impl!(
        marker::AbstractVector{I},
        supersize::AbstractVector{V},
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        hashval::AbstractVector{UInt64},
        rchset::AbstractVector{V},
        wext::AbstractVector{V},
        dwght::AbstractVector{V},
        extlen::AbstractVector{V},
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        iwlen::E,
        xw::Vector{V},
        total::V,
        neqns::V,
        external::Bool,
        mass::Bool,
        twin::Bool,
        debug::Bool,
        vwght::AbstractVector,
        xadj::AbstractVector,
        adjncy::AbstractVector,
        ndense::V,
        nwords::Int,
        dindex::AbstractVector{V},
        dnode::AbstractVector{V},
        ddeg::AbstractVector{V},
        dmask::AbstractVector{UInt64},
    ) where {I, V, E}
    @assert iwlen <= length(iw)

    # copy the adjacency structure into `iw`, dropping self loops
    pfree = one(E)

    if iszero(ndense)
        @inbounds for node in oneto(neqns)
            pe[node] = pfree
            istart = xadj[node]
            istop = xadj[node + one(V)] - one(istart)

            for i in istart:istop
                neighbor = convert(V, adjncy[i])

                if neighbor != node
                    iw[pfree] = neighbor; pfree += one(E)
                end
            end

            len[node] = convert(V, pfree - pe[node])
        end
    else
        # the dense nodes are left out of `iw`: the adjacency between any
        # node and a dense node is stored in the masks
        @inbounds for i in eachindex(dmask)
            dmask[i] = zero(UInt64)
        end

        @inbounds for node in oneto(neqns)
            j = dindex[node]

            if ispositive(j)
                dnode[j] = node
            end

            pe[node] = pfree
            istart = xadj[node]
            istop = xadj[node + one(V)] - one(istart)

            for i in istart:istop
                neighbor = convert(V, adjncy[i])

                if neighbor != node
                    k = dindex[neighbor]

                    if ispositive(k)
                        mdsetbit!(dmask, nwords, node, k)
                    elseif !ispositive(j)
                        iw[pfree] = neighbor; pfree += one(E)
                    end
                end
            end

            len[node] = convert(V, pfree - pe[node])
        end
    end

    md!_impl!(marker, supersize, superwght, degree, deghead, degnext, degprev,
        hashval, rchset, wext, dwght, extlen, pe, len,
        elen, iw, iwlen, xw, pfree, total, neqns, external, mass, twin,
        debug, vwght, ndense, nwords, dindex, dnode, ddeg, dmask)

    return degnext
end

"""
    md!_impl!(marker, supersize, superwght, degree, deghead, degnext, degprev,
        hashval, rchset, wext, dwght, extlen, pe, len,
        elen, iw, iwlen, xw, pfree, total, neqns, external, mass, twin,
        debug, vwght, ndense, nwords, dindex, dnode, ddeg, dmask)

This routine implements the exact minimum degree algorithm. It makes use
of the implicit representation of elimination graphs by quotient graphs,
and the notion of indistinguishable nodes. It also implements the
modification by minimum external degree.

The degree of a node is kept as a lower bound, which is exact for some
nodes. Let the key of a node be its bound plus 1 (with external degree)
or plus its weight (with degree). The algorithm repeats the following.
If an exact node has the minimum key, it is eliminated (`mdelim!`,
`mdupdate!`), and the nodes of its reachable set get new bounds.
Otherwise every node at the minimum key is recomputed: they are grouped
by most recent element, and each group is recomputed at once
(`mdrecompute!`).

A node of weight w stands for w indistinguishable nodes, so the degree of
a node is the weight of its closed neighborhood minus one, and its external
degree is the weight of its neighbors outside its supernode.

input parameters:

  - `neqns`: number of equations
  - `total`: sum of the vertex weights
  - `vwght`: vertex weights (truncated to integers)
  - `external`: minimize the external degree instead of the degree
  - `mass`: mass elimination (see `md`)
  - `twin`: false-twin elimination (see `md` and `mdelim!`)
  - `debug`: check every bound after every elimination
  - `iwlen`: length of `iw`
  - `(ndense, nwords, dindex, dnode, ddeg, dmask)`: the dense nodes (see
    `mdselect!`). Their lists are empty: their adjacency, and every
    adjacency to them, is stored in the masks.

updated parameters:

  - `(pe, len, elen, iw)`: quotient graph. On input, `(pe, len, iw)` holds
    the adjacency structure. For an uneliminated node, the list
    `iw[pe[node]:pe[node] + len[node] - 1]` holds `elen[node]` elements,
    sorted by elimination order, followed by its uneliminated neighbors;
    for an eliminated node (element), it holds the nodes in the element.
    An element is absorbed, and the list of a merged node released, by
    setting its length to 0. Destroyed on output.
  - `pfree`: the quotient graph is stored in `iw[1:pfree - 1]`

output parameters:

  - `degnext`: the minimum degree ordering (repurposed from degree list
    links)

working arrays:

  - `deghead`: `deghead[key]` points to the first exact node with a key,
    and `deghead[key + total + 1]` to the first node that is not exact, or
    0 if there are no such nodes
  - `degnext`: during algorithm, points to the next node in the degree list
    (or hash bucket) associated with node, or stores negative values for
    eliminated/merged nodes (-num for an eliminated node, -parent for a
    merged node). During a visit, the next node in a group. After
    algorithm, stores the final ordering.
  - `degprev`: points to the previous node in a degree list associated with
    node, or the negative of the index in `deghead` of its list (if node
    was the first in it), or 0 if the node is not in the degree lists.
    While indistinguishable nodes are detected, the hash bucket of node.
    During a visit, the first node of the group of an element.
  - `degree`: the bound on the external degree of each uneliminated
    supernode; for an element, the weight of the nodes in it
  - `supersize`: the size of the supernodes (0 means merged into another)
  - `superwght`: the weight of the supernodes. It is positive exactly for
    the uneliminated supernodes outside the reachable set being formed: it
    is negated for the nodes of that set and for the pivot, and 0 for
    merged and eliminated nodes.
  - `wext`: for an element, 0 if it is absorbed; otherwise less than `wflg`
    if it has not been met at the current elimination, and `wflg` plus the
    weight of its nodes outside the reachable set if it has. For a node of
    the reachable set, the first term of its bound (see `mdelim!`).
  - `dwght`: the weight of the dense neighbors of each node (empty if there
    are no dense nodes)
  - `hashval`: hash values, used to detect indistinguishable nodes. The
    hash buckets share `deghead` with the degree lists. They use
    its last entries, whose lists (not exact, with keys near `total`) are
    almost always empty: the first node of the bucket at `deghead[b]` is
    `-deghead[b]` if no degree list starts there, and otherwise
    `degprev[deghead[b]]`, which is restored to `-b` once the bucket is
    processed. For an element, `hashval` is the start in
    `xw` of its list of nodes outside the element being recomputed.
  - `rchset`: the reachable set of the current pivot; during a visit, the
    elements with a group
  - `extlen`: for an element, the length of its list of nodes outside the
    element being recomputed. For a node of the reachable set, `extlen` is
    the second term of its bound, and then its upper bound (see
    `mdupdate!`).
  - `xw`: the lists of nodes outside the element being recomputed (see
    `mdrecompute!`); it grows when they do not fit
  - `marker`: temporary marker vector, used with increasing tags
"""
function md!_impl!(
        marker::AbstractVector{I},
        supersize::AbstractVector{V},
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        hashval::AbstractVector{UInt64},
        rchset::AbstractVector{V},
        wext::AbstractVector{V},
        dwght::AbstractVector{V},
        extlen::AbstractVector{V},
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        iwlen::E,
        xw::Vector{V},
        pfree::E,
        total::V,
        neqns::V,
        external::Bool,
        mass::Bool,
        twin::Bool,
        debug::Bool,
        vwght::AbstractVector,
        ndense::V,
        nwords::Int,
        dindex::AbstractVector{V},
        dnode::AbstractVector{V},
        ddeg::AbstractVector{V},
        dmask::AbstractVector{UInt64},
    ) where {I, V, E}
    @assert neqns <= length(marker)
    @assert neqns <= length(supersize)
    @assert neqns <= length(superwght)
    @assert neqns <= length(degree)
    @assert twice(total + one(V)) <= length(deghead)
    @assert neqns <= length(degnext)
    @assert neqns <= length(degprev)
    @assert neqns <= length(hashval)
    @assert neqns <= length(rchset)
    @assert neqns <= length(wext)
    @assert iszero(ndense) || neqns <= length(dwght)
    @assert neqns <= length(extlen)
    @assert neqns <= length(pe)
    @assert neqns <= length(len)
    @assert neqns <= length(elen)
    @assert neqns <= length(vwght)
    @assert pfree <= iwlen + one(E) <= length(iw) + one(E)
    @assert ndense <= length(dnode)
    @assert ndense <= length(ddeg)
    @assert neqns <= length(dindex)
    @assert nwords * neqns <= length(dmask)

    # initialization for the minimum degree algorithm.
    @inbounds for node in oneto(neqns)
        marker[node] = zero(I)
        supersize[node] = one(V)
        superwght[node] = trunc(V, vwght[node])
        elen[node] = zero(V)
        degnext[node] = zero(V)
        degprev[node] = zero(V)
        wext[node] = one(V)
    end

    @inbounds for key in oneto(twice(total + one(V)))
        deghead[key] = zero(V)
    end

    # - `mindeg` is the current minimum key
    # - `num` counts the number of ordered nodes plus 1
    # - `tag` is used to facilitate marking nodes; it is of type `Int`, and
    #   fewer than 4 `neqns` are used per elimination
    # - `wflg` is the flag of `wext`, and `lemax` the largest weight of an
    #   element
    # - `offset` separates the keys of the lists that are not exact
    mindeg = total + one(V); num = one(V); tag = zero(I)
    wflg = one(V); lemax = zero(V); offset = total + one(V)

    # - `nbypass` is the number of dense nodes outside the quotient graph
    # - `nmissing` is the number of ordered pairs of such nodes that are
    #   not adjacent
    nbypass = ndense; nmissing = 0

    if ispositive(ndense)
        @inbounds for node in oneto(neqns)
            dwght[node] = zero(V)
        end

        mddenseinit!(ddeg, dmask, nwords, ndense, dnode, neqns, superwght)

        @inbounds for j in oneto(ndense)
            nmissing += convert(Int, ndense) - 1 - mdcount(dmask, nwords, dnode[j])
        end
    end

    @inbounds for node in oneto(neqns)
        deg = zero(V)
        pstart = pe[node]
        pstop = pstart + convert(E, len[node]) - one(E)

        for p in pstart:pstop
            deg += superwght[iw[p]]
        end

        if ispositive(ndense)
            # dense nodes stay out of the degree lists
            ispositive(dindex[node]) && continue
            dwght[node] = mdweight(dmask, nwords, dnode, node, superwght)
            deg += dwght[node]
        end

        degree[node] = deg
        key = mdkey(deg, superwght[node], external)

        if key > one(V)
            mindeg = mdinsert!(key, mindeg, node, true, deghead, degnext, degprev, offset)
        else
            # eliminate node of degree 0
            degnext[node] = -num; num += one(V)
        end
    end

    @inbounds while num <= neqns
        while mindeg <= total && !ispositive(deghead[mindeg]) && !ispositive(deghead[mindeg + offset])
            mindeg += one(V)
        end

        # if a dense node has a smaller key than every bound, it rejoins the
        # quotient graph and is eliminated
        dj = zero(V)

        if ispositive(nbypass)
            dkey = typemax(V)

            for j in oneto(ndense)
                dnd = dnode[j]

                if ispositive(dnd)
                    key = mdkey(ddeg[j], superwght[dnd], external)

                    if key < dkey
                        dkey = key; dj = j
                    end
                end
            end

            if mindeg <= total && mindeg <= dkey
                dj = zero(V)
            end
        end

        if ispositive(dj)
            mdnode = dnode[dj]
            nmissing -= 2 * (convert(Int, nbypass) - 1 - mdcount(dmask, nwords, mdnode))
            nbypass -= one(V)
            pfree = mdreinsert!(dj, neqns, pe, len, elen, iw, iwlen, pfree,
                superwght, nwords, dindex, dnode, dmask, rchset)
            degree[mdnode] = ddeg[dj]

            # its neighbors now reach it through their lists
            dwgt = superwght[mdnode]
            pstart = pe[mdnode]
            pstop = pstart + convert(E, len[mdnode]) - one(E)

            for p in pstart:pstop
                dwght[iw[p]] -= dwgt
            end
        elseif ispositive(deghead[mindeg])
            # remove `mdnode` from the degree structure. The exact nodes at
            # the minimum key go before the others are recomputed: their
            # eliminations may touch those nodes and give them new bounds
            mdnode = deghead[mindeg]
            mdremove!(mdnode, deghead, degnext, degprev, offset)
        else
            # recompute every node at the minimum key
            mindeg, tag = mdvisit!(mindeg, pe, len, elen, iw, superwght,
                degree, deghead, degnext, degprev, hashval, rchset, marker,
                tag, xw, neqns, extlen, dwght, nbypass, external, offset,
                debug)

            continue
        end

        degprev[mdnode] = zero(V)
        degnext[mdnode] = -num

        if num + supersize[mdnode] > neqns
            break
        end

        # the dense neighbors of `mdnode` lose it
        if ispositive(nbypass)
            mdlose!(ddeg, dmask, nwords, mdnode, superwght[mdnode])
            dS = mdweight(dmask, nwords, dnode, mdnode, superwght)
        else
            dS = zero(V)
        end

        # eliminate `mdnode` and perform quotient graph
        # transformation
        degS = degree[mdnode]

        rchsze, pfree, delta, swght, wflg, lemax, fsze, fwght = mdelim!(mdnode,
            pe, len, elen, iw, iwlen, pfree, neqns, supersize, superwght, degree,
            deghead, degnext, degprev, hashval, wext, extlen, wflg, lemax,
            rchset, dS, offset, external, twin, num + supersize[mdnode],
            nbypass, nwords, dmask, ddeg)

        debug && @assert delta == degS

        # the nodes in the reachable set, and the dense neighbors of
        # `mdnode`, become adjacent to the dense neighbors of `mdnode`
        if ispositive(nbypass)
            nmissing = mddense!(mdnode, rchsze, rchset, dwght, superwght,
                nwords, dnode, ddeg, dmask, nmissing)
        end

        # merge the indistinguishable nodes of the reachable set, eliminate
        # some of them with `mdnode` (mass elimination), assign the bounds
        # of the others, and return them to the degree structure. The
        # reachable set also loses the false twins of `mdnode`. Mass
        # elimination is skipped when there are any: a node of the reachable
        # set adjacent only to the new element and to them is not
        # indistinguishable from `mdnode`, and it must follow them (they
        # are numbered right after the supernode of `mdnode`)
        mindeg, tag = mdupdate!(mdnode, rchsze, delta, dS, swght, fwght, pe,
            len, elen, iw, supersize, superwght, degree, deghead, degnext,
            degprev, hashval, rchset, wext, marker, tag, xw,
            neqns, extlen, dwght, nwords, dmask, ddeg, nbypass, mindeg, external,
            mass && iszero(fsze), offset, debug)

        num += supersize[mdnode] + fsze; superwght[mdnode] = zero(V)

        if debug
            mdcheck(neqns, pe, len, elen, iw, supersize, superwght, degree,
                deghead, degnext, total, dwght, ndense, nwords, dindex, dnode, dmask)
        end
    end

    mdnumber!(neqns, supersize, degnext)
    return degnext
end

"""
    mdelim!(mdnode, pe, len, elen, iw, iwlen, pfree, neqns, supersize,
        superwght, degree, deghead, degnext, degprev, hashval, wext, extlen,
        wflg, lemax, rchset, dS, offset, external, twin, fnum, nbypass,
        nwords, dmask, ddeg)

This routine eliminates the node `mdnode` of minimum degree from the
adjacency structure, which is stored in the quotient graph format. It
forms the reachable set of `mdnode`, absorbs the elements adjacent to
`mdnode`, and computes, for every other element adjacent to the reachable
set, the weight of its nodes outside the reachable set. The elements with
none are absorbed (aggressive absorption). The lists of the nodes in the
reachable set are purged, and the terms of their bounds are gathered:
`wext[u]` is the weight of their uneliminated neighbors plus their largest
external part, and `extlen[u]` the sum of their other external parts (these
slots are unused for uneliminated nodes). The
nodes are placed in hash buckets, and the reachable set becomes the new
element `mdnode`.

With `twin`, the false twins of `mdnode` are found first: the nodes
outside the reachable set whose neighbors are exactly the reachable set.
Each lies in every list of the reachable set, so only the shortest is
scanned. A node of it qualifies if it has never been in a reachable set
(its adjacency is then all in its list, and its degree is exact), its
degree and dense neighbors (and, with degree, its weight) are those of
`mdnode`, and its list holds no node outside the reachable set. Once
`mdnode` is eliminated, it is simplicial and has the key of `mdnode`. It
leaves the quotient graph, numbered right after the supernode of
`mdnode`: its weight is set to 0 before the lists are purged, so it drops
out of them.

input parameters:

  - `mdnode`: node of minimum degree
  - `neqns`: number of equations
  - `iwlen`: length of `iw`
  - `dS`: the weight of the dense neighbors of `mdnode`
  - `offset`: separates the keys of the lists that are not exact
  - `external`: the keys are external degrees (the false twins need not
    have the weight of `mdnode`)
  - `twin`: find the false twins of `mdnode`
  - `fnum`: the number of the first false twin (the position after the
    supernode of `mdnode`)
  - `supersize`: the size of the supernodes
  - `(nbypass, nwords, dmask)`: the dense nodes

updated parameters:

  - `(pe, len, elen, iw)`: quotient graph
  - `pfree`: the quotient graph is stored in `iw[1:pfree - 1]`
  - `ddeg`: the degrees of the dense nodes, which lose the false twins
  - `superwght`: weight of the supernodes; on output, negative for
    `mdnode`
  - `degree`: on output, `degree[mdnode]` is the weight of the nodes in the
    new element
  - `(deghead, degnext, degprev)`: degree lists; on output, the
    nodes in the reachable set are removed from them, and the false twins
    are numbered in `degnext`
  - `hashval`: on output, the nodes in the reachable set are placed in hash
    buckets, which share `deghead` with the degree lists (`degprev` and
    `degnext` hold their buckets and links)
  - `(wext, wflg, lemax)`: external parts of the elements

output parameters:

  - `rchsze`: size of the reachable set
  - `rchset`: the reachable set
  - `(wext, extlen)`: for each node of the reachable set, the terms of its
    bound
  - `delta`: the weight of the reachable set, with the dense neighbors of
    `mdnode`
  - `swght`: the weight of `mdnode`
  - `fsze`: the number of nodes in the false twins of `mdnode`
  - `fwght`: their weight
"""
function mdelim!(
        mdnode::V,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        iwlen::E,
        pfree::E,
        neqns::V,
        supersize::AbstractVector{V},
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        hashval::AbstractVector{UInt64},
        wext::AbstractVector{V},
        extlen::AbstractVector{V},
        wflg::V,
        lemax::V,
        rchset::AbstractVector{V},
        dS::V,
        offset::V,
        external::Bool,
        twin::Bool,
        fnum::V,
        nbypass::V,
        nwords::Int,
        dmask::AbstractVector{UInt64},
        ddeg::AbstractVector{V},
    ) where {V, E}
    # the pivot leaves the quotient graph
    @inbounds swght = superwght[mdnode]; superwght[mdnode] = -swght

    # - `iw[estart:estop]` is the list of elements adjacent to `mdnode`
    # - `iw[nstart:nstop]` is the list of its uneliminated neighbors
    @inbounds estart = pe[mdnode]
    @inbounds estop = estart + convert(E, elen[mdnode]) - one(E)
    @inbounds nstart = estop + one(E)
    @inbounds nstop = estart + convert(E, len[mdnode]) - one(E)

    # form the reachable set: the weight of its nodes is negated, and they
    # are removed from the degree structure. With `twin`, `rsum` is the
    # weight of the set, and `rbest` its node with the fewest neighbors in
    # its list (`lbest`)
    rchsze = zero(V); rsum = zero(V); rbest = zero(V); lbest = typemax(V)

    @inbounds for i in estart:estop
        elmnt = iw[i]
        jstart = pe[elmnt]
        jstop = jstart + convert(E, len[elmnt]) - one(E)

        for j in jstart:jstop
            node = iw[j]; wght = superwght[node]

            if ispositive(wght)
                superwght[node] = -wght
                rchsze += one(V); rchset[rchsze] = node
                mdremove!(node, deghead, degnext, degprev, offset)

                if twin
                    rsum += wght; nlen = len[node] - elen[node]

                    if nlen < lbest
                        lbest = nlen; rbest = node
                    end
                end
            end
        end
    end

    @inbounds for i in nstart:nstop
        node = iw[i]; wght = superwght[node]

        if ispositive(wght)
            superwght[node] = -wght
            rchsze += one(V); rchset[rchsze] = node
            mdremove!(node, deghead, degnext, degprev, offset)

            if twin
                rsum += wght; nlen = len[node] - elen[node]

                if nlen < lbest
                    lbest = nlen; rbest = node
                end
            end
        end
    end

    # false twins of `mdnode`: they lie in the list of `rbest`. Each is
    # numbered (from `fnum`), leaves the degree lists and its dense
    # neighbors, and gets weight 0 so that the purge below drops it
    fsze = fwght = zero(V)

    if twin && ispositive(lbest) && lbest < typemax(V)
        tdeg = rsum + dS
        @inbounds pstart = pe[rbest] + convert(E, elen[rbest])
        @inbounds pstop = pe[rbest] + convert(E, len[rbest]) - one(E)

        @inbounds for p in pstart:pstop
            node = iw[p]; wght = superwght[node]

            # cheapest tests first: the weight is in hand
            if ispositive(wght) && (external || wght == swght) &&
                    degree[node] == tdeg && iszero(elen[node]) &&
                    (!ispositive(nbypass) || mdsame(dmask, nwords, node, mdnode))
                # every neighbor of `node` must lie in the reachable set
                same = true
                qstart = pe[node]
                qstop = qstart + convert(E, len[node]) - one(E)

                for q in qstart:qstop
                    if ispositive(superwght[iw[q]])
                        same = false
                        break
                    end
                end

                if same
                    mdremove!(node, deghead, degnext, degprev, offset)
                    ispositive(nbypass) && mdlose!(ddeg, dmask, nwords, node, wght)
                    degnext[node] = -(fnum + fsze); fsze += supersize[node]
                    superwght[node] = zero(V); len[node] = zero(V); fwght += wght
                end
            end
        end
    end

    # element absorption: the elements adjacent to `mdnode` are absorbed
    # into the new element, and the list of `mdnode` is released
    @inbounds for i in estart:estop
        elmnt = iw[i]
        len[elmnt] = zero(V); wext[elmnt] = zero(V)
    end

    @inbounds len[mdnode] = zero(V); elen[mdnode] = zero(V)

    # advance the flag of `wext` by more than the weight of any element
    # (reset it before it overflows)
    if convert(Int, wflg) + 2 * (convert(Int, lemax) + 1) >= convert(Int, typemax(V))
        @inbounds for node in oneto(neqns)
            if !iszero(wext[node])
                wext[node] = one(V)
            end
        end

        wflg = one(V)
    end

    wflg += lemax + one(V)

    # for each element adjacent to a node in the reachable set, compute
    # `wext[elmnt] - wflg`, the weight of the nodes of `elmnt` outside the
    # reachable set
    @inbounds for k in oneto(rchsze)
        rnode = rchset[k]; rwght = -superwght[rnode]
        jstart = pe[rnode]
        jstop = jstart + convert(E, elen[rnode]) - one(E)

        for j in jstart:jstop
            elmnt = iw[j]; we = wext[elmnt]

            if !iszero(we)
                if we < wflg
                    we = wflg + degree[elmnt]
                end

                wext[elmnt] = we - rwght
            end
        end
    end

    # for each node in the reachable set, do the following...
    hashmask = min(nextpow(2, max(twice(rchsze), one(V))), prevpow(2, length(deghead))) - 1

    @inbounds for k in oneto(rchsze)
        rnode = rchset[k]

        # purge inactive quotient neighbors of `rnode`: absorbed elements,
        # and eliminated nodes, merged nodes, and nodes in the reachable set
        # (they are now reachable through the new element). The elements of
        # `rnode` are `iw[p1:p2]` and its neighbors are `iw[p3:p4]`; `pn` is
        # the next free position, and `pv` the position of the first
        # neighbor after the purging. Along the way, `elmax` is the largest
        # external part, `elsum` their sum, and `nwght` the weight of the
        # neighbors.
        p1 = pe[rnode]
        p2 = p1 + convert(E, elen[rnode]) - one(E)
        p3 = p2 + one(E)
        p4 = p1 + convert(E, len[rnode]) - one(E)
        pn = p1; hash = mdmix(mdnode)
        elmax = elsum = nwght = zero(V)

        for p in p1:p2
            elmnt = iw[p]; we = wext[elmnt]

            if !iszero(we)
                dext = we - wflg

                if iszero(dext)
                    # aggressive absorption: `elmnt` lies in the reachable set
                    len[elmnt] = zero(V); wext[elmnt] = zero(V)
                else
                    iw[pn] = elmnt; pn += one(E); hash += mdmix(elmnt)
                    elmax = max(elmax, dext); elsum += dext
                end
            end
        end

        pv = pn

        for p in p3:p4
            node = iw[p]; wght = superwght[node]

            if ispositive(wght)
                iw[pn] = node; pn += one(E); hash += mdmix(node)
                nwght += wght
            end
        end

        wext[rnode] = nwght + elmax; extlen[rnode] = elsum - elmax

        # append the new element `mdnode` to the element list, which stays
        # sorted by elimination order: move the first neighbor to the end
        # (`rnode` lost at least one entry, so there is room)
        iw[pn] = iw[pv]; iw[pv] = mdnode
        elen[rnode] = convert(V, pv - p1) + one(V)
        len[rnode] = convert(V, pn - p1) + one(V)

        # and place `rnode` in a hash bucket to detect indistinguishable
        # nodes
        hashval[rnode] = hash
        bucket = convert(V, length(deghead)) - convert(V, (hash >>> 32) & hashmask)
        degprev[rnode] = bucket
        head = deghead[bucket]

        if ispositive(head)
            # a degree list starts at `deghead[bucket]`: its first node
            # holds the bucket
            bnode = degprev[head]
            degnext[rnode] = ispositive(bnode) ? bnode : zero(V)
            degprev[head] = rnode
        else
            degnext[rnode] = -head
            deghead[bucket] = -rnode
        end
    end

    # the nodes of the reachable set are active again
    @inbounds for k in oneto(rchsze)
        rnode = rchset[k]; superwght[rnode] = -superwght[rnode]
    end

    # store the reachable set as the list of the new element `mdnode`
    # (compress `iw` if necessary). It fits: it is no longer than the lists
    # released above, and no other list grew
    if pfree + convert(E, rchsze) - one(E) > iwlen
        pfree = mdcompress!(neqns, pe, len, iw, pfree)
        @assert pfree + convert(E, rchsze) - one(E) <= iwlen
    end

    @inbounds pe[mdnode] = pfree; ewght = zero(V)

    @inbounds for k in oneto(rchsze)
        rnode = rchset[k]
        iw[pfree] = rnode; pfree += one(E); ewght += superwght[rnode]
    end

    @inbounds len[mdnode] = rchsze; degree[mdnode] = ewght; wext[mdnode] = one(V)
    return rchsze, pfree, ewght + dS, swght, wflg, max(lemax, ewght), fsze, fwght
end

"""
    mdupdate!(mdnode, rchsze, delta, dS, swght, fwght, pe, len, elen, iw,
        supersize, superwght, degree, deghead, degnext, degprev, hashval, rchset, wext, marker, tag, xw, neqns, extlen, dwght,
        nwords, dmask, ddeg, nbypass, mindeg, external, mass, offset, debug)

This routine updates the degree structure after the elimination of
`mdnode`. Indistinguishable nodes in the reachable set are detected by
hashing their quotient neighbors, and merged into supernodes. With mass
elimination, each supernode whose list holds only `mdnode`, and whose
dense neighbors are those of `mdnode`, is eliminated with `mdnode`; let
`m` be their weight (0 otherwise). Each other supernode of the reachable
set gets the bound

    delta - m - w(u) + wext[u] + (weight of its dense neighbors not adjacent to `mdnode`),

raised to its old bound minus `swght + fwght + m` if it was not formed by
a merge. It is exact if it equals the upper bound, the same expression
with `extlen[u]` added. With external degree, the supernodes whose bound
is less than `max(0, delta - swght) + w(u) - 1` are recomputed at once
(`mdrecompute!`). The supernodes are returned to the degree lists.

input parameters:

  - `mdnode`: the eliminated node
  - `(rchset, rchsze)`: the reachable set of `mdnode`
  - `delta`: the weight of the reachable set, with the dense neighbors of
    `mdnode`
  - `dS`: the weight of the dense neighbors of `mdnode`
  - `swght`: the weight of `mdnode`
  - `fwght`: the weight of the false twins of `mdnode` (see `mdelim!`)
  - `(pe, len, elen, iw)`: quotient graph
  - `(wext, extlen)`: the terms of the bounds (see `mdelim!`); on output,
    `extlen[u]` is the upper bound of each node `u` of the reachable set
  - `external`: key the degree lists by external degree instead of degree
  - `mass`: mass elimination
  - `offset`: separates the keys of the lists that are not exact

updated parameters:

  - `mindeg`: new minimum key after degree update
  - `ddeg`: the degrees of the dense nodes, which lose the nodes eliminated
    with `mdnode`
  - `xw`: the lists of the eager pass (see `mdrecompute!`)
  - `(supersize, superwght, degree)`: supernodes and their bounds
  - `(deghead, degnext, degprev)`: degree lists
  - `hashval`: the hash buckets, which share `deghead` with the degree
    lists (empty on output)
  - `(marker, tag)`: marker vector
"""
@inline function mdupdate!(
        mdnode::V,
        rchsze::V,
        delta::V,
        dS::V,
        swght::V,
        fwght::V,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        supersize::AbstractVector{V},
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        hashval::AbstractVector{UInt64},
        rchset::AbstractVector{V},
        wext::AbstractVector{V},
        marker::AbstractVector{I},
        tag::I,
        xw::Vector{V},
        neqns::V,
        extlen::AbstractVector{V},
        dwght::AbstractVector{V},
        nwords::Int,
        dmask::AbstractVector{UInt64},
        ddeg::AbstractVector{V},
        nbypass::V,
        mindeg::V,
        external::Bool,
        mass::Bool,
        offset::V,
        debug::Bool,
    ) where {I, V, E}
    # the supernodes formed by a merge are marked with `mtag`
    tag += one(I); mtag = tag

    # for each hash bucket of the reachable set, do the following...
    @inbounds for k in oneto(rchsze)
        hash = degprev[rchset[k]]
        head = deghead[hash]

        # take the first node of the bucket, and empty it
        if isnegative(head)
            inode = -head; deghead[hash] = zero(V)
        elseif ispositive(head) && ispositive(degprev[head])
            inode = degprev[head]; degprev[head] = -hash
        else
            continue
        end

        # for each node `inode` in the bucket, do the following...
        while !iszero(inode) && !iszero(degnext[inode])
            # find the first node `jnode` after `inode` whose hash value and
            # list lengths match, with `jlast` before it: only then are the
            # lists compared
            jlast = inode; jnode = degnext[inode]

            while !iszero(jnode) && !(hashval[jnode] == hashval[inode] && len[jnode] == len[inode] && elen[jnode] == elen[inode])
                jlast = jnode; jnode = degnext[jnode]
            end

            if iszero(jnode)
                inode = degnext[inode]
                continue
            end

            # mark the quotient neighbors of `inode`
            tag += one(I)
            pstart = pe[inode]
            pstop = pstart + convert(E, len[inode]) - one(E)

            for p in pstart:pstop
                marker[iw[p]] = tag
            end

            # for each node `jnode` from there on in the bucket, do the
            # following...

            while !iszero(jnode)
                jnext = degnext[jnode]
                same = hashval[jnode] == hashval[inode] && len[jnode] == len[inode] && elen[jnode] == elen[inode]

                if same
                    qstart = pe[jnode]
                    qstop = qstart + convert(E, len[jnode]) - one(E)

                    for q in qstart:qstop
                        if marker[iw[q]] != tag
                            same = false
                            break
                        end
                    end
                end

                if same && ispositive(nbypass)
                    same = mdsame(dmask, nwords, inode, jnode)
                end

                if same
                    # `jnode` is indistinguishable from `inode`: merge them
                    # into a new supernode, and remove `jnode` from the bucket
                    supersize[inode] += supersize[jnode]; supersize[jnode] = zero(V)
                    superwght[inode] += superwght[jnode]; superwght[jnode] = zero(V)
                    degnext[jnode] = -inode
                    len[jnode] = zero(V); elen[jnode] = zero(V)
                    degnext[jlast] = jnext
                    marker[inode] = mtag
                else
                    jlast = jnode
                end

                jnode = jnext
            end

            inode = degnext[inode]
        end
    end

    # keep the principal nodes of the reachable set. With mass elimination,
    # a node whose list holds only `mdnode`, and whose dense neighbors are
    # those of `mdnode`, is eliminated with `mdnode` instead: with degree, it
    # is indistinguishable from `mdnode`. Its dense neighbors lose it
    nrch = zero(V); mwght = zero(V)

    @inbounds for k in oneto(rchsze)
        rnode = rchset[k]; rwght = superwght[rnode]

        if ispositive(rwght)
            degprev[rnode] = degnext[rnode] = zero(V)

            if mass && isone(len[rnode]) && (!ispositive(nbypass) || dwght[rnode] == dS)
                ispositive(nbypass) && mdlose!(ddeg, dmask, nwords, rnode, rwght)
                supersize[mdnode] += supersize[rnode]; supersize[rnode] = zero(V)
                superwght[rnode] = zero(V); degnext[rnode] = -mdnode
                len[rnode] = elen[rnode] = zero(V); mwght += rwght
            else
                nrch += one(V); rchset[nrch] = rnode
            end
        end
    end

    # the floor of the eager pass is that of `mdnode`. The other nodes lose
    # the nodes eliminated with it, and its false twins, as they lose
    # `mdnode`
    floor = max(zero(V), delta - swght)
    delta -= mwght; swght += mwght + fwght
    @inbounds degree[mdnode] -= mwght

    # for each principal node in the reachable set, assign its bound; with
    # external degree, chain the nodes for the eager pass through `degnext`,
    # in order
    zhead = ztail = zero(V)

    @inbounds for k in oneto(nrch)
        rnode = rchset[k]; rwght = superwght[rnode]
        lo = delta - rwght + wext[rnode]

        if ispositive(nbypass)
            lo += dwght[rnode] - dS
        end

        hi = lo + extlen[rnode]

        # a supernode not formed by a merge lost at most `swght` neighbors
        if marker[rnode] != mtag
            lo = max(lo, degree[rnode] - swght)
        end

        degree[rnode] = lo; extlen[rnode] = hi

        if external && lo < hi && lo < floor + rwght - one(V)
            if iszero(ztail)
                zhead = rnode
            else
                degnext[ztail] = rnode
            end

            ztail = rnode
        end
    end

    # the eager pass
    if ispositive(zhead)
        mindeg, tag = mdrecompute!(zhead, mdnode, pe, len, elen, iw,
            superwght, degree, deghead, degnext, degprev, hashval, marker,
            tag, xw, neqns, extlen, dwght, nbypass, mindeg, external, offset,
            debug)
    end

    # return the other nodes to the degree structure
    @inbounds for k in oneto(nrch)
        rnode = rchset[k]

        if iszero(degprev[rnode])
            key = mdkey(degree[rnode], superwght[rnode], external)
            mindeg = mdinsert!(key, mindeg, rnode, degree[rnode] == extlen[rnode], deghead, degnext, degprev, offset)
        end
    end

    return mindeg, tag
end

"""
    mdvisit!(key, pe, len, elen, iw, superwght, degree, deghead, degnext,
        degprev, hashval, rchset, marker, tag, xw, neqns, extlen, dwght,
        nbypass, external, offset, debug)

This routine recomputes every node with key `key` that is not exact. The
nodes are grouped by their most recent element, and each group is
recomputed at once (`mdrecompute!`). Returns the new minimum key and the
tag.

working arrays:

  - `degprev`: for an element, the first node of its group
  - `degnext`: for a node taken from the list, the next node in its group
  - `rchset`: the elements with a group (the reachable set is not in use
    between eliminations)
"""
function mdvisit!(
        key::V,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        hashval::AbstractVector{UInt64},
        rchset::AbstractVector{V},
        marker::AbstractVector{I},
        tag::I,
        xw::Vector{V},
        neqns::V,
        extlen::AbstractVector{V},
        dwght::AbstractVector{V},
        nbypass::V,
        external::Bool,
        offset::V,
        debug::Bool,
    ) where {I, V, E}
    # take the whole list, and group it by most recent element
    tag += one(I); gtag = tag; esze = zero(V); mindeg = key
    @inbounds node = deghead[key + offset]; deghead[key + offset] = zero(V)

    @inbounds while ispositive(node)
        nxnode = degnext[node]
        degnext[node] = degprev[node] = zero(V)
        elmnt = iw[pe[node] + convert(E, elen[node]) - one(E)]

        if marker[elmnt] != gtag
            marker[elmnt] = gtag; degprev[elmnt] = zero(V)
            esze += one(V); rchset[esze] = elmnt
        end

        degnext[node] = degprev[elmnt]; degprev[elmnt] = node
        node = nxnode
    end

    # recompute each group
    @inbounds for k in oneto(esze)
        elmnt = rchset[k]

        mindeg, tag = mdrecompute!(degprev[elmnt], elmnt, pe, len, elen, iw,
            superwght, degree, deghead, degnext, degprev, hashval, marker,
            tag, xw, neqns, extlen, dwght, nbypass, mindeg, external, offset,
            debug)
    end

    return mindeg, tag
end

"""
    mdrecompute!(zhead, elmnt, pe, len, elen, iw, superwght, degree,
        deghead, degnext, degprev, hashval, marker, tag, xw, neqns, extlen,
        dwght, nbypass, mindeg, external, offset, debug)

This routine recomputes the external degrees of the nodes of the chain that
starts at `zhead` and is linked by `degnext`, whose most recent element is
`elmnt`. The nodes of `elmnt` are marked, each
other element of the nodes is scanned once to list its nodes outside
`elmnt`, and the external degree of each node is the weight of `elmnt`, of
the union of these lists and of its uneliminated neighbors, and of its
dense neighbors, minus its own weight. The nodes become exact and are
inserted into the degree lists. Returns the new minimum key and the tag.

The elements in the lists of all the nodes (the core) are united once, as
they are scanned, so each node only adds the lists of its other elements.
This keeps groups whose nodes share many overlapping elements cheap.

working arrays:

  - `(hashval, extlen, xw)`: the nodes of each element `f` outside `elmnt`
    are listed in `xw[s:s + extlen[f] - 1]`, with `s = hashval[f]` (the
    hash values of elements are not in use). `xw` grows first if it is too
    short.
  - `marker`: the nodes of `elmnt` are marked `etag`; while the elements
    are counted, `marker[f] - base` is the number of nodes that list `f`;
    the core and the nodes of its lists are marked `ctag`, the other
    elements `stag`, and the union of each node `utag`.
"""
@inline function mdrecompute!(
        zhead::V,
        elmnt::V,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        hashval::AbstractVector{UInt64},
        marker::AbstractVector{I},
        tag::I,
        xw::Vector{V},
        neqns::V,
        extlen::AbstractVector{V},
        dwght::AbstractVector{V},
        nbypass::V,
        mindeg::V,
        external::Bool,
        offset::V,
        debug::Bool,
    ) where {I, V, E}
    # mark the nodes of `elmnt`
    tag += one(I); etag = tag
    @inbounds estart = pe[elmnt]
    @inbounds estop = estart + convert(E, len[elmnt]) - one(E)

    @inbounds for j in estart:estop
        node = iw[j]

        if ispositive(superwght[node])
            marker[node] = etag
        end
    end

    # count the nodes of the group
    gsize = 0; znode = zhead

    @inbounds while ispositive(znode)
        gsize += 1; znode = degnext[znode]
    end

    # mark the other elements of the nodes with the number of nodes that list
    # them, `marker[f] - base`, and count their entries (elements and nodes
    # are marked in the same vector: an element is never an uneliminated
    # node)
    tag += one(I); base = tag; tag += convert(I, gsize); need = zero(E)

    znode = zhead

    @inbounds while ispositive(znode)
        znext = degnext[znode]
        p1 = pe[znode]
        p2 = p1 + convert(E, elen[znode]) - one(E)

        for p in p1:p2
            f = iw[p]

            if f != elmnt
                if marker[f] <= base
                    marker[f] = base + one(I); need += convert(E, len[f])
                else
                    marker[f] += one(I)
                end
            end
        end

        znode = znext
    end

    # the lists of nodes outside `elmnt` go into `xw` (grow it if necessary)
    if need > length(xw)
        resize!(xw, max(convert(Int, need), 2 * length(xw)))
    end

    # scan each other element of the nodes once, and list its nodes outside
    # `elmnt`. The elements listed by every node of the group (the core) are
    # marked `ctag`, and the union of their lists, also marked `ctag`, is
    # taken once here: its weight is `cwght`
    tag += one(I); stag = tag; tag += one(I); ctag = tag; pos = one(E)
    full = base + convert(I, gsize); cwght = zero(V)

    znode = zhead

    @inbounds while ispositive(znode)
        znext = degnext[znode]
        p1 = pe[znode]
        p2 = p1 + convert(E, elen[znode]) - one(E)

        for p in p1:p2
            f = iw[p]

            if f != elmnt && base < marker[f] <= full
                core = marker[f] == full
                marker[f] = core ? ctag : stag; hashval[f] = pos % UInt64; fstart = pos
                jstart = pe[f]
                jstop = jstart + convert(E, len[f]) - one(E)

                for j in jstart:jstop
                    node = iw[j]

                    if ispositive(superwght[node]) && marker[node] != etag
                        xw[pos] = node; pos += one(E)

                        if core && marker[node] != ctag
                            marker[node] = ctag; cwght += superwght[node]
                        end
                    end
                end

                extlen[f] = convert(V, pos - fstart)
            end
        end

        znode = znext
    end

    # for each node, take the union of the lists and of its uneliminated
    # neighbors
    @inbounds ewght = degree[elmnt]

    znode = zhead

    @inbounds while ispositive(znode)
        znext = degnext[znode]
        tag += one(I); utag = tag; deg = cwght
        p1 = pe[znode]
        p2 = p1 + convert(E, elen[znode]) - one(E)
        p4 = p1 + convert(E, len[znode]) - one(E)

        for p in p1:p2
            f = iw[p]

            if f != elmnt && marker[f] != ctag
                qstart = hashval[f] % E
                qstop = qstart + convert(E, extlen[f]) - one(E)

                for q in qstart:qstop
                    node = xw[q]

                    if marker[node] != utag && marker[node] != ctag
                        marker[node] = utag; deg += superwght[node]
                    end
                end
            end
        end

        for p in (p2 + one(E)):p4
            node = iw[p]; wght = superwght[node]

            if ispositive(wght) && marker[node] != etag && marker[node] != utag && marker[node] != ctag
                marker[node] = utag; deg += wght
            end
        end

        deg += ewght - superwght[znode]

        if ispositive(nbypass)
            deg += dwght[znode]
        end

        debug && @assert deg >= degree[znode]
        degree[znode] = deg
        key = mdkey(deg, superwght[znode], external)
        mindeg = mdinsert!(key, mindeg, znode, true, deghead, degnext, degprev, offset)
        znode = znext
    end

    return mindeg, tag
end

"""
    mdremove!(node, deghead, degnext, degprev, offset)

This routine removes `node` from the degree lists.
"""
@inline function mdremove!(
        node::V,
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        offset::V,
    ) where {V}
    @inbounds pvnode = degprev[node]

    if !iszero(pvnode)
        @inbounds nxnode = degnext[node]

        if ispositive(nxnode)
            @inbounds degprev[nxnode] = pvnode
        end

        if ispositive(pvnode)
            @inbounds degnext[pvnode] = nxnode
        else
            @inbounds deghead[-pvnode] = nxnode
        end

        @inbounds degprev[node] = zero(V)
    end

    return
end

"""
    mdinsert!(key, mindeg, node, isexact, deghead, degnext,
        degprev, offset)

This routine inserts `node` into the exact (`isexact`) or other degree
list with key `key`, and returns the new minimum key.
"""
@inline function mdinsert!(
        key::V,
        mindeg::V,
        node::V,
        isexact::Bool,
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        offset::V,
    ) where {V}
    index = isexact ? key : key + offset
    @inbounds firstnode = deghead[index]
    @inbounds deghead[index] = node
    @inbounds degnext[node] = firstnode
    @inbounds degprev[node] = -index

    if ispositive(firstnode)
        @inbounds degprev[firstnode] = node
    end

    return min(key, mindeg)
end

# the key of a node: its external degree plus 1, or its degree plus 1
@inline function mdkey(deg::V, wght::V, external::Bool) where {V}
    return external ? deg + one(V) : deg + wght
end

# a list is hashed as the sum of a mix of its entries, so that lists with
# the same sum of entries rarely collide
@inline function mdmix(x::Integer)
    z = (x % UInt64) * 0x9e3779b97f4a7c15
    return z ⊻ (z >>> 29)
end

# ---------------------------------------------------------------------
# dense nodes
# ---------------------------------------------------------------------
#
# The dense nodes are kept out of the quotient graph. Fill never passes
# through an uneliminated node, so the quotient graph of the remaining nodes
# represents their elimination graph exactly. The adjacency between any node
# and the dense nodes is kept in bit masks: when a node is eliminated, its
# neighbors become a clique, so every node in its reachable set (and every
# dense neighbor) acquires its dense neighbors. A dense node rejoins the
# quotient graph when its degree is minimum, and is then eliminated.
#
#   - `ndense`: number of dense nodes
#   - `nwords`: number of 64-bit words in a mask
#   - `dindex`: `dindex[node]` is the index of a dense node, or 0
#   - `dnode`: `dnode[j]` is the dense node with index `j`; it is negated
#     once the node rejoins the quotient graph
#   - `ddeg`: `ddeg[j]` is the weighted external degree of dense node `j`
#   - `dmask`: `dmask[(node - 1) * nwords + w]` is word `w` of the mask of
#     the dense neighbors of `node`

@inline function mdoffset(nwords::Int, node::Integer)
    return (convert(Int, node) - 1) * nwords
end

@inline function mdsetbit!(dmask::AbstractVector{UInt64}, nwords::Int, node::Integer, j::Integer)
    jj = convert(Int, j) - 1
    @inbounds dmask[mdoffset(nwords, node) + (jj >> 6) + 1] |= one(UInt64) << (jj & 63)
    return
end

# number of dense neighbors of `node`
function mdcount(dmask::AbstractVector{UInt64}, nwords::Int, node::Integer)
    off = mdoffset(nwords, node); cnt = 0

    @inbounds for w in 1:nwords
        cnt += count_ones(dmask[off + w])
    end

    return cnt
end

# weight of the dense neighbors of `node`
function mdweight(dmask::AbstractVector{UInt64}, nwords::Int, dnode::AbstractVector{V}, node::Integer, superwght::AbstractVector{V}) where {V}
    off = mdoffset(nwords, node); wght = zero(V)

    @inbounds for w in 1:nwords
        word = dmask[off + w]

        while !iszero(word)
            j = ((w - 1) << 6) + trailing_zeros(word) + 1
            wght += superwght[abs(dnode[j])]
            word &= word - one(UInt64)
        end
    end

    return wght
end

# do `inode` and `jnode` have the same dense neighbors?
function mdsame(dmask::AbstractVector{UInt64}, nwords::Int, inode::Integer, jnode::Integer)
    ioff = mdoffset(nwords, inode); joff = mdoffset(nwords, jnode)

    @inbounds for w in 1:nwords
        dmask[ioff + w] == dmask[joff + w] || return false
    end

    return true
end

"""
    mddenseinit!(ddeg, dmask, nwords, ndense, dnode, neqns, vwght)

This routine computes the degrees of the dense nodes. On input, the masks
hold the original adjacency to the dense nodes.
"""
function mddenseinit!(ddeg::AbstractVector{V}, dmask::AbstractVector{UInt64}, nwords::Int, ndense::V, dnode::AbstractVector{V}, neqns::V, vwght::AbstractVector{V}) where {V}
    @inbounds for j in oneto(ndense)
        ddeg[j] = zero(V)
    end

    @inbounds for node in oneto(neqns)
        off = mdoffset(nwords, node); wght = vwght[node]

        for w in 1:nwords
            word = dmask[off + w]

            while !iszero(word)
                j = ((w - 1) << 6) + trailing_zeros(word) + 1
                ddeg[j] += wght
                word &= word - one(UInt64)
            end
        end
    end

    return
end

"""
    mdlose!(ddeg, dmask, nwords, node, wght)

This routine removes `node`, which is about to be eliminated, from the
degrees of its dense neighbors.
"""
function mdlose!(ddeg::AbstractVector{V}, dmask::AbstractVector{UInt64}, nwords::Int, node::V, wght::V) where {V}
    off = mdoffset(nwords, node)

    @inbounds for w in 1:nwords
        word = dmask[off + w]

        while !iszero(word)
            j = ((w - 1) << 6) + trailing_zeros(word) + 1
            ddeg[j] -= wght
            word &= word - one(UInt64)
        end
    end

    return
end

"""
    mddense!(mdnode, rchsze, rchset, dwght, superwght, nwords, dnode, ddeg,
        dmask, nmissing)

This routine updates the adjacency to the dense nodes after the elimination
of `mdnode`: its neighbors become a clique, so every node in the reachable
set, and every dense neighbor of `mdnode`, becomes adjacent to every dense
neighbor of `mdnode`. Each new adjacency is added to the dense weight
`dwght` of the node, or to the degree `ddeg` of the dense node. Returns the
updated number `nmissing` of ordered pairs of nonadjacent dense nodes.
"""
function mddense!(
        mdnode::V,
        rchsze::V,
        rchset::AbstractVector{V},
        dwght::AbstractVector{V},
        superwght::AbstractVector{V},
        nwords::Int,
        dnode::AbstractVector{V},
        ddeg::AbstractVector{V},
        dmask::AbstractVector{UInt64},
        nmissing::Int,
    ) where {V}
    moff = mdoffset(nwords, mdnode)

    # skip if `mdnode` has no dense neighbors
    empty = true

    @inbounds for w in 1:nwords
        if !iszero(dmask[moff + w])
            empty = false
            break
        end
    end

    empty && return nmissing

    @inbounds for k in oneto(rchsze)
        rnode = rchset[k]
        roff = mdoffset(nwords, rnode); rdeg = zero(V); rwght = superwght[rnode]

        for w in 1:nwords
            mword = dmask[moff + w]; rword = dmask[roff + w]
            new = mword & ~rword

            while !iszero(new)
                j = ((w - 1) << 6) + trailing_zeros(new) + 1
                rdeg += superwght[dnode[j]]; ddeg[j] += rwght
                new &= new - one(UInt64)
            end

            dmask[roff + w] = mword | rword
        end

        dwght[rnode] += rdeg
    end

    # the dense neighbors of `mdnode` become pairwise adjacent
    if ispositive(nmissing)
        @inbounds for w in 1:nwords
            word = dmask[moff + w]

            while !iszero(word)
                j = ((w - 1) << 6) + trailing_zeros(word) + 1
                word &= word - one(UInt64)
                dj = dnode[j]; joff = mdoffset(nwords, dj); jdeg = zero(V)

                for v in 1:nwords
                    mword = dmask[moff + v]

                    if v == w
                        # a node is not its own neighbor
                        mword &= ~(one(UInt64) << ((j - 1) & 63))
                    end

                    new = mword & ~dmask[joff + v]

                    while !iszero(new)
                        l = ((v - 1) << 6) + trailing_zeros(new) + 1
                        jdeg += superwght[dnode[l]]; nmissing -= 1
                        new &= new - one(UInt64)
                    end

                    dmask[joff + v] |= mword
                end

                ddeg[j] += jdeg
            end
        end
    end

    return nmissing
end

"""
    mdreinsert!(j, neqns, pe, len, elen, iw, iwlen, pfree, superwght,
        nwords, dindex, dnode, dmask, nbrs)

This routine returns the dense node with index `j` to the quotient graph.
Its neighbors are the uneliminated principal nodes, other than dense nodes,
whose masks contain `j`; they become its list, and it is appended to each
of their lists (which are moved to the end of `iw` to make room). The node
is then removed from every mask. Returns the updated `pfree`.

The lists fit after a compression. Outside a rejoin, each list is no
longer than in the quotient graph that keeps the dense nodes, so the lists
take at most `nnz` entries. Until the dense node is eliminated, the rejoin
adds fewer than `2 neqns`: its list and one entry per neighbor, or, while a
list is copied, the entries appended so far and that list.

working arrays:

  - `nbrs`: the neighbors of the dense node
"""
function mdreinsert!(
        j::V,
        neqns::V,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        iwlen::E,
        pfree::E,
        superwght::AbstractVector{V},
        nwords::Int,
        dindex::AbstractVector{V},
        dnode::AbstractVector{V},
        dmask::AbstractVector{UInt64},
        nbrs::AbstractVector{V},
    ) where {V, E}
    @inbounds dnd = dnode[j]; dnode[j] = -dnd
    w = ((convert(Int, j) - 1) >> 6) + 1
    bit = one(UInt64) << ((convert(Int, j) - 1) & 63)
    cnt = zero(V)

    @inbounds for node in oneto(neqns)
        i = mdoffset(nwords, node) + w

        if !iszero(dmask[i] & bit)
            dmask[i] &= ~bit

            if !ispositive(dindex[node]) && ispositive(superwght[node])
                # move the list of `node` to the end of `iw` and append
                # `dnd` to it
                nlen = len[node]

                if pfree + convert(E, nlen) > iwlen
                    pfree = mdcompress!(neqns, pe, len, iw, pfree)
                    @assert pfree + convert(E, nlen) <= iwlen
                end

                pstart = pe[node]; pe[node] = pfree

                for p in pstart:(pstart + convert(E, nlen) - one(E))
                    iw[pfree] = iw[p]; pfree += one(E)
                end

                iw[pfree] = dnd; pfree += one(E)
                len[node] = nlen + one(V)
                cnt += one(V); nbrs[cnt] = node
            end
        end
    end

    if pfree + convert(E, cnt) - one(E) > iwlen
        pfree = mdcompress!(neqns, pe, len, iw, pfree)
        @assert pfree + convert(E, cnt) - one(E) <= iwlen
    end

    @inbounds pe[dnd] = pfree

    @inbounds for k in oneto(cnt)
        iw[pfree] = nbrs[k]; pfree += one(E)
    end

    @inbounds len[dnd] = cnt; elen[dnd] = zero(V)
    return pfree
end

"""
    mdcompress!(neqns, pe, len, iw, pfree)

This routine compresses the quotient graph storage `iw`, releasing the
space used by absorbed elements, merged nodes, and purged entries. The
first entry of each list is marked in a bitmap, so the released space is
skipped 64 entries at a time. Returns the new value of `pfree`.

input parameters:

  - `neqns`: number of equations
  - `len`: lengths of the lists (0 for released lists)

updated parameters:

  - `(pe, iw)`: quotient graph storage
  - `pfree`: the quotient graph is stored in `iw[1:pfree - 1]`
"""
function mdcompress!(
        neqns::V,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        iw::AbstractVector{V},
        pfree::E,
    ) where {V, E}
    # store `-node` in the first entry of the list of each node, saving the
    # entry in `pe[node]`, and mark the entry in `heads`
    heads = zeros(UInt64, cld(convert(Int, pfree) - 1, 64))

    @inbounds for node in oneto(neqns)
        if ispositive(len[node])
            p = pe[node]
            pe[node] = convert(E, iw[p])
            iw[p] = -node
            i = convert(Int, p) - 1
            heads[(i >> 6) + 1] |= one(UInt64) << (i & 63)
        end
    end

    # move the lists to the front of `iw`, in order
    pdst = one(E)

    @inbounds for w in eachindex(heads)
        bits = heads[w]

        while !iszero(bits)
            psrc = convert(E, ((w - 1) << 6) + trailing_zeros(bits) + 1)
            bits &= bits - one(UInt64)
            node = -iw[psrc]; psrc += one(E)
            iw[pdst] = convert(V, pe[node]); pe[node] = pdst; pdst += one(E)

            for _ in 2:len[node]
                iw[pdst] = iw[psrc]; pdst += one(E); psrc += one(E)
            end
        end
    end

    return pdst
end

"""
    mdnumber!(neqns, supersize, degnext)

This routine performs the final step in producing the permutation and
inverse permutation vectors in the minimum degree ordering algorithm.

input parameters:

  - `neqns`: number of equations
  - `supersize`: size of supernodes (>0 for principals, 0 for merged)

updated parameters:

  - `degnext`: on input, stores -ordering for principals, -parent for merged.
    on output, stores the final positive ordering for all nodes.
"""
function mdnumber!(neqns::V, supersize::AbstractVector{V}, degnext::AbstractVector{V}) where {V}
    # first pass: convert principals to positive ordering
    # and do path compression for merged nodes.
    # we repurpose `supersize` as `mergelastnum` for principals.
    @inbounds for node in oneto(neqns)
        if ispositive(supersize[node])
            # principal: negate to get positive ordering
            # and use `supersize` as `mergelastnum`
            supersize[node] = degnext[node] = -degnext[node]
        else
            # merged node: trace to `root` and compress path
            root = -degnext[node]

            while iszero(supersize[root])
                root = -degnext[root]
            end

            # path compression: point all nodes on path to `root`
            while node != root
                next = -degnext[node]
                degnext[node] = -root
                node = next
            end
        end
    end

    # second pass: assign orderings
    # all merged nodes now point directly to `root`
    @inbounds for node in oneto(neqns)
        if iszero(supersize[node])
            root = -degnext[node]
            degnext[node] = supersize[root] += one(V)
        end
    end

    return
end

# debugging: recompute the external degree of every uneliminated principal
# node from the quotient graph, and compare it to its bound
function mdcheck(
        neqns::V,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        supersize::AbstractVector{V},
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        total::V,
        dwght::AbstractVector{V},
        ndense::V,
        nwords::Int,
        dindex::AbstractVector{V},
        dnode::AbstractVector{V},
        dmask::AbstractVector{UInt64},
    ) where {V, E}
    marker = zeros(V, neqns)

    # the exact nodes are those in the exact degree lists
    isexact = falses(neqns)

    for key in oneto(total)
        node = deghead[key]

        while ispositive(node)
            isexact[node] = true; node = degnext[node]
        end
    end

    isactive(node) = !isnegative(degnext[node]) && ispositive(supersize[node])
    isbypassed(node) = ispositive(dindex[node]) && ispositive(dnode[dindex[node]])

    for node in oneto(neqns)
        isactive(node) || continue
        ispositive(ndense) && isbypassed(node) && continue
        deg = zero(V); marker[node] = node

        if ispositive(ndense)
            dwgt = mdweight(dmask, nwords, dnode, node, superwght)
            @assert dwght[node] == dwgt
            deg += dwgt
        end

        # `iw[estart:estop]` are the elements of `node`, and
        # `iw[nstart:nstop]` are its neighbors
        estart = pe[node]
        estop = estart + convert(E, elen[node]) - one(E)
        nstart = estop + one(E)
        nstop = estart + convert(E, len[node]) - one(E)

        for p in nstart:nstop
            neighbor = iw[p]

            if isactive(neighbor) && marker[neighbor] != node
                marker[neighbor] = node; deg += superwght[neighbor]
            end
        end

        for p in estart:estop
            elmnt = iw[p]
            @assert isnegative(degnext[elmnt]) && ispositive(len[elmnt])
            qstart = pe[elmnt]
            qstop = qstart + convert(E, len[elmnt]) - one(E)

            for q in qstart:qstop
                neighbor = iw[q]

                if isactive(neighbor) && marker[neighbor] != node
                    marker[neighbor] = node; deg += superwght[neighbor]
                end
            end
        end

        @assert degree[node] <= deg
        isexact[node] && @assert degree[node] == deg
    end

    return
end

end
