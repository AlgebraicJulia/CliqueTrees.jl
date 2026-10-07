#= A collection of routines to find an exact minimum degree ordering.
   Based on the algorithm FastMinDegree by Robert Cummings, Matthew
   Fahrbach, and Animesh Fatehpuria.
   For information on the minimum degree algorithm see the articles:
   A fast minimum degree algorithm and matching lower bound by Robert
   Cummings, Matthew Fahrbach, and Animesh Fatehpuria, SODA 2021
   The evolution of the minimum degree algorithm by Alan George and
   Joseph Liu, SIAM Rev. 31 pp. 1 - 19, 1989
=#

module MDLib

using Base: oneto
using FillArrays
using ..Utilities

export md

"""
    md(neqns, xadj, adjncy; external=false, mass=false)
    md(neqns, vwght, xadj, adjncy; external=false, mass=false)

Compute an exact minimum degree ordering of the graph `(xadj, adjncy)`.
Returns `index`, where `index[node]` is the position of `node`. The vertex
weights `vwght` must be positive integers.

  - `external`: minimize the external degree instead of the degree
  - `mass`: use mass elimination. The ordering remains exact for the
    degree, but not for the external degree.
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
        weight < 1 && throw(ArgumentError("vertex weights must be positive"))
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

function md(::Type{V}, neqns::Integer, total::Integer, vwght::AbstractVector, xadj::AbstractVector{E}, adjncy::AbstractVector; external::Bool = false, mass::Bool = false, debug::Bool = false) where {V, E}
    @assert neqns < length(xadj)
    n = convert(V, neqns)
    iszero(n) && return FVector{V}(undef, 0)
    @inbounds nnz = xadj[neqns + 1] - one(E)

    # the quotient graph never needs more than `nnz` entries of `iw`;
    # the rest is room to grow between compressions
    iwlen = twice(nnz) + convert(E, twice(neqns))

    newvwght = FVector{V}(undef, n)
    marker = FVector{V}(undef, n)
    supersize = FVector{V}(undef, n)
    superwght = FVector{V}(undef, n)
    degree = FVector{V}(undef, n)
    deghead = FVector{V}(undef, total + 1)
    degnext = FVector{V}(undef, n)
    degprev = FVector{V}(undef, n)
    hashhead = FVector{V}(undef, nextpow(2, n))
    rchset = FVector{V}(undef, n)
    xset = FVector{V}(undef, n)
    yset = FVector{V}(undef, n)
    pe = FVector{E}(undef, n)
    len = FVector{V}(undef, n)
    elen = FVector{V}(undef, n)
    iw = FVector{V}(undef, iwlen)

    @inbounds for node in oneto(n)
        newvwght[node] = trunc(V, vwght[node])
    end

    md_impl!(marker, supersize, superwght, degree, deghead, degnext,
        degprev, hashhead, rchset, xset, yset, pe, len, elen, iw, iwlen,
        convert(V, total), n, external, mass, debug, newvwght, xadj, adjncy)

    return degnext
end

function md_impl!(
        marker::AbstractVector{I},
        supersize::AbstractVector{V},
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        hashhead::AbstractVector{V},
        rchset::AbstractVector{V},
        xset::AbstractVector{V},
        yset::AbstractVector{V},
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        iwlen::E,
        total::V,
        neqns::V,
        external::Bool,
        mass::Bool,
        debug::Bool,
        vwght::AbstractVector{V},
        xadj::AbstractVector,
        adjncy::AbstractVector,
    ) where {I, V, E}
    @assert iwlen <= length(iw)

    # copy the adjacency structure into `iw`, dropping self loops
    pfree = one(E)

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

    md!_impl!(marker, supersize, superwght, degree, deghead, degnext,
        degprev, hashhead, rchset, xset, yset, pe, len, elen, iw, iwlen,
        pfree, total, neqns, external, mass, debug, vwght)

    return degnext
end

"""
    md!_impl!(marker, supersize, superwght, degree, deghead, degnext,
        degprev, hashhead, rchset, xset, yset, pe, len, elen, iw, iwlen,
        pfree, total, neqns, external, mass, debug, vwght)

This routine implements the exact minimum degree algorithm. It makes use
of the implicit representation of elimination graphs by quotient graphs,
and the notion of indistinguishable nodes. It also implements the
modifications by mass elimination and minimum external degree.

The degree of every node is known exactly at all times. When a node is
eliminated, its reachable set is formed one generalized element at a time;
merging an element can only create fill edges between the nodes it adds
and the nodes of the reachable set it does not contain, so only these
pairs are tested for adjacency in the quotient graph. Each new fill edge is
added to the degrees of its two endpoints.

A node of weight w stands for w indistinguishable nodes, so the degree of a
node is the weight of its closed neighborhood minus one, and its external
degree is the weight of its neighbors outside its supernode.

input parameters:

  - `neqns`: number of equations
  - `total`: sum of the vertex weights
  - `vwght`: vertex weights
  - `external`: minimize the external degree instead of the degree
  - `mass`: use mass elimination; the ordering remains exact for the
    degree, but not for the external degree
  - `debug`: recompute and check all degrees after every elimination
  - `iwlen`: length of `iw`

updated parameters:

  - `(pe, len, elen, iw)`: quotient graph. On input, `(pe, len, iw)` holds
    the adjacency structure. For an uneliminated node, the list
    `iw[pe[node]:pe[node] + len[node] - 1]` holds `elen[node]` elements
    followed by its uneliminated neighbors; for an eliminated node
    (element), it holds the nodes in the element. An element is absorbed,
    and the list of a merged node released, by setting its length to 0.
    Destroyed on output.
  - `pfree`: the quotient graph is stored in `iw[1:pfree - 1]`

output parameters:

  - `degnext`: the minimum degree ordering (repurposed from degree list
    links)

working arrays:

  - `deghead`: points to first node with key deg, or 0 if there are no
    such nodes. The key of a node is its degree plus 1, or its external
    degree plus 1 if `external` is true.
  - `degnext`: during algorithm, points to the next node in the degree list
    (or hash bucket) associated with node, or stores negative values for
    eliminated/merged nodes (-num for an eliminated node, -parent for a
    merged node). After algorithm, stores the final ordering.
  - `degprev`: points to the previous node in a degree list associated with
    node, or the negative of the key of node (if node was the first in its
    degree list), or 0 if the node is not in the degree lists. While
    indistinguishable nodes are detected, the hash bucket of node.
  - `degree`: the exact weighted external degree of each uneliminated
    supernode; for an element, the number of nodes in it
  - `supersize`: the size of the supernodes (0 means merged into another);
    negative for the nodes in the reachable set of the current pivot
  - `superwght`: the weight of the supernodes (0 means merged into another)
  - `hashhead`: heads of the hash buckets used to detect indistinguishable
    nodes
  - `rchset`: the reachable set of the current pivot
  - `xset`, `yset`: work vectors for the degree update
  - `marker`: a temporary marker vector
"""
function md!_impl!(
        marker::AbstractVector{I},
        supersize::AbstractVector{V},
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        hashhead::AbstractVector{V},
        rchset::AbstractVector{V},
        xset::AbstractVector{V},
        yset::AbstractVector{V},
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        iwlen::E,
        pfree::E,
        total::V,
        neqns::V,
        external::Bool,
        mass::Bool,
        debug::Bool,
        vwght::AbstractVector{V},
    ) where {I, V, E}
    @assert neqns <= length(marker)
    @assert neqns <= length(supersize)
    @assert neqns <= length(superwght)
    @assert neqns <= length(degree)
    @assert total < length(deghead)
    @assert neqns <= length(degnext)
    @assert neqns <= length(degprev)
    @assert neqns <= length(hashhead)
    @assert neqns <= length(rchset)
    @assert neqns <= length(xset)
    @assert neqns <= length(yset)
    @assert neqns <= length(pe)
    @assert neqns <= length(len)
    @assert neqns <= length(elen)
    @assert neqns <= length(vwght)
    @assert pfree <= iwlen + one(E) <= length(iw) + one(E)

    # initialization for the minimum degree algorithm.
    @inbounds for node in oneto(neqns)
        marker[node] = zero(I)
        supersize[node] = one(V)
        superwght[node] = vwght[node]
        elen[node] = zero(V)
        degnext[node] = zero(V)
        degprev[node] = zero(V)
    end

    @inbounds for deg in oneto(total + one(V))
        deghead[deg] = zero(V)
    end

    @inbounds for hash in eachindex(hashhead)
        hashhead[hash] = zero(V)
    end

    # - `mindeg` is the current minimum key
    # - `num` counts the number of ordered nodes plus 1
    # - `tag` is used to facilitate marking nodes
    # - `maxtag` bounds `tag` at the start of each elimination: fewer
    #   than 5 `neqns` tags are used per elimination
    mindeg = total + one(V); num = one(V); tag = zero(I)
    maxtag = typemax(I) - five(I) * convert(I, neqns) - eight(I)

    @inbounds for node in oneto(neqns)
        deg = zero(V)
        pstart = pe[node]
        pstop = pstart + convert(E, len[node]) - one(E)

        for p in pstart:pstop
            deg += vwght[iw[p]]
        end

        degree[node] = deg

        # the key of a node is its degree plus 1, or its external degree
        # plus 1
        if external
            key = deg + one(V)
        else
            key = deg + vwght[node]
        end

        if key > one(V)
            mindeg = mdinsert!(key, mindeg, node, deghead, degnext, degprev)
        else
            # eliminate node of degree 0
            degnext[node] = -num; num += one(V)
        end
    end

    @inbounds while num <= neqns
        while !ispositive(deghead[mindeg])
            mindeg += one(V)
        end

        # remove `mdnode` from the degree structure.
        mdnode = deghead[mindeg]
        mdnextnode = degnext[mdnode]
        deghead[mindeg] = mdnextnode

        if ispositive(mdnextnode)
            degprev[mdnextnode] = -mindeg
        end

        degprev[mdnode] = zero(V)
        degnext[mdnode] = -num

        if num + supersize[mdnode] > neqns
            break
        end

        # eliminate `mdnode` and perform quotient graph
        # transformation (reset `tag` value if necessary)
        if tag > maxtag
            tag = zero(I)

            for node in oneto(neqns)
                marker[node] = zero(I)
            end
        end

        rchsze, pfree, tag = mdelim!(mdnode, pe, len, elen, iw, iwlen,
            pfree, neqns, supersize, superwght, degree, deghead, degnext,
            degprev, hashhead, marker, tag, rchset, xset, yset, mass)

        # update the degrees of the nodes in the reachable set, merge the
        # indistinguishable ones, and return them to the degree structure
        mindeg, tag = mdupdate!(mdnode, rchsze, rchset, pe, len, elen, iw,
            supersize, superwght, degree, deghead, degnext, degprev,
            hashhead, marker, tag, mindeg, external)

        num += supersize[mdnode]

        if debug
            mdcheck(neqns, pe, len, elen, iw, supersize, superwght, degree, degnext)
        end
    end

    mdnumber!(neqns, supersize, degnext)
    return degnext
end

"""
    mdelim!(mdnode, pe, len, elen, iw, iwlen, pfree, neqns, supersize,
        superwght, degree, deghead, degnext, degprev, hashhead, marker, tag,
        rchset, xset, yset, mass)

This routine eliminates the node `mdnode` of minimum degree from the
adjacency structure, which is stored in the quotient graph format. The
reachable set of `mdnode` is formed one generalized element at a time, and
the fill edges created by each element are added to the degrees of the
nodes involved. The routine then transforms the quotient graph
representation of the elimination graph: the reachable set becomes the
new element `mdnode`.

input parameters:

  - `mdnode`: node of minimum degree
  - `neqns`: number of equations
  - `iwlen`: length of `iw`
  - `tag`: tag value
  - `mass`: merge the nodes of the reachable set with no other neighbors
    with `mdnode` (mass elimination)

updated parameters:

  - `(pe, len, elen, iw)`: quotient graph
  - `pfree`: the quotient graph is stored in `iw[1:pfree - 1]`
  - `(supersize, superwght)`: the size and weight of the supernodes; on
    output, `supersize` is negative for the nodes in the reachable set
  - `degree`: degrees of the nodes in the reachable set, increased by the
    new fill edges; on output, `degree[mdnode]` is the number of nodes in
    the new element
  - `(deghead, degnext, degprev)`: degree lists; on output, the nodes in
    the reachable set are removed from them
  - `hashhead`: on output, the nodes in the reachable set are placed in
    hash buckets (`degprev` and `degnext` hold their buckets and links)
  - `marker`: marker vector

output parameters:

  - `rchsze`: size of the reachable set
  - `rchset`: the reachable set

working arrays:

  - `xset`, `yset`: work vectors for the degree update
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
        hashhead::AbstractVector{V},
        marker::AbstractVector{I},
        tag::I,
        rchset::AbstractVector{V},
        xset::AbstractVector{V},
        yset::AbstractVector{V},
        mass::Bool,
    ) where {I, V, E}
    # - `rchsze` is the number of supernodes in the reachable set
    # - `rchcnt` is the number of nodes in the reachable set
    # - `iw[estart:estop]` is the list of elements adjacent to `mdnode`
    # - `iw[nstart:nstop]` is the list of uneliminated neighbors of `mdnode`
    rchsze = zero(V); rchcnt = zero(V)
    @inbounds estart = pe[mdnode]
    @inbounds estop = estart + convert(E, elen[mdnode]) - one(E)
    @inbounds nstart = estop + one(E)
    @inbounds nstop = estart + convert(E, len[mdnode]) - one(E)

    # merge the elements adjacent to `mdnode` into the reachable set,
    # one at a time. If `elmnt` is the next element, the only new fill
    # edges are between `yset`, the nodes of `elmnt` not yet in the
    # reachable set, and `xset`, the nodes of the reachable set not in
    # `elmnt`.
    @inbounds for i in estart:estop
        elmnt = iw[i]
        tag += one(I); ysze = zero(V)
        jstart = pe[elmnt]
        jstop = jstart + convert(E, len[elmnt]) - one(E)

        for j in jstart:jstop
            node = iw[j]

            # skip eliminated and merged nodes
            if !isnegative(degnext[node])
                marker[node] = tag

                # nodes in the reachable set have negative `supersize`
                if ispositive(supersize[node])
                    ysze += one(V); yset[ysze] = node
                end
            end
        end

        iszero(ysze) && continue

        if ispositive(rchsze)
            xsze = zero(V)

            for k in oneto(rchsze)
                node = rchset[k]

                if marker[node] < tag
                    xsze += one(V); xset[xsze] = node
                end
            end

            if ispositive(xsze)
                tag = mdfill!(xset, xsze, yset, ysze, pe, len, elen, iw,
                    superwght, degree, marker, tag)
            end
        end

        # add the nodes in `yset` to the reachable set
        for k in oneto(ysze)
            node = yset[k]
            supersize[node] = -supersize[node]
            rchcnt -= supersize[node]
            rchsze += one(V); rchset[rchsze] = node
        end
    end

    # merge the uneliminated neighbors of `mdnode` into the reachable set,
    # one at a time: each is an element of size two, so `xset` is the whole
    # reachable set
    @inbounds for i in nstart:nstop
        node = iw[i]

        if !isnegative(degnext[node]) && ispositive(supersize[node])
            if ispositive(rchsze)
                yset[one(V)] = node
                tag = mdfill!(rchset, rchsze, yset, one(V), pe, len, elen,
                    iw, superwght, degree, marker, tag)
            end

            supersize[node] = -supersize[node]
            rchcnt -= supersize[node]
            rchsze += one(V); rchset[rchsze] = node
        end
    end

    # element absorption: the elements adjacent to `mdnode` are absorbed
    # into the new element, and the list of `mdnode` is released
    @inbounds for i in estart:estop
        len[iw[i]] = zero(V)
    end

    @inbounds len[mdnode] = zero(V); elen[mdnode] = zero(V)

    # aggressive absorption: for each element `elmnt` adjacent to a node in
    # the reachable set, compute `marker[elmnt] - etag`, the number of nodes
    # of `elmnt` outside the reachable set. If it is 0, `elmnt` is absorbed.
    # This is only done for large reachable sets: on meshes it rarely finds
    # anything.
    absorb = rchsze >= convert(V, 64); etag = zero(I)

    if absorb
        tag += one(I); etag = tag; dmax = zero(V)

        @inbounds for k in oneto(rchsze)
            rnode = rchset[k]
            rsize = convert(I, -supersize[rnode])
            jstart = pe[rnode]
            jstop = jstart + convert(E, elen[rnode]) - one(E)

            for j in jstart:jstop
                elmnt = iw[j]

                if ispositive(len[elmnt])
                    if marker[elmnt] < etag
                        marker[elmnt] = etag + convert(I, degree[elmnt])
                        dmax = max(dmax, degree[elmnt])
                    end

                    marker[elmnt] -= rsize
                end
            end
        end

        # all the markers set above are below the new `tag`
        tag = etag + convert(I, dmax) + one(I)
    end

    # for each node in the reachable set, do the following...
    # (`nrch` counts the nodes that stay in the reachable set)
    hashmask = min(nextpow(2, max(twice(rchsze), one(V))), length(hashhead)) - 1
    nrch = zero(V)

    @inbounds for k in oneto(rchsze)
        rnode = rchset[k]

        # if `rnode` is in the degree list structure...
        pvnode = degprev[rnode]

        if !iszero(pvnode)
            # then remove `rnode` from the structure
            nxnode = degnext[rnode]

            if ispositive(nxnode)
                degprev[nxnode] = pvnode
            end

            if ispositive(pvnode)
                degnext[pvnode] = nxnode
            else
                deghead[-pvnode] = nxnode
            end
        end

        # purge inactive quotient neighbors of `rnode`: absorbed elements,
        # and eliminated nodes, merged nodes, and nodes in the reachable set
        # (they are now reachable through the new element). The elements of
        # `rnode` are `iw[p1:p2]` and its neighbors are `iw[p3:p4]`; `pn` is
        # the next free position, and `pv` the position of the first
        # neighbor after the purging.
        p1 = pe[rnode]
        p2 = p1 + convert(E, elen[rnode]) - one(E)
        p3 = p2 + one(E)
        p4 = p1 + convert(E, len[rnode]) - one(E)
        pn = p1; hash = mdnode % UInt

        for p in p1:p2
            elmnt = iw[p]

            if ispositive(len[elmnt])
                if absorb && marker[elmnt] == etag
                    len[elmnt] = zero(V)
                else
                    iw[pn] = elmnt; pn += one(E); hash += elmnt % UInt
                end
            end
        end

        pv = pn

        for p in p3:p4
            node = iw[p]

            if !isnegative(degnext[node]) && ispositive(supersize[node])
                iw[pn] = node; pn += one(E); hash += node % UInt
            end
        end

        if mass && pn == p1
            # if no active neighbor after the purging, then merge `rnode`
            # with `mdnode` (mass elimination)
            rchcnt += supersize[rnode]
            supersize[mdnode] -= supersize[rnode]; supersize[rnode] = zero(V)
            superwght[mdnode] += superwght[rnode]; superwght[rnode] = zero(V)
            degnext[rnode] = -mdnode
            len[rnode] = zero(V); elen[rnode] = zero(V)
        else
            # else place the new element `mdnode` at the front of the list:
            # move the first neighbor to the end, and the first element to
            # the end of the element list (`rnode` lost at least one entry,
            # so there is room)
            iw[pn] = iw[pv]; iw[pv] = iw[p1]; iw[p1] = mdnode
            elen[rnode] = convert(V, pv - p1) + one(V)
            len[rnode] = convert(V, pn - p1) + one(V)

            # and place `rnode` in a hash bucket to detect indistinguishable
            # nodes
            hash = ((hash * 0x9e3779b97f4a7c15) >>> 32) & hashmask + 1
            degprev[rnode] = convert(V, hash)
            degnext[rnode] = hashhead[hash]
            hashhead[hash] = rnode
            nrch += one(V); rchset[nrch] = rnode
        end
    end

    rchsze = nrch

    # store the reachable set as the list of the new element `mdnode`
    # (compress `iw` if necessary)
    if pfree + convert(E, rchsze) - one(E) > iwlen
        pfree = mdcompress!(neqns, pe, len, iw, pfree)
        @assert pfree + convert(E, rchsze) - one(E) <= iwlen
    end

    @inbounds pe[mdnode] = pfree

    @inbounds for k in oneto(rchsze)
        iw[pfree] = rchset[k]; pfree += one(E)
    end

    @inbounds len[mdnode] = rchsze
    @inbounds degree[mdnode] = rchcnt
    return rchsze, pfree, tag
end

"""
    mdfill!(xset, xsze, yset, ysze, pe, len, elen, iw, superwght, degree,
        marker, tag)

This routine adds the fill edges between the nodes of `xset` and the nodes
of `yset` to the degrees of their endpoints. Two nodes are joined by a fill
edge if they are not adjacent in the quotient graph: neither is a neighbor
of the other, and they belong to no common element. For each node in the
smaller set, its quotient neighbors are marked, and the element lists of
the nodes in the other set are scanned.

input parameters:

  - `(xset, xsze)`, `(yset, ysze)`: the two sets of nodes
  - `(pe, len, elen, iw)`: quotient graph
  - `superwght`: the weight of the supernodes
  - `tag`: tag value

updated parameters:

  - `degree`: degrees of the nodes in `xset` and `yset`
  - `marker`: marker vector
"""
function mdfill!(
        xset::AbstractVector{V},
        xsze::V,
        yset::AbstractVector{V},
        ysze::V,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        marker::AbstractVector{I},
        tag::I,
    ) where {I, V, E}
    if xsze <= ysze
        sset = xset; ssze = xsze; tset = yset; tsze = ysze
    else
        sset = yset; ssze = ysze; tset = xset; tsze = xsze
    end

    # for each node `snode` in the smaller set, do the following...
    @inbounds for k in oneto(ssze)
        snode = sset[k]
        tag += one(I)

        # mark the quotient neighbors (elements and nodes) of `snode`
        pstart = pe[snode]
        pstop = pstart + convert(E, len[snode]) - one(E)

        for p in pstart:pstop
            marker[iw[p]] = tag
        end

        swght = superwght[snode]; sdeg = zero(V)

        # for each node `tnode` in the other set, do the following...
        for l in oneto(tsze)
            tnode = tset[l]

            # if `tnode` is a neighbor of `snode`, they are adjacent
            marker[tnode] == tag && continue

            # if `tnode` belongs to an element of `snode`, they are adjacent
            adjacent = false
            qstart = pe[tnode]
            qstop = qstart + convert(E, elen[tnode]) - one(E)

            for q in qstart:qstop
                if marker[iw[q]] == tag
                    adjacent = true
                    break
                end
            end

            # otherwise, `snode` and `tnode` are joined by a fill edge
            if !adjacent
                sdeg += superwght[tnode]
                degree[tnode] += swght
            end
        end

        degree[snode] += sdeg
    end

    return tag
end

"""
    mdupdate!(mdnode, rchsze, rchset, pe, len, elen, iw, supersize,
        superwght, degree, deghead, degnext, degprev, hashhead, marker, tag,
        mindeg, external)

This routine updates the degree structure after the elimination of
`mdnode`. Indistinguishable nodes in the reachable set are detected by
hashing their quotient neighbors, and merged into supernodes. The weight of
`mdnode` is removed from the degrees of the remaining nodes, which are
returned to the degree lists.

input parameters:

  - `mdnode`: the eliminated node
  - `(rchset, rchsze)`: the reachable set of `mdnode`
  - `(pe, len, elen, iw)`: quotient graph
  - `external`: key the degree lists by external degree instead of degree
  - `tag`: tag value

updated parameters:

  - `mindeg`: new minimum key after degree update
  - `(supersize, superwght, degree)`: supernodes and their degrees
  - `(deghead, degnext, degprev)`: degree lists
  - `hashhead`: hash buckets (empty on output)
  - `marker`: marker vector
"""
function mdupdate!(
        mdnode::V,
        rchsze::V,
        rchset::AbstractVector{V},
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
        hashhead::AbstractVector{V},
        marker::AbstractVector{I},
        tag::I,
        mindeg::V,
        external::Bool,
    ) where {I, V, E}
    # for each hash bucket of the reachable set, do the following...
    @inbounds for k in oneto(rchsze)
        hash = degprev[rchset[k]]
        inode = hashhead[hash]
        iszero(inode) && continue
        hashhead[hash] = zero(V)

        # for each node `inode` in the bucket, do the following...
        while !iszero(inode) && !iszero(degnext[inode])
            # mark the quotient neighbors of `inode`
            tag += one(I)
            pstart = pe[inode]
            pstop = pstart + convert(E, len[inode]) - one(E)

            for p in pstart:pstop
                marker[iw[p]] = tag
            end

            # for each node `jnode` after `inode` in the bucket, do the
            # following...
            jlast = inode; jnode = degnext[inode]

            while !iszero(jnode)
                jnext = degnext[jnode]
                same = len[jnode] == len[inode] && elen[jnode] == elen[inode]

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

                if same
                    # `jnode` is indistinguishable from `inode`: merge them
                    # into a new supernode, and remove `jnode` from the bucket
                    supersize[inode] += supersize[jnode]; supersize[jnode] = zero(V)
                    degree[inode] -= superwght[jnode]
                    superwght[inode] += superwght[jnode]; superwght[jnode] = zero(V)
                    degnext[jnode] = -inode
                    len[jnode] = zero(V); elen[jnode] = zero(V)
                    degnext[jlast] = jnext
                else
                    jlast = jnode
                end

                jnode = jnext
            end

            inode = degnext[inode]
        end
    end

    # for each principal node in the reachable set, do the following...
    @inbounds swght = superwght[mdnode]

    @inbounds for k in oneto(rchsze)
        rnode = rchset[k]

        if isnegative(supersize[rnode])
            supersize[rnode] = -supersize[rnode]

            # remove `mdnode` from the degree of `rnode`, and return `rnode`
            # to the degree structure
            degree[rnode] -= swght

            if external
                deg = degree[rnode] + one(V)
            else
                deg = degree[rnode] + superwght[rnode]
            end

            mindeg = mdinsert!(deg, mindeg, rnode, deghead, degnext, degprev)
        end
    end

    return mindeg, tag
end

"""
    mdinsert!(deg, mindeg, node, deghead, degnext, degprev)

This routine inserts `node` into the degree list `deg`, and returns the new
minimum key.
"""
function mdinsert!(
        deg::V,
        mindeg::V,
        node::V,
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
    ) where {V}
    @inbounds firstnode = deghead[deg]
    @inbounds deghead[deg] = node
    @inbounds degnext[node] = firstnode
    @inbounds degprev[node] = -deg

    if ispositive(firstnode)
        @inbounds degprev[firstnode] = node
    end

    return min(deg, mindeg)
end

"""
    mdcompress!(neqns, pe, len, iw, pfree)

This routine compresses the quotient graph storage `iw`, releasing the
space used by absorbed elements, merged nodes, and purged entries. Returns
the new value of `pfree`.

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
    # entry in `pe[node]`
    @inbounds for node in oneto(neqns)
        if ispositive(len[node])
            p = pe[node]
            pe[node] = convert(E, iw[p])
            iw[p] = -node
        end
    end

    # move the lists to the front of `iw`, in order
    psrc = pdst = one(E); pend = pfree - one(E)

    @inbounds while psrc <= pend
        node = -iw[psrc]; psrc += one(E)

        if ispositive(node)
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

# debugging: recompute the degree of every uneliminated principal node
# from the quotient graph, and compare it to `degree`
function mdcheck(
        neqns::V,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        supersize::AbstractVector{V},
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        degnext::AbstractVector{V},
    ) where {V, E}
    marker = zeros(V, neqns)
    isactive(node) = !isnegative(degnext[node]) && ispositive(supersize[node])

    for node in oneto(neqns)
        isactive(node) || continue
        deg = zero(V); marker[node] = node

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

        @assert deg == degree[node]
    end

    return
end

end
