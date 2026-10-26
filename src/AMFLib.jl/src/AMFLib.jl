#= A collection of routines to find an approximate minimum fill ordering.
   The algorithm is the exact minimum external degree algorithm of MDLib,
   with the degree replaced by an estimate of the fill. Everything else is
   kept: the quotient graph, element absorption, the detection of
   indistinguishable nodes, mass elimination, false twins, the dense nodes,
   and the degrees kept as lower bounds and made exact only for the nodes of
   minimum key, in batches that share their most recent element.
   The key of a supernode of weight w and external degree d is
       (d^2 - l^2 - disc) / (w + 1),
   where l is the weight of its newest clique outside it: the element formed
   when it was last in a reachable set, with the dense neighbors of that
   pivot. Its elimination makes the neighborhood a clique, and the pairs in
   the newest clique are already adjacent, so d^2 - l^2 bounds twice the
   fill from above. This is the approximate minimum mean local fill of
   Rothberg and Eisenstat, with w + 1 in place of w. The key grows with d,
   so a key computed from a lower bound on the degree is a lower bound on
   the key, and the degrees are made exact only as in MD.
   The nodes outside a reachable set keep their keys, but the elimination
   makes some pairs of their neighbors adjacent. With `discount`, each such
   node loses those pairs (`disc`): the pairs between its parts in the
   reachable set, which are the part of its newest element and each of its
   uneliminated neighbors there. It keeps the largest such discount since it
   was last in a reachable set, so its key remains an upper bound.
   For information on the minimum degree algorithm see the articles:
   The evolution of the minimum degree algorithm by Alan George and
   Joseph Liu, SIAM Rev. 31 pp. 1 - 19, 1989
   A fast minimum degree algorithm and matching lower bound by Robert
   Cummings, Matthew Fahrbach, and Animesh Fatehpuria, SODA 2021
   For information on the approximate minimum fill algorithm see the article:
   Node selection strategies for bottom-up sparse matrix ordering by Edward
   Rothberg and Stanley Eisenstat, SIAM J. Matrix Anal. Appl. 19 pp. 682 -
   695, 1998
=#

module AMFLib

using Base: oneto
using FillArrays
using ..Utilities
using ..MDLib: mdndense, mdselect!, mdclasses!, mdremove!, mdinsert!, mdprefetch,
    mdmix, mdsetbit!, mdcount, mdweight, mdsame, mddenseinit!, mdlose!, mddense!, mdreinsert!,
    mdcompress!, mdnumber!

export amf

"""
    amf(neqns[, vwght], xadj, adjncy; discount=true, dense=10.0, mass=true, twin=true, compress=true)

Compute an approximate minimum fill permutation of a graph with n vertices.
Returns the permutation `perm` and the inverse permutation `invp`.

  - `neqns`: the number of vertices
  - `vwght`: vertex weights, finite and positive; a vertex of weight w stands
    for indistinguishable vertices of total weight w (default 1). The
    ordering does not depend on the unit in which they are measured.
  - `xadj`, `adjncy`: the graph; the neighbors of `v` are
    `adjncy[xadj[v]:xadj[v + 1] - 1]`, and self loops are ignored
  - `discount`: lower the keys of the nodes next to each new element by the
    fill that it removes
  - `dense`: vertices with more than `max(16, dense * sqrt(neqns))` neighbors
    are handled as dense (a negative value disables this)
  - `mass`: mass elimination
  - `twin`: false-twin detection
  - `compress`: merge the vertices with the same closed neighborhood first
"""
function amf(neqns::V, xadj::AbstractVector, adjncy::AbstractVector{V}; kwargs...) where {V}
    vwght = Ones{V}(neqns)
    return amf(neqns, vwght, xadj, adjncy; kwargs...)
end

function amf(neqns::V, vwght::AbstractVector, xadj::AbstractVector, adjncy::AbstractVector{V}; kwargs...) where {V}
    @assert neqns <= length(vwght)

    # the weights as integers, `res` of them to the smallest weight (see
    # `amfweights`)
    wght, res = amfweights(neqns, vwght)
    total = 0

    @inbounds for node in oneto(neqns)
        total += convert(Int, wght[node])
    end

    # the working arrays use 32-bit integers when they suffice
    @inbounds nnz = convert(Int, xadj[neqns + 1]) - 1

    if neqns < typemax(Int32) ÷ 8 && total < typemax(Int32) ÷ 4 && nnz < typemax(Int32)
        index = amf(Int32, neqns, total, res, wght, xadj, adjncy; kwargs...)
    else
        index = amf(Int64, neqns, total, res, wght, xadj, adjncy; kwargs...)
    end

    invp = Vector{V}(undef, neqns)
    perm = Vector{V}(undef, neqns)

    @inbounds for node in oneto(neqns)
        invp[node] = convert(V, index[node])
        perm[invp[node]] = node
    end

    return perm, invp
end

# This routine computes the ordering with working arrays of integer type `V`;
# `total` is the sum of the integer weights `vwght`, and `res` the number of
# them to the smallest weight. The other parameters are those of `amf`.
function amf(::Type{V}, neqns::Integer, total::Integer, res::Float64, vwght::AbstractVector, xadj::AbstractVector, adjncy::AbstractVector; discount::Bool = true, dense::Real = 10.0, mass::Bool = true, twin::Bool = true, compress::Bool = true) where {V}
    @assert neqns < length(xadj)
    n = convert(V, neqns)
    @inbounds nnz = convert(Int, xadj[neqns + 1]) - 1

    # the lists never take more than `nnz` entries of `iw`, except while a
    # dense node rejoins, which adds fewer than `2 neqns` (see `mdreinsert!`).
    # The extra `max(nnz ÷ 5, neqns) + neqns` covers that, and leaves room
    # between compressions. Positions in `iw` are 32-bit integers when they
    # suffice, whatever the type of `xadj`
    iwlen = nnz + max(nnz ÷ 5, convert(Int, neqns)) + convert(Int, neqns)

    # 32-bit positions (`E`) when they suffice, and 32-bit tags (`I`) when
    # they leave room for many steps between resets
    if iwlen < typemax(Int32) && n < typemax(Int32) ÷ 64
        index = amf(V, Int32, Int32, n, convert(V, total), res, iwlen, vwght, xadj, adjncy, discount, dense, mass, twin, compress)
    elseif iwlen < typemax(Int32)
        index = amf(V, Int32, Int64, n, convert(V, total), res, iwlen, vwght, xadj, adjncy, discount, dense, mass, twin, compress)
    elseif n < typemax(Int32) ÷ 64
        index = amf(V, Int64, Int32, n, convert(V, total), res, iwlen, vwght, xadj, adjncy, discount, dense, mass, twin, compress)
    else
        index = amf(V, Int64, Int64, n, convert(V, total), res, iwlen, vwght, xadj, adjncy, discount, dense, mass, twin, compress)
    end

    return index
end

# This routine allocates the working arrays, with vertices, weights and
# degrees of type `V`, positions in `iw` of type `E`, and tags of type `I`,
# and computes the ordering.
#
# The keys are kept in the degree lists of MD, by bucket. There are `nbuck`
# buckets: a key `k` below `nfine` (in units of the smallest weight squared)
# goes to bucket `floor(k) + 1`, and a larger one to a bucket evenly spaced
# in `sqrt(k)`, `scale` buckets per unit of `sqrt(k)` from `sfine = sqrt(nfine)`
# up to the largest key.
function amf(::Type{V}, ::Type{E}, ::Type{I}, n::V, total::V, res::Float64, iwlen::Int, vwght::AbstractVector, xadj::AbstractVector, adjncy::AbstractVector, discount::Bool, dense::Real, mass::Bool, twin::Bool, compress::Bool) where {V, E, I}
    nbuck = max(n, one(V))
    nfine = convert(Float64, max(nbuck ÷ convert(V, 2), one(V)))
    sfine = sqrt(nfine)

    # the square root of a key is at most the total weight, in units of the
    # smallest weight
    smax = convert(Float64, total) / res

    if smax > sfine
        scale = (convert(Float64, nbuck) - nfine - 1.0) / (smax - sfine)
    else
        scale = 0.0
    end

    marker = FVector{I}(undef, n)
    superwght = FVector{V}(undef, n)
    degree = FVector{V}(undef, n)
    deghead = FVector{V}(undef, 2 * (nbuck + 1))
    degnext = FVector{V}(undef, n)
    degprev = FVector{V}(undef, n)
    wext = FVector{V}(undef, n)
    extlen = FVector{V}(undef, n)
    pe = FVector{E}(undef, n)
    len = FVector{V}(undef, n)
    elen = FVector{V}(undef, n)
    iw = FVector{V}(undef, iwlen)
    heads = FVector{UInt64}(undef, cld(iwlen, 64))
    iwlen = convert(E, iwlen)

    # the keys
    fkey = FVector{Float64}(undef, n)
    lpw = FVector{V}(undef, n)
    disc = FVector{Float64}(undef, n)
    dexact = FVector{Bool}(undef, n)

    # the discounts (`vsum`, `enew` and `vlist` are needed only with
    # `discount`)
    if discount
        dlen = n
    else
        dlen = zero(V)
    end

    vsum = FVector{Float64}(undef, twice(convert(Int, dlen)))
    enew = FVector{V}(undef, dlen)
    vlist = FVector{V}(undef, dlen)

    @inbounds for i in eachindex(vsum)
        vsum[i] = 0.0
    end

    @inbounds for node in oneto(dlen)
        enew[node] = zero(V)
    end

    # select the dense nodes (`dindex` and `dwght` are needed only if there
    # are any)
    ndense = mdndense(n, xadj, dense); slen = zero(V)

    if ispositive(ndense)
        slen = n
    end

    dindex = FVector{V}(undef, slen)
    ndense = mdselect!(dindex, n, xadj, adjncy, dense, ndense)
    nwords = cld(convert(Int, ndense), 64)
    dwght = FVector{V}(undef, slen)
    dnode = FVector{V}(undef, ndense)
    ddeg = FVector{V}(undef, ndense)
    dmask = FVector{UInt64}(undef, nwords * convert(Int, n))

    amf_impl!(marker, superwght, degree, deghead, degnext, degprev,
        wext, dwght, extlen, pe, len,
        elen, iw, iwlen, heads, total, n, mass,
        twin, compress, vwght, xadj,
        adjncy, ndense, nwords, dindex, dnode, ddeg, dmask,
        fkey, lpw, disc, dexact, nbuck, nfine, sfine, scale, res,
        discount, vsum, enew, vlist)

    return degnext
end

# This routine copies the adjacency structure into `iw`, and computes the
# ordering (`amfmain!`).
function amf_impl!(
        marker::AbstractVector{I},
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        wext::AbstractVector{V},
        dwght::AbstractVector{V},
        extlen::AbstractVector{V},
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        iwlen::E,
        heads::AbstractVector{UInt64},
        total::V,
        neqns::V,
        mass::Bool,
        twin::Bool,
        compress::Bool,
        vwght::AbstractVector,
        xadj::AbstractVector,
        adjncy::AbstractVector,
        ndense::V,
        nwords::Int,
        dindex::AbstractVector{V},
        dnode::AbstractVector{V},
        ddeg::AbstractVector{V},
        dmask::AbstractVector{UInt64},
        fkey::AbstractVector{Float64},
        lpw::AbstractVector{V},
        disc::AbstractVector{Float64},
        dexact::AbstractVector{Bool},
        nbuck::V,
        nfine::Float64,
        sfine::Float64,
        scale::Float64,
        res::Float64,
        discount::Bool,
        vsum::AbstractVector{Float64},
        enew::AbstractVector{V},
        vlist::AbstractVector{V},
    ) where {I, V, E}
    @assert iwlen <= length(iw)

    # copy the adjacency structure into `iw`, dropping self loops, and keep
    # the sum of the closed neighborhood of each node in `wext` (-1 for a
    # dense node)
    pfree = one(E); nmerge = zero(V)

    if iszero(ndense)
        @inbounds for node in oneto(neqns)
            pe[node] = pfree; deg = zero(V); hash = convert(UInt64, node)
            istart = xadj[node]
            istop = xadj[node + one(V)] - one(istart)

            for i in istart:istop
                neighbor = convert(V, adjncy[i])

                if neighbor != node
                    iw[pfree] = neighbor; pfree += one(E)
                    deg += trunc(V, vwght[neighbor])
                    hash += convert(UInt64, neighbor)
                end
            end

            len[node] = convert(V, pfree - pe[node]); degree[node] = deg
            wext[node] = convert(V, hash & convert(UInt64, typemax(V)))
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

            pe[node] = pfree; deg = zero(V); hash = convert(UInt64, node)
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
                        deg += trunc(V, vwght[neighbor])
                        hash += convert(UInt64, neighbor)
                    end
                end
            end

            len[node] = convert(V, pfree - pe[node]); degree[node] = deg

            # dense nodes are not merged
            if ispositive(j)
                wext[node] = -one(V)
            else
                wext[node] = convert(V, hash & convert(UInt64, typemax(V)))
            end
        end
    end

    # with `compress`, find the classes of nodes with the same closed
    # neighborhood: `extlen[node]` is the first node of the class of `node`,
    # or 0 if `node` is first. They are merged before the first elimination
    # (`amfmain!`)
    if compress
        nmerge = mdclasses!(neqns, pe, len, iw, wext, extlen, marker, deghead,
            degprev, elen, superwght, degnext, nwords, dmask)
    end

    # the lists of the first nodes of the classes keep only the first nodes
    # of the classes (the other nodes of a class adjacent to `node` are
    # adjacent to the first one, which `node`'s list holds too), and are
    # packed at the front of `iw` (the writes trail the reads)
    if ispositive(nmerge)
        p = one(E)

        @inbounds for node in oneto(neqns)
            if iszero(extlen[node])
                pstart = pe[node]; pstop = pstart + convert(E, len[node]) - one(E)
                pe[node] = p

                # branch-free: every entry is written, and the next one
                # overwrites it unless it is kept
                for q in pstart:pstop
                    neighbor = iw[q]
                    iw[p] = neighbor; p += convert(E, iszero(extlen[neighbor]))
                end

                len[node] = convert(V, p - pe[node])
            end
        end

        pfree = p
    end

    amfmain!(marker, superwght, degree, deghead, degnext, degprev,
        wext, dwght, extlen, pe, len,
        elen, iw, iwlen, heads, pfree, total, neqns, mass, twin,
        ispositive(nmerge), vwght, ndense, nwords, dindex, dnode, ddeg, dmask,
        fkey, lpw, disc, dexact, nbuck, nfine, sfine, scale, res,
        discount, vsum, enew, vlist)

    return degnext
end

# This routine implements the approximate minimum fill algorithm on the
# minimum external degree algorithm of MDLib. It makes use of the implicit
# representation of elimination graphs by quotient graphs, and the notion of
# indistinguishable nodes.
#
# The external degree of a node is kept as a lower bound, which is exact for
# some nodes. The key of a node is an upper bound on twice its fill over its
# weight plus 1 (see `amfkey`), computed from its bound, and it grows with
# the bound: it is a lower bound on the key of the node unless the node is
# exact. The keys are kept in buckets (see `amfbucket`). The algorithm
# repeats the following. In a fine bucket, the keys differ by less than 1;
# in a coarse one, by more. If the minimum bucket is fine and holds an exact
# node, or is coarse and its smallest key is that of an exact node, that
# node is eliminated (`amfelim!`, `amfupdate!`), and the nodes of its
# reachable set get new bounds and keys. Otherwise every node in the bucket
# that is not exact is recomputed: they are grouped by most recent element,
# and each group is recomputed at once (`amfrecompute!`).
#
# A node of weight w stands for w indistinguishable nodes, so its external
# degree is the weight of its neighbors outside its supernode.
#
# input parameters:
#
#   - `neqns`: number of equations
#   - `total`: sum of the vertex weights
#   - `vwght`: vertex weights (integers)
#   - `mass`: mass elimination. Each node of the reachable set whose only
#     neighbors are those of the new element (it lies in no other element,
#     has no other neighbors, and has the same dense neighbors as the pivot)
#     is eliminated with the pivot.
#   - `twin`: false-twin elimination. Each node outside the reachable set
#     whose neighbors are exactly the reachable set of the pivot, which has
#     never been in a reachable set, and whose dense neighbors are those of
#     the pivot, is eliminated right after the pivot's supernode (see
#     `amfelim!`).
#   - `merged`: merge the classes in `extlen` first. On input, `extlen[node]`
#     is the first node of the class of `node` (nodes with the same closed
#     neighborhood), or 0 if `node` is first (see `mdclasses!`)
#   - `iwlen`: length of `iw`
#   - `(ndense, nwords, dindex, dnode, ddeg, dmask)`: the dense nodes (see
#     `mdselect!`). Their lists are empty: their adjacency, and every
#     adjacency to them, is stored in the masks. They are kept out of the
#     quotient graph until one of them has the minimum key, with
#     indistinguishable nodes required to have the same dense neighbors.
#   - `(nbuck, nfine, sfine, scale)`: the buckets (see `amf`)
#   - `res`: the number of integer weights to the smallest weight
#   - `discount`: discount the keys of the nodes adjacent to each reachable
#     set (see `amfdiscount!`)
#
# updated parameters:
#
#   - `(pe, len, elen, iw)`: quotient graph. On input, `(pe, len, iw)` holds
#     the adjacency structure (with `merged`, the lists of the first nodes of
#     the classes, which hold only first nodes), and `degree` the weight of
#     the neighbors of each node. For an uneliminated node, the list
#     `iw[pe[node]:pe[node] + len[node] - 1]` holds `elen[node]` elements,
#     sorted by elimination order, followed by its uneliminated neighbors;
#     for an eliminated node (element), it holds the nodes in the element.
#     An element is absorbed, and the list of a merged node released, by
#     setting its length to 0. Destroyed on output.
#   - `pfree`: the quotient graph is stored in `iw[1:pfree - 1]`
#
# output parameters:
#
#   - `degnext`: the approximate minimum fill ordering (repurposed from
#     degree list links)
#
# working arrays:
#
#   - `deghead`: `deghead[bucket]` points to the first exact node in a
#     bucket, and `deghead[bucket + nbuck + 1]` to the first node that is not
#     exact, or 0 if there are no such nodes
#   - `degnext`: during algorithm, points to the next node in the degree list
#     (or hash bucket) associated with node, or stores negative values for
#     eliminated/merged nodes (`-(neqns + rank)` for an eliminated principal
#     node, the `rank`-th supernode eliminated, and -parent for a merged
#     node). During a visit, the next node in a group. After algorithm,
#     stores the final ordering.
#   - `degprev`: points to the previous node in a degree list associated with
#     node, or the negative of the index in `deghead` of its list (if node
#     was the first in it), or 0 if the node is not in the degree lists.
#     While indistinguishable nodes are detected, the hash bucket of node.
#     During a visit, the first node of the group of an element.
#   - `degree`: the bound on the external degree of each uneliminated
#     supernode; for an element, the weight of the nodes in it
#   - `superwght`: the weight of the supernodes. It is positive exactly for
#     the uneliminated supernodes outside the reachable set being formed: it
#     is negated for the nodes of that set and for the pivot, and 0 for
#     merged and eliminated nodes.
#   - `wext`: for an element, 0 if it is absorbed; otherwise less than `wflg`
#     if it has not been met at the current elimination, and `wflg` plus the
#     weight of its nodes outside the reachable set if it has. For a node of
#     the reachable set, the first term of its bound (see `amfelim!`).
#   - `dwght`: the weight of the dense neighbors of each node (empty if there
#     are no dense nodes)
#   - hash buckets, used to detect indistinguishable nodes, share `deghead`
#     with the degree lists. They use its last entries, whose lists (not
#     exact, with the largest keys) are almost always empty: the first node
#     of the bucket at `deghead[b]` is `-deghead[b]` if no degree list starts
#     there, and otherwise `degprev[deghead[b]]`, which is restored to `-b`
#     once the bucket is processed. The hash of a node of the reachable set
#     is kept in `marker` (see `amfelim!`).
#   - `extlen`: for an element, the number of its nodes outside the element
#     being recomputed, which come first in its list (see `amfrecompute!`).
#     For a node of the reachable set, `extlen` is the second term of its
#     bound, and then its upper bound (see `amfupdate!`).
#   - `marker`: temporary marker vector, used with increasing tags. A node
#     of the reachable set holds its hash there, as a negative number, until
#     the indistinguishable nodes are merged
#   - `heads`: the first entries of the lists, while `iw` is compressed (see
#     `mdcompress!`)
#   - `fkey`: the key of each node in the degree lists
#   - `lpw`: the weight of the newest clique of each node outside it (the
#     new element and the dense neighbors of the pivot, when the node was
#     last in a reachable set), or 0
#   - `disc`: the discount of each node (see `amfdiscount!`)
#   - `dexact`: whether the bound of each node is its external degree
#   - `(vsum, vlist)`: with `discount`, the nodes adjacent to the reachable
#     set, and their neighbors in it (see `amfelim!`)
#   - `enew`: with `discount`, the newest element of each node, or 0
function amfmain!(
        marker::AbstractVector{I},
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        wext::AbstractVector{V},
        dwght::AbstractVector{V},
        extlen::AbstractVector{V},
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        iwlen::E,
        heads::AbstractVector{UInt64},
        pfree::E,
        total::V,
        neqns::V,
        mass::Bool,
        twin::Bool,
        merged::Bool,
        vwght::AbstractVector,
        ndense::V,
        nwords::Int,
        dindex::AbstractVector{V},
        dnode::AbstractVector{V},
        ddeg::AbstractVector{V},
        dmask::AbstractVector{UInt64},
        fkey::AbstractVector{Float64},
        lpw::AbstractVector{V},
        disc::AbstractVector{Float64},
        dexact::AbstractVector{Bool},
        nbuck::V,
        nfine::Float64,
        sfine::Float64,
        scale::Float64,
        res::Float64,
        discount::Bool,
        vsum::AbstractVector{Float64},
        enew::AbstractVector{V},
        vlist::AbstractVector{V},
    ) where {I, V, E}
    @assert neqns <= length(marker)
    @assert neqns <= length(superwght)
    @assert neqns <= length(degree)
    @assert twice(nbuck + one(V)) <= length(deghead)
    @assert neqns <= length(degnext)
    @assert neqns <= length(degprev)
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
    @assert iszero(ndense) || neqns <= length(dindex)
    @assert nwords * neqns <= length(dmask)
    @assert neqns <= length(fkey)
    @assert neqns <= length(lpw)
    @assert neqns <= length(disc)
    @assert neqns <= length(dexact)
    @assert !discount || twice(convert(Int, neqns)) <= length(vsum)
    @assert !discount || neqns <= length(enew)
    @assert !discount || neqns <= length(vlist)

    # initialization for the approximate minimum fill algorithm.
    @inbounds for node in oneto(neqns)
        marker[node] = zero(I)
        superwght[node] = trunc(V, vwght[node])
        elen[node] = zero(V)
        degnext[node] = zero(V)
        degprev[node] = zero(V)
        wext[node] = one(V)
    end

    @inbounds for bucket in oneto(twice(nbuck + one(V)))
        deghead[bucket] = zero(V)
    end

    # merge the classes of nodes with the same closed neighborhood
    # (`extlen`): each node joins the first node of its class. The degree of
    # the first node counted the other nodes of its class, which are not
    # external to the supernode
    if merged
        @inbounds for node in oneto(neqns)
            r = extlen[node]

            if ispositive(r)
                superwght[r] += superwght[node]; superwght[node] = zero(V)
                degnext[node] = -r; len[node] = zero(V)
            end
        end

        @inbounds for node in oneto(neqns)
            if iszero(extlen[node])
                degree[node] -= superwght[node] - trunc(V, vwght[node])
            end

            extlen[node] = zero(V)
        end
    end

    # - `mindeg` is the current minimum bucket
    # - `rank` counts the eliminated supernodes plus 1, and `wleft` is the
    #   weight of the uneliminated nodes
    # - `tag` is used to facilitate marking nodes; the tags are reset
    #   before they overflow
    # - `wflg` is the flag of `wext`, and `lemax` the largest weight of an
    #   element
    # - `offset` separates the buckets of the lists that are not exact
    mindeg = nbuck + one(V); rank = one(V); wleft = total; tag = zero(I)
    wflg = one(V); lemax = zero(V); offset = nbuck + one(V)

    # - `nbypass` is the number of dense nodes outside the quotient graph
    # - `nmissing` is the number of ordered pairs of such nodes that are
    #   not adjacent
    # - `dmin` is a lower bound on the buckets of the dense nodes, and
    #   `dwmax` their largest weight (dense nodes are never merged)
    nbypass = ndense; nmissing = 0; dmin = zero(V); dwmax = zero(V)

    if ispositive(ndense)
        @inbounds for node in oneto(neqns)
            dwght[node] = zero(V)
        end

        mddenseinit!(ddeg, dmask, nwords, ndense, dnode, neqns, superwght)

        @inbounds for j in oneto(ndense)
            nmissing += convert(Int, ndense) - 1 - mdcount(dmask, nwords, dnode[j])
            dwmax = max(dwmax, superwght[dnode[j]])
        end
    end

    # `degree` holds the weight of the neighbors of each node outside its
    # supernode (summed while the adjacency was copied)
    @inbounds for node in oneto(neqns)
        # dense nodes and merged nodes stay out of the degree lists
        if ispositive(superwght[node]) && (iszero(ndense) || !ispositive(dindex[node]))
            deg = degree[node]

            if ispositive(ndense)
                dwght[node] = mdweight(dmask, nwords, dnode, node, superwght)
                deg += dwght[node]
            end

            degree[node] = deg

            # the node has no newest clique yet: its key is deg^2 / (w + 1)
            lpw[node] = zero(V); disc[node] = 0.0; dexact[node] = true

            if ispositive(deg)
                mindeg = amfinsert!(node, superwght[node], deg, mindeg, fkey, lpw, disc, dexact,
                    deghead, degnext, degprev, offset, nbuck, nfine, sfine, scale, res)
            else
                # eliminate node of degree 0
                degnext[node] = -(neqns + rank); rank += one(V); wleft -= superwght[node]
            end
        end
    end

    # each step uses fewer than `8 neqns + 64` tags: reset them before they
    # overflow
    tagmax = typemax(I) - convert(I, 8) * convert(I, neqns) - convert(I, 64)

    @inbounds while ispositive(wleft)
        # find the pivot `mdnode`. If no node of the minimum bucket can be
        # taken, the nodes there that are not exact are recomputed, and the
        # search starts over
        mdnode = zero(V)

        while iszero(mdnode)
            if tag > tagmax
                for node in oneto(neqns)
                    marker[node] = zero(I)
                end

                tag = zero(I)
            end

            while mindeg <= nbuck && !ispositive(deghead[mindeg]) && !ispositive(deghead[mindeg + offset])
                mindeg += one(V)
            end

            # if a dense node has a smaller key than every node in the
            # buckets, it rejoins the quotient graph and is eliminated.
            # `dmin` is a lower bound on the buckets of the dense nodes:
            # they are scanned only if it is below the minimum bucket
            dj = zero(V)

            if ispositive(nbypass) && !(mindeg <= nbuck && mindeg <= dmin)
                dkey = Inf

                for j in oneto(ndense)
                    dnd = dnode[j]

                    if ispositive(dnd)
                        key = amfkey(ddeg[j], superwght[dnd], zero(V), 0.0, res)

                        if key < dkey
                            dkey = key; dj = j
                        end
                    end
                end

                dmin = amfbucket(dkey, nfine, sfine, scale, nbuck)

                if mindeg <= nbuck && mindeg <= dmin
                    dj = zero(V)
                end
            end

            if ispositive(dj)
                mdnode = dnode[dj]
                nmissing -= 2 * (convert(Int, nbypass) - 1 - mdcount(dmask, nwords, mdnode))
                nbypass -= one(V)
                pfree = mdreinsert!(dj, neqns, pe, len, elen, iw, iwlen, pfree, heads,
                    superwght, nwords, dindex, dnode, dmask, wext)
                degree[mdnode] = ddeg[dj]

                # its neighbors now reach it through their lists
                dwgt = superwght[mdnode]
                pstart = pe[mdnode]
                pstop = pstart + convert(E, len[mdnode]) - one(E)

                for p in pstart:pstop
                    dwght[iw[p]] -= dwgt
                end
            else
                # the bucket `mindeg`. In a fine bucket the keys differ by
                # less than 1, and its first exact node is taken, as in MD.
                # In a coarse bucket the exact node of smallest key is
                # taken, unless a node that is not exact has a smaller key.
                # If no node is taken, the nodes of the bucket that are not
                # exact are recomputed, and the search starts over
                bucket = mindeg; cand = zero(V)

                if convert(Float64, bucket) <= nfine
                    cand = deghead[bucket]
                else
                    kmin = Inf; node = deghead[bucket]

                    while ispositive(node)
                        if fkey[node] < kmin
                            kmin = fkey[node]; cand = node
                        end

                        node = degnext[node]
                    end

                    node = deghead[bucket + offset]

                    while ispositive(node)
                        if fkey[node] < kmin
                            cand = zero(V)
                            break
                        end

                        node = degnext[node]
                    end
                end

                if ispositive(cand)
                    mdnode = cand
                    mdremove!(mdnode, deghead, degnext, degprev, offset)

                    # the new head of the list is likely the next pivot:
                    # ask for its fields and its list while `mdnode` is
                    # eliminated
                    nxnode = deghead[bucket]

                    if ispositive(nxnode)
                        mdprefetch(superwght, nxnode)
                        mdprefetch(degree, nxnode); mdprefetch(len, nxnode)
                        mdprefetch(elen, nxnode); mdprefetch(iw, pe[nxnode])
                    end
                else
                    # recompute every node of the bucket that is not exact
                    mindeg, tag = amfvisit!(bucket, pe, len, elen, iw, superwght,
                        degree, deghead, degnext, degprev, marker,
                        tag, neqns, extlen, dwght, nbypass, offset,
                        fkey, lpw, disc, dexact, nbuck, nfine, sfine, scale, res)
                end
            end
        end

        degprev[mdnode] = zero(V)
        degnext[mdnode] = -(neqns + rank)

        # `mdnode` is the last supernode
        if superwght[mdnode] >= wleft
            break
        end

        # the dense neighbors of `mdnode` lose it
        if ispositive(nbypass)
            # the key of a dense node of degree `d` is at least that of a
            # node of weight `dwmax` and degree `d`
            dlow = mdlose!(ddeg, dmask, nwords, mdnode, superwght[mdnode])
            dS = mdweight(dmask, nwords, dnode, mdnode, superwght)

            if dlow < typemax(V)
                dkey = amfkey(dlow, dwmax, zero(V), 0.0, res)
                dmin = min(dmin, amfbucket(dkey, nfine, sfine, scale, nbuck))
            end
        else
            dS = zero(V)
        end

        # eliminate `mdnode` and perform quotient graph
        # transformation
        rchsze, pfree, delta, swght, wflg, lemax, fsze, fwght, nvlist = amfelim!(mdnode,
            pe, len, elen, iw, iwlen, pfree, heads, neqns, superwght, degree,
            deghead, degnext, degprev, marker, wext, extlen, wflg, lemax,
            dS, offset, twin, rank + one(V),
            nbypass, nwords, dmask, ddeg, discount, vsum, vlist)

        # the nodes in the reachable set, and the dense neighbors of
        # `mdnode`, become adjacent to the dense neighbors of `mdnode`
        if ispositive(nbypass)
            nmissing = mddense!(mdnode, pe[mdnode], rchsze, iw, dwght, superwght,
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
        mindeg, tag, mwght = amfupdate!(mdnode, rchsze, delta, dS, swght, fwght, pe,
            len, elen, iw, superwght, degree, deghead, degnext,
            degprev, wext, marker, tag,
            neqns, extlen, dwght, nwords, dmask, ddeg, nbypass, mindeg,
            mass && iszero(fsze), offset, fkey, lpw, disc, dexact, nbuck,
            nfine, sfine, scale, res, discount, vsum, enew)

        # with `discount`, the nodes adjacent to the new element lose the
        # pairs of their neighbors that it made adjacent
        if discount
            mindeg = amfdiscount!(nvlist, vlist, vsum, enew, superwght,
                degree, wext, wflg, mindeg, deghead, degnext, degprev, offset,
                fkey, lpw, disc, dexact, nbuck, nfine, sfine, scale, res)
        end

        rank += one(V) + fsze; wleft -= swght + fwght + mwght
        superwght[mdnode] = zero(V)

        if ispositive(dS) && (ispositive(fsze) || mass)
            dmin = zero(V)
        end
    end

    mdnumber!(neqns, degnext, degprev, degree)
    return degnext
end

# This routine eliminates the node `mdnode` of minimum key from the
# adjacency structure, which is stored in the quotient graph format. It
# forms the reachable set of `mdnode`, absorbs the elements adjacent to
# `mdnode`, and computes, for every other element adjacent to the reachable
# set, the weight of its nodes outside the reachable set. The elements with
# none are absorbed (aggressive absorption). The lists of the nodes in the
# reachable set are purged, and the terms of their bounds are gathered:
# `wext[u]` is the weight of their uneliminated neighbors plus their largest
# external part, and `extlen[u]` the sum of their other external parts (these
# slots are unused for uneliminated nodes). The nodes are placed in hash
# buckets, and the reachable set becomes the new element `mdnode`.
#
# With `discount`, the purge also lists the nodes outside the reachable set
# that lie in the lists of its nodes (`vlist`), and sums, for each, the
# weights of its neighbors in the reachable set and their squares, node by
# node: `vsum[2 node - 1]` and `vsum[2 node]` (see `amfdiscount!`). The sums
# are 0 outside the step, so a node is listed when its first sum leaves 0.
#
# With `twin`, the false twins of `mdnode` are found first: the nodes
# outside the reachable set whose neighbors are exactly the reachable set.
# Each lies in every list of the reachable set, so only the shortest is
# scanned. A node of it qualifies if it has never been in a reachable set
# (its adjacency is then all in its list, and its degree is exact), its
# degree and dense neighbors are those of `mdnode`, and its list holds no
# node outside the reachable set. Once `mdnode` is eliminated, it is
# simplicial, so it creates no fill. It leaves the quotient graph, numbered
# right after the supernode of
# `mdnode`: its weight is set to 0 before the lists are purged, so it drops
# out of them.
#
# The reachable set is written to `iw` as it is formed, and becomes the list of
# the new element. If `mdnode` is adjacent to no element, its list of
# uneliminated neighbors holds the reachable set, which is formed there, in
# place. Otherwise it is formed at `pfree`, after a compression if the tail of
# `iw` is too short: it holds fewer than `neqns` nodes, and at most
# `degree[mdnode] - dS` (each node weighs at least 1). That fits after a
# compression: outside a rejoin, the lists take at most `nnz` entries, and
# `iw` has at least `nnz + 2 neqns` (see `amf`); a rejoining dense node has no
# element.
#
# input parameters:
#
#   - `mdnode`: node of minimum key
#   - `neqns`: number of equations
#   - `iwlen`: length of `iw`
#   - `dS`: the weight of the dense neighbors of `mdnode`
#   - `offset`: separates the buckets of the lists that are not exact
#   - `twin`: find the false twins of `mdnode`
#   - `fnum`: the rank of the first false twin (the rank after that of
#     `mdnode`)
#   - `(nbypass, nwords, dmask)`: the dense nodes
#   - `discount`: list the nodes adjacent to the reachable set
#
# updated parameters:
#
#   - `(pe, len, elen, iw)`: quotient graph
#   - `pfree`: the quotient graph is stored in `iw[1:pfree - 1]`
#   - `heads`: working array of `mdcompress!`
#   - `ddeg`: the degrees of the dense nodes, which lose the false twins
#   - `superwght`: weight of the supernodes; on output, negative for
#     `mdnode`
#   - `degree`: on output, `degree[mdnode]` is the weight of the nodes in the
#     new element
#   - `(deghead, degnext, degprev)`: degree lists; on output, the
#     nodes in the reachable set are removed from them, and the false twins
#     are numbered in `degnext`
#   - `marker`: on output, the nodes in the reachable set are placed in hash
#     buckets, which share `deghead` with the degree lists (`degprev` and
#     `degnext` hold their buckets and links), and each holds its hash in
#     `marker`, as a negative number, so that it is never taken for a tag
#   - `(wext, wflg, lemax)`: external parts of the elements
#
# output parameters:
#
#   - `rchsze`: size of the reachable set, which is the list of the new
#     element `mdnode`
#   - `(wext, extlen)`: for each node of the reachable set, the terms of its
#     bound
#   - `delta`: the weight of the reachable set, with the dense neighbors of
#     `mdnode`
#   - `swght`: the weight of `mdnode`
#   - `fsze`: the number of false twins of `mdnode` (supernodes)
#   - `fwght`: their weight
#   - `nvlist`: with `discount`, the number of nodes in `vlist`
#   - `(vlist, vsum)`: with `discount`, the nodes adjacent to the reachable
#     set, and their neighbors in it
function amfelim!(
        mdnode::V,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        iwlen::E,
        pfree::E,
        heads::AbstractVector{UInt64},
        neqns::V,
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        marker::AbstractVector{I},
        wext::AbstractVector{V},
        extlen::AbstractVector{V},
        wflg::V,
        lemax::V,
        dS::V,
        offset::V,
        twin::Bool,
        fnum::V,
        nbypass::V,
        nwords::Int,
        dmask::AbstractVector{UInt64},
        ddeg::AbstractVector{V},
        discount::Bool,
        vsum::AbstractVector{Float64},
        vlist::AbstractVector{V},
    ) where {I, V, E}
    # the pivot leaves the quotient graph
    @inbounds swght = superwght[mdnode]; superwght[mdnode] = -swght

    # the reachable set is written to `iw[rstart:rstop]` as it is formed:
    # in place if `mdnode` is adjacent to no element, and otherwise at the
    # end of `iw` (compress it first if the tail is too short)
    inplace = @inbounds iszero(elen[mdnode])

    if inplace
        @inbounds rstart = pe[mdnode]
    else
        @inbounds need = convert(E, min(degree[mdnode] - dS, neqns))

        if pfree + need - one(E) > iwlen
            pfree = mdcompress!(neqns, pe, len, iw, pfree, heads)
            @assert pfree + need - one(E) <= iwlen
        end

        rstart = pfree
    end

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

        # ask for the length and the list of the next element
        nxelmnt = iw[min(i + one(E), estop)]
        mdprefetch(len, nxelmnt); mdprefetch(iw, pe[nxelmnt])

        jstart = pe[elmnt]
        jstop = jstart + convert(E, len[elmnt]) - one(E)

        for j in jstart:jstop
            node = iw[j]; wght = superwght[node]

            if ispositive(wght)
                superwght[node] = -wght
                iw[rstart + convert(E, rchsze)] = node; rchsze += one(V)
                mdremove!(node, deghead, degnext, degprev, offset)

                # the fields and the list of `node` are read only after
                # the reachable set is formed: ask for them now
                mdprefetch(len, node); mdprefetch(pe, node); mdprefetch(elen, node)
                mdprefetch(degree, node); mdprefetch(iw, pe[node])

                if twin
                    rsum += wght; nlen = len[node] - elen[node]

                    if nlen < lbest
                        lbest = nlen; rbest = node
                    end
                end
            end
        end

        # element absorption: `elmnt` is absorbed into the new element
        len[elmnt] = zero(V); wext[elmnt] = zero(V)
    end

    @inbounds for i in nstart:nstop
        node = iw[i]; wght = superwght[node]

        if ispositive(wght)
            # (in place, the write trails the read)
            superwght[node] = -wght
            iw[rstart + convert(E, rchsze)] = node; rchsze += one(V)
            mdremove!(node, deghead, degnext, degprev, offset)
            mdprefetch(len, node); mdprefetch(pe, node); mdprefetch(elen, node)
            mdprefetch(degree, node); mdprefetch(iw, pe[node])

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

    if twin && lbest < typemax(V)
        tdeg = rsum + dS
        @inbounds pstart = pe[rbest] + convert(E, elen[rbest])
        @inbounds pstop = pe[rbest] + convert(E, len[rbest]) - one(E)

        @inbounds for p in pstart:pstop
            node = iw[p]; wght = superwght[node]

            # cheapest tests first: the weight is in hand
            if ispositive(wght) && degree[node] == tdeg && iszero(elen[node]) &&
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

                    if ispositive(nbypass)
                        mdlose!(ddeg, dmask, nwords, node, wght)
                    end

                    degnext[node] = -(neqns + fnum + fsze); fsze += one(V)
                    superwght[node] = zero(V); len[node] = zero(V); fwght += wght
                end
            end
        end
    end

    # the list of `mdnode` is released (its elements were absorbed above):
    # the reachable set takes its place
    rstop = rstart + convert(E, rchsze) - one(E)
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
    @inbounds for r in rstart:rstop
        rnode = iw[r]; rwght = -superwght[rnode]
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

    # with `discount`, `nvlist` is the number of nodes in `vlist`
    nvlist = 0

    # for each node in the reachable set, do the following... The hash
    # buckets are many (32 to 64 per node), so that a node is seldom in a
    # bucket with another
    hashmask = min(nextpow(2, max(32 * convert(Int, rchsze), 1)), prevpow(2, length(deghead))) - 1

    @inbounds for r in rstart:rstop
        rnode = iw[r]

        # purge inactive quotient neighbors of `rnode`: absorbed elements,
        # and eliminated nodes, merged nodes, and nodes in the reachable set
        # (they are now reachable through the new element). The elements of
        # `rnode` are `iw[p1:p2]` and its neighbors are `iw[p3:p4]`; `pn` is
        # the next free position, and `pv` the position of the first
        # neighbor after the purging. Along the way, `elmax` is the largest
        # external part, `elsum` their sum, `nwght` the weight of the
        # neighbors, and `hash` the sum of the entries kept.
        p1 = pe[rnode]
        p2 = p1 + convert(E, elen[rnode]) - one(E)
        p3 = p2 + one(E)
        p4 = p1 + convert(E, len[rnode]) - one(E)
        pn = p1; hash = zero(UInt64)
        elmax = elsum = nwght = zero(V)

        for p in p1:p2
            elmnt = iw[p]; we = wext[elmnt]

            if !iszero(we)
                dext = we - wflg

                if iszero(dext)
                    # aggressive absorption: `elmnt` lies in the reachable set
                    len[elmnt] = zero(V); wext[elmnt] = zero(V)
                else
                    iw[pn] = elmnt; pn += one(E); hash += elmnt % UInt64
                    elmax = max(elmax, dext); elsum += dext
                end
            end
        end

        pv = pn

        if discount
            # each `node` lies outside the reachable set, next to `rnode`:
            # add the weight of `rnode` to its sums, and its square
            rw = convert(Float64, -superwght[rnode]); rw2 = rw * rw

            for p in p3:p4
                node = iw[p]; wght = superwght[node]

                if ispositive(wght)
                    iw[pn] = node; pn += one(E); hash += node % UInt64
                    nwght += wght
                    q = twice(convert(Int, node)); rv = vsum[q - 1]

                    if iszero(rv)
                        nvlist += 1; vlist[nvlist] = node
                    end

                    vsum[q - 1] = rv + rw; vsum[q] += rw2
                end
            end
        else
            for p in p3:p4
                node = iw[p]; wght = superwght[node]

                if ispositive(wght)
                    iw[pn] = node; pn += one(E); hash += node % UInt64
                    nwght += wght
                end
            end
        end

        wext[rnode] = nwght + elmax; extlen[rnode] = elsum - elmax

        # append the new element `mdnode` to the element list, which stays
        # sorted by elimination order: move the first neighbor to the end
        # (`rnode` lost at least one entry, so there is room)
        iw[pn] = iw[pv]; iw[pv] = mdnode
        elen[rnode] = convert(V, pv - p1) + one(V)
        len[rnode] = convert(V, pn - p1) + one(V)

        # and place `rnode` in a hash bucket, by the mixed sum of its list, to
        # detect indistinguishable nodes
        hash = mdmix(hash)
        marker[rnode] = -convert(I, hash & convert(UInt64, typemax(I))) - one(I)
        bucket = convert(V, length(deghead)) - convert(V, (hash >>> 32) & hashmask)
        degprev[rnode] = bucket
        head = deghead[bucket]

        if ispositive(head)
            # a degree list starts at `deghead[bucket]`: its first node
            # holds the bucket
            bnode = degprev[head]

            if ispositive(bnode)
                degnext[rnode] = bnode
            else
                degnext[rnode] = zero(V)
            end

            degprev[head] = rnode
        else
            degnext[rnode] = -head
            deghead[bucket] = -rnode
        end
    end

    # the reachable set is the list of the new element `mdnode`: make its
    # nodes active again
    ewght = zero(V)

    @inbounds for r in rstart:rstop
        rnode = iw[r]; rwght = -superwght[rnode]; superwght[rnode] = rwght
        ewght += rwght
    end

    if !inplace
        pfree = rstop + one(E)
    end

    @inbounds pe[mdnode] = rstart; len[mdnode] = rchsze
    @inbounds degree[mdnode] = ewght; wext[mdnode] = one(V)
    return rchsze, pfree, ewght + dS, swght, wflg, max(lemax, ewght), fsze, fwght, nvlist
end

# This routine updates the degree structure after the elimination of
# `mdnode`. Indistinguishable nodes in the reachable set are detected by
# hashing their quotient neighbors, and merged into supernodes. With mass
# elimination, each supernode whose list holds only `mdnode`, and whose
# dense neighbors are those of `mdnode`, is eliminated with `mdnode`; let
# `m` be their weight (0 otherwise). Each other supernode of the reachable
# set gets the bound
#
#     delta - m - w(u) + wext[u] + (weight of its dense neighbors not adjacent to `mdnode`),
#
# raised to its old bound minus `swght + fwght + m` if it was not formed by
# a merge. It is exact if it equals the upper bound, the same expression
# with `extlen[u]` added. Its newest clique is now the new element with the
# dense neighbors of `mdnode`, of weight `delta - m`, and its discount is 0.
# The supernodes are returned to the degree lists by their keys.
#
# With `discount`, the nodes adjacent to the reachable set counted the
# weights of `inode` and `jnode` apart (see `amfelim!`) when `jnode` is
# merged into `inode`: their pairs are adjacent, so the sums of the squares
# take them in. And the newest element of each supernode is `mdnode`.
#
# input parameters:
#
#   - `mdnode`: the eliminated node
#   - `rchsze`: the size of the reachable set of `mdnode`, which is its list
#   - `delta`: the weight of the reachable set, with the dense neighbors of
#     `mdnode`
#   - `dS`: the weight of the dense neighbors of `mdnode`
#   - `swght`: the weight of `mdnode`
#   - `fwght`: the weight of the false twins of `mdnode` (see `amfelim!`)
#   - `(pe, len, elen, iw)`: quotient graph
#   - `(wext, extlen)`: the terms of the bounds (see `amfelim!`); on output,
#     `extlen[u]` is the upper bound of each node `u` of the reachable set
#   - `mass`: mass elimination
#   - `offset`: separates the buckets of the lists that are not exact
#   - `(nbuck, nfine, sfine, scale, res)`: the buckets and the weights (see
#     `amf`)
#   - `discount`: keep the sums of the nodes adjacent to the reachable set,
#     and the newest elements
#
# updated parameters:
#
#   - `mindeg`: new minimum bucket after degree update
#   - `ddeg`: the degrees of the dense nodes, which lose the nodes eliminated
#     with `mdnode`
#   - `(superwght, degree)`: supernodes and their bounds
#   - `(fkey, lpw, disc, dexact)`: the keys, newest cliques, discounts and
#     exact bounds
#   - `(vsum, enew)`: with `discount`, see `amfelim!` and `amfmain!`
#   - `(deghead, degnext, degprev)`: degree lists
#   - `marker`: the hashes of the nodes of the reachable set (see `amfelim!`),
#     then tags; the hash buckets share `deghead` with the degree lists
#     (empty on output)
#   - `(marker, tag)`: marker vector
@inline function amfupdate!(
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
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        wext::AbstractVector{V},
        marker::AbstractVector{I},
        tag::I,
        neqns::V,
        extlen::AbstractVector{V},
        dwght::AbstractVector{V},
        nwords::Int,
        dmask::AbstractVector{UInt64},
        ddeg::AbstractVector{V},
        nbypass::V,
        mindeg::V,
        mass::Bool,
        offset::V,
        fkey::AbstractVector{Float64},
        lpw::AbstractVector{V},
        disc::AbstractVector{Float64},
        dexact::AbstractVector{Bool},
        nbuck::V,
        nfine::Float64,
        sfine::Float64,
        scale::Float64,
        res::Float64,
        discount::Bool,
        vsum::AbstractVector{Float64},
        enew::AbstractVector{V},
    ) where {I, V, E}
    # the supernodes formed by a merge are marked with `mtag`
    tag += one(I); mtag = tag

    # the reachable set is `iw[rstart:rstop]`
    @inbounds rstart = pe[mdnode]; rstop = rstart + convert(E, rchsze) - one(E)

    # for each hash bucket of the reachable set, do the following...
    @inbounds for r in rstart:rstop
        hash = degprev[iw[r]]
        head = deghead[hash]

        # take the first node of the bucket, and empty it (`inode` stays 0
        # if it is empty, or was taken before)
        inode = zero(V)

        if isnegative(head)
            inode = -head; deghead[hash] = zero(V)
        elseif ispositive(head) && ispositive(degprev[head])
            inode = degprev[head]; degprev[head] = -hash
        end

        # for each node `inode` in the bucket, do the following...
        while !iszero(inode) && !iszero(degnext[inode])
            # find the first node `jnode` after `inode` whose hash value and
            # list lengths match, with `jlast` before it: only then are the
            # lists compared
            # (the hash of `inode` is read once: a merge overwrites it)
            jlast = inode; jnode = degnext[inode]; ihash = marker[inode]

            while !iszero(jnode) && !(marker[jnode] == ihash && len[jnode] == len[inode] && elen[jnode] == elen[inode])
                jlast = jnode; jnode = degnext[jnode]
            end

            if !iszero(jnode)
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
                    same = marker[jnode] == ihash && len[jnode] == len[inode] && elen[jnode] == elen[inode]

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
                        # with `discount`, the neighbors of `inode` outside
                        # the reachable set, which are those of `jnode`,
                        # counted them as two of their parts: they become
                        # one
                        if discount
                            wi = convert(Float64, superwght[inode])
                            wj = convert(Float64, superwght[jnode])

                            for p in (pstart + convert(E, elen[inode])):pstop
                                vsum[twice(convert(Int, iw[p]))] += 2.0 * wi * wj
                            end
                        end

                        # `jnode` is indistinguishable from `inode`: merge
                        # them into a new supernode, and remove `jnode` from
                        # the bucket
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
            end

            inode = degnext[inode]
        end
    end

    # keep the principal nodes of the reachable set. With mass elimination,
    # a node whose list holds only `mdnode`, and whose dense neighbors are
    # those of `mdnode`, is eliminated with `mdnode` instead: with degree, it
    # is indistinguishable from `mdnode`. Its dense neighbors lose it
    mwght = zero(V)

    if mass
        @inbounds for r in rstart:rstop
            rnode = iw[r]; rwght = superwght[rnode]

            if ispositive(rwght) && isone(len[rnode]) && (!ispositive(nbypass) || dwght[rnode] == dS)
                if ispositive(nbypass)
                    mdlose!(ddeg, dmask, nwords, rnode, rwght)
                end

                superwght[rnode] = zero(V); degprev[rnode] = zero(V); degnext[rnode] = -mdnode
                len[rnode] = elen[rnode] = zero(V); mwght += rwght
            end
        end
    end

    # the other nodes lose the nodes eliminated with `mdnode`, and its false
    # twins, as they lose `mdnode`
    delta -= mwght; swght += mwght + fwght
    @inbounds degree[mdnode] -= mwght

    # for each principal node in the reachable set, assign its bound and its
    # key, and return it to the degree structure. Its newest clique is the
    # new element with the dense neighbors of `mdnode`, of weight `delta`
    @inbounds for r in rstart:rstop
        rnode = iw[r]; rwght = superwght[rnode]

        if ispositive(rwght)
            degprev[rnode] = degnext[rnode] = zero(V)
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
            lpw[rnode] = delta - rwght; disc[rnode] = 0.0; dexact[rnode] = lo == hi

            if discount
                enew[rnode] = mdnode
            end

            mindeg = amfinsert!(rnode, rwght, lo, mindeg, fkey, lpw, disc, dexact,
                deghead, degnext, degprev, offset, nbuck, nfine, sfine, scale, res)
        end
    end

    return mindeg, tag, mwght
end

# This routine recomputes every node of bucket `bucket` that is not exact. The
# nodes are grouped by their most recent element, and each group is
# recomputed at once (`amfrecompute!`). Returns the new minimum bucket and
# the tag.
#
# working arrays:
#
#   - `degprev`: for an element, the first node of its group; for the node
#     of a group taken first from the list, the corresponding node of the
#     next group (the groups are chained in order)
#   - `degnext`: for a node taken from the list, the next node in its group
function amfvisit!(
        bucket::V,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        marker::AbstractVector{I},
        tag::I,
        neqns::V,
        extlen::AbstractVector{V},
        dwght::AbstractVector{V},
        nbypass::V,
        offset::V,
        fkey::AbstractVector{Float64},
        lpw::AbstractVector{V},
        disc::AbstractVector{Float64},
        dexact::AbstractVector{Bool},
        nbuck::V,
        nfine::Float64,
        sfine::Float64,
        scale::Float64,
        res::Float64,
    ) where {I, V, E}
    # take the whole list, and group it by most recent element. The
    # groups are chained, in order, from `ghead` through `degprev` of the
    # node of each group taken first
    tag += one(I); gtag = tag; mindeg = bucket; ghead = gtail = zero(V)
    @inbounds node = deghead[bucket + offset]; deghead[bucket + offset] = zero(V)

    @inbounds while ispositive(node)
        nxnode = degnext[node]

        # ask for what the next node needs here, and for what this one
        # needs in its recomputation
        mdprefetch(pe, nxnode); mdprefetch(elen, nxnode)
        mdprefetch(len, node); mdprefetch(degree, node)

        degnext[node] = degprev[node] = zero(V)
        elmnt = iw[pe[node] + convert(E, elen[node]) - one(E)]

        if marker[elmnt] != gtag
            marker[elmnt] = gtag; degprev[elmnt] = zero(V)

            if iszero(gtail)
                ghead = node
            else
                degprev[gtail] = node
            end

            gtail = node
        end

        degnext[node] = degprev[elmnt]; degprev[elmnt] = node
        node = nxnode
    end

    # recompute each group (read the link first: the recomputation
    # inserts the node into a degree list)
    gnode = ghead

    @inbounds while ispositive(gnode)
        gnext = degprev[gnode]
        elmnt = iw[pe[gnode] + convert(E, elen[gnode]) - one(E)]

        mindeg, tag = amfrecompute!(degprev[elmnt], elmnt, pe, len, elen, iw,
            superwght, degree, deghead, degnext, degprev, marker,
            tag, neqns, extlen, dwght, nbypass, mindeg, offset,
            fkey, lpw, disc, dexact, nbuck, nfine, sfine, scale, res)

        gnode = gnext
    end

    return mindeg, tag
end

# This routine recomputes the external degrees of the nodes of the chain that
# starts at `zhead` and is linked by `degnext`, whose most recent element is
# `elmnt`. The nodes of `elmnt` are marked, the list of each other element of
# the nodes is scanned once and partitioned in place, its nodes outside
# `elmnt` first, and the external degree of each node is the weight of
# `elmnt`, of the union of these nodes and of its uneliminated neighbors, and
# of its dense neighbors, minus its own weight. The nodes become exact and are
# inserted into the degree lists by their new keys. Returns the new minimum
# bucket and the tag.
#
# The elements in the lists of all the nodes (the core) are united once, as
# they are scanned, so each node only adds its other elements. This keeps
# groups whose nodes share many overlapping elements cheap.
#
# working arrays:
#
#   - `extlen`: the list of each element `f` is partitioned: its `extlen[f]`
#     nodes outside `elmnt` come first, in order, then its nodes in `elmnt`.
#     Merged and eliminated nodes are dropped, so `len[f]` can shrink.
#   - `marker`: the nodes of `elmnt` are marked `etag`; while the elements
#     are counted, `marker[f] - base` is the number of nodes that list `f`;
#     the core and the nodes of its lists are marked `ctag`, the other
#     elements `stag`, and the union of each node `utag`.
@inline function amfrecompute!(
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
        marker::AbstractVector{I},
        tag::I,
        neqns::V,
        extlen::AbstractVector{V},
        dwght::AbstractVector{V},
        nbypass::V,
        mindeg::V,
        offset::V,
        fkey::AbstractVector{Float64},
        lpw::AbstractVector{V},
        disc::AbstractVector{Float64},
        dexact::AbstractVector{Bool},
        nbuck::V,
        nfine::Float64,
        sfine::Float64,
        scale::Float64,
        res::Float64,
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
    # them, `marker[f] - base` (elements and nodes are marked in the same
    # vector: an element is never an uneliminated node)
    tag += one(I); base = tag; tag += convert(I, gsize)

    znode = zhead

    @inbounds while ispositive(znode)
        znext = degnext[znode]
        p1 = pe[znode]
        p2 = p1 + convert(E, elen[znode]) - one(E)

        for p in p1:p2
            f = iw[p]

            if f != elmnt
                if marker[f] <= base
                    marker[f] = base + one(I)
                else
                    marker[f] += one(I)
                end
            end
        end

        znode = znext
    end

    # scan each other element of the nodes once, and partition its list. The
    # elements listed by every node of the group (the core) are marked
    # `ctag`, and the union of their nodes outside `elmnt`, also marked
    # `ctag`, is taken once here: its weight is `cwght`
    tag += one(I); stag = tag; tag += one(I); ctag = tag
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

                if core
                    marker[f] = ctag
                else
                    marker[f] = stag
                end

                # the nodes outside `elmnt` go to the front of the list
                # (`pout` is the next position for them, and `pin` for the
                # nodes in `elmnt`), and merged and eliminated nodes are
                # dropped
                jstart = pe[f]
                jstop = jstart + convert(E, len[f]) - one(E)
                pout = pin = jstart

                for j in jstart:jstop
                    node = iw[j]; wght = superwght[node]

                    if ispositive(wght)
                        if marker[node] != etag
                            iw[pin] = iw[pout]; iw[pout] = node
                            pout += one(E); pin += one(E)

                            if core && marker[node] != ctag
                                marker[node] = ctag; cwght += wght
                            end
                        else
                            iw[pin] = node; pin += one(E)
                        end
                    end
                end

                len[f] = convert(V, pin - jstart); extlen[f] = convert(V, pout - jstart)
            end
        end

        znode = znext
    end

    # for each node, take the union of the nodes outside `elmnt` of its
    # elements, and of its uneliminated neighbors
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
                qstart = pe[f]
                qstop = qstart + convert(E, extlen[f]) - one(E)

                for q in qstart:qstop
                    node = iw[q]

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

        degree[znode] = deg; dexact[znode] = true
        mindeg = amfinsert!(znode, superwght[znode], deg, mindeg, fkey, lpw, disc, dexact,
            deghead, degnext, degprev, offset, nbuck, nfine, sfine, scale, res)
        znode = znext
    end

    return mindeg, tag
end

# This routine discounts the keys of the nodes outside the reachable set of
# the pivot that are adjacent to it. The reachable set is now a clique, so the
# pairs of their neighbors in it are adjacent. A node `node` counted the
# pairs between its parts in the reachable set: the part `y` of its newest
# element `elmnt` (the pairs inside it were never counted), and each of its
# uneliminated neighbors there, of total weight `rv` and squared weight `rs`
# (summed in `amfelim!`, by supernode). It loses
#
#     2 y rv + rv^2 - rs
#
# if that is more than its discount. Every node with such pairs lies in the
# list of a node of the reachable set: otherwise its neighbors there lie in
# its newest element. The part of `elmnt` in the reachable set is its weight
# minus `wext[elmnt] - wflg` if `elmnt` was met at this step (`wext[elmnt]`
# at least `wflg`, see `amfelim!`), and 0 otherwise. The sums are reset to 0.
# Returns the new minimum bucket.
#
# input parameters:
#
#   - `nvlist`: the number of nodes in `vlist`
#   - `vlist`: the nodes outside the reachable set adjacent to it
#   - `enew`: the newest element of each node, or 0
#   - `superwght`: weight of the supernodes
#   - `degree`: the bounds of the nodes; for an element, its weight
#   - `(wext, wflg)`: external parts of the elements
#
# updated parameters:
#
#   - `vsum`: the weight of the neighbors of each node in the reachable set,
#     and its square, by supernode (`vsum[2 node - 1]` and `vsum[2 node]`)
#   - `mindeg`: the minimum bucket
#   - `(fkey, disc)`: the keys and discounts
#   - `(deghead, degnext, degprev)`: degree lists
function amfdiscount!(
        nvlist::Int,
        vlist::AbstractVector{V},
        vsum::AbstractVector{Float64},
        enew::AbstractVector{V},
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        wext::AbstractVector{V},
        wflg::V,
        mindeg::V,
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        offset::V,
        fkey::AbstractVector{Float64},
        lpw::AbstractVector{V},
        disc::AbstractVector{Float64},
        dexact::AbstractVector{Bool},
        nbuck::V,
        nfine::Float64,
        sfine::Float64,
        scale::Float64,
        res::Float64,
    ) where {V}
    @inbounds for k in oneto(nvlist)
        node = vlist[k]; wght = superwght[node]
        q = twice(convert(Int, node)); rv = vsum[q - 1]; rs = vsum[q]
        vsum[q - 1] = 0.0; vsum[q] = 0.0

        if ispositive(wght)
            # the part of the newest element of `node` in the reachable set
            y = 0.0; elmnt = enew[node]

            if ispositive(elmnt)
                we = wext[elmnt]

                if we >= wflg
                    y = max(convert(Float64, degree[elmnt] - (we - wflg)), 0.0)
                end
            end

            # the pairs between its parts in the reachable set
            d = 2.0 * y * rv + rv * rv - rs

            if d > disc[node]
                disc[node] = d
                mdremove!(node, deghead, degnext, degprev, offset)
                mindeg = amfinsert!(node, wght, degree[node], mindeg, fkey, lpw, disc, dexact,
                    deghead, degnext, degprev, offset, nbuck, nfine, sfine, scale, res)
            end
        end
    end

    return mindeg
end

# This routine inserts `node`, which is in no list, into the degree lists:
# into the bucket of its key, computed from its weight `wght` and its degree
# (or lower bound) `deg`, in the exact list if `dexact[node]`. Returns the new
# minimum bucket.
@inline function amfinsert!(
        node::V,
        wght::V,
        deg::V,
        mindeg::V,
        fkey::AbstractVector{Float64},
        lpw::AbstractVector{V},
        disc::AbstractVector{Float64},
        dexact::AbstractVector{Bool},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        offset::V,
        nbuck::V,
        nfine::Float64,
        sfine::Float64,
        scale::Float64,
        res::Float64,
    ) where {V}
    @inbounds key = amfkey(deg, wght, lpw[node], disc[node], res)
    @inbounds fkey[node] = key
    bucket = amfbucket(key, nfine, sfine, scale, nbuck)
    @inbounds isexact = dexact[node]
    return mdinsert!(bucket, mindeg, node, isexact, deghead, degnext, degprev, offset)
end

# the key of a supernode of weight `wght` and degree `deg`, whose newest
# clique weighs `lpw` outside it, with the discount `disc`: an upper bound on
# twice its fill, over its weight plus 1, in units of the smallest weight
# (`res` integer weights to it)
@inline function amfkey(deg::V, wght::V, lpw::V, disc::Float64, res::Float64) where {V}
    d = convert(Float64, deg); l = convert(Float64, lpw)
    return max(d * d - l * l - disc, 0.0) / ((convert(Float64, wght) + res) * res)
end

# the bucket of key `key` (see `amf`)
@inline function amfbucket(key::Float64, nfine::Float64, sfine::Float64, scale::Float64, nbuck::V) where {V}
    if key < nfine
        bucket = floor(key) + 1.0
    else
        bucket = nfine + 1.0 + floor((sqrt(key) - sfine) * scale)
    end

    return convert(V, min(bucket, convert(Float64, nbuck)))
end

# This routine converts the vertex weights to integers: a weight `w` becomes
# `round(res * w / wmin)`, at least 1, where `wmin` is the smallest weight.
# The resolution `res` is 1 if every weight is a multiple of `wmin`, and 16
# otherwise, less if the total would not fit in 60 bits. So the ordering does
# not depend on the unit of the weights. Returns the weights and `res`.
function amfweights(neqns::V, vwght::Ones) where {V}
    return Ones{Int}(neqns), 1.0
end

function amfweights(neqns::V, vwght::AbstractVector) where {V}
    wmin = Inf; wsum = 0.0; whole = true

    @inbounds for node in oneto(neqns)
        w = convert(Float64, vwght[node])

        if !(isfinite(w) && ispositive(w))
            throw(ArgumentError("vertex weights must be finite and positive, but weight $node is $(vwght[node])"))
        end

        wmin = min(wmin, w); wsum += w
    end

    @inbounds for node in oneto(neqns)
        r = convert(Float64, vwght[node]) / wmin

        if abs(r - round(r)) > 1.0e-9 * r
            whole = false
        end
    end

    if whole
        res = 1.0
    else
        res = 16.0
    end

    if res * wsum / wmin > 2.0^60
        res = 2.0^60 * wmin / wsum
    end

    wght = FVector{Int}(undef, neqns)

    @inbounds for node in oneto(neqns)
        wght[node] = max(1, round(Int, res * convert(Float64, vwght[node]) / wmin))
    end

    return wght, res
end

end
