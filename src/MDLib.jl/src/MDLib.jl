#= A collection of routines to find an exact minimum degree ordering, or an
   exact minimum external degree ordering.
   The degree of every node is kept as a lower bound. Exact degrees are
   computed only for the nodes of minimum bound, once no exact node of
   minimum bound is left, in batches that share their most recent element,
   so each element is scanned once per batch.
   With external degree, an eager pass at each elimination also recomputes
   the nodes whose bound lies within their own weight of the floor.
   For information on the minimum degree algorithm see the articles:
   The evolution of the minimum degree algorithm by Alan George and
   Joseph Liu, SIAM Rev. 31 pp. 1 - 19, 1989
   A fast minimum degree algorithm and matching lower bound by Robert
   Cummings, Matthew Fahrbach, and Animesh Fatehpuria, SODA 2021
=#

module MDLib

using Base: oneto
using FillArrays
using ..Utilities

export md

"""
    md(neqns[, vwght], xadj, adjncy; external=true, dense=10.0, mass=true, twin=true, compress=true)

Compute a minimum degree permutation of a graph with n vertices and m edges in
O(mn) time and O(m + n) space. Returns the inverse permutation `invp`.

  - `neqns`: the number of vertices
  - `vwght`: vertex weights, positive integers; a vertex of weight w stands
    for w indistinguishable vertices (default 1)
  - `xadj`, `adjncy`: the graph; the neighbors of `v` are
    `adjncy[xadj[v]:xadj[v + 1] - 1]`, and self loops are ignored
  - `external`: minimize the external degree instead of the degree
  - `dense`: vertices with more than `max(16, dense * sqrt(neqns))` neighbors
    are handled as dense (a negative value disables this)
  - `mass`: mass elimination
  - `twin`: false-twin detection
  - `compress`: merge the vertices with the same closed neighborhood first
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
    @inbounds nnz = convert(Int, xadj[neqns + 1]) - 1

    if neqns < typemax(Int32) ÷ 8 && total < typemax(Int32) ÷ 4 && nnz < typemax(Int32)
        index = md(Int32, neqns, total, vwght, xadj, adjncy; kwargs...)
    else
        index = md(Int64, neqns, total, vwght, xadj, adjncy; kwargs...)
    end

    return convert(Vector{V}, index)
end

# This routine computes the ordering with working arrays of integer type `V`;
# `total` is the sum of the vertex weights. The other parameters are those of
# `md`.
function md(::Type{V}, neqns::Integer, total::Integer, vwght::AbstractVector, xadj::AbstractVector, adjncy::AbstractVector; external::Bool = true, dense::Real = 10.0, mass::Bool = true, twin::Bool = true, compress::Bool = true) where {V}
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
        index = md(V, Int32, Int32, n, convert(V, total), iwlen, vwght, xadj, adjncy, external, dense, mass, twin, compress)
    elseif iwlen < typemax(Int32)
        index = md(V, Int32, Int64, n, convert(V, total), iwlen, vwght, xadj, adjncy, external, dense, mass, twin, compress)
    elseif n < typemax(Int32) ÷ 64
        index = md(V, Int64, Int32, n, convert(V, total), iwlen, vwght, xadj, adjncy, external, dense, mass, twin, compress)
    else
        index = md(V, Int64, Int64, n, convert(V, total), iwlen, vwght, xadj, adjncy, external, dense, mass, twin, compress)
    end

    return index
end

# This routine allocates the working arrays, with vertices, weights and
# degrees of type `V`, positions in `iw` of type `E`, and tags of type `I`,
# and computes the ordering.
function md(::Type{V}, ::Type{E}, ::Type{I}, n::V, total::V, iwlen::Int, vwght::AbstractVector, xadj::AbstractVector, adjncy::AbstractVector, external::Bool, dense::Real, mass::Bool, twin::Bool, compress::Bool) where {V, E, I}
    marker = FVector{I}(undef, n)
    superwght = FVector{V}(undef, n)
    degree = FVector{V}(undef, n)
    deghead = FVector{V}(undef, 2 * (total + 1))
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

    # select the dense nodes (`dindex` and `dwght` are needed only if there
    # are any)
    ndense = mdndense(n, xadj, dense); dlen = zero(V)

    if ispositive(ndense)
        dlen = n
    end

    dindex = FVector{V}(undef, dlen)
    ndense = mdselect!(dindex, n, xadj, adjncy, dense, ndense)
    nwords = cld(convert(Int, ndense), 64)
    dwght = FVector{V}(undef, dlen)
    dnode = FVector{V}(undef, ndense)
    ddeg = FVector{V}(undef, ndense)
    dmask = FVector{UInt64}(undef, nwords * convert(Int, n))

    md_impl!(marker, superwght, degree, deghead, degnext, degprev,
        wext, dwght, extlen, pe, len,
        elen, iw, iwlen, heads, total, n, external, mass,
        twin, compress, vwght, xadj,
        adjncy, ndense, nwords, dindex, dnode, ddeg, dmask)

    return degnext
end

# This routine counts the nodes with more than `max(16, dense * sqrt(neqns))`
# neighbors: the candidates of `mdselect!` (0 if `dense` is negative).
function mdndense(neqns::V, xadj::AbstractVector, dense::Real) where {V}
    ndense = zero(V)

    if !isnegative(dense)
        thresh = max(16.0, dense * sqrt(convert(Float64, neqns)))

        @inbounds for node in oneto(neqns)
            if convert(Float64, xadj[node + one(V)] - xadj[node]) > thresh
                ndense += one(V)
            end
        end
    end

    return ndense
end

# This routine selects the dense nodes: the nodes with more than
# `max(16, dense * sqrt(neqns))` neighbors. To keep the space linear, the
# masks may use at most `nnz / 2` words, so if there are too many dense
# nodes, only those of highest degree are selected (the lowest-numbered
# among nodes of equal degree). `ndense` is the number of candidates (see
# `mdndense`). Returns the number of dense nodes and sets `dindex` (empty if
# there are no candidates).
function mdselect!(dindex::AbstractVector{V}, neqns::V, xadj::AbstractVector{E}, adjncy::AbstractVector, dense::Real, ndense::V) where {V, E}
    maxdense = ndense

    if ispositive(ndense)
        thresh = max(16.0, dense * sqrt(convert(Float64, neqns)))

        @inbounds for node in oneto(neqns)
            dindex[node] = zero(V)
        end

        # at most `64 * maxwords` dense nodes, where
        # `maxwords * neqns <= nnz / 2`
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
            sort!(cand; by = node -> (xadj[node] - xadj[node + one(V)], node))
            resize!(cand, maxdense); sort!(cand)
        end

        @inbounds for j in eachindex(cand)
            dindex[cand[j]] = convert(V, j)
        end
    end

    return maxdense
end

# This routine copies the adjacency structure into `iw`, and computes the
# ordering (`mdmain!`).
function md_impl!(
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
        external::Bool,
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
    # or 0 if `node` is first. They are merged before the minimum degree
    # algorithm starts (`mdmain!`)
    if compress
        nmerge = mdclasses!(neqns, pe, len, iw, wext, extlen, marker, deghead,
            degprev, nwords, dmask)
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

    mdmain!(marker, superwght, degree, deghead, degnext, degprev,
        wext, dwght, extlen, pe, len,
        elen, iw, iwlen, heads, pfree, total, neqns, external, mass, twin,
        ispositive(nmerge), vwght, ndense, nwords, dindex, dnode, ddeg, dmask)

    return degnext
end

# This routine finds the classes of nodes with the same closed
# neighborhood. Such nodes are indistinguishable: with degree, merging them
# keeps the ordering exact. `wext[node]` holds the sum of the closed
# neighborhood of `node` (-1 for a dense node, which is not merged). The
# first node of each class is kept in a hash table (buckets in `deghead`,
# chains in `degprev`). A node is compared with the first nodes in its bucket
# that have the same sum and length, by marking the closed neighborhood of
# the first node `r` with `r` (kept marked while `cur == r`): it joins `r` if
# it is marked and every other neighbor is marked (the lengths are equal),
# and, with dense nodes, the masks are equal. Sets `extlen[node]` to the
# first node of its class, or 0. Returns the number of nodes merged.
#
# input parameters:
#
#   - `neqns`: number of equations
#   - `(pe, len, iw)`: the adjacency structure
#   - `wext`: the sums of the closed neighborhoods
#   - `(nwords, dmask)`: the masks of the dense neighbors (see `mdselect!`)
#
# output parameters:
#
#   - `extlen`: the first node of the class of each node, or 0
#
# working arrays:
#
#   - `marker`: marks of the closed neighborhoods
#   - `deghead`: buckets of the hash table
#   - `degprev`: chains of the hash table
function mdclasses!(
        neqns::V,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        iw::AbstractVector{V},
        wext::AbstractVector{V},
        extlen::AbstractVector{V},
        marker::AbstractVector{I},
        deghead::AbstractVector{V},
        degprev::AbstractVector{V},
        nwords::Int,
        dmask::AbstractVector{UInt64},
    ) where {I, V, E}
    nmerge = zero(V); cur = zero(V)

    # buckets: a power of two between `neqns` and `2 neqns`
    nb = 1

    while nb < convert(Int, neqns)
        nb <<= 1
    end

    bmask = convert(UInt64, nb - 1)

    @inbounds for b in oneto(nb)
        deghead[b] = zero(V)
    end

    @inbounds for node in oneto(neqns)
        marker[node] = zero(I)
    end

    @inbounds for node in oneto(neqns)
        h = wext[node]; extlen[node] = zero(V)

        if !isnegative(h)
            nlen = len[node]; pstart = pe[node]; pstop = pstart + convert(E, nlen) - one(E)
            b = convert(Int, mdmix(h) & bmask) + 1
            r = deghead[b]; found = false

            while ispositive(r)
                if wext[r] == h && len[r] == nlen
                    tag = convert(I, r)

                    if cur != r
                        cur = r; marker[r] = tag
                        rstart = pe[r]; rstop = rstart + convert(E, nlen) - one(E)

                        for q in rstart:rstop
                            marker[iw[q]] = tag
                        end
                    end

                    same = marker[node] == tag

                    if same
                        for q in pstart:pstop
                            u = iw[q]

                            if u != r && marker[u] != tag
                                same = false
                                break
                            end
                        end
                    end

                    if same && ispositive(nwords)
                        same = mdsame(dmask, nwords, node, r)
                    end

                    if same
                        extlen[node] = r; nmerge += one(V); found = true
                        break
                    end
                end

                r = degprev[r]
            end

            if !found
                degprev[node] = deghead[b]; deghead[b] = node
            end
        end
    end

    return nmerge
end

# This routine implements the exact minimum degree algorithm. It makes use
# of the implicit representation of elimination graphs by quotient graphs,
# and the notion of indistinguishable nodes. It also implements the
# modification by minimum external degree.
#
# The degree of a node is kept as a lower bound, which is exact for some
# nodes. Let the key of a node be its bound plus 1 (with external degree)
# or plus its weight (with degree). The algorithm repeats the following.
# If an exact node has the minimum key, it is eliminated (`mdelim!`,
# `mdupdate!`), and the nodes of its reachable set get new bounds.
# Otherwise every node at the minimum key is recomputed: they are grouped
# by most recent element, and each group is recomputed at once
# (`mdrecompute!`).
#
# A node of weight w stands for w indistinguishable nodes, so the degree of
# a node is the weight of its closed neighborhood minus one, and its external
# degree is the weight of its neighbors outside its supernode.
#
# input parameters:
#
#   - `neqns`: number of equations
#   - `total`: sum of the vertex weights
#   - `vwght`: vertex weights (truncated to integers)
#   - `external`: minimize the external degree instead of the degree. Nodes
#     are indistinguishable if they lie in the reachable set of a pivot and
#     have the same elements and neighbors there; elements contained in a
#     reachable set are absorbed into the new element (aggressive
#     absorption). The ordering is exact for these supernodes.
#   - `mass`: mass elimination. Each node of the reachable set whose only
#     neighbors are those of the new element (it lies in no other element,
#     has no other neighbors, and has the same dense neighbors as the pivot)
#     is eliminated with the pivot. With degree such a node is
#     indistinguishable from the pivot, so the ordering remains exact. With
#     `external` it need not be, and the ordering is no longer exact.
#   - `twin`: false-twin elimination. Each node outside the reachable set
#     whose neighbors are exactly the reachable set of the pivot, which has
#     never been in a reachable set, and whose dense neighbors (and, with
#     degree, weight) are those of the pivot, is eliminated right after the
#     pivot's supernode. With degree it is then simplicial and of minimum
#     degree, so the ordering remains exact. With `external` it need not be,
#     and the ordering is no longer exact (see `mdelim!`).
#   - `merged`: merge the classes in `extlen` first. On input, `extlen[node]`
#     is the first node of the class of `node` (nodes with the same closed
#     neighborhood), or 0 if `node` is first (see `mdclasses!`)
#   - `iwlen`: length of `iw`
#   - `(ndense, nwords, dindex, dnode, ddeg, dmask)`: the dense nodes (see
#     `mdselect!`). Their lists are empty: their adjacency, and every
#     adjacency to them, is stored in the masks. They are kept out of the
#     quotient graph until one of them has minimum degree, so the ordering
#     remains exact; with `external`, indistinguishable nodes must also have
#     the same dense neighbors.
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
#   - `degnext`: the minimum degree ordering (repurposed from degree list
#     links)
#
# working arrays:
#
#   - `deghead`: `deghead[key]` points to the first exact node with a key,
#     and `deghead[key + total + 1]` to the first node that is not exact, or
#     0 if there are no such nodes
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
#     the reachable set, the first term of its bound (see `mdelim!`).
#   - `dwght`: the weight of the dense neighbors of each node (empty if there
#     are no dense nodes)
#   - hash buckets, used to detect indistinguishable nodes, share `deghead`
#     with the degree lists. They use its last entries, whose lists (not
#     exact, with keys near `total`) are almost always empty: the first node
#     of the bucket at `deghead[b]` is `-deghead[b]` if no degree list starts
#     there, and otherwise `degprev[deghead[b]]`, which is restored to `-b`
#     once the bucket is processed. The hash of a node of the reachable set
#     is kept in `marker` (see `mdelim!`).
#   - `extlen`: for an element, the number of its nodes outside the element
#     being recomputed, which come first in its list (see `mdrecompute!`).
#     For a node of the reachable set, `extlen` is the second term of its
#     bound, and then its upper bound (see `mdupdate!`).
#   - `marker`: temporary marker vector, used with increasing tags. A node
#     of the reachable set holds its hash there, as a negative number, until
#     the indistinguishable nodes are merged
#   - `heads`: the first entries of the lists, while `iw` is compressed (see
#     `mdcompress!`)
function mdmain!(
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
        external::Bool,
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
    ) where {I, V, E}
    @assert neqns <= length(marker)
    @assert neqns <= length(superwght)
    @assert neqns <= length(degree)
    @assert twice(total + one(V)) <= length(deghead)
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

    # initialization for the minimum degree algorithm.
    @inbounds for node in oneto(neqns)
        marker[node] = zero(I)
        superwght[node] = trunc(V, vwght[node])
        elen[node] = zero(V)
        degnext[node] = zero(V)
        degprev[node] = zero(V)
        wext[node] = one(V)
    end

    @inbounds for key in oneto(twice(total + one(V)))
        deghead[key] = zero(V)
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

    # - `mindeg` is the current minimum key
    # - `rank` counts the eliminated supernodes plus 1, and `wleft` is the
    #   weight of the uneliminated nodes
    # - `tag` is used to facilitate marking nodes; the tags are reset
    #   before they overflow
    # - `wflg` is the flag of `wext`, and `lemax` the largest weight of an
    #   element
    # - `offset` separates the keys of the lists that are not exact
    mindeg = total + one(V); rank = one(V); wleft = total; tag = zero(I)
    wflg = one(V); lemax = zero(V); offset = total + one(V)

    # - `nbypass` is the number of dense nodes outside the quotient graph
    # - `nmissing` is the number of ordered pairs of such nodes that are
    #   not adjacent
    nbypass = ndense; nmissing = 0; dmin = zero(V)

    if ispositive(ndense)
        @inbounds for node in oneto(neqns)
            dwght[node] = zero(V)
        end

        mddenseinit!(ddeg, dmask, nwords, ndense, dnode, neqns, superwght)

        @inbounds for j in oneto(ndense)
            nmissing += convert(Int, ndense) - 1 - mdcount(dmask, nwords, dnode[j])
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
            key = mdkey(deg, superwght[node], external)

            if key > one(V)
                mindeg = mdinsert!(key, mindeg, node, true, deghead, degnext, degprev, offset)
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
        # find the pivot `mdnode`. If no node at the minimum key is exact,
        # they are all recomputed, and the search starts over
        mdnode = zero(V)

        while iszero(mdnode)
            if tag > tagmax
                for node in oneto(neqns)
                    marker[node] = zero(I)
                end

                tag = zero(I)
            end

            while mindeg <= total && !ispositive(deghead[mindeg]) && !ispositive(deghead[mindeg + offset])
                mindeg += one(V)
            end

            # if a dense node has a smaller key than every bound, it rejoins
            # the quotient graph and is eliminated. `dmin` is a lower bound on
            # the keys of the dense nodes: they are scanned only if it is
            # below the minimum key
            dj = zero(V)

            if ispositive(nbypass) && !(mindeg <= total && mindeg <= dmin)
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

                dmin = dkey

                if mindeg <= total && mindeg <= dkey
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
            elseif ispositive(deghead[mindeg])
                # remove `mdnode` from the degree structure. The exact nodes
                # at the minimum key go before the others are recomputed:
                # their eliminations may touch those nodes and give them new
                # bounds
                mdnode = deghead[mindeg]
                mdremove!(mdnode, deghead, degnext, degprev, offset)

                # the new head of the list is likely the next pivot: ask for
                # its fields and its list while `mdnode` is eliminated
                @inbounds nxnode = deghead[mindeg]

                if ispositive(nxnode)
                    mdprefetch(superwght, nxnode)
                    mdprefetch(degree, nxnode); mdprefetch(len, nxnode)
                    mdprefetch(elen, nxnode); @inbounds mdprefetch(iw, pe[nxnode])
                end
            else
                # recompute every node at the minimum key
                mindeg, tag = mdvisit!(mindeg, pe, len, elen, iw, superwght,
                    degree, deghead, degnext, degprev, marker,
                    tag, neqns, extlen, dwght, nbypass, external, offset)
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
            # a key is at least the degree plus 1
            dlow = mdlose!(ddeg, dmask, nwords, mdnode, superwght[mdnode])
            dS = mdweight(dmask, nwords, dnode, mdnode, superwght)

            if dlow < typemax(V)
                dmin = min(dmin, dlow + one(V))
            end
        else
            dS = zero(V)
        end

        # eliminate `mdnode` and perform quotient graph
        # transformation
        rchsze, pfree, delta, swght, wflg, lemax, fsze, fwght = mdelim!(mdnode,
            pe, len, elen, iw, iwlen, pfree, heads, neqns, superwght, degree,
            deghead, degnext, degprev, marker, wext, extlen, wflg, lemax,
            dS, offset, external, twin, rank + one(V),
            nbypass, nwords, dmask, ddeg)

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
        mindeg, tag, mwght = mdupdate!(mdnode, rchsze, delta, dS, swght, fwght, pe,
            len, elen, iw, superwght, degree, deghead, degnext,
            degprev, wext, marker, tag,
            neqns, extlen, dwght, nwords, dmask, ddeg, nbypass, mindeg, external,
            mass && iszero(fsze), offset)

        rank += one(V) + fsze; wleft -= swght + fwght + mwght
        superwght[mdnode] = zero(V)

        if ispositive(dS) && (ispositive(fsze) || mass)
            dmin = zero(V)
        end
    end

    mdnumber!(neqns, degnext, degprev, degree)
    return degnext
end

# This routine eliminates the node `mdnode` of minimum degree from the
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
# With `twin`, the false twins of `mdnode` are found first: the nodes
# outside the reachable set whose neighbors are exactly the reachable set.
# Each lies in every list of the reachable set, so only the shortest is
# scanned. A node of it qualifies if it has never been in a reachable set
# (its adjacency is then all in its list, and its degree is exact), its
# degree and dense neighbors (and, with degree, its weight) are those of
# `mdnode`, and its list holds no node outside the reachable set. Once
# `mdnode` is eliminated, it is simplicial and has the key of `mdnode`. It
# leaves the quotient graph, numbered right after the supernode of
# `mdnode`: its weight is set to 0 before the lists are purged, so it drops
# out of them.
#
# The reachable set is written to `iw` as it is formed, and becomes the list of
# the new element. If `mdnode` is adjacent to no element, its list of
# uneliminated neighbors holds the reachable set, which is formed there, in
# place. Otherwise it is formed at `pfree`, after a compression if the tail of
# `iw` is too short: it holds at most `degree[mdnode] - dS` nodes (each node
# weighs at least 1). That fits after a compression: outside a rejoin, the
# lists take at most `nnz` entries, and `iw` has at least `nnz + 2 neqns`
# (see `md`); a rejoining dense node has no element.
#
# input parameters:
#
#   - `mdnode`: node of minimum degree
#   - `neqns`: number of equations
#   - `iwlen`: length of `iw`
#   - `dS`: the weight of the dense neighbors of `mdnode`
#   - `offset`: separates the keys of the lists that are not exact
#   - `external`: the keys are external degrees (the false twins need not
#     have the weight of `mdnode`)
#   - `twin`: find the false twins of `mdnode`
#   - `fnum`: the rank of the first false twin (the rank after that of
#     `mdnode`)
#   - `(nbypass, nwords, dmask)`: the dense nodes
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
function mdelim!(
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
        external::Bool,
        twin::Bool,
        fnum::V,
        nbypass::V,
        nwords::Int,
        dmask::AbstractVector{UInt64},
        ddeg::AbstractVector{V},
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
        @inbounds need = convert(E, degree[mdnode] - dS)

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

        for p in p3:p4
            node = iw[p]; wght = superwght[node]

            if ispositive(wght)
                iw[pn] = node; pn += one(E); hash += node % UInt64
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
    return rchsze, pfree, ewght + dS, swght, wflg, max(lemax, ewght), fsze, fwght
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
# with `extlen[u]` added. With external degree, the supernodes whose bound
# is less than `max(0, delta - m - swght) + w(u) - 1` are recomputed at once
# (`mdrecompute!`). The supernodes are returned to the degree lists.
#
# input parameters:
#
#   - `mdnode`: the eliminated node
#   - `rchsze`: the size of the reachable set of `mdnode`, which is its list
#   - `delta`: the weight of the reachable set, with the dense neighbors of
#     `mdnode`
#   - `dS`: the weight of the dense neighbors of `mdnode`
#   - `swght`: the weight of `mdnode`
#   - `fwght`: the weight of the false twins of `mdnode` (see `mdelim!`)
#   - `(pe, len, elen, iw)`: quotient graph
#   - `(wext, extlen)`: the terms of the bounds (see `mdelim!`); on output,
#     `extlen[u]` is the upper bound of each node `u` of the reachable set
#   - `external`: key the degree lists by external degree instead of degree
#   - `mass`: mass elimination
#   - `offset`: separates the keys of the lists that are not exact
#
# updated parameters:
#
#   - `mindeg`: new minimum key after degree update
#   - `ddeg`: the degrees of the dense nodes, which lose the nodes eliminated
#     with `mdnode`
#   - `(superwght, degree)`: supernodes and their bounds
#   - `(deghead, degnext, degprev)`: degree lists
#   - `marker`: the hashes of the nodes of the reachable set (see `mdelim!`),
#     then tags; the hash buckets share `deghead` with the degree lists
#     (empty on output)
#   - `(marker, tag)`: marker vector
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
        external::Bool,
        mass::Bool,
        offset::V,
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
    # twins, as they lose `mdnode`. The floor of the eager pass is that of
    # `mdnode`, without the nodes eliminated with it
    delta -= mwght; floor = max(zero(V), delta - swght)
    swght += mwght + fwght
    @inbounds degree[mdnode] -= mwght

    # for each principal node in the reachable set, assign its bound. With
    # degree, return it to the degree structure at once; with external
    # degree, chain the nodes for the eager pass through `degnext`, in order,
    # and the others in a second chain
    zhead = ztail = ohead = otail = zero(V)

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

            if !external
                key = mdkey(lo, rwght, external)
                mindeg = mdinsert!(key, mindeg, rnode, lo == hi, deghead, degnext, degprev, offset)
            elseif lo < hi && lo < floor + rwght - one(V)
                if iszero(ztail)
                    zhead = rnode
                else
                    degnext[ztail] = rnode
                end

                ztail = rnode
            else
                if iszero(otail)
                    ohead = rnode
                else
                    degnext[otail] = rnode
                end

                otail = rnode
            end
        end
    end

    # the eager pass
    if ispositive(zhead)
        mindeg, tag = mdrecompute!(zhead, mdnode, pe, len, elen, iw,
            superwght, degree, deghead, degnext, degprev, marker,
            tag, neqns, extlen, dwght, nbypass, mindeg, external, offset)
    end

    # with external degree, return the other nodes to the degree structure
    # (the eager pass leaves their links alone: they are in no list)
    rnode = ohead

    @inbounds while ispositive(rnode)
        rnext = degnext[rnode]
        key = mdkey(degree[rnode], superwght[rnode], external)
        mindeg = mdinsert!(key, mindeg, rnode, degree[rnode] == extlen[rnode], deghead, degnext, degprev, offset)
        rnode = rnext
    end

    return mindeg, tag, mwght
end

# This routine recomputes every node with key `key` that is not exact. The
# nodes are grouped by their most recent element, and each group is
# recomputed at once (`mdrecompute!`). Returns the new minimum key and the
# tag.
#
# working arrays:
#
#   - `degprev`: for an element, the first node of its group; for the node
#     of a group taken first from the list, the corresponding node of the
#     next group (the groups are chained in order)
#   - `degnext`: for a node taken from the list, the next node in its group
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
        marker::AbstractVector{I},
        tag::I,
        neqns::V,
        extlen::AbstractVector{V},
        dwght::AbstractVector{V},
        nbypass::V,
        external::Bool,
        offset::V,
    ) where {I, V, E}
    # take the whole list, and group it by most recent element. The
    # groups are chained, in order, from `ghead` through `degprev` of the
    # node of each group taken first
    tag += one(I); gtag = tag; mindeg = key; ghead = gtail = zero(V)
    @inbounds node = deghead[key + offset]; deghead[key + offset] = zero(V)

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

        mindeg, tag = mdrecompute!(degprev[elmnt], elmnt, pe, len, elen, iw,
            superwght, degree, deghead, degnext, degprev, marker,
            tag, neqns, extlen, dwght, nbypass, mindeg, external, offset)

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
# inserted into the degree lists. Returns the new minimum key and the tag.
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
        marker::AbstractVector{I},
        tag::I,
        neqns::V,
        extlen::AbstractVector{V},
        dwght::AbstractVector{V},
        nbypass::V,
        mindeg::V,
        external::Bool,
        offset::V,
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

        degree[znode] = deg
        key = mdkey(deg, superwght[znode], external)
        mindeg = mdinsert!(key, mindeg, znode, true, deghead, degnext, degprev, offset)
        znode = znext
    end

    return mindeg, tag
end

# This routine removes `node` from the degree lists.
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

# This routine inserts `node` into the exact (`isexact`) or other degree
# list with key `key`, and returns the new minimum key.
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
    index = key

    if !isexact
        index += offset
    end

    @inbounds firstnode = deghead[index]
    @inbounds deghead[index] = node
    @inbounds degnext[node] = firstnode
    @inbounds degprev[node] = -index

    if ispositive(firstnode)
        @inbounds degprev[firstnode] = node
    end

    return min(key, mindeg)
end

const MDPREFETCH = """
    declare void @llvm.prefetch.p0i8(i8*, i32, i32, i32)

    define void @entry(i$(Sys.WORD_SIZE) %0) #0 {
        %p = inttoptr i$(Sys.WORD_SIZE) %0 to i8*
        call void @llvm.prefetch.p0i8(i8* %p, i32 0, i32 3, i32 1)
        ret void
    }

    attributes #0 = { alwaysinline }
    """

# This routine asks the processor to bring `A[i]` into its caches (an LLVM
# prefetch: read, keep in all levels). It is a hint: it never faults, so `i`
# may lie outside `A`, and it does nothing unless `A` is dense. The pointer is
# written as `i8*`, which LLVM 15 (Julia 1.10) reads as a typed pointer and
# later versions upgrade.
@inline mdprefetch(A::AbstractVector, i::Integer) = nothing

@inline function mdprefetch(A::DenseVector, i::Integer)
    ptr = convert(UInt, pointer(A, i))
    Base.llvmcall((MDPREFETCH, "entry"), Cvoid, Tuple{UInt}, ptr)
    return
end

# the key of a node: its external degree plus 1, or its degree plus 1
@inline function mdkey(deg::V, wght::V, external::Bool) where {V}
    if external
        key = deg + one(V)
    else
        key = deg + wght
    end

    return key
end

# the sum of the entries of a list is mixed, so that the hash buckets are
# used evenly (lists with the same sum share a bucket, and are compared)
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
    same = true

    @inbounds for w in 1:nwords
        if dmask[ioff + w] != dmask[joff + w]
            same = false
            break
        end
    end

    return same
end

# This routine computes the degrees of the dense nodes. On input, the masks
# hold the original adjacency to the dense nodes.
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

# This routine removes `node`, which is about to be eliminated, from the
# degrees of its dense neighbors. Returns the lowest of their new degrees
# (`typemax(V)` if it has none).
function mdlose!(ddeg::AbstractVector{V}, dmask::AbstractVector{UInt64}, nwords::Int, node::V, wght::V) where {V}
    off = mdoffset(nwords, node); dlow = typemax(V)

    @inbounds for w in 1:nwords
        word = dmask[off + w]

        while !iszero(word)
            j = ((w - 1) << 6) + trailing_zeros(word) + 1
            ddeg[j] -= wght; dlow = min(dlow, ddeg[j])
            word &= word - one(UInt64)
        end
    end

    return dlow
end

# This routine updates the adjacency to the dense nodes after the elimination
# of `mdnode`: its neighbors become a clique, so every node in the reachable
# set, and every dense neighbor of `mdnode`, becomes adjacent to every dense
# neighbor of `mdnode`. The reachable set is `iw[rstart:rstart + rchsze - 1]`.
# Each new adjacency is added to the dense weight
# `dwght` of the node, or to the degree `ddeg` of the dense node. Returns the
# updated number `nmissing` of ordered pairs of nonadjacent dense nodes.
function mddense!(
        mdnode::V,
        rstart::E,
        rchsze::V,
        iw::AbstractVector{V},
        dwght::AbstractVector{V},
        superwght::AbstractVector{V},
        nwords::Int,
        dnode::AbstractVector{V},
        ddeg::AbstractVector{V},
        dmask::AbstractVector{UInt64},
        nmissing::Int,
    ) where {V, E}
    moff = mdoffset(nwords, mdnode)

    # skip if `mdnode` has no dense neighbors
    empty = true

    @inbounds for w in 1:nwords
        if !iszero(dmask[moff + w])
            empty = false
            break
        end
    end

    if !empty
        @inbounds for r in rstart:(rstart + convert(E, rchsze) - one(E))
            rnode = iw[r]
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
    end

    return nmissing
end

# This routine returns the dense node with index `j` to the quotient graph.
# Its neighbors are the uneliminated principal nodes, other than dense nodes,
# whose masks contain `j`; they become its list, and it is appended to each
# of their lists (which are moved to the end of `iw` to make room). The node
# is then removed from every mask. Returns the updated `pfree`.
#
# The lists fit after a compression. Outside a rejoin, each list is no
# longer than in the quotient graph that keeps the dense nodes, so the lists
# take at most `nnz` entries. Until the dense node is eliminated, the rejoin
# adds fewer than `2 neqns`: its list and one entry per neighbor, or, while a
# list is copied, the entries appended so far and that list.
#
# working arrays:
#
#   - `wext`: chains the neighbors of the dense node, in order (it is not in
#     use for uneliminated nodes between eliminations)
#   - `heads`: working array of `mdcompress!`
function mdreinsert!(
        j::V,
        neqns::V,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        iwlen::E,
        pfree::E,
        heads::AbstractVector{UInt64},
        superwght::AbstractVector{V},
        nwords::Int,
        dindex::AbstractVector{V},
        dnode::AbstractVector{V},
        dmask::AbstractVector{UInt64},
        wext::AbstractVector{V},
    ) where {V, E}
    @inbounds dnd = dnode[j]; dnode[j] = -dnd
    w = ((convert(Int, j) - 1) >> 6) + 1
    bit = one(UInt64) << ((convert(Int, j) - 1) & 63)
    cnt = zero(V); head = tail = zero(V)

    @inbounds for node in oneto(neqns)
        i = mdoffset(nwords, node) + w

        if !iszero(dmask[i] & bit)
            dmask[i] &= ~bit

            if !ispositive(dindex[node]) && ispositive(superwght[node])
                # move the list of `node` to the end of `iw` and append
                # `dnd` to it
                nlen = len[node]

                if pfree + convert(E, nlen) > iwlen
                    pfree = mdcompress!(neqns, pe, len, iw, pfree, heads)
                    @assert pfree + convert(E, nlen) <= iwlen
                end

                pstart = pe[node]; pe[node] = pfree

                for p in pstart:(pstart + convert(E, nlen) - one(E))
                    iw[pfree] = iw[p]; pfree += one(E)
                end

                iw[pfree] = dnd; pfree += one(E)
                len[node] = nlen + one(V)
                cnt += one(V)

                if iszero(tail)
                    head = node
                else
                    wext[tail] = node
                end

                tail = node
            end
        end
    end

    if pfree + convert(E, cnt) - one(E) > iwlen
        pfree = mdcompress!(neqns, pe, len, iw, pfree, heads)
        @assert pfree + convert(E, cnt) - one(E) <= iwlen
    end

    @inbounds pe[dnd] = pfree

    node = head

    @inbounds for k in oneto(cnt)
        iw[pfree] = node; pfree += one(E); node = wext[node]
    end

    @inbounds len[dnd] = cnt; elen[dnd] = zero(V)
    return pfree
end

# This routine compresses the quotient graph storage `iw`, releasing the
# space used by absorbed elements, merged nodes, and purged entries. The
# first entry of each list is marked in a bitmap, so the released space is
# skipped 64 entries at a time. Returns the new value of `pfree`.
#
# input parameters:
#
#   - `neqns`: number of equations
#   - `len`: lengths of the lists (0 for released lists)
#
# updated parameters:
#
#   - `(pe, iw)`: quotient graph storage
#   - `pfree`: the quotient graph is stored in `iw[1:pfree - 1]`
#
# working arrays:
#
#   - `heads`: the bitmap, of at least `cld(pfree - 1, 64)` words
function mdcompress!(
        neqns::V,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        iw::AbstractVector{V},
        pfree::E,
        heads::AbstractVector{UInt64},
    ) where {V, E}
    nheads = cld(convert(Int, pfree) - 1, 64)

    @inbounds for w in 1:nheads
        heads[w] = zero(UInt64)
    end

    # store `-node` in the first entry of the list of each node, saving the
    # entry in `pe[node]`, and mark the entry in `heads`

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

    @inbounds for w in 1:nheads
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

# This routine computes the ordering. On input, `degnext[node]` is
# `-(neqns + rank)` for a principal node, the `rank`-th supernode eliminated,
# and `-parent` for a node merged into another (or eliminated with it). The
# nodes of each supernode are counted from these trees, and the supernodes laid
# out in the order of their ranks: the principal first, then its other nodes in
# increasing order. On output, `degnext[node]` is the position of `node`.
#
# working arrays:
#
#   - `size`: for a principal, the number of nodes in its supernode, then the
#     next position in it
#   - `start`: for a rank, the size of its supernode, then its first position
function mdnumber!(neqns::V, degnext::AbstractVector{V}, size::AbstractVector{V}, start::AbstractVector{V}) where {V}
    @inbounds for node in oneto(neqns)
        size[node] = zero(V)
    end

    # count the nodes of each supernode, pointing every merged node at its
    # principal (path compression)
    nrank = zero(V)

    @inbounds for node in oneto(neqns)
        x = -degnext[node]

        if x > neqns
            size[node] += one(V); nrank = max(nrank, x - neqns)
        else
            root = x

            while -degnext[root] <= neqns
                root = -degnext[root]
            end

            j = node

            while j != root
                k = -degnext[j]; degnext[j] = -root; j = k
            end

            size[root] += one(V)
        end
    end

    # the first position of each rank
    @inbounds for node in oneto(neqns)
        x = -degnext[node]

        if x > neqns
            start[x - neqns] = size[node]
        end
    end

    pos = one(V)

    @inbounds for r in oneto(nrank)
        sz = start[r]; start[r] = pos; pos += sz
    end

    # the principals, then the other nodes in increasing order
    @inbounds for node in oneto(neqns)
        x = -degnext[node]

        if x > neqns
            size[node] = degnext[node] = start[x - neqns]
        end
    end

    @inbounds for node in oneto(neqns)
        x = -degnext[node]

        if ispositive(x)
            degnext[node] = size[x] += one(V)
        end
    end

    return
end

end
