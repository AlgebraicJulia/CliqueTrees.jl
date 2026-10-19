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
    md(neqns, xadj, adjncy; external=false, mass=false, dense=10.0, complement=64)
    md(neqns, vwght, xadj, adjncy; external=false, mass=false, dense=10.0, complement=64)

Compute an exact minimum degree ordering of the graph `(xadj, adjncy)`.
Returns `index`, where `index[node]` is the position of `node`. The vertex
weights `vwght` must be positive integers.

  - `external`: minimize the external degree instead of the degree
  - `mass`: use mass elimination. The ordering remains exact for the
    degree, but not for the external degree.
  - `dense`: nodes with more than `max(16, dense * sqrt(neqns))` neighbors
    are kept out of the quotient graph until one of them has minimum
    degree (a negative value disables this). The ordering remains exact.
  - `complement`: pivots of degree at least `complement` may count fill
    as the complement of the existing adjacency, instead of testing pairs.
    The ordering does not depend on it.
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

function md(::Type{V}, neqns::Integer, total::Integer, vwght::AbstractVector, xadj::AbstractVector{E}, adjncy::AbstractVector; external::Bool = false, mass::Bool = false, dense::Real = 10.0, complement::Integer = 64, debug::Bool = false) where {V, E}
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

    # select the dense nodes
    dindex = FVector{V}(undef, n)
    ndense = mdselect!(dindex, n, xadj, adjncy, dense)
    nwords = cld(convert(Int, ndense), 64)
    dnode = FVector{V}(undef, ndense)
    ddeg = FVector{V}(undef, ndense)
    dmask = FVector{UInt64}(undef, nwords * convert(Int, n))
    dns = MDDense(ndense, nwords, dindex, dnode, ddeg, dmask)

    md_impl!(marker, supersize, superwght, degree, deghead, degnext,
        degprev, hashhead, rchset, xset, yset, pe, len, elen, iw, iwlen,
        convert(V, total), n, external, mass, debug, newvwght, xadj, adjncy, dns,
        convert(V, min(complement, typemax(V))))

    return degnext
end

"""
    MDDense

The nodes kept out of the quotient graph ("dense nodes"). Fill never passes
through an uneliminated node, so the quotient graph of the remaining nodes
represents their elimination graph exactly. The adjacency between any node
and the dense nodes is kept in bit masks: when a node is eliminated, its
neighbors become a clique, so every node in its reachable set (and every
dense neighbor) acquires its dense neighbors. A dense node rejoins the
quotient graph when its degree is minimum, and is then eliminated.

  - `ndense`: number of dense nodes
  - `nwords`: number of 64-bit words in a mask
  - `dindex`: `dindex[node]` is the index of a dense node, or 0
  - `dnode`: `dnode[j]` is the dense node with index `j`; it is negated
    once the node rejoins the quotient graph
  - `ddeg`: `ddeg[j]` is the weighted external degree of dense node `j`
  - `dmask`: `dmask[(node - 1) * nwords + w]` is word `w` of the mask of
    the dense neighbors of `node`
"""
struct MDDense{V, A <: AbstractVector{V}, M <: AbstractVector{UInt64}}
    ndense::V
    nwords::Int
    dindex::A
    dnode::A
    ddeg::A
    dmask::M
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
        degof(node) = @inbounds xadj[node + one(V)] - xadj[node]
        partialsort!(cand, 1:maxdense; by = degof, rev = true)
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
        dns::MDDense{V},
        complement::V,
    ) where {I, V, E}
    @assert iwlen <= length(iw)

    # copy the adjacency structure into `iw`, dropping self loops
    pfree = one(E)

    if iszero(dns.ndense)
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
        dindex = dns.dindex; dnode = dns.dnode; dmask = dns.dmask
        nwords = dns.nwords

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

    md!_impl!(marker, supersize, superwght, degree, deghead, degnext,
        degprev, hashhead, rchset, xset, yset, pe, len, elen, iw, iwlen,
        pfree, total, neqns, external, mass, debug, vwght, dns, complement)

    return degnext
end

"""
    md!_impl!(marker, supersize, superwght, degree, deghead, degnext,
        degprev, hashhead, rchset, xset, yset, pe, len, elen, iw, iwlen,
        pfree, total, neqns, external, mass, debug, vwght, dns, complement)

This routine implements the exact minimum degree algorithm. It makes use
of the implicit representation of elimination graphs by quotient graphs,
and the notion of indistinguishable nodes. It also implements the
modifications by mass elimination and minimum external degree.

The degree of every node is known exactly at all times. When a node is
eliminated, its reachable set is formed one generalized element at a time;
merging an element can only create fill edges between the nodes it adds
and the nodes of the reachable set it does not contain, so only these
pairs are tested for adjacency in the quotient graph. Each new fill edge is
added to the degrees of its two endpoints. For pivots of large degree, the
fill of a merge is instead counted as a complement: the existing adjacency
between the old and new nodes of the reachable set is enumerated, and every
other pair is a fill edge, so fill edges are never visited.

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
  - `dns`: the dense nodes (see `MDDense`). Their lists are empty: their
    adjacency, and every adjacency to them, is stored in the masks.
  - `complement`: minimum degree of a pivot for complement counting

updated parameters:

  - `(pe, len, elen, iw)`: quotient graph. On input, `(pe, len, iw)` holds
    the adjacency structure. For an uneliminated node, the list
    `iw[pe[node]:pe[node] + len[node] - 1]` holds `elen[node]` elements,
    sorted by elimination order, followed by its uneliminated neighbors;
    for an eliminated node
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
        dns::MDDense{V},
        complement::V,
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
    @assert dns.ndense <= length(dns.dnode)
    @assert dns.ndense <= length(dns.ddeg)
    @assert neqns <= length(dns.dindex)
    @assert dns.nwords * neqns <= length(dns.dmask)

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
    #   than 6 `neqns` tags are used per elimination
    mindeg = total + one(V); num = one(V); tag = zero(I)
    maxtag = typemax(I) - six(I) * convert(I, neqns) - eight(I)

    # - `maxlen` bounds the length of every list (a purged list never grows)
    maxlen = zero(V)

    # - `ndense` is the number of dense nodes
    # - `nbypass` is the number of dense nodes outside the quotient graph
    # - `nmissing` is the number of ordered pairs of such nodes that are
    #   not adjacent
    ndense = dns.ndense; nbypass = ndense; nmissing = 0

    if ispositive(ndense)
        mddenseinit!(dns, neqns, vwght)

        @inbounds for j in oneto(ndense)
            nmissing += convert(Int, ndense) - 1 - mdcount(dns, dns.dnode[j])
        end
    end

    @inbounds for node in oneto(neqns)
        deg = zero(V); maxlen = max(maxlen, len[node])
        pstart = pe[node]
        pstop = pstart + convert(E, len[node]) - one(E)

        for p in pstart:pstop
            deg += vwght[iw[p]]
        end

        if ispositive(ndense)
            # dense nodes stay out of the degree lists
            ispositive(dns.dindex[node]) && continue
            deg += mdweight(dns, node, vwght)
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
        while mindeg <= total && !ispositive(deghead[mindeg])
            mindeg += one(V)
        end

        # if a dense node has a smaller key than every other node, it
        # rejoins the quotient graph and is eliminated
        dj = zero(V)

        if ispositive(nbypass)
            dkey = typemax(V)

            for j in oneto(ndense)
                dnode = dns.dnode[j]

                if ispositive(dnode)
                    if external
                        key = dns.ddeg[j] + one(V)
                    else
                        key = dns.ddeg[j] + superwght[dnode]
                    end

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
            mdnode = dns.dnode[dj]
            nmissing -= 2 * (convert(Int, nbypass) - 1 - mdcount(dns, mdnode))
            nbypass -= one(V)
            pfree, mdlen = mdreinsert!(dj, dns, neqns, pe, len, elen, iw,
                iwlen, pfree, supersize, degnext, rchset)
            maxlen = max(maxlen, mdlen)
            degree[mdnode] = dns.ddeg[dj]
        else
            # remove `mdnode` from the degree structure.
            mdnode = deghead[mindeg]
            mdnextnode = degnext[mdnode]
            deghead[mindeg] = mdnextnode

            if ispositive(mdnextnode)
                degprev[mdnextnode] = -mindeg
            end
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

        # the dense neighbors of `mdnode` lose it
        if ispositive(nbypass)
            mdlose!(dns, mdnode, superwght[mdnode])
        end

        rchsze, pfree, tag = mdelim!(mdnode, pe, len, elen, iw, iwlen,
            pfree, neqns, supersize, superwght, degree, deghead, degnext,
            degprev, hashhead, marker, tag, rchset, xset, yset, mass, maxlen,
            dns, nbypass, complement)

        # the nodes in the reachable set, and the dense neighbors of
        # `mdnode`, become adjacent to the dense neighbors of `mdnode`
        if ispositive(nbypass)
            nmissing = mddense!(dns, mdnode, rchsze, rchset, degree,
                superwght, nmissing)
        end

        # update the degrees of the nodes in the reachable set, merge the
        # indistinguishable ones, and return them to the degree structure
        mindeg, tag = mdupdate!(mdnode, rchsze, rchset, pe, len, elen, iw,
            supersize, superwght, degree, deghead, degnext, degprev,
            hashhead, marker, tag, mindeg, external, dns, nbypass)

        num += supersize[mdnode]

        if debug
            mdcheck(neqns, pe, len, elen, iw, supersize, superwght, degree,
                degnext, dns)
        end
    end

    mdnumber!(neqns, supersize, degnext)
    return degnext
end

# ---------------------------------------------------------------------
# dense nodes
# ---------------------------------------------------------------------

@inline function mdoffset(dns::MDDense, node::Integer)
    return (convert(Int, node) - 1) * dns.nwords
end

@inline function mdsetbit!(dmask::AbstractVector{UInt64}, nwords::Int, node::Integer, j::Integer)
    jj = convert(Int, j) - 1
    @inbounds dmask[(convert(Int, node) - 1) * nwords + (jj >> 6) + 1] |= one(UInt64) << (jj & 63)
    return
end

# number of dense neighbors of `node`
function mdcount(dns::MDDense, node::Integer)
    off = mdoffset(dns, node); cnt = 0

    @inbounds for w in 1:dns.nwords
        cnt += count_ones(dns.dmask[off + w])
    end

    return cnt
end

# weight of the dense neighbors of `node`
function mdweight(dns::MDDense, node::Integer, superwght::AbstractVector{V}) where {V}
    off = mdoffset(dns, node); wght = zero(V)

    @inbounds for w in 1:dns.nwords
        word = dns.dmask[off + w]

        while !iszero(word)
            j = ((w - 1) << 6) + trailing_zeros(word) + 1
            wght += superwght[dns.dnode[j]]
            word &= word - one(UInt64)
        end
    end

    return wght
end

# is the set of dense neighbors of `inode` equal to (`subset = false`) or
# contained in (`subset = true`) that of `jnode`?
function mdcompare(dns::MDDense, inode::Integer, jnode::Integer, subset::Bool)
    ioff = mdoffset(dns, inode); joff = mdoffset(dns, jnode)

    @inbounds for w in 1:dns.nwords
        iword = dns.dmask[ioff + w]; jword = dns.dmask[joff + w]

        if subset
            iszero(iword & ~jword) || return false
        else
            iword == jword || return false
        end
    end

    return true
end

"""
    mddenseinit!(dns, neqns, vwght)

This routine computes the degrees of the dense nodes. On input, the masks
hold the original adjacency to the dense nodes.
"""
function mddenseinit!(dns::MDDense{V}, neqns::V, vwght::AbstractVector{V}) where {V}
    @inbounds for j in oneto(dns.ndense)
        dns.ddeg[j] = zero(V)
    end

    @inbounds for node in oneto(neqns)
        off = mdoffset(dns, node); wght = vwght[node]

        for w in 1:dns.nwords
            word = dns.dmask[off + w]

            while !iszero(word)
                j = ((w - 1) << 6) + trailing_zeros(word) + 1
                dns.ddeg[j] += wght
                word &= word - one(UInt64)
            end
        end
    end

    return
end

"""
    mdlose!(dns, node, wght)

This routine removes `node`, which is about to be eliminated, from the
degrees of its dense neighbors.
"""
function mdlose!(dns::MDDense{V}, node::V, wght::V) where {V}
    off = mdoffset(dns, node)

    @inbounds for w in 1:dns.nwords
        word = dns.dmask[off + w]

        while !iszero(word)
            j = ((w - 1) << 6) + trailing_zeros(word) + 1
            dns.ddeg[j] -= wght
            word &= word - one(UInt64)
        end
    end

    return
end

"""
    mddense!(dns, mdnode, rchsze, rchset, degree, superwght, nmissing)

This routine updates the adjacency to the dense nodes after the elimination
of `mdnode`: its neighbors become a clique, so every node in the reachable
set, and every dense neighbor of `mdnode`, becomes adjacent to every dense
neighbor of `mdnode`. Each new adjacency is added to the degrees of its two
endpoints. Returns the updated number `nmissing` of ordered pairs of
nonadjacent dense nodes.
"""
function mddense!(dns::MDDense{V}, mdnode::V, rchsze::V, rchset::AbstractVector{V},
        degree::AbstractVector{V}, superwght::AbstractVector{V}, nmissing::Int) where {V}
    nwords = dns.nwords; dmask = dns.dmask; dnode = dns.dnode; ddeg = dns.ddeg
    moff = mdoffset(dns, mdnode)

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
        roff = mdoffset(dns, rnode); rdeg = zero(V); rwght = superwght[rnode]

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

        degree[rnode] += rdeg
    end

    # the dense neighbors of `mdnode` become pairwise adjacent
    if ispositive(nmissing)
        @inbounds for w in 1:nwords
            word = dmask[moff + w]

            while !iszero(word)
                j = ((w - 1) << 6) + trailing_zeros(word) + 1
                word &= word - one(UInt64)
                dj = dnode[j]; joff = mdoffset(dns, dj); jdeg = zero(V)

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
    mdreinsert!(j, dns, neqns, pe, len, elen, iw, iwlen, pfree, supersize,
        degnext, nbrs)

This routine returns the dense node with index `j` to the quotient graph.
Its neighbors are the uneliminated principal nodes, other than dense nodes,
whose masks contain `j`; they become its list, and it is appended to each
of their lists (which are moved to the end of `iw` to make room). The node
is then removed from every mask. Returns the updated `pfree` and the
length of the longest list created.

working arrays:

  - `nbrs`: the neighbors of the dense node
"""
function mdreinsert!(j::V, dns::MDDense{V}, neqns::V, pe::AbstractVector{E},
        len::AbstractVector{V}, elen::AbstractVector{V}, iw::AbstractVector{V},
        iwlen::E, pfree::E, supersize::AbstractVector{V},
        degnext::AbstractVector{V}, nbrs::AbstractVector{V}) where {V, E}
    dmask = dns.dmask; nwords = dns.nwords; dindex = dns.dindex
    dnode = dns.dnode[j]; dns.dnode[j] = -dnode
    w = ((convert(Int, j) - 1) >> 6) + 1
    bit = one(UInt64) << ((convert(Int, j) - 1) & 63)
    cnt = zero(V); maxlen = zero(V)

    @inbounds for node in oneto(neqns)
        i = (convert(Int, node) - 1) * nwords + w

        if !iszero(dmask[i] & bit)
            dmask[i] &= ~bit

            if !ispositive(dindex[node]) && !isnegative(degnext[node]) && ispositive(supersize[node])
                # move the list of `node` to the end of `iw` and append
                # `dnode` to it
                nlen = len[node]

                if pfree + convert(E, nlen) > iwlen
                    pfree = mdcompress!(neqns, pe, len, iw, pfree)
                    @assert pfree + convert(E, nlen) <= iwlen
                end

                pstart = pe[node]; pe[node] = pfree

                for p in pstart:(pstart + convert(E, nlen) - one(E))
                    iw[pfree] = iw[p]; pfree += one(E)
                end

                iw[pfree] = dnode; pfree += one(E)
                len[node] = nlen + one(V); maxlen = max(maxlen, len[node])
                cnt += one(V); nbrs[cnt] = node
            end
        end
    end

    if pfree + convert(E, cnt) - one(E) > iwlen
        pfree = mdcompress!(neqns, pe, len, iw, pfree)
        @assert pfree + convert(E, cnt) - one(E) <= iwlen
    end

    @inbounds pe[dnode] = pfree

    @inbounds for k in oneto(cnt)
        iw[pfree] = nbrs[k]; pfree += one(E)
    end

    @inbounds len[dnode] = cnt; elen[dnode] = zero(V)
    return pfree, max(maxlen, cnt)
end

"""
    mdelim!(mdnode, pe, len, elen, iw, iwlen, pfree, neqns, supersize,
        superwght, degree, deghead, degnext, degprev, hashhead, marker, tag,
        rchset, xset, yset, mass, maxlen, dns, nbypass, complement)

This routine eliminates the node `mdnode` of minimum degree from the
adjacency structure, which is stored in the quotient graph format. The
reachable set of `mdnode` is formed one generalized element at a time, and
the fill edges created by each element are added to the degrees of the
nodes involved, either by testing pairs of nodes or by counting the fill as
the complement of the existing adjacency. The routine then transforms the
quotient graph representation of the elimination graph: the reachable set
becomes the new element `mdnode`.

input parameters:

  - `mdnode`: node of minimum degree
  - `neqns`: number of equations
  - `iwlen`: length of `iw`
  - `tag`: tag value
  - `mass`: merge the nodes of the reachable set with no other neighbors
    with `mdnode` (mass elimination)
  - `maxlen`: bound on the length of every list
  - `dns`, `nbypass`: the dense nodes, and the number of them outside the
    quotient graph
  - `complement`: minimum degree of a pivot for complement counting

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
  - complement counting also uses the free tail of `iw`, `hashhead`
    (indexed by node), and `degprev` and `elen` of the elements; they are
    restored before the reachable set is processed further
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
        maxlen::V,
        dns::MDDense{V},
        nbypass::V,
        complement::V,
    ) where {I, V, E}
    # - `iw[estart:estop]` is the list of elements adjacent to `mdnode`
    @inbounds estart = pe[mdnode]
    @inbounds estop = estart + convert(E, elen[mdnode]) - one(E)

    # form the reachable set: `rchsze` is the number of supernodes in it,
    # and `rchcnt` the number of nodes. Pivots of large degree may count
    # fill as a complement (`mdreachc!`)
    @inbounds if degree[mdnode] >= complement && iwlen - pfree >= convert(E, 64)
        rchsze, rchcnt, tag = mdreachc!(mdnode, pe, len, elen, iw, iwlen,
            pfree, supersize, superwght, degree, degnext, degprev, hashhead,
            marker, tag, rchset, xset, yset, maxlen)
    else
        rchsze, rchcnt, tag = mdreach!(mdnode, pe, len, elen, iw, supersize,
            superwght, degree, degnext, marker, tag, rchset, xset, yset, maxlen)
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

        # (with dense nodes, `rnode` must have no dense neighbor that
        # `mdnode` lacks)
        if mass && pn == p1 && (iszero(nbypass) || mdcompare(dns, rnode, mdnode, true))
            # if no active neighbor after the purging, then merge `rnode`
            # with `mdnode` (mass elimination)
            if ispositive(nbypass)
                mdlose!(dns, rnode, superwght[rnode])
            end

            rchcnt += supersize[rnode]
            supersize[mdnode] -= supersize[rnode]; supersize[rnode] = zero(V)
            superwght[mdnode] += superwght[rnode]; superwght[rnode] = zero(V)
            degnext[rnode] = -mdnode
            len[rnode] = zero(V); elen[rnode] = zero(V)
        else
            # else append the new element `mdnode` to the element list, which
            # stays sorted by elimination order: move the first neighbor to
            # the end (`rnode` lost at least one entry, so there is room)
            iw[pn] = iw[pv]; iw[pv] = mdnode
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
    mdreach!(mdnode, pe, len, elen, iw, supersize, superwght, degree,
        degnext, marker, tag, rchset, xset, yset, maxlen)

This routine forms the reachable set of `mdnode`, testing pairs of nodes
for fill edges (`mdfill!`). Returns the number of supernodes and of nodes
in the reachable set, and the updated tag.
"""
function mdreach!(
        mdnode::V,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        supersize::AbstractVector{V},
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        degnext::AbstractVector{V},
        marker::AbstractVector{I},
        tag::I,
        rchset::AbstractVector{V},
        xset::AbstractVector{V},
        yset::AbstractVector{V},
        maxlen::V,
    ) where {I, V, E}
    rchsze = zero(V); rchcnt = zero(V)
    @inbounds estart = pe[mdnode]
    @inbounds estop = estart + convert(E, elen[mdnode]) - one(E)
    @inbounds nstart = estop + one(E)
    @inbounds nstop = estart + convert(E, len[mdnode]) - one(E)

    # merge the elements adjacent to `mdnode` into the reachable set,
    # one at a time, most recent first. If `elmnt` is the next element, the
    # only new fill edges are between Y, the nodes of `elmnt` not yet in the
    # reachable set, and X, the nodes of the reachable set not in `elmnt`.
    @inbounds for k in oneto(elen[mdnode])
        elmnt = iw[nstart - convert(E, k)]
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

            for l in oneto(rchsze)
                node = rchset[l]

                if marker[node] < tag
                    xsze += one(V); xset[xsze] = node
                end
            end

            if ispositive(xsze)
                tag = mdfill!(xset, xsze, yset, ysze, pe, len, elen, iw,
                    superwght, degree, degnext, marker, tag, maxlen)
            end
        end

        # add the nodes in `yset` to the reachable set
        for l in oneto(ysze)
            node = yset[l]
            supersize[node] = -supersize[node]
            rchcnt -= supersize[node]
            rchsze += one(V); rchset[rchsze] = node
        end
    end

    # merge the uneliminated neighbors of `mdnode` into the reachable set,
    # one at a time: each is an element of size two, so X is the whole
    # reachable set
    @inbounds for i in nstart:nstop
        node = iw[i]

        if !isnegative(degnext[node]) && ispositive(supersize[node])
            if ispositive(rchsze)
                yset[one(V)] = node
                tag = mdfill!(rchset, rchsze, yset, one(V), pe, len, elen,
                    iw, superwght, degree, degnext, marker, tag, maxlen)
            end

            supersize[node] = -supersize[node]
            rchcnt -= supersize[node]
            rchsze += one(V); rchset[rchsze] = node
        end
    end

    return rchsze, rchcnt, tag
end

"""
    mdreachc!(mdnode, pe, len, elen, iw, iwlen, pfree, supersize, superwght,
        degree, degnext, degprev, hashhead, marker, tag, rchset, xset, yset,
        maxlen)

This routine forms the reachable set of `mdnode` like `mdreach!`, but each
merge may count its fill as a complement. Returns the number of supernodes
and of nodes in the reachable set, and the updated tag.
"""
function mdreachc!(
        mdnode::V,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        iwlen::E,
        pfree::E,
        supersize::AbstractVector{V},
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        hashhead::AbstractVector{V},
        marker::AbstractVector{I},
        tag::I,
        rchset::AbstractVector{V},
        xset::AbstractVector{V},
        yset::AbstractVector{V},
        maxlen::V,
    ) where {I, V, E}
    rchsze = zero(V); rchcnt = zero(V)
    @inbounds estart = pe[mdnode]
    @inbounds estop = estart + convert(E, elen[mdnode]) - one(E)
    @inbounds nstart = estop + one(E)
    @inbounds nstop = estart + convert(E, len[mdnode]) - one(E)

    # merge the elements adjacent to `mdnode` into the reachable set, one
    # at a time, most recent first. If `elmnt` is the next element, the only
    # new fill edges join Y, the nodes of `elmnt` not yet in the reachable
    # set, to X, the nodes of the reachable set not in `elmnt`. Each merge
    # either tests the pairs of X × Y (`mdfill!`), or counts the fill as a
    # complement (`mdcount!`): for each node of Y, the nodes of X that are
    # already adjacent to it are enumerated, and the rest are fill. The
    # second way never touches a fill edge; it is used when the enumeration
    # is estimated to be cheaper.
    #
    # complement counting keeps, for every element, a bucket of the nodes
    # of the reachable set that belong to it. The buckets live in the free
    # tail of `iw`: entry `k` occupies `iw[pfree + 2k - 2]` (node) and
    # `iw[pfree + 2k - 1]` (next entry), and the touched elements are listed
    # downwards from `iw[iwlen]`. The head and size of the bucket of an
    # element are kept in its `degprev` and `elen`, which are otherwise
    # unused (and zero) for elements. The increments of the nodes of X are
    # accumulated lazily: `hashhead[node]` holds the sum of the weights of Y
    # at the time `node` joined the reachable set, plus corrections, and the
    # increment is `wsum - hashhead[node]`, where `wsum` is the final sum.
    # - `comp` is false once the free tail of `iw` is full
    # - `built` is true once the buckets exist; they are built (for the
    #   whole reachable set) at the first merge whose pairs outnumber twice
    #   the entries the buckets need
    # - `rchelen` is the number of entries the buckets need
    comp = true; built = false; rchelen = 0
    wsum = zero(V); rchwght = zero(V); nentry = 0; ntouch = 0

    @inbounds for k in oneto(elen[mdnode])
        elmnt = iw[nstart - convert(E, k)]
        tag += one(I); mtag = tag
        ysze = zero(V); ywght = zero(V); xsze = rchsze; xwght = rchwght
        jstart = pe[elmnt]
        jstop = jstart + convert(E, len[elmnt]) - one(E)

        for j in jstart:jstop
            node = iw[j]

            # skip eliminated and merged nodes
            if !isnegative(degnext[node])
                marker[node] = mtag

                # nodes in the reachable set have negative `supersize`
                if ispositive(supersize[node])
                    ysze += one(V); yset[ysze] = node; ywght += superwght[node]
                else
                    xsze -= one(V); xwght -= superwght[node]
                end
            end
        end

        iszero(ysze) && continue

        if ispositive(xsze)
            if comp && !built && 2 * rchelen <= convert(Int, xsze) * convert(Int, ysze)
                built = true

                for l in oneto(rchsze)
                    nentry, ntouch, comp = mdbucket!(rchset[l], pe, elen, iw,
                        iwlen, pfree, degprev, nentry, ntouch)

                    comp || break
                end
            end

            if comp && built && mdcomplement(yset, ysze, elmnt, xsze, pe, len, elen, iw)
                # the nodes of the reachable set in `elmnt` are not in X
                if xsze < rchsze
                    for j in jstart:jstop
                        node = iw[j]

                        if !isnegative(degnext[node]) && isnegative(supersize[node])
                            hashhead[node] += ywght
                        end
                    end
                end

                tag = mdcount!(yset, ysze, elmnt, xwght, mtag, tag, pe, len,
                    elen, iw, pfree, degnext, degprev, supersize, superwght,
                    degree, hashhead, marker)
                wsum += ywght
            else
                xsze = zero(V)

                for l in oneto(rchsze)
                    node = rchset[l]

                    if marker[node] != mtag
                        xsze += one(V); xset[xsze] = node
                    end
                end

                tag = mdfill!(xset, xsze, yset, ysze, pe, len, elen, iw,
                    superwght, degree, degnext, marker, tag, maxlen)
            end
        end

        # add the nodes in `yset` to the reachable set
        for l in oneto(ysze)
            node = yset[l]
            supersize[node] = -supersize[node]
            rchcnt -= supersize[node]
            rchsze += one(V); rchset[rchsze] = node
            rchwght += superwght[node]; rchelen += convert(Int, elen[node])
            hashhead[node] = wsum

            if comp && built
                nentry, ntouch, comp = mdbucket!(node, pe, elen, iw, iwlen,
                    pfree, degprev, nentry, ntouch)
            end
        end
    end

    # merge the uneliminated neighbors of `mdnode` into the reachable set,
    # one at a time: each is an element of size two, so X is the whole
    # reachable set
    @inbounds for i in nstart:nstop
        node = iw[i]

        if !isnegative(degnext[node]) && ispositive(supersize[node])
            if ispositive(rchsze)
                tag += one(I); yset[one(V)] = node

                if comp && built && mdcomplement(yset, one(V), zero(V), rchsze, pe, len, elen, iw)
                    tag = mdcount!(yset, one(V), zero(V), rchwght, tag, tag,
                        pe, len, elen, iw, pfree, degnext, degprev, supersize,
                        superwght, degree, hashhead, marker)
                    wsum += superwght[node]
                else
                    tag = mdfill!(rchset, rchsze, yset, one(V), pe, len, elen,
                        iw, superwght, degree, degnext, marker, tag, maxlen)
                end
            end

            supersize[node] = -supersize[node]
            rchcnt -= supersize[node]
            rchsze += one(V); rchset[rchsze] = node
            rchwght += superwght[node]; rchelen += convert(Int, elen[node])
            hashhead[node] = wsum

            if comp && built
                nentry, ntouch, comp = mdbucket!(node, pe, elen, iw, iwlen,
                    pfree, degprev, nentry, ntouch)
            end
        end
    end

    # apply the lazy increments, and empty the buckets
    @inbounds for l in oneto(rchsze)
        node = rchset[l]
        degree[node] += wsum - hashhead[node]
        hashhead[node] = zero(V)
    end

    @inbounds for t in 1:ntouch
        elmnt = iw[iwlen - convert(E, t) + one(E)]
        degprev[elmnt] = zero(V); elen[elmnt] = zero(V)
    end

    return rchsze, rchcnt, tag
end

"""
    mdcomplement(yset, ysze, elmnt, xsze, pe, len, elen, iw)

This routine decides whether a merge counts its fill as a complement. The
enumeration visits, for each node of Y, the buckets of its elements other
than `elmnt` and its uneliminated neighbors; the pairwise test visits
X × Y. The enumeration is chosen if it visits at most twice as many
entries.
"""
@inline function mdcomplement(
        yset::AbstractVector{V},
        ysze::V,
        elmnt::V,
        xsze::V,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
    ) where {V, E}
    budget = 2 * convert(Int, xsze) * convert(Int, ysze); cost = 0

    @inbounds for l in oneto(ysze)
        ynode = yset[l]
        p1 = pe[ynode]; p2 = p1 + convert(E, elen[ynode]) - one(E)
        cost += convert(Int, len[ynode] - elen[ynode])

        for q in p1:p2
            f = iw[q]

            if f != elmnt
                cost += convert(Int, elen[f])
            end
        end

        cost > budget && return false
    end

    return true
end

"""
    mdbucket!(node, pe, elen, iw, iwlen, pfree, degprev, nentry, ntouch)

This routine adds `node`, which has joined the reachable set, to the
buckets of its elements. Returns the updated numbers of entries and touched
elements, and `false` if the free tail of `iw` is full (some buckets are
then incomplete, and complement counting stops for this elimination).
"""
@inline function mdbucket!(
        node::V,
        pe::AbstractVector{E},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        iwlen::E,
        pfree::E,
        degprev::AbstractVector{V},
        nentry::Int,
        ntouch::Int,
    ) where {V, E}
    @inbounds p1 = pe[node]
    @inbounds p2 = p1 + convert(E, elen[node]) - one(E)
    room = convert(Int, iwlen - pfree) + 1
    maxentry = convert(Int, typemax(V)) - 1

    @inbounds for q in p1:p2
        f = iw[q]

        # one entry (two slots) and possibly one touched element
        if room - 2 * nentry - ntouch < 3 || nentry >= maxentry
            return nentry, ntouch, false
        end

        if iszero(elen[f])
            ntouch += 1; iw[iwlen - convert(E, ntouch) + one(E)] = f
        end

        nentry += 1
        slot = pfree + convert(E, 2 * nentry - 2)
        iw[slot] = node; iw[slot + one(E)] = degprev[f]
        degprev[f] = convert(V, nentry); elen[f] += one(V)
    end

    return nentry, ntouch, true
end

"""
    mdcount!(yset, ysze, elmnt, xwght, mtag, tag, pe, len, elen, iw, pfree,
        degnext, degprev, supersize, superwght, degree, hashhead, marker)

This routine counts the fill of a merge as a complement. For each node
`ynode` of Y, the nodes of X already adjacent to it are enumerated: the
nodes in the buckets of its elements (other than `elmnt`) and its
uneliminated neighbors in the reachable set, skipping the nodes of `elmnt`
(marked `mtag`) and counting each node once. The fill of `ynode` is the
weight of X, `xwght`, minus the weight of those nodes, and each of them
records that it is not joined to `ynode` by a fill edge (`hashhead`).
"""
function mdcount!(
        yset::AbstractVector{V},
        ysze::V,
        elmnt::V,
        xwght::V,
        mtag::I,
        tag::I,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        pfree::E,
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        supersize::AbstractVector{V},
        superwght::AbstractVector{V},
        degree::AbstractVector{V},
        hashhead::AbstractVector{V},
        marker::AbstractVector{I},
    ) where {I, V, E}
    @inbounds for l in oneto(ysze)
        ynode = yset[l]; ywght = superwght[ynode]
        tag += one(I); adj = zero(V)
        p1 = pe[ynode]; p2 = p1 + convert(E, elen[ynode]) - one(E)
        p4 = p1 + convert(E, len[ynode]) - one(E)

        for q in p1:p2
            f = iw[q]
            f == elmnt && continue
            b = degprev[f]

            while ispositive(b)
                slot = pfree + convert(E, 2 * convert(Int, b) - 2)
                node = iw[slot]; b = iw[slot + one(E)]
                mark = marker[node]

                if mark != mtag && mark != tag
                    marker[node] = tag; adj += superwght[node]; hashhead[node] += ywght
                end
            end
        end

        for q in (p2 + one(E)):p4
            node = iw[q]

            if isnegative(supersize[node]) && !isnegative(degnext[node])
                mark = marker[node]

                if mark != mtag && mark != tag
                    marker[node] = tag; adj += superwght[node]; hashhead[node] += ywght
                end
            end
        end

        degree[ynode] += xwght - adj
    end

    return tag
end

"""
    mdfill!(xset, xsze, yset, ysze, pe, len, elen, iw, superwght, degree,
        degnext, marker, tag, maxlen)

This routine adds the fill edges between the nodes of `xset` and the nodes
of `yset` to the degrees of their endpoints. Two nodes are joined by a fill
edge if they are not adjacent in the quotient graph: neither is a neighbor
of the other, and they belong to no common element. For each node `snode`
of one set, its quotient neighbors are marked, and for each node `tnode` of
the other set, its elements are checked against the marks.

The marked set is the smaller one, unless its lists are longer in total than
those of the other set (and than four entries per pair). The marking then
costs O(Σ min(len(s), len(t))) over the pairs. Element lists are sorted by
elimination order, so if the elements of `tnode` are many more than those of
`snode`, the latter are found by binary search instead (`mdsearch`). A pair
therefore costs O(1 + min(len(s), len(t)) log Δ), and by Lemma 3.5 of
Cummings, Fahrbach, and Fatehpuria (each pair u, v is tested at most
min(d(u), d(v)) times, where d is the original degree), the routine costs
O(Σ min(d(u), d(v))² log Δ) = O(m² log Δ) in total.

input parameters:

  - `(xset, xsze)`, `(yset, ysze)`: the two sets of nodes
  - `(pe, len, elen, iw)`: quotient graph
  - `superwght`: the weight of the supernodes
  - `degnext`: `-degnext[elmnt]` is the elimination order of an element
  - `tag`: tag value
  - `maxlen`: bound on the length of every list

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
        degnext::AbstractVector{V},
        marker::AbstractVector{I},
        tag::I,
        maxlen::V,
    ) where {I, V, E}
    if xsze <= ysze
        sset = xset; ssze = xsze; tset = yset; tsze = ysze
    else
        sset = yset; ssze = ysze; tset = xset; tsze = xsze
    end

    # mark the other set instead if its lists are shorter in total (unless
    # every list has at most `4tsze` entries, which the pairs pay for anyway)
    if four(V) * tsze < maxlen
        slen = 0

        @inbounds for k in oneto(ssze)
            slen += convert(Int, len[sset[k]])
        end

        if slen > 4 * convert(Int, ssze) * convert(Int, tsze)
            tlen = 0

            @inbounds for l in oneto(tsze)
                tlen += convert(Int, len[tset[l]])
            end

            if tlen < slen
                sset, tset = tset, sset
                ssze, tsze = tsze, ssze
            end
        end
    end

    # for each node `snode` in the marked set, do the following...
    @inbounds for k in oneto(ssze)
        snode = sset[k]
        tag += one(I)

        # mark the quotient neighbors (elements and nodes) of `snode`
        pstart = pe[snode]; selen = elen[snode]
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

            # if `tnode` belongs to an element of `snode`, they are adjacent.
            # Shared elements tend to be recent, so the elements of `tnode`
            # are checked from the most recent, unless they are many more
            # than those of `snode`
            qstart = pe[tnode]; telen = elen[tnode]

            if telen <= eight(V) * selen + convert(V, 64)
                adjacent = false
                qstop = qstart + convert(E, telen)

                for j in oneto(telen)
                    if marker[iw[qstop - convert(E, j)]] == tag
                        adjacent = true
                        break
                    end
                end
            else
                adjacent = mdsearch(iw, degnext, pstart, selen, qstart, telen)
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
    mdsearch(iw, degnext, astart, alen, bstart, blen)

This routine tests whether the element lists `iw[astart:astart + alen - 1]`
and `iw[bstart:bstart + blen - 1]` share an element, by binary search for
each element of the first list (most recent first) in the second. The lists
are sorted by elimination order: `elmnt` was eliminated before `other` if
`degnext[elmnt] > degnext[other]`, so `degnext` decreases along a list. The
keys are searched for in increasing order of `degnext`, so the elements of
the second list after the position of one key are more recent than every
remaining key: each search is confined to the part of the second list
before the previous one, and stops early if that part is empty.
"""
function mdsearch(
        iw::AbstractVector{V},
        degnext::AbstractVector{V},
        astart::E,
        alen::V,
        bstart::E,
        blen::V,
    ) where {V, E}
    # the remaining keys can only be in `iw[bstart:bstop]`
    bstop = bstart + convert(E, blen) - one(E)

    @inbounds for k in oneto(alen)
        key = degnext[iw[astart + convert(E, alen - k)]]
        lo = bstart; hi = bstop

        while lo <= hi
            mid = lo + ((hi - lo) >>> 1)
            midkey = degnext[iw[mid]]

            if midkey == key
                return true
            elseif midkey > key
                lo = mid + one(E)
            else
                hi = mid - one(E)
            end
        end

        # `iw[hi]` is the last element older than `key`
        hi < bstart && return false
        bstop = hi
    end

    return false
end

"""
    mdupdate!(mdnode, rchsze, rchset, pe, len, elen, iw, supersize,
        superwght, degree, deghead, degnext, degprev, hashhead, marker, tag,
        mindeg, external, dns, nbypass)

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
  - `dns`, `nbypass`: the dense nodes, and the number of them outside the
    quotient graph; indistinguishable nodes must have the same dense
    neighbors

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
        dns::MDDense{V},
        nbypass::V,
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

                if same && ispositive(nbypass)
                    same = mdcompare(dns, inode, jnode, false)
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
        dns::MDDense{V},
    ) where {V, E}
    marker = zeros(V, neqns)
    isactive(node) = !isnegative(degnext[node]) && ispositive(supersize[node])
    isbypassed(node) = ispositive(dns.dindex[node]) && ispositive(dns.dnode[dns.dindex[node]])

    # the degree of a dense node outside the quotient graph is the weight
    # of the active nodes whose masks contain it
    if ispositive(dns.ndense)
        ddeg = zeros(V, dns.ndense)

        for node in oneto(neqns)
            isactive(node) || continue
            off = mdoffset(dns, node)

            for w in 1:dns.nwords
                word = dns.dmask[off + w]

                while !iszero(word)
                    j = ((w - 1) << 6) + trailing_zeros(word) + 1
                    word &= word - one(UInt64)
                    dnode = dns.dnode[j]
                    @assert ispositive(dnode) && dnode != node

                    # adjacency between dense nodes is symmetric
                    if isbypassed(node)
                        jj = convert(Int, dns.dindex[node]) - 1
                        @assert !iszero(dns.dmask[mdoffset(dns, dnode) + (jj >> 6) + 1] & (one(UInt64) << (jj & 63)))
                    end

                    ddeg[j] += superwght[node]
                end
            end
        end

        for j in oneto(dns.ndense)
            ispositive(dns.dnode[j]) && @assert ddeg[j] == dns.ddeg[j]
        end
    end

    for node in oneto(neqns)
        isactive(node) || continue
        ispositive(dns.ndense) && isbypassed(node) && continue
        deg = zero(V); marker[node] = node

        if ispositive(dns.ndense)
            deg += mdweight(dns, node, superwght)
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

        @assert deg == degree[node]
    end

    return
end

end
