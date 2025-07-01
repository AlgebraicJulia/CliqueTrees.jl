#= A collection of routines to find an MMD (multiple minimum degree) ordering.
   Written by Erik Demaine, eddemaine@uwaterloo.ca
   Based on a Fortran 77 code written by Joseph Liu.
   For information on the minimum degree algorithm see the articles:
   The evolution of the minimum degree algorithm by Alan George and
   Joseph Liu, SIAM Rev. 31 pp. 1 - 19, 1989.0
   Modification of the minimum degree algorithm by multiple
   elimination, ACM Trans. Math. Soft. 2 pp.141 - 152, 1985
=#

module MMDLib

using ArgCheck
using Base: oneto
using FillArrays
using FixedSizeArrays
using ..Utilities

export mmd

function mmd(neqns::V, xadj::AbstractVector, adjncy::AbstractVector{V}; kwargs...) where {V}
    vwght = Ones{V}(neqns)
    return mmd(neqns, vwght, xadj, adjncy; kwargs...)
end

function mmd(neqns::V, vwght::AbstractVector, xadj::AbstractVector, adjncy::AbstractVector{V}; kwargs...) where {V}
    @argcheck neqns <= length(vwght)
    newvwght = FixedSizeVector{V}(undef, neqns)

    @inbounds for node in oneto(neqns)
        newvwght[node] = trunc(V, vwght[node])
    end

    return mmd(neqns, newvwght, xadj, adjncy; kwargs...)
end

function mmd(neqns::V, vwght::AbstractVector{V}, xadj::AbstractVector{E}, adjncy::AbstractVector{V}; delta::Integer = 0) where {V, E}
    @argcheck neqns <= length(vwght)
    @argcheck neqns < length(xadj)
    @inbounds nnz = xadj[neqns + one(V)] - one(E)
    total = zero(V)

    @inbounds for node in oneto(neqns)
        total += vwght[node]
    end

    invp = FixedSizeVector{V}(undef, neqns)
    marker = FixedSizeVector{Int}(undef, neqns)
    mergeparent = FixedSizeVector{V}(undef, neqns)
    needsupdate = FixedSizeVector{V}(undef, neqns)
    supersize = FixedSizeVector{V}(undef, neqns)
    superwght = FixedSizeVector{V}(undef, neqns)
    elimnext = FixedSizeVector{V}(undef, neqns)
    deghead = FixedSizeVector{V}(undef, total)
    degnext = FixedSizeVector{V}(undef, neqns)
    degprev = FixedSizeVector{V}(undef, neqns)
    newadjncy = FixedSizeVector{V}(undef, nnz)

    mmd_impl!(invp, marker, mergeparent, needsupdate, supersize,
        superwght, elimnext, deghead, degnext, degprev, newadjncy,
        total, neqns, convert(V, delta), vwght, xadj, adjncy)

    return convert(Vector{V}, invp)
end

function mmd_impl!(
        invp::AbstractVector{V},
        marker::AbstractVector{I},
        mergeparent::AbstractVector{V},
        needsupdate::AbstractVector{V},
        supersize::AbstractVector{V},
        superwght::AbstractVector{V},
        elimnext::AbstractVector{V},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        newadjncy::AbstractVector{V},
        total::V,
        neqns::V,
        delta::V,
        vwght::AbstractVector{V},
        xadj::AbstractVector{E},
        adjncy::AbstractVector{V},
    ) where {I, V, E}
    @argcheck neqns < length(xadj)
    @inbounds nnz = xadj[neqns + one(V)] - one(E)

    @inbounds for i in oneto(nnz)
        newadjncy[i] = adjncy[i]
    end

    mmd!_impl!(
        invp, marker, mergeparent, needsupdate, supersize,
        superwght, elimnext, deghead, degnext, degprev, total,
        neqns, convert(V, delta), vwght, xadj, newadjncy)

    return invp
end

"""
    mmd!(neqns, xadj, adjncy, delta)

This routine implements the minimum degree algorithm. It makes use of the
implicit representation of elimination graphs by quotient graphs, and the
notion of indistinguishable nodes. It also implements the modifications by
multiple elimination and minimum external degree.

input parameters:

  - `neqns`: number of equations
  - `(xadj, adjncy)`: adjacency structure
  - `delta`: tolerance value for multiple elimination

output parameters:

  - `invp`: the minimum degree ordering

working arrays:

  - `deghead`: points to first node with degree deg, or 0 if there
    are no such nodes
  - `degnext`: points to the next node in the degree list
    associated with node, or 0 if node was the last in the
    degree list
  - `degprev`: points to the previous node in a degree list
    associated with node, or the negative of the degree of
    node (if node was the last in the degree list), or 0
    if the node is not in the degree lists
  - `supersize`: the size of the supernodes
  - `elimhead`: points to the first node eliminated in the current pass
  - `elimnext`: points to the next node in a eliminated supernode
    or 0 if there are no more after node
  - `marker`: a temporary marker vector
  - `mergeparent`: the parent map for the merged forest
  - `needsupdate`: positive iff node needs a degree update (0 otherwise)
"""
function mmd!_impl!(
        invp::AbstractVector{V},
        marker::AbstractVector{I},
        mergeparent::AbstractVector{V},
        needsupdate::AbstractVector{V},
        supersize::AbstractVector{V},
        superwght::AbstractVector{V},
        elimnext::AbstractVector{V},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        total::V,
        neqns::V,
        delta::V,
        vwght::AbstractVector{V},
        xadj::AbstractVector{E},
        adjncy::AbstractVector{V},
    ) where {I, V, E}
    @argcheck neqns <= length(invp)
    @argcheck neqns <= length(marker)
    @argcheck neqns <= length(mergeparent)
    @argcheck neqns <= length(needsupdate)
    @argcheck neqns <= length(supersize)
    @argcheck neqns <= length(superwght)
    @argcheck neqns <= length(elimnext)
    @argcheck total <= length(deghead)
    @argcheck neqns <= length(degnext)
    @argcheck neqns <= length(degprev)
    @argcheck neqns <= length(vwght)
    @argcheck neqns < length(xadj)

    # initialization for the minimum degree algorithm.
    @inbounds for node in oneto(neqns)
        invp[node] = zero(V)
        marker[node] = zero(I)
        mergeparent[node] = zero(V)
        needsupdate[node] = zero(V)
        supersize[node] = one(V)
        superwght[node] = vwght[node]
        elimnext[node] = zero(V)
    end

    @inbounds for node in oneto(total)
        deghead[node] = zero(V)
    end

    # - `mindeg` is the current minimum degree
    # - `num` counts the number of ordered nodes plus 1
    # - `tag` is used to facilitate marking nodes
    mindeg = total; num = one(V); tag = one(I)

    @inbounds for node in oneto(neqns)
        istart = xadj[node]; istop = xadj[node + one(V)] - one(E)

        if istart <= istop
            ndeg = vwght[node]

            for i in istart:istop
                ndeg += vwght[adjncy[i]]
            end

            fnode = deghead[ndeg]
            deghead[ndeg] = node
            degnext[node] = fnode

            if ispositive(fnode)
                degprev[fnode] = node
            end

            degprev[node] = -ndeg
            mindeg = min(mindeg, ndeg)
        else
            # eliminate isolated node
            marker[node] = maxint(I)
            invp[node] = num; num += one(V)
        end
    end

    @inbounds while num <= neqns
        while !ispositive(deghead[mindeg])
            mindeg += one(V)
        end

        # use value of `delta` to set up `mindeglimit`, which governs
        # when a degree update is to be performed.
        mindeglimit = max(mindeg, mindeg + delta)
        elimhead = zero(V)

        while true
            mdnode = deghead[mindeg]

            while !ispositive(mdnode)
                mindeg += one(V)

                if mindeg > mindeglimit
                    @goto pass
                end

                mdnode = deghead[mindeg]
            end

            # remove `mdnode` from the degree structure.
            mdnextnode = degnext[mdnode]
            deghead[mindeg] = mdnextnode

            if ispositive(mdnextnode)
                degprev[mdnextnode] = -mindeg
            end

            invp[mdnode] = num

            if num + supersize[mdnode] > neqns
                @goto main
            end

            # eliminate `mdnode` and perform quotient graph
            # transformation (reset `tag` value if necessary)
            tag += one(I)

            if tag >= maxint(I)
                tag = one(I)

                for node in oneto(neqns)
                    if marker[node] < maxint(I)
                        marker[node] = zero(I)
                    end
                end
            end

            mmdelim!(mdnode, xadj, adjncy, deghead, degnext, degprev,
                supersize, superwght, elimnext, marker, tag, mergeparent,
                needsupdate, invp)

            num += supersize[mdnode]
            elimnext[mdnode] = elimhead
            elimhead = mdnode
        end

        @label pass

        # update degrees of the nodes involved in the
        # minimum degree node's elimination
        if num > neqns
            @goto main
        end

        mindeg, tag = mmdupdate!(neqns, elimhead, vwght, xadj, adjncy,
            delta, mindeg, deghead, degnext, degprev, supersize, superwght,
            elimnext, marker, tag, mergeparent, needsupdate, invp)
    end

    @label main

    mergelastnum = degnext
    mmdnumber!(neqns, invp, mergeparent, mergelastnum)

    return invp
end

"""
    mmdelim!(mdnode, xadj, adjncy, deghead, degnext, degprev, supersize, elimnext, marker, tag, mergeparent, needsupdate, invp)

This routine eliminates the node mdnode of
minimum degree from the adjacency structure, which
is stored in the quotient Graph format. It also
transforms the quotient Graph representation of the
elimination graph.

input parameters:

  - `mdnode`: node of minimum degree
  - `tag`: tag value
  - `invp`: the inverse of an incomplete minimum degree ordering
    (it is zero at positions where the ordering is unknown)

updated parameters:

  - `(xadj, adjncy)`: updated adjacency structure
  - `deghead`: points to first node with degree deg, or 0 if there
    are no such nodes
  - `degnext`: points to the next node in the degree list
    associated with node, or 0 if node was the last in the
    degree list
  - `degprev`: points to the previous node in a degree list
    associated with node, or the negative of the degree of
    node (if node was the last in the degree list), or 0
    if the node is not in the degree lists
  - `supersize`: the size of the supernodes
  - `elimnext`: points to the next node in a eliminated supernode
    or 0 if there are no more after node
  - `marker`: a temporary marker vector
  - `mergeparent`: the parent map for the merged forest
  - `needsupdate`: positive iff the node needs a degree update (0 otherwise)
"""
function mmdelim!(
        mdnode::V,
        xadj::AbstractVector{E},
        adjncy::AbstractVector{V},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        supersize::AbstractVector{V},
        superwght::AbstractVector{V},
        elimnext::AbstractVector{V},
        marker::AbstractVector{I},
        tag::I,
        mergeparent::AbstractVector{V},
        needsupdate::AbstractVector{V},
        invp::AbstractVector{V},
    ) where {I, V, E}
    # find reachable set and place in data structure
    @inbounds marker[mdnode] = tag

    # - `elmnt` points to the beginning of the list of eliminated
    #   neighbors of `mdnode`
    # - `rloc` gives the storage location
    #   for the next reachable node.
    elmnt = zero(V)
    @inbounds rloc = xadj[mdnode]; rlmt = xadj[mdnode + one(V)] - one(E)

    @inbounds for i in rloc:rlmt
        neighbor = adjncy[i]

        if iszero(neighbor)
            break
        end

        if marker[neighbor] < tag
            marker[neighbor] = tag

            if iszero(invp[neighbor])
                adjncy[rloc] = neighbor
                rloc += one(E)
            else
                elimnext[neighbor] = elmnt
                elmnt = neighbor
            end
        end
    end

    # merge with reachable nodes from generalized elements
    @inbounds while ispositive(elmnt)
        adjncy[rlmt] = -elmnt
        j = xadj[elmnt]; jstop = xadj[elmnt + one(V)]
        node = adjncy[j]

        while !iszero(node)
            if node < zero(V)
                j = xadj[-node]; jstop = xadj[one(V) - node]
            else
                if marker[node] < tag && !isnegative(degnext[node])
                    marker[node] = tag

                    # use storage from eliminated nodes
                    # if necessary.
                    while rloc >= rlmt
                        link = -adjncy[rlmt]
                        rloc = xadj[link]; rlmt = xadj[link + one(V)] - one(E)
                    end

                    adjncy[rloc] = node; rloc += one(E)
                end

                j += one(E)
            end

            if j >= jstop
                break
            end

            node = adjncy[j]
        end

        elmnt = elimnext[elmnt]
    end

    if rloc <= rlmt
        @inbounds adjncy[rloc] = zero(V)
    end

    # for each node in the reachable set, do the following...
    @inbounds i = xadj[mdnode]; istop = xadj[mdnode + one(V)]
    @inbounds rnode = adjncy[i]

    @inbounds while !iszero(rnode)
        if isnegative(rnode)
            i = xadj[-rnode]; istop = xadj[one(V) - rnode]
        else
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

            # purge inactive quotient neighbors of `rnode`
            xqnbr = xadj[rnode]

            for j in xadj[rnode]:(xadj[rnode + one(V)] - one(E))
                neighbor = adjncy[j]

                if iszero(neighbor)
                    break
                end

                if marker[neighbor] < tag
                    adjncy[xqnbr] = neighbor; xqnbr += one(E)
                end
            end

            # if no active neighbor after the purging...
            nqnbrs = convert(V, xqnbr - xadj[rnode])

            if !ispositive(nqnbrs)
                # then merge `rnode` with `mdnode`
                supersize[mdnode] += supersize[rnode]; supersize[rnode] = zero(V)
                superwght[mdnode] += superwght[rnode]; superwght[rnode] = zero(V)
                mergeparent[rnode] = mdnode
                marker[rnode] = maxint(I)
            else
                # else flag `rnode` for degree update, and
                # add `mdnode` as a neighbor of `rnode`
                needsupdate[rnode] = nqnbrs + one(V)
                adjncy[xqnbr] = mdnode; xqnbr += one(E)

                if xqnbr < xadj[rnode + one(V)]
                    adjncy[xqnbr] = zero(V)
                end
            end

            degprev[rnode] = zero(V)
            i += one(E)
        end

        if i >= istop
            break
        end

        rnode = adjncy[i]
    end

    return
end

"""
    mmdupdate!(neqns, elimhead, vwght, xadj, adjncy, delta, mindeg, deghead, degnext, degprev, supersize, elimnext, marker, tag, mergeparent, needsupdate, invp)

This routine updates the degrees of nodes
after a multiple elimination step.

input parameters:

  - `elimhead`: the beginning of the list of eliminated
    nodes (i.e. newly formed elements)
  - `neqns`: number of equations
  - `(xadj, adjncy)`: adjacency structure
  - `delta`: tolerance value for multiple elimination
  - `invp`: the inverse of an incomplete minimum degree ordering.
    (it is zero at positions where the ordering is unknown)

updated parameters:

  - `mindeg`: new minimum degree after degree update
  - `deghead`: points to first node with degree deg, or 0 if there
    are no such nodes
  - `degnext`: points to the next node in the degree list
    associated with node, or 0 if node was the last in the
    degree list
  - `degprev`: points to the previous node in a degree list
    associated with node, or the negative of the degree of
    node (if node was the last in the degree list), or 0
    if the node is not in the degree lists
  - `supersize`: the size of the supernodes
  - `elimnext`: points to the next node in a eliminated supernode
    or 0 if there are no more after node
  - `marker`: marker vector for degree update
  - `tag`: tag value
  - `mergeparent`: the parent map for the merged forest
  - `needsupdate`: positive iff node needs update (0 otherwise)
"""
function mmdupdate!(
        neqns::V,
        elimhead::V,
        vwght::AbstractVector{V},
        xadj::AbstractVector{E},
        adjncy::AbstractVector{V},
        delta::V,
        mindeg::V,
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        supersize::AbstractVector{V},
        superwght::AbstractVector{V},
        elimnext::AbstractVector{V},
        marker::AbstractVector{I},
        tag::I,
        mergeparent::AbstractVector{V},
        needsupdate::AbstractVector{V},
        invp::AbstractVector{V},
    ) where {I, V, E}
    mindeglimit = mindeg + delta
    elimnode = elimhead

    # for each of the newly formed element, do the following
    # (reset `tag` value if necessary)
    @inbounds while ispositive(elimnode)
        mtag = tag + convert(I, mindeglimit)

        if mtag >= maxint(I)
            tag = one(I)
            mtag = tag + convert(I, mindeglimit)

            for node in oneto(neqns)
                if marker[node] < maxint(I)
                    marker[node] = zero(I)
                end
            end
        end

        # create two linked lists from nodes associated
        # with `elmnt`: one with two neighbors (`q2head`) in
        # adjacency structure, and the other with more
        # than two neighbors (`qxhead`)
        #
        # also compute `elimwght`, weight of the nodes in this element
        q2head = qxhead = zero(V); elimwght = vwght[elimnode]
        i = xadj[elimnode]; istop = xadj[elimnode + one(V)]
        enode = adjncy[i]

        while !iszero(enode)
            if isnegative(enode)
                i = xadj[-enode]; istop = xadj[one(V) - enode]
            else
                if !iszero(supersize[enode])
                    elimwght += superwght[enode]
                    marker[enode] = mtag

                    # if `enode` requires a degree update,
                    # then do the following
                    if ispositive(needsupdate[enode])
                        # place either in `qxhead` or `q2head` lists
                        if !istwo(needsupdate[enode])
                            elimnext[enode] = qxhead; qxhead = enode
                        else
                            elimnext[enode] = q2head; q2head = enode
                        end
                    end
                end

                i += one(E)
            end

            if i >= istop
                break
            end

            enode = adjncy[i]
        end

        # For each enode in q2 list, do the following.
        enode = q2head

        while ispositive(enode)
            if ispositive(needsupdate[enode])
                tag += one(I)
                deg = elimwght

                # identify the other adjacent element neighbor
                istart = xadj[enode]
                neighbor = adjncy[istart]

                if neighbor == elimnode
                    neighbor = adjncy[istart + one(E)]
                end

                # if neighbor is uneliminated, increase degree count
                if iszero(invp[neighbor])
                    deg += superwght[neighbor]
                else
                    # otherwise, for each node in the 2nd element,
                    # do the following.
                    i = xadj[neighbor]
                    istop = xadj[neighbor + one(V)]
                    node = adjncy[i]

                    while !iszero(node)
                        if isnegative(node)
                            i = xadj[-node]
                            istop = xadj[one(V) - node]
                        else
                            if node != enode && !iszero(supersize[node])
                                if marker[node] < tag
                                    # case when `node` is not yet considered
                                    marker[node] = tag
                                    deg += superwght[node]
                                elseif ispositive(needsupdate[node])
                                    # case when `node` is indistinguishable from
                                    # `enode`
                                    #
                                    # merge them into a new supernode
                                    if istwo(needsupdate[node])
                                        supersize[enode] += supersize[node]; supersize[node] = zero(V)
                                        superwght[enode] += superwght[node]; superwght[node] = zero(V)
                                        marker[node] = maxint(I)
                                        mergeparent[node] = enode
                                    end

                                    needsupdate[node] = zero(V)
                                    degprev[node] = zero(V)
                                end
                            end

                            i += one(E)
                        end

                        if i >= istop
                            break
                        end

                        node = adjncy[i]
                    end
                end

                deg, mindeg = updateexternaldegree!(deg, mindeg, enode,
                    superwght, deghead, degnext, degprev, needsupdate)
            end

            enode = elimnext[enode]
        end

        # for each `enode` in the qx list, do the following.
        enode = qxhead

        while ispositive(enode)
            if ispositive(needsupdate[enode])
                tag += one(I)
                deg = elimwght

                # for each unmarked neighbor of `enode`,
                # do the following.
                for i in xadj[enode]:(xadj[enode + one(V)] - one(E))
                    neighbor = adjncy[i]

                    if iszero(neighbor)
                        break
                    end

                    if marker[neighbor] < tag
                        marker[neighbor] = tag

                        # if uneliminated, include it in
                        # degree count
                        if iszero(invp[neighbor])
                            deg += superwght[neighbor]
                        else
                            # if eliminated, include unmarked
                            # nodes in this element into the
                            # degree count
                            j = xadj[neighbor]; jstop = xadj[neighbor + one(V)]
                            node = adjncy[j]

                            while !iszero(node)
                                if isnegative(node)
                                    j = xadj[-node]; jstop = xadj[one(V) - node]
                                else
                                    if marker[node] < tag
                                        marker[node] = tag
                                        deg += superwght[node]
                                    end

                                    j += one(E)
                                end

                                if j >= jstop
                                    break
                                end

                                node = adjncy[j]
                            end
                        end
                    end
                end

                # update external degree of `enode` in degree
                # structure, and `mindeg` if necessary
                deg, mindeg = updateexternaldegree!(deg, mindeg, enode,
                    superwght, deghead, degnext, degprev, needsupdate)
            end

            # get next `enode` in current element.
            enode = elimnext[enode]
        end

        # get next element in the list.
        tag = mtag
        elimnode = elimnext[elimnode]
    end

    return mindeg, tag
end

function updateexternaldegree!(
        deg::V,
        mindeg::V,
        enode::V,
        superwght::AbstractVector{V},
        deghead::AbstractVector{V},
        degnext::AbstractVector{V},
        degprev::AbstractVector{V},
        needsupdate::AbstractVector{V},
    ) where {V}
    @inbounds deg -= superwght[enode]
    @inbounds firstnode = deghead[deg]
    @inbounds deghead[deg] = enode
    @inbounds degnext[enode] = firstnode
    @inbounds degprev[enode] = -deg
    @inbounds needsupdate[enode] = zero(V)

    if ispositive(firstnode)
        @inbounds degprev[firstnode] = enode
    end

    return deg, min(deg, mindeg)
end

"""
    mmnumber!(neqns, invp, mergeparent)

This routine performs the final step in
producing the permutation and inverse permutation
vectors in the multiple elimination version of the
minimum degree ordering algorithm.

input parameters:

  - `neqns`: number of equations

updated paremeters

  - `invp`: on input, new number for roots in merged forest
    on output, this plus remaining inverse of perm
  - `mergeparent`: the parent map for the merged forest (compressed)

working arrays:

  - `mergelastnum`: last number used for a merged tree rooted at r
"""
function mmdnumber!(neqns::V, invp::AbstractVector{V}, mergeparent::AbstractVector{V}, mergelastnum::AbstractVector{V}) where {V}
    @inbounds for i in oneto(neqns)
        if iszero(mergeparent[i])
            mergelastnum[i] = invp[i]
        else
            mergelastnum[i] = zero(V)
        end
    end

    # for each node which has been merged, do the following.
    @inbounds for node in oneto(neqns)
        parent = mergeparent[node]

        if ispositive(parent)
            # trace the merged tree until one which has
            # not been merged, call it `root`
            root = zero(V)

            while ispositive(parent)
                root = parent
                parent = mergeparent[parent]
            end

            # number `node` after `root`
            invp[node] = mergelastnum[root] += one(V)

            # shorten the merged tree.
            while node != root
                parent = mergeparent[node]
                mergeparent[node] = root
                node = parent
            end
        end
    end

    return
end

function maxint(::Type{I}) where {I}
    return typemax(I) - convert(I, 100000)
end

end
