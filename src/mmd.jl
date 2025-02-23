####################################################
# Original implementation:                         #
#    https://github.com/PetrKryslUCSD/Sparspak.jl  #
####################################################

function genmmd(
    neqns::V, xadj::AbstractVector{E}, adjncy::AbstractVector{V}, delta::V, maxint::V
) where {V,E}
    return genmmd!(neqns, xadj, copy(adjncy), delta, maxint)
end

# PURPOSE - This routine implements the minimum degree algorithm. It makes use of the
#   implicit representation of elimination graphs by quotient graphs, and the
#   notion of indistinguishable nodes. It also implements the modifications by
#   multiple elimination and minimum external degree.
# 
# INPUT PARAMETERS -
#   neqns - Number of equations.
#   (xadj, adjncy) - The adjacency structure.
#   delta - Tolerance value for multiple elimination.
# 
# OUTPUT PARAMETERS -
#   invp - The minimum degree ordering.
# 
# WORKING ARRAYS -
#   deghead - Points to first node with degree deg, or 0 if there
#             are no such nodes.
#   degnext - Points to the next node in the degree list
#             associated with node, or 0 if node was the last in the
#             degree list.
#   degprev - Points to the previous node in a degree list
#             associated with node, or the negative of the degree of
#             node (if node was the last in the degree list), or 0
#             if the node is not in the degree lists.
#   supersize - The size of the supernodes.
#   elimhead - Points to the first node eliminated in the current pass
#              Using elimnext, one can determine all nodes just
#              eliminated.
#   elimnext - Points to the next node in a eliminated supernode
#              or 0 if there are no more after node.
#   marker - A temporary marker vector.
#   mergeparent - the parent map for the merged forest.
#   needsupdate - > 0 iff node needs degree update. (0 otherwise).
function genmmd!(
    neqns::V, xadj::AbstractVector{E}, adjncy::AbstractVector{V}, delta::V, maxint::V
) where {V,E}
    # ------------------------------------------------
    # Initialization for the minimum degree algorithm.
    # ------------------------------------------------
    invp = zeros(V, neqns)
    deghead = zeros(V, neqns)
    marker = zeros(V, neqns)
    mergeparent = zeros(V, neqns)
    needsupdate = zeros(V, neqns)
    degnext = Vector{V}(undef, neqns)
    degprev = Vector{V}(undef, neqns)
    supersize = ones(V, neqns)
    elimnext = zeros(V, neqns)

    @inbounds for node in oneto(neqns)
        ndeg = convert(V, xadj[node + one(V)] - xadj[node])
        fnode = deghead[ndeg + one(V)]
        deghead[ndeg + one(V)] = node
        degnext[node] = fnode

        if fnode > zero(V)
            degprev[fnode] = node
        end

        degprev[node] = -ndeg
    end

    # ----------------------------------------------
    # num counts the number of ordered nodes plus 1.
    # ----------------------------------------------
    num = one(V)

    # -----------------------------
    # Eliminate all isolated nodes.
    # -----------------------------
    mdnode = deghead[begin]

    @inbounds while mdnode > zero(V)
        marker[mdnode] = maxint
        invp[mdnode] = num
        num += one(V)
        mdnode = degnext[mdnode]
    end

    deghead[begin] = zero(V)
    tag = one(V)
    mindeg = one(V)

    # ----------------------------------------
    # Search for node of the minimum degree.
    # mindeg is the current minimum degree;
    # tag is used to facilitate marking nodes.
    # ----------------------------------------
    @inbounds while num <= neqns
        while deghead[mindeg + one(V)] <= zero(V)
            mindeg += one(V)
        end

        # -------------------------------------------------------
        # Use value of delta to set up mindeglimit, which governs
        # when a degree update is to be performed.
        # -------------------------------------------------------
        mindeglimit = max(mindeg, mindeg + delta)
        elimhead = zero(V)

        while true
            mdnode = deghead[mindeg + one(V)]

            while mdnode <= zero(V)
                mindeg += one(V)

                if mindeg > mindeglimit
                    @goto pass
                end

                mdnode = deghead[mindeg + one(V)]
            end

            # ----------------------------------------
            # Remove mdnode from the degree structure.
            # ----------------------------------------
            mdnextnode = degnext[mdnode]
            deghead[mindeg + one(V)] = mdnextnode

            if mdnextnode > zero(V)
                degprev[mdnextnode] = -mindeg
            end

            invp[mdnode] = num

            if num + supersize[mdnode] > neqns
                @goto main
            end

            # ----------------------------------------------
            # Eliminate mdnode and perform quotient graph
            # transformation. (Reset tag value if necessary.)
            # ----------------------------------------------
            tag += one(V)

            if tag >= maxint
                tag = one(V)
                marker[marker .< maxint] .= zero(V)
            end

            mmdelim!(
                mdnode,
                xadj,
                adjncy,
                deghead,
                degnext,
                degprev,
                supersize,
                elimnext,
                marker,
                maxint,
                tag,
                mergeparent,
                needsupdate,
                invp,
            )
            num += supersize[mdnode]
            elimnext[mdnode] = elimhead
            elimhead = mdnode
        end

        @label pass

        # -------------------------------------------
        # Update degrees of the nodes involved in the
        # minimum degree nodes elimination.
        # -------------------------------------------
        if num > neqns
            @goto main
        end

        mindeg, tag = mmdupdate!(
            elimhead,
            xadj,
            adjncy,
            delta,
            mindeg,
            deghead,
            degnext,
            degprev,
            supersize,
            elimnext,
            marker,
            maxint,
            tag,
            mergeparent,
            needsupdate,
            invp,
        )
    end

    @label main

    mmdnumber!(neqns, invp, mergeparent)

    return invp
end

# PURPOSE - This routine eliminates the node mdnode of
#   minimum degree from the adjacency structure, which
#   is stored in the quotient Graph format. It also
#   transforms the quotient Graph representation of the
#   elimination graph.
#
# INPUT PARAMETERS -
#   mdnode - Node of minimum degree.
#   tag - Tag value.
#   invp - The inverse of an incomplete minimum degree ordering.
#          (It is zero at positions where the ordering is unknown.)
#
# UPDATED PARAMETERS -
#   (xadj, adjncy) - Updated adjacency structure (xadj is not updated).
#   deghead - Points to first node with degree deg, or 0 if there
#             are no such nodes.
#   degnext - Points to the next node in the degree list
#             associated with node, or 0 if node was the last in the
#             degree list.
#   degprev - Points to the previous node in a degree list
#             associated with node, or the negative of the degree of
#             node (if node was the last in the degree list), or 0
#             if the node is not in the degree lists.
#   supersize - The size of the supernodes.
#   elimnext - Points to the next node in a eliminated supernode
#              or 0 if there are no more after node.
#   marker - A temporary marker vector.
#   mergeparent - the parent map for the merged forest.
#   needsupdate - > 0 iff node needs update. (0 otherwise)
function mmdelim!(
    mdnode::V,
    xadj::AbstractVector{E},
    adjncy::AbstractVector{V},
    deghead::AbstractVector{V},
    degnext::AbstractVector{V},
    degprev::AbstractVector{V},
    supersize::AbstractVector{V},
    elimnext::AbstractVector{V},
    marker::AbstractVector{V},
    maxint::V,
    tag::V,
    mergeparent::AbstractVector{V},
    needsupdate::AbstractVector{V},
    invp::AbstractVector{V},
) where {V,E}
    # -----------------------------------------------
    # Find reachable set and place in data structure.
    # -----------------------------------------------
    marker[mdnode] = tag

    # --------------------------------------------------------
    # elmnt points to the beginning of the list of eliminated
    # neighbors of mdnode, and rloc gives the storage location
    # for the next reachable node.
    # --------------------------------------------------------
    elmnt = zero(V)
    rloc = xadj[mdnode]
    rlmt = xadj[mdnode + one(V)] - one(E)

    @inbounds for i in xadj[mdnode]:(xadj[mdnode + one(V)] - one(E))
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

    # -----------------------------------------------------
    # Merge with reachable nodes from generalized elements.
    # -----------------------------------------------------
    @inbounds while elmnt > zero(V)
        adjncy[rlmt] = -elmnt
        j = xadj[elmnt]
        jstop = xadj[elmnt + one(V)]
        node = adjncy[j]

        while !iszero(node)
            if node < zero(V)
                j = xadj[-node]
                jstop = xadj[one(V) - node]
            else
                if marker[node] < tag && degnext[node] >= zero(V)
                    marker[node] = tag

                    # ---------------------------------
                    # Use storage from eliminated nodes
                    # if necessary.
                    # ---------------------------------
                    while rloc >= rlmt
                        link = -adjncy[rlmt]
                        rloc = xadj[link]
                        rlmt = xadj[link + one(V)] - one(E)
                    end

                    adjncy[rloc] = node
                    rloc += one(E)
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
        adjncy[rloc] = zero(V)
    end

    # -------------------------------------------------------
    # For each node in the reachable set, do the following...
    # -------------------------------------------------------
    i = xadj[mdnode]
    istop = xadj[mdnode + one(V)]
    rnode = adjncy[i]

    @inbounds while !iszero(rnode)
        if rnode < zero(V)
            i = xadj[-rnode]
            istop = xadj[one(V) - rnode]
        else

            # --------------------------------------------
            # If rnode is in the degree list structure ...
            # --------------------------------------------
            pvnode = degprev[rnode]

            if !iszero(pvnode)

                # -------------------------------------
                # then remove rnode from the structure.
                # -------------------------------------
                nxnode = degnext[rnode]

                if nxnode > zero(V)
                    degprev[nxnode] = pvnode
                end

                if pvnode > zero(V)
                    degnext[pvnode] = nxnode
                else
                    deghead[one(V) - pvnode] = nxnode
                end
            end

            # ----------------------------------------
            # Purge inactive quotient nabors of rnode.
            # ----------------------------------------
            xqnbr = xadj[rnode]

            for j in xadj[rnode]:(xadj[rnode + one(V)] - one(E))
                neighbor = adjncy[j]

                if iszero(neighbor)
                    break
                end

                if marker[neighbor] < tag
                    adjncy[xqnbr] = neighbor
                    xqnbr += one(E)
                end
            end

            # ---------------------------------------
            # If no active nabor after the purging...
            # ---------------------------------------
            nqnbrs::V = xqnbr - xadj[rnode]

            if nqnbrs <= zero(V)
                # -----------------------------
                # then merge rnode with mdnode.
                # -----------------------------
                supersize[mdnode] += supersize[rnode]
                supersize[rnode] = zero(V)
                mergeparent[rnode] = mdnode
                marker[rnode] = maxint
            else
                # --------------------------------------
                # else flag rnode for degree update, and
                # add mdnode as a neighbor of rnode.
                # --------------------------------------
                needsupdate[rnode] = nqnbrs + one(V)
                adjncy[xqnbr] = mdnode
                xqnbr += one(E)

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
end

# PURPOSE - This routine updates the degrees of nodes
#   after a multiple elimination step.
#
# INPUT PARAMETERS -
#   elimhead - The beginning of the list of eliminated
#              nodes (i.e. newly formed elements).
#   neqns - Number of equations.
#   (xadj, adjncy) - Adjacency structure.
#   delta - Tolerance value for multiple elimination.
#   invp - The inverse of an incomplete minimum degree ordering.
#          (It is zero at positions where the Ordering is unknown.)
#
# UPDATED PARAMETERS -
#   mindeg - New minimum degree after degree update.
#   deghead - Points to first node with degree deg, or 0 if there
#             are no such nodes.
#   degnext - Points to the next node in the degree list
#             associated with node, or 0 if node was the last in the
#             degree list.
#   degprev - Points to the previous node in a degree list
#             associated with node, or the negative of the degree of
#             node (if node was the last in the degree list), or 0
#             if the node is not in the degree lists.
#   supersize - The size of the supernodes.
#   elimnext - Points to the next node in a eliminated supernode
#              or 0 if there are no more after node.
#   marker - Marker vector for degree update.
#   tag - Tag value.
#   mergeparent - The parent map for the merged forest.
#   needsupdate - > 0 iff node needs update. (0 otherwise)
function mmdupdate!(
    elimhead::V,
    xadj::AbstractVector{E},
    adjncy::AbstractVector{V},
    delta::V,
    mindeg::V,
    deghead::AbstractVector{V},
    degnext::AbstractVector{V},
    degprev::AbstractVector{V},
    supersize::AbstractVector{V},
    elimnext::AbstractVector{V},
    marker::AbstractVector{V},
    maxint::V,
    tag::V,
    mergeparent::AbstractVector{V},
    needsupdate::AbstractVector{V},
    invp::AbstractVector{V},
) where {V,E}
    mindeglimit = mindeg + delta
    deg = enode = zero(V)
    elimnode = elimhead

    # -------------------------------------------------------
    # For each of the newly formed element, do the following.
    # (Reset tag value if necessary.)
    # -------------------------------------------------------
    @inbounds while elimnode > zero(V)
        mtag = tag + mindeglimit

        if mtag >= maxint
            tag = one(V)
            mtag = tag + mindeglimit
            marker[marker .< maxint] .= zero(V)
        end

        # -------------------------------------------------
        # Create two linked lists from nodes associated
        # with elmnt: one with two neighbors (q2head) in
        # adjacency structure, and the other with more
        # than two nabors (qxhead).  Also compute elimsize,
        # number of nodes in this element.
        # -------------------------------------------------
        q2head = qxhead = elimsize = zero(V)
        i = xadj[elimnode]
        istop = xadj[elimnode + one(V)]
        enode = adjncy[i]

        while !iszero(enode)
            if enode < zero(V)
                i = xadj[-enode]
                istop = xadj[one(V) - enode]
            else
                if !iszero(supersize[enode])
                    elimsize += supersize[enode]
                    marker[enode] = mtag

                    # ----------------------------------
                    # If enode requires a degree update,
                    # then do the following.
                    # ----------------------------------
                    if needsupdate[enode] > zero(V)
                        # ---------------------------------------
                        # Place either in qxhead or q2head lists.
                        # ---------------------------------------
                        if needsupdate[enode] != two(V)
                            elimnext[enode] = qxhead
                            qxhead = enode
                        else
                            elimnext[enode] = q2head
                            q2head = enode
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

        # --------------------------------------------
        # For each enode in q2 list, do the following.
        # --------------------------------------------
        enode = q2head

        while enode > zero(V)
            if needsupdate[enode] > zero(V)
                tag += one(V)
                deg = elimsize

                # ------------------------------------------
                # Identify the other adjacent element nabor.
                # ------------------------------------------
                istart = xadj[enode]
                neighbor = adjncy[istart]

                if neighbor == elimnode
                    neighbor = adjncy[istart + one(E)]
                end

                # ---------------------------------------------------
                # If neighbor is uneliminated, increase degree count.
                # ---------------------------------------------------
                if iszero(invp[neighbor])
                    deg += supersize[neighbor]
                else
                    # --------------------------------------------
                    # Otherwise, for each node in the 2nd element,
                    # do the following.
                    # --------------------------------------------
                    i = xadj[neighbor]
                    istop = xadj[neighbor + one(V)]
                    node = adjncy[i]

                    while !iszero(node)
                        if node < zero(V)
                            i = xadj[-node]
                            istop = xadj[one(V) - node]
                        else
                            if node != enode && !iszero(supersize[node])
                                if marker[node] < tag
                                    # -------------------------------------
                                    # Case when node is not yet considered.
                                    # -------------------------------------
                                    marker[node] = tag
                                    deg += supersize[node]
                                elseif needsupdate[node] > zero(V)
                                    # ----------------------------------------
                                    # Case when node is indistinguishable from
                                    # enode. Merge them into a new supernode.
                                    # ----------------------------------------
                                    if needsupdate[node] == two(V)
                                        supersize[enode] += supersize[node]
                                        supersize[node] = zero(V)
                                        marker[node] = maxint
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

                deg, mindeg = mmdupdateexternaldegree!(
                    deg, mindeg, enode, supersize, deghead, degnext, degprev, needsupdate
                )
            end

            enode = elimnext[enode]
        end

        # ------------------------------------------------
        # For each enode in the qx list, do the following.
        # ------------------------------------------------
        enode = qxhead

        while enode > zero(V)
            if needsupdate[enode] > zero(V)
                tag += one(V)
                deg = elimsize

                # ------------------------------------
                # For each unmarked neighbor of enode,
                # do the following.
                # ------------------------------------
                for i in xadj[enode]:(xadj[enode + one(V)] - one(E))
                    neighbor = adjncy[i]

                    if iszero(neighbor)
                        break
                    end

                    if marker[neighbor] < tag
                        marker[neighbor] = tag

                        # ------------------------------
                        # If uneliminated, include it in
                        # deg count.
                        # ------------------------------
                        if iszero(invp[neighbor])
                            deg += supersize[neighbor]
                        else
                            # -------------------------------
                            # If eliminated, include unmarked
                            # nodes in this element into the
                            # degree count.
                            # -------------------------------
                            j = xadj[neighbor]
                            jstop = xadj[neighbor + one(V)]
                            node = adjncy[j]

                            while !iszero(node)
                                if node < zero(V)
                                    j = xadj[-node]
                                    jstop = xadj[one(V) - node]
                                else
                                    if marker[node] < tag
                                        marker[node] = tag
                                        deg += supersize[node]
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

                # -------------------------------------------
                # Update external degree of enode in degree
                # structure, and mdeg (min deg) if necessary.
                # -------------------------------------------
                deg, mindeg = mmdupdateexternaldegree!(
                    deg, mindeg, enode, supersize, deghead, degnext, degprev, needsupdate
                )
            end

            # ----------------------------------
            # Get next enode in current element.
            # ----------------------------------
            enode = elimnext[enode]
        end

        # -----------------------------
        # Get next element in the list.
        # -----------------------------
        tag = mtag
        elimnode = elimnext[elimnode]
    end

    return mindeg, tag
end

function mmdupdateexternaldegree!(
    deg::V,
    mindeg::V,
    enode::V,
    supersize::AbstractVector{V},
    deghead::AbstractVector{V},
    degnext::AbstractVector{V},
    degprev::AbstractVector{V},
    needsupdate::AbstractVector{V},
) where {V}
    @inbounds begin
        deg -= supersize[enode]
        firstnode = deghead[deg + one(V)]
        deghead[deg + one(V)] = enode
        degnext[enode] = firstnode
        degprev[enode] = -deg
        needsupdate[enode] = zero(V)

        if firstnode > zero(V)
            degprev[firstnode] = enode
        end
    end

    return deg, min(deg, mindeg)
end

# PURPOSE - This routine performs the final step in
#   producing the permutation and inverse permutation
#   vectors in the multiple elimination version of the
#   minimum degree ordering algorithm.
#
# INPUT PARAMETERS -
#   neqns - Number of equations.
#
# UPDATED PARAMETERS -
#   invp - On input, new number for roots in merged forest.
#          On output, this plus remaining inverse of perm.
#   mergeparent - the parent map for the merged forest (compressed).
#
# WORKING ARRAYS -
#   mergelastnum - Last number used for a merged tree rooted at r.
function mmdnumber!(neqns::V, invp::Vector{V}, mergeparent::Vector{V}) where {V}
    mergelastnum = zeros(V, neqns)

    @inbounds for i in oneto(neqns)
        if iszero(mergeparent[i])
            mergelastnum[i] = invp[i]
        end
    end

    # ------------------------------------------------------
    # For each node which has been merged, do the following.
    # ------------------------------------------------------
    @inbounds for node in oneto(neqns)
        parent = mergeparent[node]

        if parent > zero(V)
            # -----------------------------------------
            # Trace the merged tree until one which has
            # not been merged, call it root.
            # -----------------------------------------
            root = zero(V)

            while parent > zero(V)
                root = parent
                parent = mergeparent[parent]
            end

            # -----------------------
            # Number node after root.
            # -----------------------
            invp[node] = mergelastnum[root] += one(V)

            # ------------------------
            # Shorten the merged tree.
            # ------------------------
            while node != root
                parent = mergeparent[node]
                mergeparent[node] = root
                node = parent
            end
        end
    end
end
