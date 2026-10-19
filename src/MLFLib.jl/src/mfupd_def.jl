#**********************************************************************
#**********************************************************************
#*****   MFUPD_DEF ... DEFICIENCY UPDATING   **************************
#**********************************************************************
#**********************************************************************
#
#     AUTHORS:
#       ESMOND G. NG, LAWRENCE BERKELEY NATIONAL LABORAOTY
#       BARRY W. PEYTON, DALTON STATE COLLEGE
#
#     LAST UPDATED:
#       2025-09-22
#       2026-10-19 (GROUPED, WORD-PARALLEL MARKING OF THE NEW CLIQUE
#                   MEMBERS; SEE NOTE BELOW.)
#
#**********************************************************************
#
#     PURPOSE:
#       THIS SUBROUTINE USES WING-HUANG UPDATING TO UPDATE THE
#       DEFICIENCIES OF ALL VERTICES WHOSE DEFICIENCIES ARE INFLUENCED
#       BY THE ELIMINATION OF ENODE.
#
#     NOTE:
#       THE NEW MEMBERS OF ENODE'S ELIMINATION CLIQUE (ECLIQ(FNODE:
#       NNODES)) ARE PROCESSED IN GROUPS OF UP TO 64.  EACH VERTEX
#       CARRIES A 64-BIT MASK OF THE GROUP MEMBERS IT IS ADJACENT TO,
#       SO THE NEIGHBORHOOD OF EACH FILL PARTNER WNODE IS SCANNED ONCE
#       PER GROUP RATHER THAN ONCE PER FILL EDGE.  THE DEFICIENCIES ARE
#       EXACTLY THOSE OF THE PER-EDGE WING-HUANG UPDATE; ONLY THE ORDER
#       IN WHICH CHANGED EXTERNAL NODES ARE RECORDED (AND HENCE TIE-
#       BREAKING IN THE HEAP) MAY DIFFER.  WITH GROUPSIZE = 1 THE
#       ORIGINAL VISITING ORDER IS REPRODUCED EXACTLY.
#
#     INPUT PARAMETERS:
#       ENODE       - ELIMINATION NODE.
#       NEQNS       - NUMBER OF EQUATIONS.
#       MAXINT      - MAXIMUM INTEGER USED IN MARKER(*).
#       ADJLEN      - LENGTH OF ADJNCY(*).
#       XADJ(*)     - ARRAY OF LENGTH NEQNS+1, CONTAINING THE
#                     POINTERS FOR THE CURRENT QUOTIENT GRAPH.
#       ADJNCY(*)   - ARRAY OF LENGTH ADJLEN, CONTAINING THE CURRENT
#                     QUOTIENT GRAPH.
#       NVTXS(*)    - ARRAY OF LENGTH NEQNS, CONTAINING THE NUMBER
#                     VERTEX NEIGHBORS FOR AN UNELIMINATED NODE, AND
#                     THE NUMBER OF NODES IN THE CLIQUE FOR AN
#                     ELIMINATED NODE.
#       WORK(*)     - ARRAY OF LENGTH NEQNS, CONTAINING THE NUMBER
#                     OF ELIMINATION CLIQUES TO WHICH AN UNELIMINATED
#                     NODE BELONGS.
#       QSIZE(*)    - ARRAY OF LENGTH NEQNS,
#                       <0 - NEGATIVE OF SUPERNODE SIZE FOR EACH NODE
#                            IN ENODE'S ELIMINATION CLIQUE.
#                       >0 - SUPERNODE SIZE FOR EACH UNELIMINATED 
#                            NODE NOT IN ENODE'S ELIMINATION 
#                            CLIQUE.
#                       =0 - ABSORBED.
#       NNODES      - NUMBER OF (SUPER)NODES IN ENODE'S ELIMINATION
#                     CLIQUE.
#       FNODE       - POINTER TO FIRST (SUPER)NODE IN ENODE'S
#                     ELIMINATION CLIQUE MERGED INTO THE CLIQUE
#                     AFTER THE (SUPER)NODES OF QUOTIENT GRAPH
#                     NEIGHBOR CLIQUE "LAST CLIQUE".
#
#     MODIFIED PARAMETERS:
#       NNODES2     - INITIALLY ZERO.  ON OUTPUT, IT IS THE NUMBER
#                     OF EXTERNAL NODES WHOSE DEFICIENCIES HAVE BEEN
#                     CHANGED BY THIS ROUTINE.
#       ECLIQ(*)    - ARRAY OF LENGTH NEQNS.
#                     ON INPUT CONTAINS ENODE'S ELIMINATION CLIQUE
#                     AND MAY OR MAY NOT CONTAIN THE EXTERNAL NODES
#                     WHOSE DEFICIENCIES WILL CHANGE.
#                     ON OUTPUT, CONTAINS ENODE'S ELIMINATION CLIQUE
#                     AND THE EXTERNAL NODES WHOSE DEFICIENCIES HAVE
#                     CHANGED.
#       DEGREE(*)   - ARRAY OF LENGTH NEQNS, RECORDS THE DEGREE
#                     OF EACH NODE UNODE IN THE CURRENT ELIMINATION
#                     GRAPH PLUS ONE.
#       DEFNCY(*)   - ARRAY OF LENGTH NEQNS, CONTAINING THE 
#                     DEFICIENCIES OF THE UNELIMINATED NODES.
#                     SINCE DEFICIENCIES CAN BE VERY LARGE, THIS
#                     ARRAY IS STORED AS LONG INTEGER (INTEGER*8).
#       TAG         - TAG FOR MARKING PROCESSES IN MARKER(*).
#       MARKER(*)   - ARRAY OF LENGTH NEQNS, USED FOR MARKING TO 
#                     PREVENT PROCESSING SAME NODE MORE THAN ONCE
#                     OR PROCESSING AN ABSORBED NODE.
#       CHANGED(*)  - ARRAY OF LENGTH NEQNS,
#                       -1 - DEFICIENCY IS UNCHANGED
#                        1 - DEFICIENCY IS POTENTIALLY CHANGED
#
#   WORKING PARAMETERS:
#       DEGINC(*)   - ARRAY OF LENGTH NEQNS, USED TO ACCUMULATE
#                     THE CHANGE IN DEGREES OF THE NODES IN THE
#                     ELIMINATION CLIQUE.
#       UMASK(*)    - ARRAY OF LENGTH NEQNS, ALL ZERO ON INPUT AND
#                     OUTPUT.  BIT G-1 OF UMASK(X) IS SET WHEN X IS
#                     A NEIGHBOR OF THE G-TH MEMBER OF THE CURRENT
#                     GROUP.
#       TOUCH(*)    - ARRAY OF LENGTH NEQNS, LISTS THE NODES WHOSE
#                     UMASK IS NONZERO (SO THEY CAN BE CLEARED).
#       GQ(*)       - ARRAY OF LENGTH 64, QSIZE OF EACH GROUP MEMBER.
#       GUCNT(*)    - ARRAY OF LENGTH 64, EXTERNAL WEIGHT OF EACH
#                     GROUP MEMBER'S NEIGHBORHOOD.
#       GPQ(*)      - ARRAY OF LENGTH 64, SUM OF QSIZE OVER EACH
#                     GROUP MEMBER'S FILL PARTNERS.
#       GACC(*)     - ARRAY OF LENGTH 64, SUM OVER EACH GROUP
#                     MEMBER'S FILL PARTNERS WNODE AND THEIR COMMON
#                     EXTERNAL NEIGHBORS XNODE OF QSIZE(WNODE) *
#                     QSIZE(XNODE).
#
#**********************************************************************
#

# sum of the (negative) supernode sizes gq(g) of the group members g in mask m;
# heavy marks the members whose size is not one
@inline function grp_sum(m::GMASK, heavy::GMASK, gq::AbstractVector{W}) where {W <: Integer}
    # count every member as size one, then correct the few heavy ones
    s = -convert(W, count_ones(m))
    h = m & heavy

    @inbounds while !iszero(h)
        s += gq[trailing_zeros(h) + 1] + one(W)
        h &= h - one(GMASK)
    end

    return s
end

@inline function grp_sum(m::GMASK, heavy::GMASK, gq::AbstractVector{W}) where {W}
    iszero(m & heavy) && return -convert(W, count_ones(m))
    s = zero(W)

    @inbounds while !iszero(m)
        s += gq[trailing_zeros(m) + 1]
        m &= m - one(GMASK)
    end

    return s
end

function mfupd_def(
        enode::V,
        neqns::V,
        adjlen::E,
        xadj::AbstractVector{E},
        adjncy::AbstractVector{V},
        nvtxs::AbstractVector{V},
        work::AbstractVector{V},
        qsize::AbstractVector{W},
        nnodes::V,
        fnode::V,
        ecliq::AbstractVector{V},
        degree::AbstractVector{W},
        defncy::AbstractVector{W},
        tag::I,
        marker::AbstractVector{I},
        changed::AbstractVector{Bool},
        deginc::AbstractVector{W},
        umask::AbstractVector{GMASK},
        touch::AbstractVector{V},
        gq::AbstractVector{W},
        gucnt::AbstractVector{W},
        gpq::AbstractVector{W},
        gacc::AbstractVector{W},
    ) where {V, E, I, W}

    #       -------------------
    #       LOCAL VARIABLES ...
    #       -------------------

    #       ------------------------------------------------------------
    #       INITIALIZE THE CHANGES IN THE DEGREES TO ZERO.
    #       K: LAST OCCUPIED LOCATION IN THE ELIMINATION CLIQUE (ECLIQ).
    #       ------------------------------------------------------------
    @inbounds for j in oneto(nnodes)
        unode = ecliq[j]
        deginc[unode] = zero(W)
    end

    k = nnodes

    j = fnode

    @inbounds while j <= nnodes
        jhi = min(nnodes, j + convert(V, GROUPSIZE) - one(V))
        ng = Int(jhi - j) + 1
        allbits = ng == GROUPSIZE ? typemax(GMASK) : (one(GMASK) << ng) - one(GMASK)
        heavy = zero(GMASK)
        ntouch = 0

        #           -----------------------------------------
        #           MARK THE NEIGHBORHOODS OF THE GROUP MEMBERS.
        #           -----------------------------------------
        for g in 1:ng
            unode = ecliq[j + convert(V, g - 1)]
            bit = one(GMASK) << (g - 1)
            gq[g] = qsize[unode]
            qsize[unode] == -one(W) || (heavy |= bit)
            ucount = zero(W)
            istart = xadj[unode]
            istop = istart + convert(E, nvtxs[unode]) - one(E)

            for i in istart:istop
                wnode = adjncy[i]

                if marker[wnode] < maxint(I)
                    mw = umask[wnode]

                    if iszero(mw & bit)
                        iszero(mw) && (ntouch += 1; touch[ntouch] = wnode)
                        umask[wnode] = mw | bit
                        ispositive(qsize[wnode]) && (ucount += qsize[wnode])
                    end
                end
            end

            cstart = xadj[unode] + convert(E, nvtxs[unode])
            cstop = cstart + convert(E, work[unode]) - one(E)

            for c in cstart:cstop
                cnode = adjncy[c]
                istart = xadj[cnode]
                istop = istart + convert(E, nvtxs[cnode]) - one(E)

                for i in istart:istop
                    wnode = adjncy[i]

                    if marker[wnode] < maxint(I)
                        mw = umask[wnode]

                        if iszero(mw & bit)
                            iszero(mw) && (ntouch += 1; touch[ntouch] = wnode)
                            umask[wnode] = mw | bit
                            ispositive(qsize[wnode]) && (ucount += qsize[wnode])
                        end
                    end
                end
            end

            gucnt[g] = ucount
            gpq[g] = zero(W)
            gacc[g] = zero(W)
        end

        #           ---------------------------------------------------
        #           FOR EVERY EARLIER CLIQUE MEMBER WNODE = ECLIQ(I) ...
        #           PM: GROUP MEMBERS AFTER WNODE THAT ARE NOT ADJACENT
        #           TO IT, I.E. WNODE'S FILL PARTNERS IN THIS GROUP.
        #           ---------------------------------------------------
        for i in oneto(jhi - one(V))
            g0 = Int(i) - Int(j) + 2

            after = if g0 <= 1
                allbits
            elseif g0 > ng
                zero(GMASK)
            else
                allbits & ~((one(GMASK) << (g0 - 1)) - one(GMASK))
            end

            wnode = ecliq[i]
            pm = after & ~umask[wnode]
            iszero(pm) && continue

            qw = qsize[wnode]
            sqpm = grp_sum(pm, heavy, gq)

            #               DEGREE INCREMENTS DUE TO THE NEW FILL EDGES.
            deginc[wnode] -= sqpm
            bits = pm

            while !iszero(bits)
                g = trailing_zeros(bits) + 1
                deginc[ecliq[j + convert(V, g - 1)]] -= qw
                gpq[g] += qw
                bits &= bits - one(GMASK)
            end

            #               BUMP TAG (USED TO DEDUPLICATE WNODE'S NEIGHBORHOOD).
            if tag <= maxint(I) - two(I)
                tag += one(I)
            else
                tag = one(I)

                for t in oneto(neqns)
                    if marker[t] < maxint(I)
                        marker[t] = zero(I)
                    end
                end
            end

            qext = zero(W)
            bw = zero(W)
            istart = xadj[wnode]
            istop = istart + convert(E, nvtxs[wnode]) - one(E)

            for i2 in istart:istop
                xnode = adjncy[i2]

                if marker[xnode] < tag
                    marker[xnode] = tag
                    qx = qsize[xnode]
                    m = umask[xnode] & pm

                    if ispositive(qx)
                        qext += qx

                        if !iszero(m)
                            sq = grp_sum(m, heavy, gq)
                            defncy[xnode] -= qw * sq
                            bw += qx * sq
                            bits = m

                            while !iszero(bits)
                                g = trailing_zeros(bits) + 1
                                gacc[g] += qw * qx
                                bits &= bits - one(GMASK)
                            end

                            if !changed[xnode]
                                k += one(V)
                                ecliq[k] = xnode
                                changed[xnode] = true
                            end
                        end
                    elseif !iszero(m)
                        defncy[xnode] -= qw * grp_sum(m, heavy, gq)
                    end
                end
            end

            cstart = xadj[wnode] + convert(E, nvtxs[wnode])
            cstop = cstart + convert(E, work[wnode]) - one(E)

            for c in cstart:cstop
                cnode = adjncy[c]
                istart = xadj[cnode]
                istop = istart + convert(E, nvtxs[cnode]) - one(E)

                for i2 in istart:istop
                    xnode = adjncy[i2]

                    if marker[xnode] < tag
                        marker[xnode] = tag
                        qx = qsize[xnode]
                        m = umask[xnode] & pm

                        if ispositive(qx)
                            qext += qx

                            if !iszero(m)
                                sq = grp_sum(m, heavy, gq)
                                defncy[xnode] -= qw * sq
                                bw += qx * sq
                                bits = m

                                while !iszero(bits)
                                    g = trailing_zeros(bits) + 1
                                    gacc[g] += qw * qx
                                    bits &= bits - one(GMASK)
                                end

                                if !changed[xnode]
                                    k += one(V)
                                    ecliq[k] = xnode
                                    changed[xnode] = true
                                end
                            end
                        elseif !iszero(m)
                            defncy[xnode] -= qw * grp_sum(m, heavy, gq)
                        end
                    end
                end
            end

            #               WNODE GAINS NEW MISSING PAIRS: ITS EXTERNAL
            #               NEIGHBORS THAT ARE NOT ADJACENT TO EACH PARTNER.
            defncy[wnode] -= qext * sqpm - bw
        end

        #           EACH UNODE GAINS NEW MISSING PAIRS: ITS EXTERNAL
        #           NEIGHBORS THAT ARE NOT ADJACENT TO EACH PARTNER.
        for g in 1:ng
            unode = ecliq[j + convert(V, g - 1)]
            defncy[unode] -= gucnt[g] * gpq[g] - gacc[g]
        end

        for t in 1:ntouch
            umask[touch[t]] = zero(GMASK)
        end

        j = jhi + one(V)
    end

    #       -------------------------------------------
    #       APPLY THE DEGREE INCREMENTS TO THE DEGREES.
    #       -------------------------------------------
    @inbounds for j in oneto(nnodes)
        unode = ecliq[j]
        degree[unode] += deginc[unode]
    end

    #       --------------------------------------------
    #       RECORD THE NUMBER OF EXTERNAL VERTICES WHOSE
    #       DEFICIENCIES HAVE BEEN UPDATED.
    #       --------------------------------------------
    nnodes2 = k - nnodes

    #       ------------------------------------------------------
    #       CHECK IF THE DEFICIENCY OF THE ELIMINATED NODE ENODE
    #       IS ZERO.  STOP IF IT IS NOT. 
    #       ------------------------------------------------------
    @inbounds if abs(defncy[enode]) > 1e-5
        error("ELIMINATED NODE HAS NONZERO DEFICIENCY: $enode => $(defncy[enode])")
    end

    return tag, nnodes2
end
