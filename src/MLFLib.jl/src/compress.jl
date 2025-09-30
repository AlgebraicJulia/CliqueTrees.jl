#**********************************************************************
#**********************************************************************
#*****   COMPRESS  ..... COMPRESS INITIAL GRAPH   *********************
#**********************************************************************
#**********************************************************************
#
#     AUTHORS:
#       ESMOND G. NG, LAWRENCE BERKELEY NATIONAL LABORAOTY
#       BARRY W. PEYTON, DALTON STATE COLLEGE
#
#     LAST UPDATED:
#       2025-09-22
#
#**********************************************************************
#
#     PURPOSE:
#       THIS ROUTINE COMPRESSES A GRAPH BY REPRESENTING EACH SET OF
#       INDISTINGUISHABLE NODES AS A SINGLE NODE.  THE UNCOMPRESSED
#       INPUT GRAPH IS WRITTEN OVER BY THE NEW COMPRESSED GRAPH.
#
#     INPUT PARAMETERS:
#       NEQNS       - NUMBER OF EQUATIONS.
#       MAXINT      - MAXIMUM INTEGER IN MARKING ARRAYS; USED HERE 
#                     TO SIGNIFY ABSORPTION.  (USING MARKER(*))
#       ADJLEN      - LENGTH OF ADJNCY(*).
#
#     MODIFIED PARAMETERS:
#       XADJ(*)     - ARRAY OF LENGTH NEQNS+1, CONTAINING INITIAL
#                     POINTERS TO THE ADJACENCY STRUCTURE.  ON 
#                     OUTPUT, CONTAINS POINTERS FOR THE COMPRESSED
#                     ADJACENCY STRUCTURE.
#       ADJNCY(*)   - ARRAY OF LENGTH ADJLEN, CONTAINING INITIAL
#                     ADJACENCY STRUCTURE.  ON OUTPUT, CONTAINS THE
#                     COMPRESSED ADJACENCY STRUCTURE.
#
#     OUTPUT PARAMETERS:
#       DEGREE(*)   - ARRAY OF LENGTH NEQNS, NUMBER OF NODES IN
#                     THE UNCOMPRESSED GRAPH ADJACENT TO THE NODE,
#                     PLUS ONE.  (I.E., |N(V) \CUP {V}|)
#       MARKER(*)   - ARRAY OF LENGTH NEQNS,
#                       0      -  REPRESENTATIVE NODE IN COMPRESSED
#                                 GRAPH.
#                       MAXINT -  ABSORBED NODE IN COMPRESSED GRAPH.
#       WORK(*)     - ARRAY OF LENGTH NEQNS,
#                       0      -  REPRESENTATIVE NODE IN COMPRESSED
#                                 GRAPH.
#                       <0     -  -JNODE, WHERE THE NODE IS ABSORBED
#                                 INTO REPRESENTATIVE NODE JNODE.
#       QSIZE(*)    - ARRAY OF LENGTH NEQNS,
#                       0      -  ABSORBED NODE IN COMPRESSED GRAPH.
#                       >0     -  NUMBER OF NODES IN THE SUPERNODE 
#                                 REPRESENTED BY THE NODE.
#
#     WORKING PARAMETERS:
#       HASH(*)     - ARRAY OF LENGTH NEQNS, USED TO CONTAIN HASH
#                     VALUES TO GROUP THE NODES TOGETHER.  THIS
#                     IS A LONG-INTEGER (I.E., INTEGER*8) ARRAY
#                     BECAUSE THE SUMS STORED IN THE HASH ARRAY
#                     MAY BE VERY LARGE.
#       UMARK(*)    - ARRAY OF LENGTH NEQNS, USED TO MARK
#                     NEIGHBORS OF NODES.
#       XADJ2(*)    - ARRAY OF LENGTH NEQNS, USED TO COMPUTE
#                     POINTERS TO NEW COMPRESSED ADJACENCY STRUCTURE.
#
#**********************************************************************
#
function compress(
        neqns::Int,
        maxint::Int,
        adjlen::Int,
        vwght::AbstractVector{W},
        xadj::AbstractVector{Int},
        adjncy::AbstractVector{Int},
        degree::AbstractVector{W},
        marker::AbstractVector{Int},
        work::AbstractVector{Int},
        qsize::AbstractVector{W},
        qnmbr::AbstractVector{Int},
        hash::AbstractVector{Int},
        umark::AbstractVector{Int},
        xadj2::AbstractVector{Int}
    ) where {W}
    
    #       -------------------
    #       LOCAL VARIABLES ...
    #       -------------------
    
    #       -----------------------------------------------------------
    #       INITIALIZE VARIOUS VECTORS.
    #       NOTE THAT DEGREE COUNTS THE VERTEX AS A NEIGHBOR OF ITSELF.
    #       -----------------------------------------------------------
    tag = 0

    for jnode in oneto(neqns)
        qsize[jnode] = vwght[jnode]
        qnmbr[jnode] = 1
        marker[jnode] = 0
        work[jnode] = 0
        umark[jnode] = 0
        hash[jnode] = jnode

        kstart = xadj[jnode]
        kstop = xadj[jnode + 1] - 1
        dj = vwght[jnode]

        for k in kstart:kstop
            knode = adjncy[k]
            dj += vwght[knode]
        end

        degree[jnode] = dj
    end
    
    #       ----------------------------------
    #       COMPUTE HASH SCORES FOR EACH NODE.
    #       ----------------------------------
    kstart = xadj[1]

    for jnode in oneto(neqns)
        kstop = xadj[jnode + 1] - 1

        for k in kstart:kstop
            knode = adjncy[k]
            hash[jnode] += knode
        end

        kstart = kstop + 1
    end
    
    #       -----------------------
    #       FOR EACH NODE JNODE ...
    #       -----------------------
    for jnode in oneto(neqns)
        
        #           -------------------------------------------------------
        #           IF JNODE IS NOT ALREADY ABSORBED IN A PREVIOUS NODE ...
        #           -------------------------------------------------------
        if !iszero(qsize[jnode])
            
            #               ----------------------------------------------
            #               MARK JNODE AND ITS NEIGHBORS WITH THE NEW TAG.
            #               ----------------------------------------------
            tag += 1
            umark[jnode] = tag
            kstart = xadj[jnode]
            kstop = xadj[jnode + 1] - 1

            for k in kstart:kstop
                knode = adjncy[k]
                umark[knode] = tag
            end
            
            #               ------------------------------------
            #               FOR EACH NEIGHBOR KNODE OF JNODE ...
            #               ------------------------------------
            kstart = xadj[jnode]
            kstop = xadj[jnode + 1] - 1

            for k in kstart:kstop
                knode = adjncy[k]
                #                   ----------------------------------------------
                #                   IF KNODE HAS THE SAME DEGREE AND HASH VALUE AS
                #                   JNODE AND KNODE IS GREATER THAN JNODE
                #                   ----------------------------------------------
                if degree[knode] == degree[jnode] && hash[knode] == hash[jnode] && knode > jnode
                    
                    #                       ----------------------------------------------
                    #                       CHECK IF KNODE AND JNODE ARE INDISINGUISHABLE.
                    #                       ----------------------------------------------
                    indist = true
                    istart = xadj[knode]
                    istop = xadj[knode + 1] - 1

                    for i in istart:istop
                        inode = adjncy[i]

                        if umark[inode] < tag
                            indist = false
                        end
                    end
                    #                       -----------------------------------------
                    #                       IF JNODE AND KNODE ARE INDISTINGUISHABLE,
                    #                       THEN
                    #                       -----------------------------------------
                    if indist
                        #                           ---------------------------
                        #                           KNODE IS MERGED INTO JNODE.
                        #                           ---------------------------
                        work[knode] = -jnode
                        qsize[jnode] += qsize[knode]
                        qnmbr[jnode] += qnmbr[knode]
                        marker[knode] = maxint
                        qsize[knode] = zero(W)
                        qnmbr[knode] = 0
                    end
                end
                #                   ----------------------------
                #                   NEXT NEIGHBOR KNODE OF JNODE
                #                   ----------------------------
            end
            
        end
        #           ----------------
        #           NEXT NODE JNODE.
        #           ----------------
    end
    
    #       ---------------------------------
    #       COMPRESS THE ADJACENCY STRUCTURE.
    #       ---------------------------------
    nxtloc = 1
    #       -----------------------
    #       FOR EACH NODE JNODE ...
    #       -----------------------
    for jnode in oneto(neqns)
        #           ----------------------------------------
        #           IF JNODE IS ABSORBED BY ANOTHER NODE ...
        #           ----------------------------------------
        if iszero(qsize[jnode])
            #               -------------------------------------------
            #               JNODE WILL HAVE AN EMPTY LIST OF NEIGHBORS.
            #               -------------------------------------------
            xadj2[jnode] = nxtloc
        else
            #               -------------------------------
            #               JNODE IS A REPRESENTATIVE NODE.
            #               COMPRESS JNODE'S NEIGHBOR LIST.
            #               -------------------------------
            kstart = xadj[jnode]
            kstop = xadj[jnode + 1] - 1

            for k in kstart:kstop
                knode = adjncy[k]

                if !iszero(qsize[knode])
                    adjncy[nxtloc] = knode
                    nxtloc += 1
                end
            end

            xadj2[jnode] = nxtloc
        end
    end
    
    #       ------------------
    #       COPY NEW POINTERS.
    #       ------------------
    xadj[1] = 1

    for k in oneto(neqns)
        xadj[k + 1] = xadj2[k]
    end
    
    return
end
