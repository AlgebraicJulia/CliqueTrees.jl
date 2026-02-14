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
#
#**********************************************************************
#
#     PURPOSE:
#       THIS SUBROUTINE USES WING-HUANG UPDATING TO UPDATE THE
#       DEFICIENCIES OF ALL VERTICES WHOSE DEFICIENCIES ARE INFLUENCED
#       BY THE ELIMINATION OF ENODE.
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
#       UTAG        - USED TO TAG THE NEIGHBORS OF UNODE FOR
#                     THE WING-HUANG UPDATING PROCESS.
#       UMARK(*)    - ARRAY OF LENGTH NEQNS, USED TO MARK THE
#                     NEIGHBORS OF UNODE FOR THE W-H UPDATING
#                     PROCESS.
#       CHANGED(*)  - ARRAY OF LENGTH NEQNS,
#                       -1 - DEFICIENCY IS UNCHANGED
#                        1 - DEFICIENCY IS POTENTIALLY CHANGED
#
#   WORKING PARAMETERS:
#       DSET(*)     - ARRAY OF LENGTH NEQNS, USED TO STORE THE 
#                     BACK-EDGES OF THE DEFICIENCY OF ENODE,
#                     INCIDENT, IN TURN, UPON EACH MEMBER UNODE
#                     OF ENODE'S ELIMINATION CLIQUE.
#       DEGINC(*)   - ARRAY OF LENGTH NEQNS, USED TO ACCUMULATE
#                     THE CHANGE IN DEGREES OF THE NODES IN THE
#                     ELIMINATION CLIQUE.
#
#**********************************************************************
#
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
        utag::I,
        umark::AbstractVector{I},
        changed::AbstractVector{Bool},
        dset::AbstractVector{V},
        deginc::AbstractVector{W},
    ) where {V, E, I, W}
    
    #       -------------------
    #       LOCAL VARIABLES ...
    #       -------------------
    
    #       ------------------------------------------------------------
    #       INITIALIZE THE CHANGES IN THE DEGREES TO ZERO.
    #       K: LAST OCCUPIED LOCATION IN THE ELIMINATION CLIQUE (ECLIQ).
    #       ------------------------------------------------------------
    for j in oneto(nnodes)
        unode = ecliq[j]
        deginc[unode] = zero(W)
    end
    
    k = nnodes
    
    #       -----------------------------------------------------
    #       FOR EACH NODE UNODE IN ENODE'S ELIMINATION CLIQUE ...
    #       (STARTING FIRST WITH NODE FNODE)
    #       -----------------------------------------------------
    for j in fnode:nnodes
        
        #           ----------
        #           GET UNODE.
        #           ----------
        unode = ecliq[j]
        qu = abs(qsize[unode])
        
        #           ---------
        #           BUMP TAG.
        #           ---------
        if tag <= maxint(I) - convert(I, j) - one(I)
            tag0 = tag
            tag += convert(I, j)
        else
            tag0 = zero(I)
            tag = convert(I, j)

            for i in oneto(neqns)
                if marker[i] < maxint(I)
                    marker[i] = zero(I)
                end
            end
        end

        #           ----------
        #           BUMP UTAG.
        #           ----------
        if utag <= maxint(I) - two(I)
            utag += one(I)
        else
            utag = one(I)

            for i in oneto(neqns)
                if marker[i] < maxint(I)
                    umark[i] = zero(I)
                end
            end
        end

        #           ------------------------------------------------
        #           INITIALIZE NUMBER OF EXTERNAL NEIGHBORS OF UNODE
        #           TO ZERO.
        #           ------------------------------------------------
        ucount = zero(W)
        #           ------------------------------------------
        #           FOR EVERY UNABSORBED VERTEX NEIGHBOR WNODE
        #           OF UNODE ...
        #           ------------------------------------------
        istart = xadj[unode]
        istop = istart + convert(E, nvtxs[unode]) - one(E)

        for i in istart:istop
            wnode = adjncy[i]
            #               --------------------------------------------
            #               IF WNODE IS A FIRST-TIME UNABSORBED NEIGHBOR
            #               OF UNODE ...
            #               --------------------------------------------
            if umark[wnode] < utag
                #                   --------------------------------------
                #                   IF WNODE IS NOT IN ENODE'S ELIMINATION
                #                   CLIQUE ...
                #                   --------------------------------------
                if ispositive(qsize[wnode])
                    #                       ------------------------------------
                    #                       COUNT WNODE AS AN EXTERNAL NEIGHBOR.
                    #                       ------------------------------------
                    ucount += qsize[wnode]
                end
                #                   --------------------
                #                   ... THEN MARK WNODE.
                #                   --------------------
                umark[wnode] = utag
            end
        end
        #           --------------------------------------------
        #           FOR EVERY CLIQUE NEIGHBOR CNODE OF UNODE ...
        #           --------------------------------------------
        cstart = xadj[unode] + convert(E, nvtxs[unode])
        cstop = cstart + convert(E, work[unode]) - one(E)

        for c in cstart:cstop
            cnode = adjncy[c]
            #               ----------------------------------------
            #               FOR EVERY NODE WNODE IN CLIQUE CNODE ...
            #               ----------------------------------------
            istart = xadj[cnode]
            istop = istart + convert(E, nvtxs[cnode]) - one(E)

            for i in istart:istop
                wnode = adjncy[i]
                #                   -----------------------------------
                #                   IF WNODE IS A FIRST-TIME UNABSORBED
                #                   NEIGHBOR OF UNODE ...
                #                   -----------------------------------
                if umark[wnode] < utag
                    #                       --------------------------
                    #                       IF WNODE IS NOT IN ENODE'S
                    #                       ELIMINATION CLIQUE ...
                    #                       --------------------------
                    if ispositive(qsize[wnode])
                        #                           ------------------------------------
                        #                           COUNT WNODE AS AN EXTERNAL NEIGHBOR.
                        #                           ------------------------------------
                        ucount += qsize[wnode]
                    end
                    #                       --------------------
                    #                       ... THEN MARK WNODE.
                    #                       --------------------
                    umark[wnode] = utag
                end
            end
        end
        
        #           ----------------------------------
        #           FOR EVERY EARLIER WNODE IN ENODE'S
        #           ELIMINATION CLIQUE ...
        #           (NOTE THAT UNODE = ECLIQ(J))
        #           ----------------------------------
        dvtxs = zero(V)

        for i in oneto(j - one(V))
            wnode = ecliq[i]
            #               ---------------------------------------
            #               IF WNODE IS NOT A NEIGHBOR OF UNODE ...
            #               ---------------------------------------
            if umark[wnode] < utag
                #                   ---------------------------------------------
                #                   ... THEN THERE IS A NEW FILL EDGE JOINING
                #                   UNODE AND WNODE.
                #                   ACCUMULATE THE CONTRIBUTION OF THE FILL
                #                   EDGE TO UNODE'S AND WNODE'S DEGREE INCREMENT.
                #                   ---------------------------------------------
                deginc[unode] -= qsize[wnode]
                deginc[wnode] += qu
                dvtxs += one(V)
                dset[dvtxs] = wnode
            end
        end

        #           ---------------------------------------------
        #           FOR EACH NEW FILL NEIGHBOR WNODE OF UNODE ...
        #           ---------------------------------------------
        for w in oneto(dvtxs)

            #               --------------------------------------
            #               INITIALIZE COUNTS FOR UNODE AND WNODE.
            #               --------------------------------------
            wnode = dset[w]
            cntu = ucount
            cntw = zero(W)
            tag0 += one(I)
            uwfill = qsize[unode] * qsize[wnode]

            #               -------------------------------------------------------
            #               FOR EVERY UNABSORBED VERTEX NEIGHBOR XNODE OF WNODE ...
            #               -------------------------------------------------------
            istart = xadj[wnode]
            istop = istart + convert(E, nvtxs[wnode]) - one(E)

            for i in istart:istop
                xnode = adjncy[i]

                if marker[xnode] < tag0
                    #                       ------------------------------------
                    #                       FIRST TIME VISIT TO UNABSORBED XNODE
                    #                       AS A NEIGHBOR OF WNODE.
                    #                       ------------------------------------
                    if ispositive(qsize[xnode])
                        #                           --------------------------------
                        #                           XNODE IS NOT A MEMBER OF ENODE'S
                        #                           ELIMINATION CLIQUE.
                        #                           --------------------------------
                        if umark[xnode] == utag
                            #                               ----------------------------------
                            #                               XNODE IS ALSO A NEIGHBOR OF UNODE.
                            #                               ----------------------------------
                            
                            #                               ---------------------------
                            #                               DECREMENT UNODE'S COUNT AND
                            #                               DEFICIENCY OF XNODE.
                            #                               ---------------------------
                            cntu -= qsize[xnode]
                            defncy[xnode] -= uwfill
                            #                               ---------------------------------
                            #                               IF XNODE'S DEFICIENCY IS CHANGED
                            #                               FOR THE FIRST TIME, THEN STORE IT
                            #                               AND MARK IT AS CHANGED.
                            #                               ---------------------------------
                            if !changed[xnode]
                                k += one(V)
                                ecliq[k] = xnode
                                changed[xnode] = true
                            end
                        else
                            #                               ------------------------------------
                            #                               XNODE IS NOT A NEIGHBOR OF UNODE ...
                            #                               SO IT WILL INCREASE THE DEFICIENCY
                            #                               OF WNODE.
                            #                               ------------------------------------
                            cntw += qsize[xnode]
                        end
                    else
                        #                           ----------------------------
                        #                           XNODE IS A MEMBER OF ENODE'S
                        #                           ELIMINATION CLIQUE.
                        #                           ----------------------------
                        if umark[xnode] == utag
                            #                               ---------------------------------
                            #                               ... AND IT IS ADJACENT TO UNODE.
                            #                               SO DECREMENT DEFICIENCY OF XNODE.
                            #                               ---------------------------------
                            defncy[xnode] -= uwfill
                        end
                    end
                    #                       ---------------------------------------------
                    #                       MARK XNODE AS VISITED AS A NEIGHBOR OF WNODE.
                    #                       ---------------------------------------------
                    marker[xnode] = tag0
                end
            end

            #               --------------------------------------------
            #               FOR EVERY CLIQUE NEIGHBOR CNODE OF WNODE ...
            #               --------------------------------------------
            cstart = xadj[wnode] + convert(E, nvtxs[wnode])
            cstop = cstart + convert(E, work[wnode]) - one(E)

            for c in cstart:cstop
                cnode = adjncy[c]
                #                   ----------------------------------
                #                   FOR EVERY UNABSORBED NODE XNODE IN
                #                   CLIQUE CNODE ...
                #                   ----------------------------------
                istart = xadj[cnode]
                istop = istart + convert(E, nvtxs[cnode]) - one(E)

                for i in istart:istop
                    xnode = adjncy[i]

                    if marker[xnode] < tag0
                        #                           ------------------------------------
                        #                           FIRST TIME VISIT TO UNABSORBED XNODE
                        #                           AS A NEIGHBOR OF WNODE.
                        #                           ------------------------------------
                        if ispositive(qsize[xnode])
                            #                               --------------------------------
                            #                               XNODE IS NOT A MEMBER OF ENODE'S
                            #                               ELIMINATION CLIQUE.
                            #                               --------------------------------
                            if umark[xnode] == utag
                                #                                   ------------------------
                                #                                   XNODE IS ALSO A NEIGHBOR
                                #                                   OF UNODE.
                                #                                   ------------------------
                                #                                   ---------------------------
                                #                                   DECREMENT UNODE'S COUNT AND
                                #                                   DEFICIENCY OF XNODE.
                                #                                   ---------------------------
                                cntu -= qsize[xnode]
                                defncy[xnode] -= uwfill
                                #                                   ---------------------------------
                                #                                   IF XNODE'S DEFICIENCY IS CHANGED
                                #                                   FOR THE FIRST TIME, ADD IT TO THE
                                #                                   LIST OF NODES WITH CHANGED
                                #                                   DEFICIENCIES.
                                #                                   ---------------------------------
                                if !changed[xnode]
                                    k += one(V)
                                    ecliq[k] = xnode
                                    changed[xnode] = true
                                end
                            else
                                #                                   -----------------------
                                #                                   XNODE IS NOT A NEIGHBOR
                                #                                   OF UNODE ...
                                #                                   SO IT WILL INCREASE THE
                                #                                   DEFICIENCY OF WNODE.
                                #                                   -----------------------
                                cntw += qsize[xnode]
                            end
                        else
                            #                               ----------------------------
                            #                               XNODE IS A MEMBER OF ENODE'S
                            #                               ELIMINATION CLIQUE.
                            #                               ----------------------------
                            if umark[xnode] == utag
                                #                                   ---------------------------------
                                #                                   ... AND IT IS ADJACENT TO UNODE.
                                #                                   SO DECREMENT DEFICIENCY OF XNODE.
                                #                                   ---------------------------------
                                defncy[xnode] -= uwfill
                            end
                        end
                        #                           -----------------------------------
                        #                           MARK XNODE AS VISITED AS A NEIGHBOR
                        #                           OF WNODE.
                        #                           -----------------------------------
                        marker[xnode] = tag0
                    end
                end
            end
            
            #               ---------------------------------------------
            #               WING-HUANG UPDATES FOR UNODE AND WNODE DUE TO
            #               FILL JOINING UNODE AND WNODE.
            #               NOTE: QSIZE IS NEGATIVE.
            #               ---------------------------------------------
            defncy[unode] -= cntu * qsize[wnode]
            defncy[wnode] -= cntw * qsize[unode]
            
            #               -----------
            #               NEXT WNODE.
            #               -----------
        end
        #           -----------
        #           NEXT UNODE.
        #           -----------
    end
    
    #       -------------------------------------------
    #       APPLY THE DEGREE INCREMENTS TO THE DEGREES.
    #       -------------------------------------------
    for j in oneto(nnodes)
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
    if abs(defncy[enode]) > 1e-5
        error("ELIMINATED NODE HAS NONZERO DEFICIENCY: $enode => $(defncy[enode])")
    end
    
    return tag, utag, nnodes2
end
