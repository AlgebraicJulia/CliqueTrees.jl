#**********************************************************************
#**********************************************************************
#*****   MFINIT_DEF ... W-H INITIAL DEFICIENCIES   ********************
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
#       THIS SUBROUTINE COMPUTES INITIAL DEFICIENCIES.  IT INITIALIZES
#       AN EMPTY GRAPH, AND THEN IT ADDS THE EDGES OF THE COMPRESSED
#       GRAPH TO THE GROWING GRAPH, PERFORMING WING-HUANG UPDATES AS
#       IT GOES.
#
#     INPUT PARAMETERS:
#       NEQNS       - NUMBER OF EQUATIONS.
#       ADJLEN      - LENGTH OF ADJNCY(*).
#       ADJLEN2     - LENGTH OF ADJ2(*).
#       XADJ(*)     - ARRAY OF LENGTH NEQNS+1, CONTAINING POINTERS
#                     TO THE ADJACENCY STRUCTURE OF THE COMPRESSED
#                     GRAPH.
#       ADJNCY(*)   - ARRAY OF LENGTH ADJLEN, CONTAINING THE
#                     ADACENCY STRUCTURE OF THE COMPRESSED GRAPH.
#       DEGREE(*)   - ARRAY OF LENGTH NEQNS, CONTAINING THE DEGREE
#                     OF EACH REPRESENTATIVE NODE IN THE ORIGINAL 
#                     GRAPH, PLUS ONE.
#       QSIZE(*)    - SUPERNODE SIZE:
#                        0 -  ABSORBED VERTEX
#                       >0 -  SUPERNODE SIZE OF A REPRESENTATIVE
#                             VERTEX
#
#     OUTPUT PARAMETERS:
#       DEFNCY(*)   - ARRAY OF LENGTH NEQNS, CONTAINING THE EXACT 
#                     DEFICIENCY OF EACH REPRESENTATIVE NODE IN THE
#                     INITIAL COMPRESSED GRAPH.  SINCE A DEFICIENCY 
#                     VALUE CAN BE VERY LARGE, THIS ARRAY IS OF
#                     TYPE LONG INTEGER (INTEGER*8).
#       
#     WORKING PARAMETERS:
#       MARKER(*)   - ARRAY OF LENGTH NEQNS, USED FOR A MARKING
#                     PROCESS IN THIS ROUTINE.
#       DHEAD(*)    - ARRAY OF LENGTH NEQNS, HEADS OF THE DEGREE 
#                     LISTS.
#       DFORW(*)    - ARRAY OF LENGTH NEQNS, FORWARD LINKS FOR THE 
#                     SINGLY-LINKED DEGREE LISTS.
#       DEGTMP(*)   - ARRAY OF LENGTH NEQNS, MAINTAINS THE DEGREE
#                     IN THE DYNAMICALLY GROWING GRAPH WHERE EDGES
#                     ARE ADDED FOR THE REQUIRED WH UPDATES.
#       XADJ2(*)    - ARRAY OF LENGTH NEQNS, CONTAINS THE POINTERS
#                     TO THE TO GROWING ADJACENCY LISTS.
#       LEN2(*)     - ARRAY OF LENGTH NEQNS, CONTAINS THE LENGTHS
#                     OF THE GROWING ADJACENCY LISTS.
#       ADJ2(*)     - ARRAY OF LENGTH ADJLEN2, CONTAINS THE GROWING
#                     ADJACENCY LISTS.
#
#**********************************************************************
#
function mfinit_def(
        neqns::V,
        adjlen::E,
        adjlen2::E,
        xadj::AbstractVector{E},
        adjncy::AbstractVector{V},
        degree::AbstractVector{W},
        qsize::AbstractVector{W},
        defncy::AbstractVector{W},
        marker::AbstractVector{I},
        dhead::AbstractVector{V},
        dforw::AbstractVector{V},
        degtmp::AbstractVector{W},
        xadj2::AbstractVector{E},
        len2::AbstractVector{V},
        adj2::AbstractVector{V}
    ) where {V, E, I, W}
    
    #       -------------------
    #       LOCAL VARIABLES ...
    #       -------------------
    
    #       -----------------------------------------
    #       INITIALIZE MARKER VECTOR AND EMPTY LISTS.
    #       -----------------------------------------
    for jnode in oneto(neqns)
        dhead[jnode] = zero(V)

        if !iszero(qsize[jnode])
            marker[jnode] = zero(I)
        end
    end

    #       --------------------------------------------------
    #       INITIALIZE ZERO DEFICIENCY SCORES AND EMPTY GRAPH.
    #       --------------------------------------------------
    tag = zero(I)

    for jnode in oneto(neqns)
        if !iszero(qsize[jnode])
            xadj2[jnode] = xadj[jnode]
            len2[jnode] = zero(V)
            defncy[jnode] = zero(W)
            degtmp[jnode] = zero(W)
        end
    end

    #       -------------------------
    #       FOR EACH VERTEX JNODE ...
    #       -------------------------
    maxdeg = zero(V)

    for jnode = oneto(neqns)
        #           ---------------------------------------
        #           IF JNODE IS A REPRESENTATIVE VERTEX ...
        #           ---------------------------------------
        if !iszero(qsize[jnode])
            #               ---------------------------------------------
            #               COMPUTE JNODE'S NUMBER OF NEIGHBORS IN THE
            #               COMPRESSED GRAPH AND PLACE IT IN DEGREE LIST.
            #               ---------------------------------------------
            degre = convert(V, xadj[jnode + one(V)] - xadj[jnode])
            maxdeg = max(maxdeg, degre)
            degp1 = degre + one(V)
            nxtnod = dhead[degp1]
            dforw[jnode] = nxtnod
            dhead[degp1] = jnode
        end
    end

    #       *********************************************************
    #       BUILD THE GRAPH FROM SCRATCH WHILE COMPUTING DEFICIENCIES
    #       USING WING-HUANG UPDATING.
    #       *********************************************************

    #       -------------------------------------------------------
    #       DO WHILE THERE ARE VERTICES WHOSE EDGES ARE YET TO BE
    #       ADDED TO THE ADJACENCY STRUCTURE UNDER CONSTRUCTION ...
    #       -------------------------------------------------------
    while ispositive(maxdeg)

        #           ----------------------------------------------
        #           GET VERTEX UNODE OF MAXIMUM DEGREE WHOSE EDGES
        #           HAVE NOT YET BEEN ADDED TO THE NEW GRAPH.
        #           ----------------------------------------------
        degp1 = maxdeg + one(V)
        unode = dhead[degp1]

        if iszero(unode)
            maxdeg -= one(V)
        else
            
            #               ----------------------------------
            #               REMOVE UNODE FROM ITS DEGREE LIST.
            #               ----------------------------------
            nxtnod = dforw[unode]
            dhead[degp1] = nxtnod
            
            #               ------------------------------------------
            #               MARK THE NEIGHBORS OF UNODE IN THE CURRENT
            #               TRANSIENT GRAPH.
            #               ------------------------------------------
            tag += one(I)

            for w in xadj2[unode]:xadj2[unode] + convert(E, len2[unode]) - one(E)
                wnode = adj2[w]
                marker[wnode] = tag
            end

            #               ----------------------------------------------------
            #               FOR EACH NEIGHBOR OF UNODE TO BE ADDED AS A NEIGHBOR
            #               IN THE NEW GRAPH ...
            #               ----------------------------------------------------
            for w in xadj[unode]:xadj[unode + one(V)] - one(E)
                wnode = adjncy[w]

                if marker[wnode] < tag

                    #                       ----------------------------------------
                    #                       INITIALIZE COUNTS USED FOR W-H UPDATING.
                    #                       ----------------------------------------
                    ucount = degtmp[unode]
                    cntu = ucount
                    cntw = zero(W)
                    uwfill = qsize[unode] * qsize[wnode]

                    #                       --------------------------------------------
                    #                       PERFORM WING-HUANG UPDATES FOR THE NEW EDGE.
                    #                       --------------------------------------------
                    for x in xadj2[wnode]:xadj2[wnode] + convert(E, len2[wnode]) - one(E)
                        xnode = adj2[x]

                        if marker[xnode] == tag
                            defncy[xnode] -= uwfill
                            cntu -= qsize[xnode]
                        else
                            cntw += qsize[xnode]
                        end
                    end

                    defncy[unode] += cntu * qsize[wnode]
                    defncy[wnode] += cntw * qsize[unode]

                    #                       ---------------------------------------
                    #                       ADD THE EDGE JOINING UNODE AND WNODE TO
                    #                       THE GRAPH (AND UPDATE DEGREES IN THE
                    #                       TRANSIENT GRAPH).
                    #                       ---------------------------------------
                    adj2[xadj2[unode] + convert(E, len2[unode])] = wnode
                    len2[unode] += one(V)

                    adj2[xadj2[wnode] + convert(E, len2[wnode])] = unode
                    len2[wnode] += one(V)
                    
                    degtmp[wnode] += qsize[unode]
                    degtmp[unode] += qsize[wnode]
                    
                    #                       --------------------------------------------
                    #                       MARK WNODE AS A NEW NEIGHBOR OF UNODE IN G'.
                    #                       --------------------------------------------
                    marker[wnode] = tag
                    
                end
                #                   -----------
                #                   NEXT WNODE.
                #                   -----------
            end
        end
        #           -----------
        #           NEXT UNODE.
        #           -----------
    end
    
    return
end
