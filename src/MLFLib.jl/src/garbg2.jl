#**********************************************************************
#**********************************************************************
#*****   GARBG2 ..... GARBAGE COLLECTION   ****************************
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
#       THIS SUBROUTINE COMPRESSES THE ADJACENCY STRUCTURE USED BY
#       THE MINIMUM DEFICIENCY ROUTINES.  UPON COMPLETION, ALL
#       "ACTIVE" DATA IS STORED CONTIGUOUSLY IN THE FIRST PART OF
#       ADJNCY(*), WHILE UNUSED SPACE OCCURS AS A SINGLE BLOCK THAT
#       TRAILS THE ACTIVE PORTION OF THE DATA STRUCTURE.
#
#     INPUT PARAMETERS:
#       NEQNS       - NUMBER OF EQUATIONS.
#       ADJLEN      - LENGTH OF ADJNCY(*).
#       ECHEAD      - HEAD OF ELIMINATION CLIQUE LIST.
#       ECFORW(*)   - ARRAY OF SIZE NEQNS.
#                       ELIMINATED NOT IN PLACE: FORWARD LINKS OF
#                         A LIST OF THE ELIMINATION CLIQUES ORDERED
#                         BY THE ELIMINATION.
#                       OTHERWISE: -1.
#       NVTXS(*)    - ARRAY OF LENGTH NEQNS+1.
#                       ACTIVE ELIMINATED: NUMBER OF VERTICES IN THE
#                         ELIMINATION CLIQUE.
#                       MERGED ELIMINATED: -1.
#                       UNELIMINATED: NUMBER OF UNELIMINATED VERTEX
#                         NEIGHBORS IN CURRENT QUOTIENT GRAPH.
#                       ABSORBED: -1.
#       WORK(*)     - ARRAY OF LENGTH NEQNS.
#                       ACTIVE ELIMINATED: >=0; NOT USED.
#                       UNELIMINATED: NUMBER OF ELIMINATION
#                         CLIQUES VERTEX IS ADJACENT TO IN THE
#                         QUOTIENT GRAPH.
#                       ABSORBED: MAPS ABSORBED VERTEX TO MINUS
#                         THE ABSORBING VERTEX (I.E., -ABSORBEE).
#       INVP(*)     - ARRAY OF LENGTH NEQNS.
#                       ELIMINATED: -NUM, WHERE NUM IS THE NUMBER
#                         ASSIGNED BY THE ELIMINATION ORDER.
#
#     MODIFIED PARAMETERS:
#       XADJ(*)     - ARRAY OF LENGTH NEQNS+1.
#                     XADJ(JNODE) POINTS TO THE BEGINNING OF JNODE'S 
#                     NEIGHBORS IN ADJNCY(*).
#                     NOTE: XADJ(JNODE+1) DOES **NOT** GENERALLY
#                     POINT ONE LOCATION BEYOND THE END OF JNODE.
#                     TERMINATION SEES THE SAME FUNCTIONALITY, BUT
#                     NOW ALL UNUSED SPACE IN ADJNCY(*) HAS BEEN
#                     MOVED TO THE END.
#       ADJNCY(*)   - ARRAY OF LENGTH ADJLEN.
#                     ADJACENCY LISTS.  THE VERTEX NEIGHBORS OF 
#                     VERTEX JNODE ARE LOCATED IN ADJLEN(XADJ(JNODE)),
#                     ADJLEN(XADJ(JNODE)+1), ... ,
#                     ADJLEN(XADJ(JNODE)+NVTXS(JNODE)-1).
#
#     OUTPUT PARAMETER:
#       NXTLOC      - POINTS TO THE FIRST AVAILABLE LOCATION IN THE
#                     UNOCCUPIED BLOCK TO THE RIGHT OF THE ACTIVE 
#                     DATA STRUCTURE IN ADJNCY(*).  UPON TERMINATION, 
#                     ALL UNUSED SPACE IN ADJNCY(*) NOW OCCURS IN A 
#                     SINGLE BLOCK TO THE RIGHT OF THE ACTIVE DATA 
#                     STRUCTURE.
#
#**********************************************************************
#
function garbg2(
        neqns::Int,
        adjlen::Int,
        echead::Int,
        ecforw::AbstractVector{Int},
        nvtxs::AbstractVector{Int},
        work::AbstractVector{Int},
        invp::AbstractVector{Int},
        xadj::AbstractVector{Int},
        adjncy::AbstractVector{Int},
        nxtloc::Int
    )
    
    #       -------------------
    #       LOCAL VARIABLES ...
    #       -------------------
    
    #       ---------------
    #       INITIALIZATION.
    #       ---------------
    #       PRINT *,' '
    #       PRINT *,'ENTER GARBAGE COLLECTION'
    fstloc = 1
    nxtlc2 = fstloc
    
    #       -----------------------------------------
    #       FOR THE NEXT VERTEX NXTNOD (IN ORDER) ...
    #       -----------------------------------------
    for nxtnod in oneto(neqns)
        #           -------------------------------------
        #           ... IF NXTNOD IS AN ACTIVE VERTEX ...
        #               I.E. UNMERGED AND UNABSORBED
        #           -------------------------------------
        if !isnegative(nvtxs[nxtnod])
            #               ---------------------------------
            #               ... IF NXTNOD IS UNELIMINATED ...
            #               ---------------------------------
            if !isnegative(invp[nxtnod])
                #                   --------------------------------
                #                   ... THEN COPY NXTNOD'S LIST INTO
                #                       ITS NEW LOCATION.
                #                   --------------------------------
                #                   PRINT *,'NXTNOD,NVTXS,WORK:',NXTNOD,NVTXS(NXTNOD),
                #    &                      WORK(NXTNOD)
                jstart = xadj[nxtnod]
                jstop = jstart + nvtxs[nxtnod] + work[nxtnod] - 1
                nxtlc2 = fstloc

                for j in jstart:jstop
                    #                       PRINT *,'JNODE, NXTLC2, J:',ADJNCY(NXTLC2),
                    #    &                          NXTLC2, J
                    adjncy[nxtlc2] = adjncy[j]
                    nxtlc2 += 1
                end
                #                   ----------------------------
                #                   ... AND RECORD NEW LOCATION.
                #                   ----------------------------
                xadj[nxtnod] = fstloc
                fstloc = nxtlc2
            end
        end
    end
    
    #       --------------------------------------------------------
    #       FOR THE NEXT VERTEX (NXTNOD) ELIMINATED NOT IN PLACE ...
    #       (IN ORDER BY ELIMINATION)
    #       --------------------------------------------------------
    nxtnod = echead

    while !iszero(nxtnod)
        #           PRINT *,'ELIMINATION CLIQUES'
        #           PRINT *,'NXTNOD,NVTXS     :',NXTNOD,NVTXS(NXTNOD)
        #           --------------------------------------
        #           IF NXTNOD REMAINS AN ACTIVE (UNMERGED)
        #           ELIMINATION CLIQUE ...
        #           --------------------------------------
        if !isnegative(nvtxs[nxtnod])
            #               --------------------------------------------------
            #               ... THEN COPY NXTNOD'S LIST INTO ITS NEW LOCATION.
            #               --------------------------------------------------
            jstart = xadj[nxtnod]
            jstop = jstart + nvtxs[nxtnod] - 1
            nxtlc2 = fstloc

            for j in jstart:jstop
                #                   PRINT *,'JNODE, NXTLC2, J:',ADJNCY(NXTLC2),NXTLC2, J
                adjncy[nxtlc2] = adjncy[j]
                nxtlc2 += 1
            end
            #               ----------------------------
            #               ... AND RECORD NEW LOCATION.
            #               ----------------------------
            xadj[nxtnod] = fstloc
            fstloc = nxtlc2
        end
        #           ----------------
        #           GET NEXT CLIQUE.
        #           ----------------
        nxtnod = ecforw[nxtnod]
    end
    
    #       ---------------------------------------
    #       ... AND RECORD NEXT AVAILABLE POSITION.
    #       ---------------------------------------
    nxtloc = nxtlc2
    
    return nxtloc
end
