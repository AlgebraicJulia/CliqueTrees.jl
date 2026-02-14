#**********************************************************************
#**********************************************************************
#*****   MFINIT_VANILLA --- INITIAL DEFICIENCIES   ********************
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
#       THIS SUBROUTINE COMPUTES THE INITIAL DEFICIENCIES USING A
#       STRAIGHFORWARD APPROACH.  THE DEFICIENCIES ARE USED BY THE
#       CODE TO CHECK THE WING-HUANG INITIAL DEFICIENCES WHENEVER THE
#       CODE IS CHECKING FOR CORRECTNESS.  
#
#     INPUT PARAMETERS:
#       NEQNS       - NUMBER OF EQUATIONS.
#       ADJLEN      - LENGTH OF THE ADJNCY(*) ARRAY.
#       XADJ(*)     - ARRAY OF LENGTH NEQNS+1, CONTAINING THE    
#                     POINTERS TO THE ADJACENCY STRUCTURE OF THE
#                     COMPRESSED GRAPH.
#       ADJNCY(*)   - ARRAY OF LENGTH ADJLEN, CONTAINING THE 
#                     ADJACENCY STRUCTURE OF THE COMPRESSED GRAPH.
#       DEGREE(*)   - ARRAY OF LENGTH NEQNS, CONTAINING THE DEGREE
#                     OF EACH NODE BACK IN THE UNCOMPRESSED GRAPH
#                     PLUS ONE.
#       QSIZE(*)    - ARRAY OF LENGTH NEQNS,
#                        0 -  NODE IS ABSORBED IN THE COMPRESSED
#                             GRAPH.
#                       >0 -  NUMBER OF NODES ABSORBED BY THE NODE
#                             IN THE COMPRESSED GRAPH PLUS ONE.
#
#     OUTPUT PARAMETER:
#       DEFCY2(*)   - ARRAY OF LENGTH NEQNS, CONTAINING THE
#                     DEFICIENCY OF EACH REPRESENTATIVE NODE IN
#                     THE COMPRESSED GRAPH.
#
#     WORKING PARAMETER:
#       UMARK(*)    - ARRAY OF LENGTH NEQNS, USED TO MARK (AND
#                     UNMARK) THE NEIGHBORS OF A NODE (UNODE).
#
#**********************************************************************
#
function mfinit_vanilla(
        neqns::V,
        adjlen::E,
        xadj::AbstractVector{E},
        adjncy::AbstractVector{V},
        degree::AbstractVector{W},
        qsize::AbstractVector{W},
        defcy2::AbstractVector{W},
        umark::AbstractVector{I},
    ) where {V, E, I, W}
    
    #       -------------------
    #       LOCAL VARIABLES ...
    #       -------------------
    
    #       ----------------
    #       INITIALIZATIONS.
    #       ----------------
    for unode in oneto(neqns)
        if !iszero(qsize[unode])
            umark[unode] = zero(I)
        end
    end
    
    #       -----------------------
    #       FOR EACH NODE UNODE ...
    #       -----------------------
    for unode in oneto(neqns)
        #           PRINT *, 'UNODE:', UNODE
        
        #           ----------------------------------------------
        #           IF UNODE IS A REPRESENTATIVE NODE (UNABSORBED)
        #           IN THE COMPRESSED GRAPH ...
        #           ----------------------------------------------
        if !iszero(qsize[unode])
            
            #               ---------------------------------------
            #               INITIALIZE UNODE'S DEFICIENCY COUNT AND
            #               ITS ACTIVE NEIGHBORHOOD SIZE.
            #               ---------------------------------------
            def = zero(W)
            ucount = degree[unode] - qsize[unode]
            #               ----------------------------
            #               MARK THE NEIGHBORS OF UNODE.
            #               ----------------------------
            for w in xadj[unode]:xadj[unode + one(V)] - one(E)
                wnode = adjncy[w]
                umark[wnode] = one(I)
            end
            #               ------------------------------------
            #               FOR EACH NEIGHBOR WNODE OF UNODE ...
            #               ------------------------------------
            for w = xadj[unode]:xadj[unode + one(V)] - one(E)
                wnode = adjncy[w]
                qw = qsize[wnode]
                ucount -= qw
                cnt = ucount
                #                   PRINT *,'   WNODE,CNT:',WNODE,CNT
                #                   -------------------------------
                #                   COMPUTE WNODE'S CONTRIBUTION TO
                #                   UNODE'S DEFICIENCY.
                #                   -------------------------------
                for x in xadj[wnode]:xadj[wnode + one(V)] - one(E)
                    xnode = adjncy[x]

                    if isone(umark[xnode])
                        cnt -= qsize[xnode]
                        #                           PRINT *,'      XNODE,CNT:',XNODE,CNT
                    end
                end
                #                   ------------------------------------
                #                   ACCUMULATE WNODE'S CONTRIBUTION TO
                #                   UNODE'S DEFICIENCY AND UNMARK WNODE.
                #                   ------------------------------------
                def += cnt * qw
                #                   PRINT *,'   WNODE,CNT:',WNODE,CNT
                umark[wnode] = zero(I)
            end
            #               -------------------------------------
            #               RECORD THE DEFICIENCY SCORE OF UNODE.
            #               -------------------------------------
            defcy2[unode] = def
            
        end
        
    end
    
    return
end
