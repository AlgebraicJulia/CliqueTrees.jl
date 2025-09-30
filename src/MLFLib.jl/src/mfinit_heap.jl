#**********************************************************************
#**********************************************************************
#*****   MFINIT_HEAP ... INITIALIZE VECTORS AND HEAP     **************
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
#       THIS SUBROUTINE INITIALIZES KEY VECTORS IN THE DATA STRUCTURE
#       AND INSERTS THE ACTIVE VERTICES IN THE HEAP STRUCTURE AND
#       MAKES THE HEAP.
#
#     INPUT PARAMETERS:
#       NEQNS       - NUMBER OF EQUATIONS.
#       XADJ(*)     - ARRAY OF LENGTH NEQNS+1, CONTAINING POINTERS
#                     TO THE ADJACENCY STRUCTURE OF THE COMPRESSED
#                     GRAPH.
#       DEGREE(*)   - ARRAY OF LENGTH NEQNS, CONTAINING THE DEGREE
#                     OF EACH REPRESENTATIVE NODE IN THE ORIGINAL 
#                     GRAPH, PLUS ONE.
#       QSIZE(*)    - SUPERNODE SIZE:
#                       0  -  ABSORBED VERTEX
#                       >0 -  SUPERNODE SIZE OF A REPRESENTATIVE
#                             VERTEX
#       DEFNCY(*)   - ARRAY OF LENGTH NEQNS, CONTAINING THE EXACT 
#                     DEFICIENCY OF EACH REPRESENTATIVE NODE IN
#                     THE INITIAL COMPRESSED GRAPH (IF COMPRESSION
#                     WAS DONE).
#
#     OUTPUT PARAMETERS:
#       HEAP(*)     - ARRAY OF LENGTH 2*NEQNS, CONTAINING THE 
#                     PRIORITY HEAP OF VERTICES BASED ON
#                     DEFICIENCY-DEPENDENT SCORES.
#       HEAPSIZE    - NUMBER OF VERTICES IN THE HEAP.
#       HEAPINV(*)  - ARRAY OF LENGTH NEQNS, CONTAINING POINTERS
#                     FROM THE VERTICES TO THEIR LOCATIONS IN THE
#                     HEAP.
#       MARKER(*)   - ARRAY OF LENGTH NEQNS, INITIALIZED TO ZERO
#                     FOR MARKING PROCESSES IN SUBSEQUENT
#                     PROCESSING AFTER THIS ROUTINE.
#       NVTXS(*)    - ARRAY OF LENGTH NEQNS, NUMBER OF VERTEX
#                     NEIGHBORS IN INITIAL QUOTIENT GRAPH.
#                       >= 0  FOR EACH REPRESENTATIVE NODE.
#                       -1    FOR EACH ABSORBED VERTEX.
#       WORK(*)     - ARRAY OF LENGTH NEQNS, NUMBER OF CLIQUE
#                     NEIGHBORS IN INITIAL QUOTIENT GRAPH.
#                       0     FOR EACH REPRESENTATIVE NODE.
#       ECFORW(*)   - ARRAY OF LENGTH NEQNS, INITIALLY EVERY NODE
#                     IS NOT IN THE ELIMINATION CLIQUE LIST (THEY
#                     ARE NOT ELIMINATION CLIQUES).
#                       -1    FOR EACH REPRESENTATIVE NODE.
#       CHANGED(*)  - ARRAY OF LENGTH NEQNS, INITIALLY EVERY NODE
#                     IS RECORDED AS NOT HAVING ITS DEFICIENCY
#                     CURRENTLY CHANGED.
#                       -1    FOR EACH REPRESENTATIVE NODE.
#       UMARK(*)    - ARRAY OF LENGTH NEQNS, ARRAY FOR LATER
#                     WING-HUANG MARKING PROCESSES; IT IS
#                     INITIALIZED TO ZERO HERE.
#                       0     FOR EACH REPRESENTATIVE NODE.
#       INVP(*)     - ARRAY OF LENGTH NEQNS, INITIALLY ZERO FOR
#                     EVERY REPRESENTATIVE VERTEX TO INDICATE THAT
#                     IT HAS NOT BEEN ELIMINATED.
#
#**********************************************************************
#
function mfinit_heap(
        neqns::Int,
        xadj::AbstractVector{Int},
        degree::AbstractVector{W},
        qsize::AbstractVector{W},
        defncy::AbstractVector{W},
        heap::AbstractVector{W},
        heapinv::AbstractVector{Int},
        marker::AbstractVector{Int},
        nvtxs::AbstractVector{Int},
        work::AbstractVector{Int},
        ecforw::AbstractVector{Int},
        changed::AbstractVector{Bool},
        umark::AbstractVector{Int},
        invp::AbstractVector{Int},
    ) where {W}
    
    #       ---------------------------------------------------------
    #       MORE INITIALIZATIONS AND ALSO PLACE EACH NODE AND ITS
    #       DEFICIENCY-BASED SCORE IN THE HEAP STRUCTURE IN ARBITRARY
    #       ORDER.
    #       ---------------------------------------------------------
    
    heapcnt = 0
    
    #       -------------------------
    #       FOR EACH VERTEX JNODE ...
    #       -------------------------
    for jnode in oneto(neqns)
        
        #           ----------------------------------------------
        #           NVTXS(JNODE) WILL BE -1 FOR ABSORBED VERTICES.
        #           ----------------------------------------------
        nvtxs[jnode] = -1
        
        #           -------------------------------------
        #           IF JNODE IS A REPRESENTATIVE NODE ...
        #           -------------------------------------
        if !iszero(qsize[jnode])
            
            #               ---------------------------
            #               INITIALIZE VARIOUS VECTORS.
            #               ---------------------------
            ecforw[jnode] = -1
            changed[jnode] = false
            umark[jnode] = 0
            work[jnode] = 0
            invp[jnode] = 0
            marker[jnode] = 0
            nvtxs[jnode] = xadj[jnode + 1] - xadj[jnode]
            
            #               ------------------------------------------
            #               DJ WILL BE THE DEGREE OF JNODE (PLUS ONE).
            #               ------------------------------------------
            def = defncy[jnode]
            
            #               -------------------------------
            #               COMPUTE DEFICIENCY-BASED SCORE.
            #               -------------------------------
            
            #               ---------------------------------------------
            #               INSERT JNODE AND ITS DEFICIENCY-BASED SCORE
            #               INTO THE HEAP IN THE NEXT AVAILABLE LOCATION.
            #               ---------------------------------------------
            heapcnt += 1
            heapinv[jnode] = heapcnt
            heap[twice(heapcnt) - 1] = def
            heap[twice(heapcnt)] = jnode
            
        end
        
    end
    
    #       ------------------------------
    #       IMPOSE HEAP ORDER ON THE HEAP.
    #       ------------------------------
    heapsize = build_heap(heap, heapinv, heapcnt)
    
    return heapsize
end
