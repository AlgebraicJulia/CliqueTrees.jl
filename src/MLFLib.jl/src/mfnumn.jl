#**********************************************************************
#**********************************************************************
#*****   MFNUMN .....  VERTEX NUMBER STEP - NATURAL   *****************
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
#       THIS SUBROUTINE NUMBERS THE VERTICES OF THE ORIGINAL GRAPH 
#       PASSED INTO THE MINIMUM LOCAL FILL ORDERING ROUTINES.
#       WITHIN EACH SUPERNODE, THE REPRESENTATIVE VERTEX IS NUMBERED
#       FIRST, AND THE REMAINING VERTICES ARE NUMBERED IN THE NATURAL
#       ORDERING IN THE ORIGINAL ORDERING.
#
#     INPUT PARAMETERS:
#       NEQNS       - NUMBER OF COLUMN VERTICES.
#       WORK(*)     - ARRAY OF LENGTH NEQNS.
#                       ELIMINATED REPRESENTATIVE: >=0
#                       ABSORBED: MAPS ABSORBED VERTEX TO MINUS THE
#                         ABSORBING VERTEX (I.E., -ABSORBEE).
#       INVP(*)     - ARRAY OF LENGTH NEQNS.
#                       ELIMINATED REPRESENTATIVE: -NUM, WHERE NUM
#                         IS THE NUMBER ASSIGNED BY THE ELIMINATION
#                         ORDER.
#                       ABSORBED: USELESS DATA.
#
#     MODIFIED PARAMETERS:
#       WORK(*)     - SHORTENED SUPERNODE FOREST.
#       INVP(*)     - ENDS UP WITH THE INVERSE PERMUTATION VECTOR, 
#                     WHICH MAPS OLD NUMBERS TO NEW.
#
#     OUTPUT PARAMETER:
#       PERM(*)     - ARRAY OF LENGTH NEQNS.
#                     ENDS UP WITH THE PERMUTATION VECTOR, WHICH 
#                     MAPS NEW NUMBERS TO OLD.
#
#**********************************************************************
#
function mfnumn(neqns::V, work::AbstractVector{V}, invp::AbstractVector{V}, perm::AbstractVector{V}) where {V}
    
    #       -------------------
    #       LOCAL VARIABLES ...
    #       -------------------
    
    #       -------------------------------------------------------
    #       INITIALIZATION AND TAKE CARE OF REPRESENTATIVE COLUMNS.
    #       -------------------------------------------------------
    for jcol in oneto(neqns)
        perm[jcol] = zero(V)
    end

    for jcol in oneto(neqns)
        if !isnegative(work[jcol])
            perm[-invp[jcol]] = jcol
        end
    end
    
    #       -----------------------------------------------------
    #       MAIN LOOP -- TAKE CARE OF NON-REPRESENTATIVE COLUMNS.
    #       FOR EACH COLUMN JCOL ...
    #       -----------------------------------------------------
    for jcol in oneto(neqns)
        
        #           ---------------------------
        #           SKIP REPRESENTATIVE COLUMN.
        #           ---------------------------
        if isnegative(work[jcol])
            
            #               ----------------------------
            #               FIND ROOT OF SUPERNODE TREE.
            #               ----------------------------
            parent = jcol

            while isnegative(work[parent])
                parent = -work[parent]
            end
            
            #               -----------------------
            #               NUMBER JCOL AFTER ROOT.
            #               -----------------------
            root = parent
            invp[root] -= one(V)
            perm[-invp[root]] = jcol
            
            #               --------------------
            #               SHORTEN MERGED TREE.
            #               --------------------
            parent = jcol
            nextp = -work[parent]

            while ispositive(nextp)
                work[parent] = -root
                parent = nextp
                nextp = -work[parent]
            end
            
        end
        
    end
    
    #       --------------------------------
    #       COMPUTE INVERSE PERMUTATION TOO.
    #       --------------------------------
    for jcol in oneto(neqns)
        invp[perm[jcol]] = jcol
    end
    
    return
end
