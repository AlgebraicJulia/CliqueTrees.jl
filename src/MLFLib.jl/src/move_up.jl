#**********************************************************************
#**********************************************************************
#*****   MOVE_UP ... MOVE A NODE UP A HEAP   **************************
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
#       MOVE_UP MOVES A GIVEN ELEMENT UP A BINARY TREE TO MAKE SURE
#       THE BINARY TREE IS A HEAP.
#
#       EACH ELEMENT HAS TWO FIELDS: (WEIGHT,VERTEX).  THE VERTEX
#       FIELD REFERS TO A VERTEX AND THE WEIGHT FIELD REFERS TO A
#       WEIGHT ASSOCIATED WITH THE VERETEX.  THE VERTICES ARE ASSUMED
#       TO BE UNIQUE INTEGERS FROM 1 TO N.  THE HEAP IS CONSTRUCTED
#       USING THE WEIGHTS.
#
#     INPUT PARAMETERS:
#       HEAPSIZE    - IT IS THE NUMBER OF ELEMENTS IN HEAP.
#       VTX_PTR     - THE LOCATION OF THE ELEMENT TO BE MOVED UP.
#
#     MODIFIED PARAMETERS:
#       HEAP(*)     - IT IS AN ARRAY OF SIZE 2*HEAPSIZE.  ON INPUT,
#                     IT CONTAINS A SET OF NODES THAT ARE IN A HEAP,
#                     WITH HEAP(2*K-1) CONTAINING THE WEIGHT OF NODE
#                     K AND HEAP(2*K) CONTAINING THE CORRESPONDING
#                     VERTEX.  ON OUTPUT, THE NODES ARE REARRANGED
#                     TO FORM A HEAP.
#       V2HEAP(*)   - V2HEAP IS A MAPPING OF THE VERTICES TO
#                     THE NODES IN HEAP.
#
#**********************************************************************
#
function move_up(heap::AbstractVector{W}, heapsize::Int, vtx_ptr::Int, v2heap::AbstractVector{Int}) where {W}
    
    #       -------------------
    #       LOCAL VARIABLES ...
    #       -------------------
    
    #       ------------------------------------------------------
    #       MOVE THE GIVEN NODE IN LOCATION VTX_PTR UP IN HEAP AND
    #       ENSURE THAT HEAP IS A HEAP.
    #       ------------------------------------------------------
    v = vtx_ptr
    #       PRINT *, 'FLD_INDEX = ', VTX_PTR
    
    while ispositive(v - 1)
        #           PRINT *, 'NODE = ', V
        
        #           ------------------------------------------------
        #           DO THE FOLLOWING, AS LONG AS THE ELEMENT HAS NOT
        #           REACHED THE ROOT OF THE BINARY TREE.
        #           ------------------------------------------------
        
        #           ----------------
        #           P IS THE PARENT.
        #           ----------------
        p = half(v)
        
        #           -----------------------------------------------
        #           V_PTR AND P_PTR POINT TO WHERE V AND ITS PARENT
        #           ARE LOCATED IN HEAP.
        #           -----------------------------------------------
        v_ptr = twice(v)
        p_ptr = twice(p)
        
        #           --------------------------------------------
        #           V_WT AND P_WT ARE THE WEIGHT FIELDS OF V AND
        #           ITS PARENT.
        #           --------------------------------------------
        v_wt = heap[v_ptr - 1]
        p_wt = heap[p_ptr - 1]
        
        #           PRINT *, 'VAL, VAL1 = ', V_WT, P_WT
        if v_wt >= p_wt
            
            #               ----------------------------------------------
            #               WE ARE DONE MOVING ELEMENT UP THE BINARY TREE.
            #               ----------------------------------------------
            v = 0
            
        else
            
            #               ----------------------------------------------
            #               HERE V_WT < P_WT, SO THE WEIGHT FIELD OF THE
            #               THE NODE IS LESS THAN THAT OF THE PARENT.  THE
            #               TWO SHOULD BE SWAPPED.
            #               ----------------------------------------------
            
            c_vtx = convert(Int, heap[v_ptr])
            v2heap[c_vtx] = p
            
            p_vtx = convert(Int, heap[p_ptr])
            v2heap[p_vtx] = v
            #               PRINT *, 'VTX, FLD_NEXT, IPOINT, FLD_NOW = ',
            #    &                    C_VTX, P, P_VTX, V
            
            heap[v_ptr] = heap[p_ptr]
            heap[p_ptr] = convert(W, c_vtx)
            
            heap[v_ptr - 1] = p_wt
            heap[p_ptr - 1] = v_wt
            
            v = p
            
        end
        
    end
    
    return
end
