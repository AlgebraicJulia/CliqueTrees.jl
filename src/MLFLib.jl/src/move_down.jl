#**********************************************************************
#**********************************************************************
#*****   MOVE_DOWN ... MOVE NODE DOWN A HEAP   ************************
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
#       MOVE_DOWN MOVES A GIVEN ELEMENT DOWN A BINARY TREE TO MAKE
#       SURE THE BINARY TREE IS A HEAP.
#
#       EACH ELEMENT HAS TWO FIELDS: (WEIGHT,VERTEX).  THE VERTEX
#       FIELD REFERS TO A VERTEX AND THE WEIGHT FIELD REFERS TO A
#       WEIGHT ASSOCIATED WITH THE VERETEX.  THE VERTICES ARE ASSUMED
#       TO BE UNIQUE INTEGERS FROM 1 TO N.  THE HEAP IS CONSTRUCTED
#       USING THE WEIGHTS.
#
#     INPUT PARAMETERS:
#       HEAPSIZE    - IT IS THE NUMBER OF ELEMENTS IN HEAP.
#       VTX_PTR     - THE LOCATION OF THE ELEMENT TO BE MOVED DOWN.
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
function move_down(heap::AbstractVector{W}, heapsize::Int, vtx_ptr::Int, v2heap::AbstractVector{Int}) where {W}
    
    #       -------------------
    #       LOCAL VARIABLES ...
    #       -------------------
    
    infty = typemax(W) - convert(W, 100000)
    
    #       --------------------------------------------------------
    #       MOVE THE GIVEN NODE IN LOCATION VTX_PTR DOWN IN HEAP AND
    #       ENSURE THAT HEAP IS A HEAP.
    #       --------------------------------------------------------
    v = vtx_ptr
    #       PRINT *, 'FLD_NOW, FLD_INDEX = ', V, VTX_PTR
    
    while v <= heapsize
        
        #           ----------------------------------------------
        #           DO THE FOLLOWING, AS LONG AS V HAS NOT REACHED
        #           THE BOTTOM OF THE BINARY TREE.
        #           ----------------------------------------------
        
        #           ------------------------
        #           V_WT IS THE WEIGHT OF V.
        #           ------------------------
        v_wt = heap[twice(v) - 1]
        #           PRINT *, 'FLD_NOW, VAL = ', V, V_WT
        
        #           ----------------------------------------
        #           C_LEFT AND C_RIGHT ARE THE CHILDREN OF V
        #           IN THE BINARY TREE.
        #           ----------------------------------------
        c_left = twice(v)
        c_right = c_left + 1
        #           PRINT *, 'FLD_NEXT1, FLD_NEXT2 = ',
        #    &                C_LEFT, C_RIGHT
        
        #           ---------------------------------
        #           GET THE WEIGHT'S OF THE CHILDREN.
        #           ---------------------------------
        if c_left <= heapsize
            wt_left = heap[twice(c_left) - 1]
        else
            wt_left = infty
        end
        
        if c_right <= heapsize
            wt_right = heap[twice(c_right) - 1]
        else
            wt_right = infty
        end
        
        #           PRINT *, 'VAL1, VAL2 = ', WT_LEFT, WT_RIGHT
        
        #           ----------------------------
        #           DECIDE WHICH PATH TO FOLLOW.
        #           ----------------------------
        if v_wt <= wt_left && v_wt <= wt_right
            
            #               ------------------------------------------
            #               THE WEIGHT OF V IS LESS THAN OR EQUAL TO
            #               THOSE OF THE TWO CHILDREN, SO WE ARE DONE.
            #               ------------------------------------------
            v = heapsize + 1
            #               PRINT *, 'DONE ...'
            
        else
            
            #               ----------------------------------------
            #               THE WEIGHT OF V IS GREATER THAN THAT OF
            #               AT LEAST ONE OF THE CHILDERN.  DETERMINE
            #               WHICH PATH TO TRAVERSE.
            #               ----------------------------------------
            
            if wt_left <= wt_right
                c = c_left
            else
                c = c_right
            end
            #               PRINT *, 'FLD_NEXT = ', P
            
            #               ---------------------
            #               SWAP V AND THE CHILD.
            #               ---------------------
            
            v_ptr = twice(v)
            c_ptr = twice(c)
            #               PRINT *, 'FLD_NEXT1, FLD_NEXT2 = ',
            #    &                    V_PTR, C_PTR
            
            v_vtx = convert(Int, heap[v_ptr])
            #               PRINT *, 'VTX = ', V_VTX
            v2heap[v_vtx] = c
            #               PRINT *, 'FLD_NEXT = ', P
            
            c_vtx = convert(Int, heap[c_ptr])
            v2heap[c_vtx] = v
            #               PRINT *, 'IPOINT, FLD_NOW = ', C_VTX, V
            
            heap[v_ptr] = heap[c_ptr]
            heap[c_ptr] = v_vtx
            #               PRINT *, 'HEAP(FLD_NEXT1), HEAP(FLD_NEXT2) = ',
            #    &                    HEAP(V_PTR), HEAP(C_PTR)
            
            v_ptr -= 1
            c_ptr -= 1
            heap[v_ptr] = heap[c_ptr]
            heap[c_ptr] = v_wt
            #               PRINT *, 'HEAP(FLD_NEXT1), HEAP(FLD_NEXT2) = ',
            #    &                    HEAP(V_PTR), HEAP(C_PTR)
            
            v = c
            
        end
        
    end
    
    return
end
