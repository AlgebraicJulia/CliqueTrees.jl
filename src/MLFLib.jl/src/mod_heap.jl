#**********************************************************************
#**********************************************************************
#*****   MOD_HEAP ... REBUILD A HEAP AFTER A NODE HAS BEEN   **********
#*****                MODIFIED                               **********
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
#       MOD_HEAP MODIFIES AN ELEMENT IN A HEAP AND REBUILDS THE HEAP.
#
#       EACH ELEMENT HAS TWO FIELDS: (WEIGHT,VERTEX).  THE VERTEX
#       FIELD REFERS TO A VERTEX AND THE WEIGHT FIELD REFERS TO A
#       WEIGHT ASSOCIATED WITH THE VERETEX.  THE VERTICES ARE ASSUMED
#       TO BE UNIQUE INTEGERS FROM 1 TO N.  THE HEAP IS CONSTRUCTED
#       USING THE WEIGHTS.  THE WEIGHT OF A GIVEN VERTEX IS TO BE
#       MODIFIED.
#     
#     INPUT PARAMETERS:
#       HEAPSIZE    - IT IS THE NUMBER OF ELEMENTS IN HEAP.
#       VTX         - THE ELEMENT WITH VERTEX = VTX IS TO BE
#                     MODIFED. 
#       WT          - THE NEW WEIGHT OF THE ELEMENT WITH
#                     VERTEX = VTX.
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
function mod_heap(heap::AbstractVector{W}, heapsize::Int, v2heap::AbstractVector{Int}, vtx::Int, wt::W) where {W}
    
    #       -------------------
    #       LOCAL VARIABLES ...
    #       -------------------
    
    #       ---------------------------------------------------
    #       GET THE LOCATION OF THE NODE CONTAINING VERTEX VTX.
    #       ---------------------------------------------------
    i = v2heap[vtx]
    
    #       -----------------------------------------------------
    #       MODIFY THE WEIGHT FIELD OF ELEMENT WITH VERTEX = VTX.
    #       -----------------------------------------------------
    heap[twice(i) - 1] = wt
    
    #       ------------------------------------------------------
    #       HEAP MAY NO LONGER BE A HEAP AFTER THE WEIGHT FIELD OF
    #       AN ELEMENT IS MODIFIED.  IT HAS TO BE REBUILT INTO A
    #       HEAP.
    #       ------------------------------------------------------
    
    if ispositive(i - 1)
        #           ----------------------------------------------------
        #           THE MODIFIED ELEMENT IS NOT AT THE TOP OF THE BIANRY
        #           TREE, SO IT HAS A PARENT, GIVEN BY I/2.
        #           ----------------------------------------------------
        p = half(i)

        if heap[twice(p) - 1] > wt
            #               ------------------------------------------------
            #               THE WEIGHT FIELD OF THE PARENT IS LARGER THAN THE
            #               WEIGHT OF THE MODIFIED ELEMENT, SO THE MODIFIED
            #               ELEMENT HAS TO BE MOVED UP THE BINARY TREE TO
            #               REBUILD THE HEAP.
            #               ------------------------------------------------
            move_up(heap, heapsize, i, v2heap)
            return
        end
    end
    
    #       -------------------------------------------------------
    #       EITHER THE MODIFIED ELEMENT IS AT THE TOP OF THE BINARY
    #       TREE (WITH I = 1), OR ITS WEIGHT FIELD IS LESS THAN OR
    #       EQUAL TO THE WEIGHT FIELD OF THE PARENT IN THE BINARY
    #       TREE.
    #
    #       THE BINARY TREE ROOTED AT THE MODIFIED ELEMENT MAY NOT
    #       BE A HEAP ANYMORE.  IT MAY BE NECESSARY TO REBUILD THE
    #       HEAP BY TRAVERSING DOWN THE BINARY TREE, STARTING FROM
    #       THE MODIFIED ELEMENT.
    #       -------------------------------------------------------
    move_down(heap, heapsize, i, v2heap)
    
    return
end
