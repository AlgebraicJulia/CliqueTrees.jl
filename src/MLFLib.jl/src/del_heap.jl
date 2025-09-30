#**********************************************************************
#**********************************************************************
#*****   DEL_HEAP ... DELETE A NODE FROM A HEAP   *********************
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
#       DEL_HEAP DELETES AN ELEMENT FROM A HEAP AND REBUILDS THE HEAP.
#
#       EACH ELEMENT HAS TWO FIELDS: (WEIGHT,VERTEX).  THE VERTEX
#       FIELD REFERS TO A VERTEX AND THE WEIGHT FIELD REFERS TO A
#       WEIGHT ASSOCIATED WITH THE VERETEX.  THE VERTICES ARE ASSUMED
#       TO BE UNIQUE INTEGERS FROM 1 TO N.  THE HEAP IS CONSTRUCTED
#       USING THE WEIGHTS.
#     
#     INPUT PARAMETERS:
#       HEAPSIZE    - IT IS THE NUMBER OF ELEMENTS IN HEAP.
#       VTX         - THE ELEMENT WITH VERTEX = VTX IS TO BE REMOVED
#                     FROM THE HEAP.
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
function del_heap(heap::AbstractVector{W}, heapsize::Int, v2heap::AbstractVector{Int}, vtx::Int) where {W}
    
    #       -------------------
    #       LOCAL VARIABLES ...
    #       -------------------
    
    #       --------------------------------
    #       VTX IS THE VERTEX TO BE REMOVED.
    #       --------------------------------
    v = vtx

    if iszero(heapsize)
        return heapsize
    end
    
    #       ---------------------------------------------------
    #       I GIVES THE LOCATION OF THE VERTEX VTX IN THE HEAP.
    #       ---------------------------------------------------
    i = v2heap[v]

    if i < heapsize
        
        #           -----------------------------------------
        #           INDICATE THAT THE ELEMENT WITH VERTEX = V
        #           IS NO LONGER IN THE HEAP.
        #           -----------------------------------------
        v2heap[v] = 0
        
        #           ----------------------------------------------------
        #           THE ELEMENT BEING REMOVED OCCUPIES HEAP(2*I)-1 AND
        #           HEAP(2*I).  MOVE THE LAST ELEMENT IN THE HEAP TO THE
        #           POSITION OCCUPIED BY THE DELETED ELEMENT.
        #           ----------------------------------------------------
        lp = twice(heapsize)
        ip = twice(i)
        
        v = convert(Int, heap[lp])
        heap[ip] = v
        v2heap[v] = i
        heapsize -= 1
        
        wt = heap[lp - 1]
        
        #           -------------------------------------------------
        #           CALLING MOD_HEAP TO PUT THE WEIGHT IN THE CORRECT
        #           PLACE AND REBUILD THE HEAP.
        #           -------------------------------------------------
        mod_heap(heap, heapsize, v2heap, v, wt)
        
    else
        
        #           ---------------------------------------------
        #           THE ELEMENT BEING REMOVED IS THE LAST ELEMENT
        #           IN THE HEAP, SO THERE IS NOT MUCH TO DO.
        #           ---------------------------------------------
        v2heap[v] = 0
        heapsize -= 1
    end
    
    return heapsize
end
