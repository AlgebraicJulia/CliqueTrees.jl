#**********************************************************************
#**********************************************************************
#*****   BUILD_HEAP ... BUILD A HEAP   ********************************
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
#       BUILD_HEAP CREATES A BINARY HEAP OUT OF A SET OF N ELEMENTS.
#
#       EACH ELEMENT HAS TWO FIELDS: (WEIGHT,VERTEX).  THE VERTEX
#       FIELD REFERS TO A VERTEX AND THE WEIGHT FIELD REFERS TO A
#       WEIGHT ASSOCIATED WITH THE VERETEX.  THE VERTICES ARE ASSUMED
#       TO BE UNIQUE INTEGERS FROM 1 TO N.  THE HEAP IS CONSTRUCTED
#       USING THE WEIGHTS.
#     
#     INPUT PARAMETER:
#       N           - IT IS THE NUMBER OF INPUT ELEMENTS.
#
#     MODIFIED PARAMETERS:
#       HEAP(*)     - IT IS AN ARRAY OF SIZE 2*N.  ON INPUT,
#                     IT CONTAINS A SET OF NODES (IN ARBITRARY
#                     ORDER).  HEAP(2*K-1) CONTAINS THE WEIGHT
#                     OF NODE K AND HEAP(2*K) CONTAINS THE
#                     CORRESPONDING VERTEX.  ON OUTPUT, THE
#                     NODES ARE REARRANGED TO FORM A HEAP.
#       V2HEAP(*)   - V2HEAP IS A MAPPING OF THE VERTICES TO
#                     THE NODES IN THE HEAP.
#
#     OUTPUT PARAMETER:
#       HEAPSIZE    - IT IS THE NUMBER OF ELEMENTS IN HEAP.
#
#**********************************************************************
#
function build_heap(heap::AbstractVector{W}, v2heap::AbstractVector{Int}, n::Int) where {W}
    
    #       -------------------
    #       LOCAL VARIABLES ...
    #       -------------------
    
    heapsize = n

    for node in reverse(oneto(half(n)))
        move_down(heap, heapsize, node, v2heap)
    end

    return heapsize
end
