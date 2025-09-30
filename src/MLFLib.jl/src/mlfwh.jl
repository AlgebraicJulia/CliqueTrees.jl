#**********************************************************************
#**********************************************************************
#*****   MLFWH ... COMPUTE A MIMUM LOCAL FILL ORDERING   **************
#*****             USING THE WING-HUANG UPDATES          **************
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
#       THIS IS A SIMPLE INTERFACE FOR INVOKING GENMF_WH, WHICH
#       COMPUTES A MINMUM LOCAL FILL ORDERING USING THE WING-HUANG
#       UPDATES.
#
#     INPUT PARAMETERS:
#       N           - NUMBER OF EQUATIONS.
#       NNZ         - NUMBER OF NONZERO ENTRIES (NOT COUNTING THE
#                     DIAGONAL ELEMENTS) IN THE MATRIX.
#       ADJLEN      - LENGTH OF THE ADJACENCY STRUCTURE (ADJNCY(*)).
#                     IDEALLY AT LEAST N GREATER THAN INITIAL
#                     XADJ(N+1)-1 (SEE IFLAG).
#       XADJ(*)     - ARRAY OF LENGTH N+1, CONTAINING POINTERS
#                     TO THE ADJACENCY STRUCTURE.
#       ADJNCY(*)   - ARRAY OF LENGTH ADJLEN, CONTAINING THE
#                     ADJACENCY STRUCTURE.
#
#     MODIFIED PARAMETERS:
#       XADJ(*)     - DESTROYED ON OUTPUT.
#       ADJNCY(*)   - DESTROYED ON OUTPUT.
#
#     OUTPUT PARAMETERS:
#       PERM(*)     - ENDS UP WITH THE PERMUTATION VECTOR, WHICH
#                     MAPS NEW NUMBERS TO OLD.
#       INVP(*)     - ENDS UP WITH THE INVERSE PERMUTATION VECTOR,
#                     WHICH MAPS OLD NUMBERS TO NEW.
#       NOFNZ       - NUMBER OF OFF-DIAGONAL NONZEROS IN THE FACTOR
#                     MATRIX PRODUCED BY THE ORDERING.  IT IS A
#                     LONG INTEGER (INTEGER*8).
#       IFLAG       - ERROR FLAG
#                        0 -  NO ERROR OR WARNING.
#                        1 -  WARNING, ADJLEN IS SMALL; IT IS
#                             BETWEEN INITIAL XADJ(N+1)-1 AND
#                             INITIAL XADJ(N+1)+N-1,
#                             INCLUSIVE.  MAY CAUSE EXCESSIVE
#                             GARBAGE COLLECTION.
#                       -1 -  ERROR, ADJLEN IS SMALLER THAN INITIAL
#                             XADJ(N+1)-1.  INSUFFICIENT ROOM
#                             IN ADJNCY(*) FOR THE ADJACENCY
#                             STRUCTURE, OR INCORRECT ADJLEN HAS
#                             BEEN INPUT.
#                       -2 -  FAIL TO ALLOCATE WORK SPACE.
#                       -3 -  FAIL TO DEALLOCATE WORK SPACE.
#                       -4 -  FAIL TO ALLOCATE AND DEALLOCATE
#                             WORK SPACE.
#
#**********************************************************************
#
function mlf(n::Int, vwght::AbstractVector{W}, xadj::AbstractVector{Int}, adjncy::AbstractVector{Int}) where {W}
    nnz = xadj[n + 1] - 1
    adjlen = length(adjncy)
    
    #       -------------------
    #       LOCAL VARIABLES ...
    #       -------------------
    perm = FVector{Int}(undef, n)
    invp = FVector{Int}(undef, n)
    
    for i in oneto(n)
        perm[i] = i
        invp[i] = i
    end
    
    #       -----------------------------
    #       SET POINTERS FOR WORK ARRAYS.
    #       -----------------------------
    adjlen2 = nnz
    
    #       --------------------
    #       ALLOCATE WORK SPACE.
    #       --------------------
    changed = FVector{Bool}(undef, n) 
    degree  = FVector{W}(undef, n)
    ecforw  = FVector{Int}(undef, n)
    ecliq   = FVector{Int}(undef, n)
    heapinv = FVector{Int}(undef, n)
    len2    = FVector{Int}(undef, n)
    marker  = FVector{Int}(undef, n)
    nvtxs   = FVector{Int}(undef, n)
    qsize   = FVector{W}(undef, n)
    qnmbr   = FVector{Int}(undef, n)
    umark   = FVector{Int}(undef, n)
    work    = FVector{Int}(undef, n)
    work1   = FVector{Int}(undef, n)
    work2   = FVector{W}(undef, n)
    xadj2   = FVector{Int}(undef, n)
    adj2    = FVector{Int}(undef, adjlen2)
   
    defncy  = FVector{W}(undef, n)
    heap    = FVector{W}(undef, twice(n))
    
    maxint  = 2_000_000_000
    
    #       ------------------------------------------------------------
    #       DEFFLAG INDICATES HOW THE INITIAL DEFICIENCIES ARE COMPUTED.
    #       A ZERO VALUE MEANS THAT THE INITIAL DEFICIENCIES WILL BE
    #       COMPUTING USING A STRAIGHTFORWARD WAY.
    #       A NONZERO VALUE MEANS THAT THE INITIAL DEFICIENCIES WILL BE
    #       COMPUTED BY ADDING THE EDGES OF THE GRAPH ONE AT A TIME TO
    #       AN INITIALLY EMPTY GRAPH AND PERFORMING WING-HUANG UPDATING
    #       FOR EACH NEW EDGE.
    #       ------------------------------------------------------------
    defflag = true
    
    #       ----------------------------------------------
    #       GENMF_WH COMPUTES A MINMUM LOCAL FILL ORDERING
    #       USING WING-HUANG UPDATES.
    #       ----------------------------------------------
    nofnz, gbgcnt, iflag = genmf_wh(
        n, maxint, adjlen, adjlen2, vwght, xadj,
        adjncy, defflag, perm, invp,
        marker, nvtxs, work,
        qsize, qnmbr, ecforw, defncy, adj2, changed,
        degree, umark, heap, heapinv, ecliq,
        xadj2, len2, work1, work2,
    )
    
    return perm, invp, nofnz, iflag
end
