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
#       SCORE       - NODE-SELECTION SCORE (AN MLFSCORE). DEFAULTS
#                     TO MINFILL, THE MINIMUM LOCAL FILL SCORE.
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
function mlf!(n::V, vwght::AbstractVector{W}, xadj::AbstractVector{E}, adjncy::AbstractVector{V}) where {V, E, W}
    return mlf!(MinFill, n, vwght, xadj, adjncy)
end

function mlf!(score::MLFScore, n::V, vwght::AbstractVector{W}, xadj::AbstractVector{E}, adjncy::AbstractVector{V}) where {V, E, W}
    nnz = xadj[n + one(V)] - one(E)
    adjlen = convert(E, length(adjncy))

    #       -------------------
    #       LOCAL VARIABLES ...
    #       -------------------
    perm = Vector{V}(undef, n)
    invp = Vector{V}(undef, n)

    @inbounds for i in oneto(n)
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
    #       (SLOTS 3 AND 21 TO 38 OF THE WORKSPACE; SEE UTILITIES.)
    ws      = workspace()
    changed = workvector(ws, Bool, 21, n)
    degree  = workvector(ws, W, 22, n)
    ecforw  = workvector(ws, V, 23, n)
    ecliq   = workvector(ws, V, 24, n)
    heapinv = workvector(ws, V, 25, n)
    len2    = workvector(ws, V, 26, n)
    marker  = workvector(ws, Int, 27, n)
    nvtxs   = workvector(ws, V, 28, n)
    qsize   = workvector(ws, W, 29, n)
    qnmbr   = workvector(ws, V, 30, n)
    umark   = workvector(ws, Int, 31, n)
    work    = workvector(ws, V, 32, n)
    work1   = workvector(ws, Int, 33, n)
    work2   = workvector(ws, W, 34, n)
    xadj2   = workvector(ws, E, 35, n)
    adj2    = workvector(ws, V, 3, adjlen2)

    defncy  = workvector(ws, W, 37, n)
    heap    = workvector(ws, SCORE, 38, twice(n))

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
        score, n, adjlen, adjlen2, vwght, xadj,
        adjncy, defflag, perm, invp,
        marker, nvtxs, work,
        qsize, qnmbr, ecforw, defncy, adj2, changed,
        degree, umark, heap, heapinv, ecliq,
        xadj2, len2, work1, work2,
    )

    return perm, invp
end
