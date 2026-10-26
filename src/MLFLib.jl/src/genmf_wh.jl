#**********************************************************************
#**********************************************************************
#*****   GENMF_WH ..... MINIMUM DEFICIENCY ORDERING   *****************
#*****                  (WITH WING-HUANG UPDATING)    *****************
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
#       THIS SUBROUTINE COMPUTES ORDERINGS BASED ON MINIMUM LOCAL
#       FILL.  IT USES WING-HUANG UPDATING TO UPDATE DEFICIENCY
#       COUNTS AFTER EACH ELIMINATION STEP.
#
#     ACKNOWLEDGEMENTS:
#       THIS CODE OWES MUCH TO PAST MINIMUM DEGREE CODES.  THE
#       DATA STRUCTURES, MODULARITY, AND VARIABLE NAMING
#       CONVENTIONS OWE MUCH TO THE MULTIPLE MINIMUM DEGREE CODE
#       OF LIU (GENMMD.F).  THE AGGRESSIVE 'HASHING' TECHNIQUE
#       FOR DETECTING INDISTINGUISHABLE VERTICES OWES MUCH TO THE
#       APPROXIMATE MINIMUM DEGREE CODE OF AMESTOY, DAVIS, AND DUFF.
#
#       REFERENCES:
#
#         P. AMESTOY, T.A. DAVIS, AND I.S. DUFF, "AN APPROXIMATE
#           MINIMUM DEGREE ALGORITHM," SIAM J. MATRIX ANAL. APPL.,
#           17 (1995), PP. 1404-1411.
#
#         J.W.H. LIU, "MODIFICATION OF THE MINIMUM DEGREE
#           ALGORITHM BY MULTIPLE ELIMINATION," ACM TRANS. MATH.
#           SOFTWARE, 11 (1985), PP. 141-153.
#
#     NOTES:
#       (1) QUOTIENT GRAPHS ARE USED TO REPRESENT THE ELIMINATION
#           GRAPHS.
#       (2) THE INPUT GRAPH IS DESTROYED DURING THE ELIMINATION
#           PROCESS.
#       (3) THE ADJACENCY STRUCTURE SHOULD BE PADDED WITH VACANT
#           ENTRIES AT THE END OF ADJNCY(*).  IT IS ADVISABLE
#           THAT THERE BE AT LEAST NEQNS SUCH VACANT ENTRIES.
#       (4) ALL ARRAYS AND VARIABLES ARE INTEGERS.
#       (5) VERTICES FALL INTO ONE OF FOUR CATEGORIES DURING THE 
#           ELIMINATION PROCESS.
#             (A) AN ACTIVE UNELIMINATED VERTEX OF THE QUOTIENT
#                 GRAPH.
#             (B) AN ACTIVE ELIMINATION CLIQUE REPRESENTATIVE
#                 IN THE QUOTIENT GRAPH.
#             (C) A MERGED ELIMINATION CLIQUE REPRESENTATIVE.
#             (D) A VERTEX ABSORBED INTO ANOTHER VERTEX'S
#                 SUPERNODE.
#           WORKING STORAGE VECTORS ARE USED IN DIFFERENT WAYS
#           FOR DIFFERENT CATEGORIES OF VERTICES.
#
#     INPUT PARAMETERS:
#       SCORE       - NODE-SELECTION SCORE (AN MLFSCORE).
#       NEQNS       - NUMBER OF EQUATIONS.
#       MAXINT      - MAXIMUM INTEGER VALUE WITH WHICH VERTICES CAN
#                     BE MARKED (SEE MARKER(*)).
#       ADJLEN      - LENGTH OF THE ADJACENCY STRUCTURE (ADJNCY(*)).
#                     IDEALLY AT LEAST NEQNS GREATER THAN INITIAL
#                     XADJ(NEQNS+1)-1 (SEE IFLAG).
#       ADJLEN2     - LENGTH OF THE WORK VECTOR (ADJ2(*)) INTO
#                     WHICH THE ADJACENCY STRUCTURE IS COPIED WHEN
#                     W-H UPDATING IS USED TO COMPUTE THE INITIAL
#                     DEFICIENCIES.
#       XADJ(*)     - ARRAY OF LENGTH NEQNS+1, CONTAINING POINTERS
#                     TO THE ADJACENCY STRUCTURE.
#       ADJNCY(*)   - ARRAY OF LENGTH ADJLEN, CONTAINING THE
#                     ADJACENCY STRUCTURE.
#       DEFFLAG     - DEFICIENCY FLAG.
#                          0 -  USING A STRAIGTFORWARD WAY TO COMPUTE
#                               THE INITIAL DEFICIENCIES.
#                       /= 0 -  COMPUTE THE DEFICIENCIES BY ADDING
#                               THE EDGES OF THE GIVEN GRAPH TO AN
#                               INITIALLY EMPTY GRAPH AND PERFORMING
#                               WING-HUANG UPDATING FOR EACH NEW EDGE.
#
#     OUTPUT PARAMETERS:
#       XADJ(*)     - DESTROYED ON OUTPUT.
#       ADJNCY(*)   - DESTROYED ON OUTPUT.
#       GBGCNT      - NUMBER OF GARBAGE COLLECTIONS USED TO
#                     RECOMPRESS THE QUOTIENT GRAPH WHEN A NEW
#                     ELIMINATION CLIQUE "RUNS OFF THE END" OF
#                     ADJNCY(*).
#       PERM(*)     - ENDS UP WITH THE PERMUTATION VECTOR, WHICH
#                     MAPS NEW NUMBERS TO OLD.
#       INVP(*)     - ENDS UP WITH THE INVERSE PERMUTATION VECTOR,
#                     WHICH MAPS OLD NUMBERS TO NEW.
#       NOFNZ       - NUMBER OF OFF-DIAGONAL NONZEROS IN THE FACTOR
#                     MATRIX PRODUCED BY THE ORDERING.
#       IFLAG       - ERROR FLAG
#                        0 -  NO ERROR OR WARNING, ADJLEN IS NO LESS
#                             THAN INITIAL XADJ(NEQNS+1)+NEQNS-1.
#                        1 -  WARNING, ADJLEN IS SMALL; IT IS
#                             BETWEEN INITIAL XADJ(NEQNS+1)-1 AND
#                             INITIAL XADJ(NEQNS+1)+NEQNS-1,
#                             INCLUSIVE.  MAY CAUSE EXCESSIVE
#                             GARBAGE COLLECTION.
#                       -1 -  ERROR, ADJLEN IS SMALLER THAN INITIAL
#                             XADJ(NEQNS+1)-1.  INSUFFICIENT ROOM
#                             IN ADJNCY(*) FOR THE ADJACENCY
#                             STRUCTURE, OR INCORRECT ADJLEN HAS
#                             BEEN INPUT.
#
#     WORK PARAMETERS:
#       INVP(*)     - ARRAY OF LENGTH NEQNS.
#                       ELIMINATED: -NUM, WHERE NUM IS THE NUMBER
#                                   ASSIGNED BY THE ELIMINATION
#                                   ORDER.
#                       OTHERWISE:  >=0
#       MARKER(*)   - ARRAY OF LENGTH NEQNS, USED TO MARK PREVIOUS
#                     VISITS TO VERTICES.
#                       NOT ABSORBED: < MAXINT.
#                       ABSORBED:     MAXINT.
#       NVTXS(*)    - ARRAY OF LENGTH NEQNS.
#                       ACTIVE ELIMINATED:  NUMBER OF VERTICES IN
#                                           THE ELIMINATION CLIQUE.
#                       MERGED ELIMINATED:  -1.
#                       UNELIMINATED:       NUMBER OF UNELIMINATED
#                                           VERTEX NEIGHBORS IN
#                                           CURRENT QUOTIENT GRAPH.
#                       ABSORBED:           -1.
#       WORK(*)     - ARRAY OF LENGTH NEQNS.
#                       ACTIVE ELIMINATED:  NOT USED AT THIS TIME.
#                       UNELIMINATED:       NUMBER OF ELIMINATION
#                                           CLIQUES VERTEX IS
#                                           ADJACENT TO IN THE
#                                           QUOTIENT GRAPH.
#                       ABSORBED:           MAPS ABSORBED VERTEX TO
#                                           MINUS THE ABSORBING
#                                           VERTEX (I.E.,
#                                           -ABSORBEE).
#       QSIZE(*)    - ARRAY OF SIZE NEQNS.
#                       UNELIMINATED: SUPERNODE SIZE.
#                       ABSORBED:     0.
#       ECFORW(*)   - ARRAY OF SIZE NEQNS.
#                       ELIMINATED: FORWARD LINKS OF A LIST OF THE 
#                                   ELIMINATION CLIQUES ORDERED BY
#                                   THE ELIMINATION.
#                       OTHERWISE:  -1.
#       DEFNCY(*)   - ARRAY OF SIZE NEQNS, CONTAINING THE
#                     DEFICIENCIES OF THE UNELIMINATED NODES
#                     COMPUTED WITH WING-HUANG UPDATING.
#       ADJ2(*)     - ARRAY OF LENGTH ADJLEN2, CONTAINS THE GROWING
#                     ADJACENCY LISTS USED TO PERFORM WING-HUANG 
#                     UPDATING DURING COMPUTATION OF INITIAL 
#                     DEFICIENCIES BY ROUTINES MFINIT_DEF.
#       CHANGED(*)  - ARRAY OF LENGTH NEQNS, USED TO MARK THE
#                     VERTICES WHOSE DEFICIENCIES ARE CHANGED BY
#                     THE CURRENT STEP OF THE ALGORITHM.
#                       -1 -  DEFICIENCY IS UNCHANGED
#                        1 -  DEFICIENCY IS POTENTIALLY CHANGED
#       DEGREE(*)   - ARRAY OF LENGTH NEQNS, USED TO STORE DEGREE
#                     INFORMATION.  DEGREE(UNODE) IS THE DEGREE
#                     OF UNODE IN THE CURRENT ELIMINATION GRAPH
#                     PLUS ONE.  THAT IS, IT IS THE SIZE OF THE
#                     CLOSED NEIGHBORHOOD.
#       HEAP(*)     - ARRAY OF SIZE 2*NEQNS, THE HEAP OF ACTIVE
#                     NODES BASED ON SOME DEFICIENCY-BASED SCORE. 
#       HEAPINV(*)  - ARRAY OF LENGTH NEQNS, GIVES LOCATION IN THE
#                     HEAP FOR EACH NODE IN THE HEAP.
#       ECLIQ(*)    - ARRAY OF LENGTH NEQNS, CONTAINS ENODE'S  
#                     ELIMINATION CLIQUE.  IT IS ALSO USED TO
#                     COLLECT OTHER NODES WHO NEED THEIR
#                     DEFICIENCY SCORES UPDATED.
#       XADJ2(*)    - ARRAY OF LENGTH NEQNS, USED TO STORE POINTERS
#                     TO THE DYNAMIC GROWING ADJACENCY LISTS NEEDED
#                     FOR COMPUTATION OF INITIAL DEFICIENCIES.
#                     USED IN OTHER CAPACITIES AS A WORK VECTOR.
#       LEN2(*)     - ARRAY OF LENGTH NEQNS, USED TO STORE THE
#                     LENGTHS OF THE DYNAMIC GROWING ADJACENCY
#                     LISTS NEEDED FOR COMPUTATION OF INITIAL
#                     DEFICIENCIES.
#                     USED IN OTHER CAPACITIES AS A WORK VECTOR.
#       WORK1(*)    - ARRAY OF LENGTH NEQNS, A WORK VECTOR.
#       UMASK(*)    - ARRAY OF LENGTH NEQNS.  BIT MASKS USED BY
#                     MFUPD_DEF TO MARK THE NEIGHBORS OF UP TO
#                     GROUPSIZE CLIQUE MEMBERS AT ONCE.  ITS STORAGE
#                     IS ALSO USED AS A MARKER VECTOR (UMARK) DURING
#                     INITIALIZATION; IT IS ZEROED BEFORE THE MAIN LOOP.
#       GQ(*), GUCNT(*), GPQ(*), GACC(*)
#                   - ARRAYS OF LENGTH 64, WORK VECTORS FOR MFUPD_DEF.
#       MATMIN      - CLIQUES OF AT LEAST MATMIN SUPERNODES ARE UPDATED
#                     BY MFUPD_MAT INSTEAD OF MFUPD_DEF (0 DISABLES THIS).
#       TWIN        - IF TRUE, THE FALSE TWINS OF EACH ELIMINATED NODE
#                     (UNTOUCHED NODES WITH THE SAME OPEN NEIGHBORHOOD)
#                     ARE ABSORBED INTO IT AND NUMBERED IN ITS BLOCK.
#                     THEY ARE SIMPLICIAL ONCE IT IS ELIMINATED, SO THE
#                     ORDERING REMAINS A GREEDY ONE FOR EVERY SCORE WHOSE
#                     VALUE AT A SIMPLICIAL NODE IS ITS MINIMUM (ALL BUT
#                     MININCDEG).
#
#**********************************************************************
#
samewght(a::Integer, b::Integer) = a == b
samewght(a, b) = isapprox(a, b)

function genmf_wh(
        score::MLFScore,
        neqns::V,
        adjlen::E,
        adjlen2::E,
        vwght::AbstractVector{W},
        xadj::AbstractVector{E},
        adjncy::AbstractVector{V},
        defflag::Bool,
        perm::AbstractVector{V},
        invp::AbstractVector{V},
        marker::AbstractVector{I},
        nvtxs::AbstractVector{V},
        work::AbstractVector{V},
        qsize::AbstractVector{W},
        qnmbr::AbstractVector{V},
        ecforw::AbstractVector{V},
        defncy::AbstractVector{W},
        adj2::AbstractVector{V},
        changed::AbstractVector{Bool},
        degree::AbstractVector{W},
        heap::AbstractVector{S},
        heapinv::AbstractVector{V},
        ecliq::AbstractVector{V},
        xadj2::AbstractVector{E},
        len2::AbstractVector{V},
        work1::AbstractVector{I},
        work2::AbstractVector{W},
        umask::AbstractVector{GMASK},
        gq::AbstractVector{W},
        gucnt::AbstractVector{W},
        gpq::AbstractVector{W},
        gacc::AbstractVector{W},
        twin::Bool,
        matmin::Int,
    ) where {V, E, I, W, S}

    #       ****************
    #       INITIALIZATIONS.
    #       ****************

    #       -----------------------------------------------------
    #       DURING INITIALIZATION, UMASK'S STORAGE SERVES AS THE
    #       MARKER VECTOR UMARK (IT IS ZEROED AGAIN BELOW).
    #       -----------------------------------------------------
    umark = reinterpret(I, umask)

    #       -------------------------------------
    #       COMPRESS THE GRAPH.
    #         INPUT:    NEQNS, MAXINT, ADJLEN
    #         MODIFIED: XADJ, ADJNCY
    #         OUTPUT:   DEGREE, MARKER, WORK, QSIZE
    #         WORK:     DEFNCY, UMARK, XADJ2
    #       -------------------------------------
    compress(
        neqns, adjlen, vwght, xadj, adjncy,
        degree, marker, work, qsize, qnmbr, work1,
        umark, xadj2
    )

    if !defflag
    
        #           -----------------------------------------------------
        #           COMPUTE INITIAL DEFICIENCIES BY THE STRAIGHTFORWARD
        #           APPROACH.
        #             INPUT:  NEQNS , ADJLEN, XADJ, ADJNCY, DEGREE, QSIZE
        #             OUTPUT: DEFNCY
        #             WORK:   UMARK
        #           -----------------------------------------------------
        mfinit_vanilla(
            neqns, adjlen, xadj, adjncy, degree,
            qsize, defncy, umark
        )
    
    else
    
        #           --------------------------------------------------------
        #           IF DEFICIENCY FLAG IS ON ...
        #           COMPUTE INITIAL DEFICIENCIES BY ADDING THE EDGES ONE
        #           AT A TIME TO AN INITIALLY EMPTY GRAPH AND PERFORMING
        #           WING-HUANG UPDATING FOR EACH NEW EDGE.
        #             INPUT:  NEQNS , ADJLEN, ADJLEN2, XADJ, ADJNCY, DEGREE,
        #               QSIZE
        #             OUTPUT: DEFNCY
        #             WORK:   MARKER, PERM, INVP, UMARK, XADJ2, LEN2  , ADJ2
        #           --------------------------------------------------------
        mfinit_def(
            neqns, adjlen, adjlen2, xadj, adjncy,
            degree, qsize, defncy, marker, perm,
            invp, work2, xadj2, len2, adj2
        )
    
    end

    #       ---------------------------------------------------
    #       QMIN IS THE SMALLEST POSITIVE VERTEX WEIGHT. IT IS
    #       ADDED TO THE SUPERNODE SIZE IN THE MINMEANFILL SCORE.
    #       ---------------------------------------------------
    qmin = zero(W)

    @inbounds for i in oneto(neqns)
        if ispositive(vwght[i])
            if iszero(qmin) || vwght[i] < qmin
                qmin = vwght[i]
            end
        end
    end

    #       ----------------------------------------------------------
    #       INITIALIZE VARIOUS VECTORS AND MAKE THE INITIAL HEAP BASED
    #       ON INITIAL DEFICIENCY-BASED SCORES.
    #         INPUT:  NEQNS, XADJ, DEGREE, QSIZE, DEFNCY, QMIN
    #         OUTPUT: HEAP, HEAPSIZE, HEAPINV, MARKER, NVTXS, WORK,
    #                 ECFORW, CHANGED, INVP
    #       ----------------------------------------------------------
    @inbounds for i in oneto(neqns)
        umask[i] = zero(GMASK)
    end

    heapsize = mfinit_heap(
        score, neqns, xadj, degree, qsize, defncy, qmin,
        heap, heapinv, marker, nvtxs,
        work, ecforw, changed, invp
    )

    #       WORKSPACE OF MFUPD_MAT
    ws = MatWork{W}()

    #       ------------------------------------------------------
    #       NOFNZ:  NUMBER OF NONZEROS IN FACTOR IS INITIALLY ZERO.
    #       ECHEAD, ECTAIL: EMPTY ELIMINATION CLIQUE LIST.
    #       GBGCNT: GARBAGE COLLECTION COUNT INITIALLY ZERO.
    #       TAG:    INITIAL MARKER VALUE IS ZERO.
    #       NXTLOC: POINTS TO FIRST LOCATION IN VACANT STORAGE
    #               REMAINING AT END OF ADJNCY(*).
    #       REMAIN: AMOUNT OF VACANT STORAGE REMAINING AT END OF
    #               ADJNCY(*).
    #       IFLAG:  TEST FOR SUFFICIENT WORK STORAGE IN ADJNCY(*).
    #       ------------------------------------------------------
    nofnz  = zero(W)
    echead = zero(V)
    ectail = zero(V)
    gbgcnt = zero(I)
    tag    = zero(I)
    nxtloc = xadj[neqns + one(V)]
    remain = adjlen - nxtloc + one(E)

    if remain >= convert(E, neqns)
        #           --------------------------------
        #           SUFFICIENT STORAGE (NO WARNING).
        #           --------------------------------
        iflag = zero(I)
    elseif !isnegative(remain) && remain <= convert(E, neqns) - one(E)
        #           ------------------------------------------------------
        #           SUFFICIENT STORAGE, BUT COULD BE TIGHT ENOUGH TO CAUSE
        #           MANY GARBAGE COLLECTIONS. (WARNING)
        #           ------------------------------------------------------
        iflag = one(I)
    else
        #           -----------------------------
        #           INSUFFICIENT STORAGE (ERROR).
        #           -----------------------------
        iflag = -one(I)
        return nofnz, gbgcnt, iflag
    end

    #       ------------------------------------------------
    #       MAIN LOOP:
    #       WHILE THERE REMAIN ANY NODES TO ELIMINATE ...
    #
    #       NUM: NEXT NUMBER TO BE ASSIGNED BY THE ORDERING.
    #       ------------------------------------------------
    num = one(V)

    @inbounds while num <= neqns

        #           --------------------------------------------
        #           DELETE THE MINIMUM-SCORE NODE FROM THE HEAP.
        #             ENODE:  MINIMUM-SCORE NODE
        #           --------------------------------------------
        enode = convert(V, heap[two(V)])
        heapsize = del_heap(heap, heapsize, heapinv, enode)
    
        #           --------------------------------------------------------
        #           COMPUTE THE NEW ELIMINATION CLIQUE FORMED BY ELIMINATING
        #           ENODE.
        #             INPUT:    ENODE, NEQNS, ADJLEN, XADJ, ADJNCY, NVTXS,
        #                       WORK
        #             MODIFIED: QSIZE, CHANGED
        #             OUTPUT:   CLQSIZ, FNODE, NNODES, ECLIQ, CLIQUE
        #           --------------------------------------------------------
        clqsiz, fnode, nnodes = elmclq(
            enode, neqns, adjlen, xadj, adjncy,
            nvtxs, work, qsize, changed, ecliq
        )

        #           -----------------------------------------------------------
        #           ABSORB THE FALSE TWINS OF ENODE: THE UNTOUCHED NODES VNODE
        #           OUTSIDE ENODE'S ELIMINATION CLIQUE R WITH N(VNODE) = R.
        #           ONCE ENODE IS ELIMINATED, R IS A CLIQUE AND EACH VNODE IS
        #           SIMPLICIAL (SCORE 0), SO VNODE IS NUMBERED IN ENODE'S
        #           BLOCK.  EVERY VNODE IS IN EVERY VERTEX LIST OF R, SO
        #           SCANNING THE SHORTEST ONE FINDS THEM ALL.
        #             QFT:  TOTAL WEIGHT OF THE ABSORBED NODES.
        #             MUFT: WEIGHTED NUMBER OF PAIRS AMONG THEM (ALL MISSING).
        #           -----------------------------------------------------------
        qft = zero(W)
        muft = zero(W)

        if twin && ispositive(nnodes)
            rnode = ecliq[1]

            for j in 2:nnodes
                if nvtxs[ecliq[j]] < nvtxs[rnode]
                    rnode = ecliq[j]
                end
            end

            istart = xadj[rnode]
            istop = istart + convert(E, nvtxs[rnode]) - one(E)

            for i in istart:istop
                vnode = adjncy[i]
                qv = qsize[vnode]

                #               VNODE IS ACTIVE, OUTSIDE R AND ENODE, UNTOUCHED,
                #               AND W(N(VNODE)) = W(R)
                if ispositive(qv) && iszero(work[vnode]) && samewght(degree[vnode] - qv, clqsiz)
                    #               ... AND EVERY ACTIVE NODE IN ITS LIST IS IN R,
                    #               AND IT HAS NNODES OF THEM: N(VNODE) = R.
                    istwin = true
                    count = zero(V)
                    kstart = xadj[vnode]
                    kstop = kstart + convert(E, nvtxs[vnode]) - one(E)

                    for k in kstart:kstop
                        knode = adjncy[k]

                        if marker[knode] < maxint(I)
                            if !isnegative(qsize[knode])
                                istwin = false
                                break
                            end

                            count += one(V)
                        end
                    end

                    if istwin && count == nnodes
                        heapsize = del_heap(heap, heapsize, heapinv, vnode)
                        muft += qft * qv
                        qft += qv
                        work[vnode] = -enode
                        marker[vnode] = maxint(I)
                        qnmbr[enode] += qnmbr[vnode]
                        qsize[vnode] = zero(W)
                        qnmbr[vnode] = zero(V)
                    end
                end
            end
        end
    
        #           ----------------------------------------------------------
        #           INITIALLY NO NNODES EXTERNAL TO ENODE'S ELIMINATION CLIQUE
        #           HAVE HAD THEIR DEFICIENCIES REDUCED. (NNODES2 = 0)
        #           ----------------------------------------------------------
    
        #           ---------------------------------------------------------
        #           PERFORM THE WING-HUANG UPDATES TO THE DEFICIENCIES DUE TO
        #           EDGES ADDED BY THE ELIMINATION OF ENODE.
        #             INPUT:    CONFLAG, CONDONE, ENODE, NEQNS, MAXINT, 
        #                       ADJLEN, XADJ, ADJNCY, NVTXS, WORK,
        #                       QSIZE, NNODES, FNODE
        #             MODIFIED: NNODES2, ECLIQ, DEGREE, DEFNCY, TAG, MARKER,
        #                       CHANGED
        #             WORK:     WORK2, UMASK, LEN2, GQ, GUCNT, GPQ, GACC
        #           ---------------------------------------------------------
        if ispositive(matmin) && nnodes >= matmin
            nnodes2 = mfupd_mat!(
                ws, neqns, enode, nnodes, ecliq, xadj,
                adjncy, nvtxs, work, qsize, marker,
                changed, degree, defncy, qft
            )
        else
            tag, nnodes2 = mfupd_def(
                enode, neqns, adjlen, xadj,
                adjncy, nvtxs, work, qsize, nnodes,
                fnode, ecliq, degree, defncy,
                tag, marker, changed, work2,
                umask, len2, gq, gucnt, gpq, gacc
            )
        end
    
        #           -------------------------------------------------------
        #           PERFORM THE ELIMINATION GRAPH TRANSFORMATION DUE TO THE
        #           ELIMINATION OF ENODE ...
        #             INPUT:    ENODE, NEQNS, ADJLEN, MAXINT, NNODES,
        #                       ECLIQ, INVP
        #             MODIFIED: TAG, XADJ, ADJNCY, NVTXS, WORK,
        #                       QSIZE, MARKER, ECHEAD, ECTAIL, ECFORW,
        #                       NXTLOC, GBGCNT, CLQSIZ, HEAPINV (AS ECOUNT)
        #             WORK:     WORK1, XADJ2
        #           -------------------------------------------------------
        tag, echead, ectail, nxtloc, gbgcnt, clqsiz = elmtra(
            enode, neqns, adjlen, nnodes,
            ecliq, invp, tag, xadj, adjncy,
            nvtxs, work, qsize, qnmbr, marker, echead,
            ectail, ecforw, nxtloc, gbgcnt, clqsiz,
            perm, len2, heapinv, ispositive(qft)
        )
    
        #           -------------------------------------------------------
        #           PERFORM FINAL WING-HUANG UPDATE TO THE DEFICIENCIES FOR
        #           EACH NODE UNODE IN ENODE'S ELIMINATION CLIQUE, DUE TO
        #           THE REMOVAL OF ENODE FROM THE GRAPH (NOT DUE TO THE NEW
        #           FILL EDGES).
        #           -------------------------------------------------------
        qe = qsize[enode]
        de = degree[enode]
        jstart = xadj[enode]
        jstop = jstart + convert(E, nvtxs[enode]) - one(E)

        for j in jstart:jstop
            unode = adjncy[j]
            du = degree[unode]
            defncy[unode] = defncy[unode] - (du - de) * qe
        end

        #           ---------------------------------------------------------
        #           ... AND THE PAIRS JOINING AN ABSORBED FALSE TWIN TO A NODE
        #           OUTSIDE R AND ENODE'S SUPERNODE, OR TO ANOTHER FALSE TWIN.
        #           (PAIRS JOINING IT TO ENODE'S SUPERNODE ARE COUNTED ABOVE.)
        #           ---------------------------------------------------------
        if ispositive(qft)
            for j in jstart:jstop
                unode = adjncy[j]
                du = degree[unode]
                defncy[unode] = defncy[unode] - (du - de - qft) * qft - muft
                degree[unode] = degree[unode] - qft
            end
        end

        #           -----------------------------------------------
        #           UPDATE THE DEGREE OF EACH NODE UNODE IN ENODE'S
        #           ELIMINATION CLIQUE TO REFLECT THE REMOVAL OF
        #           ENODE.
        #           -----------------------------------------------
        jstart = xadj[enode]
        jstop = jstart + convert(E, nvtxs[enode]) - one(E)

        for j in jstart:jstop
            unode = adjncy[j]
            degree[unode] = degree[unode] - qsize[enode]
        end
    
        #           ---------------------------------------------------------
        #           ... RECORD INFO FOR ENODE.
        #
        #           NOFNZ:  ACCUMULATES THE NUMBER OF NONZEROS IN THE FACTOR.
        #                   THIS IS RETURNED SO THAT IT CAN BE COMPARED WITH
        #                   THE RESULTS OF A SYMBOLIC FACTORIZATION.
        #           NUM:    ENODE'S NUMBER IN THE ELIMINATION ORDERING.
        #           CLQSIZ: ENODE'S ELIMINATION CLIQUE SIZE.
        #           ---------------------------------------------------------
        qe = qsize[enode]
        nofnz += qe * (qe + clqsiz) - half((qe - one(W)) * qe)
        invp[enode] = -num
        num += qnmbr[enode]
    
        #           ---------------------------------------------------------
        #           FOR EACH NODE JNODE WHOSE DEFICIENCY HAS BEEN CHANGED ...
        #           ---------------------------------------------------------
        for j in oneto(nnodes + nnodes2)
        
            jnode = ecliq[j]
        
            if iszero(qsize[jnode])
                #                   -------------------------------------
                #                   DELETE ABSORBED VERTEX FROM THE HEAP.
                #                   -------------------------------------
                heapsize = del_heap(heap, heapsize, heapinv, jnode)
            else
                #                   ------------------------------------
                #                   DJ WILL BE THE DEGREE OF JNODE.
                #                   (PLUS ONE).
                #                   QJ IS THE SIZE OF JNODE'S SUPERNODE.
                #                   ------------------------------------
                #                   -------------------------------
                #                   COMPUTE DEFICIENCY-BASED SCORE.
                #                   -------------------------------
                key = mlfscore(score, defncy, degree, qsize, qmin, jnode)
            
                #                   -------------------
                #                   RESTORE HEAP ORDER.
                #                   -------------------
                if j <= nnodes
                    #                       ------------------------------------------
                    #                       JNODE IS IN THE ELIMINATION CLIQUE,
                    #                       SO DSCORE MAY HAVE INCREASED OR DECREASED.
                    #                       ------------------------------------------
                    mod_heap(heap, heapsize, heapinv, jnode, key)
                else
                    #                       ------------------------------------------
                    #                       JNODE IS NOT IN THE ELIMINATION CLIQUE,
                    #                       SO DSCORE HAS DECREASED.
                    #                       ------------------------------------------
                    hindex = heapinv[jnode]
                    heap[twice(hindex) - one(V)] = key
                    move_up(heap, heapsize, hindex, heapinv)
                end
            end
            #               ------------------------
            #               MARK JNODE AS UNCHANGED.
            #               ------------------------
            changed[jnode] = false
            #               ----------------------
            #               NEXT NODE IN THE LIST.
            #               ----------------------
        end
    
        #           -----------------------------------------------
        #           END OF MAIN LOOP:
        #           GET NEXT NODE OF MINIMUM SCORE TO ELIMINATE ...
        #           -----------------------------------------------
    end

    #       ------------------------------------------------------
    #       NUMBER THE VERTICES ACCORDING THE THE ORDER GENERATED.
    #         INPUT:    NEQNS, WORK
    #         MODIFIED: INVP
    #         OUTPUT:   PERM 
    #       ------------------------------------------------------
    mfnumn(neqns, work, invp, perm)

    return nofnz, gbgcnt, iflag
end
