#**********************************************************************
#**********************************************************************
#*****   MFUPD_MAT ... DEFICIENCY UPDATING IN MATRIX FORM   ***********
#**********************************************************************
#**********************************************************************
#
#     PURPOSE:
#       THIS SUBROUTINE PERFORMS THE SAME UPDATE AS MFUPD_DEF, WITH THE
#       SAME RESULT: THE SAME DEFICIENCIES (EXACTLY FOR INTEGER WEIGHTS,
#       UP TO ROUNDING FOR FLOATING-POINT WEIGHTS), DEGREES, AND SET OF
#       CHANGED NODES.  ONLY THE ORDER IN WHICH THE CHANGED EXTERNAL
#       NODES ARE RECORDED MAY DIFFER.  IT IS USED FOR LARGE ELIMINATION
#       CLIQUES.
#
#       LET K BE ENODE'S ELIMINATION CLIQUE, q THE SUPERNODE SIZES,
#       S_y = N(y) ∩ K FOR EACH NODE y OUTSIDE K THAT TOUCHES K (A ROW),
#       AND F_u = K \ N[u] FOR u IN K (THE FILL PARTNERS OF u).  WITH
#
#           C[y, u] = w(S_y ∩ F_u)   FOR u IN S_y,
#
#         - y LOSES THE FILL PAIRS INSIDE S_y:  (1/2) sum_u q_u C[y, u];
#         - u IN K LOSES THE FILL PAIRS INSIDE N(u) ∩ K (COMPUTED FROM F
#           ALONE: FILL(K) MINUS THE PAIRS TOUCHING F_u ∪ {u}), AND GAINS
#           THE NEW MISSING PAIRS  w(F_u) w(X_u) - sum_y q_y C[y, u],
#           WHERE X_u IS ITS SET OF NEIGHBORS OUTSIDE K ∪ {ENODE}.
#
#       ROWS ARE BITSETS OVER K, BUILT BY OR-ING ONE BITSET PER ELEMENT
#       TOUCHING K, SO THAT EACH ELEMENT LIST IS SCANNED A FIXED NUMBER
#       OF TIMES.  THE ROWS OF F ARE KEPT AS LISTS OF THEIR NONZERO
#       WORDS.  C IS COMPUTED WITH AND + POPCOUNT.
#
#       WEIGHTS: THE (UP TO NCLASS) MOST COMMON WEIGHTS IN K, EACH HELD
#       BY MORE THAN |K| / 16 MEMBERS, ARE CLASSES q_1, q_2, ...  THEY
#       ARE FOUND WITH THE MISRA-GRIES HEAVY HITTERS ALGORITHM (ONE PASS
#       WITH NCAND = 15 COUNTERS FINDS EVERY WEIGHT HELD BY MORE THAN
#       |K| / 16 MEMBERS; A SECOND PASS COUNTS THEM).  THE OTHER MEMBERS
#       ARE HEAVY.  A WEIGHTED COUNT OF A WORD v IS
#
#           q_1 POPCOUNT(v) + sum_r (q_r - q_1) POPCOUNT(v & M_r)
#                           + (q - q_1) FOR EACH HEAVY BIT OF v,
#
#       WHERE M_r IS THE MASK OF CLASS r.  ANY NUMERIC WEIGHTS WORK.  IT
#       IS FASTEST WHEN FEW MEMBERS ARE HEAVY (INTEGER SUPERNODE SIZES,
#       OR A FEW DISTINCT FLOATING-POINT WEIGHTS).  WITHOUT ANY CLASS
#       (E.G. ALL WEIGHTS DISTINCT), q_1 = 0 AND EVERY MEMBER IS HEAVY.
#       WHEN A QUARTER OR MORE OF A LARGE CLIQUE IS HEAVY, THE HEAVY SUM
#       OF A WORD IS READ FROM BYTE TABLES (8 LOOKUPS OF PRECOMPUTED SUMS
#       OF q - q_1, ONE PER BYTE); OTHERWISE IT IS ADDED BIT BY BIT.
#
#       FALSE TWINS ABSORBED INTO ENODE (WEIGHT QFT) ARE INVISIBLE HERE,
#       AS IN MFUPD_DEF.
#
#     INPUT PARAMETERS:
#       NEQNS       - NUMBER OF EQUATIONS.
#       ENODE       - ELIMINATION NODE.
#       NNODES      - NUMBER OF (SUPER)NODES IN ENODE'S ELIMINATION
#                     CLIQUE.
#       XADJ(*)     - ARRAY OF LENGTH NEQNS+1, CONTAINING THE
#                     POINTERS FOR THE CURRENT QUOTIENT GRAPH.
#       ADJNCY(*)   - ARRAY CONTAINING THE CURRENT QUOTIENT GRAPH.
#       NVTXS(*)    - ARRAY OF LENGTH NEQNS, CONTAINING THE NUMBER
#                     OF VERTEX NEIGHBORS FOR AN UNELIMINATED NODE, AND
#                     THE NUMBER OF NODES IN THE CLIQUE FOR AN
#                     ELIMINATED NODE.
#       WORK(*)     - ARRAY OF LENGTH NEQNS, CONTAINING THE NUMBER
#                     OF ELIMINATION CLIQUES TO WHICH AN UNELIMINATED
#                     NODE BELONGS.
#       QSIZE(*)    - ARRAY OF LENGTH NEQNS,
#                       <0 - NEGATIVE OF SUPERNODE SIZE FOR EACH NODE
#                            IN ENODE'S ELIMINATION CLIQUE (AND ENODE).
#                       >0 - SUPERNODE SIZE FOR EACH UNELIMINATED
#                            NODE NOT IN ENODE'S ELIMINATION CLIQUE.
#                       =0 - ABSORBED.
#       MARKER(*)   - ARRAY OF LENGTH NEQNS; MAXINT FOR ABSORBED NODES.
#       QFT         - WEIGHT OF THE FALSE TWINS ABSORBED INTO ENODE.
#
#     MODIFIED PARAMETERS:
#       ECLIQ(*)    - ARRAY OF LENGTH NEQNS.
#                     ON INPUT CONTAINS ENODE'S ELIMINATION CLIQUE.
#                     ON OUTPUT, ALSO CONTAINS THE EXTERNAL NODES
#                     WHOSE DEFICIENCIES HAVE CHANGED.
#       CHANGED(*)  - ARRAY OF LENGTH NEQNS, MARKS THE NODES WHOSE
#                     DEFICIENCIES HAVE CHANGED.
#       DEGREE(*)   - ARRAY OF LENGTH NEQNS, THE DEGREE OF EACH NODE
#                     IN THE CURRENT ELIMINATION GRAPH PLUS ITS SIZE.
#       DEFNCY(*)   - ARRAY OF LENGTH NEQNS, THE DEFICIENCIES.
#
#     OUTPUT:
#       NNODES2     - THE NUMBER OF EXTERNAL NODES WHOSE DEFICIENCIES
#                     HAVE CHANGED (RETURNED).
#
#     WORKING PARAMETERS:
#       WS          - A MATWORK: FIVE INTEGER VECTORS OF LENGTH NEQNS
#                     (ALLOCATED ON FIRST USE), AND BITSETS OF
#                     O((|K| + ROWS + ELEMENTS) |K| / 64) WORDS FOR THE
#                     CURRENT CLIQUE.
#
#**********************************************************************
#
mutable struct MatWork{W}
    kpos::Vector{Int}        # position of each member in K (0 outside K)
    rtag::Vector{Int}        # tag of the elimination that gave a node its row
    row::Vector{Int}         # row of each outside node
    etag::Vector{Int}        # tag of the elimination that saw an element
    eidx::Vector{Int}        # index of each element touching K
    rnode::Vector{Int}       # node of each row
    S::Vector{UInt64}        # rows S_y
    KE::Vector{UInt64}       # K ∩ L_e for each element e touching K
    elist::Vector{Int}       # elements touching K
    N::Vector{UInt64}        # N(u) ∩ K for each member u
    fp::Vector{Int}          # pointers to the nonzero words of each F_u
    fk::Vector{Int}          # word index of each nonzero word of F
    fv::Vector{UInt64}       # the nonzero words of F
    fh::Vector{UInt64}       # their heavy bits
    fx::Vector{UInt64}       # their bits in classes 2, 3, ...
    H::Vector{UInt64}        # mask of the heavy members
    CM::Vector{UInt64}       # masks of classes 2, 3, ...
    KF::Vector{UInt64}       # mask of the members with fill partners
    D::Vector{UInt64}        # F_u ∪ {u}, for one member at a time
    mgq::Vector{W}           # Misra-Gries candidate weights
    mgc::Vector{Int}         # Misra-Gries counters
    cq::Vector{W}            # class weights
    qk::Vector{W}            # weight of each member
    fw::Vector{W}            # w(F_u)
    wn::Vector{W}            # w(N(u) ∩ K)
    gacc::Vector{W}          # sum_y q_y C[y, u]
    T::Vector{W}             # byte tables of the heavy weights
    tag::Int
end

function MatWork{W}() where {W}
    return MatWork{W}(Int[], Int[], Int[], Int[], Int[], Int[], UInt64[], UInt64[], Int[], UInt64[], Int[], Int[],
        UInt64[], UInt64[], UInt64[], UInt64[], UInt64[], UInt64[], UInt64[], zeros(W, NCAND), zeros(Int, NCAND),
        zeros(W, NCLASS), W[], W[], W[], W[], W[], 0)
end

const NCLASS = 4    # at most 4: mfupd_mat_kern! has three counters after the first class
const NCAND = 15    # Misra-Gries counters: finds every weight held by more than |K| / 16 members

@inline function mbit(j::Int)
    return one(UInt64) << ((j - 1) & 63)
end

@inline function mwrd(j::Int)
    return ((j - 1) >> 6) + 1
end

function halve(x::W) where {W <: Integer}
    return x ÷ convert(W, 2)
end

function halve(x)
    return x / 2
end

# sum of q - q_1 over the heavy bits h of word k, bit by bit
@inline function hsum(h::UInt64, k::Int, qk::AbstractVector{W}, q1::W) where {W}
    c = zero(W)

    @inbounds while !iszero(h)
        t = trailing_zeros(h)
        h &= h - one(UInt64)
        c += qk[(k - 1) * 64 + t + 1] - q1
    end

    return c
end

# byte tables of the heavy weights: T[2048(k - 1) + 256b + v + 1] is the sum of
# q - q_1 over the bits of the byte value v at byte b of word k
function hsum_tables!(T::AbstractVector{W}, qk::AbstractVector{W}, q1::W, nK::Int, NW::Int) where {W}
    @inbounds for k in 1:NW, b in 0:7
        o = 2048 * (k - 1) + 256b
        T[o + 1] = zero(W)

        for v in 1:255
            j = 64 * (k - 1) + 8b + trailing_zeros(v) + 1
            T[o + v + 1] = T[o + (v & (v - 1)) + 1]

            if j <= nK
                T[o + v + 1] += qk[j] - q1
            end
        end
    end

    return T
end

# sum of q - q_1 over the heavy bits h of word k, from the byte tables
@inline function hsum(h::UInt64, k::Int, T::AbstractVector{W}) where {W}
    o = 2048 * (k - 1) + 1

    @inbounds c = ((T[o + Int(h & 0xff)] + T[o + 256 + Int((h >> 8) & 0xff)]) +
        (T[o + 512 + Int((h >> 16) & 0xff)] + T[o + 768 + Int((h >> 24) & 0xff)])) +
        ((T[o + 1024 + Int((h >> 32) & 0xff)] + T[o + 1280 + Int((h >> 40) & 0xff)]) +
        (T[o + 1536 + Int((h >> 48) & 0xff)] + T[o + 1792 + Int(h >> 56)]))

    return c
end

# the heavy part, bit by bit or from the tables
@inline function hpart(::Val{false}, h::UInt64, k::Int, qk::AbstractVector{W}, q1::W, T::AbstractVector{W}) where {W}
    return hsum(h, k, qk, q1)
end

@inline function hpart(::Val{true}, h::UInt64, k::Int, qk::AbstractVector{W}, q1::W, T::AbstractVector{W}) where {W}
    return hsum(h, k, T)
end

# weighted count of the bits of v, in word k
@inline function wcount(v::UInt64, k::Int, H::AbstractVector{UInt64}, CM::AbstractVector{UInt64}, NW::Int,
        qk::AbstractVector{W}, q1::W, dq::NTuple{X, W}, tab::Val, T::AbstractVector{W}) where {W, X}
    c = q1 * convert(W, count_ones(v))

    @inbounds for x in 1:X
        c += dq[x] * convert(W, count_ones(v & CM[(x - 1) * NW + k]))
    end

    @inbounds h = v & H[k]

    if !iszero(h)
        c += hpart(tab, h, k, qk, q1, T)
    end

    return c
end

# weighted count of the bits of s in fill entry i
@inline function fcount(s::UInt64, i::Int, fk::AbstractVector{Int}, fv::AbstractVector{UInt64}, fh::AbstractVector{UInt64},
        fx::AbstractVector{UInt64}, qk::AbstractVector{W}, q1::W, dq::NTuple{X, W}, tab::Val, T::AbstractVector{W}) where {W, X}
    @inbounds c = q1 * convert(W, count_ones(s & fv[i]))

    @inbounds for x in 1:X
        c += dq[x] * convert(W, count_ones(s & fx[(i - 1) * X + x]))
    end

    @inbounds h = s & fh[i]

    if !iszero(h)
        @inbounds c += hpart(tab, h, fk[i], qk, q1, T)
    end

    return c
end

function mfupd_mat!(ws::MatWork{W}, neqns::V, enode::V, nnodes::V, ecliq::AbstractVector{V}, xadj::AbstractVector{E},
        adjncy::AbstractVector{V}, nvtxs::AbstractVector{V}, work::AbstractVector{V}, qsize::AbstractVector{W},
        marker::AbstractVector{I}, changed::AbstractVector{Bool}, degree::AbstractVector{W}, defncy::AbstractVector{W},
        qft::W) where {V, E, I, W}

    #       -------------------
    #       LOCAL VARIABLES ...
    #       -------------------
    p = Int(enode)
    nK = Int(nnodes)
    NW = cld(nK, 64)
    maxI = maxint(I)
    ws.tag += 1
    tag = ws.tag
    qp = -qsize[p]

    #       --------------------------------------------------
    #       SIZE THE WORKSPACE.  THE VECTORS OF LENGTH NEQNS ARE
    #       ALLOCATED ON FIRST USE.
    #       --------------------------------------------------
    if length(ws.kpos) < neqns
        for A in (ws.kpos, ws.rtag, ws.row, ws.etag, ws.eidx)
            resize!(A, neqns)
            fill!(A, 0)
        end
    end

    for A in (ws.qk, ws.fw, ws.wn, ws.gacc)
        if length(A) < nK
            resize!(A, 2nK)
        end
    end

    if length(ws.N) < nK * NW
        resize!(ws.N, 2nK * NW)
    end

    if length(ws.fp) < nK + 1
        resize!(ws.fp, 2nK + 2)
    end

    if length(ws.CM) < NW * NCLASS
        resize!(ws.CM, 2NW * NCLASS)
    end

    for A in (ws.KF, ws.D, ws.H)
        if length(A) < NW
            resize!(A, 2NW)
        end
    end

    kpos = ws.kpos
    rtag = ws.rtag
    row = ws.row
    etag = ws.etag
    eidx = ws.eidx
    elist = ws.elist
    qk = ws.qk
    fw = ws.fw
    wn = ws.wn
    gacc = ws.gacc
    N = ws.N
    H = ws.H
    CM = ws.CM
    mgq = ws.mgq
    mgc = ws.mgc
    cq = ws.cq

    #       ---------------------------------
    #       POSITIONS AND WEIGHTS OF MEMBERS.
    #       ---------------------------------
    wK = zero(W)

    @inbounds for j in 1:nK
        unode = Int(ecliq[j])
        kpos[unode] = j
        qk[j] = -qsize[unode]
        wK += qk[j]
    end

    #       --------------------------------------------------------
    #       PASS 1: THE ELEMENTS TOUCHING K, AND A ROW FOR EACH NODE
    #       OUTSIDE K THAT TOUCHES IT.
    #       --------------------------------------------------------
    nE = 0
    nY = 0
    empty!(elist)

    @inbounds for j in 1:nK
        unode = Int(ecliq[j])
        istart = Int(xadj[unode])
        istop = istart + Int(nvtxs[unode]) - 1
        cstop = istop + Int(work[unode])

        for i in istart:istop
            ynode = Int(adjncy[i])

            if ynode != p && marker[ynode] < maxI && ispositive(qsize[ynode]) && rtag[ynode] != tag
                nY += 1
                rtag[ynode] = tag
                row[ynode] = nY
            end
        end

        for c in (istop + 1):cstop
            cnode = Int(adjncy[c])

            if etag[cnode] != tag
                nE += 1
                etag[cnode] = tag
                eidx[cnode] = nE
                push!(elist, cnode)
            end
        end
    end

    if length(ws.KE) < nE * NW
        resize!(ws.KE, 2nE * NW + 64)
    end

    KE = ws.KE

    @inbounds for k in 1:(nE * NW)
        KE[k] = zero(UInt64)
    end

    @inbounds for e in 1:nE
        cnode = elist[e]
        eb = (e - 1) * NW
        istart = Int(xadj[cnode])
        istop = istart + Int(nvtxs[cnode]) - 1

        for i in istart:istop
            ynode = Int(adjncy[i])
            jj = kpos[ynode]

            if ispositive(jj)
                if marker[ynode] < maxI
                    KE[eb + mwrd(jj)] |= mbit(jj)
                end
            elseif ynode != p && marker[ynode] < maxI && ispositive(qsize[ynode]) && rtag[ynode] != tag
                nY += 1
                rtag[ynode] = tag
                row[ynode] = nY
            end
        end
    end

    if length(ws.rnode) < nY
        resize!(ws.rnode, 2nY + 16)
    end

    if length(ws.S) < nY * NW
        resize!(ws.S, 2nY * NW + 64)
    end

    rnode = ws.rnode
    S = ws.S

    @inbounds for k in 1:(nY * NW)
        S[k] = zero(UInt64)
    end

    @inbounds for k in 1:(nK * NW)
        N[k] = zero(UInt64)
    end

    #       --------------------------------------------------
    #       PASS 2: N(u) ∩ K FOR EACH MEMBER u, AND THE ROWS S_y.
    #       --------------------------------------------------
    @inbounds for j in 1:nK
        unode = Int(ecliq[j])
        nb = (j - 1) * NW
        istart = Int(xadj[unode])
        istop = istart + Int(nvtxs[unode]) - 1
        cstop = istop + Int(work[unode])

        for i in istart:istop
            ynode = Int(adjncy[i])

            if ynode != p && marker[ynode] < maxI && !iszero(qsize[ynode])
                jj = kpos[ynode]

                if ispositive(jj)
                    N[nb + mwrd(jj)] |= mbit(jj)
                else
                    r = row[ynode]
                    rnode[r] = ynode
                    S[(r - 1) * NW + mwrd(j)] |= mbit(j)
                end
            end
        end

        for c in (istop + 1):cstop
            eb = (eidx[Int(adjncy[c])] - 1) * NW

            for k in 1:NW
                N[nb + k] |= KE[eb + k]
            end
        end
    end

    @inbounds for e in 1:nE
        cnode = elist[e]
        eb = (e - 1) * NW
        istart = Int(xadj[cnode])
        istop = istart + Int(nvtxs[cnode]) - 1

        for i in istart:istop
            ynode = Int(adjncy[i])

            if ynode != p && marker[ynode] < maxI && ispositive(qsize[ynode])
                r = row[ynode]
                rnode[r] = ynode
                rb = (r - 1) * NW

                for k in 1:NW
                    S[rb + k] |= KE[eb + k]
                end
            end
        end
    end

    #       ---------------------------------------------------------
    #       WEIGHT CLASSES.  MISRA-GRIES: AFTER ONE PASS WITH NCAND
    #       COUNTERS, EVERY WEIGHT HELD BY MORE THAN |K| / (NCAND + 1)
    #       MEMBERS HAS A POSITIVE COUNTER.
    #       ---------------------------------------------------------
    for x in 1:NCAND
        mgc[x] = 0
    end

    @inbounds for j in 1:nK
        q = qk[j]
        found = false

        for x in 1:NCAND
            if !found && ispositive(mgc[x]) && mgq[x] == q
                mgc[x] += 1
                found = true
            end
        end

        for x in 1:NCAND
            if !found && iszero(mgc[x])
                mgq[x] = q
                mgc[x] = 1
                found = true
            end
        end

        if !found
            for x in 1:NCAND
                mgc[x] -= 1
            end
        end
    end

    #       ------------------------------------------------------
    #       COUNT THE CANDIDATES EXACTLY (-1 MARKS AN EMPTY COUNTER).
    #       ------------------------------------------------------
    for x in 1:NCAND
        if ispositive(mgc[x])
            mgc[x] = 0
        else
            mgc[x] = -1
        end
    end

    @inbounds for j in 1:nK
        q = qk[j]

        for x in 1:NCAND
            if !isnegative(mgc[x]) && mgq[x] == q
                mgc[x] += 1
            end
        end
    end

    #       --------------------------------------------------------
    #       THE CLASSES: THE (UP TO NCLASS) CANDIDATES WITH THE MOST
    #       MEMBERS, IF THEY HAVE MORE THAN |K| / 16.
    #       --------------------------------------------------------
    nclass = 0

    for r in 1:NCLASS
        xbest = 0

        for x in 1:NCAND
            if 16mgc[x] > nK
                if iszero(xbest) || mgc[x] > mgc[xbest]
                    xbest = x
                end
            end
        end

        if ispositive(xbest)
            nclass += 1
            cq[nclass] = mgq[xbest]
            mgc[xbest] = -1
        end
    end

    if ispositive(nclass)
        q1 = cq[1]
    else
        q1 = zero(W)
    end

    #       ---------------------------------------------
    #       THE CLASS MASKS (AFTER THE FIRST), AND THE MASK
    #       OF THE HEAVY MEMBERS.
    #       ---------------------------------------------
    @inbounds for k in 1:(NW * NCLASS)
        CM[k] = zero(UInt64)
    end

    @inbounds for k in 1:NW
        H[k] = zero(UInt64)
    end

    nH = 0

    @inbounds for j in 1:nK
        x = 0

        for r in 1:nclass
            if iszero(x) && qk[j] == cq[r]
                x = r
            end
        end

        if iszero(x)
            H[mwrd(j)] |= mbit(j)
            nH += 1
        elseif x > 1
            CM[(x - 2) * NW + mwrd(j)] |= mbit(j)
        end
    end

    #       ------------------------------------------------------
    #       BYTE TABLES, IF A QUARTER OR MORE OF THE MEMBERS ARE
    #       HEAVY (AND THE CLIQUE IS LARGE ENOUGH TO AMORTIZE THEM).
    #       ------------------------------------------------------
    if nK >= 128 && 4nH >= nK
        if length(ws.T) < 2048NW
            resize!(ws.T, 4096NW)
        end

        hsum_tables!(ws.T, qk, q1, nK, NW)
        tab = Val(true)
    else
        tab = Val(false)
    end

    #       --------------------------------------------------------
    #       THE FILL ROWS, THE PRODUCT, AND THE FILL PAIRS INSIDE
    #       N(u) ∩ K.  (THE KERNEL IS SPECIALIZED ON THE CLASSES.)
    #       --------------------------------------------------------
    if nclass <= 1
        fillK, k2 = mfupd_mat_kern!(ws, (), q1, tab, nK, NW, nY, wK, S, qsize, rnode, ecliq, changed, defncy)
    elseif nclass == 2
        dq = (cq[2] - q1,)
        fillK, k2 = mfupd_mat_kern!(ws, dq, q1, tab, nK, NW, nY, wK, S, qsize, rnode, ecliq, changed, defncy)
    elseif nclass == 3
        dq = (cq[2] - q1, cq[3] - q1)
        fillK, k2 = mfupd_mat_kern!(ws, dq, q1, tab, nK, NW, nY, wK, S, qsize, rnode, ecliq, changed, defncy)
    else
        dq = (cq[2] - q1, cq[3] - q1, cq[4] - q1)
        fillK, k2 = mfupd_mat_kern!(ws, dq, q1, tab, nK, NW, nY, wK, S, qsize, rnode, ecliq, changed, defncy)
    end

    #       --------------------------------------------------
    #       CLIQUE MEMBERS: THE NEW MISSING PAIRS, AND THE DEGREE.
    #       --------------------------------------------------
    @inbounds for j in 1:nK
        unode = Int(ecliq[j])
        wx = degree[unode] - qk[j] - wn[j] - qp - qft
        defncy[unode] += fw[j] * wx - gacc[j]
        degree[unode] += fw[j]
    end

    defncy[p] -= fillK

    @inbounds for j in 1:nK
        kpos[ecliq[j]] = 0
    end

    #       ------------------------------------------------------
    #       CHECK IF THE DEFICIENCY OF THE ELIMINATED NODE ENODE
    #       IS ZERO.  STOP IF IT IS NOT.
    #       ------------------------------------------------------
    if abs(defncy[p]) > deftol(degree[p])
        error("ELIMINATED NODE HAS NONZERO DEFICIENCY: $p => $(defncy[p])")
    end

    return convert(V, k2 - nK)
end

#**********************************************************************
#
#     MFUPD_MAT_KERN: THE FILL ROWS F_j = K \ N[u_j] AS LISTS OF THEIR
#     NONZERO WORDS (WITH THEIR HEAVY BITS AND CLASS MASKS), THE
#     PRODUCT OVER THE OUTSIDE ROWS, AND, FOR EACH MEMBER, THE FILL PAIRS
#     INSIDE N(u) ∩ K.  DQ HOLDS q_r - q_1 FOR THE CLASSES AFTER THE
#     FIRST.  RETURNS FILL(K) AND THE LAST POSITION USED IN ECLIQ.
#
#**********************************************************************
function mfupd_mat_kern!(ws::MatWork{W}, dq::NTuple{X, W}, q1::W, tab::Val, nK::Int, NW::Int, nY::Int, wK::W,
        S::AbstractVector{UInt64}, qsize::AbstractVector{W}, rnode::AbstractVector{Int}, ecliq::AbstractVector,
        changed::AbstractVector{Bool}, defncy::AbstractVector{W}) where {W, X}
    fp = ws.fp
    fw = ws.fw
    wn = ws.wn
    gacc = ws.gacc
    qk = ws.qk
    N = ws.N
    H = ws.H
    CM = ws.CM
    KF = ws.KF
    D = ws.D
    T = ws.T
    fk = ws.fk
    fv = ws.fv
    fh = ws.fh
    fx = ws.fx
    empty!(fk)
    empty!(fv)
    empty!(fh)
    empty!(fx)

    if iszero(nK & 63)
        lastw = typemax(UInt64)
    else
        lastw = (one(UInt64) << (nK & 63)) - one(UInt64)
    end

    #       --------------------------------------------------
    #       THE FILL ROWS.  u_j, N(u_j) ∩ K, AND F_j PARTITION K.
    #       --------------------------------------------------
    fillK = zero(W)

    @inbounds for k in 1:NW
        KF[k] = zero(UInt64)
    end

    @inbounds for j in 1:nK
        nb = (j - 1) * NW
        fp[j] = length(fk) + 1
        fwj = zero(W)

        for k in 1:NW
            f = ~N[nb + k]

            if k == mwrd(j)
                f &= ~mbit(j)
            end

            if k == NW
                f &= lastw
            end

            if !iszero(f)
                push!(fk, k)
                push!(fv, f)
                push!(fh, f & H[k])

                for x in 1:X
                    push!(fx, f & CM[(x - 1) * NW + k])
                end

                fwj += wcount(f, k, H, CM, NW, qk, q1, dq, tab, T)
            end
        end

        fw[j] = fwj
        wn[j] = wK - qk[j] - fwj
        gacc[j] = zero(W)

        if ispositive(fwj)
            KF[mwrd(j)] |= mbit(j)
        end

        fillK += qk[j] * fwj
    end

    fp[nK + 1] = length(fk) + 1
    fillK = halve(fillK)

    #       --------------------------------------------------------
    #       THE PRODUCT: EVERY OUTSIDE ROW AGAINST THE FILL ROWS OF
    #       ITS MEMBERS.  A ROW WITH FILL PAIRS IS A CHANGED NODE.
    #       --------------------------------------------------------
    k2 = nK

    @inbounds for r in 1:nY
        ynode = rnode[r]
        qy = qsize[ynode]
        rb = (r - 1) * NW
        my = zero(W)

        for k in 1:NW
            b = S[rb + k] & KF[k]

            while !iszero(b)
                t = trailing_zeros(b)
                b &= b - one(UInt64)
                j = (k - 1) * 64 + t + 1
                c1 = 0
                c2 = 0
                c3 = 0
                c4 = 0
                cw = zero(W)

                for i in fp[j]:(fp[j + 1] - 1)
                    s = S[rb + fk[i]]
                    c1 += count_ones(s & fv[i])

                    if X >= 1
                        c2 += count_ones(s & fx[(i - 1) * X + 1])
                    end

                    if X >= 2
                        c3 += count_ones(s & fx[(i - 1) * X + 2])
                    end

                    if X >= 3
                        c4 += count_ones(s & fx[(i - 1) * X + 3])
                    end

                    h = s & fh[i]

                    if !iszero(h)
                        cw += hpart(tab, h, fk[i], qk, q1, T)
                    end
                end

                cw += q1 * convert(W, c1)

                if X >= 1
                    cw += dq[1] * convert(W, c2)
                end

                if X >= 2
                    cw += dq[2] * convert(W, c3)
                end

                if X >= 3
                    cw += dq[3] * convert(W, c4)
                end

                my += qk[j] * cw
                gacc[j] += qy * cw
            end
        end

        if ispositive(my)
            defncy[ynode] -= halve(my)

            if !changed[ynode]
                k2 += 1
                ecliq[k2] = ynode
                changed[ynode] = true
            end
        end
    end

    #       ------------------------------------------------------
    #       MEMBERS: THE FILL PAIRS INSIDE N(u) ∩ K ARE FILL(K)
    #       MINUS THOSE TOUCHING D = F_u ∪ {u}.
    #       ------------------------------------------------------
    @inbounds for k in 1:NW
        D[k] = zero(UInt64)
    end

    @inbounds for j in 1:nK
        for i in fp[j]:(fp[j + 1] - 1)
            D[fk[i]] = fv[i]
        end

        D[mwrd(j)] |= mbit(j)
        touch = zero(W)
        inside = zero(W)

        for k in 1:NW
            b = D[k]

            while !iszero(b)
                t = trailing_zeros(b)
                b &= b - one(UInt64)
                a = (k - 1) * 64 + t + 1
                touch += qk[a] * fw[a]
                cw = zero(W)

                for i in fp[a]:(fp[a + 1] - 1)
                    cw += fcount(D[fk[i]], i, fk, fv, fh, fx, qk, q1, dq, tab, T)
                end

                inside += qk[a] * cw
            end
        end

        for i in fp[j]:(fp[j + 1] - 1)
            D[fk[i]] = zero(UInt64)
        end

        D[mwrd(j)] = zero(UInt64)
        defncy[ecliq[j]] -= fillK - (touch - halve(inside))
    end

    return fillK, k2
end
