# ===== sgetrs! =====

function sgetrs!(s::AbstractSemiring, side::Val{SIDE}, trans::Val{TRANS}, A::AbstractMatrix, b::AbstractVector; nt::Integer = nthreads()) where {SIDE, TRANS}
    if isforward(:L, TRANS, SIDE)
        strsx!(s, side, trans, Val(:L), A, b)
        strsx!(s, side, trans, Val(:U), A, b)
    else
        strsx!(s, side, trans, Val(:U), A, b)
        strsx!(s, side, trans, Val(:L), A, b)
    end

    return b
end

function sgetrs!(s::AbstractSemiring, side::Val{SIDE}, trans::Val{TRANS}, A::AbstractMatrix, B::AbstractMatrix{T}; nt::Integer = nthreads()) where {SIDE, TRANS, T}
    m = size(B, 1)
    n = size(B, 2)

    if SIDE === :L
        c = n
        d = m
    else
        c = m
        d = n
    end

    if nt <= 1 || c <= THRESHOLD
        AP, BP, CP = spool_st(T, m, d, n)

        if isforward(:L, TRANS, SIDE)
            strsx_st!(s, side, trans, Val(:L), A, B, AP, BP, CP)
            strsx_st!(s, side, trans, Val(:U), A, B, AP, BP, CP)
        else
            strsx_st!(s, side, trans, Val(:U), A, B, AP, BP, CP)
            strsx_st!(s, side, trans, Val(:L), A, B, AP, BP, CP)
        end
    else
        pool = spool_mt(T, nt, m, d, n)

        if isforward(:L, TRANS, SIDE)
            strsx_mt!(s, side, trans, Val(:L), A, B, pool, nt)
            strsx_mt!(s, side, trans, Val(:U), A, B, pool, nt)
        else
            strsx_mt!(s, side, trans, Val(:U), A, B, pool, nt)
            strsx_mt!(s, side, trans, Val(:L), A, B, pool, nt)
        end
    end

    return B
end
