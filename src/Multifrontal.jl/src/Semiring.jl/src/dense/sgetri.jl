# ===== sgetri! =====

function sgetri!(s::AbstractSemiring, C::AbstractMatrix{T}, A::AbstractMatrix{T}; nt::Integer = nthreads()) where {T}
    @assert size(A, 1) == size(A, 2)
    @assert size(C) == size(A)

    n = size(A, 1)
    #
    #   C ← U
    #
    szerorec!(s, C, Val(:N))
    copytri!(C, A, Val(:U))

    if nt <= 1
        AP, BP, CP = spool_st(T, n, n, n)
        #
        #   C ← U*        C ← C L* = U* L*
        #
        strtri_st!(s, Val(:U), Val(:N), C, AP, BP, CP)
        strsx_st!(s, Val(:R), Val(:N), Val(:L), Val(:U), A, C, AP, BP, CP)
    else
        pool = spool_mt(T, nt, n, n, n)
        #
        #   C ← U*        C ← C L* = U* L*
        #
        strtri_mt!(s, Val(:U), Val(:N), C, pool, nt)
        strsx_mt!(s, Val(:R), Val(:N), Val(:L), Val(:U), A, C, pool, nt)
    end

    return C
end
