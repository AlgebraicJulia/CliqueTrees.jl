function sgetri!(
        s::AbstractSemiring,
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        X::AbstractMatrix;
        nt::Integer = nthreads(),
    ) where {T, I}
    S = L.S

    fdesc = FVector{I}(undef, nfr(S))
    Tval = FVector{T}(undef, S.nFval * S.nFval)
    W = DivisionWorkspace{T}(S, ncl(S))
    pool = spool_mt(T, nt)
    #
    #   X ← U*
    #
    strtri_mt!(s, U, X, fdesc, Tval, W.Mval, pool, nt)
    #
    #   X ← X L*
    #
    strsx_mt!(s, Val(:R), Val(:N), Val(:U), L, X, W, pool, nt)

    return X
end
