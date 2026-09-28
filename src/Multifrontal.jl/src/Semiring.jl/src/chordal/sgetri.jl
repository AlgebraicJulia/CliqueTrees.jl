function sgetri!(
        s::AbstractSemiring,
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        X::AbstractMatrix;
        nt::Integer = nthreads(),
    ) where {T, I}
    W = DivisionWorkspace{T}(U.S, ncl(U.S))
    pool = spool_mt(T, nt)

    return sgetri_mt!(s, L, U, X, W, pool, nt)
end

function sgetri_mt!(
        s::AbstractSemiring,
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        X::AbstractMatrix,
        W::DivisionWorkspace{T},
        pool::AbstractVector,
        nt::Integer,
    ) where {T, I}
    #
    #   X ← U*
    #
    strtri_mt!(s, Val(:N), U, X, W, pool, nt)
    #
    #   X ← X L*
    #
    strsx_mt!(s, Val(:R), Val(:N), Val(:U), L, X, W, pool, nt, one(I), nv(L.S.res))

    return X
end
