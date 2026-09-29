function sgetri!(
        s::AbstractSemiring,
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        X::AbstractMatrix;
        nt::Integer = nthreads(),
    ) where {T, I}
    pool = spool_mt(T, nt)

    return sgetri_mt!(s, L, U, X, pool, nt)
end

function sgetri_mt!(
        s::AbstractSemiring,
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        X::AbstractMatrix,
        pool::AbstractVector,
        nt::Integer,
    ) where {T, I}
    #
    #   X ← U*
    #
    strtri_mt!(s, Val(:N), U, X, pool, nt)
    #
    #   X ← X L*
    #
    strsx_mt!(s, Val(:R), Val(:N), Val(:U), L, X, pool, nt)

    return X
end
