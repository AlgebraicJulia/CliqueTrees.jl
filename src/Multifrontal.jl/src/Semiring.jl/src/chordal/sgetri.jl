function sgetri!(
        s::AbstractSemiring,
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        X::AbstractMatrix;
        nt::Integer = nthreads(),
    ) where {T, I}
    #
    #   X ← U*
    #
    strtri!(s, Val(:N), U, X; nt)
    #
    #   X ← X L*
    #
    strsx!(s, Val(:R), Val(:N), Val(:U), L, X; nt)

    return X
end
