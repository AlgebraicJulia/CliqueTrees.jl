# ===== trrk! =====
# Triangular rank-k update: C = α * op(A) * op(B)' + β * C, writing only to the triangle.
# Assumes A * B' is symmetric, so uses syr2k with α/2 for BLAS types.

function trrk!(uplo::Val, trans::Val, α, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T <: BlasFloat}
    syr2k!(uplo, trans, α / 2, A, B, β, C)
    return
end

function trrk!(uplo::Val, trans::Val{TRANS}, α, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T, TRANS}
    if TRANS === :N
        gemmt!(uplo, trans, Val(:C), α, A, B, β, C)
    else
        gemmt!(uplo, trans, Val(:N), α, A, B, β, C)
    end
    return
end
