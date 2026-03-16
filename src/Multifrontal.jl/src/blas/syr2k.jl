# ===== syr2k! =====
# Symmetric rank-2k update: C = α(AB' + BA') + βC, writing only to the triangle.

function syr2k!(uplo::Val, trans::Val{TRANS}, α::Real, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::Real, C::AbstractMatrix{T}) where {T <: BlasFloat, TRANS}
    if T <: Complex
        BLAS.her2k!(char(uplo), char(trans), convert(T, α), A, B, β, C)
    else
        BLAS.syr2k!(char(uplo), char(trans), convert(T, α), A, B, β, C)
    end

    return
end

function syr2k!(uplo::Val{UPLO}, trans::Val{TRANS}, α, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T, TRANS, UPLO}
    gemmt!(uplo, trans, Val(:C), α, A, B, β, C)
    gemmt!(uplo, trans, Val(:C), α, B, A, one(T), C)
    return
end
