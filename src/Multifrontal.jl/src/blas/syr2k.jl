# ===== syr2k! =====
# Symmetric rank-2k update: C = α(AB' + BA') + βC, writing only to the triangle.

function syr2k!(uplo::Val, trans::Val{TRANS}, α, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T <: BlasFloat, TRANS}
    if T <: Complex
        BLAS.her2k!(char(uplo), char(trans), convert(T, α), A, B, convert(real(T), β), C)
    else
        BLAS.syr2k!(char(uplo), char(trans), convert(T, α), A, B, convert(T, β), C)
    end

    return
end

function syr2k!(uplo::Val{UPLO}, trans::Val{TRANS}, α, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T, TRANS, UPLO}
    if TRANS === :N
        tB = Val(:C)
    else
        tB = Val(:N)
    end

    gemmt!(uplo, trans, tB, α, A, B, β, C)
    gemmt!(uplo, trans, tB, α, B, A, one(T), C)
    return
end
