# ===== trrk! =====

function trrk!(uplo::Val, trans::Val{TRANS}, alpha::Real, A::AbstractMatrix{T}, B::AbstractMatrix{T}, beta::Real, C::AbstractMatrix{T}) where {T <: BlasFloat, TRANS}
    if T <: Complex
        BLAS.her2k!(char(uplo), char(trans), convert(T, alpha / 2), A, B, beta, C)
    else
        BLAS.syr2k!(char(uplo), char(trans), convert(T, alpha / 2), A, B, beta, C)
    end

    return
end

function trrk!(uplo::Val{UPLO}, trans::Val{TRANS}, alpha::Real, A::AbstractMatrix{T}, B::AbstractMatrix{T}, beta::Real, C::AbstractMatrix{T}) where {T, TRANS, UPLO}
    @inbounds for j in axes(C, 2)
        if UPLO === :L
            for i in 1:j-1
                C[i, j] = zero(T)
            end
        else
            for i in j + 1:size(C, 1)
                C[i, j] = zero(T)
            end
        end
    end

    if TRANS === :N
        mul!(C, A, B', alpha, beta)
    else
        mul!(C, A', B, alpha, beta)
    end

    return
end
