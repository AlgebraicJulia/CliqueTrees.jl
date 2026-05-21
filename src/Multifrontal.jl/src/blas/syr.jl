# ===== syr! =====

function syr!(uplo::Val, α, x::AbstractVector{T}, A::AbstractMatrix{T}) where {T <: BlasFloat}
    if T <: Complex
        BLAS.her!(char(uplo), convert(real(T), α), x, A)
    else
        BLAS.syr!(char(uplo), convert(T, α), x, A)
    end

    return
end

function syr!(::Val{UPLO}, α, x::AbstractVector{T}, A::AbstractMatrix{T}) where {T, UPLO}
    n = length(x)

    @inbounds @fastmath for j in 1:n
        αxj = α * conj(x[j])

        if UPLO === :L
            for i in j:n
                A[i, j] += x[i] * αxj
            end
        else
            for i in 1:j
                A[i, j] += x[i] * αxj
            end
        end
    end

    return
end
