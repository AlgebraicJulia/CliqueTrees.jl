# ===== gert! =====
# Triangular rank-1 update: C += α * x * yᴴ, writing only to the triangle specified by uplo

# WARNING: OVER-WRITES OTHER TRIANGLE (matches gemmt! behavior for BlasFloat)
function gert!(uplo::Val, α, x::AbstractVector{T}, y::AbstractVector{T}, C::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.ger!(convert(T, α), x, y, C)
    return
end

function gert!(uplo::Val{UPLO}, α, x::AbstractVector{T}, y::AbstractVector{T}, C::AbstractMatrix{T}) where {T, UPLO}
    n = size(C, 1)

    @inbounds @fastmath for j in 1:n
        if UPLO === :L
            rng = j:n
        else
            rng = 1:j
        end

        αyj = α * conj(y[j])

        for i in rng
            C[i, j] += x[i] * αyj
        end
    end

    return
end
