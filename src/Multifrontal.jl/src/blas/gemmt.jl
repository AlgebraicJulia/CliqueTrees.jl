# ===== gemmt! =====
# Triangular gemm: C = α * op(A) * op(B) + β * C, writing only to the triangle specified by uplo

# WARNING: OVER-WRITES OTHER TRIANGLE
function gemmt!(uplo::Val, tA::Val, tB::Val, α, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T <: BlasFloat}
    gemm!(tA, tB, α, A, B, β, C)
    return
end

function gemmt!(uplo::Val{UPLO}, tA::Val{TA}, tB::Val{TB}, α, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T, TA, UPLO, TB}
    n = size(C, 1)

    if n <= THRESHOLD
        gemmt2!(uplo, tA, tB, α, A, B, β, C)
        return
    end

    m = prevpow(2, n) >> 1

    C₁₁ = view(C,     1:m,     1:m)
    C₂₂ = view(C, m + 1:n, m + 1:n)

    if TA === :N
        A₁ = view(A,     1:m, :)
        A₂ = view(A, m + 1:n, :)
    else
        A₁ = view(A, :,     1:m)
        A₂ = view(A, :, m + 1:n)
    end

    if TB === :N
        B₁ = view(B, :,     1:m)
        B₂ = view(B, :, m + 1:n)
    else
        B₁ = view(B,     1:m, :)
        B₂ = view(B, m + 1:n, :)
    end

    if UPLO === :L
        C₂₁ = view(C, m + 1:n, 1:m)
        gemx!(tA, tB, α, A₂, B₁, β, C₂₁)
    else
        C₁₂ = view(C, 1:m, m + 1:n)
        gemx!(tA, tB, α, A₁, B₂, β, C₁₂)
    end

    gemmt!(uplo, tA, tB, α, A₁, B₁, β, C₁₁)
    gemmt!(uplo, tA, tB, α, A₂, B₂, β, C₂₂)
    return
end

# ===== gemmt2! =====

function gemmt2!(::Val{UPLO}, ::Val{:N}, ::Val{:N}, α, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T, UPLO}
    m, n = size(C)

    @inbounds @fastmath for j in axes(C, 2)
        if UPLO === :L
            rng = j:m
        else
            rng = 1:j
        end

        for i in rng
            Δ = zero(T)

            for k in axes(A, 2)
                Δ += A[i, k] * B[k, j]
            end

            if iszero(β)
                C[i, j] = α * Δ
            else
                C[i, j] = α * Δ + β * C[i, j]
            end
        end
    end

    return
end

function gemmt2!(::Val{UPLO}, ::Val{:N}, ::Val{TB}, α, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T, UPLO, TB}
    m, n = size(C)

    @inbounds @fastmath for j in axes(C, 2)
        if UPLO === :L
            rng = j:m
        else
            rng = 1:j
        end

        if iszero(β)
            for i in rng
                C[i, j] = β
            end
        else
            for i in rng
                C[i, j] *= β
            end
        end
    end

    @inbounds @fastmath for k in axes(A, 2)
        for j in axes(C, 2)
            if UPLO === :L
                rng = j:m
            else
                rng = 1:j
            end

            if TB === :C
                Bjk = conj(B[j, k])
            else
                Bjk = B[j, k]
            end

            for i in rng
                C[i, j] += α * A[i, k] * Bjk
            end
        end
    end

    return
end

function gemmt2!(::Val{UPLO}, ::Val{TA}, ::Val{:N}, α, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T, UPLO, TA}
    m, n = size(C)

    @inbounds @fastmath for j in axes(C, 2)
        if UPLO === :L
            rng = j:m
        else
            rng = 1:j
        end

        for i in rng
            Δ = zero(T)

            for k in axes(A, 1)
                if TA === :C
                    Aki = conj(A[k, i])
                else
                    Aki = A[k, i]
                end

                Δ += Aki * B[k, j]
            end

            if iszero(β)
                C[i, j] = α * Δ
            else
                C[i, j] = α * Δ + β * C[i, j]
            end
        end
    end

    return
end

function gemmt2!(::Val{UPLO}, ::Val{TA}, ::Val{TB}, α, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T, UPLO, TA, TB}
    m, n = size(C)

    @inbounds @fastmath for j in axes(C, 2)
        if UPLO === :L
            rng = j:m
        else
            rng = 1:j
        end

        for i in rng
            Δ = zero(T)

            for k in axes(A, 1)
                if TA === :C
                    Aki = conj(A[k, i])
                else
                    Aki = A[k, i]
                end

                if TB === :C
                    Bjk = conj(B[j, k])
                else
                    Bjk = B[j, k]
                end

                Δ += Aki * Bjk
            end

            if iszero(β)
                C[i, j] = α * Δ
            else
                C[i, j] = α * Δ + β * C[i, j]
            end
        end
    end

    return
end
