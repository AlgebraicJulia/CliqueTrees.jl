# ===== trmx2! =====

function trmx2!(side::Val{SIDE}, uplo::Val{UPLO}, trans::Val{TRANS}, diag::Val{DIAG}, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}) where {SIDE, UPLO, TRANS, DIAG, T}
    # Multiply uses opposite direction from solve
    if isforward(UPLO, TRANS, SIDE)
        trmx2_bwd!(side, uplo, diag, A, B)
    else
        trmx2_fwd!(side, uplo, diag, A, B)
    end
end

function trmx2_fwd!(::Val{:R}, ::Val{UPLO}, ::Val{DIAG}, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}) where {UPLO, DIAG, T}
    @inbounds @fastmath for j in axes(A, 1)
        for k in 1:j - 1
            if UPLO === :L
                Akj = A[j, k]
            else
                Akj = A[k, j]
            end

            for i in axes(B, 1)
                B[i, k] += B[i, j] * Akj
            end
        end

        if DIAG === :N
            Ajj = A[j, j]

            for i in axes(B, 1)
                B[i, j] *= Ajj
            end
        end
    end
end

function trmx2_fwd!(::Val{:L}, ::Val{UPLO}, ::Val{DIAG}, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}) where {UPLO, DIAG, T}
    @inbounds @fastmath for j in axes(A, 1)
        for i in axes(B, 2)
            for k in 1:j - 1
                if UPLO === :U
                    B[k, i] += A[k, j] * B[j, i]
                else
                    B[k, i] += A[j, k] * B[j, i]
                end
            end

            if DIAG === :N
                B[j, i] *= A[j, j]
            end
        end
    end
end

function trmx2_bwd!(::Val{:R}, ::Val{UPLO}, ::Val{DIAG}, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}) where {UPLO, DIAG, T}
    @inbounds @fastmath for j in reverse(axes(A, 1))
        if DIAG === :N
            Ajj = A[j, j]

            for i in axes(B, 1)
                B[i, j] *= Ajj
            end
        end

        for k in 1:j - 1
            if UPLO === :L
                Akj = A[j, k]
            else
                Akj = A[k, j]
            end

            for i in axes(B, 1)
                B[i, j] += B[i, k] * Akj
            end
        end
    end
end

function trmx2_bwd!(::Val{:L}, ::Val{UPLO}, ::Val{DIAG}, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}) where {UPLO, DIAG, T}
    @inbounds @fastmath for j in reverse(axes(A, 1))
        for i in axes(B, 2)
            if DIAG === :N
                B[j, i] *= A[j, j]
            end

            for k in 1:j - 1
                if UPLO === :U
                    B[j, i] += A[k, j] * B[k, i]
                else
                    B[j, i] += A[j, k] * B[k, i]
                end
            end
        end
    end
end

# ===== trmx! =====

function trmx!(side::Val{SIDE}, uplo::Val{UPLO}, trans::Val{TRANS}, diag::Val, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}) where {SIDE, UPLO, TRANS, T}
    n = size(A, 1)

    if n <= THRESHOLD
        trmx2!(side, uplo, trans, diag, A, B)
        return
    end

    m = prevpow(2, n) >> 1

    A₁₁ = view(A, 1:m, 1:m)
    A₂₂ = view(A, m+1:n, m+1:n)

    if UPLO === :L
        A₂₁ = view(A, m+1:n, 1:m)
    else
        A₂₁ = view(A, 1:m, m+1:n)
    end

    if B isa AbstractVector
        B₁ = view(B, 1:m)
        B₂ = view(B, m+1:n)
    elseif SIDE === :R
        B₁ = view(B, :, 1:m)
        B₂ = view(B, :, m+1:n)
    else
        B₁ = view(B, 1:m, :)
        B₂ = view(B, m+1:n, :)
    end

    if isforward(UPLO, TRANS, SIDE)
        trmx!(side, uplo, trans, diag, A₂₂, B₂)

        if B isa AbstractVector
            gemv!(trans, one(T), A₂₁, B₁, one(T), B₂)
        elseif SIDE === :R
            gemm!(Val(:N), trans, one(T), B₁, A₂₁, one(T), B₂)
        else
            gemm!(trans, Val(:N), one(T), A₂₁, B₁, one(T), B₂)
        end

        trmx!(side, uplo, trans, diag, A₁₁, B₁)
    else
        trmx!(side, uplo, trans, diag, A₁₁, B₁)

        if B isa AbstractVector
            gemv!(trans, one(T), A₂₁, B₂, one(T), B₁)
        elseif SIDE === :R
            gemm!(Val(:N), trans, one(T), B₂, A₂₁, one(T), B₁)
        else
            gemm!(trans, Val(:N), one(T), A₂₁, B₂, one(T), B₁)
        end

        trmx!(side, uplo, trans, diag, A₂₂, B₂)
    end
end

# ===== trmm! =====

function trmm!(side::Val, uplo::Val, tA::Val, diag::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.trmm!(char(side), char(uplo), char(tA), char(diag), α, A, B)
    return
end

function trmm!(side::Val, uplo::Val, tA::Val, diag::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    trmx!(side, uplo, tA, diag, A, B)
    lmul!(α, B)
    return
end

# ===== trmv! =====

function trmv!(uplo::Val, tA::Val, diag::Val, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T <: BlasFloat}
    BLAS.trmv!(char(uplo), char(tA), char(diag), A, b)
    return
end

function trmv!(uplo::Val, tA::Val, diag::Val, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    trmx!(Val(:L), uplo, tA, diag, A, b)
    return
end
