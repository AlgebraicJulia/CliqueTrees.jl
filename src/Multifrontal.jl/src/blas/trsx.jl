# ===== trsx2! =====

function trsx2!(side::Val{SIDE}, uplo::Val{UPLO}, trans::Val{TRANS}, diag::Val{DIAG}, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}) where {SIDE, UPLO, TRANS, DIAG, T}
    if isforward(UPLO, TRANS, SIDE)
        trsx2_fwd!(side, uplo, diag, A, B)
    else
        trsx2_bwd!(side, uplo, diag, A, B)
    end
end

function trsx2_fwd!(::Val{:R}, ::Val{UPLO}, ::Val{DIAG}, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}) where {UPLO, DIAG, T}
    @inbounds @fastmath for j in axes(A, 1)
        for k in 1:j - 1
            if UPLO === :L
                Akj = A[j, k]
            else
                Akj = A[k, j]
            end

            for i in axes(B, 1)
                B[i, j] -= B[i, k] * Akj
            end
        end

        if DIAG === :N
            iAjj = inv(A[j, j])

            for i in axes(B, 1)
                B[i, j] *= iAjj
            end
        end
    end
end

function trsx2_fwd!(::Val{:L}, ::Val{UPLO}, ::Val{DIAG}, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {UPLO, DIAG, T}
    @inbounds @fastmath for j in axes(A, 1)
        for i in axes(B, 2)
            for k in 1:j - 1
                if UPLO === :U
                    B[j, i] -= A[k, j] * B[k, i]
                else
                    B[j, i] -= A[j, k] * B[k, i]
                end
            end

            if DIAG === :N
                B[j, i] *= inv(A[j, j])
            end
        end
    end
end

function trsx2_fwd!(::Val{:L}, ::Val{UPLO}, ::Val{DIAG}, A::AbstractMatrix{T}, b::AbstractVector{T}) where {UPLO, DIAG, T}
    @inbounds @fastmath for j in axes(A, 1)
        for k in 1:j - 1
            if UPLO === :U
                b[j] -= A[k, j] * b[k]
            else
                b[j] -= A[j, k] * b[k]
            end
        end

        if DIAG === :N
            b[j] *= inv(A[j, j])
        end
    end
end

function trsx2_bwd!(::Val{:R}, ::Val{UPLO}, ::Val{DIAG}, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}) where {UPLO, DIAG, T}
    @inbounds @fastmath for j in reverse(axes(A, 1))
        if DIAG === :N
            iAjj = inv(A[j, j])

            for i in axes(B, 1)
                B[i, j] *= iAjj
            end
        end

        for k in 1:j - 1
            if UPLO === :L
                Akj = A[j, k]
            else
                Akj = A[k, j]
            end

            for i in axes(B, 1)
                B[i, k] -= B[i, j] * Akj
            end
        end
    end
end

function trsx2_bwd!(::Val{:L}, ::Val{UPLO}, ::Val{DIAG}, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {UPLO, DIAG, T}
    @inbounds @fastmath for j in reverse(axes(A, 1))
        for i in axes(B, 2)
            if DIAG === :N
                B[j, i] *= inv(A[j, j])
            end

            for k in 1:j - 1
                if UPLO === :U
                    B[k, i] -= A[k, j] * B[j, i]
                else
                    B[k, i] -= A[j, k] * B[j, i]
                end
            end
        end
    end
end

function trsx2_bwd!(::Val{:L}, ::Val{UPLO}, ::Val{DIAG}, A::AbstractMatrix{T}, b::AbstractVector{T}) where {UPLO, DIAG, T}
    @inbounds @fastmath for j in reverse(axes(A, 1))
        if DIAG === :N
            b[j] *= inv(A[j, j])
        end

        for k in 1:j - 1
            if UPLO === :U
                b[k] -= A[k, j] * b[j]
            else
                b[k] -= A[j, k] * b[j]
            end
        end
    end
end

# ===== trsx! =====

function trsx!(side::Val{SIDE}, uplo::Val{UPLO}, trans::Val{TRANS}, diag::Val, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}) where {SIDE, UPLO, TRANS, T}
    n = size(A, 1)

    if n <= THRESHOLD
        trsx2!(side, uplo, trans, diag, A, B)
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
        trsx!(side, uplo, trans, diag, A₁₁, B₁)

        if B isa AbstractVector
            gemv!(trans, -one(T), A₂₁, B₁, one(T), B₂)
        elseif SIDE === :R
            gemm!(Val(:N), trans, -one(T), B₁, A₂₁, one(T), B₂)
        else
            gemm!(trans, Val(:N), -one(T), A₂₁, B₁, one(T), B₂)
        end

        trsx!(side, uplo, trans, diag, A₂₂, B₂)
    else
        trsx!(side, uplo, trans, diag, A₂₂, B₂)

        if B isa AbstractVector
            gemv!(trans, -one(T), A₂₁, B₂, one(T), B₁)
        elseif SIDE === :R
            gemm!(Val(:N), trans, -one(T), B₂, A₂₁, one(T), B₁)
        else
            gemm!(trans, Val(:N), -one(T), A₂₁, B₂, one(T), B₁)
        end

        trsx!(side, uplo, trans, diag, A₁₁, B₁)
    end
end

# ===== trsm! =====

function trsm!(side::Val, uplo::Val, tA::Val, diag::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.trsm!(char(side), char(uplo), char(tA), char(diag), α, A, B)
    return
end

function trsm!(side::Val, uplo::Val, tA::Val, diag::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    trsx!(side, uplo, tA, diag, A, B)
    lmul!(α, B)
    return
end

# ===== trsv! =====

function trsv!(uplo::Val, tA::Val, diag::Val, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T <: BlasFloat}
    BLAS.trsv!(char(uplo), char(tA), char(diag), A, b)
    return
end

function trsv!(uplo::Val, tA::Val, diag::Val, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    trsx!(Val(:L), uplo, tA, diag, A, b)
    return
end
