# ===== trsx2! =====

function trsx2!(side::Val{SIDE}, uplo::Val{UPLO}, trans::Val{TRANS}, diag::Val{DIAG}, A::AbstractMatrix, B::AbstractVecOrMat) where {SIDE, UPLO, TRANS, DIAG}
    if isforward(UPLO, TRANS, SIDE)
        trsx2_fwd!(side, uplo, diag, A, B)
    else
        trsx2_bwd!(side, uplo, diag, A, B)
    end
end

function trsx2_fwd!(::Val{:R}, ::Val{UPLO}, ::Val{DIAG}, A::AbstractMatrix, B::AbstractVecOrMat) where {UPLO, DIAG}
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

function trsx2_fwd!(::Val{:L}, ::Val{:L}, ::Val{DIAG}, A::AbstractMatrix, B::AbstractVecOrMat) where {DIAG}
    n = size(A, 1)

    @inbounds @fastmath for i in axes(B, 2)
        for j in 1:n
            if DIAG === :N
                B[j, i] /= A[j, j]
            end

            Bji = B[j, i]

            for k in j + 1:n
                B[k, i] -= A[k, j] * Bji
            end
        end
    end
end

function trsx2_fwd!(::Val{:L}, ::Val{:U}, ::Val{DIAG}, A::AbstractMatrix, B::AbstractVecOrMat) where {DIAG}
    n = size(A, 1)

    @inbounds @fastmath for j in 1:n
        for i in axes(B, 2)
            Δ = zero(promote_eltype(A, B))

            for k in 1:j - 1
                Δ += A[k, j] * B[k, i]
            end

            B[j, i] -= Δ

            if DIAG === :N
                B[j, i] /= A[j, j]
            end
        end
    end
end

function trsx2_bwd!(::Val{:R}, ::Val{UPLO}, ::Val{DIAG}, A::AbstractMatrix, B::AbstractVecOrMat) where {UPLO, DIAG}
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

function trsx2_bwd!(::Val{:R}, ::Val{:L}, ::Val{DIAG}, A::AbstractMatrix, B::AbstractVecOrMat) where {DIAG}
    n = size(A, 1)

    @inbounds @fastmath for j in n:-1:1
        for k in j + 1:n
            Akj = A[k, j]

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

function trsx2_bwd!(::Val{:L}, ::Val{:L}, ::Val{DIAG}, A::AbstractMatrix, B::AbstractVecOrMat) where {DIAG}
    n = size(A, 1)

    @inbounds @fastmath for j in n:-1:1
        if DIAG === :N
            iAjj = inv(A[j, j])
        end

        for i in axes(B, 2)
            Δ = zero(promote_eltype(A, B))

            for k in j + 1:n
                Δ += A[k, j] * B[k, i]
            end

            B[j, i] -= Δ

            if DIAG === :N
                B[j, i] *= iAjj
            end
        end
    end
end

function trsx2_bwd!(::Val{:L}, ::Val{:U}, ::Val{DIAG}, A::AbstractMatrix, B::AbstractVecOrMat) where {DIAG}
    n = size(A, 1)

    @inbounds @fastmath for i in axes(B, 2)
        for j in n:-1:1
            if DIAG === :N
                B[j, i] /= A[j, j]
            end

            Bji = B[j, i]

            for k in 1:j - 1
                B[k, i] -= A[k, j] * Bji
            end
        end
    end
end

# ===== trsx! =====

function trsx!(side::Val, uplo::Val, trans::Val, diag::Val, A::AbstractMatrix, B::AbstractVector; nt::Integer = nthreads())
    trsx2!(side, uplo, trans, diag, A, B)
    return
end

function trsx!(side::Val{S}, uplo::Val, trans::Val, diag::Val, A::AbstractMatrix, B::AbstractMatrix; nt::Integer = nthreads()) where {S}
    if S === :L
        ncol = size(B, 2)
    else
        ncol = size(B, 1)
    end

    if nt <= 1 || ncol <= THRESHOLD
        trsx_impl!(side, uplo, trans, diag, A, B; nt)
    else
        h = ncol >> 1

        if S === :L
            B₁ = view(B, :, 1:h)
            B₂ = view(B, :, h + 1:ncol)
        else
            B₁ = view(B, 1:h, :)
            B₂ = view(B, h + 1:ncol, :)
        end

        nt₁ = nt >> 1

        task = @spawn trsx!(side, uplo, trans, diag, A, B₁; nt = nt₁)
        trsx!(side, uplo, trans, diag, A, B₂; nt = nt - nt₁)
        wait(task)
    end

    return
end

# ===== trsx_impl! =====

function trsx_impl!(side::Val{SIDE}, uplo::Val{UPLO}, trans::Val{TRANS}, diag::Val, A::AbstractMatrix, B::AbstractMatrix; nt::Integer = nthreads()) where {SIDE, UPLO, TRANS}
    n = size(A, 1)

    if n <= THRESHOLD
        trsx2!(side, uplo, trans, diag, A, B)
    else
        m = prevpow(2, n) >> 1

        A₁₁ = view(A, 1:m, 1:m)
        A₂₂ = view(A, m+1:n, m+1:n)

        if UPLO === :L
            A₂₁ = view(A, m+1:n, 1:m)
        else
            A₂₁ = view(A, 1:m, m+1:n)
        end

        if SIDE === :R
            B₁ = view(B, :, 1:m)
            B₂ = view(B, :, m+1:n)
        else
            B₁ = view(B, 1:m, :)
            B₂ = view(B, m+1:n, :)
        end

        if isforward(UPLO, TRANS, SIDE)
            trsx_impl!(side, uplo, trans, diag, A₁₁, B₁; nt)

            if SIDE === :R
                gemm!(Val(:N), trans, -1, B₁, A₂₁, 1, B₂; nt)
            else
                gemm!(trans, Val(:N), -1, A₂₁, B₁, 1, B₂; nt)
            end

            trsx_impl!(side, uplo, trans, diag, A₂₂, B₂; nt)
        else
            trsx_impl!(side, uplo, trans, diag, A₂₂, B₂; nt)

            if SIDE === :R
                gemm!(Val(:N), trans, -1, B₂, A₂₁, 1, B₁; nt)
            else
                gemm!(trans, Val(:N), -1, A₂₁, B₂, 1, B₁; nt)
            end

            trsx_impl!(side, uplo, trans, diag, A₁₁, B₁; nt)
        end
    end

    return
end

# ===== trsm! =====

function trsm!(side::Val, uplo::Val, tA::Val, diag::Val, α, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.trsm!(char(side), char(uplo), char(tA), char(diag), convert(T, α), A, B)
    return
end

function trsm!(side::Val, uplo::Val, tA::Val, diag::Val, α, A::AbstractMatrix, B::AbstractMatrix; nt::Integer = nthreads())
    trsx!(side, uplo, tA, diag, A, B; nt)
    lmul!(α, B)
    return
end

# ===== trsv! =====

function trsv!(uplo::Val, tA::Val, diag::Val, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T <: BlasFloat}
    BLAS.trsv!(char(uplo), char(tA), char(diag), A, b)
    return
end

function trsv!(uplo::Val, tA::Val, diag::Val, A::AbstractMatrix, b::AbstractVector)
    trsx!(Val(:L), uplo, tA, diag, A, b)
    return
end
