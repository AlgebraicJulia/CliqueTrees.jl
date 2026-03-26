# ===== symv! =====

function symv!(uplo::Val, α, A::AbstractMatrix{T}, x::AbstractVector{T}, β, y::AbstractVector{T}) where {T <: BlasFloat}
    if T <: Complex
        BLAS.hemv!(char(uplo), convert(T, α), A, x, convert(T, β), y)
    else
        BLAS.symv!(char(uplo), convert(T, α), A, x, convert(T, β), y)
    end

    return
end

function symv!(uplo::Val{UPLO}, α, A::AbstractMatrix, x::AbstractVector, β, y::AbstractVector) where {UPLO}
    n = length(y)

    if n <= THRESHOLD
        symv2!(uplo, α, A, x, β, y)
    else
        m = prevpow(2, n) >> 1

        A₁₁ = view(A,     1:m,     1:m)
        A₂₂ = view(A, m + 1:n, m + 1:n)

        if UPLO === :L
            A₂₁ = view(A, m + 1:n, 1:m)
        else
            A₂₁ = view(A, 1:m, m + 1:n)
        end

        x₁ = view(x,     1:m)
        x₂ = view(x, m + 1:n)
        y₁ = view(y,     1:m)
        y₂ = view(y, m + 1:n)

        symv!(uplo, α, A₁₁, x₁, β, y₁)
        symv!(uplo, α, A₂₂, x₂, β, y₂)

        if UPLO === :L
            gemv!(Val(:N), α, A₂₁, x₁, 1, y₂)
            gemv!(Val(:C), α, A₂₁, x₂, 1, y₁)
        else
            gemv!(Val(:N), α, A₂₁, x₂, 1, y₁)
            gemv!(Val(:C), α, A₂₁, x₁, 1, y₂)
        end
    end

    return
end

# ===== symv2! =====

function symv2!(::Val{:L}, α, A::AbstractMatrix, x::AbstractVector, β, y::AbstractVector)
    n = length(y)

    if iszero(β)
        @inbounds @fastmath for i in 1:n
            y[i] = β
        end
    else
        @inbounds @fastmath for i in 1:n
            y[i] *= β
        end
    end

    @inbounds @fastmath for j in 1:n
        αxj = α * x[j]
        y[j] += A[j, j] * αxj

        for i in j + 1:n
            y[i] += A[i, j] * αxj
            y[j] += conj(A[i, j]) * α * x[i]
        end
    end
    return
end

function symv2!(::Val{:U}, α, A::AbstractMatrix, x::AbstractVector, β, y::AbstractVector)
    n = length(y)

    if iszero(β)
        @inbounds @fastmath for i in 1:n
            y[i] = β
        end
    else
        @inbounds @fastmath for i in 1:n
            y[i] *= β
        end
    end

    @inbounds @fastmath for j in 1:n
        αxj = α * x[j]
        y[j] += A[j, j] * αxj

        for i in 1:j - 1
            y[i] += A[i, j] * αxj
            y[j] += conj(A[i, j]) * α * x[i]
        end
    end
    return
end

# ===== symm! =====

function symm!(side::Val, uplo::Val, α, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T <: BlasFloat}
    if T <: Complex
        BLAS.hemm!(char(side), char(uplo), convert(T, α), A, B, convert(T, β), C)
    else
        BLAS.symm!(char(side), char(uplo), convert(T, α), A, B, convert(T, β), C)
    end

    return
end

function symm!(side::Val{SIDE}, uplo::Val{UPLO}, α, A::AbstractMatrix, B::AbstractMatrix, β, C::AbstractMatrix) where {SIDE, UPLO}
    m, n = size(C)
    p = size(A, 1)

    maxdim = max(m, n, p)

    if maxdim <= THRESHOLD
        symm2!(side, uplo, α, A, B, β, C)
        return
    end

    l = prevpow(2, maxdim) >> 1

    if p == maxdim
        A₁₁ = view(A,     1:l,     1:l)
        A₂₂ = view(A, l + 1:p, l + 1:p)

        if UPLO === :L
            A₂₁ = view(A, l + 1:p, 1:l)
        else
            A₂₁ = view(A, 1:l, l + 1:p)
        end

        if SIDE === :L
            B₁ = view(B,     1:l, :)
            B₂ = view(B, l + 1:p, :)
            C₁ = view(C,     1:l, :)
            C₂ = view(C, l + 1:p, :)
        else
            B₁ = view(B, :,     1:l)
            B₂ = view(B, :, l + 1:p)
            C₁ = view(C, :,     1:l)
            C₂ = view(C, :, l + 1:p)
        end

        symm!(side, uplo, α, A₁₁, B₁, β, C₁)
        symm!(side, uplo, α, A₂₂, B₂, β, C₂)

        if SIDE === :L
            if UPLO === :L
                gemm!(Val(:N), Val(:N), α, A₂₁, B₁, 1, C₂)
                gemm!(Val(:C), Val(:N), α, A₂₁, B₂, 1, C₁)
            else
                gemm!(Val(:N), Val(:N), α, A₂₁, B₂, 1, C₁)
                gemm!(Val(:C), Val(:N), α, A₂₁, B₁, 1, C₂)
            end
        else
            if UPLO === :L
                gemm!(Val(:N), Val(:N), α, B₂, A₂₁, 1, C₁)
                gemm!(Val(:N), Val(:C), α, B₁, A₂₁, 1, C₂)
            else
                gemm!(Val(:N), Val(:C), α, B₂, A₂₁, 1, C₁)
                gemm!(Val(:N), Val(:N), α, B₁, A₂₁, 1, C₂)
            end
        end
    else
        if SIDE === :L
            B₁ = view(B, :,     1:l)
            B₂ = view(B, :, l + 1:n)
            C₁ = view(C, :,     1:l)
            C₂ = view(C, :, l + 1:n)
        else
            B₁ = view(B,     1:l, :)
            B₂ = view(B, l + 1:m, :)
            C₁ = view(C,     1:l, :)
            C₂ = view(C, l + 1:m, :)
        end

        symm!(side, uplo, α, A, B₁, β, C₁)
        symm!(side, uplo, α, A, B₂, β, C₂)
    end

    return
end

# ===== symm2! =====

function symm2!(::Val{:L}, ::Val{:L}, α, A::AbstractMatrix, B::AbstractMatrix, β, C::AbstractMatrix)
    m, n = size(C)

    @inbounds @fastmath for j in 1:n
        for i in 1:m
            Δ = α * A[i, i] * B[i, j]

            for k in i + 1:m
                Δ += α * conj(A[k, i]) * B[k, j]
            end

            for k in 1:i - 1
                Δ += α * A[i, k] * B[k, j]
            end

            if iszero(β)
                C[i, j] = Δ
            else
                C[i, j] = Δ + β * C[i, j]
            end
        end
    end

    return
end

function symm2!(::Val{:L}, ::Val{:U}, α, A::AbstractMatrix, B::AbstractMatrix, β, C::AbstractMatrix)
    m, n = size(C)

    @inbounds @fastmath for j in 1:n
        for i in 1:m
            Δ = α * A[i, i] * B[i, j]

            for k in 1:i - 1
                Δ += α * conj(A[k, i]) * B[k, j]
            end

            for k in i + 1:m
                Δ += α * A[i, k] * B[k, j]
            end

            if iszero(β)
                C[i, j] = Δ
            else
                C[i, j] = Δ + β * C[i, j]
            end
        end
    end
    return
end

function symm2!(::Val{:R}, ::Val{:L}, α, A::AbstractMatrix, B::AbstractMatrix, β, C::AbstractMatrix)
    m, n = size(C)

    @inbounds @fastmath for j in 1:n
        for i in 1:m
            Δ = α * B[i, j] * A[j, j]

            for k in j + 1:n
                Δ += α * B[i, k] * conj(A[k, j])
            end

            for k in 1:j - 1
                Δ += α * B[i, k] * A[j, k]
            end

            if iszero(β)
                C[i, j] = Δ
            else
                C[i, j] = Δ + β * C[i, j]
            end
        end
    end

    return
end

function symm2!(::Val{:R}, ::Val{:U}, α, A::AbstractMatrix, B::AbstractMatrix, β, C::AbstractMatrix)
    m, n = size(C)

    @inbounds @fastmath for j in 1:n
        for i in 1:m
            Δ = α * B[i, j] * A[j, j]

            for k in 1:j - 1
                Δ += α * B[i, k] * conj(A[k, j])
            end

            for k in j + 1:n
                Δ += α * B[i, k] * A[j, k]
            end

            if iszero(β)
                C[i, j] = Δ
            else
                C[i, j] = Δ + β * C[i, j]
            end
        end
    end

    return
end
