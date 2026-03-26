# ===== gemm! =====

function gemm!(tA::Val, tB::Val, α, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.gemm!(char(tA), char(tB), convert(T, α), A, B, convert(T, β), C)
    return
end

function gemm!(tA::Val, tB::Val, α, A::AbstractMatrix, B::AbstractMatrix, β, C::AbstractMatrix)
    gemx!(tA, tB, α, A, B, β, C)
    return
end

function gemm!(tA::Val, tB::Val, α, ::AbstractVector, A::AbstractMatrix, B::AbstractMatrix, ::AbstractVector, β, C::AbstractMatrix, ::Val{:N})
    return gemm!(tA, tB, α, A, B, β, C)
end

function gemm!(tA::Val{TA}, tB::Val{TB}, α, W::AbstractVector, A::AbstractMatrix, B::AbstractMatrix, d::AbstractVector, β, C::AbstractMatrix, ::Val{:U}) where {TA, TB}
    D = reshape(view(W, 1:length(A)), size(A))
    copyrec!(D, A)

    if TA === :N
        cmul!(Val(:R), Val(:U), D, d)
    else
        cmul!(Val(:L), Val(:U), D, d)
    end

    gemm!(tA, tB, α, D, B, β, C)
    return
end

# ===== gemv! =====

function gemv!(tA::Val, α, A::AbstractMatrix{T}, b::AbstractVector{T}, β, c::AbstractVector{T}) where {T <: BlasFloat}
    BLAS.gemv!(char(tA), convert(T, α), A, b, convert(T, β), c)
    return
end

function gemv!(tA::Val, α, A::AbstractMatrix, b::AbstractVector, β, c::AbstractVector)
    gemx!(tA, Val(:N), α, A, b, β, c)
    return
end

# ===== gemx! =====

function gemx!(tA::Val{TA}, tB::Val{TB}, α, A::AbstractMatrix, B::AbstractVecOrMat, β, C::AbstractVecOrMat) where {TA, TB}
    m = size(C, 1)
    n = size(C, 2)

    if TA === :N
        k = size(A, 2)
    else
        k = size(A, 1)
    end

    maxdim = max(m, n, k)

    if maxdim <= THRESHOLD
        gemx2!(tA, tB, α, A, B, β, C)
    else
        l = prevpow(2, maxdim) >> 1

        if m == maxdim
            if C isa AbstractVector
                C₁ = view(C,     1:l)
                C₂ = view(C, l + 1:m)
            else
                C₁ = view(C,     1:l, :)
                C₂ = view(C, l + 1:m,  :)
            end

            if TA === :N
                A₁ = view(A,     1:l, :)
                A₂ = view(A, l + 1:m,  :)
            else
                A₁ = view(A, :,     1:l)
                A₂ = view(A, :, l + 1:m)
            end

            gemx!(tA, tB, α, A₁, B, β, C₁)
            gemx!(tA, tB, α, A₂, B, β, C₂)

        elseif n == maxdim
            C₁ = view(C, :,     1:l)
            C₂ = view(C, :, l + 1:n)

            if B isa AbstractVector
                B₁ = view(B,     1:l)
                B₂ = view(B, l + 1:n)
            elseif TB === :N
                B₁ = view(B, :,     1:l)
                B₂ = view(B, :, l + 1:n)
            else
                B₁ = view(B,     1:l, :)
                B₂ = view(B, l + 1:n,  :)
            end

            gemx!(tA, tB, α, A, B₁, β, C₁)
            gemx!(tA, tB, α, A, B₂, β, C₂)

        else
            if TA === :N
                A₁ = view(A, :,     1:l)
                A₂ = view(A, :, l + 1:k)
            else
                A₁ = view(A,     1:l, :)
                A₂ = view(A, l + 1:k,  :)
            end

            if B isa AbstractVector
                B₁ = view(B,     1:l)
                B₂ = view(B, l + 1:k)
            elseif TB === :N
                B₁ = view(B,     1:l, :)
                B₂ = view(B, l + 1:k,  :)
            else
                B₁ = view(B, :,     1:l)
                B₂ = view(B, :, l + 1:k)
            end

            gemx!(tA, tB, α, A₁, B₁, β, C)
            gemx!(tA, tB, α, A₂, B₂, 1, C)
        end
    end

    return
end

function gemx2!(::Val{:N}, ::Val{:N}, α, A::AbstractMatrix, B::AbstractVecOrMat, β, C::AbstractVecOrMat)
    if iszero(β)
        @inbounds @fastmath for j in axes(C, 2)
            for i in axes(C, 1)
                C[i, j] = β
            end
        end
    else
        @inbounds @fastmath for j in axes(C, 2)
            for i in axes(C, 1)
                C[i, j] *= β
            end
        end
    end

    @inbounds @fastmath for k in axes(A, 2)
        for j in axes(C, 2)
            Bkj = α * B[k, j]

            for i in axes(C, 1)
                C[i, j] += A[i, k] * Bkj
            end
        end
    end

    return
end

function gemx2!(::Val{:N}, ::Val{TB}, α, A::AbstractMatrix, B::AbstractVecOrMat, β, C::AbstractVecOrMat) where {TB}
    if iszero(β)
        @inbounds @fastmath for j in axes(C, 2)
            for i in axes(C, 1)
                C[i, j] = β
            end
        end
    else
        @inbounds @fastmath for j in axes(C, 2)
            for i in axes(C, 1)
                C[i, j] *= β
            end
        end
    end

    @inbounds @fastmath for k in axes(A, 2)
        for j in axes(C, 2)
            if TB === :C
                Bjk = α * conj(B[j, k])
            else
                Bjk = α * B[j, k]
            end

            for i in axes(C, 1)
                C[i, j] += A[i, k] * Bjk
            end
        end
    end

    return
end

function gemx2!(::Val{TA}, ::Val{:N}, α, A::AbstractMatrix, B::AbstractVecOrMat, β, C::AbstractVecOrMat) where {TA}
    @inbounds @fastmath for j in axes(C, 2)
        for i in axes(C, 1)
            Δ = zero(promote_eltype(A, B))

            for k in axes(A, 1)
                if TA === :C
                    Δ += conj(A[k, i]) * B[k, j]
                else
                    Δ += A[k, i] * B[k, j]
                end
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

function gemx2!(::Val{TA}, ::Val{TB}, α, A::AbstractMatrix, B::AbstractVecOrMat, β, C::AbstractVecOrMat) where {TA, TB}
    @inbounds @fastmath for j in axes(C, 2)
        for i in axes(C, 1)
            Δ = zero(promote_eltype(A, B))

            for k in axes(A, 1)
                if TA === :C
                    Aki = conj(A[k, i])
                else
                    Aki = A[k, i]
                end

                if TB === :C
                    Δ += Aki * conj(B[j, k])
                else
                    Δ += Aki * B[j, k]
                end
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
