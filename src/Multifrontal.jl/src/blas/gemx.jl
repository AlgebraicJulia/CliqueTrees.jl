# ===== gemm! =====

function gemm!(tA::Val, tB::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.gemm!(char(tA), char(tB), α, A, B, β, C)
    return
end

function gemm!(tA::Val, tB::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T}
    gemx!(tA, tB, α, A, B, β, C)
    return
end

function gemm!(tA::Val, tB::Val, α::T, ::AbstractVector{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, ::AbstractVector{T}, β::T, C::AbstractMatrix{T}, ::Val{:N}) where {T}
    return gemm!(tA, tB, α, A, B, β, C)
end

function gemm!(tA::Val{TA}, tB::Val{TB}, α::T, W::AbstractVector{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, d::AbstractVector{T}, β::T, C::AbstractMatrix{T}, ::Val{:U}) where {T, TA, TB}
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

function gemv!(tA::Val, α::T, A::AbstractMatrix{T}, b::AbstractVector{T}, β::T, c::AbstractVector{T}) where {T <: BlasFloat}
    BLAS.gemv!(char(tA), α, A, b, β, c)
    return
end

function gemv!(tA::Val, α::T, A::AbstractMatrix{T}, b::AbstractVector{T}, β::T, c::AbstractVector{T}) where {T}
    gemx!(tA, Val(:N), α, A, b, β, c)
    return
end

# ===== gemx! =====

function gemx!(tA::Val{TA}, tB::Val{TB}, α::T, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, β::T, C::AbstractVecOrMat{T}) where {T, TA, TB}
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

            gemx!(tA, tB, α, A₁, B₁,      β, C)
            gemx!(tA, tB, α, A₂, B₂, one(T), C)
        end
    end

    return
end

function gemx2!(::Val{:N}, ::Val{:N}, α, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, β, C::AbstractVecOrMat{T}) where {T}
    @inbounds @fastmath for j in axes(C, 2)
        for i in axes(C, 1)
            C[i, j] *= β
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

function gemx2!(::Val{:N}, ::Val{TB}, α, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, β, C::AbstractVecOrMat{T}) where {T, TB}
    @inbounds @fastmath for j in axes(C, 2)
        for i in axes(C, 1)
            C[i, j] *= β
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

function gemx2!(::Val{TA}, ::Val{:N}, α, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, β, C::AbstractVecOrMat{T}) where {T, TA}
    @inbounds @fastmath for j in axes(C, 2)
        for i in axes(C, 1)
            Δ = zero(T)

            for k in axes(A, 1)
                if TA === :C
                    Δ += conj(A[k, i]) * B[k, j]
                else
                    Δ += A[k, i] * B[k, j]
                end
            end

            C[i, j] = α * Δ + β * C[i, j]
        end
    end

    return
end

function gemx2!(::Val{TA}, ::Val{TB}, α, A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, β, C::AbstractVecOrMat{T}) where {T, TA, TB}
    @inbounds @fastmath for j in axes(C, 2)
        for i in axes(C, 1)
            Δ = zero(T)

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

            C[i, j] = α * Δ + β * C[i, j]
        end
    end

    return
end
