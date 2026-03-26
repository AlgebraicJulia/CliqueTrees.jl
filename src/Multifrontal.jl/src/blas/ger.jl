# ===== ger2! =====

function ger2!(α, x::AbstractVector{T}, y::AbstractVector{T}, A::AbstractMatrix{T}) where {T}
    @inbounds @fastmath for j in axes(A, 2)
        αyj = α * conj(y[j])

        for i in axes(A, 1)
            A[i, j] += x[i] * αyj
        end
    end

    return
end

# ===== ger! =====

function ger!(α, x::AbstractVector{T}, y::AbstractVector{T}, A::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.ger!(convert(T, α), x, y, A)
    return
end

function ger!(α, x::AbstractVector{T}, y::AbstractVector{T}, A::AbstractMatrix{T}) where {T}
    m = size(A, 1)
    n = size(A, 2)
    maxdim = max(m, n)

    if maxdim <= THRESHOLD
        ger2!(α, x, y, A)
    else
        l = prevpow(2, maxdim) >> 1

        if m >= n
            x₁ = view(x,     1:l)
            x₂ = view(x, l + 1:m)
            A₁ = view(A,     1:l, :)
            A₂ = view(A, l + 1:m, :)

            ger!(α, x₁, y, A₁)
            ger!(α, x₂, y, A₂)
        else
            y₁ = view(y,     1:l)
            y₂ = view(y, l + 1:n)
            A₁ = view(A, :,     1:l)
            A₂ = view(A, :, l + 1:n)

            ger!(α, x, y₁, A₁)
            ger!(α, x, y₂, A₂)
        end
    end

    return
end
