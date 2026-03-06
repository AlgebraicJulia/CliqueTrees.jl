# ===== gemm! =====

function gemm!(tA::Val, tB::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.gemm!(char(tA), char(tB), α, A, B, β, C)
    return
end

function gemm!(tA::Val, tB::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T}
    mul!(C, adj(tA, A), adj(tB, B), α, β)
    return
end

function gemm!(tA::Val, tB::Val, α::T, ::AbstractVector{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, ::AbstractVector{T}, β::T, C::AbstractMatrix{T}, ::Val{:N}) where {T}
    return gemm!(tA, tB, α, A, B, β, C)
end

function gemm!(tA::Val{TA}, tB::Val{TB}, α::T, W::AbstractVector{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, d::AbstractVector{T}, β::T, C::AbstractMatrix{T}, ::Val{:U}) where {T, TA, TB}
    D = reshape(view(W, 1:length(A)), size(A))
    copyrec!(D, A)

    if TA === :N
        cmul!(Val(:R), D, d)
    else
        cmul!(Val(:L), D, d)
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
    mul!(c, adj(tA, A), b, α, β)
    return
end
