# ===== symv! =====

function symv!(uplo::Val, α::T, A::AbstractMatrix{T}, x::AbstractVector{T}, β::T, y::AbstractVector{T}) where {T <: BlasFloat}
    if T <: Complex
        BLAS.hemv!(char(uplo), α, A, x, β, y)
    else
        BLAS.symv!(char(uplo), α, A, x, β, y)
    end

    return
end

function symv!(uplo::Val, α::T, A::AbstractMatrix{T}, x::AbstractVector{T}, β::T, y::AbstractVector{T}) where {T}
    mul!(y, sym(uplo, A), x, α, β)
    return
end

# ===== symm! =====

function symm!(side::Val, uplo::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T <: BlasFloat}
    if T <: Complex
        BLAS.hemm!(char(side), char(uplo), α, A, B, β, C)
    else
        BLAS.symm!(char(side), char(uplo), α, A, B, β, C)
    end

    return
end

function symm!(side::Val{SIDE}, uplo::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T, SIDE}
    if SIDE === :L
        mul!(C, sym(uplo, A), B, α, β)
    else
        mul!(C, B, sym(uplo, A), α, β)
    end

    return
end
