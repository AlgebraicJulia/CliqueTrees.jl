# ===== trmm! =====

function trmm!(side::Val, uplo::Val, tA::Val, diag::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.trmm!(char(side), char(uplo), char(tA), char(diag), α, A, B)
    return
end

function trmm!(side::Val{SIDE}, uplo::Val, tA::Val, diag::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T, SIDE}
    if SIDE === :L
        lmul!(adj(tA, tri(uplo, diag, A)), B)
    else
        rmul!(B, adj(tA, tri(uplo, diag, A)))
    end

    lmul!(α, B)
    return
end

# ===== trmv! =====

function trmv!(uplo::Val, tA::Val, diag::Val, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T <: BlasFloat}
    BLAS.trmv!(char(uplo), char(tA), char(diag), A, b)
    return
end

function trmv!(uplo::Val, tA::Val, diag::Val, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    lmul!(adj(tA, tri(uplo, diag, A)), b)
    return
end
