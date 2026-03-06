# ===== ger! =====

function ger!(α, x::AbstractVector{T}, y::AbstractVector{T}, A::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.ger!(convert(T, α), x, y, A)
    return
end

function ger!(α, x::AbstractVector{T}, y::AbstractVector{T}, A::AbstractMatrix{T}) where {T}
    mul!(A, x, y', α, true)
    return
end
