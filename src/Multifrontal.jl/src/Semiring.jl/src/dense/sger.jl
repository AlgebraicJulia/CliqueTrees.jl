# ===== sger_kern! =====

function sger_kern!(s::AbstractSemiring, tA::Val, tB::Val, side::Val, pc::Ptr{T}, ldc::Integer, px::Ptr{T}, y::AbstractVector, nr::Integer, nc::Integer) where {T}
    Z = sizeof(T)

    @inbounds for c in oneto(nc)
        saxpy_kern!(s, tA, tB, side, pc + (c - 1) * ldc * Z, px, y[c], nr)
    end

    return
end

# ===== sger! =====

function sger!(s::AbstractSemiring, tA::Val, tB::Val, side::Val, x::AbstractVector{T}, y::AbstractVector, M::AbstractMatrix{T}) where {T}
    @assert size(M, 1) == length(x)
    @assert size(M, 2) == length(y)

    @preserve x M sger_kern!(s, tA, tB, side, pointer(M), stride(M, 2), pointer(x), y, length(x), length(y))

    return M
end
