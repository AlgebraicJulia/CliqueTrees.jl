# ===== saxpy_kern! =====

@inline function smul(s::AbstractSemiring, tA::Val, tB::Val, ::Val{:L}, v, x, c)
    return smuladd(s, x, v, c, tA, tB)
end

@inline function smul(s::AbstractSemiring, tA::Val, tB::Val, ::Val{:R}, v, x, c)
    return smuladd(s, v, x, c, tA, tB)
end

function saxpy_kern!(s::AbstractSemiring, tA::Val, tB::Val, side::Val, pc::Ptr{T}, pv::Ptr{T}, x, ni::Integer) where {T}
    W = vecwidth(T)
    Z = sizeof(T)

    i = 1

    while i + 4W - 1 <= ni
        o0 = (i - 1) * Z
        o1 = o0 + W * Z
        o2 = o1 + W * Z
        o3 = o2 + W * Z

        c0 = vload(Vec{W, T}, pc + o0); v0 = vload(Vec{W, T}, pv + o0)
        c1 = vload(Vec{W, T}, pc + o1); v1 = vload(Vec{W, T}, pv + o1)
        c2 = vload(Vec{W, T}, pc + o2); v2 = vload(Vec{W, T}, pv + o2)
        c3 = vload(Vec{W, T}, pc + o3); v3 = vload(Vec{W, T}, pv + o3)

        vstore(smul(s, tA, tB, side, v0, x, c0), pc + o0)
        vstore(smul(s, tA, tB, side, v1, x, c1), pc + o1)
        vstore(smul(s, tA, tB, side, v2, x, c2), pc + o2)
        vstore(smul(s, tA, tB, side, v3, x, c3), pc + o3)

        i += 4W
    end

    while i + W - 1 <= ni
        o0 = (i - 1) * Z
        vstore(smul(s, tA, tB, side, vload(Vec{W, T}, pv + o0), x, vload(Vec{W, T}, pc + o0)), pc + o0)
        i += W
    end

    while i <= ni
        unsafe_store!(pc, smul(s, tA, tB, side, unsafe_load(pv, i), x, unsafe_load(pc, i)), i)
        i += 1
    end

    return
end

# ===== saxpy! =====

function saxpy!(s::AbstractSemiring, tA::Val, tB::Val, side::Val, a, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    @assert length(x) == length(y)

    @preserve x y saxpy_kern!(s, tA, tB, side, pointer(y), pointer(x), a, length(y))

    return y
end
