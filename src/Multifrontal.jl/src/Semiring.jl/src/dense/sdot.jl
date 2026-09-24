# ===== sdot_kern! =====

function sdot_kern!(s::AbstractSemiring, tA::Val, tB::Val, op::Val, pa::Ptr{T}, pb::Ptr{T}, nj::Integer) where {T}
    W = vecwidth(T)
    Z = sizeof(T)

    z = szero(s, T, op)

    d0 = Vec{W, T}(z)
    d1 = Vec{W, T}(z)
    d2 = Vec{W, T}(z)
    d3 = Vec{W, T}(z)

    j = 1

    while j + 4W - 1 <= nj
        o0 = (j - 1) * Z
        o1 = o0 + W * Z
        o2 = o1 + W * Z
        o3 = o2 + W * Z

        d0 = smuladd(s, vload(Vec{W, T}, pa + o0), vload(Vec{W, T}, pb + o0), d0, tA, tB)
        d1 = smuladd(s, vload(Vec{W, T}, pa + o1), vload(Vec{W, T}, pb + o1), d1, tA, tB)
        d2 = smuladd(s, vload(Vec{W, T}, pa + o2), vload(Vec{W, T}, pb + o2), d2, tA, tB)
        d3 = smuladd(s, vload(Vec{W, T}, pa + o3), vload(Vec{W, T}, pb + o3), d3, tA, tB)

        j += 4W
    end

    d = splus(s, d0, d1, d2, d3, op)

    while j + W - 1 <= nj
        o0 = (j - 1) * Z
        d = smuladd(s, vload(Vec{W, T}, pa + o0), vload(Vec{W, T}, pb + o0), d, tA, tB)
        j += W
    end

    r = z

    for l in 1:W
        r = splus(s, r, d[l], op)
    end

    while j <= nj
        r = smuladd(s, unsafe_load(pa, j), unsafe_load(pb, j), r, tA, tB)
        j += 1
    end

    return r
end

# ===== sdot =====

function sdot(s::AbstractSemiring, tA::Val, tB::Val, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    @assert length(x) == length(y)

    op = compose(tA, tB)

    return @preserve x y sdot_kern!(s, tA, tB, op, pointer(x), pointer(y), length(x))
end
