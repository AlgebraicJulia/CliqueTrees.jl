#
# Update x and y
#
#   [ x y ] ← [ x y ] [   α ]* [     ]*      (L)
#                     [     ]  [ β   ]
#
#   [ x ] ← [ 1 α ] [ x ]                    (R)
#   [ y ]   [ β 1 ] [ y ]
#
function srotm!(s::AbstractSemiring, side::Val, α::T, β::T, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    @assert length(x) == length(y)

    if stride(x, 1) == 1 && stride(y, 1) == 1
        @preserve x y srotm_kern!(s, side, pointer(x), pointer(y), α, β, length(x))
    else
        @inbounds for i in eachindex(x, y)
            x[i], y[i] = srotm_step(s, side, x[i], y[i], α, β)
        end
    end

    return
end

function srotm_kern!(s::AbstractSemiring, side::Val, px::Ptr{T}, py::Ptr{T}, α, β, n::Integer) where {T}
    W = vecwidth(T)
    Z = sizeof(T)

    i = 1

    while i + 2W - 1 <= n
        o0 = (i - 1) * Z
        o1 = o0 + W * Z

        x0 = vload(Vec{W, T}, px + o0); y0 = vload(Vec{W, T}, py + o0)
        x1 = vload(Vec{W, T}, px + o1); y1 = vload(Vec{W, T}, py + o1)

        x0, y0 = srotm_step(s, side, x0, y0, α, β)
        x1, y1 = srotm_step(s, side, x1, y1, α, β)

        vstore(x0, px + o0); vstore(y0, py + o0)
        vstore(x1, px + o1); vstore(y1, py + o1)

        i += 2W
    end

    while i + W - 1 <= n
        o0 = (i - 1) * Z
        x0, y0 = srotm_step(s, side, vload(Vec{W, T}, px + o0), vload(Vec{W, T}, py + o0), α, β)
        vstore(x0, px + o0); vstore(y0, py + o0)
        i += W
    end

    while i <= n
        x0, y0 = srotm_step(s, side, unsafe_load(px, i), unsafe_load(py, i), α, β)
        unsafe_store!(px, x0, i); unsafe_store!(py, y0, i)
        i += 1
    end

    return
end

@inline function srotm_step(s::AbstractSemiring, ::Val{:L}, l, a, α, β)
    a2 = smuladd(s, l, α, a, Val(:N), Val(:N))
    l2 = smuladd(s, a2, β, l, Val(:N), Val(:N))
    return l2, a2
end

@inline function srotm_step(s::AbstractSemiring, ::Val{:R}, u, b, α, β)
    u2 = smuladd(s, α, b, u, Val(:N), Val(:N))
    b2 = smuladd(s, β, u, b, Val(:N), Val(:N))
    return u2, b2
end
