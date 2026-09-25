# The direct product
#
#   ℤ₂ⁿ := ℤ₂ × ⋯ × ℤ₂
#
# of cyclic groups ℤ₂.
#
#  - elements are tuples
#  - addition is elementwise exclusive or
#
struct Elementary <: MinkowskiQuantale end

# ----- semiring -----

@generated function sprod(s::Elementary, a::T, b::T, ::Val{:N}, ::Val{:N}) where {T <: Unsigned}
    function xtree(i, n)
        if n == 1
            e = :(@inbounds tab[gnib(a, $i)])
        else
            h = n ÷ 2
            e = :($(xtree(i, h)) | xswap($(xtree(i + h, h)), Val($(4h))))
        end

        return e
    end

    return quote
        $(Expr(:meta, :inline))
        tab = gtab(gbase(s, b)...)
        return $(xtree(0, 2sizeof(T)))
    end
end

function sstar(s::Elementary, a::T) where {T <: Unsigned}
    return xperp(xperp(a))
end

function issymmetric(::Type{Elementary})
    return true
end

# ----- minkowski -----

@inline function gbase(s::Elementary, a)
    w = a
    x = xswap(a, Val(1))
    y = xswap(a, Val(2))
    z = xswap(x, Val(2))

    return w, x, y, z
end

@inline function gshift(s::Elementary, x)
    return xswap(x, Val(4))
end

@inline function gprod(s::Elementary, T::NTuple{16, V}, U::NTuple{16, V}, b::E) where {V <: Vec, E <: Unsigned}
    return gprodrec(s, T, U, b, Val(sizeof(E) - 1))
end

@inline function gprodrec(s::Elementary, T::Tuple, U::Tuple, b, ::Val{0})
    return gbyte(T, U, b, 0)
end

@inline function gprodrec(s::Elementary, T::Tuple, U::Tuple, b, ::Val{K}) where {K}
    return xbytes(gbyte(T, U, b, K), Val(K)) | gprodrec(s, T, U, b, Val(K - 1))
end

# ----- helpers -----

@generated function xmask(::Type{T}, ::Val{S}) where {T, S}
    m = zero(T)

    for i in 0:8sizeof(T) - 1
        if i & S == 0
            m |= one(T) << i
        end
    end

    return m
end

@inline function xswap(x::T, ::Val{S}) where {T <: Unsigned, S}
    m = xmask(T, Val(S))
    return ((x >> S) & m) | ((x & m) << S)
end

@inline function xswap(x::Vec{W, T}, ::Val{S}) where {W, T, S}
    m = Vec{W, T}(xmask(T, Val(S)))
    return ((x >> S) & m) | ((x & m) << S)
end

@generated function xbytes(v::Vec{W, T}, ::Val{J}) where {W, T, J}
    nb = sizeof(T)

    function f(p)
        l, i = divrem(p - 1, nb)
        return l * nb + (i ⊻ J)
    end

    return quote
        $(Expr(:meta, :inline))
        reinterpret(Vec{W, T}, shufflevector(reinterpret(Vec{$(W * nb), UInt8}, v), Val($(ntuple(f, W * nb)))))
    end
end

function xperpmake(::Type{T}) where {T <: Unsigned}
    n = 8sizeof(T)
    tab = Vector{T}(undef, 256 * sizeof(T))

    for j in 0:sizeof(T) - 1
        for v in 0:255
            m = typemax(T)

            for t in 0:7
                if isodd(v >> t)
                    a = 8j + t

                    for x in 0:n - 1
                        if isodd(count_ones(x & a))
                            m &= ~(one(T) << x)
                        end
                    end
                end
            end

            tab[256j + v + 1] = m
        end
    end

    return tab
end

const XOR_PERP_8  = xperpmake(UInt8)
const XOR_PERP_16 = xperpmake(UInt16)
const XOR_PERP_32 = xperpmake(UInt32)
const XOR_PERP_64 = xperpmake(UInt64)

function xperptab(::Type{UInt8})
    return XOR_PERP_8
end

function xperptab(::Type{UInt16})
    return XOR_PERP_16
end

function xperptab(::Type{UInt32})
    return XOR_PERP_32
end

function xperptab(::Type{UInt64})
    return XOR_PERP_64
end

function xperp(a::T) where {T <: Unsigned}
    tab = xperptab(T)
    x = typemax(T)

    @inbounds for j in 0:sizeof(T) - 1
        x &= tab[256j + (((a >> 8j) & 0xff) % Int) + 1]
    end

    return x
end
