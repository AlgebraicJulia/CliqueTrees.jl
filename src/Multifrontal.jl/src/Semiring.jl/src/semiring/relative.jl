struct RelPlus <: AbstractQuantale end

struct RelProd <: AbstractQuantale end

const RelationQuantale = Union{RelPlus, RelProd}

# ----- pack / unpack -----

function pack(s::RelProd, M::AbstractMatrix{Bool})
    @assert size(M) == (8, 8)

    w = 0x0000000000000000

    for j in 1:8
        for i in 1:8
            if M[i, j]
                w |= 0x0000000000000001 << (8(j - 1) + (i - 1))
            end
        end
    end

    return w
end

function pack(s::RelPlus, M::AbstractMatrix{Bool})
    return ~pack(RelProd(), M)
end

function unpack(s::RelProd, w::UInt64)
    M = BitMatrix(undef, 8, 8)

    for j in 1:8
        for i in 1:8
            M[i, j] = isodd(w >> (8(j - 1) + (i - 1)))
        end
    end

    return M
end

function unpack(s::RelPlus, w::UInt64)
    return unpack(RelProd(), ~w)
end

# ----- semiring interface -----

function stop(s::RelationQuantale, ::Type{UInt64})
    return 0xffffffffffffffff
end

function szero(s::RelationQuantale, ::Type{UInt64})
    return 0x0000000000000000
end

function sone(s::RelationQuantale, ::Type{UInt64})
    return 0x8040201008040201
end

function splus(s::RelationQuantale, a::UInt64, b::UInt64)
    return a | b
end

function sprod(s::RelationQuantale, a::UInt64, b::UInt64)
    COL = 0x00000000000000ff
    ROW = 0x0101010101010101

    c = (COL &  a)        * (ROW &  b)       |
        (COL & (a >> 8))  * (ROW & (b >> 1)) |
        (COL & (a >> 16)) * (ROW & (b >> 2)) |
        (COL & (a >> 24)) * (ROW & (b >> 3)) |
        (COL & (a >> 32)) * (ROW & (b >> 4)) |
        (COL & (a >> 40)) * (ROW & (b >> 5)) |
        (COL & (a >> 48)) * (ROW & (b >> 6)) |
        (COL & (a >> 56)) * (ROW & (b >> 7))

    return c
end

function sstar(s::RelationQuantale, a::UInt64)
    COL = 0x00000000000000ff
    ROW = 0x0101010101010101
    I   = 0x8040201008040201

    c = a | I
    c |= (ROW &  c)       * (COL &  c)
    c |= (ROW & (c >> 1)) * (COL & (c >> 8))
    c |= (ROW & (c >> 2)) * (COL & (c >> 16))
    c |= (ROW & (c >> 3)) * (COL & (c >> 24))
    c |= (ROW & (c >> 4)) * (COL & (c >> 32))
    c |= (ROW & (c >> 5)) * (COL & (c >> 40))
    c |= (ROW & (c >> 6)) * (COL & (c >> 48))
    c |= (ROW & (c >> 7)) * (COL & (c >> 56))

    return c
end

@generated function rbc(v::Vec{W, UInt8}, ::Val{K}) where {W, K}
    function f(i)
        im1 = i - 1
        return (im1 & ~7) + K
    end

    return :(shufflevector(v, Val($(ntuple(f, W)))))
end

@inline function smuladd(s::RelationQuantale, a::Vec{W, UInt64}, b::UInt64, c::Vec{W, UInt64}) where {W}
    ROW = 0x0101010101010101

    a8 = reinterpret(Vec{8W, UInt8}, a)
    c8 = reinterpret(Vec{8W, UInt8}, c)

    x =  b       & ROW; m = (x << 8) - x; c8 |= rbc(a8, Val(0)) & reinterpret(Vec{8W, UInt8}, Vec{W, UInt64}(m))
    x = (b >> 1) & ROW; m = (x << 8) - x; c8 |= rbc(a8, Val(1)) & reinterpret(Vec{8W, UInt8}, Vec{W, UInt64}(m))
    x = (b >> 2) & ROW; m = (x << 8) - x; c8 |= rbc(a8, Val(2)) & reinterpret(Vec{8W, UInt8}, Vec{W, UInt64}(m))
    x = (b >> 3) & ROW; m = (x << 8) - x; c8 |= rbc(a8, Val(3)) & reinterpret(Vec{8W, UInt8}, Vec{W, UInt64}(m))
    x = (b >> 4) & ROW; m = (x << 8) - x; c8 |= rbc(a8, Val(4)) & reinterpret(Vec{8W, UInt8}, Vec{W, UInt64}(m))
    x = (b >> 5) & ROW; m = (x << 8) - x; c8 |= rbc(a8, Val(5)) & reinterpret(Vec{8W, UInt8}, Vec{W, UInt64}(m))
    x = (b >> 6) & ROW; m = (x << 8) - x; c8 |= rbc(a8, Val(6)) & reinterpret(Vec{8W, UInt8}, Vec{W, UInt64}(m))
    x = (b >> 7) & ROW; m = (x << 8) - x; c8 |= rbc(a8, Val(7)) & reinterpret(Vec{8W, UInt8}, Vec{W, UInt64}(m))

    return reinterpret(Vec{W, UInt64}, c8)
end
