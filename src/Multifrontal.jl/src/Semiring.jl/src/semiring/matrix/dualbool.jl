# The semiring of 4x4 dual Boolean matrices
#
#   (2³², |, ∘)
#
# - elements are 4x4 matrices over 𝔹[ε]/(ε²)
# - addition is union
# - multiplication is matrix product
#
struct DualBoolMatrix <: AbstractSemiring end

# ----- semiring interface -----

function sid(s::DualBoolMatrix, a::UInt32, ::Val{:T})
    return tr4(a)
end

function slte(s::DualBoolMatrix, a, b)
    return splus(s, a, b, Val(:N)) == b
end

function szero(s::DualBoolMatrix, ::Type{UInt32}, ::Val{:N})
    return 0x00000000
end

function sone(s::DualBoolMatrix, ::Type{UInt32}, ::Val{:N})
    return 0x08040201
end

function splus(s::DualBoolMatrix, a, b, ::Val{:N})
    return a | b
end

function sstar(s::DualBoolMatrix, a::UInt32)
    return dcompress(sstar(BoolMatrix(), dexpand(a)))
end

function isidempotent(::Type{DualBoolMatrix})
    return true
end

# ----- (N, N) products -----

function sprod(s::DualBoolMatrix, a::UInt32, b::UInt32, ::Val{:N}, ::Val{:N})
    COL = 0x00000000000000ff
    ROW = 0x0101010101010101

    x = UInt64(b); y = dmask(a)

    t = (COL &  x)        * (ROW &  y)       |
        (COL & (x >> 8))  * (ROW & (y >> 1)) |
        (COL & (x >> 16)) * (ROW & (y >> 2)) |
        (COL & (x >> 24)) * (ROW & (y >> 3))

    return (t | dtau(t >> 32)) % UInt32
end

function smuladd(s::DualBoolMatrix, a::UInt32, b::UInt32, c::UInt32, ::Val{:N}, ::Val{:N})
    return sprod(s, a, b, Val(:N), Val(:N)) | c
end

@inline function stables(s::DualBoolMatrix, b::UInt32)
    lo = ortable(b)
    hi = reinterpret(Vec{16, UInt8}, dtau(reinterpret(Vec{2, UInt64}, lo)))
    return lo, hi
end

@inline function smuladd(s::DualBoolMatrix, a::Vec{W, UInt32}, b::UInt32, c::Vec{W, UInt32}, ::Val{:N}, ::Val{:N}) where {W}
    lo, hi = stables(s, b)

    il = reinterpret(Vec{4W, UInt8},  a       & 0x0f0f0f0f)
    ih = reinterpret(Vec{4W, UInt8}, (a >> 4) & 0x0f0f0f0f)

    r = lookup(lo, il) | lookup(hi, ih)
    return c | reinterpret(Vec{W, UInt32}, r)
end

@inline function smuladd(s::DualBoolMatrix, a::UInt32, b::Vec{W, UInt32}, c::Vec{W, UInt32}, ::Val{:N}, ::Val{:N}) where {W}
    ROW = 0x0101010101010101

    b8 = reinterpret(Vec{4W, UInt8}, b)
    z8 = reinterpret(Vec{4W, UInt8}, dtau(b))
    c8 = reinterpret(Vec{4W, UInt8}, c)
    y  = dmask(a)

    x = y        & ROW; m = (x << 8) - x
    c8 |= rbc4(b8, Val(0)) & reinterpret(Vec{4W, UInt8}, Vec{W, UInt32}(m % UInt32))
    c8 |= rbc4(z8, Val(0)) & reinterpret(Vec{4W, UInt8}, Vec{W, UInt32}((m >> 32) % UInt32))

    x = (y >> 1) & ROW; m = (x << 8) - x
    c8 |= rbc4(b8, Val(1)) & reinterpret(Vec{4W, UInt8}, Vec{W, UInt32}(m % UInt32))
    c8 |= rbc4(z8, Val(1)) & reinterpret(Vec{4W, UInt8}, Vec{W, UInt32}((m >> 32) % UInt32))

    x = (y >> 2) & ROW; m = (x << 8) - x
    c8 |= rbc4(b8, Val(2)) & reinterpret(Vec{4W, UInt8}, Vec{W, UInt32}(m % UInt32))
    c8 |= rbc4(z8, Val(2)) & reinterpret(Vec{4W, UInt8}, Vec{W, UInt32}((m >> 32) % UInt32))

    x = (y >> 3) & ROW; m = (x << 8) - x
    c8 |= rbc4(b8, Val(3)) & reinterpret(Vec{4W, UInt8}, Vec{W, UInt32}(m % UInt32))
    c8 |= rbc4(z8, Val(3)) & reinterpret(Vec{4W, UInt8}, Vec{W, UInt32}((m >> 32) % UInt32))

    return reinterpret(Vec{W, UInt32}, c8)
end

@inline function smuladd(s::DualBoolMatrix, a::Vec{W, UInt32}, b::Vec{W, UInt32}, c::Vec{W, UInt32}, ::Val{:N}, ::Val{:N}) where {W}
    t = rmuladd4(convert(Vec{W, UInt64}, b), dmask(a))
    return convert(Vec{W, UInt32}, t | dtau(t >> 32)) | c
end

@inline function sprod(s::DualBoolMatrix, a::Vec{W, UInt32}, b::UInt32, ::Val{:N}, ::Val{:N}) where {W}
    return smuladd(s, a, b, zero(Vec{W, UInt32}), Val(:N), Val(:N))
end

@inline function sprod(s::DualBoolMatrix, a::UInt32, b::Vec{W, UInt32}, ::Val{:N}, ::Val{:N}) where {W}
    return smuladd(s, a, b, zero(Vec{W, UInt32}), Val(:N), Val(:N))
end

@inline function sprod(s::DualBoolMatrix, a::Vec{W, UInt32}, b::Vec{W, UInt32}, ::Val{:N}, ::Val{:N}) where {W}
    return smuladd(s, a, b, zero(Vec{W, UInt32}), Val(:N), Val(:N))
end

# ----- transposed products -----

const DUALBOOLMATRIX_PRODUCTS = (
    (:T, :N) => (:(tr4(a)), :b),
    (:N, :T) => (:a,        :(tr4(b))),
    (:T, :T) => (:(tr4(a)), :(tr4(b))),
)

const DUALBOOLMATRIX_OPERANDS = (
    (:(Vec{W, UInt32}), :UInt32,           :(Vec{W, UInt32})),
    (:UInt32,           :(Vec{W, UInt32}), :(Vec{W, UInt32})),
    (:(Vec{W, UInt32}), :(Vec{W, UInt32}), :(Vec{W, UInt32})),
)

for ((TA, TB), (x, y)) in DUALBOOLMATRIX_PRODUCTS
    tA = :(::Val{$(QuoteNode(TA))})
    tB = :(::Val{$(QuoteNode(TB))})

    @eval function sprod(s::DualBoolMatrix, a::UInt32, b::UInt32, $tA, $tB)
        return sprod(s, $x, $y, Val(:N), Val(:N))
    end

    @eval function smuladd(s::DualBoolMatrix, a::UInt32, b::UInt32, c::UInt32, $tA, $tB)
        return smuladd(s, $x, $y, c, Val(:N), Val(:N))
    end

    for (A, B, C) in DUALBOOLMATRIX_OPERANDS
        @eval @inline function smuladd(s::DualBoolMatrix, a::$A, b::$B, c::$C, $tA, $tB) where {W}
            return smuladd(s, $x, $y, c, Val(:N), Val(:N))
        end
    end
end

# ----- pack / unpack -----

function pack(s::DualBoolMatrix, P::AbstractMatrix, E::AbstractMatrix)
    @assert size(P) == size(E) == (4, 4)

    w = 0x00000000

    for i in 1:4
        for j in 1:4
            if !iszero(P[i, j])
                w |= 0x00000001 << (8(i - 1) + (j - 1))
            end

            if !iszero(E[i, j])
                w |= 0x00000001 << (8(i - 1) + (j + 3))
            end
        end
    end

    return w
end

function unpack(s::DualBoolMatrix, w::UInt32)
    P = BitMatrix(undef, 4, 4)
    E = BitMatrix(undef, 4, 4)

    for i in 1:4
        for j in 1:4
            P[i, j] = isodd(w >> (8(i - 1) + (j - 1)))
            E[i, j] = isodd(w >> (8(i - 1) + (j + 3)))
        end
    end

    return P, E
end

# ----- helpers -----

@inline function dtau(x::T) where {T <: Union{UInt32, UInt64}}
    return (x & (0x0f0f0f0f0f0f0f0f % T)) << 4
end

@inline function dtau(x::Vec{W, T}) where {W, T <: Union{UInt32, UInt64}}
    return (x & (0x0f0f0f0f0f0f0f0f % T)) << 4
end

@inline function dmask(a::UInt32)
    return UInt64(a) | (UInt64((a >> 4) & 0x0f0f0f0f) << 32)
end

@inline function dmask(a::Vec{W, UInt32}) where {W}
    x = convert(Vec{W, UInt64}, a)
    return x | (((x >> 4) & 0x0f0f0f0f) << 32)
end

@inline function dexpand(w::UInt32)
    return UInt64(w) | (UInt64(dtau(w)) << 32)
end

@inline function dcompress(w::UInt64)
    return w % UInt32
end
