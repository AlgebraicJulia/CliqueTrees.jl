# The semiring of 4x4 qualitative matrices
#
#   (2³², |, ∘)
#
# - elements are 4x4 matrices over the sign algebra {0, +, -, ?}
# - addition is union
# - multiplication is matrix product
#
struct QualMatrix <: AbstractSemiring end

# ----- semiring interface -----

function sid(s::QualMatrix, a::UInt32, ::Val{:T})
    return tr4(a)
end

function sid(s::QualMatrix, a::UInt32, ::Val{:R})
    return ~a
end

function sid(s::QualMatrix, a::UInt32, ::Val{:C})
    return ~tr4(a)
end

function slte(s::QualMatrix, a, b)
    return splus(s, a, b, Val(:N)) == b
end

function szero(s::QualMatrix, ::Type{UInt32}, ::Val{:N})
    return 0x00000000
end

function szero(s::QualMatrix, ::Type{UInt32}, ::Val{:C})
    return 0xffffffff
end

function sone(s::QualMatrix, ::Type{UInt32}, ::Val{:N})
    return 0x08040201
end

function splus(s::QualMatrix, a, b, ::Val{:N})
    return a | b
end

function splus(s::QualMatrix, a, b, ::Val{:C})
    return a & b
end

function sstar(s::QualMatrix, a::UInt32)
    return scompress(sstar(BoolMatrix(), sexpand(a)))
end

function isidempotent(::Type{QualMatrix})
    return true
end

# ----- (N, N) products -----

function sprod(s::QualMatrix, a::UInt32, b::UInt32, ::Val{:N}, ::Val{:N})
    COL = 0x00000000000000ff
    ROW = 0x0101010101010101

    x = UInt64(b); y = sexpand(a)

    t = (COL &  x)        * (ROW &  y)       |
        (COL & (x >> 8))  * (ROW & (y >> 1)) |
        (COL & (x >> 16)) * (ROW & (y >> 2)) |
        (COL & (x >> 24)) * (ROW & (y >> 3))

    return scompress(t | (nibswap(t) >> 32))
end

function smuladd(s::QualMatrix, a::UInt32, b::UInt32, c::UInt32, ::Val{:N}, ::Val{:N})
    return sprod(s, a, b, Val(:N), Val(:N)) | c
end

@inline function stables(s::QualMatrix, b::UInt32)
    return qtables(b)
end

@inline function smuladd(s::QualMatrix, a::Vec{W, UInt32}, b::UInt32, c::Vec{W, UInt32}, ::Val{:N}, ::Val{:N}) where {W}
    lo, hi = stables(s, b)

    il = reinterpret(Vec{4W, UInt8},  a       & 0x0f0f0f0f)
    ih = reinterpret(Vec{4W, UInt8}, (a >> 4) & 0x0f0f0f0f)

    r = lookup(lo, il) | lookup(hi, ih)
    return c | reinterpret(Vec{W, UInt32}, r)
end

@inline function smuladd(s::QualMatrix, a::UInt32, b::Vec{W, UInt32}, c::Vec{W, UInt32}, ::Val{:N}, ::Val{:N}) where {W}
    ROW = 0x0101010101010101

    b8 = reinterpret(Vec{4W, UInt8}, b)
    z8 = reinterpret(Vec{4W, UInt8}, nibswap(b))
    c8 = reinterpret(Vec{4W, UInt8}, c)
    y  = sexpand(a)

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

@inline function smuladd(s::QualMatrix, a::Vec{W, UInt32}, b::Vec{W, UInt32}, c::Vec{W, UInt32}, ::Val{:N}, ::Val{:N}) where {W}
    t = rmuladd4(convert(Vec{W, UInt64}, b), sexpand(a))
    return scompress(t | (nibswap(t) >> 32)) | c
end

@inline function sprod(s::QualMatrix, a::Vec{W, UInt32}, b::UInt32, ::Val{:N}, ::Val{:N}) where {W}
    return smuladd(s, a, b, zero(Vec{W, UInt32}), Val(:N), Val(:N))
end

@inline function sprod(s::QualMatrix, a::UInt32, b::Vec{W, UInt32}, ::Val{:N}, ::Val{:N}) where {W}
    return smuladd(s, a, b, zero(Vec{W, UInt32}), Val(:N), Val(:N))
end

@inline function sprod(s::QualMatrix, a::Vec{W, UInt32}, b::Vec{W, UInt32}, ::Val{:N}, ::Val{:N}) where {W}
    return smuladd(s, a, b, zero(Vec{W, UInt32}), Val(:N), Val(:N))
end

# ----- other products -----

const QUALMATRIX_PRODUCTS = (
    #  flags        left       right      dual
    (:C, :N) => (:(tr4(a)),  :(~b),      true),
    (:N, :C) => (:(~a),      :(tr4(b)),  true),
    (:T, :N) => (:(tr4(a)),  :b,         false),
    (:N, :T) => (:a,         :(tr4(b)),  false),
    (:R, :N) => (:a,         :(~b),      true),
    (:N, :R) => (:(~a),      :b,         true),
    (:T, :T) => (:(tr4(a)),  :(tr4(b)),  false),
    (:T, :R) => (:(~tr4(a)), :b,         true),
    (:T, :C) => (:(~tr4(a)), :(tr4(b)),  true),
    (:R, :T) => (:a,         :(~tr4(b)), true),
    (:C, :T) => (:(tr4(a)),  :(~tr4(b)), true),
)

const QUALMATRIX_OPERANDS = (
    (:(Vec{W, UInt32}), :UInt32,           :(Vec{W, UInt32})),
    (:UInt32,           :(Vec{W, UInt32}), :(Vec{W, UInt32})),
    (:(Vec{W, UInt32}), :(Vec{W, UInt32}), :(Vec{W, UInt32})),
)

for ((TA, TB), (x, y, dual)) in QUALMATRIX_PRODUCTS
    tA = :(::Val{$(QuoteNode(TA))})
    tB = :(::Val{$(QuoteNode(TB))})

    if dual
        prd = :(~sprod(s, $x, $y, Val(:N), Val(:N)))
        acc = :(c & ~sprod(s, $x, $y, Val(:N), Val(:N)))
    else
        prd = :(sprod(s, $x, $y, Val(:N), Val(:N)))
        acc = :(smuladd(s, $x, $y, c, Val(:N), Val(:N)))
    end

    @eval function sprod(s::QualMatrix, a::UInt32, b::UInt32, $tA, $tB)
        return $prd
    end

    @eval function smuladd(s::QualMatrix, a::UInt32, b::UInt32, c::UInt32, $tA, $tB)
        return $acc
    end

    for (A, B, C) in QUALMATRIX_OPERANDS
        @eval @inline function smuladd(s::QualMatrix, a::$A, b::$B, c::$C, $tA, $tB) where {W}
            return $acc
        end
    end
end

# ----- pack / unpack -----

function pack(s::QualMatrix, M::AbstractMatrix)
    @assert size(M) == (4, 4)

    w = 0x00000000

    for j in 1:4
        for i in 1:4
            p, n = signbits(M[i, j])

            if p
                w |= 0x00000001 << (8(i - 1) + (j - 1))
            end

            if n
                w |= 0x00000001 << (8(i - 1) + (j + 3))
            end
        end
    end

    return w
end

function unpack(s::QualMatrix, w::UInt32)
    M = Matrix{Char}(undef, 4, 4)

    for j in 1:4
        for i in 1:4
            p = isodd(w >> (8(i - 1) + (j - 1)))
            n = isodd(w >> (8(i - 1) + (j + 3)))
            M[i, j] = p ? (n ? '?' : '+') : (n ? '-' : '0')
        end
    end

    return M
end

# ----- helpers -----

function signbits(x::AbstractChar)
    x == '0' && return (false, false)
    x == '+' && return (true,  false)
    x == '-' && return (false, true)
    x == '?' && return (true,  true)
    return error("unknown sign '$x'")
end

function signbits(x::Real)
    isnan(x) && return (true, true)
    return (x > 0, x < 0)
end

@inline function nibswap(x::T) where {T <: Union{UInt32, UInt64}}
    M = 0x0f0f0f0f0f0f0f0f % T
    return ((x >> 4) & M) | ((x & M) << 4)
end

@inline function nibswap(x::Vec{W, T}) where {W, T <: Union{UInt32, UInt64}}
    M = 0x0f0f0f0f0f0f0f0f % T
    return ((x >> 4) & M) | ((x & M) << 4)
end

@inline function sexpand(w::UInt32)
    return UInt64(w) | (UInt64(nibswap(w)) << 32)
end

@inline function sexpand(w::Vec{W, UInt32}) where {W}
    x = convert(Vec{W, UInt64}, w)
    return x | (nibswap(x) << 32)
end

@inline function scompress(w::UInt64)
    return w % UInt32
end

@inline function scompress(w::Vec{W, UInt64}) where {W}
    return convert(Vec{W, UInt32}, w)
end

@inline function qtables(b::UInt32)
    lo = ortable(b)
    hi = reinterpret(Vec{16, UInt8}, nibswap(reinterpret(Vec{2, UInt64}, lo)))
    return lo, hi
end
