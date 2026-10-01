# The semiring of 8x8 binary relations
#
#   (2⁶⁴, |, ∘)
#
# - elements are binary relations
# - addition is union
# - multiplication is relative product
#
struct BoolMatrix <: AbstractSemiring end

# ----- semiring interface -----

function sid(s::BoolMatrix, a, ::Val{:T})
    return btr(a)
end

function sid(s::BoolMatrix, a, ::Val{:R})
    return ~a
end

function sid(s::BoolMatrix, a, ::Val{:C})
    return ~btr(a)
end

function slte(s::BoolMatrix, a, b)
    return splus(s, a, b, Val(:N)) == b
end

function szero(s::BoolMatrix, ::Type{UInt64}, ::Val{:N})
    return 0x0000000000000000
end

function szero(s::BoolMatrix, ::Type{UInt64}, ::Val{:C})
    return 0xffffffffffffffff
end

function sone(s::BoolMatrix, ::Type{UInt64}, ::Val{:N})
    return 0x8040201008040201
end

function splus(s::BoolMatrix, a, b, ::Val{:N})
    return a | b
end

function splus(s::BoolMatrix, a, b, ::Val{:C})
    return a & b
end

function sprod(s::BoolMatrix, a::UInt64, b::UInt64, ::Val{:N}, ::Val{:N})
    COL = 0x00000000000000ff
    ROW = 0x0101010101010101

    c = (COL &  b)        * (ROW &  a)       |
        (COL & (b >> 8))  * (ROW & (a >> 1)) |
        (COL & (b >> 16)) * (ROW & (a >> 2)) |
        (COL & (b >> 24)) * (ROW & (a >> 3)) |
        (COL & (b >> 32)) * (ROW & (a >> 4)) |
        (COL & (b >> 40)) * (ROW & (a >> 5)) |
        (COL & (b >> 48)) * (ROW & (a >> 6)) |
        (COL & (b >> 56)) * (ROW & (a >> 7))

    return c
end

@inline function sprod(s::BoolMatrix, a::Vec{W, UInt64}, b::UInt64, ::Val{:N}, ::Val{:N}) where {W}
    return smuladd(s, a, b, zero(Vec{W, UInt64}), Val(:N), Val(:N))
end

@inline function sprod(s::BoolMatrix, a::UInt64, b::Vec{W, UInt64}, ::Val{:N}, ::Val{:N}) where {W}
    return smuladd(s, a, b, zero(Vec{W, UInt64}), Val(:N), Val(:N))
end

@inline function sprod(s::BoolMatrix, a::Vec{W, UInt64}, b::Vec{W, UInt64}, ::Val{:N}, ::Val{:N}) where {W}
    return smuladd(s, a, b, zero(Vec{W, UInt64}), Val(:N), Val(:N))
end

function sprod(s::BoolMatrix, a::UInt64, b::UInt64, ::Val{:C}, ::Val{:N})
    return ~sprod(s, btr(a), ~b, Val(:N), Val(:N))
end

function sprod(s::BoolMatrix, a::UInt64, b::UInt64, ::Val{:N}, ::Val{:C})
    return ~sprod(s, ~a, btr(b), Val(:N), Val(:N))
end

function sprod(s::BoolMatrix, a::UInt64, b::UInt64, ::Val{:T}, ::Val{:N})
    return sprod(s, btr(a), b, Val(:N), Val(:N))
end

function sprod(s::BoolMatrix, a::UInt64, b::UInt64, ::Val{:N}, ::Val{:T})
    return sprod(s, a, btr(b), Val(:N), Val(:N))
end

function sprod(s::BoolMatrix, a::UInt64, b::UInt64, ::Val{:R}, ::Val{:N})
    return ~sprod(s, a, ~b, Val(:N), Val(:N))
end

function sprod(s::BoolMatrix, a::UInt64, b::UInt64, ::Val{:N}, ::Val{:R})
    return ~sprod(s, ~a, b, Val(:N), Val(:N))
end

function sprod(s::BoolMatrix, a::UInt64, b::UInt64, ::Val{:T}, ::Val{:T})
    return btr(sprod(s, b, a, Val(:N), Val(:N)))
end

function sprod(s::BoolMatrix, a::UInt64, b::UInt64, ::Val{:T}, ::Val{:R})
    return ~sprod(s, ~btr(a), b, Val(:N), Val(:N))
end

function sprod(s::BoolMatrix, a::UInt64, b::UInt64, ::Val{:T}, ::Val{:C})
    return ~btr(sprod(s, b, ~a, Val(:N), Val(:N)))
end

function sprod(s::BoolMatrix, a::UInt64, b::UInt64, ::Val{:R}, ::Val{:T})
    return ~sprod(s, a, ~btr(b), Val(:N), Val(:N))
end

function sprod(s::BoolMatrix, a::UInt64, b::UInt64, ::Val{:C}, ::Val{:T})
    return ~btr(sprod(s, ~b, a, Val(:N), Val(:N)))
end

function smuladd(s::BoolMatrix, a::UInt64, b::UInt64, c::UInt64, ::Val{:N}, ::Val{:N})
    return splus(s, sprod(s, a, b, Val(:N), Val(:N)), c, Val(:N))
end

function smuladd(s::BoolMatrix, a::UInt64, b::UInt64, c::UInt64, ::Val{:C}, ::Val{:N})
    return splus(s, sprod(s, a, b, Val(:C), Val(:N)), c, Val(:C))
end

function smuladd(s::BoolMatrix, a::UInt64, b::UInt64, c::UInt64, ::Val{:N}, ::Val{:C})
    return splus(s, sprod(s, a, b, Val(:N), Val(:C)), c, Val(:C))
end

function smuladd(s::BoolMatrix, a::UInt64, b::UInt64, c::UInt64, ::Val{:T}, ::Val{:N})
    return splus(s, sprod(s, a, b, Val(:T), Val(:N)), c, Val(:N))
end

function smuladd(s::BoolMatrix, a::UInt64, b::UInt64, c::UInt64, ::Val{:N}, ::Val{:T})
    return splus(s, sprod(s, a, b, Val(:N), Val(:T)), c, Val(:N))
end

function smuladd(s::BoolMatrix, a::UInt64, b::UInt64, c::UInt64, ::Val{:R}, ::Val{:N})
    return splus(s, sprod(s, a, b, Val(:R), Val(:N)), c, Val(:C))
end

function smuladd(s::BoolMatrix, a::UInt64, b::UInt64, c::UInt64, ::Val{:N}, ::Val{:R})
    return splus(s, sprod(s, a, b, Val(:N), Val(:R)), c, Val(:C))
end

function smuladd(s::BoolMatrix, a::UInt64, b::UInt64, c::UInt64, ::Val{:T}, ::Val{:T})
    return splus(s, sprod(s, a, b, Val(:T), Val(:T)), c, Val(:N))
end

function smuladd(s::BoolMatrix, a::UInt64, b::UInt64, c::UInt64, ::Val{:T}, ::Val{:R})
    return splus(s, sprod(s, a, b, Val(:T), Val(:R)), c, Val(:C))
end

function smuladd(s::BoolMatrix, a::UInt64, b::UInt64, c::UInt64, ::Val{:T}, ::Val{:C})
    return splus(s, sprod(s, a, b, Val(:T), Val(:C)), c, Val(:C))
end

function smuladd(s::BoolMatrix, a::UInt64, b::UInt64, c::UInt64, ::Val{:R}, ::Val{:T})
    return splus(s, sprod(s, a, b, Val(:R), Val(:T)), c, Val(:C))
end

function smuladd(s::BoolMatrix, a::UInt64, b::UInt64, c::UInt64, ::Val{:C}, ::Val{:T})
    return splus(s, sprod(s, a, b, Val(:C), Val(:T)), c, Val(:C))
end

@inline function stables(s::BoolMatrix, b::UInt64)
    return ortable(b % UInt32), ortable((b >> 32) % UInt32)
end

#
# Vector A, scalar B (the GEMM micro-kernel): two table
# lookups per byte.
#
@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::UInt64, c::Vec{W, UInt64}, ::Val{:N}, ::Val{:N}) where {W}
    lo, hi = stables(s, b)

    il = reinterpret(Vec{8W, UInt8},  a       & 0x0f0f0f0f0f0f0f0f)
    ih = reinterpret(Vec{8W, UInt8}, (a >> 4) & 0x0f0f0f0f0f0f0f0f)

    r = lookup(lo, il) | lookup(hi, ih)
    return c | reinterpret(Vec{W, UInt64}, r)
end

#
# Scalar A, vector B: broadcast row k of every B to its whole
# lane and mask the rows i of the result with A[i, k].
#
@inline function smuladd(s::BoolMatrix, a::UInt64, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:N}, ::Val{:N}) where {W}
    ROW = 0x0101010101010101

    b8 = reinterpret(Vec{8W, UInt8}, b)
    c8 = reinterpret(Vec{8W, UInt8}, c)

    x =  a       & ROW; m = (x << 8) - x; c8 |= rbc(b8, Val(0)) & reinterpret(Vec{8W, UInt8}, Vec{W, UInt64}(m))
    x = (a >> 1) & ROW; m = (x << 8) - x; c8 |= rbc(b8, Val(1)) & reinterpret(Vec{8W, UInt8}, Vec{W, UInt64}(m))
    x = (a >> 2) & ROW; m = (x << 8) - x; c8 |= rbc(b8, Val(2)) & reinterpret(Vec{8W, UInt8}, Vec{W, UInt64}(m))
    x = (a >> 3) & ROW; m = (x << 8) - x; c8 |= rbc(b8, Val(3)) & reinterpret(Vec{8W, UInt8}, Vec{W, UInt64}(m))
    x = (a >> 4) & ROW; m = (x << 8) - x; c8 |= rbc(b8, Val(4)) & reinterpret(Vec{8W, UInt8}, Vec{W, UInt64}(m))
    x = (a >> 5) & ROW; m = (x << 8) - x; c8 |= rbc(b8, Val(5)) & reinterpret(Vec{8W, UInt8}, Vec{W, UInt64}(m))
    x = (a >> 6) & ROW; m = (x << 8) - x; c8 |= rbc(b8, Val(6)) & reinterpret(Vec{8W, UInt8}, Vec{W, UInt64}(m))
    x = (a >> 7) & ROW; m = (x << 8) - x; c8 |= rbc(b8, Val(7)) & reinterpret(Vec{8W, UInt8}, Vec{W, UInt64}(m))

    return reinterpret(Vec{W, UInt64}, c8)
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:N}, ::Val{:N}) where {W}
    return bmuladd(a, b, c)
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::UInt64, c::Vec{W, UInt64}, ::Val{:C}, ::Val{:N}) where {W}
    return c & ~sprod(s, btr(a), ~b, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::UInt64, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:C}, ::Val{:N}) where {W}
    return c & ~sprod(s, btr(a), ~b, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:C}, ::Val{:N}) where {W}
    return c & ~sprod(s, btr(a), ~b, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::UInt64, c::Vec{W, UInt64}, ::Val{:N}, ::Val{:C}) where {W}
    return c & ~sprod(s, ~a, btr(b), Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::UInt64, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:N}, ::Val{:C}) where {W}
    return c & ~sprod(s, ~a, btr(b), Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:N}, ::Val{:C}) where {W}
    return c & ~sprod(s, ~a, btr(b), Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::UInt64, c::Vec{W, UInt64}, ::Val{:T}, ::Val{:N}) where {W}
    return smuladd(s, btr(a), b, c, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::UInt64, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:T}, ::Val{:N}) where {W}
    return smuladd(s, btr(a), b, c, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:T}, ::Val{:N}) where {W}
    return smuladd(s, btr(a), b, c, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::UInt64, c::Vec{W, UInt64}, ::Val{:N}, ::Val{:T}) where {W}
    return smuladd(s, a, btr(b), c, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::UInt64, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:N}, ::Val{:T}) where {W}
    return smuladd(s, a, btr(b), c, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:N}, ::Val{:T}) where {W}
    return smuladd(s, a, btr(b), c, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::UInt64, c::Vec{W, UInt64}, ::Val{:R}, ::Val{:N}) where {W}
    return c & ~sprod(s, a, ~b, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::UInt64, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:R}, ::Val{:N}) where {W}
    return c & ~sprod(s, a, ~b, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:R}, ::Val{:N}) where {W}
    return c & ~sprod(s, a, ~b, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::UInt64, c::Vec{W, UInt64}, ::Val{:N}, ::Val{:R}) where {W}
    return c & ~sprod(s, ~a, b, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::UInt64, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:N}, ::Val{:R}) where {W}
    return c & ~sprod(s, ~a, b, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:N}, ::Val{:R}) where {W}
    return c & ~sprod(s, ~a, b, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::UInt64, c::Vec{W, UInt64}, ::Val{:T}, ::Val{:T}) where {W}
    return smuladd(s, btr(a), btr(b), c, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::UInt64, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:T}, ::Val{:T}) where {W}
    return smuladd(s, btr(a), btr(b), c, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:T}, ::Val{:T}) where {W}
    return smuladd(s, btr(a), btr(b), c, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::UInt64, c::Vec{W, UInt64}, ::Val{:T}, ::Val{:R}) where {W}
    return c & ~sprod(s, ~btr(a), b, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::UInt64, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:T}, ::Val{:R}) where {W}
    return c & ~sprod(s, ~btr(a), b, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:T}, ::Val{:R}) where {W}
    return c & ~sprod(s, ~btr(a), b, Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::UInt64, c::Vec{W, UInt64}, ::Val{:T}, ::Val{:C}) where {W}
    return c & ~sprod(s, ~btr(a), btr(b), Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::UInt64, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:T}, ::Val{:C}) where {W}
    return c & ~sprod(s, ~btr(a), btr(b), Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:T}, ::Val{:C}) where {W}
    return c & ~sprod(s, ~btr(a), btr(b), Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::UInt64, c::Vec{W, UInt64}, ::Val{:R}, ::Val{:T}) where {W}
    return c & ~sprod(s, a, ~btr(b), Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::UInt64, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:R}, ::Val{:T}) where {W}
    return c & ~sprod(s, a, ~btr(b), Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:R}, ::Val{:T}) where {W}
    return c & ~sprod(s, a, ~btr(b), Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::UInt64, c::Vec{W, UInt64}, ::Val{:C}, ::Val{:T}) where {W}
    return c & ~sprod(s, btr(a), ~btr(b), Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::UInt64, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:C}, ::Val{:T}) where {W}
    return c & ~sprod(s, btr(a), ~btr(b), Val(:N), Val(:N))
end

@inline function smuladd(s::BoolMatrix, a::Vec{W, UInt64}, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:C}, ::Val{:T}) where {W}
    return c & ~sprod(s, btr(a), ~btr(b), Val(:N), Val(:N))
end

#
# Warshall's algorithm; the closure commutes with transposition,
# so the row/column convention does not matter here.
#
function sstar(s::BoolMatrix, a::UInt64)
    COL = 0x00000000000000ff
    ROW = 0x0101010101010101
    I   = 0x8040201008040201

    a |= I
    a |= (ROW &  a)       * (COL &  a)
    a |= (ROW & (a >> 1)) * (COL & (a >> 8))
    a |= (ROW & (a >> 2)) * (COL & (a >> 16))
    a |= (ROW & (a >> 3)) * (COL & (a >> 24))
    a |= (ROW & (a >> 4)) * (COL & (a >> 32))
    a |= (ROW & (a >> 5)) * (COL & (a >> 40))
    a |= (ROW & (a >> 6)) * (COL & (a >> 48))
    a |= (ROW & (a >> 7)) * (COL & (a >> 56))

    return a
end

function isidempotent(::Type{BoolMatrix})
    return true
end

# ----- pack / unpack -----

function pack(s::BoolMatrix, M::AbstractMatrix)
    @assert size(M) == (8, 8)

    w = 0x0000000000000000

    for i in 1:8
        for j in 1:8
            if !iszero(M[i, j])
                w |= 0x0000000000000001 << (8(i - 1) + (j - 1))
            end
        end
    end

    return w
end

function unpack(s::BoolMatrix, w::UInt64)
    M = BitMatrix(undef, 8, 8)

    for i in 1:8
        for j in 1:8
            M[i, j] = isodd(w >> (8(i - 1) + (j - 1)))
        end
    end

    return M
end

# ----- helpers -----

@inline function btr(a)
    b = ((a >> 7)  ⊻ a) & 0x00aa00aa00aa00aa
    a = a ⊻ b ⊻ (b << 7)

    b = ((a >> 14) ⊻ a) & 0x0000cccc0000cccc
    a = a ⊻ b ⊻ (b << 14)

    b = ((a >> 28) ⊻ a) & 0x00000000f0f0f0f0
    a = a ⊻ b ⊻ (b << 28)

    return a
end

#
# c ⊕ a b for two vectors of matrices: row i of each product takes row k
# of b wherever bit k of row i of a is set, the row selected by a byte test
#
@inline function bmuladd(a::Vec{W, UInt64}, b::Vec{W, UInt64}, c::Vec{W, UInt64}) where {W}
    a8 = reinterpret(Vec{8W, UInt8}, a)
    b8 = reinterpret(Vec{8W, UInt8}, b)
    c8 = reinterpret(Vec{8W, UInt8}, c)

    @nexprs 8 k -> begin
        c8 = vifelse((a8 & (0x01 << (k - 1))) != 0x00, c8 | rbc(b8, Val(k - 1)), c8)
    end

    return reinterpret(Vec{W, UInt64}, c8)
end

@generated function rbc(v::Vec{W, UInt8}, ::Val{K}) where {W, K}
    function f(i)
        im1 = i - 1
        return (im1 & ~7) + K
    end

    return :(shufflevector(v, Val($(ntuple(f, W)))))
end
