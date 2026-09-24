struct RelProd <: AbstractSemiring end

function isidempotent(::Type{RelProd})
    return true
end

# ----- pack / unpack -----

function pack(s::RelProd, M::AbstractMatrix)
    @assert size(M) == (8, 8)

    w = 0x0000000000000000

    for j in 1:8
        for i in 1:8
            if !iszero(M[i, j])
                w |= 0x0000000000000001 << (8(j - 1) + (i - 1))
            end
        end
    end

    return w
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

# ----- semiring interface -----

function szero(s::RelProd, ::Type{UInt64}, ::Val{:C})
    return 0xffffffffffffffff
end

function szero(s::RelProd, ::Type{UInt64}, ::Val{:N})
    return 0x0000000000000000
end

function sone(s::RelProd, ::Type{UInt64}, ::Val{:N})
    return 0x8040201008040201
end

function splus(s::RelProd, a, b, ::Val{:N})
    return a | b
end

function splus(s::RelProd, a, b, ::Val{:C})
    return a & b
end

function sprod(s::RelProd, a::UInt64, b::UInt64, ::Val{:N}, ::Val{:N})
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

function sprod(s::RelProd, a::UInt64, b::UInt64, ::Val{:C}, ::Val{:N})
    return ~sprod(s, btr(a), ~b, Val(:N), Val(:N))
end

function sprod(s::RelProd, a::UInt64, b::UInt64, ::Val{:N}, ::Val{:C})
    return ~sprod(s, ~a, btr(b), Val(:N), Val(:N))
end

function sstar(s::RelProd, a::UInt64)
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

@inline function smuladd(s::RelProd, a::Vec{W, UInt64}, b::UInt64, c::Vec{W, UInt64}, ::Val{:N}, ::Val{:N}) where {W}
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

@inline function smuladd(s::RelProd, a::Vec{W, UInt64}, b::UInt64, c::Vec{W, UInt64}, ::Val{:C}, ::Val{:N}) where {W}
    return c & ~smuladd(s, btr(a), ~b, zero(Vec{W, UInt64}), Val(:N), Val(:N))
end

@inline function smuladd(s::RelProd, a::Vec{W, UInt64}, b::UInt64, c::Vec{W, UInt64}, ::Val{:N}, ::Val{:C}) where {W}
    return c & ~smuladd(s, ~a, btr(b), zero(Vec{W, UInt64}), Val(:N), Val(:N))
end

@inline function smuladd(s::RelProd, a::UInt64, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:N}, ::Val{:N}) where {W}
    lo, hi = luts(a)
    return alut(lo, hi, b, c)
end

@inline function smuladd(s::RelProd, a::UInt64, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:C}, ::Val{:N}) where {W}
    return c & ~smuladd(s, btr(a), ~b, zero(Vec{W, UInt64}), Val(:N), Val(:N))
end

@inline function smuladd(s::RelProd, a::UInt64, b::Vec{W, UInt64}, c::Vec{W, UInt64}, ::Val{:N}, ::Val{:C}) where {W}
    return c & ~smuladd(s, ~a, btr(b), zero(Vec{W, UInt64}), Val(:N), Val(:N))
end

@inline function smuladd(s::RelProd, a::Vec{W, UInt64}, b::Vec{W, UInt64}, c::Vec{W, UInt64}, tA::Union{Val{:N}, Val{:C}}, tB::Union{Val{:N}, Val{:C}}) where {W}
    return Vec{W, UInt64}(ntuple(l -> smuladd(s, a[l], b[l], c[l], tA, tB), Val(W)))
end

function btr(a)
    b = ((a >> 7)  ⊻ a) & 0x00aa00aa00aa00aa
    a = a ⊻ b ⊻ (b << 7)

    b = ((a >> 14) ⊻ a) & 0x0000cccc0000cccc
    a = a ⊻ b ⊻ (b << 14)

    b = ((a >> 28) ⊻ a) & 0x00000000f0f0f0f0
    a = a ⊻ b ⊻ (b << 28)

    return a
end

@generated function rbc(v::Vec{W, UInt8}, ::Val{K}) where {W, K}
    function f(i)
        im1 = i - 1
        return (im1 & ~7) + K
    end

    return :(shufflevector(v, Val($(ntuple(f, W)))))
end

function tbl1(t::Vec{16, UInt8}, v::Vec{16, UInt8})
    return Vec(ccall("llvm.aarch64.neon.tbl1.v16i8", llvmcall, NTuple{16, VecElement{UInt8}},
        (NTuple{16, VecElement{UInt8}}, NTuple{16, VecElement{UInt8}}), t.data, v.data))
end

@inline function luts(a::UInt64)
    COL = 0x00000000000000ff

    function flo(i)
        n = i - 1; r = 0x00

        for j in 0:3
            if isodd(n >> j)
                r |= UInt8((a >> (8j)) & COL)
            end
        end

        return r
    end

    function fhi(i)
        n = i - 1; r = 0x00

        for j in 0:3
            if isodd(n >> j)
                r |= UInt8((a >> (8(j + 4))) & COL)
            end
        end

        return r
    end

    lo = Vec{16, UInt8}(ntuple(flo, Val(16)))
    hi = Vec{16, UInt8}(ntuple(fhi, Val(16)))

    return lo, hi
end

@generated function half(v::Vec{N, UInt8}, ::Val{O}) where {N, O}
    function f(i)
        return O + i - 1
    end

    return :(shufflevector(v, Val($(ntuple(f, N ÷ 2)))))
end

@generated function cat2(a::Vec{N, UInt8}, b::Vec{N, UInt8}) where {N}
    function f(i)
        return i - 1
    end

    return :(shufflevector(a, b, Val($(ntuple(f, 2N)))))
end

@inline function tblv(lo::Vec{16, UInt8}, hi::Vec{16, UInt8}, v::Vec{16, UInt8})
    rlo = tbl1(lo, v & 0x0f)
    rhi = tbl1(hi, v >> 0x04)
    return rlo | rhi
end

@inline function tblv(lo::Vec{16, UInt8}, hi::Vec{16, UInt8}, v::Vec{N, UInt8}) where {N}
    return cat2(tblv(lo, hi, half(v, Val(0))), tblv(lo, hi, half(v, Val(N ÷ 2))))
end

@inline function alut(lo::Vec{16, UInt8}, hi::Vec{16, UInt8}, b::Vec{W, UInt64}, c::Vec{W, UInt64}) where {W}
    bv = reinterpret(Vec{8W, UInt8}, b)
    cv = reinterpret(Vec{8W, UInt8}, c)
    return reinterpret(Vec{W, UInt64}, cv | tblv(lo, hi, bv))
end
