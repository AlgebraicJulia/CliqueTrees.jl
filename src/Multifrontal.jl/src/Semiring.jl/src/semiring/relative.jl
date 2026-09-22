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

function splus(s::RelProd, a::UInt64, b::UInt64, ::Val{:N})
    return a | b
end

function splus(s::RelProd, a::UInt64, b::UInt64, ::Val{:C})
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
    return c & ~smuladd(s, a, b, zero(Vec{W, UInt64}), Val(:N), Val(:N))
end

@inline function smuladd(s::RelProd, a::Vec{W, UInt64}, b::UInt64, c::Vec{W, UInt64}, ::Val{:N}, ::Val{:C}) where {W}
    return c & ~smuladd(s, a, b, zero(Vec{W, UInt64}), Val(:N), Val(:N))
end

function sgemx_pack_A!(s::RelProd, tA::Val{TA}, tB::Val{TB}, AP::AbstractVector, A::AbstractMatrix, ni::Int, nj::Int, z, ::Val{MR}) where {TA, TB, MR}
    @inbounds for i0 in 0:MR:ni - 1
        it = min(MR, ni - i0); ip0 = i0 * nj

        for j in 1:nj
            for ip in 1:it
                if TA === :N
                    x = A[i0 + ip, j]
                else
                    x = A[j, i0 + ip]
                end

                if TA === :C
                    x = btr(x)
                elseif TB === :C
                    x = ~x
                end

                AP[ip0 + (j - 1) * MR + ip] = x
            end

            for ip in it + 1:MR
                AP[ip0 + (j - 1) * MR + ip] = z
            end
        end
    end

    return AP
end

function sgemx_pack_B!(s::RelProd, tA::Val{TA}, tB::Val{TB}, BP::AbstractVector, B::AbstractMatrix, nk::Int, nj::Int, z) where {TA, TB}
    @inbounds for k0 in 0:SGEMX_NR:nk - 1
        kt = min(SGEMX_NR, nk - k0); kp0 = k0 * nj

        for j in 1:nj
            for kp in 1:kt
                if TB === :N
                    x = B[j, k0 + kp]
                else
                    x = B[k0 + kp, j]
                end

                if TB === :C
                    x = btr(x)
                elseif TA === :C
                    x = ~x
                end

                BP[kp0 + (j - 1) * SGEMX_NR + kp] = x
            end

            for kp in kt + 1:SGEMX_NR
                BP[kp0 + (j - 1) * SGEMX_NR + kp] = z
            end
        end
    end

    return BP
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
