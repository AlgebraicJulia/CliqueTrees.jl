struct RelPlus <: AbstractQuantale end

struct RelProd <: AbstractQuantale end

const RelationQuantale = Union{RelPlus, RelProd}

function stop(::RelPlus, ::Type{T}) where {T}
    return zero(T)
end

function stop(::RelProd, ::Type{T}) where {T}
    return ~zero(T)
end

function szero(::RelPlus, ::Type{T}) where {T}
    return ~zero(T)
end

function szero(::RelProd, ::Type{T}) where {T}
    return zero(T)
end

function sone(::RelPlus, ::Type{T}) where {T}
    return ~bid(T)
end

function sone(::RelProd, ::Type{T}) where {T}
    return bid(T)
end

function splus(::RelPlus, a, b)
    return a & b
end

function splus(::RelProd, a, b)
    return a | b
end

function sprod(::RelPlus, a, b)
    return ~bml(~a, ~b)
end

function sprod(::RelProd, a, b)
    return bml(a, b)
end

function sstar(::RelPlus, a)
    return ~bst(~a)
end

function sstar(::RelProd, a)
    return bst(a)
end

function bcm(::Type{UInt64})
    return 0x00000000000000ff
end

function bcm(::Type{Int64})
    return 255
end

function brm(::Type{UInt64})
    return 0x0101010101010101
end

function brm(::Type{Int64})
    return 72340172838076673
end

function bid(::Type{UInt64})
    return 0x8040201008040201
end

function bid(::Type{Int64})
    return -9205322385119247871
end

function bml(a::T, b::T) where {T}
    COL = bcm(T)
    ROW = brm(T)

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

function bst(a::T) where {T}
    COL = bcm(T)
    ROW = brm(T)
    I   = bid(T)

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
    ind = ntuple(W) do j
        jm1 = j - 1
        return (jm1 & ~7) + K
    end

    return :(shufflevector(v, Val($ind)))
end

function sgemx_row!(s::RelationQuantale, C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, AP::AbstractVector{T}, BP::AbstractVector{T}) where {T}
    return sgemx_rel!(s, C, A, B, AP, BP, Val(3))
end

function sgemx_rel!(s::RelationQuantale, C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, AP::AbstractVector{T}, BP::AbstractVector{T}, ::Val{TILE}) where {T, TILE}
    m = size(C, 1)
    n = size(C, 2)
    k = size(A, 2)

    V = 1 << TILE

    VecV64 = Vec{ V, T}
    Vec8V8 = Vec{8V, UInt8}

    if 8k > length(BP)
        @inbounds for j in 1:n
            for i in 1:m
                Cij = C[i, j]

                for p in 1:k
                    Cij = smuladd(s, A[i, p], B[p, j], Cij)
                end

                C[i, j] = Cij
            end
        end

        return C
    end

    mf = m - m % V

    @inbounds for i0 in 0:V:mf - V
        for p in 1:k
            pm1 = p - 1
            ap0 = i0 * k + pm1 * V

            for i in 1:V
                if s isa RelPlus
                    a = ~A[i0 + i, p]
                else
                    a =  A[i0 + i, p]
                end

                AP[ap0 + i] = a
            end
        end
    end

    ldc = stride(C, 2)

    @preserve C AP begin
        cp0 = pointer(C)
        ap0 = pointer(AP)

        @inbounds for j in 1:n
            jm1 = j - 1

            for p in 1:k
                pm1 = p - 1; q0 = 8pm1

                if s isa RelPlus
                    Bpj = ~B[p, j]
                else
                    Bpj =  B[p, j]
                end

                x =  Bpj       & brm(T); BP[q0 + 1] = (x << 8) - x
                x = (Bpj >> 1) & brm(T); BP[q0 + 2] = (x << 8) - x
                x = (Bpj >> 2) & brm(T); BP[q0 + 3] = (x << 8) - x
                x = (Bpj >> 3) & brm(T); BP[q0 + 4] = (x << 8) - x
                x = (Bpj >> 4) & brm(T); BP[q0 + 5] = (x << 8) - x
                x = (Bpj >> 5) & brm(T); BP[q0 + 6] = (x << 8) - x
                x = (Bpj >> 6) & brm(T); BP[q0 + 7] = (x << 8) - x
                x = (Bpj >> 7) & brm(T); BP[q0 + 8] = (x << 8) - x
            end

            for i0 in 0:V:mf - V
                cp = cp0 + 8(jm1 * ldc + i0)

                if s isa RelPlus
                    cV64 = ~vload(VecV64, cp)
                else
                    cV64 =  vload(VecV64, cp)
                end

                c8V8 = reinterpret(Vec8V8, cV64)

                for p in 1:k
                    pm1 = p - 1; q0 = 8pm1

                    ap = ap0 + 8(i0 * k + pm1 * V)

                    aV64 = vload(VecV64, ap)
                    a8V8 = reinterpret(Vec8V8, aV64)

                    x = BP[q0 + 1]; c8V8 |= (rbc(a8V8, Val(0)) & reinterpret(Vec8V8, VecV64(x)))
                    x = BP[q0 + 2]; c8V8 |= (rbc(a8V8, Val(1)) & reinterpret(Vec8V8, VecV64(x)))
                    x = BP[q0 + 3]; c8V8 |= (rbc(a8V8, Val(2)) & reinterpret(Vec8V8, VecV64(x)))
                    x = BP[q0 + 4]; c8V8 |= (rbc(a8V8, Val(3)) & reinterpret(Vec8V8, VecV64(x)))
                    x = BP[q0 + 5]; c8V8 |= (rbc(a8V8, Val(4)) & reinterpret(Vec8V8, VecV64(x)))
                    x = BP[q0 + 6]; c8V8 |= (rbc(a8V8, Val(5)) & reinterpret(Vec8V8, VecV64(x)))
                    x = BP[q0 + 7]; c8V8 |= (rbc(a8V8, Val(6)) & reinterpret(Vec8V8, VecV64(x)))
                    x = BP[q0 + 8]; c8V8 |= (rbc(a8V8, Val(7)) & reinterpret(Vec8V8, VecV64(x)))
                end

                if s isa RelPlus
                    cV64 = ~reinterpret(VecV64, c8V8)
                else
                    cV64 =  reinterpret(VecV64, c8V8)
                end

                vstore(cV64, cp)
            end

            for i in mf + 1:m
                Cij = C[i, j]

                for p in 1:k
                    Cij = smuladd(s, A[i, p], B[p, j], Cij)
                end

                C[i, j] = Cij
            end
        end
    end

    return C
end
