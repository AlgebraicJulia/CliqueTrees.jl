# Fixed-size bitsets for device code: UInt64 (n <= 64) or Bits{W} (n <= 64W),
# with the same operations, so a kernel written against them compiles for any W.
# Vertices are 1-based; vmin(x) = 0 for the empty set.

struct Bits{W}
    w::NTuple{W, UInt64}
end

@inline Base.:&(a::Bits{W}, b::Bits{W}) where {W} = Bits{W}(ntuple(i -> a.w[i] & b.w[i], Val(W)))
@inline Base.:|(a::Bits{W}, b::Bits{W}) where {W} = Bits{W}(ntuple(i -> a.w[i] | b.w[i], Val(W)))
@inline Base.:~(a::Bits{W}) where {W} = Bits{W}(ntuple(i -> ~a.w[i], Val(W)))
@inline Base.:(==)(a::Bits{W}, b::Bits{W}) where {W} = a.w == b.w
@inline Base.zero(::Type{Bits{W}}) where {W} = Bits{W}(ntuple(_ -> zero(UInt64), Val(W)))
@inline Base.iszero(a::Bits{W}) where {W} = all(iszero, a.w)
@inline Base.isless(a::Bits{W}, b::Bits{W}) where {W} = a.w < b.w

@inline nwords(::Type{UInt64}) = 1
@inline nwords(::Type{Bits{W}}) where {W} = W

@inline popc(x::UInt64) = Int32(count_ones(x))
@inline popc(a::Bits{W}) where {W} = Int32(sum(count_ones, a.w))

@inline vmin(x::UInt64) = x == zero(UInt64) ? Int32(0) : Int32(trailing_zeros(x) + 1)
@inline function vmin(a::Bits{W}) where {W}
    for i in 1:W
        a.w[i] != zero(UInt64) && return Int32(64(i - 1) + trailing_zeros(a.w[i]) + 1)
    end
    return Int32(0)
end

# x without its smallest element
@inline clearmin(x::UInt64) = x & (x - one(UInt64))
@inline function firstword(a::Bits{W}) where {W}
    for i in 1:W
        a.w[i] != zero(UInt64) && return i
    end
    return 0
end
@inline function clearmin(a::Bits{W}) where {W}
    j = firstword(a)
    return Bits{W}(ntuple(i -> i == j ? a.w[i] & (a.w[i] - one(UInt64)) : a.w[i], Val(W)))
end

# NB: a & ~bit(v) can be miscompiled into a rotate with an unmasked amount
# (see GPUOrd README); bit() masks the shift, and clearing uses delbit.
@inline bitof(::Type{UInt64}, v) = one(UInt64) << ((UInt32(v) - one(UInt32)) & 0x3f)
@inline function bitof(::Type{Bits{W}}, v) where {W}
    j = (Int32(v) - Int32(1)) >> 6 + Int32(1); b = one(UInt64) << ((UInt32(v) - one(UInt32)) & 0x3f)
    return Bits{W}(ntuple(i -> i == j ? b : zero(UInt64), Val(W)))
end
@inline hasbit(x::UInt64, v) = (x >> ((UInt32(v) - one(UInt32)) & 0x3f)) & one(UInt64) != zero(UInt64)
@inline function hasbit(a::Bits{W}, v) where {W}
    j = (Int32(v) - Int32(1)) >> 6 + 1
    return (a.w[j] >> ((UInt32(v) - one(UInt32)) & 0x3f)) & one(UInt64) != zero(UInt64)
end

# the first n vertices
@inline maskn(::Type{UInt64}, n) = n >= 64 ? typemax(UInt64) : (one(UInt64) << n) - one(UInt64)
@inline function maskn(::Type{Bits{W}}, n) where {W}
    return Bits{W}(ntuple(Val(W)) do i
        r = n - 64(i - 1)
        r >= 64 ? typemax(UInt64) : r <= 0 ? zero(UInt64) : (one(UInt64) << r) - one(UInt64)
    end)
end

# storage of sets as W consecutive words of an array (1-based slot)
@inline ldset(::Type{UInt64}, a, i) = @inbounds a[i]
@inline ldset(::Type{Bits{W}}, a, i) where {W} = Bits{W}(ntuple(j -> @inbounds(a[(i - 1) * W + j]), Val(W)))
@inline stset!(a, i, x::UInt64) = (@inbounds a[i] = x; nothing)
@inline function stset!(a, i, x::Bits{W}) where {W}
    for j in 1:W
        @inbounds a[(i - 1) * W + j] = x.w[j]
    end
    return nothing
end
# for stack regions addressed by word offset: words off+1 .. off+W
@inline ldw(::Type{UInt64}, st, off) = st[off + 1]
@inline ldw(::Type{Bits{W}}, st, off) where {W} = Bits{W}(ntuple(j -> st[off + j], Val(W)))
@inline stw!(st, off, x::UInt64) = (st[off + 1] = x; nothing)
@inline function stw!(st, off, x::Bits{W}) where {W}
    for j in 1:W
        st[off + j] = x.w[j]
    end
    return nothing
end

# host: a set from vertex list / to vertex list
function toset(::Type{T}, vs) where {T}
    x = zero(T)
    for v in vs
        x = x | bitof(T, v)
    end
    return x
end
function tovec(x)
    out = Int[]
    while !iszero(x)
        push!(out, vmin(x)); x = clearmin(x)
    end
    return out
end
