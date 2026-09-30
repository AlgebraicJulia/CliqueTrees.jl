# ===== byte-table lookups =====
#
# Shared by `BoolMatrix` and `QualMatrix`, which both store one
# row per byte. Right multiplication by a fixed element B acts
# on every row separately, so it is a byte-to-byte map, and it
# splits into two 16-entry table lookups, one per nibble of the
# row.

const LOOKUP_ISA = if X86 && test_cpu_feature(Base.BinaryPlatforms.CPUID.JL_X86_avx512vbmi)
    :vbmi
elseif X86 && test_cpu_feature(Base.BinaryPlatforms.CPUID.JL_X86_avx512bw)
    :avx512
elseif X86 && test_cpu_feature(JL_X86_avx2)
    :avx2
else
    :other
end

#
# The subset-OR table of the four bytes r₀, …, r₃ of w:
#
#   t[x] = ∨ { rₖ : bit k of x }
#
# built eight entries at a time: multiplying a byte by a word
# with ones in bytes x₁ < ⋯ copies it into those bytes.
#
@inline function ortable(w::UInt32)
    r0 = UInt64( w        & 0xff)
    r1 = UInt64((w >> 8)  & 0xff)
    r2 = UInt64((w >> 16) & 0xff)
    r3 = UInt64( w >> 24)

    t = r0 * 0x0100010001000100 |    # x ∈ {1, 3, 5, 7}
        r1 * 0x0101000001010000 |    # x ∈ {2, 3, 6, 7}
        r2 * 0x0101010100000000      # x ∈ {4, 5, 6, 7}

    return reinterpret(Vec{16, UInt8}, Vec{2, UInt64}((t, t | r3 * 0x0101010101010101)))
end

#
# lookup(t, i)[n] = t[i[n]] for a 16-entry table t and indices
# i[n] < 16
#
@inline function lookup(t::Vec{16, UInt8}, i::Vec{16, UInt8})
    return tbl1(t, i)
end

@inline function lookup(t::Vec{16, UInt8}, i::Vec{N, UInt8}) where {N}
    #
    #   vpermb indexes the whole register, so the table need
    #   not be repeated in every 16-byte lane
    #
    @static if LOOKUP_ISA === :vbmi
        if N == 64
            return vpermb512(tpad(t), i)
        end
    end

    @static if LOOKUP_ISA === :vbmi || LOOKUP_ISA === :avx512
        if N == 64
            return pshufb512(trep(t, Val(64)), i)
        end
    end

    @static if LOOKUP_ISA !== :other
        if N == 32
            return pshufb256(trep(t, Val(32)), i)
        end
    end

    return cat2(lookup(t, half(i, Val(0))), lookup(t, half(i, Val(N ÷ 2))))
end

@static if Sys.ARCH === :aarch64
    function tbl1(t::Vec{16, UInt8}, v::Vec{16, UInt8})
        return Vec(ccall("llvm.aarch64.neon.tbl1.v16i8", llvmcall, NTuple{16, VecElement{UInt8}},
            (NTuple{16, VecElement{UInt8}}, NTuple{16, VecElement{UInt8}}), t.data, v.data))
    end
elseif Sys.ARCH === :x86_64
    function tbl1(t::Vec{16, UInt8}, v::Vec{16, UInt8})
        return Vec(ccall("llvm.x86.ssse3.pshuf.b.128", llvmcall, NTuple{16, VecElement{UInt8}},
            (NTuple{16, VecElement{UInt8}}, NTuple{16, VecElement{UInt8}}), t.data, v.data))
    end
else
    function tbl1(t::Vec{16, UInt8}, v::Vec{16, UInt8})
        return Vec{16, UInt8}(ntuple(i -> t[(v[i] & 0x0f) + 1], Val(16)))
    end
end

@inline function vpermb512(t::Vec{64, UInt8}, i::Vec{64, UInt8})
    return Vec(ccall("llvm.x86.avx512.permvar.qi.512", llvmcall, NTuple{64, VecElement{UInt8}},
        (NTuple{64, VecElement{UInt8}}, NTuple{64, VecElement{UInt8}}), t.data, i.data))
end

@inline function pshufb512(t::Vec{64, UInt8}, i::Vec{64, UInt8})
    return Vec(ccall("llvm.x86.avx512.pshuf.b.512", llvmcall, NTuple{64, VecElement{UInt8}},
        (NTuple{64, VecElement{UInt8}}, NTuple{64, VecElement{UInt8}}), t.data, i.data))
end

@inline function pshufb256(t::Vec{32, UInt8}, i::Vec{32, UInt8})
    return Vec(ccall("llvm.x86.avx2.pshuf.b", llvmcall, NTuple{32, VecElement{UInt8}},
        (NTuple{32, VecElement{UInt8}}, NTuple{32, VecElement{UInt8}}), t.data, i.data))
end

#
# place a 16-byte table in the low lane of a 64-byte register
#
@inline function tpad(t::Vec{16, UInt8})
    return shufflevector(t, zero(Vec{16, UInt8}), Val(LOOKUP_PAD))
end

const LOOKUP_PAD = ntuple(i -> i <= 16 ? i - 1 : 16, 64)

#
# repeat a 16-byte table in every 16-byte lane
#
@generated function trep(t::Vec{16, UInt8}, ::Val{N}) where {N}
    return :(shufflevector(t, Val($(ntuple(i -> (i - 1) % 16, N)))))
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

# ===== 4 × 4 matrices with one row per byte =====
#
# `QualMatrix` and `DualBoolMatrix` store a 4 × 4 matrix as two
# 4 × 4 bit planes, one per nibble: bit 8i + j and bit 8i + j + 4.

#
# Transpose both planes in place: they are 4 × 4 blocks in the
# 8 × 8 view of `btr`, and its first two delta-swap rounds
# transpose every 4 × 4 block.
#
@inline function tr4(a)
    b = ((a >> 7)  ⊻ a) & 0x00aa00aa
    a = a ⊻ b ⊻ (b << 7)

    b = ((a >> 14) ⊻ a) & 0x0000cccc
    a = a ⊻ b ⊻ (b << 14)

    return a
end

#
# broadcast byte K of every 4-byte lane to the whole lane
#
@generated function rbc4(v::Vec{W, UInt8}, ::Val{K}) where {W, K}
    function f(i)
        im1 = i - 1
        return (im1 & ~3) + K
    end

    return :(shufflevector(v, Val($(ntuple(f, W)))))
end

# ===== semirings =====

include("relative.jl")
include("dualbool.jl")
include("qualitative.jl")
