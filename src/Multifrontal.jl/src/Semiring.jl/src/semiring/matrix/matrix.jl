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

@inline function rmuladd4(x::Vec{W, UInt64}, y::Vec{W, UInt64}) where {W}
    x8 = reinterpret(Vec{8W, UInt8}, x)
    y8 = reinterpret(Vec{8W, UInt8}, y)
    t8 = zero(Vec{8W, UInt8})

    @nexprs 4 k -> begin
        t8 = vifelse((y8 & (0x01 << (k - 1))) != 0x00, t8 | rbc(x8, Val(k - 1)), t8)
    end

    return reinterpret(Vec{W, UInt64}, t8)
end

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
include("idembool.jl")
include("qualitative.jl")

const MatrixQuantale = Union{BoolMatrix, QualMatrix, DualBoolMatrix, IdemBoolMatrix}

function strsx_fwd_upd_1!(s::MatrixQuantale, C::AbstractVecOrMat{T}, fsep::AbstractVector{I}, l₂₁::AbstractVector{T}, Rp::I, na::I, nrhs::I, trans::Val, side::Val{SIDE}) where {T, I, SIDE}
    return strsx_fwd_upd_vec_1!(s, C, fsep, l₂₁, Rp, na, nrhs, trans, side)
end

# ===== sgemx =====

function stablesize(::Type{T}, nj::Integer) where {T}
    return cld(32, sizeof(T)) * SGEMX_NR * nj
end

function spool_st(s::MatrixQuantale, ::Type{T}, ni::Integer, nj::Integer, nk::Integer) where {T}
    mr = SGEMX_MV * vecwidth(T)

    nic = min(ni, SGEMX_LEAF)
    njc = min(nj, SGEMX_LEAF)
    nkc = min(nk, SGEMX_LEAF)

    apn = cld(nic, mr) * mr * njc
    bpn = cld(nkc, SGEMX_NR) * SGEMX_NR * njc + stablesize(T, njc)
    cpn = mr * SGEMX_NR

    AP = FVector{T}(undef, apn)
    BP = FVector{T}(undef, bpn)
    CP = FVector{T}(undef, cpn)

    return AP, BP, CP
end

function sgemx2!(s::MatrixQuantale, tA::Val{:N}, tB::Val{:N}, C::AbstractMatrix{T}, A::AbstractMatrix, B::AbstractMatrix, AP::AbstractVector, BP::AbstractVector, CP::AbstractVector, mr::Val{MR} = Val(SGEMX_MV * vecwidth(T))) where {T, MR}
    ni = size(C, 1)
    nk = size(C, 2)
    nj = size(A, 2)

    if ni >= 2MR && length(BP) >= cld(nk, SGEMX_NR) * SGEMX_NR * nj + stablesize(T, nj)
        sgemx2_table!(s, tA, tB, C, A, B, AP, BP, CP, mr)
    else
        sgemx2_impl!(s, tA, tB, C, A, B, AP, BP, CP, mr)
    end

    return C
end

function sgemx2_table!(s::AbstractSemiring, tA::Val{TA}, tB::Val{TB}, C::AbstractMatrix{T}, A::AbstractMatrix, B::AbstractMatrix, AP::AbstractVector, BP::AbstractVector, CP::AbstractVector, mr::Val{MR} = Val(SGEMX_MV * vecwidth(T))) where {T, MR, TA, TB}
    ni = size(C, 1)
    nk = size(C, 2)
    nj = size(A, 2)

    z = szero(s, T, Val(:N))
    Z = sizeof(T)

    direct = nk <= SGEMX_NR
    ie = ni - ni % MR

    if !direct
        sgemx_pack_A!(s, tA, tB, AP, A, ni, nj, z, mr)
    elseif ie < ni
        sgemx_pack_A!(s, tA, tB, AP, view(A, ie + 1:ni, :), ni - ie, nj, z, mr)
    end

    sgemx_pack_B!(s, tA, tB, BP, B, nk, nj, z)

    @preserve A AP BP @inbounds for k0 in 0:SGEMX_NR:nk - 1
        kt = min(SGEMX_NR, nk - k0)
        pT = reinterpret(Ptr{UInt8}, pointer(BP, length(BP) - stablesize(T, nj) + 1))

        for i in 1:SGEMX_NR * nj
            lo, hi = stables(s, BP[k0 * nj + i])
            vstore(lo, pT + 32(i - 1))
            vstore(hi, pT + 32(i - 1) + 16)
        end

        for i0 in 0:MR:ni - 1
            it = min(MR, ni - i0)

            if direct && it == MR
                pA = pointer(A) + i0 * Z; sA = stride(A, 2)
            elseif direct
                pA = pointer(AP); sA = MR
            else
                pA = pointer(AP) + i0 * nj * Z; sA = MR
            end

            if it == MR && kt == SGEMX_NR
                @preserve C sgemx_kern_tables!(s, tA, tB, unsafe_convert(Ptr{T}, C) + (k0 * stride(C, 2) + i0) * Z, stride(C, 2), pA, sA, pT, nj, mr)
            else
                for kp in 1:kt
                    for ip in 1:it
                        CP[(kp - 1) * MR + ip] = C[i0 + ip, k0 + kp]
                    end

                    for ip in it + 1:MR
                        CP[(kp - 1) * MR + ip] = z
                    end
                end

                for kp in kt + 1:SGEMX_NR
                    for ip in 1:MR
                        CP[(kp - 1) * MR + ip] = z
                    end
                end

                @preserve CP begin
                    sgemx_kern_tables!(s, tA, tB, pointer(CP), MR, pA, sA, pT, nj, mr)
                end

                for kp in 1:kt
                    for ip in 1:it
                        C[i0 + ip, k0 + kp] = CP[(kp - 1) * MR + ip]
                    end
                end
            end
        end
    end

    return C
end

@generated function sgemx_kern_tables!(s::AbstractSemiring, tA::Val{:N}, tB::Val{:N}, pC::Ptr{T}, ldC::Int, pA::Ptr{T}, sA::Int, pT::Ptr{UInt8}, nj::Int, ::Val{MR}) where {T, MR}
    W = vecwidth(T)
    MV = MR ÷ W
    NR = SGEMX_NR
    Z = sizeof(T)
    M = 0x0f0f0f0f0f0f0f0f % T

    c(v, k) = Symbol(:c_, v, :_, k)
    il(v) = Symbol(:il_, v)
    ih(v) = Symbol(:ih_, v)

    init = Expr(:block)
    body = Expr(:block)
    term = Expr(:block)

    for k in 1:NR, v in 1:MV
        off = :(($(k - 1) * ldC + $((v - 1) * W)) * $Z)
        push!(init.args, :($(c(v, k)) = vload(Vec{$W, $T}, pC + $off)))
        push!(term.args, :(vstore($(c(v, k)), pC + $off)))
    end

    for v in 1:MV
        push!(body.args, :(a = vload(Vec{$W, $T}, pA + $((v - 1) * W * Z))))
        push!(body.args, :($(il(v)) = reinterpret(Vec{$(Z * W), UInt8}, a & $M)))
        push!(body.args, :($(ih(v)) = reinterpret(Vec{$(Z * W), UInt8}, (a >> 4) & $M)))
    end

    for k in 1:NR
        push!(body.args, :(lo = vload(Vec{16, UInt8}, pT + $(32(k - 1)))))
        push!(body.args, :(hi = vload(Vec{16, UInt8}, pT + $(32(k - 1) + 16))))

        for v in 1:MV
            push!(body.args, :($(c(v, k)) = $(c(v, k)) | reinterpret(Vec{$W, $T}, lookup(lo, $(il(v))) | lookup(hi, $(ih(v))))))
        end
    end

    return quote
        $init

        for _ in 1:nj
            $body
            pA += sA * $Z
            pT += $(32NR)
        end

        $term
        return
    end
end
