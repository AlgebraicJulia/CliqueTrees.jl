# ===== kernel v7: CUTLASS SIMT structure =====
#
# The structure of CUTLASS's SIMT SGEMM (and of cuASR and the archived C++ kernel), written against raw
# pointers so that the address arithmetic is what a CUDA C++ kernel would emit:
#
#   - warp tile 8LM × LN·TN: lanes LM (M) × LN = 32/LM (N), each with an 8 × TN register tile (TN = 8 or 4)
#     made of 2 × TN/4 groups of 4 × 4, so every fragment load is one conflict-free LDS.128 (A: rows
#     lm·4 + 0:3 and 4LM + lm·4 + 0:3 of the warp tile; B: columns ln·4 + 0:3 and, for TN = 8, 4LN + ln·4 + 0:3);
#     the block is BM/8LM × BN/(LN·TN) warps. LM = 4 (warp 32 × 8TN) is CUTLASS's lane grid; LM = 8 (warp
#     64 × 4TN, 64 × 16 for TN = 4) makes tiles of any multiple of 16 columns, so that a front of nn columns
#     needs little padding (one tile cld(nn, 16)·16 wide);
#   - shared memory As[p][m] (A is m-contiguous in global memory: no transpose) and Bs[p][n] with a pad
#     of 4 (B is k-contiguous: the transposing store is conflict-free), two stages, a register-staged
#     prefetch of the next panel and one barrier per panel;
#   - global loads as 16-byte vectors when the operands allow (aligned base, leading dimension a
#     multiple of 4, an interior tile), else as checked scalar loads of the same elements;
#   - each thread's global pointers computed once and advanced by a constant per panel; shared-memory
#     offsets in Int32, the stage switch an XOR of a byte offset; the BK steps of a panel fully unrolled
#     with constant offsets, so the main loop is loads, products and a handful of integer instructions;
#   - the epilogue a 16-byte read-⊕-write of C per 4-row group (interior tiles), checked scalars elsewhere;
#   - split-K (blockIdx().z > 1 slices of the k panels) for ⊕ = min or max (atomic_kind): each slice adds
#     its partial tile into C with the atomic ⊕ of the solve (atomic_splus!). min and max are exact and
#     do not depend on the order, so the result is bit-identical to one pass; with overwrite, C is set to
#     the semiring zero first (launch7!).
#
# Generic over the semiring (smuladd, splus, szero, sone): the same operations in the same order per
# entry as sgemx_kernel2! (bit-identical results). Operands must be column-major with unit row stride
# (strided_layout), or for A and C, columns of one picked by an index vector (indexed_layout: the
# solve's C[:, sep], gathered on the fly); others use kernel v2.

using Core: LLVMPtr
using CUDA: AS
const i32 = Int32(1)          # Int32(3) isa Int32

const F4 = NTuple{4, VecElement{Float32}}

@inline v7_ldg4(p::LLVMPtr{Float32, AS.Global}) = unsafe_load(reinterpret(LLVMPtr{F4, AS.Global}, p), 1, Val(16))
@inline v7_stg4!(p::LLVMPtr{Float32, AS.Global}, v::F4) = unsafe_store!(reinterpret(LLVMPtr{F4, AS.Global}, p), v, 1, Val(16))
@inline v7_lds4(p::LLVMPtr{Float32, AS.Shared}) = unsafe_load(reinterpret(LLVMPtr{F4, AS.Shared}, p), 1, Val(16))
@inline v7_sts4!(p::LLVMPtr{Float32, AS.Shared}, v::F4) = unsafe_store!(reinterpret(LLVMPtr{F4, AS.Shared}, p), v, 1, Val(16))
@inline v7_sts!(p::LLVMPtr{Float32, AS.Shared}, x::Float32) = unsafe_store!(p, x, 1, Val(4))
@inline v7_ld(p::LLVMPtr{Float32, AS.Global}) = unsafe_load(p, 1, Val(4))
@inline v7_st!(p::LLVMPtr{Float32, AS.Global}, x::Float32) = unsafe_store!(p, x, 1, Val(4))

@inline f4(x::F4) = (x[1].value, x[2].value, x[3].value, x[4].value)
@inline f4(a, b, c, d) = (VecElement(a), VecElement(b), VecElement(c), VecElement(d))

# (pointer, leading dimension) of a column-major matrix with unit row stride, or nothing
strided_layout(A::StridedCuMatrix) = stride(A, 1) == 1 ? (pointer(A), stride(A, 2)) : nothing
strided_layout(A::Base.ReshapedArray{T, 2, <:Union{CuVector{T}, SubArray{T, 1, <:CuArray, <:Tuple{UnitRange}}}}) where {T} =
    (pointer(parent(A)), size(A, 1))
strided_layout(A) = nothing

# (pointer, leading dimension, column indices) of a matrix whose columns are columns of a column-major
# matrix picked by a device index vector, C[:, idx] (the solve's gathers and scatters), or nothing
const DeviceIndices = Union{CuVector{<:Integer}, SubArray{<:Integer, 1, <:CuVector, <:Tuple{UnitRange}}}
indexed_layout(A::SubArray{T, 2, <:CuMatrix{T}, <:Tuple{Base.Slice, DeviceIndices}}) where {T} =
    (pointer(parent(A)), stride(parent(A), 2), A.indices[2])
indexed_layout(A) = nothing

# strided (indices nothing) or indexed
function gemm_layout(A)
    l = strided_layout(A)
    isnothing(l) || return (l..., nothing)
    return indexed_layout(A)
end

vec_ok(p, ld) = UInt(p) % 16 == 0 && ld % 4 == 0

# warp tile (rows, columns) and threads of kernel v7
v7_warp(LM, TN) = (8 * LM, (32 ÷ LM) * TN)
v7_threads(BM, BN, TN, LM) = (BM ÷ v7_warp(LM, TN)[1]) * (BN ÷ v7_warp(LM, TN)[2]) * 32

# every thread loads whole float4s of the panels, at a fixed row of A and a fixed k offset of B (the last
# float4 of a panel is skipped by the threads past its end when the threads do not divide the panel)
function v7_ok(::Type{V}, BM, BN, BK, TN, LM = 4) where {V}
    V === Float32 && TN in (4, 8) && LM in (4, 8) && BK % 4 == 0 && BK > 0 || return false
    WM, WN = v7_warp(LM, TN)
    BM % WM == 0 && BN % WN == 0 && BM > 0 && BN > 0 || return false
    NT = v7_threads(BM, BN, TN, LM)
    return NT <= 1024 && NT % (BM ÷ 4) == 0 && NT % (BK ÷ 4) == 0 && 8 * BK * (BM + BN + 4) <= 48 * 1024
end

# acc ← acc ⊕ a ⊗ bᵀ for the 8 × TN tile acc[r + 8(c - 1)]
@generated function v7_rank1(s, acc::NTuple{N, V}, a::NTuple{8, V}, b::NTuple{TN, V}) where {N, V, TN}
    terms = [:(smuladd(s, a[$r], b[$c], acc[$(r + 8 * (c - 1))], Val(:N), Val(:N))) for c in 1:TN for r in 1:8]
    return :($(Expr(:meta, :inline)); @inbounds ($(terms...),))
end

# the thread's NA float4 of the A panel (rows i0 + 4·m4 + 0:3, column k0 + p), checked when not `fast`
# (with column indices colsA, column c of A is column colsA[c] of the matrix at pA0); columns from k on
# are the semiring zero. The last float4 only when `live` (the thread is not past the end of the panel).
@generated function v7_fetch_a(pA, lda, rowA, kA, m, k, z, fast, live, pA0, colsA, ::Val{NA}, ::Val{NT}, ::Val{BM}) where {NA, NT, BM}
    # element l: float4 index e = t + l·NT, m4 = e % (BM/4), p = e ÷ (BM/4); pA points at l = 0
    loads = map(0:(NA - 1)) do l
        dm = (l * NT) % (BM ÷ 4) * 4; dp = (l * NT) ÷ (BM ÷ 4)
        @assert dm == 0             # NT is a multiple of BM / 4 (checked by the launcher)
        addr = colsA === Nothing ? :(pA + 4 * $dp * Int(lda)) :
            :((c = kA + Int32($dp); c < k ? pA0 + 4 * (Int(rowA) + (Int(@inbounds colsA[c + Int32(1)]) - 1) * Int(lda)) : pA0))
        load = quote
            p = $addr
            if fast
                f4(v7_ldg4(p))
            else
                r = rowA; c = kA + Int32($dp)
                ok = c < k
                (ok && r < m ? v7_ld(p) : z, ok && r + Int32(1) < m ? v7_ld(p + 4) : z,
                 ok && r + Int32(2) < m ? v7_ld(p + 8) : z, ok && r + Int32(3) < m ? v7_ld(p + 12) : z)
            end
        end
        l == NA - 1 ? :(live ? $load : (z, z, z, z)) : load
    end
    return :($(Expr(:meta, :inline)); ($(loads...),))
end

# the thread's NB float4 of the B panel (rows k0 + 4·k4 + 0:3, column j0 + c); the last only when `live`
@generated function v7_fetch_b(pB, ldb, kB, colB, k, n, u, fast, live, ::Val{NB}, ::Val{NT}, ::Val{BK}) where {NB, NT, BK}
    loads = map(0:(NB - 1)) do l
        dk = (l * NT) % (BK ÷ 4) * 4; dc = (l * NT) ÷ (BK ÷ 4)
        @assert dk == 0
        load = quote
            p = pB + 4 * $dc * Int(ldb)
            if fast
                f4(v7_ldg4(p))
            else
                r = kB; ok = colB + Int32($dc) < n
                (ok && r < k ? v7_ld(p) : u, ok && r + Int32(1) < k ? v7_ld(p + 4) : u,
                 ok && r + Int32(2) < k ? v7_ld(p + 8) : u, ok && r + Int32(3) < k ? v7_ld(p + 12) : u)
            end
        end
        l == NB - 1 ? :(live ? $load : (u, u, u, u)) : load
    end
    return :($(Expr(:meta, :inline)); ($(loads...),))
end

@generated function v7_stash_a!(sA, ra, live, ::Val{NA}, ::Val{NT}, ::Val{BM}) where {NA, NT, BM}
    stores = map(0:(NA - 1)) do l
        st = :(v7_sts4!(sA + $(4 * ((l * NT) ÷ (BM ÷ 4)) * BM), f4(ra[$(l + 1)]...)))
        l == NA - 1 ? :(live && $st) : st
    end
    return :($(Expr(:meta, :inline)); $(stores...); nothing)
end

@generated function v7_stash_b!(sB, rb, live, ::Val{NB}, ::Val{NT}, ::Val{BK}, ::Val{LDB}) where {NB, NT, BK, LDB}
    stores = Expr[]
    for l in 0:(NB - 1), v in 1:4
        dc = (l * NT) ÷ (BK ÷ 4)
        st = :(v7_sts!(sB + $(4 * ((v - 1) * LDB + dc)), rb[$(l + 1)][$v]))
        push!(stores, l == NB - 1 ? :(live && $st) : st)
    end
    return :($(Expr(:meta, :inline)); $(stores...); nothing)
end

# the BK steps of one panel: a loop that loads the fragments of step p + 1 before the products of step p.
# Not unrolled (llvm.loop.unroll.disable): unrolled, LLVM issues all 4·BK fragment loads of the panel
# first (16·BK registers on top of the 64 accumulators: 255 registers and 1 KB of spills at BK = 8);
# rolled, two fragment sets are live and the loop costs a few integer instructions per 128 products.
@inline function v7_frags(ra, rb, p::Int32, ::Val{BM}, ::Val{LDB}, ::Val{TN}, ::Val{LM}) where {BM, LDB, TN, LM}
    oa = p * Int32(4 * BM); ob = p * Int32(4 * LDB)
    a0 = f4(v7_lds4(ra + oa)); a1 = f4(v7_lds4(ra + (oa + Int32(16 * LM))))
    b0 = f4(v7_lds4(rb + ob))
    TN == 4 && return (a0..., a1...), b0
    b1 = f4(v7_lds4(rb + (ob + Int32(16 * (32 ÷ LM)))))
    return (a0..., a1...), (b0..., b1...)
end

@generated function v7_panel(s, acc, ra, rb, ::Val{BK}, ::Val{BM}, ::Val{LDB}, ::Val{TN}, ::Val{LM}) where {BK, BM, LDB, TN, LM}
    return quote
        $(Expr(:meta, :inline))
        a, b = v7_frags(ra, rb, Int32(0), Val($BM), Val($LDB), Val($TN), Val($LM))
        p = Int32(1)

        while p < Int32($BK)
            an, bn = v7_frags(ra, rb, p, Val($BM), Val($LDB), Val($TN), Val($LM))
            acc = v7_rank1(s, acc, a, b)
            a = an; b = bn
            p += Int32(1)
            $(Expr(:loopinfo, (Symbol("llvm.loop.unroll.disable"),)))
        end

        return v7_rank1(s, acc, a, b)
    end
end

# ===== kernel v8: v7 with packed adds and 3-input mins (sm_100, min-plus / max-plus Float32) =====
#
# On sm_100, add.f32x2 (FADD2) adds a pair of registers to a broadcast scalar, and the 3-input min.f32
# (FMNMX3) takes two products at once, so two k-steps of a 2 × 1 piece of the tile,
#
#   acc[r, c] ← min(acc[r, c], a[r] + b[c], a′[r] + b′[c])      (r and r + 1 at once)
#
# cost 2 FADD2 + 2 FMNMX3: one instruction per multiply-add instead of two (FADD + FMNMX). The same
# operations as v2 per entry (min is exact and commutative): bit-identical results. Julia's LLVM
# scalarizes <2 x float> adds and emits 2-input mins, so both are inline PTX.

const F2 = NTuple{2, VecElement{Float32}}

# FADD2 and FMNMX3 are native on sm_100 and sm_103 only (ptxas splits both on sm_110 and sm_12x)
pair_ok(s, ::Type{V}) where {V} = min3_ok(s, V) && device_profile().capability.major == 10

@inline addbc(a::F2, b::Float32) = Base.llvmcall(("""
    define <2 x float> @entry(<2 x float> %a, float %b) #0 {
        %ai = bitcast <2 x float> %a to i64
        %b0 = insertelement <2 x float> undef, float %b, i32 0
        %b1 = insertelement <2 x float> %b0, float %b, i32 1
        %bi = bitcast <2 x float> %b1 to i64
        %r = call i64 asm "add.f32x2 \$0, \$1, \$2;", "=l,l,l"(i64 %ai, i64 %bi)
        %rf = bitcast i64 %r to <2 x float>
        ret <2 x float> %rf
    }
    attributes #0 = { alwaysinline }""", "entry"), F2, Tuple{F2, Float32}, a, b)

# acc ← acc ⊕ a ⊗ bᵀ ⊕ a′ ⊗ b′ᵀ, two k-steps, for the 8 × TN tile acc[r + 8(c - 1)]
@generated function v8_rank2(op, acc::NTuple{N, Float32}, a::NTuple{8, Float32}, b::NTuple{TN, Float32},
        a1::NTuple{8, Float32}, b1::NTuple{TN, Float32}) where {N, TN}
    stmts = Expr[]; terms = Vector{Any}(undef, N)
    for c in 1:TN, r in 1:2:7
        x = Symbol(:x, r, :_, c); y = Symbol(:y, r, :_, c)
        push!(stmts, :($x = addbc((VecElement(a[$r]), VecElement(a[$(r + 1)])), b[$c])))
        push!(stmts, :($y = addbc((VecElement(a1[$r]), VecElement(a1[$(r + 1)])), b1[$c])))
        terms[r + 8 * (c - 1)] = :(pair3(op, acc[$(r + 8 * (c - 1))], $x[1].value, $y[1].value))
        terms[r + 1 + 8 * (c - 1)] = :(pair3(op, acc[$(r + 1 + 8 * (c - 1))], $x[2].value, $y[2].value))
    end
    return :($(Expr(:meta, :inline)); @inbounds begin $(stmts...); ($(terms...),) end)
end

# the BK steps of one panel in pairs, the fragments of the next pair loaded before the products
@generated function v8_panel(op, acc, ra, rb, ::Val{BK}, ::Val{BM}, ::Val{LDB}, ::Val{TN}, ::Val{LM}) where {BK, BM, LDB, TN, LM}
    @assert iseven(BK)
    return quote
        $(Expr(:meta, :inline))
        a, b = v7_frags(ra, rb, Int32(0), Val($BM), Val($LDB), Val($TN), Val($LM))
        a1, b1 = v7_frags(ra, rb, Int32(1), Val($BM), Val($LDB), Val($TN), Val($LM))
        p = Int32(2)

        while p < Int32($BK)
            an, bn = v7_frags(ra, rb, p, Val($BM), Val($LDB), Val($TN), Val($LM))
            an1, bn1 = v7_frags(ra, rb, p + Int32(1), Val($BM), Val($LDB), Val($TN), Val($LM))
            acc = v8_rank2(op, acc, a, b, a1, b1)
            a = an; b = bn; a1 = an1; b1 = bn1
            p += Int32(2)
            $(Expr(:loopinfo, (Symbol("llvm.loop.unroll.disable"),)))
        end

        return v8_rank2(op, acc, a, b, a1, b1)
    end
end

# C ← C ⊕ acc (C ← acc with OW): 16-byte read-modify-write per 4-row group (interior tiles with aligned
# C), or checked scalars. The loads of each half of the tile (4 columns) are issued before its stores:
# a load cannot move above an earlier store to C (the compiler cannot prove that they do not alias), so
# one read-modify-write per entry would serialize N round trips to memory, which bounds small-k GEMMs.
# (with column indices colsC, column j of C is column colsC[j] of the matrix at pC0)
@generated function v7_store!(s, pC, ldc, acc::NTuple{N}, gi, gj, m, n, fast, pC0, colsC, ::Val{OW}, ::Val{LM}) where {N, OW, LM}
    vec = Expr[]; sca = Expr[]
    sym(x...) = Symbol(x...)
    LN = 32 ÷ LM
    # the address of row gi + dr of the thread's column dc
    col(dr, dc) = colsC === Nothing ? :(pC + 4 * ($dr + $dc * Int(ldc))) :
        :(pC0 + 4 * (Int(gi) + $dr + (Int(@inbounds colsC[gj + Int32($(dc + 1))]) - 1) * Int(ldc)))

    for half in 0:(N ÷ 32 - 1)
        cols = (4 * half + 1):(4 * half + 4)
        for c in cols, g in 1:2
            dc = c <= 4 ? c - 1 : 4 * LN + c - 5      # column offset within the thread's tile
            dr = 4 * LM * (g - 1)                      # row group offset
            push!(vec, :($(sym(:p, c, :_, g)) = $(col(dr, dc))))
            OW || push!(vec, :($(sym(:o, c, :_, g)) = f4(v7_ldg4($(sym(:p, c, :_, g))))))
            for v in 0:3
                # addresses are recomputed at the store (cheap): 32 live 64-bit addresses would spill
                push!(sca, :($(sym(:in, c, :_, g, :_, v)) = gi + Int32($(dr + v)) < m && gj + Int32($dc) < n))
                OW || push!(sca, :($(sym(:x, c, :_, g, :_, v)) = $(sym(:in, c, :_, g, :_, v)) ? v7_ld($(col(dr + v, dc))) : acc[1]))
            end
        end
        for c in cols, g in 1:2
            dc = c <= 4 ? c - 1 : 4 * LN + c - 5
            dr = 4 * LM * (g - 1)
            e = 8 * (c - 1) + 4 * (g - 1)              # acc index of the group's first row
            o = sym(:o, c, :_, g)
            val = OW ? :(f4(acc[$(e + 1)], acc[$(e + 2)], acc[$(e + 3)], acc[$(e + 4)])) :
                :(f4(splus(s, acc[$(e + 1)], $o[1], Val(:N)), splus(s, acc[$(e + 2)], $o[2], Val(:N)),
                     splus(s, acc[$(e + 3)], $o[3], Val(:N)), splus(s, acc[$(e + 4)], $o[4], Val(:N))))
            push!(vec, :(v7_stg4!($(sym(:p, c, :_, g)), $val)))
            for v in 0:3
                x = OW ? :(acc[$(e + v + 1)]) : :(splus(s, acc[$(e + v + 1)], $(sym(:x, c, :_, g, :_, v)), Val(:N)))
                push!(sca, :($(sym(:in, c, :_, g, :_, v)) && v7_st!($(col(dr + v, dc)), $x)))
            end
        end
    end

    return quote
        $(Expr(:meta, :inline))
        @inbounds if fast
            $(vec...)
        else
            $(sca...)
        end
        return
    end
end

# split-K needs a native atomic ⊕ (atomic_kind): min or max (exact, so the order of the slices does not
# matter); never + (plus-times would round differently) or a compare-and-swap loop
splitk_ok(s, ::Type{V}) where {V} = V === Float32 && atomic_kind(s, Val(:N), V) isa Union{Val{:min}, Val{:max}}

# C ← C ⊕ acc atomically (one slice of a split-K GEMM): the solve's atomic ⊕ (atomic_splus!) per entry,
# skipped for the semiring zero z (C ⊕ z = C). Entry (gi + i, j) of the matrix at pC0 is entry (i, j) of
# the leading-dimension ldc matrix at pR (the thread's first row), so the indices are small constants.
@generated function v7_red!(s, pR, ldc, acc::NTuple{N}, gi, gj, m, n, z, colsC, ::Val{LM}) where {N, LM}
    LN = 32 ÷ LM
    stmts = Expr[]
    for c in 1:(N ÷ 8)
        dc = c <= 4 ? c - 1 : 4 * LN + c - 5
        j = Symbol(:j, c)
        push!(stmts, colsC === Nothing ? :($j = $(dc + 1)) :
            :($j = gj + Int32($dc) < n ? Int(@inbounds colsC[gj + Int32($(dc + 1))]) : 1))
        for g in 1:2, v in 0:3
            dr = 4 * LM * (g - 1) + v
            e = 8 * (c - 1) + 4 * (g - 1) + v + 1
            push!(stmts, :(gi + Int32($dr) < m && gj + Int32($dc) < n && acc[$e] !== z &&
                atomic_splus!(s, Val(:N), R, $(dr + 1), $j, acc[$e])))
        end
    end

    return quote
        $(Expr(:meta, :inline))
        R = CuDeviceArray{Float32, 2, AS.Global}(pR, (Int(ldc), Int(typemax(Int32))))
        @inbounds begin
            $(stmts...)
        end
        return
    end
end

# C ← z (the m × n matrix at pC0, or its columns colsC): before the slices of an overwriting split-K GEMM
function sgemx_kernel7_fill!(pC0::UInt64, ldc::Int32, m::Int32, n::Int32, colsC, z::Float32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = blockIdx().y

    @inbounds while j <= n && i <= m
        c = colsC === nothing ? Int(j) : Int(colsC[j])
        v7_st!(reinterpret(LLVMPtr{Float32, AS.Global}, pC0) + 4 * (Int(i) - 1 + (c - 1) * Int(ldc)), z)
        j += gridDim().y
    end

    return
end

function sgemx_kernel7!(s::AbstractSemiring, pC0::UInt64, ldc::Int32, pA0::UInt64, lda::Int32, pB0::UInt64, ldb::Int32, m::Int32, n::Int32, k::Int32,
        vecA::Bool, vecB::Bool, vecC::Bool, colsA, colsC, ::Val{BM}, ::Val{BN}, ::Val{BK}, ::Val{TN}, ::Val{OW}, ::Val{PAIR} = Val(false),
        ::Val{LM} = Val(4), ::Val{SPLIT} = Val(false), pps::Int32 = Int32(0)) where {BM, BN, BK, TN, OW, PAIR, LM, SPLIT}
    V = Float32
    WM = 8 * LM; WN = (32 ÷ LM) * TN         # warp tile: lanes LM × 32/LM with 8 × TN register tiles
    NT = (BM ÷ WM) * (BN ÷ WN) * 32
    FA = BM * BK ÷ 4; FB = BK * BN ÷ 4        # float4 of a panel of A, of B
    NA = cld(FA, NT)                          # float4 of A per thread per panel (the last maybe not)
    NB = cld(FB, NT)
    LDB = BN + 4
    ABYTES = Int32(4 * BM * BK)
    BBYTES = Int32(4 * LDB * BK)

    pA = reinterpret(LLVMPtr{V, AS.Global}, pA0)
    pB = reinterpret(LLVMPtr{V, AS.Global}, pB0)
    pC = reinterpret(LLVMPtr{V, AS.Global}, pC0)

    As = CuStaticSharedArray(V, BM * BK * 2)
    Bs = CuStaticSharedArray(V, LDB * BK * 2)
    sAp = pointer(As); sBp = pointer(Bs)

    t = threadIdx().x - Int32(1)
    i0 = (blockIdx().y - Int32(1)) * Int32(BM)          # column tiles vary fastest (see launch!)
    j0 = (blockIdx().x - Int32(1)) * Int32(BN)
    innerA = i0 + Int32(BM) <= m
    innerB = j0 + Int32(BN) <= n
    z = szero(s, V, Val(:N))
    u = sone(s, V, Val(:N))
    liveA = FA % NT == 0 || t + Int32((NA - 1) * NT) < Int32(FA)
    liveB = FB % NT == 0 || t + Int32((NB - 1) * NT) < Int32(FB)

    # the k panels of this block, q0 + 1 : q1: all, or (split-K) those of slice blockIdx().z. Only the last
    # panel of all can be partial, so the bounds checks are against k in every slice.
    q0 = SPLIT ? (blockIdx().z - Int32(1)) * pps : Int32(0)
    q1 = SPLIT ? min(cld(k, Int32(BK)), q0 + pps) : cld(k, Int32(BK))
    kb = q0 * Int32(BK)

    # global: thread t loads float4 rows i0 + 4·m4 of column p (A), rows 4·k4 of column j0 + c (B)
    m4 = t % Int32(BM ÷ 4); pa = t ÷ Int32(BM ÷ 4)
    k4 = t % Int32(BK ÷ 4); cb = t ÷ Int32(BK ÷ 4)
    rowA = i0 + Int32(4) * m4
    colB = j0 + cb
    gA = pA + 4 * (Int(rowA) + Int(kb + pa) * Int(lda))
    gB = pB + 4 * (Int(kb + Int32(4) * k4) + Int(colB) * Int(ldb))
    stepA = 4 * BK * Int(lda)
    stepB = 4 * BK

    # shared: store offsets (bytes) and fragment read offsets
    wA = Int32(4) * (pa * Int32(BM) + Int32(4) * m4)
    wB = Int32(4) * (Int32(4) * k4 * Int32(LDB) + cb)
    warp = t ÷ Int32(32); lane = t % Int32(32)
    wm = warp % Int32(BM ÷ WM); wn = warp ÷ Int32(BM ÷ WM)
    lm = lane % Int32(LM); ln = lane ÷ Int32(LM)
    rA = Int32(4) * (wm * Int32(WM) + lm * Int32(4))
    rB = Int32(4) * (wn * Int32(WN) + ln * Int32(4))

    acc = ntuple(_ -> z, Val(8 * TN))

    @inbounds begin
        full = kb + Int32(BK) <= k
        ra = v7_fetch_a(gA, lda, rowA, kb + pa, m, k, z, vecA & innerA & full, liveA, pA, colsA, Val(NA), Val(NT), Val(BM))
        rb = v7_fetch_b(gB, ldb, kb + Int32(4) * k4, colB, k, n, u, vecB & innerB & full, liveB, Val(NB), Val(NT), Val(BK))
        v7_stash_a!(sAp + wA, ra, liveA, Val(NA), Val(NT), Val(BM))
        v7_stash_b!(sBp + wB, rb, liveB, Val(NB), Val(NT), Val(BK), Val(LDB))
        sync_threads()
        cur = Int32(0)
        q = q0 + Int32(1)

        while q <= q1
            more = q < q1

            if more
                gA += stepA; gB += stepB
                k0 = q * Int32(BK)
                full = k0 + Int32(BK) <= k
                ra = v7_fetch_a(gA, lda, rowA, k0 + pa, m, k, z, vecA & innerA & full, liveA, pA, colsA, Val(NA), Val(NT), Val(BM))
                rb = v7_fetch_b(gB, ldb, k0 + Int32(4) * k4, colB, k, n, u, vecB & innerB & full, liveB, Val(NB), Val(NT), Val(BK))
            end

            acc = if PAIR
                v8_panel(min3_op(s), acc, sAp + (cur * ABYTES + rA), sBp + (cur * BBYTES + rB), Val(BK), Val(BM), Val(LDB), Val(TN), Val(LM))
            else
                v7_panel(s, acc, sAp + (cur * ABYTES + rA), sBp + (cur * BBYTES + rB), Val(BK), Val(BM), Val(LDB), Val(TN), Val(LM))
            end

            if more
                cur ⊻= Int32(1)
                v7_stash_a!(sAp + (cur * ABYTES + wA), ra, liveA, Val(NA), Val(NT), Val(BM))
                v7_stash_b!(sBp + (cur * BBYTES + wB), rb, liveB, Val(NB), Val(NT), Val(BK), Val(LDB))
            end

            sync_threads()
            q += Int32(1)
        end
    end

    gi = i0 + wm * Int32(WM) + lm * Int32(4)
    gj = j0 + wn * Int32(WN) + ln * Int32(4)

    if SPLIT
        v7_red!(s, pC + 4 * (Int(gi) + (colsC === nothing ? Int(gj) * Int(ldc) : 0)), ldc, acc, gi, gj, m, n, z, colsC, Val(LM))
    else
        v7_store!(s, pC + 4 * (Int(gi) + Int(gj) * Int(ldc)), ldc, acc, gi, gj, m, n, vecC & innerA & innerB, pC, colsC, Val(OW), Val(LM))
    end

    return
end

# launch kernel v7 if the operands allow it (Float32, unit row strides; A and C may also pick their
# columns by a device index vector); false otherwise. Of C and B only columns j0 + 1 : j0 + nc. With
# split > 1 the k panels are cut into (up to) that many slices, one block per tile and slice, whose
# partial products are added into C atomically (C set to the semiring zero first with overwrite); only
# for an atomic ⊕ that is min or max (splitk_ok; otherwise one slice), and never in place.
function launch7!(s::AbstractSemiring, C::AbstractMatrix{V}, A::AbstractMatrix{V}, B::AbstractMatrix{V}, ::Val{BM}, ::Val{BN}, ::Val{BK}, ::Val{OW};
        tn::Integer = 8, lm::Integer = 4, pair::Bool = false, maxregs::Integer = 0, split::Integer = 1, j0::Integer = 0,
        nc::Integer = size(C, 2) - j0) where {V, BM, BN, BK, OW}
    v7_ok(V, BM, BN, BK, tn, lm) || return false
    pair && !pair_ok(s, V) && return false
    lc = gemm_layout(C); la = gemm_layout(A); lb = strided_layout(B)
    (isnothing(lc) || isnothing(la) || isnothing(lb)) && return false
    m = size(C, 1); n = nc; k = size(A, 2)
    (pc, ldc, ic), (pa, lda, ia), (pb, ldb) = lc, la, lb

    if j0 > 0
        isnothing(ic) ? (pc += 4 * j0 * ldc) : (ic = view(ic, (j0 + 1):(j0 + nc)))
        pb += 4 * j0 * ldb
    end

    npanel = cld(k, BK)
    pps = cld(npanel, clamp(split, 1, npanel))         # panels per slice
    nz = splitk_ok(s, V) ? cld(npanel, pps) : 1         # slices
    NT = v7_threads(BM, BN, tn, lm)
    # 256-thread blocks are capped so that 2 fit per SM; smaller ones fit 3 or more without a cap
    maxregs = maxregs > 0 ? maxregs : pair ? V8_MAXREGS : NT >= 256 ? V7_MAXREGS : V8_MAXREGS
    # a block must fit the register file: 64K per SM in 4 quarters, each holding every 4th warp
    maxregs = min(maxregs, 16384 ÷ (32 * cld(NT ÷ 32, 4)) ÷ 8 * 8)
    blocks = (cld(n, BN), cld(m, BM), nz)
    blocks[2] <= 65535 || throw(ArgumentError("sgemx_gpu!: $m rows need more than 65535 row tiles of $BM"))

    if nz > 1 && OW                                       # the slices add into C: start from the semiring zero
        @cuda threads = 256 blocks = (cld(m, 256), min(n, 65535)) sgemx_kernel7_fill!(UInt64(UInt(pc)), ldc % Int32, m % Int32, n % Int32, ic, szero(s, V, Val(:N)))
    end

    # raw addresses (a CuPtr would arrive as a generic-space pointer)
    @cuda threads = NT blocks = blocks maxregs = maxregs sgemx_kernel7!(s, UInt64(UInt(pc)), ldc % Int32, UInt64(UInt(pa)), lda % Int32, UInt64(UInt(pb)), ldb % Int32,
        m % Int32, n % Int32, k % Int32, vec_ok(pa, lda), vec_ok(pb, ldb), vec_ok(pc, ldc), ia, ic, Val(BM), Val(BN), Val(BK), Val(Int(tn)), Val(OW), Val(pair),
        Val(Int(lm)), Val(nz > 1), pps % Int32)
    return true
end

# 2 blocks of 256 threads per SM; blocks of fewer threads, and v8 (two k-steps of fragments live), would
# spill at 128
const V7_MAXREGS = 128
const V8_MAXREGS = 168
