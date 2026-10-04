module SemiringGPU

# GPU kernels for CliqueTrees.Multifrontal.Semiring.
#
# sgemx_gpu!(s, C, A, B) computes C ← C ⊕ A ⊗ B in the semiring s, with
# the same semantics as the CPU kernel Semiring.sgemx!. The kernel is
# generic: it only calls smuladd, szero, and sone, so every semiring
# whose operations compile for the GPU works unchanged.

using CUDA
using SparseArrays: SparseMatrixCSC
import SparseArrays
import FileWatching

# A package (`using SemiringGPU`, precompiled), a submodule of CliqueTrees, or loaded by the development
# scripts with include("src/SemiringGPU.jl") (into Main, not precompiled: edits take effect at once).
const PACKAGED = parentmodule(@__MODULE__) !== Main

if nameof(parentmodule(@__MODULE__)) === :CliqueTrees
    using ..CliqueTrees
else
    using CliqueTrees
end

const Semiring = CliqueTrees.Multifrontal.Semiring

using .Semiring: AbstractSemiring, smuladd, splus, szero, sone

export sgemx_gpu!, GPUSLU, rmul_gpu!, sssp_gpu!, SSSPPlan, sgetrf_gpu!, mlu_gpu, FactorPlan, factorize!, closure_gpu!, closure_gpu, precompute_ops!
export DeviceProfile, device_profile, measure!
export MultiGPUSLU, closure_multigpu!
export GPUConfig, config, with_config
export apsp_gpu, apsp_gpu!, apsp_plan, APSPPlan

# ===== device overrides =====
#
# Semiring.vmin(x, y) is ifelse(x < y, x, y) on x86 hosts (a host-side
# @static choice that the GPU compilation inherits), which ptxas compiles
# to FSETP + FSEL. On the GPU, llvm.minnum gives the same result whenever
# the accumulator is not NaN (in particular +∞ + -∞ = NaN is absorbed, as
# upstream intends) and is a single FMNMX instruction. This halves the cost
# of a min-plus multiply-add.
#
CUDA.@device_override @inline Semiring.vmin(x::Float32, y::Float32) = ccall("llvm.minnum.f32", llvmcall, Float32, (Float32, Float32), x, y)
CUDA.@device_override @inline Semiring.vmax(x::Float32, y::Float32) = ccall("llvm.maxnum.f32", llvmcall, Float32, (Float32, Float32), x, y)
CUDA.@device_override @inline Semiring.vmin(x::Float64, y::Float64) = ccall("llvm.minnum.f64", llvmcall, Float64, (Float64, Float64), x, y)
CUDA.@device_override @inline Semiring.vmax(x::Float64, y::Float64) = ccall("llvm.maxnum.f64", llvmcall, Float64, (Float64, Float64), x, y)

include("device.jl")
include("config.jl")

# ===== tiling =====
#
# Each thread block computes a BM × BN tile of C, streaming BK-deep
# panels of A and B through shared memory. Each of its TX × TY threads
# keeps a TM × TN register tile of C; the rows owned by thread (tx, ty)
# are tx, tx + TX, ..., and its columns are ty, ty + TY, ..., so that a
# warp reads consecutive shared-memory words (no bank conflicts) or
# one broadcast word.

struct Tiling{BM, BN, BK, TM, TN} end

const TILING_LARGE = Tiling{128, 128, 8, 8, 8}()   # 256 threads
const TILING_MID = Tiling{128, 64, 8, 8, 4}()      # 256 threads
const TILING_SMALL = Tiling{64, 64, 8, 4, 4}()     # 256 threads
const TILING_N16 = Tiling{256, 16, 8, 16, 1}()     # 256 threads, skinny outputs (n ≤ 16)
const TILING_N32 = Tiling{128, 32, 8, 8, 2}()      # 256 threads, skinny outputs (n ≤ 32)

# overwrite = true computes C ← A ⊗ B instead (C is not read). C may then be A itself (C = A ⊗ B in
# place) when the output is one tile wide (size(C, 2) ≤ BN), since every block reads all of its rows
# of A before it writes them; see inplace_ok. `inplace` says whether C and A may share memory: by
# default Base.mightalias, which callers must override when A reaches C's columns through an index
# view (mightalias cannot tell). A configuration that is not one tile wide, or splits k, then writes
# a scratch output that is copied into C (launch!).
#
# The kernel (version and tiling) is chosen per GPU and shape class by select_gemm (autotuned and
# cached, see below); passing `tiling` forces the tiling (with config().gemm_kernel, or version 2).
function sgemx_gpu!(s::AbstractSemiring, C::AbstractMatrix{V}, A::AbstractMatrix{V}, B::AbstractMatrix{V};
        tiling::Union{Nothing, Tiling} = nothing, overwrite::Bool = false, inplace::Bool = Base.mightalias(C, A)) where {V}
    @assert size(C, 1) == size(A, 1)
    @assert size(C, 2) == size(B, 2)
    @assert size(A, 2) == size(B, 1)

    m = size(C, 1)
    n = size(C, 2)
    k = size(A, 2)

    if m > 0 && n > 0 && k > 0
        cfg = isnothing(tiling) ? select_gemm(s, C, A, B, overwrite, inplace) : GemmConfig(forced_version(s, V), tiling)
        launch!(s, C, A, B, cfg, Val(overwrite); inplace)
    elseif overwrite && m > 0 && n > 0
        fill!(C, szero(s, V, Val(:N)))
    end

    return C
end

# ===== kernel selection =====
#
# Which tile shape is fastest depends on the GPU, not only on the shape: on the closure's GEMMs,
# 128 × 128 tiles are best for large shapes on an L4 (sm_89) but 128 × 64 on an RTX PRO 6000 (sm_120),
# and the paired-step kernel (version 4) wins on B200 (sm_100) but loses on sm_120
# (bench/gemm_shapes.jl). So each (GPU, semiring, element type, overwrite, in place, shape class)
# is tuned once: the candidates run on the real operands into a scratch output, the fastest is kept,
# in memory and in a file in the Julia depot, so later sessions only compile the winners. Without
# tuning (SEMIRINGGPU_TUNE=off, during graph capture, on concurrent streams, or when the scratch
# output does not fit) a heuristic picks the kernel. Every candidate gives bit-identical results.

# A configuration is a kernel version and its tiling; kernels v7 / v8 (gemm_simt.jl) also take
#   - lm: the lanes of a warp along M, 4 (warp tile 32 × 8TN) or 8 (warp tile 64 × 4TN, so tile widths
#     in steps of 16 for TN = 4);
#   - bn ≤ 0: a tile width fitted to the output, -bn tiles of cld(cld(n, -bn), WN)·WN columns (WN the
#     warp tile's width): a front nn columns wide is then padded by less than WN columns;
#   - split > 1: split-K, that many slices of the k panels merged by the atomic ⊕ (min or max only,
#     splitk_ok);
#   - rem = 1: the last n % bn columns by a second launch with a narrow tile.
# In place, the last two (and widths below n) go through a scratch output (launch!).
struct GemmConfig
    version::Int
    bm::Int; bn::Int; bk::Int; tm::Int; tn::Int
    lm::Int; split::Int; rem::Int
end

GemmConfig(v::Integer, bm::Integer, bn::Integer, bk::Integer, tm::Integer, tn::Integer) = GemmConfig(v, bm, bn, bk, tm, tn, 4, 1, 0)
GemmConfig(v::Integer, ::Tiling{BM, BN, BK, TM, TN}) where {BM, BN, BK, TM, TN} = GemmConfig(v, BM, BN, BK, TM, TN)
GemmConfig(c::GemmConfig; split::Integer = c.split, rem::Integer = c.rem) = GemmConfig(c.version, c.bm, c.bn, c.bk, c.tm, c.tn, c.lm, split, rem)
tiling(c::GemmConfig) = Tiling{c.bm, c.bn, c.bk, c.tm, c.tn}()

# a plain tiling (every kernel version can run it), as the entries of the tuning file's 7-field lines
classic(c::GemmConfig) = c.bn > 0 && c.lm == 4 && c.split == 1 && c.rem == 0

# the tile width for an output n columns wide
gemm_bn(c::GemmConfig, n::Integer) = c.bn > 0 ? c.bn : (WN = v7_warp(c.lm, c.tn)[2]; cld(cld(n, -c.bn), WN) * WN)

# can c compute an output n columns wide in place (one tile wide, one slice, one launch)? (else an in-place
# GEMM goes through a scratch output)
inplace_config(c::GemmConfig, n::Integer) = gemm_bn(c, n) >= n && c.split == 1 && c.rem == 0

# a configuration some kernel can run (entries of the tuning file are checked before use)
classic_ok(c::GemmConfig) = all(>(0), (c.bm, c.bn, c.bk, c.tm, c.tn)) &&
    c.bm % c.tm == 0 && c.bn % c.tn == 0 && c.bm * c.bn ÷ (c.tm * c.tn) <= 1024 && c.bm <= 256 && c.bn <= 256 && c.bk <= 32
valid_config(c::GemmConfig) = c.version in (2, 4, 6, 7, 8) && (classic(c) ? classic_ok(c) :
    c.version in (7, 8) && c.tm == 8 && c.tn in (4, 8) && c.lm in (4, 8) && c.bn != 0 && -16 <= c.bn <= 256 &&
    0 < c.bm <= 256 && 0 < c.bk <= 32 && 1 <= c.split <= 64 && c.rem in (0, 1))

# tuned choices are keyed by a hash of the kernels' source files: any change to them invalidates them
const GEMM_TUNE_KEY = "gemm-" * string(hash((read(@__FILE__, String), read(joinpath(@__DIR__, "gemm_simt.jl"), String))); base = 16)
const TUNE_ROUNDS = 5
const GEMM_TABLE = Dict{String, GemmConfig}()
const GEMM_TABLE_LOCK = ReentrantLock()
const GEMM_TABLE_LOADED = Ref(false)

gemm_table_file() = get(ENV, "SEMIRINGGPU_TUNE_FILE", joinpath(first(DEPOT_PATH), "semiringgpu", "gemm_tuning.tsv"))

const BUCKETS = (16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096)
bucket(x::Integer) = (i = findfirst(>=(x), BUCKETS); isnothing(i) ? 8192 : BUCKETS[i])
mbucket(m::Integer) = m <= 2048 ? 2048 : m <= 8192 ? 8192 : m <= 32768 ? 32768 : 131072

# the full name of a type, the same in every session (string(T) leaves out the modules of names that the
# session has imported, so two processes could key the same semiring differently); with parameters, as
# MaxPlus, MaxMin and MaxProd are all DualQuantale{…}
type_name(T) = sprint(show, T; context = :module => Core)

# the shape class: the buckets of m and k; the row tiles (of 128) per SM, which with n decide how many
# waves of blocks a GEMM fills (so whether split-K pays); n to a multiple of 16 up to 256 columns and of
# 64 up to 1024 (the tile padding: a front of 71 columns is not one of 96)
nclass(n::Integer) = n <= 256 ? cld(n, 16) * 16 : n <= 1024 ? cld(n, 64) * 64 : bucket(n)
const RCLASSES = (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16)
rclass(m::Integer, nsm::Integer) = (r = cld(m, 128) / nsm; i = findfirst(>=(r), RCLASSES); isnothing(i) ? "r>16" : "r$(RCLASSES[i])")

# `indexed`: C or A picks its columns by an index vector (tuned apart: the plain-tiling kernels read such
# operands through generic indexing, an index load per entry, 2–3× slower than kernel v7 / v8 there)
function gemm_key(s, ::Type{V}, m, n, k, overwrite, inplace, indexed::Bool = false) where {V}
    p = device_profile()
    key = join((GEMM_TUNE_KEY, p.name, "sm_$(p.capability.major)$(p.capability.minor)", "$(p.nsm)SM", type_name(typeof(s)), V,
        overwrite ? "ow" : "acc", inplace ? "inplace" : "out", mbucket(m), rclass(m, p.nsm), nclass(n), bucket(k)), "|")
    return indexed ? key * "|idx" : key
end

function load_gemm_table!()
    GEMM_TABLE_LOADED[] && return
    GEMM_TABLE_LOADED[] = true
    path = gemm_table_file()
    isfile(path) || return

    try
        for line in eachline(path)                   # skip malformed lines (e.g. a torn write)
            f = split(line, '\t')
            v = length(f) in (7, 10) ? tryparse.(Int, f[2:end]) : nothing
            (isnothing(v) || any(isnothing, v)) && continue
            c = GemmConfig(v...)
            valid_config(c) && (GEMM_TABLE[f[1]] = c)
        end
    catch e
        @warn "SemiringGPU: could not read the GEMM tuning file $path; kernels will be retimed" exception = e maxlog = 1
    end
end

# a line of the tuning file: key, version, bm, bn, bk, tm, tn, and lm, split, rem unless they are the defaults
gemm_fields(key, c::GemmConfig) = classic(c) ? (key, c.version, c.bm, c.bn, c.bk, c.tm, c.tn) :
    (key, c.version, c.bm, c.bn, c.bk, c.tm, c.tn, c.lm, c.split, c.rem)

# appended under a lock file, so that concurrent processes (e.g. cluster jobs sharing a depot) never
# interleave lines; a reader skips any malformed line
function save_gemm_entry(key, c::GemmConfig)
    path = gemm_table_file()

    try
        mkpath(dirname(path))
        FileWatching.Pidfile.mkpidlock(path * ".lock"; stale_age = 60) do
            open(path, "a") do io
                println(io, join(gemm_fields(key, c), '\t'))
            end
        end
    catch e
        @warn "SemiringGPU: could not save GEMM tuning to $path, so later sessions will retime the kernels; " *
            "set SEMIRINGGPU_TUNE_FILE to a writable file" exception = e maxlog = 1
    end
end

function gemm_candidates(n::Integer, inplace::Bool, min3::Bool = false, v7::Bool = false, pair::Bool = false)
    c = if n <= 16
        [GemmConfig(2, TILING_N16), GemmConfig(2, TILING_N32)]
    elseif n <= 32
        [GemmConfig(2, TILING_N32), GemmConfig(4, TILING_N32), GemmConfig(2, TILING_SMALL)]
    elseif n <= 64
        [GemmConfig(2, TILING_SMALL), GemmConfig(2, TILING_MID), GemmConfig(4, TILING_MID), GemmConfig(2, TILING_N32)]
    else
        [GemmConfig(2, TILING_LARGE), GemmConfig(4, TILING_LARGE), GemmConfig(2, TILING_MID), GemmConfig(4, TILING_MID),
         GemmConfig(2, TILING_SMALL), GemmConfig(2, TILING_N32)]
    end

    vs = pair ? (7, 8) : (7,)                    # CUTLASS SIMT structure (Float32, strided operands)
    if v7 && n > 32                              # and with packed adds and 3-input mins (sm_100)
        for v in vs
            append!(c, [GemmConfig(v, 128, 64, 8, 8, 8), GemmConfig(v, 64, 64, 8, 8, 8), GemmConfig(v, 128, 64, 16, 8, 4), GemmConfig(v, 64, 64, 8, 8, 4)])
            n > 64 && append!(c, [GemmConfig(v, 128, 128, 8, 8, 8), GemmConfig(v, 64, 128, 8, 8, 8), GemmConfig(v, 128, 128, 16, 8, 4)])
        end
    end

    if v7                                        # tile widths fitted to n (warps of 64 rows: widths in steps of 16 or 32),
        for (bn, tn) in fitted_widths(n, inplace), bm in (64, 128)         # in v8 where it runs (it always won there)
            f = GemmConfig(pair ? 8 : 7, bm, bn, 8, 8, tn, 8, 1, 0)
            v7_threads(bm, gemm_bn(f, n), tn, 8) <= 512 && push!(c, f)
        end
    end

    if min3 && n > 32                            # v2's layout with an explicit 3-input min (sm_100+)
        push!(c, GemmConfig(6, TILING_MID))
        n > 64 && push!(c, GemmConfig(6, TILING_LARGE))
    end

    inplace && filter!(x -> inplace_config(x, n), c)
    return unique!(c)
end

# (bn, TN) of the fitted widths worth a candidate for an output n columns wide: one tile up to 256
# columns (bn = -1: 8 × 4 thread tiles, widths in steps of 16, and 8 × 8, steps of 32); for wider
# outputs (not in place) the width of 64-256 columns that pads n least, if less than the fixed widths 64
# and 128 do (a fixed width: the shape class holds other n, which a fixed number of tiles would fit badly)
function fitted_widths(n::Integer, inplace::Bool)
    out = Tuple{Int, Int}[]
    if n <= 256
        push!(out, (-1, 4))
        n > 32 && push!(out, (-1, 8))
    elseif !inplace
        fixed = min(cld(n, 64) * 64, cld(n, 128) * 128)
        for (tn, w) in ((4, 16), (8, 32))
            bn = argmin(bn -> (cld(n, bn) * bn, -bn), w * cld(64, w):w:256)
            cld(n, bn) * bn < fixed && push!(out, (bn, tn))
        end
    end
    return out
end

# phase 2 of the tuning, from the fastest kernel v7 / v8 configurations c of phase 1: split-K into 2-8
# slices of at least 2 panels while the blocks stay below 24 per SM (a few waves of the small blocks; min
# and max only: splitk_ok), and the last n % bn columns by a second, narrow launch (in place, both go
# through a scratch output: launch!)
function gemm_refinements(cs, m::Integer, n::Integer, k::Integer, splitk::Bool, nsm::Integer)
    out = GemmConfig[]

    for c in cs
        bn = gemm_bn(c, n); tiles = cld(m, c.bm) * cld(n, bn)

        for S in (2, 3, 4, 6, 8)
            splitk && 2S <= cld(k, c.bk) && tiles * S <= 24nsm && push!(out, GemmConfig(c; split = S))
        end

        c.bn > 0 && n > bn && n % bn != 0 && push!(out, GemmConfig(c; rem = 1))
    end

    return unique!(out)
end

# without tuning: from the shape and the GPU size only. With strided Float32 operands, kernel v7 in
# 128 × 64 tiles (64 × 64 for n ≤ 64) is the fastest or within a few percent of it on the closure's
# shapes on an RTX 5060 Laptop, L4 and RTX PRO 6000 (bench/gemm_shapes.jl), except for small k
# (≤ 64), where the 8 × 4 thread tiles of TILING_MID win. In place (one tile wide) above 64 columns,
# 64-row tiles of the output's width rounded up to 32 (warps of 64 × 32) win on the B200 and RTX PRO
# 6000 (1.0-1.4× the 64 × 128 tiles). (Wider in-place outputs go through a scratch output: launch!.)
function heuristic_gemm(s, ::Type{V}, m::Integer, n::Integer, k::Integer, inplace::Bool, strided::Bool = false) where {V}
    if config().gemm_kernel == 0 && strided && V === Float32 && 32 < n && 64 < k
        inplace && 64 < n <= 256 && return GemmConfig(7, 64, -1, 8, 8, 8, 8, 1, 0)
        return n <= 64 ? GemmConfig(7, 64, 64, 8, 8, 8) : GemmConfig(7, 128, 64, 8, 8, 8)
    end

    v = forced_version(s, V)
    t = if n <= 16
        TILING_N16
    elseif n <= 32
        TILING_N32
    elseif n <= 64                    # a 128-wide tile would be at least half empty (2.2× slower on 27000 × 64 × 64)
        TILING_SMALL
    elseif n <= 256 || k <= 96        # 128 × 64 is best or within 5% on all four test GPUs for these
        TILING_MID
    elseif cld(m, 128) * cld(n, 128) >= 2 * device_profile().nsm
        TILING_LARGE
    else
        TILING_MID
    end

    inplace && (tiling_bn(t) < n) && (t = TILING_LARGE)
    return GemmConfig(v, t)
end

tiling_bn(::Tiling{BM, BN}) where {BM, BN} = BN

# the heuristic's tiling (benchmarks)
choose_tiling(m::Integer, n::Integer, k::Integer = 1024) = tiling(heuristic_gemm(Semiring.MinPlus(), Float32, m, n, k, false))

# the kernel version config().gemm_kernel forces (2 when it is automatic, or when version 6 does not
# apply to this semiring and element type)
function forced_version(s, ::Type{V}) where {V}
    v = config().gemm_kernel

    if v == 6 && !min3_ok(s, V)
        @warn "SemiringGPU: gemm_kernel = 6 needs Float32 min-plus or max-plus on compute capability 10.0+; using kernel 2" maxlog = 1
        return 2
    end

    if v == 7 && V !== Float32
        @warn "SemiringGPU: gemm_kernel = 7 needs Float32; using kernel 2" maxlog = 1
        return 2
    end

    if v == 8 && !pair_ok(s, V)
        @warn "SemiringGPU: gemm_kernel = 8 needs Float32 min-plus or max-plus on compute capability 10.x; using kernel 7" maxlog = 1
        return V === Float32 ? 7 : 2
    end

    return v == 0 ? 2 : v
end

function select_gemm(s::AbstractSemiring, C::AbstractMatrix{V}, A::AbstractMatrix, B::AbstractMatrix, overwrite::Bool, inplace::Bool = Base.mightalias(C, A)) where {V}
    m, n = size(C); k = size(A, 2)
    cf = config()
    v7 = V === Float32 && !isnothing(gemm_layout(C)) && !isnothing(gemm_layout(A)) && !isnothing(strided_layout(B))
    (cf.gemm_tune && cf.gemm_kernel == 0 && TUNING[] && !CUDA.is_capturing()) || return heuristic_gemm(s, V, m, n, k, inplace, v7)
    key = gemm_key(s, V, m, n, k, overwrite, inplace, v7 && !(isnothing(indexed_layout(C)) && isnothing(indexed_layout(A))))

    c = lock(GEMM_TABLE_LOCK) do
        load_gemm_table!()
        get(GEMM_TABLE, key, nothing)
    end

    isnothing(c) || return c
    more = v7 ? cs -> gemm_refinements(cs, m, n, k, splitk_ok(s, V), device_profile().nsm) : nothing
    c = tune_gemm(s, C, A, B, overwrite, gemm_candidates(n, inplace, min3_ok(s, V), v7, v7 && pair_ok(s, V)), more; inplace)
    isnothing(c) && return heuristic_gemm(s, V, m, n, k, inplace, v7)

    lock(GEMM_TABLE_LOCK) do
        GEMM_TABLE[key] = c
        save_gemm_entry(key, c)
    end

    return c
end

# the fastest candidate on these operands, written into a scratch output (C and A are not modified);
# nothing if the scratch output does not fit comfortably. With `more`, a second round times the
# refinements more(best two kernel v7 / v8 configurations) against the fastest of the first. In place,
# a configuration that needs a scratch output of its own is timed with it (launch!). When C picks its
# columns by an index vector, so does the scratch output (its own columns, in order): the same kernels.
function tune_gemm(s, C::AbstractMatrix{V}, A, B, overwrite::Bool, cands, more = nothing; inplace::Bool = false) where {V}
    m, n = size(C)
    m * n * sizeof(V) <= available_memory() ÷ 4 || return nothing
    W = CuMatrix{V}(undef, m, n)
    overwrite ? fill!(W, szero(s, V, Val(:N))) : (W .= C)
    ic = indexed_layout(C)
    Wc = isnothing(ic) ? W : SubArray(W, (Base.Slice(axes(W, 1)), scratch_indices(ic[3], n)))
    cands, t = time_gemms(s, Wc, A, B, overwrite, cands, inplace)

    if !isnothing(more) && !isempty(cands)
        top = filter(c -> c.version in (7, 8), cands[sortperm(t)])
        extra = more(top[1:min(2, end)])

        if !isempty(extra)
            cands, t = time_gemms(s, Wc, A, B, overwrite, [cands[argmin(t)]; extra], inplace)
        end
    end

    CUDA.unsafe_free!(W)
    return isempty(cands) ? nothing : cands[argmin(t)]
end

# the indices 1:n as a device vector of the type of idx (a vector, or a view of one)
scratch_indices(idx::CuVector{I}, n) where {I} = CuVector{I}(1:n)
scratch_indices(idx::SubArray{I, 1, <:CuVector}, n) where {I} = view(CuVector{I}(1:n), 1:n)

# (the candidates that launch, their best times): round-robin, best of rounds, since clocks drift
# (power caps, boost) and candidates timed one after the other are not comparable
function time_gemms(s, W, A, B, overwrite::Bool, cands, inplace::Bool)
    # compile and warm up; a candidate this GPU cannot launch (resources) is dropped, not fatal
    cands = filter(cands) do c
        try
            launch!(s, W, A, B, c, Val(overwrite); inplace); true
        catch e
            e isa CUDA.CuError || e isa ArgumentError || occursin("exceeds", sprint(showerror, e)) || rethrow()
            @debug "SemiringGPU: GEMM candidate $c cannot launch here" exception = e
            false
        end
    end
    t = fill(Inf, length(cands))
    isempty(cands) && return cands, t

    CUDA.synchronize()

    for _ in 1:TUNE_ROUNDS, (i, c) in enumerate(cands)
        t[i] = min(t[i], CUDA.@elapsed(launch!(s, W, A, B, c, Val(overwrite); inplace)))
    end

    return cands, t
end

# can C = A ⊗ B be computed in place (C === A) for an output n columns wide? (some kernel is one tile wide)
inplace_ok(n::Integer) = n <= 128

# C ← C ⊕ A ⊗ B (C ← A ⊗ B with OW) by the configuration c. In place (C among A's columns), a
# configuration that is not one tile wide, or splits k, writes a scratch output that is then copied into
# C: its blocks would otherwise overwrite columns of A that other blocks have yet to read.
function launch!(s::AbstractSemiring, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, c::GemmConfig, ow::Val{OW} = Val(false);
        inplace::Bool = false) where {OW}
    if inplace && !inplace_config(c, size(C, 2))
        W = CuMatrix{eltype(C)}(undef, size(C))
        OW || copyto!(W, C)
        launch!(s, W, A, B, c, ow)
        copyto!(C, W)
        CUDA.unsafe_free!(W)
        return
    end

    if !classic(c)
        launch_v7!(s, C, A, B, c, ow) && return
        # operands (or a width) kernel v7 cannot take: kernel v2
        n = size(C, 2)
        return launch!(s, C, A, B, GemmConfig(2, n <= 16 ? TILING_N16 : n <= 32 ? TILING_N32 : n <= 64 ? TILING_SMALL : TILING_LARGE), ow; inplace)
    end

    # function barrier: the tiling becomes a type
    return launch!(s, C, A, B, tiling(c), Val(c.version), ow)
end

# a configuration of kernel v7 / v8 with a fitted tile width, split-K or a remainder launch; false if the
# operands do not allow kernel v7
function launch_v7!(s::AbstractSemiring, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, c::GemmConfig, ::Val{OW}) where {OW}
    n = size(C, 2)
    bn = gemm_bn(c, n)
    pair = c.version == 8
    q = c.rem > 0 && n > bn ? n - n % bn : n          # the columns of the main launch

    launch7!(s, C, A, B, Val(c.bm), Val(bn), Val(c.bk), Val(OW); tn = c.tn, lm = c.lm, pair, split = c.split, nc = q) || return false
    q < n || return true
    # the remainder: tiles of 64 rows (twice the blocks of 128: a narrow strip is little work) and 8 × 4
    # thread tiles, as narrow as the last n - q columns allow
    r = n - q
    launch7!(s, C, A, B, Val(64), Val(cld(r, 16) * 16), Val(c.bk), Val(OW); tn = 4, lm = 8, pair, split = c.split, j0 = q, nc = r) ||
        error("sgemx_gpu!: kernel v7 cannot run the remainder of $c")
    return true
end

function launch!(s::AbstractSemiring, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, ::Tiling{BM, BN, BK, TM, TN}, ::Val{VER}, ::Val{OW}) where {BM, BN, BK, TM, TN, VER, OW}
    TX = BM ÷ TM
    TY = BN ÷ TN
    #
    # The column tiles are in blockIdx.x, so consecutive blocks share a row strip of A:
    # each strip streams from DRAM about once while B (small) stays in L2. With the row tiles fastest,
    # A is re-read once per column tile when it does not fit in L2.
    #
    blocks = (cld(size(C, 2), BN), cld(size(C, 1), BM))
    blocks[2] <= 65535 || throw(ArgumentError("sgemx_gpu!: $(size(C, 1)) rows need more than 65535 row tiles of $BM"))
    if VER == 8
        launch7!(s, C, A, B, Val(BM), Val(BN), Val(BK), Val(OW); tn = TN, pair = true) && return
        launch7!(s, C, A, B, Val(BM), Val(BN), Val(BK), Val(OW); tn = TN) && return
        @cuda threads = TX * TY blocks = blocks sgemx_kernel2!(s, C, A, B, Val(BM), Val(BN), Val(BK), Val(TM), Val(TN), Val(OW))
    elseif VER == 7
        launch7!(s, C, A, B, Val(BM), Val(BN), Val(BK), Val(OW); tn = TN) && return     # else (layout, type, tile): kernel v2
        @cuda threads = TX * TY blocks = blocks sgemx_kernel2!(s, C, A, B, Val(BM), Val(BN), Val(BK), Val(TM), Val(TN), Val(OW))
    elseif VER == 6
        @cuda threads = TX * TY blocks = blocks sgemx_kernel6!(s, C, A, B, Val(BM), Val(BN), Val(BK), Val(TM), Val(TN), Val(OW))
    elseif VER == 4
        @cuda threads = TX * TY blocks = blocks sgemx_kernel4!(s, C, A, B, Val(BM), Val(BN), Val(BK), Val(TM), Val(TN), Val(OW))
    else
        @cuda threads = TX * TY blocks = blocks sgemx_kernel2!(s, C, A, B, Val(BM), Val(BN), Val(BK), Val(TM), Val(TN), Val(OW))
    end
    return
end

# ===== kernel v2: pipelined =====
#
# Block tiles of BM × BN with BK-deep panels, with the structure of cuASR /
# CUTLASS's 2-stage SIMT GEMM and TropicalGEMM's thread mapping:
#
#   - two shared-memory buffers: while the block computes on one k-panel,
#     the next panel is fetched from global memory into registers and then
#     stored into the other buffer, so there is one barrier per panel;
#   - each thread owns groups of VW = 4 contiguous rows (columns) of its
#     register tile, so its shared-memory reads are contiguous 4-vectors;
#   - Bs is stored n-contiguous (Bs[j, p]) and padded by 4 to avoid bank
#     conflicts on the transposing store (cuASR pads transposed layouts).
#
# Row r ∈ 1:TM of the register tile is row g VW TX + VW tx + v of the
# block tile, with g = (r - 1) ÷ VW and v = (r - 1) % VW (VW = 1 gives the
# strided mapping).

const GEMM_UNROLL = 2               # LLVM unroll count of the panel loop of sgemx_kernel4!
const BPAD = 4

function sgemx_kernel2!(s::AbstractSemiring, C::AbstractMatrix{V}, A::AbstractMatrix{V}, B::AbstractMatrix{V},
        ::Val{BM}, ::Val{BN}, ::Val{BK}, ::Val{TM}, ::Val{TN}, ::Val{OW} = Val(false)) where {V, BM, BN, BK, TM, TN, OW}
    TX = BM ÷ TM
    TY = BN ÷ TN
    NT = TX * TY
    VWM = TM % 4 == 0 ? 4 : 1
    VWN = TN % 4 == 0 ? 4 : 1
    NA = cld(BM * BK, NT)       # A elements each thread prefetches per panel
    NB = cld(BK * BN, NT)

    m = size(C, 1)
    n = size(C, 2)
    k = size(A, 2)

    As = CuStaticSharedArray(V, (BM, BK, 2))
    Bs = CuStaticSharedArray(V, (BN + BPAD, BK, 2))

    t  = threadIdx().x - 1
    tx = t % TX
    ty = t ÷ TX
    i0 = (blockIdx().y - 1) * BM        # column tiles vary fastest (see launch!)
    j0 = (blockIdx().x - 1) * BN

    z = szero(s, V, Val(:N))
    u = sone(s, V, Val(:N))
    acc = ntuple(_ -> z, Val(TM * TN))
    npanel = cld(k, BK)

    @inbounds begin
        ra = fetch_a(A, i0, 0, m, k, t, z, Val(BM), Val(BK), Val(NT), Val(NA))
        rb = fetch_b(B, j0, 0, n, k, t, u, Val(BN), Val(BK), Val(NT), Val(NB))
        stash_a!(As, ra, 1, t, Val(BM), Val(BK), Val(NT), Val(NA))
        stash_b!(Bs, rb, 1, t, Val(BN), Val(BK), Val(NT), Val(NB))
        sync_threads()

        for q in 1:npanel
            cur = isodd(q) ? 1 : 2
            nxt = 3 - cur

            if q < npanel
                k0 = q * BK
                ra = fetch_a(A, i0, k0, m, k, t, z, Val(BM), Val(BK), Val(NT), Val(NA))
                rb = fetch_b(B, j0, k0, n, k, t, u, Val(BN), Val(BK), Val(NT), Val(NB))
            end

            for p in 1:BK
                a = load_frag(As, p, cur, tx, Val(TX), Val(TM), Val(VWM))
                b = load_frag(Bs, p, cur, ty, Val(TY), Val(TN), Val(VWN))
                acc = rank1_update(s, acc, a, b)
            end

            if q < npanel
                stash_a!(As, ra, nxt, t, Val(BM), Val(BK), Val(NT), Val(NA))
                stash_b!(Bs, rb, nxt, t, Val(BN), Val(BK), Val(NT), Val(NB))
            end

            sync_threads()
        end
    end

    store_tile2!(s, C, acc, i0, j0, tx, ty, Val(TX), Val(TY), Val(TM), Val(VWM), Val(VWN), Val(OW))
    return
end

# ===== kernel v4: lean main loop =====
#
# sgemx_kernel2!'s tiles, buffers and thread mapping, with the main loop rebuilt so that it needs
# fewer registers and hides shared-memory latency even at one or two warps per scheduler (which is
# what a 64-accumulator tile leaves on any NVIDIA GPU):
#
#   - all index and bounds arithmetic in Int32 (64-bit compares and their live operands were most of
#     sgemx_kernel2!'s 162 registers);
#   - the BK steps of a panel fully unrolled, the fragments of step p + 1 loaded before the products
#     of step p, so the shared-memory loads overlap the arithmetic instead of stalling it;
#   - bounds checks only on edge tiles: a panel inside the matrices loads without predicates;
#   - steps taken in pairs, acc ← (acc ⊕ a₀b₀) ⊕ a₁b₁, so that ptxas can fuse the two ⊕ = min/max
#     into one 3-input FMNMX3 on sm_100 (bit-identical: the same two operations).
#
# Same semiring operations in the same order per entry as sgemx_kernel2! (bit-identical results).

function sgemx_kernel4!(s::AbstractSemiring, C::AbstractMatrix{V}, A::AbstractMatrix{V}, B::AbstractMatrix{V},
        ::Val{BM}, ::Val{BN}, ::Val{BK}, ::Val{TM}, ::Val{TN}, ::Val{OW} = Val(false)) where {V, BM, BN, BK, TM, TN, OW}
    TX = BM ÷ TM
    TY = BN ÷ TN
    NT = TX * TY
    VWM = TM % 4 == 0 ? 4 : 1
    VWN = TN % 4 == 0 ? 4 : 1
    NA = cld(BM * BK, NT)
    NB = cld(BK * BN, NT)

    m = size(C, 1) % Int32
    n = size(C, 2) % Int32
    k = size(A, 2) % Int32

    As = CuStaticSharedArray(V, (BM, BK, 2))
    Bs = CuStaticSharedArray(V, (BN + BPAD, BK, 2))

    t  = (threadIdx().x - 1) % Int32
    tx = t % Int32(TX)
    ty = t ÷ Int32(TX)
    i0 = (blockIdx().y - 1) % Int32 * Int32(BM)
    j0 = (blockIdx().x - 1) % Int32 * Int32(BN)
    innerA = i0 + Int32(BM) <= m
    innerB = j0 + Int32(BN) <= n

    z = szero(s, V, Val(:N))
    u = sone(s, V, Val(:N))
    acc = ntuple(_ -> z, Val(TM * TN))
    npanel = cld(k, Int32(BK))

    @inbounds begin
        full = Int32(BK) <= k
        ra = fetch_a4(A, i0, Int32(0), m, k, t, z, innerA & full, Val(BM), Val(BK), Val(NT), Val(NA))
        rb = fetch_b4(B, j0, Int32(0), n, k, t, u, innerB & full, Val(BN), Val(BK), Val(NT), Val(NB))
        stash_a!(As, ra, 1, t, Val(BM), Val(BK), Val(NT), Val(NA))
        stash_b!(Bs, rb, 1, t, Val(BN), Val(BK), Val(NT), Val(NB))
        sync_threads()

        q = Int32(1)

        while q <= npanel
            cur = isodd(q) ? 1 : 2
            nxt = 3 - cur

            if q < npanel
                k0 = q * Int32(BK)
                full = k0 + Int32(BK) <= k
                ra = fetch_a4(A, i0, k0, m, k, t, z, innerA & full, Val(BM), Val(BK), Val(NT), Val(NA))
                rb = fetch_b4(B, j0, k0, n, k, t, u, innerB & full, Val(BN), Val(BK), Val(NT), Val(NB))
            end

            acc = panel_update(s, acc, As, Bs, cur, tx, ty, Val(BK), Val(TX), Val(TY), Val(TM), Val(TN), Val(VWM), Val(VWN), Val(GEMM_UNROLL))

            if q < npanel
                stash_a!(As, ra, nxt, t, Val(BM), Val(BK), Val(NT), Val(NA))
                stash_b!(Bs, rb, nxt, t, Val(BN), Val(BK), Val(NT), Val(NB))
            end

            sync_threads()
            q += Int32(1)
        end
    end

    store_tile4!(s, C, acc, i0, j0, tx, ty, m, n, Val(TX), Val(TY), Val(TM), Val(VWM), Val(VWN), Val(OW))
    return
end

# ===== 3-input min / max (sm_100+) =====
#
# PTX's three-input min.f32 / max.f32 (PTX ISA 8.8, sm_100+): one FMNMX3 on B200. Julia's LLVM backend
# only emits the two-input form, which ptxas does not fuse, so kernel v6 issues them explicitly.
#
min3_op(::Semiring.MinPlus) = Val(:min)
min3_op(::Semiring.DualQuantale{Semiring.MinPlus}) = Val(:max)       # MaxPlus
min3_op(s) = nothing
min3_ok(s, ::Type{V}) where {V} = V === Float32 && !isnothing(min3_op(s)) && device_profile().capability >= v"10.0"

@inline min3(a::Float32, b::Float32, c::Float32) = Base.llvmcall(("""
    define float @entry(float %a, float %b, float %c) #0 {
        %r = call float asm "min.f32 \$0, \$1, \$2, \$3;", "=f,f,f,f"(float %a, float %b, float %c)
        ret float %r
    }
    attributes #0 = { alwaysinline }""", "entry"), Float32, Tuple{Float32, Float32, Float32}, a, b, c)
@inline max3(a::Float32, b::Float32, c::Float32) = Base.llvmcall(("""
    define float @entry(float %a, float %b, float %c) #0 {
        %r = call float asm "max.f32 \$0, \$1, \$2, \$3;", "=f,f,f,f"(float %a, float %b, float %c)
        ret float %r
    }
    attributes #0 = { alwaysinline }""", "entry"), Float32, Tuple{Float32, Float32, Float32}, a, b, c)

# ===== kernel v6: v2's layout with an explicit 3-input min (sm_100+, min-plus / max-plus Float32) =====
#
# On sm_100 NVIDIA's compiler turns min(min(acc, a₀ + b₀), a₁ + b₁) into one 3-input min (FMNMX3), so a
# pair of k-steps costs 2 adds and 1 min instead of 2 and 2. Julia's LLVM backend emits 2-input mins,
# which ptxas does not fuse (0 FMNMX3 in the SASS of v2 and v4 on B200), so the 3-input min is issued
# here explicitly. The shared-memory layout and thread mapping are v2's (conflict-free 16-byte loads);
# interior panels load without bounds checks, with 32-bit index arithmetic (as v4).
#
@inline pair3(::Val{:min}, acc::Float32, x::Float32, y::Float32) = min3(acc, x, y)
@inline pair3(::Val{:max}, acc::Float32, x::Float32, y::Float32) = max3(acc, x, y)

@generated function rank2_min3(op, acc::NTuple{N, Float32}, a0::NTuple{TM, Float32}, b0::NTuple{TN, Float32}, a1::NTuple{TM, Float32}, b1::NTuple{TN, Float32}) where {N, TM, TN}
    terms = [:(pair3(op, acc[$(r + (c - 1) * TM)], a0[$r] + b0[$c], a1[$r] + b1[$c])) for c in 1:TN for r in 1:TM]
    return :($(Expr(:meta, :inline)); @inbounds ($(terms...),))
end

function sgemx_kernel6!(s::AbstractSemiring, C::AbstractMatrix{Float32}, A::AbstractMatrix{Float32}, B::AbstractMatrix{Float32},
        ::Val{BM}, ::Val{BN}, ::Val{BK}, ::Val{TM}, ::Val{TN}, ::Val{OW} = Val(false)) where {BM, BN, BK, TM, TN, OW}
    V = Float32
    TX = BM ÷ TM
    TY = BN ÷ TN
    NT = TX * TY
    VWM = TM % 4 == 0 ? 4 : 1
    VWN = TN % 4 == 0 ? 4 : 1
    NA = cld(BM * BK, NT)
    NB = cld(BK * BN, NT)
    op = min3_op(s)

    m = size(C, 1) % Int32
    n = size(C, 2) % Int32
    k = size(A, 2) % Int32

    As = CuStaticSharedArray(V, (BM, BK, 2))
    Bs = CuStaticSharedArray(V, (BN + BPAD, BK, 2))

    t  = (threadIdx().x - 1) % Int32
    tx = t % Int32(TX)
    ty = t ÷ Int32(TX)
    i0 = (blockIdx().y - 1) % Int32 * Int32(BM)
    j0 = (blockIdx().x - 1) % Int32 * Int32(BN)
    innerA = i0 + Int32(BM) <= m
    innerB = j0 + Int32(BN) <= n

    z = szero(s, V, Val(:N))
    u = sone(s, V, Val(:N))
    acc = ntuple(_ -> z, Val(TM * TN))
    npanel = cld(k, Int32(BK))

    @inbounds begin
        full = Int32(BK) <= k
        ra = fetch_a4(A, i0, Int32(0), m, k, t, z, innerA & full, Val(BM), Val(BK), Val(NT), Val(NA))
        rb = fetch_b4(B, j0, Int32(0), n, k, t, u, innerB & full, Val(BN), Val(BK), Val(NT), Val(NB))
        stash_a!(As, ra, 1, t, Val(BM), Val(BK), Val(NT), Val(NA))
        stash_b!(Bs, rb, 1, t, Val(BN), Val(BK), Val(NT), Val(NB))
        sync_threads()

        q = Int32(1)

        while q <= npanel
            cur = isodd(q) ? 1 : 2
            nxt = 3 - cur

            if q < npanel
                k0 = q * Int32(BK)
                full = k0 + Int32(BK) <= k
                ra = fetch_a4(A, i0, k0, m, k, t, z, innerA & full, Val(BM), Val(BK), Val(NT), Val(NA))
                rb = fetch_b4(B, j0, k0, n, k, t, u, innerB & full, Val(BN), Val(BK), Val(NT), Val(NB))
            end

            for p in 1:2:BK
                a0 = load_frag(As, p, cur, tx, Val(TX), Val(TM), Val(VWM))
                b0 = load_frag(Bs, p, cur, ty, Val(TY), Val(TN), Val(VWN))
                a1 = load_frag(As, p + 1, cur, tx, Val(TX), Val(TM), Val(VWM))
                b1 = load_frag(Bs, p + 1, cur, ty, Val(TY), Val(TN), Val(VWN))
                acc = rank2_min3(op, acc, a0, b0, a1, b1)
            end

            if q < npanel
                stash_a!(As, ra, nxt, t, Val(BM), Val(BK), Val(NT), Val(NA))
                stash_b!(Bs, rb, nxt, t, Val(BN), Val(BK), Val(NT), Val(NB))
            end

            sync_threads()
            q += Int32(1)
        end
    end

    store_tile2!(s, C, acc, Int(i0), Int(j0), Int(tx), Int(ty), Val(TX), Val(TY), Val(TM), Val(VWM), Val(VWN), Val(OW))
    return
end

# the A panel elements of thread t (as fetch_a), unchecked when the whole panel is in range
@generated function fetch_a4(A, i0, k0, m, k, t, z, inside, ::Val{BM}, ::Val{BK}, ::Val{NT}, ::Val{NA}) where {BM, BK, NT, NA}
    fast = [quote
        e = t + Int32($(l * NT))
        ($(l * NT + NT <= BM * BK) || e < Int32($(BM * BK))) ? A[i0 + e % Int32($BM) + Int32(1), k0 + e ÷ Int32($BM) + Int32(1)] : z
    end for l in 0:(NA - 1)]
    slow = [quote
        e = t + Int32($(l * NT))
        gi = i0 + e % Int32($BM) + Int32(1); gp = k0 + e ÷ Int32($BM) + Int32(1)
        (e < Int32($(BM * BK)) && gi <= m && gp <= k) ? A[gi, gp] : z
    end for l in 0:(NA - 1)]
    return :($(Expr(:meta, :inline)); @inbounds inside ? ($(fast...),) : ($(slow...),))
end

@generated function fetch_b4(B, j0, k0, n, k, t, u, inside, ::Val{BN}, ::Val{BK}, ::Val{NT}, ::Val{NB}) where {BN, BK, NT, NB}
    fast = [quote
        e = t + Int32($(l * NT))
        ($(l * NT + NT <= BK * BN) || e < Int32($(BK * BN))) ? B[k0 + e % Int32($BK) + Int32(1), j0 + e ÷ Int32($BK) + Int32(1)] : u
    end for l in 0:(NB - 1)]
    slow = [quote
        e = t + Int32($(l * NT))
        gp = k0 + e % Int32($BK) + Int32(1); gj = j0 + e ÷ Int32($BK) + Int32(1)
        (e < Int32($(BK * BN)) && gp <= k && gj <= n) ? B[gp, gj] : u
    end for l in 0:(NB - 1)]
    return :($(Expr(:meta, :inline)); @inbounds inside ? ($(fast...),) : ($(slow...),))
end

# the BK steps of one panel as a loop that loads the fragments of step p + 1 before the products of
# step p (one step of register lookahead: 16 extra registers for an 8 × 8 tile, where a full unroll
# lets the compiler hoist every load of the panel and needs about 128). LLVM unrolls it GEMM_UNROLL
# times; two consecutive steps form the chain acc ← (acc ⊕ a₀b₀) ⊕ a₁b₁ that ptxas fuses into
# FMNMX3 on sm_100.
@generated function panel_update(s, acc::NTuple{N, V}, As, Bs, cur, tx, ty, ::Val{BK}, ::Val{TX}, ::Val{TY}, ::Val{TM}, ::Val{TN}, ::Val{VWM}, ::Val{VWN}, ::Val{U} = Val(2)) where {N, V, BK, TX, TY, TM, TN, VWM, VWN, U}
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            a = load_frag(As, 1, cur, tx, Val($TX), Val($TM), Val($VWM))
            b = load_frag(Bs, 1, cur, ty, Val($TY), Val($TN), Val($VWN))
            p = Int32(1)

            while p < Int32($BK)
                an = load_frag(As, p + Int32(1), cur, tx, Val($TX), Val($TM), Val($VWM))
                bn = load_frag(Bs, p + Int32(1), cur, ty, Val($TY), Val($TN), Val($VWN))
                acc = rank1_update(s, acc, a, b)
                a = an; b = bn
                p += Int32(1)
                $(Expr(:loopinfo, (Symbol("llvm.loop.unroll.count"), U)))
            end

            acc = rank1_update(s, acc, a, b)
        end
        return acc
    end
end

@generated function rank2_update(s::AbstractSemiring, acc::NTuple{N, V}, a0::NTuple{TM, V}, b0::NTuple{TN, V}, a1::NTuple{TM, V}, b1::NTuple{TN, V}) where {N, V, TM, TN}
    terms = Expr[]

    for c in 1:TN, r in 1:TM
        push!(terms, :(smuladd(s, a1[$r], b1[$c], smuladd(s, a0[$r], b0[$c], acc[$(r + (c - 1) * TM)], Val(:N), Val(:N)), Val(:N), Val(:N))))
    end

    return quote
        $(Expr(:meta, :inline))
        @inbounds return ($(terms...),)
    end
end

@generated function store_tile4!(s, C, acc::NTuple{N}, i0, j0, tx, ty, m, n, ::Val{TX}, ::Val{TY}, ::Val{TM}, ::Val{VWM}, ::Val{VWN}, ::Val{OW}) where {N, TX, TY, TM, VWM, VWN, OW}
    stores = Expr[]

    for e in 1:N
        r = (e - 1) % TM + 1
        c = (e - 1) ÷ TM + 1
        ri = (r - 1) ÷ VWM * VWM * TX + (r - 1) % VWM + 1
        cj = (c - 1) ÷ VWN * VWN * TY + (c - 1) % VWN + 1

        push!(stores, quote
            gi = i0 + Int32($ri) + Int32($VWM) * tx
            gj = j0 + Int32($cj) + Int32($VWN) * ty

            if gi <= m && gj <= n
                C[gi, gj] = $(OW ? :(acc[$e]) : :(splus(s, acc[$e], C[gi, gj], Val(:N))))
            end
        end)
    end

    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $(stores...)
        end
        return
    end
end

# the NA elements of the A panel (rows i0+1:i0+BM, columns k0+1:k0+BK) that thread t loads, coalesced along columns
@generated function fetch_a(A, i0, k0, m, k, t, z, ::Val{BM}, ::Val{BK}, ::Val{NT}, ::Val{NA}) where {BM, BK, NT, NA}
    loads = [quote
        e = t + $(l * NT)
        i = e % $BM; p = e ÷ $BM
        gi = i0 + i + 1; gp = k0 + p + 1
        (e < $(BM * BK) && gi <= m && gp <= k) ? A[gi, gp] : z
    end for l in 0:(NA - 1)]
    return :($(Expr(:meta, :inline)); @inbounds ($(loads...),))
end

@generated function fetch_b(B, j0, k0, n, k, t, u, ::Val{BN}, ::Val{BK}, ::Val{NT}, ::Val{NB}) where {BN, BK, NT, NB}
    loads = [quote
        e = t + $(l * NT)
        p = e % $BK; j = e ÷ $BK
        gp = k0 + p + 1; gj = j0 + j + 1
        (e < $(BK * BN) && gp <= k && gj <= n) ? B[gp, gj] : u
    end for l in 0:(NB - 1)]
    return :($(Expr(:meta, :inline)); @inbounds ($(loads...),))
end

@generated function stash_a!(As, ra, buf, t, ::Val{BM}, ::Val{BK}, ::Val{NT}, ::Val{NA}) where {BM, BK, NT, NA}
    stores = [quote
        e = t + $(l * NT)
        if e < $(BM * BK)
            As[e % $BM + 1, e ÷ $BM + 1, buf] = ra[$(l + 1)]
        end
    end for l in 0:(NA - 1)]
    return :($(Expr(:meta, :inline)); @inbounds begin $(stores...) end; nothing)
end

@generated function stash_b!(Bs, rb, buf, t, ::Val{BN}, ::Val{BK}, ::Val{NT}, ::Val{NB}) where {BN, BK, NT, NB}
    stores = [quote
        e = t + $(l * NT)
        if e < $(BK * BN)
            Bs[e ÷ $BK + 1, e % $BK + 1, buf] = rb[$(l + 1)]
        end
    end for l in 0:(NB - 1)]
    return :($(Expr(:meta, :inline)); @inbounds begin $(stores...) end; nothing)
end

# register fragment: entries r = 1:TM of row (or column) p of the panel, VW-contiguous groups
@generated function load_frag(S, p, buf, tx, ::Val{TX}, ::Val{TM}, ::Val{VW}) where {TX, TM, VW}
    loads = [:(S[$((r - 1) ÷ VW * VW * TX + (r - 1) % VW + 1) + $VW * tx, p, buf]) for r in 1:TM]
    return :($(Expr(:meta, :inline)); @inbounds ($(loads...),))
end

# C ← acc ⊕ C (or C ← acc with OW) for the thread's register tile. For tiles of at most 32 values
# all loads of C come first, then all stores: written as one load-⊕-store per entry, the loads cannot
# move above earlier stores (the compiler cannot prove that they do not alias), so each load waits for
# the previous store and a small-k GEMM is bound by this read-modify-write. For 8 × 8 tiles the 64
# extra registers would make the main loop spill, and those tiles are compute-bound anyway.
@generated function store_tile2!(s, C, acc::NTuple{N}, i0, j0, tx, ty, ::Val{TX}, ::Val{TY}, ::Val{TM}, ::Val{VWM}, ::Val{VWN}, ::Val{OW} = Val(false)) where {N, TX, TY, TM, VWM, VWN, OW}
    pos = map(1:N) do e
        r = (e - 1) % TM + 1
        c = (e - 1) ÷ TM + 1
        ri = (r - 1) ÷ VWM * VWM * TX + (r - 1) % VWM + 1
        cj = (c - 1) ÷ VWN * VWN * TY + (c - 1) % VWN + 1
        (ri, cj)
    end
    idx(e) = quote
        $(Symbol(:gi, e)) = i0 + $(pos[e][1]) + $VWM * tx
        $(Symbol(:gj, e)) = j0 + $(pos[e][2]) + $VWN * ty
        $(Symbol(:in, e)) = $(Symbol(:gi, e)) <= m && $(Symbol(:gj, e)) <= n
    end
    body = Expr[idx(e) for e in 1:N]

    if OW
        for e in 1:N
            push!(body, :($(Symbol(:in, e)) && (C[$(Symbol(:gi, e)), $(Symbol(:gj, e))] = acc[$e])))
        end
    elseif N <= 32
        for e in 1:N
            push!(body, :($(Symbol(:c, e)) = $(Symbol(:in, e)) ? C[$(Symbol(:gi, e)), $(Symbol(:gj, e))] : acc[$e]))
        end
        for e in 1:N
            push!(body, :($(Symbol(:in, e)) && (C[$(Symbol(:gi, e)), $(Symbol(:gj, e))] = splus(s, acc[$e], $(Symbol(:c, e)), Val(:N)))))
        end
    else
        for e in 1:N
            push!(body, :($(Symbol(:in, e)) && (C[$(Symbol(:gi, e)), $(Symbol(:gj, e))] = splus(s, acc[$e], C[$(Symbol(:gi, e)), $(Symbol(:gj, e))], Val(:N)))))
        end
    end

    return quote
        $(Expr(:meta, :inline))
        m = size(C, 1); n = size(C, 2)
        @inbounds begin
            $(body...)
        end
        return
    end
end

# acc ← acc ⊕ a ⊗ bᵀ on a TM × TN register tile stored column-major in a
# tuple. A separate function, so that `acc` is never captured by a
# closure that reassigns it (which would box it).
# Unrolled explicitly: an ntuple closure of TM * TN calls is not always
# inlined, and an outlined call per multiply-add is ruinous on the GPU.
@generated function rank1_update(s::AbstractSemiring, acc::NTuple{N, V}, a::NTuple{TM, V}, b::NTuple{TN, V}) where {N, V, TM, TN}
    terms = Expr[]

    for c in 1:TN, r in 1:TM
        push!(terms, :(smuladd(s, a[$r], b[$c], acc[$(r + (c - 1) * TM)], Val(:N), Val(:N))))
    end

    return quote
        $(Expr(:meta, :inline))
        @inbounds return ($(terms...),)
    end
end


include("gemm_simt.jl")
include("amalgamate.jl")
include("sgetrs.jl")
include("layered.jl")
include("sgetrf.jl")
include("multigpu.jl")
include("api.jl")

# Runtime state never outlives a session: a precompiled image must not carry device buffers, device
# profiles or tuning tables from precompilation, and the settings come from this session's environment.
function reset_caches!()
    lock(() -> (empty!(GEMM_TABLE); GEMM_TABLE_LOADED[] = false), GEMM_TABLE_LOCK)
    lock(() -> empty!(PROFILES), PROFILES_LOCK)
    lock(() -> empty!(AMALGAMATIONS), AMALGAMATIONS_LOCK)
    lock(() -> empty!(TRSM_WS), TRSM_WS)
    lock(() -> empty!(GRAPH_SCRATCH), GRAPH_SCRATCH)
    # (dropped, not freed: in a session loaded from a precompiled image the pointer is not ours)
    settle_arena()
    lock(() -> (ARENA.mem = nothing; ARENA.size = 0; ARENA.busy = false; ARENA.off = false), ARENA_LOCK)
    return
end

function __init__()
    init_config!()
    reset_caches!()
end

PACKAGED && include("precompile.jl")

end
