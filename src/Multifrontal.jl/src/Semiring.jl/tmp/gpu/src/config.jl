# ===== configuration =====
#
# Every tunable choice of the solver lives in one immutable GPUConfig, held in a ScopedValue: the
# defaults apply everywhere, and `with_config(f; kw...)` changes them only for the code that runs
# inside `f` (and the tasks it starts), so concurrent callers, threads and GPUs never see each
# other's settings. None of these change results beyond rounding of non-idempotent semirings
# (plus-times); for min-plus, max-plus and max-min every setting gives bit-identical output.
#
# The defaults are the measured best (or within a few percent of it) on an RTX 5060 Laptop, L4,
# RTX PRO 6000 and B200 (bench/portable.jl, bench/gemm_shapes.jl).

using Base.ScopedValues: ScopedValue, with

"""
    GPUConfig(; kw...)

Settings of the GPU solver (see `with_config`).

Defaults can be set per process with environment variables `SEMIRINGGPU_<SETTING>` (for example
`SEMIRINGGPU_MERGE=1`), read when the module loads.

Dense GEMM
- `gemm_tune = true`: pick the GEMM kernel per GPU and shape class by timing candidates once, and
  cache the choice on disk (`SEMIRINGGPU_TUNE=off` in the environment turns the default off).
- `gemm_kernel = 0`: 0 lets the solver choose; 2, 4 or 6 forces that kernel version (testing).

Solve and closure
- `merge = 128`, `merge_alpha = 0.5`: merge chains of fronts into fronts of up to `merge` pivots for
  the solve, while the semiring-zero padding stays below `merge_alpha` of the merged block (1 = off).
- `skip_fill = true`: do not fill the n × n result before the solve; rows are zeroed only where needed.
- `layered_min_rows = 4096`: from this many right-hand sides on, the L sweep below the top of the tree
  runs as one layered launch per layer (fewer rows use level-by-level launches).
- `layer_size = 0`: fronts per region of the layered sweep (0: about √(2 · fronts)).
- `layer_cache = true`: the layered sweep keeps each row's recent values in shared memory (slots,
  placed by a host-side plan; layered.jl); false runs the plain layered walk.
- `layer_slots = 0`: slots per row of that cache (0: from the device's shared memory, at most 64).

Numeric factorization
- `factor_merge = 128`: merge chains of fronts at the top of the tree for the GPU factorization (1 = off).
- `fused_front = true`: LU and both triangular closures of a ≤ 64-pivot diagonal block in one kernel.
- `direct_assembly = true`: add children's updates straight into the factor blocks (no front matrix).
"""
Base.@kwdef struct GPUConfig
    gemm_tune::Bool = get(ENV, "SEMIRINGGPU_TUNE", "auto") != "off"
    gemm_kernel::Int = 0
    merge::Int = 128
    merge_alpha::Float64 = 0.5
    skip_fill::Bool = true
    layered_min_rows::Int = 4096
    layer_size::Int = 0
    layer_cache::Bool = true
    layer_slots::Int = 0
    factor_merge::Int = 128
    fused_front::Bool = true
    direct_assembly::Bool = true
end

# a copy of c with some fields replaced
function GPUConfig(c::GPUConfig; kw...)
    for k in keys(kw)
        hasfield(GPUConfig, k) || throw(ArgumentError("GPUConfig has no setting `$k`; settings: $(join(fieldnames(GPUConfig), ", "))"))
    end
    return GPUConfig(; (f => get(kw, f, getfield(c, f)) for f in fieldnames(GPUConfig))...)
end

function check(c::GPUConfig)
    c.gemm_kernel in (0, 2, 4, 6) || throw(ArgumentError("gemm_kernel must be 0 (automatic), 2, 4 or 6, not $(c.gemm_kernel)"))
    c.merge >= 1 && c.factor_merge >= 1 || throw(ArgumentError("merge widths must be at least 1"))
    0 <= c.merge_alpha || throw(ArgumentError("merge_alpha must be nonnegative"))
    c.layered_min_rows >= 1 && c.layer_size >= 0 || throw(ArgumentError("layered_min_rows must be ≥ 1 and layer_size ≥ 0"))
    c.layer_slots >= 0 || throw(ArgumentError("layer_slots must be ≥ 0"))
    return c
end

# process-wide defaults may be set by environment variables SEMIRINGGPU_<SETTING> (e.g.
# SEMIRINGGPU_MERGE=1, SEMIRINGGPU_SKIP_FILL=false), read once when the module loads
function settings_from_env(env = ENV)
    kw = Pair{Symbol, Any}[]

    for f in fieldnames(GPUConfig)
        v = get(env, "SEMIRINGGPU_" * uppercase(String(f)), nothing)
        isnothing(v) && continue
        T = fieldtype(GPUConfig, f)
        push!(kw, f => (T === Bool ? parse(Bool, v) : parse(T, v)))
    end

    return (; kw...)
end

# the process defaults, read from the environment when the module is loaded (init_config!, called by
# __init__: a precompiled module must see the session's environment, not that of precompilation);
# CONFIG holds the settings changed by with_config, or nothing outside any with_config
const DEFAULT_CONFIG = Ref{GPUConfig}()
const CONFIG = ScopedValue{Union{Nothing, GPUConfig}}(nothing)

init_config!() = (DEFAULT_CONFIG[] = check(GPUConfig(GPUConfig(); settings_from_env()...)); nothing)
init_config!()

"The settings in effect (see `with_config`)."
config() = something(CONFIG[], DEFAULT_CONFIG[])

"""
    with_config(f; kw...)

Run `f()` with some settings of `GPUConfig` changed, e.g. `with_config(() -> closure_gpu(G); merge = 1)`.
The change is scoped: other tasks, threads and GPUs keep their own settings.
"""
with_config(f; kw...) = with(f, CONFIG => check(GPUConfig(config(); kw...)))

# internal: false while work is issued on several streams at once, where the GEMM autotuner cannot
# time candidates cleanly (they then use the heuristic choice)
const TUNING = ScopedValue(true)
without_tuning(f) = with(f, TUNING => false)

# internal: a Dict{Symbol, Float64} to time the factorization kernels by phase (synchronizes), or nothing
const FTIMER = ScopedValue{Any}(nothing)

# ===== input checks =====

"""
    check_semiring(s, T)

Throw an `ArgumentError` when the semiring `s` cannot be computed exactly with element type `T` by
the GPU kernels: the semiring zero must stay zero under ⊗ with finite values (else, for example,
an integer infinity overflows when a weight is added to it) and be the identity of ⊕.
"""
function check_semiring(s::AbstractSemiring, ::Type{T}) where {T}
    z = szero(s, T, Val(:N)); o = sone(s, T, Val(:N))
    samples = T <: Integer ? T[0, 1, 2, 100, typemax(T) >> 4] : T[0, 1, 2, 100]

    for w in samples
        x = Semiring.sprod(s, z, w, Val(:N), Val(:N))
        y = Semiring.sprod(s, w, z, Val(:N), Val(:N))

        if !isequal(x, z) || !isequal(y, z)
            throw(ArgumentError("$(nameof(typeof(s))) with $T: zero ⊗ $w = $x is not the semiring zero $z " *
                (T <: Integer ? "(the integer infinity overflows). Use Float32 or Float64 weights, or an integer semiring whose zero absorbs under ⊗." : "")))
        end

        isequal(splus(s, z, w, Val(:N)), w) || throw(ArgumentError("$(nameof(typeof(s))) with $T: zero ⊕ $w ≠ $w"))
    end

    isequal(Semiring.sprod(s, o, T(1), Val(:N), Val(:N)), T(1)) || throw(ArgumentError("$(nameof(typeof(s))) with $T: one ⊗ 1 ≠ 1"))
    return nothing
end
