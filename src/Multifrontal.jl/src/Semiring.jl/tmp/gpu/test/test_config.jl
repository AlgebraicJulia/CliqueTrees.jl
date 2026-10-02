# GPUConfig, with_config, settings_from_env, without_tuning and check_semiring; and the closure is
# bit-identical under every setting (min-plus).
#
#   julia --project=. -t auto test/test_config.jl
#
# The defaults under test are the built-in ones: SEMIRINGGPU_* variables (other than the tuning file)
# are removed before the module loads, and the effect of setting them is checked in subprocesses.
for k in collect(keys(ENV))
    startswith(k, "SEMIRINGGPU_") && delete!(ENV, k)
end
const TUNE_FILE = joinpath(mktempdir(), "gemm_tuning.tsv")      # fresh: the closure test checks what is written
ENV["SEMIRINGGPU_TUNE_FILE"] = TUNE_FILE
const SRC = joinpath(@__DIR__, "..", "src", "SemiringGPU.jl")
include(SRC)
include(joinpath(@__DIR__, "graphs.jl"))
using .SemiringGPU
using .SemiringGPU: Semiring, settings_from_env, without_tuning, check_semiring, TUNING
using .Semiring: MinPlus, MaxPlus, MinMax, MaxMin, PlusProd, MinProd, MaxProd, MinPlusLaw, MaxPlusLaw, MaxProdLaw,
    AndOr, OrAnd, AbstractSemiring, ChordalSLU, mlu
using CUDA, SparseArrays, Random, Test

const PROJECT = dirname(Base.active_project())
const FIELDS = fieldnames(GPUConfig)

# run `code` in a fresh process that loads the module with extra environment variables
function load_with_env(env, code = "")
    script = "include($(repr(SRC))); using .SemiringGPU; $code"
    cmd = addenv(`$(Base.julia_cmd()) --project=$PROJECT --startup-file=no --color=no -e $script`, env...)
    out = IOBuffer()
    p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out))
    return p.exitcode, String(take!(out))
end

const KNOWN_DEFAULTS = (gemm_tune = true, gemm_kernel = 0, merge = 128, merge_alpha = 0.5, skip_fill = true, layered_min_rows = 4096,
                        layer_size = 0, factor_merge = 128, fused_front = true, direct_assembly = true)
# optional settings: tested when GPUConfig has them
has(f) = hasfield(GPUConfig, f)

# a valid non-default value of every setting
function other_value(f)
    d = getfield(GPUConfig(), f)
    f == :gemm_kernel && return 4
    d isa Bool && return !d
    d isa Integer && return d + 1
    d isa AbstractFloat && return d / 2
    error("no test value for the setting $f::$(typeof(d)); add one to test_config.jl")
end

# the error inside a failed task (or a nested CompositeException), for @sync blocks
rootcause(e) = e isa CompositeException ? rootcause(first(e.exceptions)) : e isa TaskFailedException ? rootcause(e.task.exception) : e

@testset "test_config" begin

@testset "GPUConfig" begin
    @testset "defaults match the docstring" begin
        doc = string(@doc GPUConfig)
        documented = Dict(Symbol(m[1]) => strip(m[2]) for m in eachmatch(r"`(\w+) = ([^`]+)`", doc) if Symbol(m[1]) in FIELDS)
        d = GPUConfig()

        for f in FIELDS
            @test haskey(documented, f)                     # every setting is documented with its default
            haskey(documented, f) || continue
            T = fieldtype(GPUConfig, f)
            @test getfield(d, f) == parse(T, documented[f])
        end
        @test config() == d                                 # in effect at the top level
        # the values themselves (bench/portable.jl's measured best); settings added later are covered by
        # the docstring check above
        for (f, v) in KNOWN_DEFAULTS
            @test hasfield(GPUConfig, f) && getfield(d, f) == v
        end
    end

    @testset "GPUConfig(c; kw...) overrides" begin
        c = GPUConfig()
        c2 = GPUConfig(c; merge = 1, skip_fill = false, merge_alpha = 0.25)
        @test (c2.merge, c2.skip_fill, c2.merge_alpha) == (1, false, 0.25)
        for f in FIELDS
            f in (:merge, :skip_fill, :merge_alpha) || @test getfield(c2, f) == getfield(c, f)
        end
        @test c == GPUConfig()                              # immutable: c is unchanged
        @test GPUConfig(c2) == c2                           # no keywords: a copy
        @test GPUConfig(c2; merge = 128, skip_fill = true, merge_alpha = 0.5) == c
        c3 = GPUConfig(c; merge = Int32(7))                 # converted to the field type
        @test c3.merge === 7

        @test_throws ArgumentError GPUConfig(c; foo = 1)
        @test_throws ArgumentError GPUConfig(c; merge = 1, Merge = 2)
        err = try GPUConfig(c; nonsense = 1) catch e; e end
        @test occursin("nonsense", sprint(showerror, err))  # names the unknown key
        @test occursin("merge", sprint(showerror, err))     # and lists the valid ones
        @test_throws Exception GPUConfig(c; merge = 1.5)    # not an integer
        @test_throws Exception GPUConfig(c; skip_fill = "no")
    end

    @testset "invalid values are rejected" begin
        bad = [(gemm_kernel = 3,), (gemm_kernel = 1,), (gemm_kernel = 5,), (gemm_kernel = -1,), (gemm_kernel = 8,),
               (merge = 0,), (merge = -2,), (factor_merge = 0,), (merge_alpha = -0.1,), (merge_alpha = NaN,),
               (layered_min_rows = 0,), (layer_size = -1,)]
        has(:layer_slots) && push!(bad, (layer_slots = -1,))
        for kw in bad
            ran = Ref(false)
            @test_throws ArgumentError with_config(() -> (ran[] = true); kw...)
            @test !ran[]                                     # f never runs
            @test_throws ArgumentError SemiringGPU.check(GPUConfig(GPUConfig(); kw...))
            @test config() == GPUConfig()
        end
        # the edges of the valid ranges
        for kw in [(gemm_kernel = 0,), (gemm_kernel = 2,), (gemm_kernel = 4,), (gemm_kernel = 6,), (merge = 1,), (factor_merge = 1,),
                   (merge_alpha = 0.0,), (merge_alpha = 1.0,), (layered_min_rows = 1,), (layer_size = 0,), (layer_size = 1,)]
            @test with_config(() -> config(); kw...) == GPUConfig(GPUConfig(); kw...)
        end
    end
end

@testset "with_config scoping" begin
    d = GPUConfig()

    @testset "nesting and return values" begin
        @test with_config(() -> 42; merge = 1) == 42
        @test with_config(() -> config()) == d               # no settings: unchanged
        with_config(merge = 1) do
            @test config().merge == 1
            @test config().skip_fill
            with_config(skip_fill = false) do
                @test config().merge == 1                    # outer setting kept
                @test !config().skip_fill
                with_config(merge = 8) do                    # inner overrides outer
                    @test config().merge == 8
                    @test !config().skip_fill
                end
                @test config().merge == 1
            end
            @test config().skip_fill
            @test config().merge == 1
        end
        @test config() == d
    end

    @testset "restored after an exception" begin
        @test_throws ErrorException with_config(() -> error("boom"); merge = 1, gemm_kernel = 4)
        @test config() == d
        r = try
            with_config(merge = 2) do
                with_config(merge = 3) do
                    throw(DomainError(config().merge))
                end
            end
        catch e
            e
        end
        @test r isa DomainError && r.val == 3
        @test config() == d
    end

    @testset "tasks started inside inherit the setting" begin
        with_config(merge = 1, layer_size = 5) do
            @test fetch(Threads.@spawn config().merge) == 1
            @test fetch(@async config().layer_size) == 5
            @test fetch(Threads.@spawn fetch(Threads.@spawn config().merge)) == 1     # grandchildren too
        end
        # a task keeps the scope it was started in, even when it runs after the scope has exited
        go = Channel{Nothing}(1)
        t = with_config(merge = 9) do
            Threads.@spawn (take!(go); config().merge)
        end
        @test config().merge == 128
        put!(go, nothing)
        @test fetch(t) == 9
    end

    @testset "tasks outside the scope do not see it" begin
        go = Channel{Nothing}(1); seen = Channel{Int}(1)
        outside = Threads.@spawn (take!(go); put!(seen, config().merge); config().skip_fill)
        with_config(merge = 1, skip_fill = false) do
            put!(go, nothing)                                # the outside task reads while this scope is active
            @test take!(seen) == 128
            @test config().merge == 1
        end
        @test fetch(outside) == true

        # concurrent tasks, each in its own scope, never see each other's settings
        nt = 16
        ok = fill(false, nt)
        @sync for i in 1:nt
            Threads.@spawn with_config(merge = i, layer_size = 100 + i) do
                good = true
                for _ in 1:200
                    good &= config().merge == i && config().layer_size == 100 + i
                    rand() < 0.3 && yield()
                end
                ok[i] = good
            end
        end
        @test all(ok)
        @test config() == d

        # a with_config inside a task does not leak to its parent
        fetch(Threads.@spawn with_config(() -> nothing; merge = 3))
        @test config() == d
    end
end

@testset "settings_from_env" begin
    @test settings_from_env(Dict{String, String}()) == (;)
    @test settings_from_env(Dict("PATH" => "/bin", "SEMIRINGGPU_TUNE_FILE" => "/x", "SEMIRINGGPU_FOO" => "1",
                                 "SEMIRINGGPU_merge" => "1", "MERGE" => "1")) == (;)          # unrelated keys ignored

    kw = settings_from_env(Dict("SEMIRINGGPU_MERGE" => "8", "SEMIRINGGPU_SKIP_FILL" => "false", "SEMIRINGGPU_MERGE_ALPHA" => "0.25",
                                "SEMIRINGGPU_GEMM_TUNE" => "0", "SEMIRINGGPU_FUSED_FRONT" => "1", "SEMIRINGGPU_LAYER_SIZE" => "3"))
    @test Dict(pairs(kw)) == Dict(:merge => 8, :skip_fill => false, :merge_alpha => 0.25, :gemm_tune => false, :fused_front => true, :layer_size => 3)
    @test kw.merge isa Int && kw.skip_fill isa Bool && kw.merge_alpha isa Float64 && kw.gemm_tune isa Bool

    # every setting can be set this way (a non-default value of each), and the result is a valid GPUConfig
    vals = Dict(f => other_value(f) for f in FIELDS)
    kw = settings_from_env(Dict("SEMIRINGGPU_" * uppercase(String(f)) => string(v) for (f, v) in vals))
    @test Set(keys(kw)) == Set(FIELDS)
    c = SemiringGPU.check(GPUConfig(GPUConfig(); kw...))
    @test all(getfield(c, f) == vals[f] && typeof(getfield(c, f)) == fieldtype(GPUConfig, f) for f in FIELDS)
    @test all(getfield(c, f) != getfield(GPUConfig(), f) for f in FIELDS)
    @test settings_from_env(Dict("SEMIRINGGPU_MERGE_ALPHA" => "1e-1", "SEMIRINGGPU_SKIP_FILL" => "true")) == (merge_alpha = 0.1, skip_fill = true)

    # garbage is an error, for every field type
    for (k, v) in [("MERGE", "abc"), ("MERGE", "1.5"), ("MERGE", ""), ("GEMM_KERNEL", "two"), ("LAYER_SIZE", "0x"),
                   ("SKIP_FILL", "yes"), ("SKIP_FILL", "2"), ("FUSED_FRONT", ""), ("GEMM_TUNE", "off"),
                   ("MERGE_ALPHA", "half"), ("MERGE_ALPHA", "")]
        @test_throws ArgumentError settings_from_env(Dict("SEMIRINGGPU_" * k => v))
    end
end

@testset "SEMIRINGGPU_* variables when the module loads" begin
    code, out = load_with_env(Dict("SEMIRINGGPU_MERGE" => "1", "SEMIRINGGPU_SKIP_FILL" => "false", "SEMIRINGGPU_MERGE_ALPHA" => "0.25",
                                   "SEMIRINGGPU_LAYERED_MIN_ROWS" => "7", "SEMIRINGGPU_TUNE" => "off"),
        "c = config(); println(\"CONFIG \", join([string(f, '=', getfield(c, f)) for f in fieldnames(GPUConfig)], ' '))")
    @test code == 0
    m = match(r"CONFIG (.*)", out)
    @test m !== nothing
    if m !== nothing
        got = Dict(Symbol(first(split(x, '='))) => last(split(x, '=')) for x in split(m[1]))
        @test got[:merge] == "1" && got[:skip_fill] == "false" && got[:merge_alpha] == "0.25" && got[:layered_min_rows] == "7"
        @test got[:gemm_tune] == "false"                      # SEMIRINGGPU_TUNE=off
        @test got[:gemm_kernel] == "0" && got[:factor_merge] == "128" && got[:fused_front] == "true"     # the rest: defaults
    else
        println(out)
    end

    code, out = load_with_env(Dict("SEMIRINGGPU_GEMM_KERNEL" => "3"))     # parses, but invalid
    @test code != 0
    @test occursin("ArgumentError", out) && occursin("gemm_kernel", out)

    code, out = load_with_env(Dict("SEMIRINGGPU_MERGE" => "lots"))        # does not parse
    @test code != 0
    @test occursin("ArgumentError", out)
end

@testset "without_tuning" begin
    @test TUNING[]
    without_tuning() do
        @test !TUNING[]
        with_config(merge = 1) do                            # independent of the settings
            @test !TUNING[]
        end
        @test fetch(Threads.@spawn TUNING[]) == false        # tasks started inside inherit it
    end
    @test TUNING[]
    @test without_tuning(() -> 7) == 7
    @test_throws ErrorException without_tuning(() -> error("boom"))
    @test TUNING[]
    go = Channel{Nothing}(1)
    outside = Threads.@spawn (take!(go); TUNING[])
    without_tuning() do
        put!(go, nothing)
        @test fetch(outside) == true                         # not visible outside the scope
    end
    @test TUNING[]
end

@testset "check_semiring" begin
    for s in (MinPlus(), MaxPlus()), T in (Int32, Int64)     # the integer infinity overflows under ⊗
        @test_throws ArgumentError check_semiring(s, T)
        msg = try check_semiring(s, T); "" catch e; sprint(showerror, e) end
        @test occursin(string(T), msg) && occursin("Float32", msg)                # says why and what to use
    end
    @test check_semiring(MaxMin(), Int32) === nothing        # documented (api.jl): max-min works on Int32

    # every semiring that works on floats is accepted with Float32 and Float64
    floats = [MinPlus(), MaxPlus(), MinMax(), MaxMin(), PlusProd(), MinProd(), MaxProd(), MinPlusLaw(), MaxPlusLaw(), MaxProdLaw()]
    for s in floats, T in (Float32, Float64)
        @test check_semiring(s, T) === nothing
    end

    # every semiring the package exports is either in that list, Boolean (no float operations), or rejected
    # cleanly (MinProdLaw: ∞ ⊗ 0 = NaN under its ⊗); nothing else may happen
    boolean = (AndOr, OrAnd)
    exported = [getfield(Semiring, n) for n in names(Semiring)]
    scalar = filter(x -> x isa Type && x <: AbstractSemiring && isconcretetype(x) && applicable(x) && !occursin("Matrix", string(x)), exported)
    @test length(scalar) >= length(floats)
    for S in scalar
        any(s -> s isa S, floats) && continue
        S in boolean && continue
        for T in (Float32, Float64)
            r = try check_semiring(S(), T); :accepted catch e; e isa ArgumentError ? :rejected : e end
            @test r in (:accepted, :rejected)
            r == :rejected && println("  check_semiring rejects $(S()) with $T: ", try check_semiring(S(), T) catch e; first(split(sprint(showerror, e), '\n')) end)
        end
    end
end

@testset "closure identical under every setting" begin
    Random.seed!(17)
    s = MinPlus()

    # disconnected (unreachable pairs), with a dense block, so that some fronts are large
    function pieces(n, ::Type{T}) where {T}
        I = Int[]; J = Int[]
        for _ in 1:3n
            u = rand(1:n); v = clamp(u + rand(-12:12), 1, n)
            u != v && rand() < 0.7 && (push!(I, u); push!(J, v))
        end
        for i in 1:30, j in (i + 1):30; rand() < 0.5 && (push!(I, i); push!(J, j)); end
        W = T.(rand(0:100, length(I)))
        return sparse(vcat(I, J), vcat(J, I), vcat(W, W), n, n, min)
    end

    function closure(A; kw...)
        with_config(; kw...) do                               # the whole pipeline (bench/portable.jl), small thresholds
            F = ChordalSLU(s, A); copyto!(F, A)
            P = FactorPlan(F; large = 16, graph = false, nstreams = 8); factorize!(P)
            G = GPUSLU(P; large = 64); precompute_ops!(G)
            D = Array(closure_gpu(G)); p = Array(G.rperm)
            H = similar(D); H[p, p] = D                      # original labels
            H
        end
    end

    settings = [(merge = 1,), (merge = 8, merge_alpha = 0.0), (skip_fill = false,), (layered_min_rows = 1,),
                (layered_min_rows = 1, layer_size = 1), (gemm_kernel = 2,), (gemm_kernel = 4,), (gemm_kernel = 6,),
                (factor_merge = 1,), (fused_front = false,), (direct_assembly = false,),
                (merge = 1, skip_fill = false, layered_min_rows = 1, gemm_kernel = 4, factor_merge = 1, fused_front = false, direct_assembly = false)]
    has(:layer_cache) && push!(settings, (layered_min_rows = 1, layer_cache = false))
    has(:layer_slots) && push!(settings, (layered_min_rows = 1, layer_slots = 1), (layered_min_rows = 1, layer_slots = 3, layer_size = 1))

    nlines() = isfile(TUNE_FILE) ? countlines(TUNE_FILE) : 0

    for (i, (name, A)) in enumerate([("grid3 9³", grid3(9, Float32)), ("pieces 700", pieces(700, Float32))])
        before = nlines()
        notune = closure(A; gemm_tune = false)
        i == 1 && @test !isfile(TUNE_FILE)                    # tuning off: the whole pipeline writes nothing
        @test nlines() == before
        ref = closure(A)
        @test isequal(ref, Matrix(mlu(s, A)))                # the defaults against the CPU closure
        @test isequal(notune, ref)
        name == "pieces 700" && @test any(isinf, ref)        # unreachable pairs
        for kw in settings
            H = closure(A; kw...)
            ok = isequal(H, ref)
            ok || println("  $name: $(kw) differs in $(count(.!isequal.(H, ref))) entries")
            @test ok
        end
        i == 1 && println("  tuning file after the default closures: ", nlines(), " entries")
    end
end

end # test_config
