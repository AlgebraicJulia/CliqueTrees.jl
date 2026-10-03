# The GEMM autotuner and its on-disk cache (SEMIRINGGPU_TUNE_FILE), and that every GEMM kernel gives the
# same result as kernel version 2.
#
#   julia --project=. -t auto test/test_tuning.jl
#
# What is observable (src/SemiringGPU.jl, kernel selection): the tuning file (one tab-separated line per
# tuned shape class: key, version, bm, bn, bk, tm, tn, appended under `<file>.lock`), and the in-memory
# table SemiringGPU.GEMM_TABLE. gemm_table_file() reads SEMIRINGGPU_TUNE_FILE at call time, but the
# file is read only once per process (at the first tuned GEMM), so reading it back, concurrent writers
# and SEMIRINGGPU_TUNE=off are tested in fresh processes. A retuned shape class always appends a line,
# so "read back without retuning" is "the file did not grow".
for k in collect(keys(ENV))
    startswith(k, "SEMIRINGGPU_") && delete!(ENV, k)
end
const DIR = mktempdir()
const TUNE_FILE = joinpath(DIR, "gemm_tuning.tsv")
ENV["SEMIRINGGPU_TUNE_FILE"] = TUNE_FILE
const SRC = joinpath(@__DIR__, "..", "src", "SemiringGPU.jl")
include(SRC)
using .SemiringGPU
using .SemiringGPU: Semiring, GemmConfig, GEMM_TABLE, GEMM_TUNE_KEY, gemm_key, gemm_candidates, min3_ok, without_tuning
using .Semiring: MinPlus, MaxPlus, MaxMin, PlusProd, szero
using CUDA, Random, Test

const PROJECT = dirname(Base.active_project())
const TILINGS = (SemiringGPU.TILING_LARGE, SemiringGPU.TILING_MID, SemiringGPU.TILING_SMALL, SemiringGPU.TILING_N32, SemiringGPU.TILING_N16)

readlines_or_empty(path) = isfile(path) ? readlines(path) : String[]

# (key, GemmConfig) of a well-formed line, else nothing
function parse_entry(line)
    f = split(line, '\t')
    length(f) == 7 || return nothing
    v = tryparse.(Int, f[2:7])
    any(isnothing, v) && return nothing
    return String(f[1]), GemmConfig(v...)
end

cfgstring(c::GemmConfig) = join((c.version, c.bm, c.bn, c.bk, c.tm, c.tn), ',')
entry_line(key, c::GemmConfig) = join((key, c.version, c.bm, c.bn, c.bk, c.tm, c.tn), '\t')

# a shape class: (m, n, k, overwrite, inplace); in place means C === A (so k == n and overwrite)
shape_arg(sh) = join(Int.(sh), ',')
shape_key(sh) = gemm_key(MinPlus(), Float32, sh[1], sh[2], sh[3], sh[4], sh[5])

# ===== child process =====
#
#   child.jl MODE shape...      MODE = run (GEMMs on the shapes), race (load the table, compile, then wait
#                               for TT_WAIT before tuning; touch TT_READY when ready)
# Prints KEY, GEMM_TUNE, then one line per shape: SHAPE <shape> KEY <key> CFG <config or none> OK <bool>
# (or CRASH <exception>), and SAVED <entries added to the table>.
const CHILD = joinpath(DIR, "child.jl")
write(CHILD, """
const MODE = ARGS[1]
const SHAPES = [Tuple(parse.(Int, split(x, ','))) for x in ARGS[2:end]]
include($(repr(SRC)))
using .SemiringGPU
using .SemiringGPU: Semiring, GEMM_TABLE, gemm_key, gemm_candidates, min3_ok
using CUDA, Random
const s = Semiring.MinPlus()
println("KEY ", SemiringGPU.GEMM_TUNE_KEY)
println("GEMM_TUNE ", config().gemm_tune)
Random.seed!(1)
operand(m, n) = CuArray(Float32.(rand(0:9, m, n)))

function run_shape(m, n, k, ow, inplace)
    ow = ow != 0; inplace = inplace != 0
    A = operand(m, k); B = operand(k, n); C = CuArray(Float32.(rand(20:40, m, n)))
    ref = with_config(gemm_kernel = 2) do                       # version 2, heuristic tiling, never tuned
        Array(sgemx_gpu!(s, inplace ? similar(A, m, n) : copy(C), A, B; overwrite = ow))
    end
    X = inplace ? copy(A) : copy(C)
    inplace ? sgemx_gpu!(s, X, X, B; overwrite = true) : sgemx_gpu!(s, X, A, B; overwrite = ow)
    key = gemm_key(s, Float32, m, n, k, ow, inplace)
    c = get(GEMM_TABLE, key, nothing)
    return key, c, Array(X) == ref
end

lock(SemiringGPU.GEMM_TABLE_LOCK) do; SemiringGPU.load_gemm_table!(); end   # now, so that SAVED counts only new entries
if MODE == "race"
    for (m, n, k, ow, inplace) in SHAPES                        # compile every candidate before the start signal
        A = operand(m, k); B = operand(k, n); W = CUDA.zeros(Float32, m, n)
        for c in gemm_candidates(n, Bool(inplace), min3_ok(s, Float32), true, SemiringGPU.pair_ok(s, Float32))
            SemiringGPU.launch!(s, W, A, B, c, Val(Bool(ow)))
        end
    end
    CUDA.synchronize()
    touch(ENV["TT_READY"])
    t0 = time()
    while !isfile(ENV["TT_WAIT"])
        time() - t0 > 600 && (println("TIMEOUT waiting for the other process"); exit(4))
        sleep(0.01)
    end
end

n0 = length(GEMM_TABLE)
for sh in SHAPES
    try
        key, c, ok = run_shape(sh...)
        println("SHAPE ", join(sh, ','), " KEY ", key, " CFG ", c === nothing ? "none" : join((c.version, c.bm, c.bn, c.bk, c.tm, c.tn), ','), " OK ", ok)
    catch e
        println("SHAPE ", join(sh, ','), " CRASH ", typeof(e), ": ", first(split(sprint(showerror, e), '\\n')))
    end
end
println("SAVED ", length(GEMM_TABLE) - n0)
""")

function child_cmd(mode, shapes; env = Dict{String, String}())
    cmd = `$(Base.julia_cmd()) --project=$PROJECT --startup-file=no --color=no -t 1 $CHILD $mode $(map(shape_arg, shapes))`
    return addenv(cmd, env...)
end

function run_child(mode, shapes; env = Dict{String, String}())
    out = IOBuffer()
    p = run(pipeline(ignorestatus(child_cmd(mode, shapes; env)); stdout = out, stderr = out))
    return parse_child(p.exitcode, String(take!(out)))
end

function parse_child(code, text)
    key = (m = match(r"^KEY (\S+)"m, text); m === nothing ? nothing : String(m[1]))
    tune = (m = match(r"^GEMM_TUNE (\S+)"m, text); m === nothing ? nothing : m[1] == "true")
    saved = (m = match(r"^SAVED (\d+)"m, text); m === nothing ? nothing : parse(Int, m[1]))
    shapes = Dict{String, Any}()
    for m in eachmatch(r"^SHAPE (\S+) KEY (.+) CFG (\S+) OK (\S+)$"m, text)
        shapes[m[1]] = (key = String(m[2]), cfg = String(m[3]), ok = m[4] == "true")
    end
    for m in eachmatch(r"^SHAPE (\S+) CRASH (.*)$"m, text)
        shapes[m[1]] = (crash = String(m[2]),)
    end
    return (; code, text, key, tune, saved, shapes)
end

# the child loaded the same source (the key hashes it); otherwise the file changed during the test
function same_source(r)
    r.key == GEMM_TUNE_KEY && return true
    println("child process saw tuning key $(r.key), this process $GEMM_TUNE_KEY: src/SemiringGPU.jl changed during the test?\n", r.text)
    return false
end

# a check that documents a known bug: broken while the bug is there, a pass once it is fixed
function known_bug(ok::Bool, what)
    ok || println("  KNOWN BUG (test_broken): ", what)
    return ok ? (@test ok) : (@test_broken ok)
end

# shape classes tuned in this process (mbucket(m), bucket(n), bucket(k) differ between them)
const S_ACC = (300, 150, 200, false, false)
const S_OW = (300, 150, 200, true, false)
const S_INPLACE = (3000, 100, 100, true, true)
const S_PLANT = (9000, 300, 40, false, false)        # tuned only in the reading process, from a planted entry
const S_STALE = (300, 40, 600, false, false)         # only stale or malformed lines in the file

Random.seed!(2)
s = MinPlus()
operand(m, n; V = Float32) = CuArray(V.(rand(0:9, m, n)))

@testset "test_tuning" begin

@testset "tuning file location" begin
    @test SemiringGPU.gemm_table_file() == TUNE_FILE
    withenv("SEMIRINGGPU_TUNE_FILE" => joinpath(DIR, "elsewhere.tsv")) do        # read at call time
        @test SemiringGPU.gemm_table_file() == joinpath(DIR, "elsewhere.tsv")
    end
    withenv("SEMIRINGGPU_TUNE_FILE" => nothing) do                                # default: in the depot
        @test SemiringGPU.gemm_table_file() == joinpath(first(DEPOT_PATH), "semiringgpu", "gemm_tuning.tsv")
    end
    @test SemiringGPU.gemm_table_file() == TUNE_FILE
    @test !isfile(TUNE_FILE)
    @test startswith(GEMM_TUNE_KEY, "gemm-")
end

@testset "tuning a shape writes one entry with the current key" begin
    A = operand(300, 200); B = operand(200, 150); C0 = operand(300, 150)
    sgemx_gpu!(s, copy(C0), A, B)
    L = readlines_or_empty(TUNE_FILE)
    @test length(L) == 1
    e = parse_entry(L[1])
    @test e !== nothing
    k1, c1 = e
    @test k1 == shape_key(S_ACC)
    @test startswith(k1, GEMM_TUNE_KEY * "|")
    @test occursin("|MinPlus|Float32|acc|out|", k1)
    @test c1 in gemm_candidates(150, false, min3_ok(s, Float32), true, SemiringGPU.pair_ok(s, Float32))
    @test GEMM_TABLE[k1] == c1
    @test !ispath(TUNE_FILE * ".lock")                # the lock is released

    # the same shape class again (other sizes in the same buckets): answered from memory
    sgemx_gpu!(s, operand(290, 140), operand(290, 230), operand(230, 140))
    @test length(readlines_or_empty(TUNE_FILE)) == 1

    # overwrite and in place are their own classes; in place only tiles one tile wide
    sgemx_gpu!(s, copy(C0), A, B; overwrite = true)
    X = operand(3000, 100); T = operand(100, 100)
    sgemx_gpu!(s, X, X, T; overwrite = true)
    L = readlines_or_empty(TUNE_FILE)
    @test length(L) == 3
    E = Dict(parse_entry(l) for l in L)
    @test Set(keys(E)) == Set(shape_key.((S_ACC, S_OW, S_INPLACE)))
    @test E[shape_key(S_OW)] in gemm_candidates(150, false, min3_ok(s, Float32), true, SemiringGPU.pair_ok(s, Float32))
    @test E[shape_key(S_INPLACE)] in gemm_candidates(100, true, min3_ok(s, Float32), true, SemiringGPU.pair_ok(s, Float32))
    @test E[shape_key(S_INPLACE)].bn >= 100
    @test all(E[k] == GEMM_TABLE[k] for k in keys(E))
end

@testset "no tuning when it is off or does not apply" begin
    n0 = length(readlines_or_empty(TUNE_FILE))
    new_class() = (operand(700, 96), operand(700, 48), operand(48, 96))         # C, A, B of a class not tuned yet
    k_new = gemm_key(s, Float32, 700, 96, 48, false, false)
    checks = [
        "gemm_tune = false" => () -> with_config(() -> sgemx_gpu!(s, new_class()...); gemm_tune = false),
        "gemm_kernel = 2" => () -> with_config(() -> sgemx_gpu!(s, new_class()...); gemm_kernel = 2),
        "gemm_kernel = 4" => () -> with_config(() -> sgemx_gpu!(s, new_class()...); gemm_kernel = 4),
        "gemm_kernel = 6" => () -> with_config(() -> sgemx_gpu!(s, new_class()...); gemm_kernel = 6),
        "without_tuning" => () -> without_tuning(() -> sgemx_gpu!(s, new_class()...)),
        "forced tiling" => () -> sgemx_gpu!(s, new_class()...; tiling = SemiringGPU.TILING_MID),
        "k = 0" => () -> sgemx_gpu!(s, operand(700, 96), operand(700, 0), operand(0, 96)),
        "k = 0, overwrite" => () -> sgemx_gpu!(s, operand(700, 96), operand(700, 0), operand(0, 96); overwrite = true),
        "m = 0" => () -> sgemx_gpu!(s, operand(0, 96), operand(0, 48), operand(48, 96)),
    ]
    for (name, f) in checks
        f(); CUDA.synchronize()
        grew = length(readlines_or_empty(TUNE_FILE)) != n0 || haskey(GEMM_TABLE, k_new)
        grew && println("  tuned with $name")
        @test !grew
    end

    # degenerate shapes keep their semantics: k = 0 adds nothing, or gives the semiring zero with overwrite
    C = operand(70, 30)
    @test Array(sgemx_gpu!(s, copy(C), operand(70, 0), operand(0, 30))) == Array(C)
    @test all(==(Inf32), Array(sgemx_gpu!(s, copy(C), operand(70, 0), operand(0, 30); overwrite = true)))

    # during CUDA graph capture the heuristic kernel is used (compiled first, outside the capture)
    C, A, B = new_class()
    ref = with_config(() -> Array(sgemx_gpu!(s, copy(C), A, B)); gemm_tune = false)
    CUDA.synchronize()
    Cg = copy(C)
    graph = CUDA.capture() do
        sgemx_gpu!(s, Cg, A, B)
    end
    exe = CUDA.instantiate(graph)
    CUDA.launch(exe); CUDA.synchronize()
    @test Array(Cg) == ref
    @test length(readlines_or_empty(TUNE_FILE)) == n0
    @test !haskey(GEMM_TABLE, k_new)

    # and the same class is tuned once tuning applies again
    sgemx_gpu!(s, new_class()...)
    @test length(readlines_or_empty(TUNE_FILE)) == n0 + 1
    @test haskey(GEMM_TABLE, k_new)
end

@testset "a fresh process reads the entries back and does not retune" begin
    mine = Dict(parse_entry(l) for l in readlines_or_empty(TUNE_FILE))
    @test haskey(mine, shape_key(S_ACC))
    path = joinpath(DIR, "readback.tsv")
    planted = GemmConfig(2, SemiringGPU.TILING_N32)  # a valid candidate (n > 64) the tuner would rarely pick
    @test planted in gemm_candidates(S_PLANT[2], false, min3_ok(s, Float32), true, SemiringGPU.pair_ok(s, Float32))
    kstale = shape_key(S_STALE)
    stale = replace(kstale, GEMM_TUNE_KEY => "gemm-0123456789abcdef")
    @test stale != kstale
    garbage = ["", "garbage", "a\tb\tc\td\te\tf\tg", "$kstale\t2\t128", "$kstale\t2\t128\t128\t8\t8\t8\t9",
               "$kstale\t2\t128\t128\tx\t8\t8", "$kstale\t2\t128\t128\t8\t8\t", "\t\t\t\t\t\t", "$kstale 2 128 128 8 8 8"]
    lines0 = vcat(readlines_or_empty(TUNE_FILE), [entry_line(stale, GemmConfig(2, SemiringGPU.TILING_N16))], garbage,
                  [entry_line(shape_key(S_PLANT), planted)])
    write(path, join(lines0, '\n') * '\n')

    r = run_child("run", [S_ACC, S_INPLACE, S_PLANT, S_STALE]; env = Dict("SEMIRINGGPU_TUNE_FILE" => path))
    @test r.code == 0
    @test same_source(r)
    r.code == 0 || println(r.text)
    sh = r.shapes
    @test all(x -> haskey(x, :ok) && x.ok, values(sh))            # every result equals version 2's
    @test length(sh) == 4
    if r.code == 0 && length(sh) == 4
        @test sh[shape_arg(S_ACC)].cfg == cfgstring(mine[shape_key(S_ACC)])           # what this process tuned
        @test sh[shape_arg(S_INPLACE)].cfg == cfgstring(mine[shape_key(S_INPLACE)])
        @test sh[shape_arg(S_PLANT)].cfg == cfgstring(planted)                        # what the file says
        lines1 = readlines(path)
        @test lines1[1:length(lines0)] == lines0                                       # appended only
        new = lines1[(length(lines0) + 1):end]
        @test length(new) == 1                                                         # only the stale class is tuned
        e = parse_entry(only(new))
        @test e !== nothing && e[1] == kstale                                          # under the current key
        @test e !== nothing && e[2] in gemm_candidates(S_STALE[2], false, min3_ok(s, Float32), true, SemiringGPU.pair_ok(s, Float32))
        @test r.saved == 1
    end
    @test !ispath(path * ".lock")
end

@testset "two processes tuning at once leave a well-formed file" begin
    # 28 shape classes, the same in both processes (both tune every one: each loaded the empty table first)
    shapes = vec([(100 + 7n, n, k, ow, false) for n in (16, 32, 64, 96, 128, 192, 256), k in (16, 128), ow in (false, true)])
    keys_expected = Set(shape_key.(shapes))
    @test length(keys_expected) == length(shapes)
    path = joinpath(DIR, "race.tsv")
    ready = [joinpath(DIR, "ready$i") for i in 1:2]
    logs = [joinpath(DIR, "race$i.log") for i in 1:2]
    ios = [open(l, "w") for l in logs]
    procs = [run(pipeline(ignorestatus(child_cmd("race", shapes; env = Dict("SEMIRINGGPU_TUNE_FILE" => path,
                 "TT_READY" => ready[i], "TT_WAIT" => ready[3 - i]))); stdout = ios[i], stderr = ios[i]); wait = false) for i in 1:2]
    foreach(wait, procs)
    foreach(close, ios)
    rs = [parse_child(procs[i].exitcode, read(logs[i], String)) for i in 1:2]

    for r in rs
        @test r.code == 0
        r.code == 0 || println(r.text)
        @test same_source(r)
        @test r.saved == length(shapes)
        @test all(x -> haskey(x, :ok) && x.ok, values(r.shapes))
    end
    L = readlines_or_empty(path)
    E = parse_entry.(L)
    @test length(L) == 2 * length(shapes)                          # nothing lost
    @test all(!isnothing, E)                                       # no torn or interleaved lines
    @test endswith(read(path, String), '\n')
    good = filter(!isnothing, E)
    counts = Dict{String, Int}()
    for (k, _) in good; counts[k] = get(counts, k, 0) + 1; end
    @test Set(keys(counts)) == keys_expected
    @test all(==(2), values(counts))                               # each class once per process
    @test all(c in gemm_candidates(sh[2], false, min3_ok(s, Float32), true, SemiringGPU.pair_ok(s, Float32)) for sh in shapes for (k, c) in good if k == shape_key(sh))
    @test !ispath(path * ".lock")
end

@testset "tuning off writes nothing (SEMIRINGGPU_TUNE=off)" begin
    path = joinpath(DIR, "off.tsv")
    r = run_child("run", [S_ACC, S_OW, S_INPLACE, S_PLANT]; env = Dict("SEMIRINGGPU_TUNE_FILE" => path, "SEMIRINGGPU_TUNE" => "off"))
    @test r.code == 0
    r.code == 0 || println(r.text)
    @test r.tune === false
    @test all(x -> haskey(x, :ok) && x.ok, values(r.shapes))
    @test all(x -> x.cfg == "none", values(r.shapes))              # never in the table
    @test r.saved == 0
    @test !ispath(path)
    @test !ispath(path * ".lock")

    r = run_child("run", [S_ACC]; env = Dict("SEMIRINGGPU_TUNE_FILE" => path, "SEMIRINGGPU_GEMM_TUNE" => "false"))
    @test r.code == 0 && r.tune === false && r.saved == 0
    @test !ispath(path)
end

@testset "tuning keys" begin
    k(s, V = Float32) = gemm_key(s, V, 300, 150, 200, false, false)
    @test k(MinPlus()) != k(MinPlus(), Float64)
    @test k(MinPlus()) != k(MaxPlus())
    @test gemm_key(s, Float32, 300, 150, 200, false, false) != gemm_key(s, Float32, 300, 150, 200, true, false)
    @test gemm_key(s, Float32, 300, 100, 100, true, false) != gemm_key(s, Float32, 300, 100, 100, true, true)
    @test gemm_key(s, Float32, 300, 150, 200, false, false) != gemm_key(s, Float32, 3000, 150, 200, false, false)
    # semirings with different kernels must not share tuned choices: MaxPlus and MaxMin are both
    # DualQuantale{...}, and nameof(typeof(s)) drops the parameter
    known_bug(k(MaxPlus()) != k(MaxMin()), "gemm_key gives MaxPlus and MaxMin (both DualQuantale) the same key")
    # the consequence on GPUs where tuning picks version 6 for max-plus (min3_ok, sm_100+): max-min then
    # runs kernel 6, which only exists for min-plus / max-plus. Simulated by storing that choice.
    if min3_ok(MaxPlus(), Float32) && k(MaxPlus()) == k(MaxMin())
        key = k(MaxPlus())
        old = get(GEMM_TABLE, key, nothing)
        GEMM_TABLE[key] = GemmConfig(6, SemiringGPU.TILING_MID)          # as tuning max-plus may store it
        A = operand(300, 200); B = operand(200, 150); C = operand(300, 150)
        ref = with_config(() -> Array(sgemx_gpu!(MaxMin(), copy(C), A, B)); gemm_kernel = 2)
        ok = try
            Array(sgemx_gpu!(MaxMin(), copy(C), A, B)) == ref
        catch e
            println("  max-min GEMM after a max-plus v6 choice: ", typeof(e))
            false
        end
        old === nothing ? delete!(GEMM_TABLE, key) : (GEMM_TABLE[key] = old)
        known_bug(ok, "a max-min GEMM uses the version-6 kernel tuned for max-plus (InvalidIRError)")
    end
end

@testset "entries of the current key are not validated" begin
    # a corrupted or hand-edited file: well-formed lines whose numbers are not a candidate kernel
    path = joinpath(DIR, "bad.tsv")
    S_ZERO = (300, 150, 200, false, false); S_WIDE = (5000, 100, 100, true, true)
    write(path, entry_line(shape_key(S_ZERO), GemmConfig(2, 0, 0, 0, 0, 0)) * "\n" *
                entry_line(shape_key(S_WIDE), GemmConfig(2, SemiringGPU.TILING_N32)) * "\n")  # bn = 32 < n = 100 in place
    r = run_child("run", [S_ZERO, S_WIDE]; env = Dict("SEMIRINGGPU_TUNE_FILE" => path))
    @test r.code == 0
    @test same_source(r)
    z = get(r.shapes, shape_arg(S_ZERO), (crash = "missing",))
    w = get(r.shapes, shape_arg(S_WIDE), (crash = "missing",))
    known_bug(haskey(z, :ok) && z.ok, "a current-key entry with a zero tiling crashes the GEMM ($(get(z, :crash, "")))")
    known_bug(haskey(w, :ok) && w.ok, "an in-place entry with bn < n is used and gives wrong results")
end

@testset "every kernel gives the result of version 2" begin
    shapes = [(70, 33, 17), (130, 20, 9), (64, 64, 64), (200, 129, 31), (257, 300, 70), (1, 1, 1), (5, 200, 3), (300, 16, 300), (40, 128, 8)]
    # MaxMin before MaxPlus: see "tuning keys" (a max-plus version-6 choice would be reused for max-min)
    for (sr, V) in [(MinPlus(), Float32), (MaxMin(), Float32), (MaxPlus(), Float32), (MinPlus(), Float64), (PlusProd(), Float64)]
        z = szero(sr, V, Val(:N))
        gen(m, n) = sr isa PlusProd ? CuArray(V.(rand(m, n) .- 0.3)) : CuArray(map(x -> rand() < 0.1 ? z : V(x), rand(0:9, m, n)))
        same(x, y) = sr isa PlusProd ? isapprox(x, y; rtol = 1e-12) : isequal(x, y)
        nbad = 0; ncmp = 0
        for (m, n, k) in shapes, ow in (false, true)
            A = gen(m, k); B = gen(k, n); C = gen(m, n)
            ref = with_config(() -> Array(sgemx_gpu!(sr, copy(C), A, B; overwrite = ow)); gemm_kernel = 2)
            runs = Pair{String, Function}["tuned" => () -> sgemx_gpu!(sr, copy(C), A, B; overwrite = ow)]
            for v in (2, 4, 6)
                push!(runs, "v$v heuristic" => () -> with_config(() -> sgemx_gpu!(sr, copy(C), A, B; overwrite = ow); gemm_kernel = v))
                for t in TILINGS
                    push!(runs, "v$v $(typeof(t).parameters)" => () -> with_config(() -> sgemx_gpu!(sr, copy(C), A, B; overwrite = ow, tiling = t); gemm_kernel = v))
                end
            end
            if ow && k == n && n <= 128                                  # in place: C === A, one tile wide
                push!(runs, "in place tuned" => () -> (X = copy(A); sgemx_gpu!(sr, X, X, B; overwrite = true)))
                for v in (2, 4, 6), t in TILINGS
                    SemiringGPU.tiling_bn(t) >= n || continue
                    push!(runs, "in place v$v $(typeof(t).parameters)" => () -> (X = copy(A); with_config(() -> sgemx_gpu!(sr, X, X, B; overwrite = true, tiling = t); gemm_kernel = v)))
                end
            end
            for (name, f) in runs
                ncmp += 1
                got = Array(f())
                if !same(got, ref)
                    nbad += 1
                    nbad <= 5 && println("  $sr $V $((m, n, k)) overwrite=$ow $name: differs from version 2")
                end
            end
        end
        println("  ", rpad("$sr $V", 32), ncmp, " kernel runs against version 2: ", nbad == 0 ? "identical" : "$nbad differ")
        @test nbad == 0
    end
end

end # test_tuning
