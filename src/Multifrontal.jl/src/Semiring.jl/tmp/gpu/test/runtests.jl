# Test runner: every test file in its own fresh Julia process (no module state leaks between files),
# output captured to one log file per test, then a summary table. Exits nonzero if anything failed.
#
#   julia --project=. test/runtests.jl [quick|full] [names...] [--seed=N] [--logdir=DIR]
#
#   quick (default)  test_config test_tuning test_multigpu test_solve test_factor test_closure
#                    test_api (if present) sass_guard, stress.jl 50
#   full             quick + stress.jl 800, stress_large.jl 30, sanitize_small (plain run, no sanitizer)
#   names...         run only these, e.g. `quick test_tuning stress` (the `test_` prefix and `.jl` may be
#                    left out; a file name selects all of its entries, e.g. `stress` → stress[50] in quick,
#                    stress[50] and stress[800] in full). A name outside the suite (e.g. test_cuda_backend,
#                    the archived C++ backend, which no suite runs) is run when asked for explicitly.
#   --seed=N         first seed of the randomized stress files (default: theirs, 1)
#   --logdir=DIR     where the logs go (default: $SEMIRINGGPU_TEST_LOGDIR, else a new temporary directory)
#   --list           print what would run (file, arguments, timeout) and exit
#
# Each child runs `julia --project=<this project> -t auto` (SEMIRINGGPU_TEST_THREADS overrides `auto`),
# with SEMIRINGGPU_TUNE_FILE set to a file in the log directory unless it is already set, so the
# suite never reads or writes the GEMM tuning cache in the Julia depot. Other SEMIRINGGPU_* variables
# are passed through (e.g. SEMIRINGGPU_MERGE=1 runs the suite with that default; test_config clears them).
#
# A test fails when its process exits nonzero (Test-based files, crashes), runs past its timeout, prints
# a failure line of the script-style tests (`FAIL`, `ERROR seed=`, `N failures` with N > 0,
# `sass guard: N suspicious`), or does not print its completion line.

using Printf

const TESTDIR = @__DIR__
const PROJECT = let p = Base.active_project()
    (p === nothing || !isfile(p)) ? dirname(TESTDIR) : dirname(p)
end

struct Entry
    name::String                      # unique, e.g. "stress[800]"
    file::String                      # in test/
    args::Vector{String}
    timeout::Float64                  # seconds
    done::Union{Nothing, Regex}       # completion line the output must contain
end

Entry(name, file; args = String[], timeout = 3600, done = nothing) = Entry(name, file, args, timeout, done)

const MIN = 60.0
const HOUR = 3600.0

# every runnable entry; the suites below select from it
const CATALOG = [
    Entry("test_config", "test_config.jl"; timeout = 30MIN),
    Entry("test_tuning", "test_tuning.jl"; timeout = 30MIN),
    Entry("test_multigpu", "test_multigpu.jl"; timeout = 30MIN),
    Entry("test_solve", "test_solve.jl"; timeout = 45MIN),
    Entry("test_factor", "test_factor.jl"; timeout = 45MIN),
    Entry("test_closure", "test_closure.jl"; timeout = 45MIN),
    Entry("test_api", "test_api.jl"; timeout = 45MIN),
    Entry("sass_guard", "sass_guard.jl"; timeout = 30MIN, done = r"sass guard: all ok"),
    Entry("stress[50]", "stress.jl"; args = ["50"], timeout = 1HOUR, done = r"stress: 50 cases, 0 failures"),
    Entry("stress[800]", "stress.jl"; args = ["800"], timeout = 6HOUR, done = r"stress: 800 cases, 0 failures"),
    Entry("stress_large[30]", "stress_large.jl"; args = ["30"], timeout = 6HOUR, done = r"stress_large: 30 cases, 0 failures"),
    Entry("sanitize_small", "sanitize_small.jl"; timeout = 2HOUR, done = r"sanitize workload done"),
    Entry("test_cuda_backend", "test_cuda_backend.jl"; timeout = 1HOUR),        # archived C++ backend: only by name
]

const QUICK = ["test_config", "test_tuning", "test_multigpu", "test_solve", "test_factor", "test_closure",
               "test_api", "sass_guard", "stress[50]"]
const FULL = vcat(QUICK, ["stress[800]", "stress_large[30]", "sanitize_small"])

entry(name) = only(filter(e -> e.name == name, CATALOG))
exists(e::Entry) = isfile(joinpath(TESTDIR, e.file))

# ===== command line =====

function parse_args(args)
    mode = "quick"; names = String[]; seed = nothing; list = false
    logdir = get(ENV, "SEMIRINGGPU_TEST_LOGDIR", "")

    for (i, a) in enumerate(args)
        if i == 1 && a in ("quick", "full")
            mode = a
        elseif startswith(a, "--seed=")
            seed = parse(Int, a[8:end])
        elseif startswith(a, "--logdir=")
            logdir = a[10:end]
        elseif a == "--list"
            list = true
        elseif a in ("-h", "--help")
            println(read(@__FILE__, String) |> s -> join(Iterators.takewhile(startswith("#"), split(s, '\n')), '\n'))
            exit(0)
        elseif startswith(a, "-")
            error("unknown option $a (see the header of test/runtests.jl)")
        else
            push!(names, a)
        end
    end

    return mode, names, seed, logdir, list
end

# the entries a name selects: within the suite first, then anywhere in the catalog
function resolve(name, suite::Vector{Entry})
    base = replace(name, r"\.jl$" => "")
    cands = unique([base, "test_" * base])
    matches(e) = e.name in cands || replace(e.file, r"\.jl$" => "") in cands
    found = filter(matches, suite)
    isempty(found) && (found = filter(matches, CATALOG))
    isempty(found) && error("no test named `$name`; known: $(join(map(e -> e.name, CATALOG), ", "))")
    return found
end

function select(mode, names)
    suite = [entry(n) for n in (mode == "full" ? FULL : QUICK)]
    isempty(names) && return filter(exists, suite)     # test_api.jl only when present
    sel = unique(vcat([resolve(n, suite) for n in names]...))
    missing = filter(!exists, sel)
    isempty(missing) || error("test file(s) not found: $(join(map(e -> e.file, missing), ", "))")
    return sel
end

# ===== GPU check =====

const GPU_PROBE = """
using CUDA
ok = try
    CUDA.functional(true); true
catch e
    println("reason: ", sprint(showerror, e)); false
end
ok || exit(3)
d = CUDA.device()
info = try
    string(", ", round(CUDA.totalmem(d) / 2^30; digits = 1), " GiB, CUDA runtime ", CUDA.runtime_version(), ", driver ", CUDA.driver_version())
catch
    ""
end
println(CUDA.name(d), " (sm_", CUDA.capability(d).major, CUDA.capability(d).minor, info, "), ", length(CUDA.devices()), " device(s)")
"""

function check_gpu()
    out = IOBuffer()
    cmd = `$(Base.julia_cmd()) --project=$PROJECT --startup-file=no --color=no -e $GPU_PROBE`
    p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out))
    return success(p), strip(String(take!(out)))
end

# ===== running one test =====

const FAIL_LINE = [r"\bFAIL\b", r"\bERROR seed=", r"sass guard: \d+ suspicious"]

function failure_lines(text)
    bad = String[]

    for line in split(text, '\n')
        if any(r -> occursin(r, line), FAIL_LINE)
            push!(bad, line)
        else
            m = match(r"\b(\d+) failures\b", line)
            m !== nothing && parse(Int, m[1]) > 0 && push!(bad, line)
        end
    end

    return bad
end

safe(name) = replace(name, r"[^A-Za-z0-9_.-]" => "_")

function run_entry(e::Entry, logdir, env, seed, threads)
    log = joinpath(logdir, safe(e.name) * ".log")
    args = copy(e.args)
    seed !== nothing && e.file in ("stress.jl", "stress_large.jl") && push!(args, string(seed))
    cmd = `$(Base.julia_cmd()) --project=$PROJECT --startup-file=no --color=no -t $threads $(joinpath(TESTDIR, e.file)) $args`
    cmd = Cmd(setenv(ignorestatus(cmd), env); dir = PROJECT)
    timedout = Ref(false)
    t0 = time()

    proc = open(log, "w") do io
        println(io, "# ", cmd.exec |> x -> join(x, " "))
        flush(io)
        p = run(pipeline(cmd; stdout = io, stderr = io); wait = false)
        watchdog = Timer(e.timeout) do _
            if process_running(p)
                timedout[] = true
                kill(p, Base.SIGTERM)
                for _ in 1:30
                    process_running(p) || break
                    sleep(1)
                end
                process_running(p) && kill(p, Base.SIGKILL)
            end
        end
        wait(p)
        close(watchdog)
        p
    end

    secs = time() - t0
    text = read(log, String)
    bad = failure_lines(text)
    reasons = String[]
    timedout[] && push!(reasons, "timed out after $(round(Int, e.timeout)) s")
    proc.exitcode != 0 && push!(reasons, "exit code $(proc.exitcode)" * (proc.termsignal != 0 ? ", signal $(proc.termsignal)" : ""))
    isempty(bad) || push!(reasons, "$(length(bad)) failure line(s)")
    e.done !== nothing && !occursin(e.done, text) && push!(reasons, "no completion line /$(e.done.pattern)/")
    return (; e, ok = isempty(reasons), secs, log, reasons, bad, text)
end

function tail(text, n)
    lines = split(rstrip(text), '\n')
    return join(lines[max(1, end - n + 1):end], '\n')
end

# ===== main =====

function main(args)
    mode, names, seed, logdir, list, tests = try
        mode, names, seed, logdir, list = parse_args(args)
        mode, names, seed, logdir, list, select(mode, names)
    catch e
        e isa ErrorException || e isa ArgumentError || rethrow()
        println(stderr, "runtests.jl: ", e isa ErrorException ? e.msg : sprint(showerror, e))
        exit(2)
    end
    if list
        for e in tests
            a = vcat(e.args, seed !== nothing && e.file in ("stress.jl", "stress_large.jl") ? [string(seed)] : String[])
            @printf("%-18s test/%s %s  (timeout %.0f min)\n", e.name, e.file, join(a, " "), e.timeout / 60)
        end
        exit(0)
    end

    logdir = isempty(logdir) ? mktempdir(; prefix = "semiringgpu-tests-", cleanup = false) : (mkpath(logdir); abspath(logdir))
    threads = get(ENV, "SEMIRINGGPU_TEST_THREADS", "auto")

    println("SemiringGPU tests: $mode, $(length(tests)) file(s) | project $PROJECT | Julia $VERSION | -t $threads")
    PROJECT == dirname(TESTDIR) || println("  note: running against the active project $PROJECT, not $(dirname(TESTDIR))")
    flush(stdout)
    ok, gpu = check_gpu()

    if !ok
        println(stderr, "\nNo functional CUDA GPU: these tests need an NVIDIA GPU that CUDA.jl can use.")
        isempty(gpu) || println(stderr, "  ", replace(gpu, "\n" => "\n  "))
        println(stderr, "Nothing was run.")
        exit(2)
    end

    println("GPU: $gpu")
    println("logs: $logdir\n")
    env = copy(ENV)
    haskey(env, "SEMIRINGGPU_TUNE_FILE") || (env["SEMIRINGGPU_TUNE_FILE"] = joinpath(logdir, "gemm_tuning.tsv"))
    results = []

    for (i, e) in enumerate(tests)
        @printf("[%d/%d] %-18s ... ", i, length(tests), e.name)
        flush(stdout)
        r = run_entry(e, logdir, env, seed, threads)
        push!(results, r)
        @printf("%s (%.1f s)\n", r.ok ? "pass" : "FAILED", r.secs)

        if !r.ok
            println("    ", join(r.reasons, "; "), "  (log: $(r.log))")
            for l in first(r.bad, 10)
                println("    | ", l)
            end
            println("    last lines:")
            println("    | ", replace(tail(r.text, 25), "\n" => "\n    | "))
        end
        flush(stdout)
    end

    nfail = count(r -> !r.ok, results)
    w = maximum(r -> length(r.e.name), results; init = 4)
    println("\n", rpad("test", w), "  result  seconds  log")
    println("-"^(w + 2 + 6 + 2 + 7 + 2 + 3))
    for r in results
        @printf("%s  %-6s  %7.1f  %s\n", rpad(r.e.name, w), r.ok ? "pass" : "FAIL", r.secs, r.log)
    end
    println("-"^(w + 2 + 6 + 2 + 7 + 2 + 3))
    @printf("%s  %d/%d passed, %.1f s total\n", rpad("", w), length(results) - nfail, length(results), sum(r -> r.secs, results; init = 0.0))

    exit(nfail == 0 ? 0 : 1)
end

main(ARGS)
