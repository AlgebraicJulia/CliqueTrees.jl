# Elimination orderings of our methods for one frontier instance (one process, 1 thread).
#   julia -t 1 --project=<mtenv> orders.jl <name> <method> <budgets> <outdir> [<frontier dir>] [<hbt seed>]
# method: bw | nd | hbt | hbtnd      budgets: comma list of seconds (ignored by bw / nd)
# Writes <outdir>/<name>__<method>__T<budget>__s<seed>.ord : line 1 a TSV header with
#   name method budget seed seconds bagwidth n lb ; line 2 the ordering (vertex ids, eliminated first..last)
# Methods (draft Sec. 5.2, via tncorpus/prof/common.jl, read-only):
#   bw    = BestWidth(MF(), MF(; strategy = 1), AMF(), MMD())
#   nd    = BestWidth of 8 ND(bw, METISND(; seed = s); width = 16, level = 12, scale = 100, imbalance = 300)
#   hbt   = SafeRules(SafeSeparators(HBT(; time = T, threads = 1, seed), 4 minimal greedy orderings; maxmissing = 50))
#   hbtnd = the same with HBT(; init = nd)
include(joinpath(get(ENV, "TNCORPUS_DIR", "/path/to/tn-work/tncorpus"), "prof", "common.jl"))

name = ARGS[1]; method = ARGS[2]; budgets = parse.(Float64, split(ARGS[3], ','))
outdir = ARGS[4]
fdir = length(ARGS) >= 5 ? ARGS[5] : get(ENV, "FRONTIER_DIR", "/path/to/tn-work/frontier")
seed = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 1
mkpath(outdir)
# watchdog: HBT can overrun its budget (seen: syc53_m12_marg6, T = 100 s, still running after 56 min).
# An external process kills this one after 2 x (largest budget) + 900 s (warmup, ND, preprocessing);
# a budget whose ordering never appears is reported as an overrun.
const WATCHDOG = 2 * maximum(budgets) + 900
# (polls every 10 s and exits as soon as this process is gone, so it never signals a reused pid)
let p = getpid(), lim = round(Int, WATCHDOG)
    run(`sh -c "t=0; while kill -0 $p 2>/dev/null; do sleep 10; t=\$((t+10)); if [ \$t -ge $lim ]; then kill -9 $p; exit 0; fi; done"`; wait = false)
end

function wu()
    inst = TNCorpus.load("qc_rand_google53_27_20")
    w = inst.weights; g = TNCorpus.bipartite(inst)
    if method == "bw"
        permutation(w, g, bwalg())
    elseif method == "nd"
        permutation(w, g, nd8())
    elseif method == "hbt"
        permutation(w, g, pipeline(HBT(; time = 0.5, threads = 1)))
    else
        permutation(w, g, pipeline(HBT(; time = 0.5, threads = 1, init = nd8())))
    end
end
wu()

inst = TNCorpus.load(name; dir = fdir)
w = inst.weights; g = TNCorpus.bipartite(inst)
lb = lowerbound(w, g, MMW())

function emit(T, t, o)
    f = joinpath(outdir, "$(name)__$(method)__T$(Int(T))__s$(seed).ord")
    open(f * ".tmp", "w") do io
        println(io, join((name, method, T, seed, t, treewidth(w, g, o), inst.n, lb), '\t'))
        println(io, join(o, ' '))
    end
    mv(f * ".tmp", f; force = true)
    println("$(name) $(method) T=$(T) s=$(seed): $(round(t; digits=3)) s, bag width $(treewidth(w, g, o)) (lb $(lb))")
    flush(stdout)
end

if method in ("bw", "nd")
    alg = method == "bw" ? bwalg() : nd8()
    GC.gc(); t = @elapsed (global o, _ = permutation(w, g, alg))
    emit(0, t, o)
else
    for T in budgets
        h = method == "hbt" ? HBT(; time = T, threads = 1, seed) : HBT(; time = T, threads = 1, seed, init = nd8())
        GC.gc(); local t = @elapsed (global o, _ = permutation(w, g, pipeline(h)))
        emit(T, t, o)
    end
end
