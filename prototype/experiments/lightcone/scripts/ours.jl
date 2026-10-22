# usage: julia -t 2 --project=mtenv ours.jl <listfile> <outdir> <exact_limit_s>
# listfile: "<id> <edge list path>" per line (edge file: "n m", then 1-based "u v").
# Pass 1: bw, hbt T=0.2, hbt T=1, hbtnd T=1 on every graph. Pass 2: exact (SafeRules+SafeSeparators+PIDBT) in ascending size, each in a
# spawned task bounded by exact_limit; stops after the first timeout (larger graphs marked TIMEOUT) and exits (killing the runaway task).
# Appends rows "<id>\t<method>\t<seconds>\t<bag width>\t<lower bound>" to <outdir>/rows.tsv, orders to <outdir>/<id>__<method>.ord
include("/home/itaykadosh/Projects/GATAS/tn-work/tncorpus/prof/common.jl")
using Graphs
lst = [split(l) for l in readlines(ARGS[1])]; outdir = ARGS[2]; limit = parse(Float64, ARGS[3])
mkpath(outdir)
function readg(path)
    ls = readlines(path); n, m = parse.(Int, split(ls[1])); g = SimpleGraph(n)
    for l in ls[2:end]; u, v = parse.(Int, split(l)); u != v && add_edge!(g, u, v); end
    g
end
function wu()
    g = SimpleGraph(30); for i in 1:29; add_edge!(g, i, i+1); end; for i in 1:25; add_edge!(g, i, i+5); end
    w = ones(30)
    permutation(w, g, pipeline(HBT(; time = 0.2, threads = 1))); permutation(w, g, pipeline(HBT(; time = 0.2, threads = 1, init = nd8())))
    permutation(w, g, SafeRules(SafeSeparators(PIDBT(), minalgs(); maxmissing = 50))); permutation(w, g, bwalg())
end
wu()
function emit(id, meth, t, w, g, o, lb)
    open(joinpath(outdir, "$(id)__$(meth).ord"), "w") do io; println(io, join(o, ' ')); end
    open(joinpath(outdir, "rows.tsv"), "a") do io; println(io, join((id, meth, t, treewidth(w, g, o), lb), '\t')); end
    println("$id $meth $(round(t; digits=3)) s width $(treewidth(w, g, o)) lb $lb"); flush(stdout)
end
gs = [readg(p) for (_, p) in lst]
function timed(alg, w, g)
    GC.gc(); t0 = time(); o, _ = permutation(w, g, alg); (time() - t0, o)
end
for (k, (id, _)) in enumerate(lst)
    g = gs[k]; w = ones(nv(g)); lb = lowerbound(w, g, MMW())
    t, o = timed(bwalg(), w, g); emit(id, "bw", t, w, g, o, lb)
    for (meth, T, init) in (("hbt0.2", 0.2, false), ("hbt1", 1.0, false), ("hbtnd1", 1.0, true))
        h = init ? HBT(; time = T, threads = 1, init = nd8()) : HBT(; time = T, threads = 1)
        t, o = timed(pipeline(h), w, g); emit(id, meth, t, w, g, o, lb)
    end
end
dead = false
for k in sortperm(nv.(gs))
    id = lst[k][1]; g = gs[k]; w = ones(nv(g)); lb = lowerbound(w, g, MMW())
    if dead
        open(joinpath(outdir, "rows.tsv"), "a") do io; println(io, join((id, "exact", "skip", "TIMEOUT", lb), '\t')); end; continue
    end
    task = Threads.@spawn timed(SafeRules(SafeSeparators(PIDBT(), minalgs(); maxmissing = 50)), w, g)
    if timedwait(() -> istaskdone(task), limit) == :ok
        t, o = fetch(task); emit(id, "exact", t, w, g, o, lb)
    else
        global dead = true
        open(joinpath(outdir, "rows.tsv"), "a") do io; println(io, join((id, "exact", limit, "TIMEOUT", lb), '\t')); end
        println("$id exact TIMEOUT"); flush(stdout)
    end
end
println("FIN"); flush(stdout); exit(0)
