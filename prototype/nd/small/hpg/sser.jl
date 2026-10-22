# Serial baselines for the small-graph sweep (sgpu.jl): Richard's new serial MF1 (CliqueTrees in this env)
# usage: julia --project=RENV -t8 sser.jl
# env: OUT, LEAVES, CORPUS, DATA, SINGLE (comma list), LTAGS (comma list of leaf files NAME_LL, without .jls)
# single: time = min of 3 runs of permutation(BipartiteGraph(g), MF(strategy=1)) (conversion excluded, as sbench.jl)
# leaves: 1 thread = the leaves one after another; 8 threads = Threads.@threads :dynamic over the leaves
#         (min of 2 / 3 runs; BipartiteGraph conversion excluded, its time is in the row); fill from CliqueTrees
using CliqueTrees, Graphs, SparseArrays, Serialization, Random
const DIR = joinpath(@__DIR__, "..", "..")
include(joinpath(DIR, "gens.jl"))
include(ENV["CORPUS"] * "/load.jl"); using .TNCorpus
const dirs = split(get(ENV, "DATA", ""), ":")
const host = gethostname()
const out = open(ENV["OUT"], "a")
const ALG = MF(; strategy = 1)

function loadbin(name)
    startswith(name, "corpus:") && return TNCorpus.load(name[8:end]).graph
    for d in dirs
        f = joinpath(d, name * ".bin")
        if isfile(f)
            n, I, J = deserialize(f)
            return SimpleGraph(sparse(vcat(I, J), vcat(J, I), true, n, n, |))
        end
    end
    startswith(name, "lat2_") && return lat2(parse(Int, name[6:end]))
    startswith(name, "lat3_") && return lat3(parse(Int, name[6:end]))
    error("no data for $name")
end

# nnz(L), flops, tw from the supernodes (public API; same definition as the GPU counters)
function fillq(g, o)
    n = nv(g)
    @assert length(o) == n && isperm(o) "invalid permutation"
    _, tree = cliquetree(g; alg = o)
    nz = 0; fl = 0.0; mx = 0
    for bag in tree
        r = length(residual(bag)); s = length(separator(bag))
        nz += r * s + (r * (r + 1)) >> 1
        fl += r * Float64(s)^2 + Float64(s) * r * (r + 1) + r * (r + 1) * (2r + 1) / 6
        mx = max(mx, r + s)
    end
    return nz, fl, mx - 1
end

row(part, name, n, m, cfg, nnz, fl, tw, t, extra) =
    (println(out, join((host, "cpu", part, name, n, m, cfg, nnz, fl, tw, round(t, digits = 5), extra), '\t')); flush(out))

let g = lat2(10), bg = BipartiteGraph(g); permutation(bg, ALG); end
lst(k) = [String(x) for x in split(get(ENV, k, ""), ",") if !isempty(x)]

for name in lst("SINGLE")
    g = try loadbin(name) catch err; println("SKIP $name: $err"); continue end
    bg = BipartiteGraph(g)
    t = Inf; o = nothing
    for r in 1:3
        t1 = @elapsed (o1 = permutation(bg, ALG)[1])
        t1 < t && (t = t1; o = o1)
    end
    q = fillq(g, o)
    row("single", name, nv(g), ne(g), "serial-MF1", q[1], q[2], q[3], t, "")
    println(rpad(name, 22), " MF1 t=", round(t, digits = 4), " nnz=", q[1]); flush(stdout)
end

for tag in lst("LTAGS")
    f = joinpath(ENV["LEAVES"], tag * ".jls")
    isfile(f) || (println("no leaves $tag"); continue)
    gs = deserialize(f)
    nl = length(gs); ntot = sum(nv.(gs)); mtot = sum(ne.(gs))
    tc = @elapsed bgs = [BipartiteGraph(x) for x in gs]
    orders = Vector{Any}(undef, nl)
    for i in 1:min(nl, 4); permutation(bgs[i], ALG); end
    t1 = Inf
    for r in 1:2
        t = @elapsed for i in 1:nl
            orders[i] = permutation(bgs[i], ALG)[1]
        end
        t1 = min(t1, t)
    end
    tp = Inf
    for r in 1:3
        t = @elapsed Threads.@threads :dynamic for i in 1:nl
            permutation(bgs[i], ALG)
        end
        tp = min(tp, t)
    end
    qs = [fillq(gs[i], orders[i]) for i in 1:nl]
    nz = sum(q[1] for q in qs); fl = sum(q[2] for q in qs); tw = maximum(q[3] for q in qs)
    row("batch", tag, ntot, mtot, "serial-MF1-1t", nz, fl, tw, t1, "leaves=$nl;gps=$(round(nl / t1, digits = 1));tconv=$(round(tc, digits = 4))")
    row("batch", tag, ntot, mtot, "serial-MF1-$(Threads.nthreads())t", nz, fl, tw, tp,
        "leaves=$nl;gps=$(round(nl / tp, digits = 1));tconv=$(round(tc, digits = 4))")
    println(rpad(tag, 22), " leaves=", nl, " 1t=", round(t1, digits = 3), " s  ", Threads.nthreads(), "t=", round(tp, digits = 3), " s  nnz=", nz)
    flush(stdout)
end
close(out)
println("DONE")
