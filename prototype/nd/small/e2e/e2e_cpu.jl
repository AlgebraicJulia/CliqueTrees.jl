# Serial baselines of e2e_gpu.jl: Richard's new serial MF1 (CliqueTrees in this env), one graph per thread.
# usage: julia --project=RENV -t32 e2e_cpu.jl
# env: OUT, LEAVES, LTAGS, SETS (sweep sets of e2e_gpu.jl), SINGLE, DATA, CORPUS, NTH (thread counts, e.g. 1,8,32)
# set rows: T threads = T worker tasks pulling graphs from a shared counter (largest first), min of 3 runs (1 thread:
# min of 2); permutation(BipartiteGraph(g), MF(strategy=1)), the conversion excluded (its total time is tconv);
# fill nnz from CliqueTrees (cliquetree of the 1-thread orders).
using CliqueTrees, Graphs, SparseArrays, Serialization, Random
const DIR = pwd()
include(joinpath(DIR, "gens.jl"))
include(ENV["CORPUS"] * "/load.jl"); using .TNCorpus
const dirs = split(get(ENV, "DATA", ""), ":")
const host = gethostname()
const out = open(ENV["OUT"], "a")
const ALG = MF(; strategy = 1)
lst(k) = [String(x) for x in split(get(ENV, k, ""), ",") if !isempty(x)]
row(part, name, B, n, m, cfg, nnz, t, extra) =
    (println(out, join((host, "cpu", part, name, B, n, m, cfg, nnz, round(t, digits = 5), extra), '\t')); flush(out))

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

function fillnnz(g, o)
    _, tree = cliquetree(g; alg = o)
    nz = 0
    for bag in tree
        r = length(residual(bag)); s = length(separator(bag))
        nz += r * s + (r * (r + 1)) >> 1
    end
    return nz
end

# T workers over the graphs (largest first)
function runpool(bgs, ord, T, orders)
    nxt = Threads.Atomic{Int}(1)
    @sync for _ in 1:T
        Threads.@spawn while true
            i = Threads.atomic_add!(nxt, 1)
            i > length(ord) && break
            b = ord[i]
            o = permutation(bgs[b], ALG)[1]
            orders === nothing || (orders[b] = o)
        end
    end
end

let g = lat2(10); permutation(BipartiteGraph(g), ALG); end
NTH = parse.(Int, lst("NTH"))
function setrows(tag, gs)
    B = length(gs); n = sum(nv.(gs)); m = sum(ne.(gs))
    tc = @elapsed bgs = [BipartiteGraph(x) for x in gs]
    ord = sortperm(nv.(gs); rev = true)
    orders = Vector{Any}(undef, B)
    runpool(bgs, ord[1:min(B, 4)], 1, nothing)
    nz = 0
    for T in NTH
        T > Threads.nthreads() && continue
        t = Inf
        for r in 1:(T == 1 ? 2 : 3)
            GC.gc(false)
            t = min(t, @elapsed runpool(bgs, ord, T, T == 1 ? orders : nothing))
        end
        T == 1 && (nz = sum(fillnnz(gs[b], orders[b]) for b in 1:B))
        row("batch", tag, B, n, m, "serial-MF1-$(T)t", nz, t, "tconv=$(round(tc, digits = 4))")
        println(rpad(tag, 26), " B=", B, " ", T, "t=", round(t, digits = 4), " s"); flush(stdout)
    end
end

for tag in lst("LTAGS")
    f = joinpath(ENV["LEAVES"], tag * ".jls")
    isfile(f) && setrows(tag, filter(x -> nv(x) >= 2, deserialize(f)))
end
for f in sort(readdir(ENV["SETS"]))
    endswith(f, ".jls") && setrows(replace(f, ".jls" => ""), deserialize(joinpath(ENV["SETS"], f)))
end
for name in lst("SINGLE")
    g = try loadbin(name) catch err; println("SKIP $name"); continue end
    bg = BipartiteGraph(g)
    t = Inf; o = nothing
    for r in 1:3
        t1 = @elapsed (o1 = permutation(bg, ALG)[1])
        t1 < t && (t = t1; o = o1)
    end
    row("single", name, 1, nv(g), ne(g), "serial-MF1-1t", fillnnz(g, o), t, "")
    println(rpad(name, 22), " MF1 t=", round(t, digits = 4)); flush(stdout)
end
close(out)
println("DONE")
