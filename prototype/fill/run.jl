# usage: julia --project=../rounds/genv run.jl NAME   (NAME = mtx basename in data/, or a TN graph name)
const DIR = @__DIR__
const R = joinpath(DIR, "..", "rounds")
using CliqueTrees, Graphs, SparseArrays, MatrixMarket, Metis, AMD, CUDA, Random
include(joinpath(R, "Rounds.jl")); using .Rounds
include(joinpath(R, "GRounds.jl")); using .GRounds
include(joinpath(R, "evalutil.jl")); include(joinpath(R, "gens.jl"))
include(joinpath(R, "..", "tncorpus", "load.jl")); using .TNCorpus
const PARAMD = joinpath(R, "..", "dl_paramd", "ParAMD", "build", "paramd")
const name = ARGS[1]
const skip = Set(split(get(ENV, "SKIP", ""), ","))
const only = get(ENV, "ONLY", "")
out = open(joinpath(DIR, "results.tsv"), "a")

function loadg(name)
    f = joinpath(DIR, "data", name * ".mtx")
    if isfile(f)
        A = mmread(f); B = SparseMatrixCSC(A .!= 0); B = B + B'
        g = SimpleGraph(B); return g, "ssmc"
    elseif startswith(name, "lat2_")
        return lat2(parse(Int, name[6:end])), "tn"
    else
        return TNCorpus.load(name).graph, "tn"
    end
end
function writemtx(path, g)
    open(path, "w") do io
        println(io, "%%MatrixMarket matrix coordinate pattern symmetric")
        println(io, nv(g), " ", nv(g), " ", ne(g))
        for e in edges(g); println(io, max(src(e), dst(e)), " ", min(src(e), dst(e))); end
    end
end
function paramd(path, nt; reps)
    best = Inf; o = nothing
    for _ in 1:reps
        txt = read(addenv(`$PARAMD --matrix $path --algo paramd --mult 1.1 --lim 8192 --mem 3.0 --seed 1 --sym`, "OMP_NUM_THREADS" => string(nt), "OMP_PROC_BIND" => "true", "OMP_PLACES" => "cores", "PATH" => ENV["PATH"]), String)
        t = parse(Float64, match(r"Ordering Time = ([0-9.eE+-]+)", txt)[1])
        best = min(best, t); o = parse.(Int, split(read(path * ".perm", String))) .+ 1
    end
    return o, best
end

# warmup
let g = lat2(12)
    bg = BipartiteGraph(g)
    for a in (CliqueTrees.AMD(), AMF(), MMD(), MF(), MF(strategy = 1), CliqueTrees.METIS()); permutation(bg, a); end
    for sc in (:md, :mf); GRounds.grounds(ones(nv(g)), g, RoundsAlg(; score = sc, relax = 0.1)); end
end

g, src_ = loadg(name); n = nv(g); bg = BipartiteGraph(g); u = ones(n); ui = ones(Int, n)
reps = n > 150_000 ? 1 : 2
path = joinpath(DIR, "tmp_" * name * ".mtx")
function rec(meth, o, t)
    @assert length(o) == n && isperm(o) "invalid permutation: $meth"
    cw = colweights(ui, g, o)
    nnzL = sum(cw); fl = sum(Float64(c)^2 for c in cw); tw = maximum(cw) - 1
    println(out, join((name, n, ne(g), meth, nnzL, fl, tw, t), '\t')); flush(out)
    println(rpad(name, 18), rpad(meth, 14), " nnzL=", nnzL, " flops=", round(fl, sigdigits = 4), " tw=", tw, " t=", round(t, digits = 3)); flush(stdout)
end
function tm(f; reps)
    local r; best = Inf
    for _ in 1:reps; t = @elapsed(r = f()); best = min(best, t); end
    return r, best
end
meths = [
    "AMD" => () -> permutation(bg, CliqueTrees.AMD())[1],
    "AMF" => () -> permutation(bg, AMF())[1],
    "MMD" => () -> permutation(bg, MMD())[1],
    "METIS-ND" => () -> permutation(bg, CliqueTrees.METIS())[1],
    "MF" => () -> permutation(bg, MF())[1],
    "MF1" => () -> permutation(bg, MF(strategy = 1))[1],
    "GPU-MF" => () -> GRounds.grounds(u, g, RoundsAlg(; score = :mf, relax = 0.1))[1],
    "GPU-MD" => () -> GRounds.grounds(u, g, RoundsAlg(; score = :md, relax = 0.1))[1],
]
for (m, f) in meths
    (m in skip || (!isempty(only) && !(m in split(only, ",")))) && continue
    o, t = tm(f; reps); rec(m, o, t)
end
if !("ParAMD" in skip) && (isempty(only) || occursin("ParAMD", only))
    writemtx(path, g)
    for nt in (1, 8)
        o, t = paramd(path, nt; reps); rec("ParAMD-t$nt", o, t)
    end
    rm(path; force = true); rm(path * ".perm"; force = true)
end
