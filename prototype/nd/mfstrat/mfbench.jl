# Serial ordering baselines (CPU), one CliqueTrees version per process.
# usage: julia --project=ENV -t1 sbench.jl TAG OUT ALGS name...
#   TAG  : label for the CliqueTrees version (e.g. rmain = Richard's origin/main, mt = our old baseline)
#   ALGS : comma list of MF1, MF, AMF, AMD, MMD
#   name : mtx basename / .bin basename (dirs in DATA, ':'-separated), lat2_L, lat3_L, or corpus:NAME
#          (TN corpus at CORPUS; unit weights everywhere: fill metrics)
# appends: host tag graph n m alg nnzL flops tw time  (time = min over reps after a warm-up on a tiny graph)
using CliqueTrees, Graphs, SparseArrays, Serialization, Random, AMD, Metis
const RDIR = @__DIR__
include(joinpath(RDIR, "gens.jl"))
const DIRS = split(get(ENV, "DATA", joinpath(RDIR, "..", "fill", "data")), ":")
const HAVEMM = Base.find_package("MatrixMarket") !== nothing
HAVEMM && @eval using MatrixMarket
if haskey(ENV, "CORPUS")
    include(ENV["CORPUS"] * "/load.jl"); @eval using .TNCorpus
end

function loadg(name)
    startswith(name, "corpus:") && return TNCorpus.load(name[8:end]).graph
    for d in DIRS
        f = joinpath(d, name * ".bin")
        if isfile(f)
            n, I, J = deserialize(f)
            return SimpleGraph(sparse(vcat(I, J), vcat(J, I), true, n, n, |))
        end
        f = joinpath(d, name * ".mtx")
        if isfile(f) && HAVEMM
            A = mmread(f); B = SparseMatrixCSC(A .!= 0); B = B + B'
            return SimpleGraph(B)
        end
    end
    startswith(name, "lat2_") && return lat2(parse(Int, name[6:end]))
    startswith(name, "lat3_") && return lat3(parse(Int, name[6:end]))
    error("no data for $name")
end

mkalg(a) = startswith(a, "MF") && length(a) == 3 ? MF(; strategy = parse(Int, a[3])) : a == "AMF" ? AMF() : a == "AMD" ? CliqueTrees.AMD() : a == "METIS" ? CliqueTrees.METIS() : error("alg $a")

# nnz(L) (diagonal included), flops = sum of squared column counts, tw = max column count - 1,
# from the supernodes of the elimination order (public API only, so it works on every CliqueTrees version)
function fillq(g, o)
    n = nv(g)
    @assert length(o) == n && isperm(o) "invalid permutation"
    n == 0 && return 0, 0.0, -1
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

tag, outf, algs = ARGS[1], ARGS[2], split(ARGS[3], ",")
let g = lat2(10), bg = BipartiteGraph(g)
    for a in algs; permutation(bg, mkalg(a)); end
end
out = open(outf, "a"); host = gethostname()
tmax = parse(Float64, get(ENV, "TMAX", "600"))
for name in ARGS[4:end]
    g = try loadg(name) catch err; println("SKIP $name: ", sprint(showerror, err)); continue end
    n = nv(g); bg = BipartiteGraph(g)
    for a in algs
        alg = mkalg(a)
        t = Inf; o = nothing
        reps = n < 1_000_000 ? 3 : 1
        for r in 1:reps
            t1 = @elapsed (o1 = permutation(bg, alg)[1])
            t1 < t && (t = t1; o = o1)
            t1 > tmax / 4 && break
        end
        q = fillq(g, o)
        println(out, join((host, tag, name, n, ne(g), a, q[1], q[2], q[3], t), '\t')); flush(out)
        println(rpad(name, 22), rpad(a, 5), " nnzL=", q[1], " tw=", q[3], " t=", round(t, digits = 4)); flush(stdout)
        o = nothing; GC.gc()
    end
end
close(out)
