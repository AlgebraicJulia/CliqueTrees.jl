# ParAMD vs serial AMF/MMD and the round orderings: weighted and unit-weight widths
const DIR = @__DIR__
include(joinpath(DIR, "Rounds.jl")); using .Rounds
using CUDA
include(joinpath(DIR, "GRounds.jl")); using .GRounds
include(joinpath(DIR, "evalutil.jl")); include(joinpath(DIR, "gens.jl"))
include(get(ENV, "CORPUS", joinpath(DIR, "..", "tncorpus")) * "/load.jl"); using .TNCorpus
const PARAMD = ENV["PARAMD"]
out = open(ENV["OUT"], "a")
threads = parse.(Int, split(get(ENV, "PTHREADS", "1,8"), ","))
function writemtx(path, g)
    open(path, "w") do io
        println(io, "%%MatrixMarket matrix coordinate pattern symmetric")
        println(io, nv(g), " ", nv(g), " ", ne(g))
        for e in edges(g)
            println(io, max(src(e), dst(e)), " ", min(src(e), dst(e)))
        end
    end
end
function paramd(path, nt; mult = 1.1)
    best = Inf; o = nothing
    for rep in 1:3
        txt = read(addenv(`$PARAMD --matrix $path --algo paramd --mult $mult --lim 8192 --mem 3.0 --seed 1 --sym`, "OMP_NUM_THREADS" => string(nt), "OMP_PROC_BIND" => "true", "OMP_PLACES" => "cores", "PATH" => ENV["PATH"]), String)
        t = parse(Float64, match(r"Ordering Time = ([0-9.eE+-]+)", txt)[1])
        best = min(best, t)
        o = parse.(Int, split(read(path * ".perm", String))) .+ 1
    end
    return o, best
end
insts = Any[]
for L in parse.(Int, filter(!isempty, split(get(ENV, "LAT2", "64,128,256,512,724"), ",")))
    push!(insts, ("lat2_$L", () -> (g = lat2(L); (g, randw(nv(g))))))
end
for nm in split(get(ENV, "NAMES", ""), ",")
    isempty(nm) && continue
    push!(insts, (String(nm), () -> (i = TNCorpus.load(String(nm)); (i.graph, i.weights))))
end
let g = lat2(12), w = randw(nv(g))
    for sc in (:md, :mf); GRounds.grounds(w, g, RoundsAlg(; score = sc, relax = 0.1)); end
    permutation(w, g, AMF()); permutation(w, g, MMD())
end
for (name, ld) in insts
    g, w = ld(); u = ones(nv(g))
    path = joinpath(DIR, "mtx", name * ".mtx"); writemtx(path, g)
    function rec(meth, o, t)
        wd, lc = quality(w, g, o); wu, lu = quality(u, g, o)
        println(out, join((gethostname(), name, nv(g), meth, wd, lc, wu, lu, t), '\t')); flush(out)
        println(rpad(name, 20), rpad(meth, 22), " w=", round(wd, digits = 1), " unit=", round(wu, digits = 1), " t=", round(t, digits = 4)); flush(stdout)
    end
    for nt in threads
        o, t = paramd(path, nt); rec("ParAMD-t$nt", o, t)
    end
    for (an, a) in ("AMF" => AMF(), "MMD" => MMD())
        for (wn, ww) in (("w", w), ("u", u))
            t = minimum(@elapsed(permutation(ww, g, a)) for _ in 1:2)
            rec("$an-$wn", permutation(ww, g, a)[1], t)
        end
    end
    for sc in (:md, :mf), relax in (0.1, 0.5), (wn, ww) in (("w", w), ("u", u))
        alg = RoundsAlg(; score = sc, relax)
        GRounds.grounds(ww, g, alg)
        t = minimum(@elapsed(GRounds.grounds(ww, g, alg)) for _ in 1:2)
        rec("gpu-$sc-r$relax-$wn", GRounds.grounds(ww, g, alg)[1], t)
    end
    rm(path; force = true); rm(path * ".perm"; force = true)
end
