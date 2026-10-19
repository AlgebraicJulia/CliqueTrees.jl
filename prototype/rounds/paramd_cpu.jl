# ParAMD thread scaling vs serial AMF/MMD and the CPU rounds (same node)
const DIR = @__DIR__
include(joinpath(DIR, "Rounds.jl")); using .Rounds
include(joinpath(DIR, "evalutil.jl")); include(joinpath(DIR, "gens.jl"))
include(get(ENV, "CORPUS", joinpath(DIR, "..", "tncorpus")) * "/load.jl"); using .TNCorpus
const PARAMD = ENV["PARAMD"]; const MTX = ENV["MTXDIR"]
out = open(ENV["OUT"], "a")
threads = parse.(Int, split(get(ENV, "PTHREADS", "1,2,4,8,16,32,64"), ","))
function writemtx(path, g)
    open(path, "w") do io
        println(io, "%%MatrixMarket matrix coordinate pattern symmetric")
        println(io, nv(g), " ", nv(g), " ", ne(g))
        for e in edges(g)
            println(io, max(src(e), dst(e)), " ", min(src(e), dst(e)))
        end
    end
end
function paramd(path, nt)
    best = Inf; o = nothing
    for rep in 1:3
        txt = read(addenv(`$PARAMD --matrix $path --algo paramd --mult 1.1 --lim 8192 --mem 3.0 --seed 1 --sym`, "OMP_NUM_THREADS" => string(nt), "OMP_PROC_BIND" => "close", "OMP_PLACES" => "cores", "PATH" => ENV["PATH"]), String)
        best = min(best, parse(Float64, match(r"Ordering Time = ([0-9.eE+-]+)", txt)[1]))
        o = parse.(Int, split(read(path * ".perm", String))) .+ 1
    end
    return o, best
end
insts = Any[]
for L in parse.(Int, filter(!isempty, split(get(ENV, "LAT2", "256,512,1024,2048"), ",")))
    push!(insts, ("lat2_$L", () -> (g = lat2(L); (g, randw(nv(g))))))
end
for nm in filter(!isempty, split(get(ENV, "NAMES", ""), ","))
    push!(insts, (String(nm), () -> (i = TNCorpus.load(String(nm)); (i.graph, i.weights))))
end
for (name, ld) in insts
    g, w = ld(); u = ones(nv(g))
    path = joinpath(MTX, name * ".mtx"); writemtx(path, g)
    function rec(meth, o, t)
        wd, lc = quality(w, g, o); wu, lu = quality(u, g, o)
        println(out, join((gethostname(), name, nv(g), meth, wd, lc, wu, lu, t), '\t')); flush(out)
        println(rpad(name, 20), rpad(meth, 22), " w=", round(wd, digits = 1), " unit=", round(wu, digits = 1), " t=", round(t, digits = 4)); flush(stdout)
    end
    for nt in threads
        o, t = paramd(path, nt); rec("ParAMD-t$nt", o, t)
    end
    for (an, a) in ("AMF" => AMF(), "MMD" => MMD()), (wn, ww) in (("w", w), ("u", u))
        permutation(ww, g, a)
        t = @elapsed o = permutation(ww, g, a)[1]
        rec("$an-$wn", o, t)
    end
    rm(path; force = true); rm(path * ".perm"; force = true)
end
