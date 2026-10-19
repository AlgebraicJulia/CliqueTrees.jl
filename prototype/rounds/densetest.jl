# dense-fill families: GPU rounds with the dense end on the CPU (MF or AMF)
const DIR = @__DIR__
include(joinpath(DIR, "Rounds.jl")); using .Rounds
using CUDA
include(joinpath(DIR, "GRounds.jl")); using .GRounds
include(joinpath(DIR, "evalutil.jl")); include(joinpath(DIR, "gens.jl"))
include(get(ENV, "CORPUS", joinpath(DIR, "..", "tncorpus")) * "/load.jl"); using .TNCorpus
out = open(ENV["OUT"], "a")
names = split(ENV["NAMES"], ",")
serialmax = parse(Int, get(ENV, "SERIALMAX", "25000"))
tailsel = parse(Int, get(ENV, "TAILSEL", "128"))
let g = lat2(12), w = randw(nv(g))
    GRounds.grounds(w, g, RoundsAlg(; score = :mf, relax = 0.5, tailsel = 128, tailmax = 10^9, tailalg = AMF()))
    permutation(w, g, AMF()); permutation(w, g, MF())
end
for nm in names
    inst = TNCorpus.load(String(nm)); g = inst.graph; w = inst.weights
    function rec(meth, o, t, extra = "")
        wd, lc = quality(w, g, o)
        println(out, join((gethostname(), nm, nv(g), meth, wd, lc, t, extra), '\t')); flush(out)
        println(rpad(nm, 22), rpad(meth, 18), " w=", round(wd, digits = 1), " t=", round(t, digits = 3), " ", extra); flush(stdout)
    end
    for (an, a) in ("AMF" => AMF(), "MMD" => MMD(), "MF" => MF(), "MF1" => MF(; strategy = 1))
        (an in ("MF", "MF1") && nv(g) > serialmax) && continue
        get(ENV, "SERIAL", "1") == "0" && an in ("MF", "MF1") && continue
        t = @elapsed o = try permutation(w, g, a)[1] catch; nothing end
        o === nothing || rec(an, o, t)
    end
    for (vn, sc, tl) in (("gpu-mf+MF", :mf, nothing), ("gpu-mf+AMF", :mf, AMF()), ("gpu-mf1+AMF", :mf1, AMF()), ("gpu-md+AMF", :md, AMF()))
        (tl === nothing && nv(g) > serialmax) && continue
        alg = RoundsAlg(; score = sc, relax = 0.5, tailsel, tailmax = 10^9, tailalg = tl)
        tm = Dict{Symbol,Any}()
        t = @elapsed o = GRounds.grounds(w, g, alg; timing = tm)[1]
        rec(vn, o, t, "gpu=$(round(tm[:gpu],digits=3));cpu_post=$(round(tm[:cpu_post],digits=3));rounds=$(tm[:rounds]);tail=$(tm[:tail])")
    end
end
