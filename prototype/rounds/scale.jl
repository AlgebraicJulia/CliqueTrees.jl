# scaling benchmark: time and width against n
#   MODES=serial,gpu,cpu  FAMS=lat2,lat3,corpus:lattice2d,...  OUT=file.tsv
const DIR = @__DIR__
include(joinpath(DIR, "Rounds.jl")); using .Rounds
include(joinpath(DIR, "evalutil.jl")); include(joinpath(DIR, "gens.jl"))
include(get(ENV, "CORPUS", joinpath(DIR, "..", "tncorpus")) * "/load.jl"); using .TNCorpus
modes = split(get(ENV, "MODES", "serial,gpu"), ",")
if "fcpu" in modes
    include(joinpath(DIR, "FRounds.jl")); using .FRounds
end
if "gpu" in modes
    using CUDA
    include(joinpath(DIR, "GRounds.jl")); using .GRounds
end
fams = split(ENV["FAMS"], ",")
nmax = parse(Int, get(ENV, "NMAX", "1100000"))
serialmax = parse(Int, get(ENV, "SERIALMAX", "100000000"))
relax = parse(Float64, get(ENV, "RELAX", "0.5"))
scores = Symbol.(split(get(ENV, "SCORES", "md,mf,mf1"), ","))
out = open(ENV["OUT"], "a")
host = gethostname(); nt = Threads.nthreads()
gpuname = "gpu" in modes ? CUDA.name(CUDA.device()) : "-"

function instances(fam)
    if fam == "lat2"
        Ls = parse.(Int, split(get(ENV, "LAT2", "16,23,32,45,64,90,128,181,256,362,512,724"), ","))
        return [("lat2_$L", 2L^2, () -> (g = lat2(L); (g, randw(nv(g))))) for L in Ls]
    elseif fam == "lat3"
        Ls = parse.(Int, split(get(ENV, "LAT3", "5,6,8,10,13,16,20,25"), ","))
        return [("lat3_$L", 3L^3, () -> (g = lat3(L); (g, randw(nv(g))))) for L in Ls]
    else
        f = split(fam, ":")[2]
        rows = sort(TNCorpus.select(; source = "ladder", family = String(f), nmax); by = r -> r.n)
        return [(r.name, r.n, () -> (inst = TNCorpus.load(r); (inst.graph, inst.weights))) for r in rows]
    end
end

serials = ["MMD" => MMD(), "AMF" => AMF(), "MF" => MF(), "MF1" => MF(; strategy = 1)]
function emit(fam, name, g, w, method, t, o, extra = "")
    wd, lc = o === nothing ? (NaN, NaN) : quality(w, g, o)
    println(out, join((host, gpuname, nt, fam, name, nv(g), ne(g), method, wd, lc, t, extra), '\t')); flush(out)
    println(rpad(name, 22), rpad(method, 14), " w=", round(wd, digits = 1), " t=", round(t, digits = 4), " ", extra); flush(stdout)
end
# warm-up
let g = lat2(12), w = randw(nv(g))
    for (_, a) in serials; permutation(w, g, a); end
    for sc in scores
        "cpu" in modes && rounds(w, g, RoundsAlg(; score = sc, select = :luby, relax))
        "fcpu" in modes && FRounds.frounds(w, g, RoundsAlg(; score = sc, select = :luby, relax))
        "gpu" in modes && GRounds.grounds(w, g, RoundsAlg(; score = sc, select = :luby, relax); poolfactor = 1.1)
    end
end
for fam in fams, (name, n, load) in instances(fam)
    n > nmax && continue
    g, w = load()
    reps = nv(g) < 200_000 ? 3 : 1
    if "serial" in modes && nv(g) <= serialmax
        for (an, a) in serials
            local o = nothing; t = Inf
            for _ in 1:reps
                t1 = @elapsed (o = try permutation(w, g, a)[1] catch e; nothing end)
                t = min(t, t1)
            end
            emit(fam, name, g, w, an, t, o)
        end
    end
    for sc in scores
        alg = RoundsAlg(; score = sc, select = :luby, relax, tailsel = parse(Int, get(ENV, "TAILSEL", "128")), tailmax = 10^9)
        if "gpu" in modes
            local o; t = Inf; tm = Dict{Symbol,Any}()
            GRounds.grounds(w, g, alg)    # untimed: device clocks and pool warm
            for _ in 1:max(reps, 2)
                tmi = Dict{Symbol,Any}()
                t1 = @elapsed (o = GRounds.grounds(w, g, alg; timing = tmi)[1])
                t1 < t && (t = t1; tm = tmi)
            end
            emit(fam, name, g, w, "gpu-$sc", t, o, join(["$k=$(v isa Float64 ? round(v, digits=4) : v)" for (k, v) in sort(collect(tm))], ";"))
        end
        if "fcpu" in modes
            local o; t = Inf; tm = Dict{Symbol,Any}()
            for _ in 1:reps
                tmi = Dict{Symbol,Any}()
                t1 = @elapsed (o = FRounds.frounds(w, g, alg; timing = tmi)[1])
                t1 < t && (t = t1; tm = tmi)
            end
            emit(fam, name, g, w, "fcpu-$sc", t, o, join(["$k=$(v isa Float64 ? round(v, digits=4) : v)" for (k, v) in sort(collect(tm))], ";"))
        end
        if "cpu" in modes
            local o; t = Inf; tm = Dict{Symbol,Any}()
            for _ in 1:reps
                tmi = Dict{Symbol,Any}()
                t1 = @elapsed (o = rounds(w, g, alg; timing = tmi)[1])
                t1 < t && (t = t1; tm = tmi)
            end
            emit(fam, name, g, w, "cpu-$sc", t, o, join(["$k=$(v isa Float64 ? round(v, digits=4) : v)" for (k, v) in sort(collect(tm))], ";"))
        end
    end
    GC.gc()
end
close(out)
