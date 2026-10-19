# Batch GPU runs in one process.
#   julia --project=env scripts/batch.jl <tasks file> <results tsv> <tree dir>
# task line (whitespace separated): name tncpath seedtag seedfiles(comma) k=v;k=v (runpipe opts; "-" for none)
# Appends one row per task: name seedtag opts device nch nth T Wt rseed  best@{0.3,1,3,10,30,100,300} final_log2C W nsliced seed_log2C ttot
# and writes <tree dir>/<name>__<seedtag>__<optstag>.{T*,final}.tree. Tasks already in the results file are skipped.
const HERE = @__DIR__
include(joinpath(HERE, "..", "src", "BlockAnneal.jl"))
using .BlockAnneal, CUDA, Printf
const BA = BlockAnneal
include(joinpath(HERE, "..", "src", "Pipe.jl"))

tasks, resf, tdir = ARGS[1], ARGS[2], ARGS[3]
mkpath(tdir)
done = Set{String}()
isfile(resf) && for l in eachline(resf)
    f = split(l, '\t'); length(f) >= 3 && push!(done, join(f[1:3], '|'))
end
marks = [0.3, 1, 3, 10, 30, 100, 300, 1000]
compiled = Ref(false)
for line in eachline(tasks)
    (isempty(strip(line)) || startswith(line, "#")) && continue
    name, path, stag, sfiles, ostr = split(line)
    key = join((name, stag, ostr), '|')
    key in done && continue
    opts = Dict{String,String}(split(x, '='; limit = 2) |> y -> (y[1] => y[2]) for x in split(ostr, ';') if occursin('=', x))
    net = BA.readnet(path; name)
    seeds = [(BA.tree_from_ssa(net, BA.read_tree(f)[2]), BA.read_tree(f)[3]) for f in split(sfiles, ',')]
    s0 = minimum(BA.hostcost(net, CHs, BA.slicemask(net, sl)).log2C for (CHs, sl) in seeds)
    if !compiled[]
        let A = BA.BAnnealer(net, seeds; nch = 2); BA.temper!(A; time = 0.01, rounds = 4); end
        let A = BA.BAnnealer(net, seeds; nch = 2, shared = false); BA.temper!(A; time = 0.01, rounds = 4, Wt = 20.0, nslice = 2, pheur = 0.5); end
        compiled[] = true
    end
    otag = replace(ostr, ';' => '_', '=' => "")
    r = try
        runpipe(net, seeds, opts; out = joinpath(tdir, "$(name)__$(stag)__$(otag)"))
    catch e
        @warn "task failed" name stag ostr exception = (e, catch_backtrace())
        continue
    end
    E, CH, sl, h = r.best
    row = (name, stag, ostr, CUDA.name(CUDA.device()), r.info.nch, r.info.nth, get(opts, "time", "30"), get(opts, "Wt", "Inf"),
           get(opts, "rseed", "1"), (@sprintf("%.4f", traceat(r.trace, m)) for m in marks)..., @sprintf("%.4f", h.log2C),
           h.W, length(sl), @sprintf("%.4f", s0), @sprintf("%.2f", r.ttot))
    open(resf, "a") do io
        println(io, join(row, '\t'))
    end
    @printf("%s %s %s: %.4f (seed %.4f) W %.1f sliced %d [%d chains]\n", name, stag, ostr, h.log2C, s0, h.W, length(sl), r.info.nch)
    flush(stdout)
end
