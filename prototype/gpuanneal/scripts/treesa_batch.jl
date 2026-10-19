# TreeSA batch: julia -t <thr> --project=treesa_env scripts/treesa_batch.jl <list: name tncpath per line> <niters comma> <outdir> <results tsv> [sc_target]
using OMEinsumContractionOrders, Printf
module HostIO
using Printf
include(joinpath(@__DIR__, "..", "src", "hostio.jl"))
end
using .HostIO: readnet, write_tree
lst, nis, odir, resf = ARGS[1], parse.(Int, split(ARGS[2], ',')), ARGS[3], ARGS[4]
sct = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : Inf
mkpath(odir)
score = isfinite(sct) ? ScoreFunction(sc_target = sct, sc_weight = 1.0) : ScoreFunction(sc_target = 1e4, sc_weight = 0.0)
function tossa(ne, nt)
    ssa = Tuple{Int,Int}[]; nxt = Ref(nt)
    go(x) = OMEinsumContractionOrders.isleaf(x) ? x.tensorindex - 1 : begin
        a = go(x.args[1]); b = go(x.args[2]); push!(ssa, (a, b)); id = nxt[]; nxt[] += 1; id
    end
    go(ne); ssa
end
nthr = Threads.nthreads()
warm = false
for line in eachline(lst)
    isempty(strip(line)) && continue
    name, path = split(line)
    net = readnet(path; name)
    code = OMEinsumContractionOrders.EinCode([collect(t) for t in net.tens], collect(net.open))
    sizes = Dict(i => net.dims[i] for i in 1:net.n)
    global warm
    if !warm; optimize_code(code, sizes, TreeSA(ntrials = 1, niters = 1, score = score)); warm = true; end
    for ni in nis
        tag = "$(name)__treesa_t$(nthr)_ni$(ni)" * (isfinite(sct) ? "_W$(Int(sct))" : "")
        isfile(joinpath(odir, tag * ".tree")) && continue
        t = @elapsed (opt = optimize_code(code, sizes, TreeSA(ntrials = nthr, niters = ni, score = score)))
        sl = Int[]
        if isfinite(sct)
            t += @elapsed (s = slice_code(opt, sizes, TreeSASlicer(score = ScoreFunction(sc_target = sct), ntrials = nthr, niters = ni)))
            sl = collect(Int, s.slicing); opt = s.eins
        end
        cc = contraction_complexity(opt, sizes)
        write_tree(joinpath(odir, tag * ".tree"), net.nt, tossa(opt, net.nt), sl; meta = Dict("t" => @sprintf("%.3f", t),
            "treesa_tc" => @sprintf("%.6f", cc.tc), "niters" => string(ni), "threads" => string(nthr)))
        open(resf, "a") do io
            println(io, join((name, "treesa", nthr, ni, isfinite(sct) ? sct : "Inf", @sprintf("%.3f", t), @sprintf("%.4f", cc.tc), cc.sc, length(sl)), '\t'))
        end
        @printf("%s ni=%d %.2fs tc %.3f\n", name, ni, t, cc.tc); flush(stdout)
    end
end
