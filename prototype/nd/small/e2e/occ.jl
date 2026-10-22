# Throughput of the one-block-per-graph batch kernel vs block width and registers (blocks per SM), end to end,
# on large batches (B >> SMs). usage (code dir): julia --project=ENV -t8 occ.jl ; env: OUT, LEAVES, SETS, OTAGS
# Rows: config nw<warps>-r<maxregs> (0 = all a block can have), min of 3 calls, and a timed call for occ (resident blocks
# per SM), regs, lmem (local memory per thread: spills), kernel time. Fill compared with the first config.
const DIR = pwd()
using CUDA, Serialization, Graphs
include(joinpath(DIR, "Rounds.jl")); using .Rounds
include(joinpath(DIR, "GRounds.jl")); using .GRounds
const ALG = RoundsAlg(; score = :mf1, relax = 0.1)
const out = open(ENV["OUT"], "a")
fmt(x) = x isa Float64 ? round(x, digits = 5) : x
tstr(tm) = join(["$k=$(fmt(v))" for (k, v) in sort(collect(tm); by = first)], ";")
cfgs = [(16, 0), (8, 0), (8, 128), (4, 0), (4, 128), (4, 96), (2, 0), (2, 128), (32, 0)]
let g = [path_graph(40), path_graph(30)]
    for (nw, mr) in cfgs; GRounds.grounds_batch(g, ALG; nw, maxregs = mr); end
    GRounds.grounds_batch(g, ALG; nw = 16, twins = :sort)
end
lst(k) = [String(x) for x in split(get(ENV, k, ""), ",") if !isempty(x)]
for tag in lst("OTAGS")
    f = isfile(joinpath(ENV["LEAVES"], tag * ".jls")) ? joinpath(ENV["LEAVES"], tag * ".jls") : joinpath(ENV["SETS"], tag * ".jls")
    gs = filter(x -> nv(x) >= 2, deserialize(f)); B = length(gs)
    ref = nothing
    for (nw, mr) in cfgs
        try
            GRounds.grounds_batch(gs, ALG; nw, maxregs = mr)
            t = Inf; inf = nothing
            for r in 1:3
                GC.gc(false)
                t1 = @elapsed ((o, i1) = GRounds.grounds_batch(gs, ALG; nw, maxregs = mr))
                t1 < t && (t = t1; inf = i1)
            end
            ref === nothing && (ref = inf.nnz)
            tm = Dict{Symbol, Any}(); GRounds.grounds_batch(gs, ALG; nw, maxregs = mr, chunks = 1, timing = tm)
            println(out, join((gethostname(), tag, B, sum(nv.(gs)), "nw$nw-r$mr", sum(inf.nnz), fmt(t), "same=$(inf.nnz == ref);" * tstr(tm)), '\t')); flush(out)
            println(rpad(tag, 24), " nw=$nw maxregs=$mr  t=", round(t, digits = 4), " occ=", tm[:occ], " regs=", tm[:regs], " lmem=", tm[:lmem],
                " kernel=", round(tm[:kernel], digits = 4), inf.nnz == ref ? "" : " FILLDIFF"); flush(stdout)
        catch err
            println("FAILED $tag nw=$nw mr=$mr: ", first(sprint(showerror, err), 300)); flush(stdout)
        end
        GC.gc(); CUDA.reclaim()
    end
    # twin detection: sort (as grounds_csr) vs hash table (same representatives)
    for tw in (:sort, :hash)
        tm = Dict{Symbol, Any}(); GRounds.grounds_batch(gs, ALG; nw = 16, chunks = 1, twins = tw, timing = tm)
        println(rpad(tag, 24), " twins=$tw pre_twins=", round(1000 * tm[:pre_twins], digits = 2), " ms"); flush(stdout)
    end
end
close(out)
println("DONE")
