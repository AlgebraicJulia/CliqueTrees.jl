# End-to-end GPU batch MF1 (grounds_batch, GSmall.jl after the host-path work) vs the v2 host path, on the B200.
# usage: julia --project=ENV -t8 e2e_gpu.jl            (run from the code dir: GRounds.jl + GSmall.jl new, old/ = v2)
# env: OUT (tsv), LEAVES (ND leaf sets NAME_LL.jls of sgpu.jl), SETS (dir: the sweep sets written here, read by e2e_cpu.jl),
#      DATA (':'-separated dirs with NAME.bin), CORPUS, LTAGS (leaf sets), SINGLE (full graphs 30k-100k), REP (label of this pass),
#      SKIPOLD=1 (no v2 runs), STOPFILE (keeper mode: repeat until this file exists)
# Every row: wall time of the whole call (host graphs in, host permutations out), min of 3 after a warm call;
# the stage breakdown is a 4th, timed run (it syncs the GPU between stages). Fill: per-graph nnz/flops/tw of the new
# path must equal the v2 path (printed as FILLDIFF otherwise); every order is checked to be a permutation.
const DIR = pwd()
using CUDA, Serialization, SparseArrays, Random, Metis, Graphs
include(joinpath(DIR, "Rounds.jl")); using .Rounds
include(joinpath(DIR, "evalutil.jl")); include(joinpath(DIR, "gens.jl"))
include(ENV["CORPUS"] * "/load.jl"); using .TNCorpus
include(joinpath(DIR, "GRounds.jl")); using .GRounds
const OLD = get(ENV, "SKIPOLD", "0") != "1"
OLD && (include(joinpath(DIR, "old", "GRounds.jl")); using .GRoundsOld)
const host = gethostname()
const nsm = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
const out = open(ENV["OUT"], "a")
const REP = get(ENV, "REP", "a")
const ALG = RoundsAlg(; score = :mf1, relax = 0.1)
const dirs = split(get(ENV, "DATA", ""), ":")
lst(k) = [String(x) for x in split(get(ENV, k, ""), ",") if !isempty(x)]
fmt(x) = x isa Float64 ? round(x, digits = 5) : x
tstr(tm) = join(["$k=$(fmt(v))" for (k, v) in sort(collect(tm); by = first)], ";")
row(part, name, B, n, m, cfg, nnz, t, extra) =
    (println(out, join((host, REP, part, name, B, n, m, cfg, nnz, fmt(t), extra), '\t')); flush(out))

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

# warm-up: every configuration compiled
let g = [lat2(30), lat2(10), path_graph(40)]
    for nw in (8, 16, 32)
        GRounds.grounds_batch(g, ALG; nw); GRounds.grounds_batch(g, ALG; nw, chunks = 2, timing = Dict{Symbol, Any}())
        OLD && GRoundsOld.grounds_batch(g, ALG; nw)
    end
    w = ones(nv(g[1]))
    for kw in ((;), (mode = :auto,), (mode = :grid, G = 4), (mode = :grid, G = nsm))
        GRounds.grounds(w, g[1], ALG; kw...)
    end
end
println(CUDA.name(CUDA.device()), " nsm=", nsm, " threads=", Threads.nthreads(), " auto_nw=", GRounds.auto_nw()); flush(stdout)

function best3(f)
    f(); t = Inf; r = nothing
    for i in 1:3
        GC.gc(false)
        t1 = @elapsed (ri = f())
        t1 < t && (t = t1; r = ri)
    end
    return t, r
end

function batchset(tag, gs)
    B = length(gs); n = sum(nv.(gs)); m = sum(ne.(gs)); sz = sort(nv.(gs))
    desc = "nmin=$(sz[1]);nmed=$(sz[cld(B, 2)]);nmax=$(sz[end])"
    ref = nothing
    if OLD
        t, (o, inf) = best3(() -> GRoundsOld.grounds_batch(gs, ALG; nw = 16))
        tm = Dict{Symbol, Any}(); GRoundsOld.grounds_batch(gs, ALG; nw = 16, timing = tm)
        ref = (inf.nnz, inf.flops, inf.tw)
        row("batch", tag, B, n, m, "v2-nw16", sum(inf.nnz), t, desc * ";" * tstr(tm))
        println(rpad(tag, 26), " B=", B, " v2 nw16 t=", round(t, digits = 4)); flush(stdout)
    end
    cfgs = get(ENV, "CFGS", "v4") == "v4" ?
        Any[("new-nw16", (nw = 16,)), ("new-nw8", (nw = 8,)), ("new-nw16-c1", (nw = 16, chunks = 1)),
            ("new-nw16-min", (nw = 16, arena = :min, chunks = 1))] :
        # v5: defaults (block shape from batch_cfg, chunks from the volume, device-side growth), and ablations
        Any[("new", (;)), ("new-c1", (chunks = 1,)), ("new-nw16", (nw = 16,)), ("new-nosp", (spare = 0,)),
            ("new-min-c1", (nw = 16, arena = :min, chunks = 1, spare = 0))]
    get(ENV, "CFGS", "v4") == "v4" && sz[end] > 20000 && push!(cfgs, ("new-nw32", (nw = 32,)))
    for (lab, kw) in cfgs
        try
            t, (o, inf) = best3(() -> GRounds.grounds_batch(gs, ALG; kw...))
            @assert all(length(o[b]) == nv(gs[b]) && isperm(o[b]) for b in 1:B) "invalid permutation"
            q = (inf.nnz, inf.flops, inf.tw)
            if ref === nothing
                ref = q
            elseif q != ref
                println("FILLDIFF $tag $lab: ", count(q[1] .!= ref[1]), " graphs differ in nnz"); flush(stdout)
            end
            tm = Dict{Symbol, Any}(); GRounds.grounds_batch(gs, ALG; timing = tm, kw...)
            row("batch", tag, B, n, m, lab, sum(inf.nnz), t, desc * ";same=$(q == ref);" * tstr(tm))
            println(rpad(tag, 26), " ", rpad(lab, 14), " t=", round(t, digits = 4), " nw=", get(tm, :nw, "?"), "/", get(tm, :maxregs, "?"),
                " launches=", inf.launches, " grows=", inf.grows, " dgrows=", get(tm, :dgrows, 0),
                " pre=", round(tm[:pre], digits = 4), " gpu=", round(tm[:gpu], digits = 4), " post=", round(tm[:cpu_post], digits = 4),
                q == ref ? "" : " FILLDIFF"); flush(stdout)
        catch err
            println("FAILED $tag $lab: ", first(sprint(showerror, err), 300)); flush(stdout)
        end
        GC.gc(); CUDA.reclaim()
    end
    # L50k-type sets (fewer graphs than SMs): every graph its own cooperative grid of nsm/B blocks, on its own
    # stream (one task per graph), all at once
    if B < nsm && sz[end] > 20000 && get(ENV, "CFGS", "v4") == "v4"
        G = max(1, nsm ÷ B)
        f() = (ts = [Threads.@spawn GRounds.grounds(ones(nv(g)), g, ALG; mode = :grid, G = $G) for g in gs]; fetch.(ts))
        try
            t, _ = best3(f)
            row("batch", tag, B, n, m, "grid-streams-G$G", "", t, desc)
            println(rpad(tag, 26), " grid streams G=$G t=", round(t, digits = 4)); flush(stdout)
        catch err
            println("FAILED $tag grid-streams: ", first(sprint(showerror, err), 300)); flush(stdout)
        end
    end
end

# ND leaves (sgpu.jl ndleaves): bound 1000 for the sweep pool
function ndleaves(g, L)
    keep = Vector{Vector{Int}}()
    stack = Any[collect(1:nv(g))]
    while !isempty(stack)
        vs = pop!(stack)
        if length(vs) <= L
            push!(keep, vs); continue
        end
        sg, vmap = induced_subgraph(g, vs)
        part = Metis.separator(sg)
        a = vmap[part .== 1]; b = vmap[part .== 2]
        (isempty(a) || isempty(b)) && (push!(keep, vs); continue)
        push!(stack, a); push!(stack, b)
    end
    return keep
end

function sweepsets()
    sdir = ENV["SETS"]; mkpath(sdir)
    pools = Dict{Int, Vector{Any}}()
    for L in (2000, 10000, 50000)
        pools[L] = Any[]
        for g in ("ecology2", "thermal2", "G3_circuit", "lat2_724", "audikw_1")
            f = joinpath(ENV["LEAVES"], "$(g)_L$L.jls")
            isfile(f) && append!(pools[L], filter(x -> nv(x) >= L ÷ 4, deserialize(f)))
        end
    end
    pools[1000] = Any[]
    for name in (get(ENV, "NOND", "0") == "1" ? () : ("ecology2", "thermal2", "G3_circuit"))
        g = loadbin(name)
        lv = ndleaves(g, 1000)
        append!(pools[1000], filter(x -> nv(x) >= 250, [induced_subgraph(g, vs)[1] for vs in lv]))
    end
    rng = MersenneTwister(1)
    counts = Dict(1000 => (10, 30, 100, 300, 1000, 3000), 2000 => (10, 30, 100, 300, 1000, 3000),
        10000 => (3, 10, 30, 100, 300, 800), 50000 => (3, 10, 30, 100))
    tags = String[]
    cmax = parse(Int, get(ENV, "SWEEPMAX", "100000"))
    for L in (1000, 2000, 10000, 50000), c in counts[L]
        p = pools[L]
        (isempty(p) || c > cmax) && continue
        idx = c <= length(p) ? randperm(rng, length(p))[1:c] : rand(rng, 1:length(p), c)
        tag = "sweep_L$(L)_x$c"
        serialize(joinpath(sdir, tag * ".jls"), p[idx]); push!(tags, tag)
    end
    println("pools: ", join(["L$L: $(length(pools[L]))" for L in sort(collect(keys(pools)))], ", ")); flush(stdout)
    return tags
end

function single(name)
    g = try loadbin(name) catch err; println("LOAD FAILED $name"); return end
    n = nv(g); w = ones(n)
    for (lab, kw) in (("replay", (;)), ("auto", (mode = :auto,)), ("grid$nsm", (mode = :grid, G = nsm)))
        try
            t, (o, _) = best3(() -> GRounds.grounds(w, g, ALG; kw...))
            @assert isperm(o)
            tm = Dict{Symbol, Any}(); GRounds.grounds(w, g, ALG; timing = tm, kw...)
            row("single", name, 1, n, ne(g), lab, tm[:nnz], t, tstr(tm))
            println(rpad(name, 22), rpad(lab, 9), " t=", round(t, digits = 4), " csr=", round(tm[:csr], digits = 4), " pre=",
                round(tm[:pre], digits = 4), " gpu=", round(tm[:gpu], digits = 4), " post=", round(tm[:cpu_post], digits = 4)); flush(stdout)
        catch err
            println("FAILED $name $lab: ", first(sprint(showerror, err), 300)); flush(stdout)
        end
    end
end

stopf = get(ENV, "STOPFILE", "")
tags = isempty(stopf) && get(ENV, "SETSREADY", "0") != "1" ? sweepsets() :
    [replace(f, ".jls" => "") for f in readdir(ENV["SETS"]) if endswith(f, ".jls")]
while true
    for tag in lst("LTAGS")
        batchset(tag, filter(x -> nv(x) >= 2, deserialize(joinpath(ENV["LEAVES"], tag * ".jls"))))
        !isempty(stopf) && isfile(stopf) && break
    end
    for tag in tags
        batchset(tag, deserialize(joinpath(ENV["SETS"], tag * ".jls")))
        !isempty(stopf) && isfile(stopf) && break
    end
    isempty(stopf) && for name in lst("SINGLE"); single(name); end
    (isempty(stopf) || isfile(stopf)) && break
end
close(out)
println("DONE")
