# B200 sweep of the small-graph modes of GPU-MF1 (GSmall.jl, SMALL_NOTES.md).
# usage: julia --project=ENV -t2 sgpu.jl
# env: OUT (tsv), LEAVES (dir for the ND leaves, read by sser.jl), CORPUS, DATA (':'-separated dirs with NAME.bin),
#      SINGLE (comma list of graphs), NDG (comma list of graphs for the ND-leaf batches), LMAX (leaf sizes),
#      RACE (comma list), PROFG (comma list: phase profiles)
# Fill config everywhere: RoundsAlg(score=:mf1, relax=0.1), unit weights; nnz(L), flops, tw are the device
# counters (checked against a CliqueTrees recount on the replay runs); every order is checked to be a permutation.
const DIR = joinpath(@__DIR__, "..", "..")
using CUDA, Serialization, SparseArrays, Random, Metis
include(joinpath(DIR, "Rounds.jl")); using .Rounds
include(joinpath(DIR, "evalutil.jl")); include(joinpath(DIR, "gens.jl"))
include(ENV["CORPUS"] * "/load.jl"); using .TNCorpus
include(joinpath(DIR, "GRounds.jl")); using .GRounds

const host = gethostname()
const gpuname = CUDA.name(CUDA.device())
const dirs = split(get(ENV, "DATA", ""), ":")
const nsm = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
const out = open(ENV["OUT"], "a")

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

fmt(x) = x isa Float64 ? round(x, digits = 5) : x
tstr(tm) = join(["$k=$(fmt(v))" for (k, v) in sort(collect(tm); by = first)], ";")
row(part, name, n, m, cfg, nnz, fl, tw, t, extra) =
    (println(out, join((host, gpuname, part, name, n, m, cfg, nnz, fl, tw, fmt(t), extra), '\t')); flush(out))

function fillq(g, o)
    cw = colweights(ones(Int, nv(g)), g, o)
    return sum(cw), sum(Float64(c)^2 for c in cw), maximum(cw) - 1
end

const ALG = RoundsAlg(; score = :mf1, relax = 0.1)

# warm-up (compilation of every kernel configuration)
let g = lat2(30), w = ones(nv(g))
    for kw in ((;), (mode = :block, nw = 8), (mode = :block, nw = 16), (mode = :grid, G = 2), (mode = :auto,))
        GRounds.grounds(w, g, ALG; kw...); GRounds.grounds(w, g, ALG; timing = Dict{Symbol, Any}(), prof = true, kw...)
    end
    for nw in (4, 8, 16); GRounds.grounds_batch([g, lat2(10)], ALG; nw); end
    GRounds.grounds_race(w, g, ALG; mode = :auto)
end
println(gpuname, " nsm=", nsm, " threads=", Threads.nthreads()); flush(stdout)

# ---- single graphs: every mode, min of 3 runs after a warm run
function single(name)
    g = try loadbin(String(name)) catch err; println("LOAD FAILED $name: $err"); return end
    n = nv(g); m = ne(g); w = ones(n)
    Gs = unique(filter(G -> G >= 2, [16, 32, 64, nsm]))
    cfgs = Any[("replay", (;)), ("block8", (mode = :block, nw = 8)), ("block16", (mode = :block, nw = 16))]
    for G in Gs; push!(cfgs, ("grid$G", (mode = :grid, G = G))); end
    push!(cfgs, ("auto", (mode = :auto,)))
    ref = nothing
    for (lab, kw) in cfgs
        try
            GRounds.grounds(w, g, ALG; kw...)
            t = Inf; tm = Dict{Symbol, Any}(); o = nothing
            for r in 1:3
                tmi = Dict{Symbol, Any}()
                t1 = @elapsed (oi = GRounds.grounds(w, g, ALG; timing = tmi, kw...)[1])
                t1 < t && (t = t1; tm = tmi; o = oi)
                t1 > 5 && break
            end
            @assert length(o) == n && isperm(o) "invalid permutation"
            q = (tm[:nnz], Float64(tm[:flops]), tm[:tw])
            if ref === nothing
                ref = q
                rc = fillq(g, o)
                rc == (q[1], q[2], q[3]) || println("COUNTER MISMATCH $name: device $q recount $rc")
            end
            q == ref || println("FILL DIFF $name $lab: $q vs replay $ref")
            row("single", name, n, m, lab, q[1], q[2], q[3], t, tstr(tm))
            println(rpad(name, 22), rpad(lab, 9), " n=", n, " t=", round(t, digits = 4), " rounds=", tm[:rounds], " nnz=", q[1],
                q == ref ? "" : " DIFF"); flush(stdout)
        catch err
            println("FAILED $name $lab: ", first(sprint(showerror, err), 400)); flush(stdout)
            row("single", name, n, m, lab, "error", "", "", "", "")
        end
        GC.gc(false)
    end
end

# ---- phase profiles (cycles of block 1 per round and phase)
function profile(name)
    g = loadbin(String(name)); n = nv(g); w = ones(n)
    for (lab, kw) in (("block16", (mode = :block, nw = 16)), ("grid16", (mode = :grid, G = 16)), ("grid$nsm", (mode = :grid, G = nsm)))
        GRounds.grounds(w, g, ALG; kw...)
        tm = Dict{Symbol, Any}()
        t = @elapsed GRounds.grounds(w, g, ALG; timing = tm, prof = true, kw...)
        r = tm[:rounds]
        s = join(["$k=$(round(tm[Symbol("prof_", k)] / r / 1000, digits = 2))" for k in GRounds.PROF_NAMES[1:10]], ";")
        row("prof", name, n, ne(g), lab, tm[:nnz], tm[:flops], tm[:tw], t, "rounds=$r;kcyc_round=$(round(tm[:prof_total] / r / 1000, digits = 2));" * s)
        println(rpad(name, 22), rpad(lab, 9), " kcyc/round=", round(tm[:prof_total] / r / 1000, digits = 1), " ", s); flush(stdout)
    end
end

# ---- ND leaves: recursive METIS vertex separators; a part becomes a leaf of size bound L
# once it has at most L vertices (parts of all bounds come from the same recursion)
function ndleaves(g, Ls)
    leaves = Dict(L => Vector{Vector{Int}}() for L in Ls)
    Lmax = maximum(Ls); Lmin = minimum(Ls)
    stack = Any[(collect(1:nv(g)), false)]
    while !isempty(stack)
        vs, _ = pop!(stack)
        k = length(vs)
        for L in Ls
            # leaf of bound L: at most L vertices, and its parent had more than L (emitted once)
            k <= L && push!(leaves[L], vs)
        end
        k <= Lmin && continue
        sg, vmap = induced_subgraph(g, vs)
        part = Metis.separator(sg)
        a = vmap[part .== 1]; b = vmap[part .== 2]
        (isempty(a) || isempty(b)) && continue
        push!(stack, (a, true)); push!(stack, (b, true))
    end
    # keep, for each bound, only the maximal parts (a part below L whose parent was also below L is dropped)
    for L in Ls
        sets = leaves[L]
        sort!(sets; by = length, rev = true)
        mark = falses(nv(g)); keep = Vector{Vector{Int}}()
        for vs in sets
            any(mark[v] for v in vs) && continue
            mark[vs] .= true; push!(keep, vs)
        end
        leaves[L] = keep
    end
    return leaves
end

function batches(name, Ls)
    g = loadbin(String(name)); n0 = nv(g)
    t = @elapsed lv = ndleaves(g, Ls)
    println(rpad(name, 22), " ND leaves in ", round(t, digits = 1), " s: ", join(["L=$L: $(length(lv[L]))" for L in Ls], ", ")); flush(stdout)
    for L in Ls
        gs = [induced_subgraph(g, vs)[1] for vs in lv[L]]
        filter!(x -> nv(x) >= 2, gs)
        sz = nv.(gs)
        tag = "$(name)_L$L"
        serialize(joinpath(ENV["LEAVES"], tag * ".jls"), gs)
        nl = length(gs); ntot = sum(sz)
        desc = "leaves=$nl;nmin=$(minimum(sz));nmed=$(sort(sz)[cld(nl, 2)]);nmax=$(maximum(sz));nsum=$ntot;n0=$n0"
        for nw in (4, 8, 16)
            try
                GRounds.grounds_batch(gs, ALG; nw)
                tb = Inf; tm = Dict{Symbol, Any}(); res = nothing
                for r in 1:3
                    tmi = Dict{Symbol, Any}()
                    t1 = @elapsed (ri = GRounds.grounds_batch(gs, ALG; nw, timing = tmi))
                    t1 < tb && (tb = t1; tm = tmi; res = ri)
                end
                orders, info = res
                @assert all(length(orders[i]) == sz[i] && isperm(orders[i]) for i in 1:nl)
                row("batch", tag, ntot, sum(ne.(gs)), "batch-nw$nw", sum(info.nnz), sum(Float64.(info.flops)), maximum(info.tw), tb,
                    desc * ";gps=$(round(nl / tb, digits = 1));launches=$(info.launches);grows=$(info.grows);" * tstr(tm))
                println(rpad(tag, 22), " batch nw=$nw: ", nl, " leaves in ", round(tb, digits = 4), " s (", round(nl / tb, digits = 1),
                    " graphs/s), gpu=", round(tm[:gpu], digits = 4), " launches=", info.launches, " nnz=", sum(info.nnz)); flush(stdout)
                if nw == 8
                    # exactness: the batch gives the single-graph fill (first 20 leaves, replay driver)
                    nb = 0
                    for i in 1:min(nl, 20)
                        tmi = Dict{Symbol, Any}()
                        GRounds.grounds(ones(sz[i]), gs[i], ALG; timing = tmi)
                        (tmi[:nnz], tmi[:tw]) == (info.nnz[i], info.tw[i]) || (nb += 1)
                    end
                    println("   batch vs single on $(min(nl, 20)) leaves: $(nb) differ"); flush(stdout)
                end
            catch err
                println("FAILED batch $tag nw=$nw: ", first(sprint(showerror, err), 400)); flush(stdout)
            end
            GC.gc(); CUDA.reclaim()
        end
        # the same leaves one after another (auto mode per leaf)
        GRounds.grounds(ones(sz[1]), gs[1], ALG; mode = :auto)
        ts = @elapsed begin
            snnz = 0
            for i in 1:nl
                tmi = Dict{Symbol, Any}()
                GRounds.grounds(ones(sz[i]), gs[i], ALG; mode = :auto, timing = tmi)
                snnz += tmi[:nnz]
            end
        end
        row("batch", tag, ntot, sum(ne.(gs)), "seq-auto", snnz, "", "", ts, desc * ";gps=$(round(nl / ts, digits = 1))")
        println(rpad(tag, 22), " sequential auto: ", round(ts, digits = 3), " s"); flush(stdout)
        GC.gc(); CUDA.reclaim()
    end
end

# ---- CPU/GPU race (the serial MF1 here is CliqueTrees in this env)
function race(name)
    g = loadbin(String(name)); n = nv(g); w = ones(n)
    GRounds.grounds_race(w, g, ALG; mode = :auto)
    t = @elapsed (res = GRounds.grounds_race(w, g, ALG; mode = :auto))
    o, ip, info = res
    @assert length(o) == n && isperm(o)
    r = info.winner === :gpu ? info.gpu : info.cpu
    row("race", name, n, ne(g), "race-auto", r.nnz, r.flops, r.tw, t,
        "winner=$(info.winner);tcpu=$(fmt(info.tcpu));tgpu=$(fmt(info.tgpu));cpu_nnz=$(info.cpu.nnz);gpu_nnz=$(info.gpu.nnz)")
    println(rpad(name, 22), " race: ", info.winner, " t=", round(t, digits = 4), " tcpu=", round(info.tcpu, digits = 4), " tgpu=",
        round(info.tgpu, digits = 4)); flush(stdout)
end

lst(k) = [String(x) for x in split(get(ENV, k, ""), ",") if !isempty(x)]
for name in lst("PROFG"); profile(name); end
for name in lst("SINGLE"); single(name); end
Ls = parse.(Int, lst("LMAX"))
for name in lst("NDG"); batches(name, Ls); end
for name in lst("RACE"); race(name); end
close(out)
println("DONE")
