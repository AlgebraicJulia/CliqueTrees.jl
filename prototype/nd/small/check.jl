# exactness of the persistent modes (GSmall.jl) and of the context refactor of
# GRounds.jl: same fill as GRounds before the change (small/GRounds_pre_small.jl)
# usage: julia --project=../genv --heap-size-hint=2G check.jl [QUICK=1] name...
include(joinpath(@__DIR__, "..", "perf", "pload.jl"))
using CUDA
include(joinpath(RDIR, "GRounds.jl")); using .GRounds
include(joinpath(@__DIR__, "GRounds_pre_small.jl")); using .GRoundsPS
function inst(name)
    if startswith(name, "lat")
        g, _ = loadg(name); return g, randw(nv(g))
    elseif isfile(joinpath(FDATA, name * ".mtx"))
        g, _ = loadg(name); return g, ones(nv(g))
    else
        i = TNCorpus.load(name); return i.graph, i.weights
    end
end
quick = get(ENV, "QUICK", "0") == "1"
modes = split(get(ENV, "MODES", "replay,block8,block16,grid1,grid3,batch"), ",")
q(g, o, w) = (fillq(g, o), quality(w, g, o))
nsame = 0; nbad = 0
function runmode(m, w, g, alg)
    tm = Dict{Symbol, Any}()
    if m == "replay"
        o = GRounds.grounds(w, g, alg; timing = tm)[1]
    elseif startswith(m, "block")
        o = GRounds.grounds(w, g, alg; mode = :block, nw = parse(Int, m[6:end]), timing = tm)[1]
    elseif startswith(m, "grid")
        o = GRounds.grounds(w, g, alg; mode = :grid, G = parse(Int, m[5:end]), timing = tm)[1]
    elseif m == "auto"
        o = GRounds.grounds(w, g, alg; mode = :auto, timing = tm)[1]
    end
    return o, tm
end
allg = Any[]
for name in ARGS
    g, w = inst(name)
    push!(allg, (name, g, w))
    cfgs = quick ? [(:mf1, 0.1, true, ones(nv(g))), (:mf1, 0.5, true, w)] :
        [(sc, rx, comp, ww) for sc in (:md, :mf1) for rx in (0.0, 0.1, 0.5) for comp in (true, false) for ww in (w, ones(nv(g)))]
    for (sc, rx, comp, ww) in cfgs
        alg = RoundsAlg(; score = sc, relax = rx, compress = comp)
        b = GRoundsPS.grounds(ww, g, alg)[1]
        qb = q(g, b, ww)
        for m in modes
            m in ("batch", "batchdev") && continue
            a, tm = runmode(m, ww, g, alg)
            @assert length(a) == nv(g) && isperm(a) "$name $m: not a permutation"
            qa = q(g, a, ww)
            # device counters agree with the CliqueTrees recount (unit weights)
            ok = qa == qb && (tm[:nnz], tm[:tw]) == (qa[1][1], qa[1][3])
            if ok
                global nsame += 1
            else
                global nbad += 1
                println(rpad(name, 22), " $m $sc relax=$rx compress=$comp DIFF $(qa) vs $(qb) dev=$((tm[:nnz], tm[:flops], tm[:tw]))")
            end
        end
    end
    println(rpad(name, 22), " done (same=$nsame bad=$nbad)"); flush(stdout)
end
if "batch" in modes && !isempty(allg)
    for (sc, rx) in ((:mf1, 0.1), (:md, 0.0), (:mf1, 0.5))
        alg = RoundsAlg(; score = sc, relax = rx)
        gs = [x[2] for x in allg]; ws = [x[3] for x in allg]
        # small arenas: exercises the per-graph growth and relaunch path
        # (arena = :min: growth on demand from these start sizes; :auto: sized up front; chunks > 1: pipelined streams)
        # (spare = 0: growth by host relaunches only; -1: device-side growth from a spare arena; 3000: a spare arena
        # that runs out, so both)
        for (pm, km, oc, ps, ar, ch, sp) in ((1024, 2048, 4096, 8, :min, 1, 0), (16, 16, 16, 1, :min, 1, 0), (16, 16, 16, 1, :min, 3, 0),
                (16, 16, 16, 1, :min, 1, -1), (16, 16, 16, 1, :min, 2, 3000), (1024, 2048, 4096, 8, :auto, 3, -1))
            orders, info = GRounds.grounds_batch(gs, alg; weights = ws, poolmin = pm, kmin0 = km, ocap0 = oc, poolfactor = 1.0,
                poolsmall = ps, dgroup = 16, arena = ar, chunks = ch, spare = sp)
            println("batch $sc relax=$rx arena=$km $ar chunks=$ch spare=$sp: launches=$(info.launches) grows=$(info.grows) dgrows=$(info.dgrows)")
            for (i, (name, g, w)) in enumerate(allg)
                b = GRoundsPS.grounds(w, g, alg)[1]
                a = orders[i]
                @assert length(a) == nv(g) && isperm(a)
                qa = q(g, a, w); qb = q(g, b, w)
                if qa == qb && info.nnz[i] == qa[1][1]
                    global nsame += 1
                else
                    global nbad += 1
                    println(rpad(name, 22), " batch $sc relax=$rx arena=$km $ar chunks=$ch spare=$sp DIFF $(qa) vs $(qb)")
                end
            end
        end
    end
    println("batch done")
end
# device-input batch (grounds_batch_dev, unit weights): same fill as the reference per graph
if "batchdev" in modes && !isempty(allg)
    for (sc, rx) in ((:mf1, 0.1), (:md, 0.0), (:mf1, 0.5))
        alg = RoundsAlg(; score = sc, relax = rx)
        gs = [x[2] for x in allg]
        cs = [GRounds.csr(g) for g in gs]
        loff = Int32[1]; up = Int32[1]; ua = Int32[]
        for (p, t) in cs
            push!(loff, loff[end] + length(p) - 1)
            for i in 1:(length(p) - 1); push!(up, up[end] + Int32(p[i + 1] - p[i])); end
            append!(ua, Int32.(t))
        end
        dord, info = GRounds.grounds_batch_dev(CuVector{Int32}(up), CuVector{Int32}(ua), loff, alg)
        oh = Array(dord)
        for (i, (name, g, _)) in enumerate(allg)
            w = ones(nv(g))
            a = Int.(oh[loff[i]:(loff[i + 1] - 1)])
            @assert length(a) == nv(g) && isperm(a)
            b = GRoundsPS.grounds(w, g, alg)[1]
            qa = q(g, a, w); qb = q(g, b, w)
            if qa == qb && info.nnz[i] == qa[1][1]
                global nsame += 1
            else
                global nbad += 1
                println(rpad(name, 22), " batchdev $sc relax=$rx DIFF $(qa) vs $(qb)")
            end
        end
    end
    let gs = [SimpleGraph(1), path_graph(3), complete_graph(40), star_graph(100), SimpleGraph(2), complete_graph(300)]
        cs = [GRounds.csr(g) for g in gs]
        loff = Int32[1]; up = Int32[1]; ua = Int32[]
        for (p, t) in cs
            push!(loff, loff[end] + length(p) - 1)
            for i in 1:(length(p) - 1); push!(up, up[end] + Int32(p[i + 1] - p[i])); end
            append!(ua, Int32.(t))
        end
        dord, _ = GRounds.grounds_batch_dev(CuVector{Int32}(up), CuVector{Int32}(max.(ua, Int32(1))), loff, RoundsAlg(; score = :mf1, relax = 0.1))
        oh = Array(dord)
        @assert all(isperm(oh[loff[i]:(loff[i + 1] - 1)]) for i in eachindex(gs))
    end
    println("batchdev done (same=$nsame bad=$nbad)")
end
for g in (SimpleGraph(1), SimpleGraph(5), path_graph(3), complete_graph(40), star_graph(100), complete_graph(300))
    for m in modes
        m in ("batch", "batchdev") && continue
        o, _ = runmode(m, ones(nv(g)), g, RoundsAlg(; score = :mf1, relax = 0.1))
        @assert length(o) == nv(g) && isperm(o)
    end
end
let gs = [SimpleGraph(0), SimpleGraph(1), path_graph(3), complete_graph(40), star_graph(100), complete_graph(300)]
    for ch in (1, 2, 4)
        orders, _ = GRounds.grounds_batch(gs, RoundsAlg(; score = :mf1, relax = 0.1); chunks = ch)
        @assert all(length(orders[i]) == nv(gs[i]) && isperm(orders[i]) for i in eachindex(gs))
    end
    # self loops (packed again without them), CSR input, concurrent calls from several tasks
    gl = path_graph(30); add_edge!(gl, 5, 5); add_edge!(gl, 9, 9)
    o1, i1 = GRounds.grounds_batch([gl, path_graph(30)], RoundsAlg(; score = :mf1, relax = 0.1))
    @assert i1.nnz[1] == i1.nnz[2] && isperm(o1[1])
    o2, i2 = GRounds.grounds_batch_csr([GRounds.csr(x) for x in gs], nothing, RoundsAlg(; score = :mf1, relax = 0.1))
    @assert all(isperm(o2[i]) for i in eachindex(gs))
    ts = [Threads.@spawn GRounds.grounds_batch(gs, RoundsAlg(; score = :mf1, relax = 0.1); chunks = 2) for _ in 1:4]
    @assert all(fetch(t)[2].nnz == i2.nnz for t in ts)
    println("edge cases done")
end
# tails: GPU tail without CPU handoff, CPU tail (tailalg), tail parameter
if get(ENV, "TAILS", "1") == "1"
    for (name, g, w) in allg
        for (lab, alg, kw) in (("gputail", RoundsAlg(; score = :mf1, select = :luby, relax = 0.5, tailsel = 128, tailmax = 10^9), (cpualpha = Inf,)),
                ("cputail", RoundsAlg(; score = :mf1, select = :luby, relax = 0.5, tailsel = 128, tailmax = 10^9, tailalg = CliqueTrees.MF(; strategy = 1)), (;)),
                ("tail50", RoundsAlg(; score = :mf1, relax = 0.1, tail = 50), (;)))
            b = GRoundsPS.grounds(w, g, alg; kw...)[1]; qb = q(g, b, w)
            res = Any[]
            for m in ("block16", "grid2")
                tm = Dict{Symbol, Any}()
                o = m == "block16" ? GRounds.grounds(w, g, alg; mode = :block, nw = 16, timing = tm, kw...)[1] :
                    GRounds.grounds(w, g, alg; mode = :grid, G = 2, timing = tm, kw...)[1]
                push!(res, (m, o, tm))
            end
            ob, ib = GRounds.grounds_batch([g, g], alg; weights = [w, w])
            push!(res, ("batch", ob[2], Dict(:nnz => ib.nnz[2], :tw => ib.tw[2])))
            for (m, o, tm) in res
                @assert length(o) == nv(g) && isperm(o)
                qa = q(g, o, w)
                if qa == qb && tm[:nnz] == qa[1][1]
                    global nsame += 1
                else
                    global nbad += 1
                    println(rpad(name, 22), " $lab $m DIFF $(qa) vs $(qb) dev=$(tm[:nnz])")
                end
            end
        end
    end
    println("tails done")
end
println("edge cases ok; identical fill: $nsame; differences: $nbad")
