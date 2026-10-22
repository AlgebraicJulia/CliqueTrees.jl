# GPU side of cost-aware selection (one GPU).
#
#   julia --project=<gpu env> score.jl <queue> <pooldir> <outdir> [tgms=10] [tpol=10]
#
# Per network with <pooldir>/<net>.pool (from gen.jl), writes <outdir>/<net>.gpu:
#   GS   nTD seconds cand/s maxdiff_cpu  max|maxop-width| (all TDs)        -- GPU exact bucket C of every TD
#   T    tag gpu_log2C W maxop                                            -- one per TD
#   GMS  cap ncand ngen seconds seed_best_log2C best_log2C best_maxop     -- width-capped GPU mass search
#        <ordering>                                                       -- (insertion moves around the
#                                                                            best width-<=cap TDs; a move is
#                                                                            kept only if maxop <= cap)
#   GMT  t:ncand:best_log2C ...                                          -- its improvement trace
#   POL  name src_tag start_log2C final_log2C seconds                       -- GPU reconfiguration (K=10, ILS,
#        <ssa path>                                                          K=12) of a selected tree
# maxop of the bucket tree of a PEO = the TD width (checked per TD: column maxdiff of GS), so the cap is the
# TD width cap. POL starts: "br" = br tree of the TD selected by br cost (all TDs of delta 0), "bk" = bucket
# tree of the TD selected by GPU C (delta 0), "gms" = the mass-search winner.
include(joinpath(ENV["GPUCOST_SRC"], "Host.jl")); using .Host
include(joinpath(ENV["GPUCOST_SRC"], "GPUTree.jl")); using .GPUTree
include(joinpath(ENV["GPUCOST_SRC"], "GPUReconf.jl")); using .GPUReconf
include(joinpath(ENV["GPUCOST_SRC"], "Reconf.jl")); using .Reconf
include(joinpath(ENV["GPUCOST_SRC"], "Search.jl")); using .Search
using CUDA, Random, Printf

f6(x) = @sprintf("%.6f", x)

function readpool(f)
    tds = []; ssa = Dict{String, Vector{Tuple{Int, Int}}}(); sel = []; K = NaN; ls = readlines(f); i = 1
    while i <= length(ls)
        x = split(ls[i], '\t')
        if x[1] == "NET"
            K = parse(Float64, x[4])
        elseif x[1] == "TD"
            push!(tds, (; tag = String(x[2]), mode = String(x[3]), delta = parse(Int, x[4]), width = parse(Float64, x[7]),
                br = parse(Float64, x[9]), bk = parse(Float64, x[10]), order = parse.(Int, split(ls[i + 1]))))
            i += 1
        elseif x[1] == "SSA"
            v = parse.(Int, split(ls[i + 1])); ssa[String(x[2])] = [(v[k], v[k + 1]) for k in 1:2:length(v)]
            i += 1
        elseif x[1] == "SEL"
            push!(sel, x)
        end
        i += 1
    end
    return (; K, tds, ssa, sel)
end

"""Score orderings (already open-last) on the GPU in chunks of J; returns (log2C, W, maxop)."""
function gscore(net, g, orders; J = 4096)
    n = length(orders); lc = Float32[]; W = Float32[]; mo = Float32[]
    sc = scratch(g, min(J, n))
    for lo in 1:J:n
        hi = min(n, lo + J - 1); m = hi - lo + 1
        copyto!(view(sc.orders, :, 1:m), reduce(hcat, [Int32.(o) .- Int32(1) for o in orders[lo:hi]]))
        score_orders!(sc, g, m)
        append!(lc, Array(sc.log2C)[1:m]); append!(W, Array(sc.W)[1:m]); append!(mo, Array(sc.maxop)[1:m])
    end
    return lc, W, mo
end

"""(mu + lambda) mass search over insertion moves, keeping only orderings whose maxop (= TD width) <= cap."""
function capped_search(net, g, seeds; cap, deadline, E = 16, J = 0, rng = Xoshiro(1), mem = 1 << 29,
        moves = (1, 2, 3, 4, 6, 8, 12, 16), dists = (1, 2, 4, 8, 16, 32, 64))
    J == 0 && (J = clamp(prevpow(2, mem ÷ Search.jobsize(net)), 128, 8192))
    sc = scratch(g, J)
    lc0, _, mo0 = gscore(net, g, seeds)
    pool = [(lc0[j], Int32.(seeds[j]) .- Int32(1), mo0[j]) for j in eachindex(seeds) if mo0[j] <= cap]
    isempty(pool) && return nothing
    sort!(pool; by = first)
    elites = Tuple{Float32, Vector{Int32}, Float32}[]
    for p in pool
        any(abs(p[1] - e[1]) < 1e-6 for e in elites) && continue
        push!(elites, p); length(elites) == E && break
    end
    seedbest = elites[1][1]
    base = CuArray(reduce(hcat, [e[2] for e in elites])); ncand = 0; ngen = 0; t0 = time()
    trace = [(0.0, 0, seedbest)]
    while time() < deadline
        ne = length(elites)
        bidx = Int32[rand(rng, 1:ne) for _ in 1:J]
        nm = Int32[rand(rng, moves) for _ in 1:J]; md = Int32[rand(rng, dists) for _ in 1:J]
        perturb!(sc, g, base, CuArray(bidx), CuArray(nm), CuArray(md), CuArray(rand(rng, UInt64, J)), J)
        score_orders!(sc, g, J)
        lc = Array(sc.log2C); mo = Array(sc.maxop)
        ncand += J; ngen += 1
        for j in eachindex(lc)
            mo[j] > cap && (lc[j] = Inf32)
        end
        changed = false
        for j in sortperm(lc)[1:min(4E, J)]
            isfinite(lc[j]) || break
            lc[j] < elites[end][1] || length(elites) < E || break
            any(abs(lc[j] - e[1]) < 1e-6 for e in elites) && continue
            push!(elites, (lc[j], Array(view(sc.orders, :, j)), mo[j]))
            sort!(elites; by = first); length(elites) > E && pop!(elites)
            changed = true
        end
        changed && (base = CuArray(reduce(hcat, [e[2] for e in elites])))
        changed && push!(trace, (time() - t0, ncand, elites[1][1]))
    end
    return (; order = Int.(elites[1][2]) .+ 1, log2C = elites[1][1], maxop = elites[1][3], seedbest, ncand, ngen, secs = time() - t0, trace)
end

function polish(net, tr; tpol, rng = Xoshiro(3))
    t0 = time(); dl = t0 + tpol
    cost!(tr, net); start = tr.log2C
    reconf!([tr], net; K = 10, deadline = t0 + 0.4tpol, rng, patience = 4, rounds = 400)
    ils!(tr, net; deadline = t0 + 0.8tpol, P = clamp(32768 ÷ net.T, 8, 64), K = 10, rng)
    reconf!([tr], net; K = 12, deadline = dl, rng, patience = 3)
    cost!(tr, net)
    return start, tr.log2C, time() - t0
end

function fromssa(net, ssa)
    tr = Host.tree_from_ssa(net.T, ssa)
    cost!(tr, net)
    return tr
end

function runnet(io, name, pool; tgms, tpol)
    net = readnet(name); g = gnet(net)
    tds = pool.tds; isempty(tds) && return
    t0 = time()
    lc, W, mo = gscore(net, g, [t.order for t in tds]); CUDA.synchronize()
    ts = time() - t0
    dcpu = maximum(abs.(lc .- [t.bk for t in tds]))
    dw = maximum(abs.(mo .- [t.width for t in tds]))
    println(io, join(("GS", length(tds), f6(ts), f6(length(tds) / ts), f6(dcpu), f6(dw)), '\t'))
    for (k, t) in enumerate(tds)
        println(io, join(("T", t.tag, f6(lc[k]), f6(W[k]), f6(mo[k])), '\t'))
    end
    flush(io)
    cap = pool.K + 1.0e-4
    i0 = [k for k in eachindex(tds) if tds[k].delta == 0 && mo[k] <= cap]
    pols = Tuple{String, String, Host.Tree}[]
    if !isempty(i0)
        # seeds: the 64 cheapest delta-0 TDs by GPU C
        seeds = [tds[k].order for k in i0[sortperm(lc[i0])][1:min(64, length(i0))]]
        r = capped_search(net, g, seeds; cap, deadline = time() + tgms, rng = Xoshiro(hash(name)))
        if !isnothing(r)
            println(io, join(("GMS", f6(cap), r.ncand, r.ngen, f6(r.secs), f6(r.seedbest), f6(r.log2C), f6(r.maxop)), '\t'))
            println(io, join(r.order, ' '))
            println(io, "GMT\t", join((@sprintf("%.3f:%d:%.6f", x...) for x in r.trace), ' ')); flush(io)
            push!(pols, ("gms", "gms", Search.trees_of(net, g, [r.order])[1]))
        end
        kb = i0[argmin(lc[i0])]
        push!(pols, ("bk", tds[kb].tag, Search.trees_of(net, g, [tds[kb].order])[1]))
        kr = i0[argmin([tds[k].br for k in i0])]
        haskey(pool.ssa, tds[kr].tag) && push!(pols, ("br", tds[kr].tag, fromssa(net, pool.ssa[tds[kr].tag])))
    end
    for (nm, src, tr) in pols
        a, b, s = polish(net, tr; tpol)
        println(io, join(("POL", nm, src, f6(a), f6(b), f6(s)), '\t'))
        println(io, join((string(x) for p in Host.tree_ssa(tr) for x in p), ' ')); flush(io)
    end
end

function main()
    queue, pooldir, outdir = ARGS[1], ARGS[2], ARGS[3]
    tgms = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 10.0
    tpol = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 10.0
    mkpath(outdir)
    let p = joinpath(pooldir, "gm_1fas.pool")   # warm-up
        isfile(p) && runnet(IOBuffer(), "gm_1fas", readpool(p); tgms = 1.0, tpol = 1.0)
    end
    for name in filter(!isempty, strip.(readlines(queue)))
        f = joinpath(pooldir, name * ".pool"); o = joinpath(outdir, name * ".gpu")
        (isfile(f) && !isfile(o)) || continue
        println(stderr, "[$(round(Int, time()))] $name"); flush(stderr)
        try
            open(o * ".tmp", "w") do io
                runnet(io, name, readpool(f); tgms, tpol)
                println(io, "END")
            end
            mv(o * ".tmp", o; force = true)
        catch e
            println(stderr, "FAILED $name: ", sprint(showerror, e, catch_backtrace()))
        end
    end
end
main()
