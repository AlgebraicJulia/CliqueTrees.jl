# Mass search over orderings scored by exact C (GPUTree), then batched GPU reconfiguration (Reconf).
module Search

using CUDA, Random
using ..Host
using ..GPUTree
using ..Reconf

export mass_search, gpu_pipeline, tree_from_job

"""Host.Tree of job j of the scratch (already in topological order)."""
function tree_from_job(sc, g, j)
    l, r = fetch_tree(sc, g, j)
    T = Int(g.T); tr = Host.Tree(T)
    tr.left .= l; tr.right .= r
    for p in T+1:2T-1
        tr.parent[l[p]] = p; tr.parent[r[p]] = p
    end
    tr
end

function jobsize(net)
    T = net.T; N = 2T - 1
    4 * (7N + 3T + 32 * maximum(diff(net.occptr)) + net.n)
end

"""(mu + lambda) search: `seeds` (orderings, open indices last) are scored, the best `E` distinct
(by log2C) are the elites; each generation perturbs random elites (insertion moves, strength drawn
per job) and keeps the best E. Stops at `deadline`. Returns (elite orderings, their log2C,
candidates scored, generations, scratch, elite job ids in the last scratch fill)."""
function mass_search(net::Host.Net, g::GNet, seeds::Vector{Vector{Int}}; deadline::Float64, E::Int = 16,
        J::Int = 0, rng = Xoshiro(1), mem::Int = 1 << 29, moves = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32), dists = (1, 2, 4, 8, 16, 32, 64, 128))
    n = net.n
    J == 0 && (J = clamp(prevpow(2, mem ÷ jobsize(net)), 128, 8192))
    J = max(J, length(seeds))
    sc = scratch(g, J)
    # generation 0: the seeds
    ns = length(seeds)
    copyto!(view(sc.orders, :, 1:ns), reduce(hcat, [Int32.(s) .- Int32(1) for s in seeds]))
    score_orders!(sc, g, ns)
    lc = Array(sc.log2C)[1:ns]
    pool = [(lc[j], Array(view(sc.orders, :, j))) for j in 1:ns]
    sort!(pool; by = first)
    elites = Tuple{Float32, Vector{Int32}}[]
    for p in pool
        any(abs(p[1] - e[1]) < 1e-6 for e in elites) && continue
        push!(elites, p); length(elites) == E && break
    end
    ncand = ns; ngen = 0
    base = CuArray(reduce(hcat, [e[2] for e in elites]))
    Jmax = J; tstart = time()
    while time() < deadline
        tg = time()
        ne = length(elites)
        bidx = Int32[rand(rng, 1:ne) for _ in 1:J]
        nm = Int32[rand(rng, moves) for _ in 1:J]
        md = Int32[rand(rng, dists) for _ in 1:J]
        perturb!(sc, g, base, CuArray(bidx), CuArray(nm), CuArray(md), CuArray(rand(rng, UInt64, J)), J)
        score_orders!(sc, g, J)
        lc = Array(sc.log2C)
        ncand += J; ngen += 1
        # merge
        cand = sortperm(lc)[1:min(4E, J)]
        changed = false
        for j in cand
            lc[j] < elites[end][1] || length(elites) < E || break
            any(abs(lc[j] - e[1]) < 1e-6 for e in elites) && continue
            push!(elites, (lc[j], Array(view(sc.orders, :, j))))
            sort!(elites; by = first); length(elites) > E && pop!(elites)
            changed = true
        end
        changed && (base = CuArray(reduce(hcat, [e[2] for e in elites])))
        # keep a generation under ~1/10 of the search time
        (time() - tg > (deadline - tstart) / 10 && J > 128) && (J ÷= 2)
    end
    [Int.(e[2]) .+ 1 for e in elites], [e[1] for e in elites], ncand, ngen, pool[1][1]
end

"""Score `orders` (open last) on the GPU and return their Host trees (exactly re-costed)."""
function trees_of(net, g, orders)
    sc = scratch(g, length(orders))
    copyto!(sc.orders, reduce(hcat, [Int32.(o) .- Int32(1) for o in orders]))
    score_orders!(sc, g, length(orders))
    [cost!(tree_from_job(sc, g, j), net) for j in eachindex(orders)]
end

"""Full GPU stage within `budget` seconds: mass search for `fmass` of it, then reconfiguration of the
best `M` elites with K = `K1`, then of the best tree with K = `K2` until the deadline."""
function gpu_pipeline(net::Host.Net, seeds::Vector{Vector{Int}}; budget::Float64, fmass::Float64 = 0.4, M::Int = 8,
        K1::Int = 10, K2::Int = 12, E::Int = 16, rng = Xoshiro(1), log = devnull, ils::Bool = true, P::Int = 0)
    t0 = time(); dl = t0 + budget
    g = gnet(net)
    seeds = [open_last(net, s) for s in seeds]
    tseed = time() - t0
    el, elc, ncand, ngen, seedbest = mass_search(net, g, seeds; deadline = t0 + fmass * budget, E, rng)
    tmass = time() - t0
    trees = trees_of(net, g, el[1:min(M, length(el))])
    raw = [t.log2C for t in trees]
    st1 = reconf!(trees, net; K = K1, deadline = t0 + 0.6 * budget, rng, patience = 6, rounds = 400)
    k = argmin([t.log2C for t in trees]); best = trees[k]
    mid = best.log2C
    sti = ils ? ils!(best, net; deadline = t0 + 0.9 * budget, P = P == 0 ? clamp(32768 ÷ net.T, 8, 64) : P, K = K1, rng) : (; steps = 0, improved = 0, chunks = 0)
    mid2 = best.log2C
    st2 = reconf!([best], net; K = K2, deadline = dl, rng, patience = 3)
    println(log, "  seeds ", length(seeds), " (", round(tseed; digits = 3), " s)  mass ", ncand, " cand in ", ngen, " gens, ",
        round(ncand / (tmass - tseed); digits = 0), " cand/s; raw log10C seeds ", round(seedbest * log10(2); digits = 4), " -> ", round(elc[1] * log10(2); digits = 4),
        "  reconf K", K1, " on ", length(trees), ": ", st1.rounds, " rounds ", st1.chunks, " chunks -> ", round(mid * log10(2); digits = 4),
        "  ILS ", sti.steps, " steps ", sti.improved, " impr -> ", round(mid2 * log10(2); digits = 4), "  K", K2, ": ", st2.rounds, " rounds -> ", round(best.log2C * log10(2); digits = 4), "  total ", round(time() - t0; digits = 2), " s")
    best, (; ncand, ngen, tmass = tmass - tseed, rawbest = elc[1], seedraw = seedbest, mid, final = best.log2C,
        chunks = st1.chunks + st2.chunks + sti.chunks, mid2, tgpu = st1.tgpu + st2.tgpu, total = time() - t0)
end

end
