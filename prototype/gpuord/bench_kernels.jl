# Kernel throughput: orderings scored / greedy orderings built per second,
# GPU vs CPU, piece sizes 64..512. usage: julia -t N bench_kernels.jl [quick]
using CUDA, CliqueTrees, Graphs, Random, Printf, Serialization
include(joinpath(@__DIR__, "src", "GPUOrd.jl")); using .GPUOrd

const QUICK = "quick" in ARGS
const GPUONLY = "gpuonly" in ARGS   # no CPU baselines (for GPU jobs)

# pieces that look like line-graph atoms: random geometric (2D, local), random
# 3-regular line graphs (circuit-like: every vertex in two cliques)
function mkpiece(rng, n, kind)
    if kind == :geo
        return first(euclidean_graph(n, 2; cutoff = sqrt(6.0 / (pi * n)), rng))
    else
        # line graph of a random 3-regular graph on 2n/3 vertices (n edges)
        m = 2 * cld(n, 3); m += isodd(m)
        h = random_regular_graph(m, 3; rng)
        es = collect(edges(h)); L = SimpleGraph(length(es))
        idx = Dict((src(e), dst(e)) => i for (i, e) in enumerate(es))
        for v in vertices(h)
            inc = [idx[minmax(v, u)] for u in neighbors(h, v)]
            for a in inc, b in inc
                a < b && add_edge!(L, a, b)
            end
        end
        return L
    end
end

qweights(rng, n) = Int32[round(Int32, 1000 * log2(rand(rng, (2, 2, 3, 4)))) for _ in 1:n]

function gpu_time(f; reps = 5)
    f(); CUDA.synchronize()
    ts = [CUDA.@elapsed f() for _ in 1:reps]
    return minimum(ts)
end

function bench(kind; W, P, K, rng = Xoshiro(11))
    N = 64W
    gs = [mkpiece(rng, N, kind) for _ in 1:P]
    gs = [nv(g) > N ? induced_subgraph(g, 1:N)[1] : g for g in gs]
    ws = [qweights(rng, nv(g)) for g in gs]
    sws = [Int32.(round.(Int32, w ./ 250)) for w in ws]
    tpack = @elapsed b = pack(gs, ws, W; sweights = sws)
    njobs = P * K
    pidx = CuArray(Int32.(repeat(1:P; inner = K)))
    horders = zeros(Int16, N, njobs)
    for p in 1:P, k in 1:K
        horders[1:nv(gs[p]), (p - 1) * K + k] .= randperm(rng, nv(gs[p])) .- 1
    end
    dord = CuArray(horders)
    width = CUDA.zeros(Int32, njobs); lc = CUDA.zeros(Float32, njobs); tm = CUDA.zeros(Int32, njobs)
    tscore = gpu_time(() -> score!(width, lc, tm, b, pidx, dord; invq = 1e-3))
    te2e = @elapsed begin
        d2 = CuArray(horders); score!(width, lc, tm, b, pidx, d2; invq = 1e-3); hw = Array(width); Array(lc)
    end
    hw = Array(width)

    # greedy: (a) min weighted degree, (b) min avg fill, (c) mixed; randomized
    gord = CUDA.zeros(Int16, N, njobs)
    mk(f) = CuArray([f(p, k) for p in 1:P for k in 1:K])
    prmd = mk((p, k) -> GreedyParams(1, 0, 0, 0.1f0, UInt32(k + 1000p)))
    prma = mk((p, k) -> GreedyParams(0, 0, 1, 0.1f0, UInt32(k + 1000p)))
    tgreedy_deg = gpu_time(() -> greedy!(gord, width, lc, tm, b, pidx, prmd; invq = 1e-3); reps = 3)
    tgreedy_avg = gpu_time(() -> greedy!(gord, width, lc, tm, b, pidx, prma; invq = 1e-3); reps = 3)
    hgo = Array(gord); hgw = Array(width)
    # score realistic (greedy-quality) orderings
    tscore_good = gpu_time(() -> score!(width, lc, tm, b, pidx, gord; invq = 1e-3))

    # CPU: CliqueTrees treewidth on the same orderings (1 thread), and the
    # same bitset algorithm as the GPU (1 thread and all threads)
    nc = GPUONLY ? 1 : QUICK ? min(njobs, 256) : min(njobs, 4096)
    sel = collect(1:max(1, njobs ÷ nc):njobs)[1:min(nc, end)]
    res = Dict{Symbol, Float64}()
    for (tag, H, hwref) in ((:rand, horders, hw), (:good, hgo, hgw))
        ords = [Int.(H[1:nv(gs[(j - 1) ÷ K + 1]), j]) .+ 1 for j in sel]
        gsel = [gs[(j - 1) ÷ K + 1] for j in sel]; wsel = [ws[(j - 1) ÷ K + 1] for j in sel]
        treewidth(wsel[1], gsel[1]; alg = ords[1])
        t = @elapsed for i in eachindex(sel)
            @assert treewidth(wsel[i], gsel[i]; alg = ords[i]) == hwref[sel[i]]
        end
        res[Symbol(:ct_, tag)] = t / length(sel)
        rows = [GPUOrd.cpurows(g, W) for g in gsel]
        GPUOrd.cpu_score_bitset(rows[1], wsel[1], ords[1])
        res[Symbol(:bs_, tag)] = (@elapsed for i in eachindex(sel)
            GPUOrd.cpu_score_bitset(rows[i], wsel[i], ords[i], 1e-3)
        end) / length(sel)
        Threads.@threads for i in 1:Threads.nthreads(); GPUOrd.cpu_score_bitset(rows[1], wsel[1], ords[1], 1e-3); end
        res[Symbol(:bsmt_, tag)] = (@elapsed Threads.@threads for i in eachindex(sel)
            GPUOrd.cpu_score_bitset(rows[i], wsel[i], ords[i], 1e-3)
        end) / length(sel)
    end
    # CPU greedy (CliqueTrees, one ordering each)
    npc = GPUONLY ? 1 : min(P, QUICK ? 16 : 64)
    tcg = Dict{String, Float64}()
    for (nm, a) in (("MF", MF()), ("MFavg", MF(; strategy = 1)), ("MMD", MMD()), ("AMF", AMF()), ("bw", BestWidth(MF(), MF(; strategy = 1), AMF(), MMD())))
        permutation(ws[1], gs[1]; alg = a)
        tcg[nm] = (@elapsed for p in 1:npc
            permutation(ws[p], gs[p]; alg = a)
        end) / npc
    end
    return (; kind, W, N, P, K, njobs, tpack, tscore, te2e, tscore_good, tgreedy_deg, tgreedy_avg, res, tcg,
        avgdeg = sum(ne, gs) * 2 / sum(nv, gs), randwidth = sum(hw) / njobs / 1000, goodwidth = sum(hgw) / njobs / 1000)
end

info = devinfo(); println(info); println("threads = ", Threads.nthreads())
res = []
for kind in (:geo, :line), W in (1, 2, 4, 8)
    P = 64; K = QUICK ? 64 : (W <= 2 ? 1024 : W == 4 ? 256 : 128)
    r = bench(kind; W, P, K)
    push!(res, r)
    R = r.res
    @printf("%-4s N=%3d deg=%.1f jobs=%6d | SCORE random ords (w~%.0f): GPU %.3g/s  CT %.3g/s  bitset1 %.3g/s  bitset%d %.3g/s  => GPU/CT %.0fx, GPU/bitsetMT %.1fx  (e2e GPU %.3g/s)\n",
        kind, r.N, r.avgdeg, r.njobs, r.randwidth, r.njobs / r.tscore, 1 / R[:ct_rand], 1 / R[:bs_rand], Threads.nthreads(), 1 / R[:bsmt_rand],
        R[:ct_rand] * r.njobs / r.tscore, R[:bsmt_rand] * r.njobs / r.tscore, r.njobs / r.te2e)
    @printf("      SCORE greedy ords (w~%.0f): GPU %.3g/s  CT %.3g/s  bitset1 %.3g/s  bitset%d %.3g/s  => GPU/CT %.0fx, GPU/bitsetMT %.1fx\n",
        r.goodwidth, r.njobs / r.tscore_good, 1 / R[:ct_good], 1 / R[:bs_good], Threads.nthreads(), 1 / R[:bsmt_good], R[:ct_good] * r.njobs / r.tscore_good, R[:bsmt_good] * r.njobs / r.tscore_good)
    @printf("      GREEDY GPU: min-deg %.3g/s  min-avgfill %.3g/s | CPU CliqueTrees: MF %.3g/s MFavg %.3g/s MMD %.3g/s AMF %.3g/s bw(4 algs) %.3g/s\n",
        r.njobs / r.tgreedy_deg, r.njobs / r.tgreedy_avg, 1 / r.tcg["MF"], 1 / r.tcg["MFavg"], 1 / r.tcg["MMD"], 1 / r.tcg["AMF"], 1 / r.tcg["bw"])
    flush(stdout)
end
serialize(joinpath(@__DIR__, "results", "kernels_$(replace(info.name, ' ' => '_')).jls"), (info, res))
