using CUDA, CliqueTrees, Graphs, Random
include(joinpath(@__DIR__, "src", "GPUOrd.jl")); using .GPUOrd

function refcost(w, g, order, invq)
    label, tree = cliquetree(w, g, order, Nodal())
    lc = -Inf
    for clique in tree
        x = sum(w[label[v]] for v in clique) * invq
        lc = lc == -Inf ? x : max(lc, x) + log2(1 + exp2(-abs(lc - x)))
    end
    return lc
end

function randpiece(rng, n)
    kind = rand(rng, 1:3)
    kind == 1 ? erdos_renyi(n, min(1.0, 3.5 / n); rng) :
    kind == 2 ? random_regular_graph(n, 3; rng) :
    first(euclidean_graph(n, 2; cutoff = 1.6 / sqrt(n), rng))
end

function check(; seed = 1, P = 40, K = 16, invq = 1 / 1000)
    rng = Xoshiro(seed); nbad = 0; nbadp = 0; maxerr = 0.0
    for W in (1, 2, 4, 8)
        N = 64W
        gs = [randpiece(rng, rand(rng, max(2, N ÷ 2 + 1):N) ÷ 2 * 2) for _ in 1:P]
        ws = [Int32[round(Int32, 1000 * log2(rand(rng, 2:7))) for _ in 1:nv(g)] for g in gs]
        # half the pieces have a boundary (last vertices not eliminated)
        nints = [isodd(p) ? nv(gs[p]) : rand(rng, 1:nv(gs[p])) for p in 1:P]
        b = pack(gs, ws, W; nints)
        horders = zeros(Int16, N, P * K)
        for p in 1:P, k in 1:K
            horders[1:nints[p], (p - 1) * K + k] .= randperm(rng, nints[p]) .- 1
        end
        pidx = CuArray(Int32.(repeat(1:P; inner = K)))
        width = CUDA.zeros(Int32, P * K); lcost = CUDA.zeros(Float32, P * K); tm = CUDA.zeros(Int32, P * K)
        score!(width, lcost, tm, b, pidx, CuArray(horders); invq)
        hw = Array(width); hc = Array(lcost)
        for p in 1:P, k in 1:K
            g = gs[p]; w = ws[p]; j = (p - 1) * K + k
            ord = Int.(horders[1:nints[p], j]) .+ 1
            if nints[p] == nv(g)
                ref = treewidth(w, g; alg = ord)
                hw[j] == ref || (nbad += 1)
                maxerr = max(maxerr, abs(hc[j] - refcost(w, g, ord, invq)))
            else
                ref, rc = cpu_score_bitset(cpurows(g, W), w, ord, invq)
                hw[j] == ref || (nbadp += 1)
                maxerr = max(maxerr, abs(hc[j] - rc))
            end
        end
        println("score W=$W: full-piece mismatches vs CliqueTrees $nbad, boundary-piece mismatches vs CPU bitset $nbadp, max |dlog2cost| $maxerr")
    end
    return nbad + nbadp
end

function check_greedy(; seed = 2, P = 20, K = 32, invq = 1 / 1000)
    rng = Xoshiro(seed); nbad = 0
    for W in (1, 2, 4, 8)
        N = 64W
        gs = [randpiece(rng, rand(rng, max(2, N ÷ 2 + 1):N) ÷ 2 * 2) for _ in 1:P]
        ws = [Int32[round(Int32, 1000 * log2(rand(rng, 2:7))) for _ in 1:nv(g)] for g in gs]
        sw = [Int32.(round.(Int32, w ./ 250)) for w in ws]
        nints = [isodd(p) ? nv(gs[p]) : rand(rng, 1:nv(gs[p])) for p in 1:P]
        b = pack(gs, ws, W; nints, sweights = sw)
        prm = [GreedyParams(Float32(k % 3 == 0), Float32(k % 3 == 1), Float32(k % 3 == 2), k < K ÷ 2 ? 0.0f0 : 0.3f0, UInt32(k * 7919 + p)) for p in 1:P for k in 1:K]
        pidx = CuArray(Int32.(repeat(1:P; inner = K)))
        order = CUDA.zeros(Int16, N, P * K); width = CUDA.zeros(Int32, P * K); lcost = CUDA.zeros(Float32, P * K); tm = CUDA.zeros(Int32, P * K)
        greedy!(order, width, lcost, tm, b, pidx, CuArray(prm); invq)
        ho = Array(order); hw = Array(width)
        for p in 1:P, k in 1:K
            g = gs[p]; j = (p - 1) * K + k
            ord = Int.(ho[1:nints[p], j]) .+ 1
            (sort(ord) == 1:nints[p]) || (nbad += 1; continue)
            ref = nints[p] == nv(g) ? treewidth(ws[p], g; alg = ord) : first(cpu_score_bitset(cpurows(g, W), ws[p], ord))
            ref == hw[j] || (nbad += 1)
        end
        full = filter(isodd, 1:P)
        gb = [minimum(hw[(p - 1) * K + 1:p * K]) for p in full]
        cb = [minimum(treewidth(ws[p], gs[p]; alg = a) for a in (MF(), MF(; strategy = 1), AMF(), MMD())) for p in full]
        println("greedy W=$W: bad=$nbad  GPU best-of-$K vs bw (full pieces): wins ", count(gb .< cb), " ties ", count(gb .== cb), " losses ", count(gb .> cb))
    end
    return nbad
end

# mutate + select: incumbents only improve, and stay valid permutations
function check_search(; seed = 3, P = 16, K = 64)
    rng = Xoshiro(seed); nbad = 0
    for W in (1, 4)
        N = 64W
        gs = [randpiece(rng, N ÷ 2 * 2 - 2) for _ in 1:P]
        ws = [Int32[round(Int32, 1000 * log2(rand(rng, 2:7))) for _ in 1:nv(g)] for g in gs]
        b = pack(gs, ws, W)
        best = CUDA.zeros(Int16, N, P); bestw = CUDA.fill(typemax(Int32), P); bestc = CUDA.fill(Inf32, P); bt = CUDA.zeros(Int32, P)
        pidx = CuArray(Int32.(repeat(1:P; inner = K))); bstart = CuArray(Int32.(1:K:P*K+1)); bpiece = CuArray(Int32.(1:P))
        cand = CUDA.zeros(Int16, N, P * K); width = CUDA.zeros(Int32, P * K); lcost = CUDA.zeros(Float32, P * K); tm = CUDA.zeros(Int32, P * K)
        greedy!(cand, width, lcost, tm, b, pidx, CuArray([GreedyParams(0, 0, 1, 0.1f0, rand(rng, UInt32)) for _ in 1:P*K]); invq = 1e-3)
        selectbest!(bstart, bpiece, cand, width, lcost, tm, best, bestw, bestc, bt, b)
        w0 = Array(bestw)
        for r in 1:50
            mutate!(cand, width, lcost, tm, b, pidx, best, bt, CuArray([MutateParams(rand(rng, 1:3), 0.5f0, rand(rng, (0, 8)), rand(rng, UInt32)) for _ in 1:P*K]); invq = 1e-3)
            selectbest!(bstart, bpiece, cand, width, lcost, tm, best, bestw, bestc, bt, b)
        end
        w1 = Array(bestw); hb = Array(best)
        for p in 1:P
            ord = Int.(hb[1:nv(gs[p]), p]) .+ 1
            isperm(ord) || (nbad += 1; continue)
            treewidth(ws[p], gs[p]; alg = ord) == w1[p] || (nbad += 1)
            w1[p] <= w0[p] || (nbad += 1)
        end
        println("search W=$W: bad=$nbad  mean width greedy-best $(sum(w0)/P/1000)  after 50 LS rounds $(sum(w1)/P/1000)")
    end
    return nbad
end

check(); check_greedy(); check_search()
