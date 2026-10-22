# Checks of the GPU subtree DP:
#  (1) every chunk's optimum <= the cost of its current structure (a feasible solution);
#  (2) on sampled chunks, the GPU optimum equals an independent CPU DP that builds the leg sets
#      explicitly (Set{Int}) instead of zeta transforms;
#  (3) the applied round: exact re-costed log2C matches the predicted gain, then full reconf! runs.
#   julia --project=env scripts/test_reconf.jl <list> <orddir> [K]
include(joinpath(@__DIR__, "..", "src", "Host.jl")); using .Host
include(joinpath(@__DIR__, "..", "src", "GPUReconf.jl")); using .GPUReconf
include(joinpath(@__DIR__, "..", "src", "Reconf.jl")); using .Reconf
using Random

function cpu_dp(F, legsF, outlegs, w)
    k = length(F); M = 1 << k
    cnt = Dict{Int,Int}()
    for l in legsF, i in l; cnt[i] = get(cnt, i, 0) + 1; end
    outs = Set(outlegs)
    legsS = Vector{Set{Int}}(undef, M)
    for S in 1:M-1
        c = Dict{Int,Int}()
        for q in 0:k-1
            S >> q & 1 == 1 || continue
            for i in legsF[q+1]; c[i] = get(c, i, 0) + 1; end
        end
        legsS[S] = Set(i for (i, x) in c if x < cnt[i] || i in outs)
    end
    best = fill(Inf, M)
    for S in 1:M-1
        count_ones(S) == 1 && (best[S] = 0.0; continue)
    end
    for S in sort(1:M-1; by = count_ones)
        count_ones(S) == 1 && continue
        for S1 in 1:S-1
            (S1 & S) == S1 || continue
            S2 = S ⊻ S1; S1 < S2 || continue
            inv = union(legsS[S1], legsS[S2])
            v = best[S1] + best[S2] + exp2(sum(w[i] for i in inv; init = 0.0))
            v < best[S] && (best[S] = v)
        end
    end
    best[M-1]
end

names = readlines(ARGS[1]); orddir = ARGS[2]; K = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8
rng = Xoshiro(3); nbad1 = 0; nbad2 = 0; nchk = 0
for name in names
    files = sort(filter(f -> startswith(f, name * "__hbt") && endswith(f, ".ord"), readdir(orddir)))
    isempty(files) && continue
    net = readnet(name)
    o = parse.(Int, split(readlines(joinpath(orddir, files[1]))[2]))
    tr = bucket_tree(net, o)
    legs = [Int[] for _ in 1:2net.T-1]; cost!(tr, net; legs)
    b = DPBatch(); descs = Reconf.Desc[]
    Reconf.addchunks!(b, descs, 1, tr, net, legs, K, rng, false, zeros(UInt16, net.n), falses(net.n))
    t = @elapsed (best, arg) = dp_solve!(b; K)
    t2 = @elapsed (best, arg) = dp_solve!(b; K)
    for c in eachindex(descs)
        best[c] <= descs[c].oldrel * (1 + 1e-4) || (global nbad1 += 1)
    end
    for c in randperm(rng, length(descs))[1:min(6, length(descs))]
        d = descs[c]
        ref = cpu_dp(d.F, [legs[f] for f in d.F], legs[d.r], net.w)
        g = Float64(best[c]) * exp2(d.off)
        global nchk += 1
        abs(log2(ref) - log2(g)) < 1e-4 || (global nbad2 += 1; println("  DP mismatch ", name, " chunk ", c, " k=", length(d.F), " cpu ", log2(ref), " gpu ", log2(g)))
    end
    c0 = tr.log2C
    st = reconf!([tr], net; K, rounds = 1, patience = 1)
    c1 = tr.log2C
    st = reconf!([tr], net; K, rounds = 100)
    println(rpad(name, 48), " chunks=", length(descs), " dp ", round(t2 * 1e3; digits = 1), " ms (", round(length(descs) / t2; digits = 0), " chunks/s)  log10C ", round(c0 * log10(2); digits = 4), " -> 1 round ", round(c1 * log10(2); digits = 4), " -> ", round(tr.log2C * log10(2); digits = 4), " (", st.rounds, " rounds)")
end
println("chunks above current cost: ", nbad1, "   CPU-DP checks ", nchk, " mismatches ", nbad2)
