using CUDA, CliqueTrees, Graphs, Serialization, Printf, Random
using CliqueTrees: bt_mat, bt_adj!, bt_clk, sepdp_blk!, sepdp_lab!, sepdp_rel, sepdp_srt!, sepdp_set!, sepdp_tri, sepdp_val!, sepdp_bv1!, sepdp_rot!, sepdp_ord!, FVector, FScalar
include(joinpath(@__DIR__, "SepEnumGPU.jl")); using .SepEnumGPU
function loadatoms(files; nmax = 64)
    out = []
    for f in files, (nm, v) in deserialize(f), a in v.atoms
        (2 <= a.n <= nmax) || continue
        g = SimpleGraph(a.n); for (x, y) in a.edges; add_edge!(g, x, y); end
        for r in a.rounds
            r[2] >= 0 && push!(out, (; inst = nm, g, w = a.weights, k = r[1], nsep = r[2], tcpu = r[3]))
        end
    end
    return out
end
calls = loadatoms(filter(f -> startswith(basename(f), "atoms_") && endswith(f, ".jls"), readdir(@__DIR__; join = true)))
T = Dict(:blk => 0.0, :prep => 0.0, :val => 0.0, :root => 0.0, :ord => 0.0, :sepenum => 0.0)
nblks = Int[]; found = 0
for c in calls
    g = BipartiteGraph(c.g); w = c.w; k = c.k; nvtx = nv(g)
    ts = time(); sep, _ = CliqueTrees.sepenum(w, g, k, Inf, 1); T[:sepenum] += time() - ts
    adj = bt_adj!(bt_mat(1, nvtx), g); clk = bt_clk(Inf); V = Int; W = Float64
    nsep = nv(sep); sepwgt = FVector{W}(undef, nsep); work1 = FVector{V}(undef, nvtx)
    tb = time(); lstmin, lstlen, lstwgt, lstsep, lstset = sepdp_blk!(sepwgt, adj, w, g, sep, clk); T[:blk] += time() - tb
    t0 = time()
    nblk = length(lstmin); push!(nblks, nblk)
    new2old, old2new, moved = sepdp_lab!(lstmin, lstset, nvtx, 1)
    ww, gg, ss, aa = moved ? sepdp_rel(w, g, sep, adj, new2old, old2new) : (w, g, sep, adj)
    blkmin = FVector{V}(undef, nblk); blklen = FVector{V}(undef, nblk); blkwgt = FVector{W}(undef, nblk); blksep = FVector{V}(undef, nblk); order = FVector{V}(undef, nblk)
    fst = sepdp_srt!(blkmin, blklen, blkwgt, blksep, order, work1, lstmin, lstlen, lstwgt, lstsep, nvtx)
    blkset, sepset = sepdp_set!(aa, ss, blksep, order, lstset, nvtx)
    trie = merge(sepdp_tri(fst, blklen, sepset, ww), (; clk))
    T[:prep] += time() - t0
    valk = FVector{W}(undef, nblk + 1); capptr = FVector{V}(undef, nblk + 2)
    pool = UInt64[]; captgt = V[]; bagset = FVector{UInt64}(undef, 1)
    T[:val] += @elapsed for b in 1:nblk
        capptr[b] = length(captgt) + 1
        sepdp_bv1!(valk, captgt, pool, bagset, aa, ww, ss, sepwgt, blkset, sepset, fst, trie, blkmin, blklen, blkwgt, blksep, k, b)
    end
    tr = time(); cap = sepdp_rot!(valk, capptr, captgt, pool, bagset, aa, ww, blkset, sepset, fst, trie, blklen, k); T[:root] += time() - tr
    mink = valk[nblk + 1]
    if mink <= k
        global found += 1
        T[:ord] += @elapsed begin
            perm = FVector{V}(undef, nvtx); invp = FVector{V}(undef, nvtx); tag = FScalar{V}(undef); tag[] = 0
            sepdp_ord!(perm, invp, FVector{V}(undef, nvtx + 1), work1, FVector{V}(undef, nvtx), tag, gg, ss, fst, blklen, blksep, cap)
        end
    end
end
tot = sum(values(T))
for k in (:sepenum, :blk, :prep, :val, :root, :ord)
    @printf("%-8s %.3f s (%.0f%%)\n", k, T[k], 100T[k] / tot)
end
println("rounds $(length(calls)), found $found, blocks total $(sum(nblks)), max $(maximum(nblks))")
