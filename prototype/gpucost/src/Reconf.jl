# Host driver of the batched subtree reconfiguration (GPU DP in GPUReconf).
# Every round: for every internal node r of every tree, grow a chunk rooted at r (expand the frontier
# node with the largest flops, or a random one in randomized rounds, until k = K frontier nodes),
# solve all chunks on the GPU, then apply a disjoint set of improving chunks (largest gain first;
# chunks with disjoint internal nodes are independent: a chunk's frontier legs and root legs depend
# only on leaf sets, which other chunks do not change). Trees are re-costed exactly after each round.
module Reconf

using Random
using ..Host
using ..GPUReconf

export reconf!, ils!, kick!, copytree

struct Desc
    tree::Int
    r::Int
    I::Vector{Int}
    F::Vector{Int}
    oldrel::Float64
    off::Float64
end

function grow(tr::Host.Tree, r::Int, K::Int, rng, randomized::Bool)
    T = tr.T
    I = [r]; F = [tr.left[r], tr.right[r]]
    while length(F) < K
        bi = 0; bv = -Inf
        for (q, f) in enumerate(F)
            f > T || continue
            v = randomized ? rand(rng) : tr.acc[f]
            v > bv && (bv = v; bi = q)
        end
        bi == 0 && break
        f = F[bi]; push!(I, f)
        F[bi] = tr.left[f]; push!(F, tr.right[f])
    end
    I, F
end

function addchunks!(b::DPBatch, descs::Vector{Desc}, ti::Int, tr::Host.Tree, net::Host.Net, legs, K::Int, rng, randomized::Bool, mask::Vector{UInt16}, inout::BitVector; roots = nothing)
    T = tr.T; N = 2T - 1
    touched = Int[]
    for r in (roots === nothing ? (T+1:N) : roots)
        I, F = grow(tr, r, K, rng, randomized)
        length(F) <= 2 && continue      # a single contraction: nothing to reorder
        for i in legs[r]; inout[i] = true; end
        for (q, f) in enumerate(F), i in legs[f]
            mask[i] == 0 && push!(touched, i)
            mask[i] |= UInt16(1) << (q - 1)
        end
        off = maximum(tr.acc[p] for p in I)
        oldrel = sum(exp2(tr.acc[p] - off) for p in I)
        push!(b.kk, Int32(length(F)))
        for i in touched
            push!(b.umask, mask[i]); push!(b.uin, inout[i] ? 0x01 : 0x00); push!(b.uw, Float32(net.w[i]))
            mask[i] = 0
        end
        for i in legs[r]; inout[i] = false; end
        push!(b.uptr, Int32(length(b.umask)))
        push!(b.off, Float32(off))
        push!(descs, Desc(ti, r, I, F, oldrel, off))
        empty!(touched)
    end
end

function rebuild!(tr::Host.Tree, d::Desc, arg::AbstractVector{UInt16})
    free = d.I[2:end]
    k = length(d.F); full = (1 << k) - 1
    function go(S::Int, id::Int)
        if count_ones(S) == 1
            return d.F[trailing_zeros(S)+1]
        end
        S1 = Int(arg[S+1]); S2 = S ⊻ S1
        node = id == 0 ? pop!(free) : id
        tr.left[node] = go(S1, 0); tr.right[node] = go(S2, 0)
        node
    end
    go(full, d.r)
    @assert isempty(free)
end

"""Reconfigure `trees` (Host.Tree, re-costed in place) of network `net`. Stops after `rounds` rounds,
at `deadline` (time()), or when `patience` consecutive rounds improve no tree by more than `tol`
(relative). Returns stats (rounds, chunks solved, GPU seconds)."""
function reconf!(trees::Vector{Host.Tree}, net::Host.Net; K::Int = 10, rounds::Int = 50, deadline::Float64 = Inf,
        rng = Xoshiro(1), tol = 1e-9, patience::Int = 2, randfrac::Float64 = 0.5, dirty = nothing)
    mask = zeros(UInt16, net.n); inout = falses(net.n)
    legs = [[Vector{Int}() for _ in 1:2t.T-1] for t in trees]
    for (ti, t) in enumerate(trees)
        cost!(t, net; legs = legs[ti])
    end
    nch = 0; tgpu = 0.0; nr = 0; stall = 0
    newid = Int[]
    while nr < rounds && time() < deadline
        nr += 1
        randomized = nr > 1 && rand(rng) < randfrac
        b = DPBatch(); descs = Desc[]
        for (ti, t) in enumerate(trees)
            roots = dirty === nothing ? nothing : candroots(t, dirty[ti], K)
            addchunks!(b, descs, ti, t, net, legs[ti], K, rng, randomized, mask, inout; roots)
        end
        isempty(descs) && break
        tg = time(); best, arg = dp_solve!(b; K = K); tgpu += time() - tg
        nch += length(descs)
        gains = [descs[c].oldrel - best[c] for c in eachindex(descs)]
        order = sortperm([gains[c] > 1e-5 * descs[c].oldrel ? log2(gains[c]) + descs[c].off : -Inf for c in eachindex(descs)]; rev = true)
        used = [falses(2t.T - 1) for t in trees]
        changed = falses(length(trees)); old = [t.log2C for t in trees]
        for c in order
            gains[c] > 1e-5 * descs[c].oldrel || break
            d = descs[c]; u = used[d.tree]
            any(u[p] for p in d.I) && continue
            for p in d.I; u[p] = true; end
            rebuild!(trees[d.tree], d, view(arg, :, c)); changed[d.tree] = true
        end
        improved = false
        for (ti, t) in enumerate(trees)
            if !changed[ti]
                dirty === nothing || fill!(dirty[ti], false)
                continue
            end
            renumber!(t, 2t.T - 1; newid_out = newid)
            if dirty !== nothing
                fill!(dirty[ti], false)
                for x in eachindex(used[ti])
                    used[ti][x] && (dirty[ti][newid[x]] = true)
                end
            end
            cost!(t, net; legs = legs[ti])
            t.log2C < old[ti] - tol && (improved = true)
            @assert t.log2C <= old[ti] + 1e-6 "reconf made tree $ti worse: $(old[ti]) -> $(t.log2C)"
        end
        stall = improved ? 0 : stall + 1
        stall >= patience && break
    end
    (; rounds = nr, chunks = nch, tgpu)
end

"""Chunk roots whose chunk can contain a dirty node: dirty nodes and their ancestors up to K - 2 levels."""
function candroots(t::Host.Tree, dirty::BitVector, K::Int)
    T = t.T; N = 2T - 1
    mark = falses(N)
    for x in T+1:N
        dirty[x] || continue
        y = x; d = 0
        while y != 0 && d <= K - 2
            mark[y] && d > 0 && break
            mark[y] = true; y = t.parent[y]; d += 1
        end
    end
    findall(mark)
end

function copytree(t::Host.Tree)
    Host.Tree(t.T, copy(t.left), copy(t.right), copy(t.parent), copy(t.acc), copy(t.leg), t.log2C, t.W)
end

"""`nrot` random rotations: pick an internal node p with an internal child c = (a, b) and sibling s;
replace p = (c, s) by p = (a, (b, s)) or p = (b, (a, s)). Rotations near expensive contractions are
preferred (p drawn with probability ~ flops^(1/2))."""
function kick!(t::Host.Tree, nrot::Int, rng; dirty = nothing)
    T = t.T; N = 2T - 1
    touched = Int[]
    wts = [exp2(0.5 * (t.acc[p] - t.log2C)) for p in T+1:N]
    cw = cumsum(wts)
    for _ in 1:nrot
        p = T + searchsortedfirst(cw, rand(rng) * cw[end])
        p = min(p, N)
        l, r = t.left[p], t.right[p]
        cands = Int[]; l > T && push!(cands, 1); r > T && push!(cands, 2)
        isempty(cands) && continue
        side = rand(rng, cands)
        c, s = side == 1 ? (l, r) : (r, l)
        a, b = t.left[c], t.right[c]
        rand(rng, Bool) && ((a, b) = (b, a))
        # p = (a, c'), c' = (b, s) reusing node id c
        t.left[c] = b; t.right[c] = s
        t.left[p] = a; t.right[p] = c
        t.parent[b] = c; t.parent[s] = c; t.parent[a] = p; t.parent[c] = p
        push!(touched, p, c)
    end
    newid = Int[]
    renumber!(t, N; newid_out = newid)
    if dirty !== nothing
        fill!(dirty, false)
        for x in touched; dirty[newid[x]] = true; end
    end
    t
end

"""Iterated local search with a population: each step copies the incumbent P times, kicks every copy
with a few random rotations (strength varies over the population), runs batched reconfiguration on
all copies jointly (one DP launch per round for the whole population), and keeps the best tree if
it beats the incumbent. Until `deadline`."""
function ils!(best::Host.Tree, net::Host.Net; deadline::Float64, P::Int = 16, K::Int = 10, rng = Xoshiro(2),
        rounds::Int = 12, strengths = (1, 2, 3, 4, 6, 8))
    nstep = 0; nimp = 0; nch = 0
    while time() < deadline
        dirty = [falses(2best.T - 1) for _ in 1:P]
        pop = [kick!(copytree(best), strengths[mod1(q, length(strengths))], rng; dirty = dirty[q]) for q in 1:P]
        st = reconf!(pop, net; K, rounds, deadline, rng, patience = 2, randfrac = 0.5, dirty)
        nch += st.chunks; nstep += 1
        k = argmin([t.log2C for t in pop])
        if pop[k].log2C < best.log2C - 1e-9
            b = pop[k]
            best.left = b.left; best.right = b.right; best.parent = b.parent; best.acc = b.acc; best.leg = b.leg
            best.log2C = b.log2C; best.W = b.W; nimp += 1
        end
    end
    (; steps = nstep, improved = nimp, chunks = nch)
end

end
