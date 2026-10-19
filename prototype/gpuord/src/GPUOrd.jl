module GPUOrd

# Batched, bit-parallel elimination on padded bitset adjacency matrices.
#
# A *bucket* holds P pieces with at most N = 64W vertices each. Piece p is
# stored word-major: word j (0-based) of row u is adj[u + j*N + 1, p], so the
# lanes of a warp, which own consecutive rows, touch consecutive words. The
# diagonal is set, so row v masked by the alive set is the closed bag N^+[v].
# The first nint[p] vertices of a piece are its *interior*: only they are
# eliminated (the others are the separator vertices of ancestor nodes in a
# nested dissection, eliminated higher up; they still count in bags).
#
# One warp runs one job = (piece, ordering); lane l owns the rows l, l+32, ...
# (2W of them). Eliminating v: bag = row v & alive; every lane ORs the bag
# into those of its rows that are in the bag (fill); the bag weight is a warp
# sum of the weights of the set bits. Job g works on piece pidx[g].

using CUDA
using Graphs

export Bucket, pack, score!, greedy!, mutate!, selectbest!, devinfo, GreedyParams,
    MutateParams, cpu_score_bitset, cpurows, jobkey

# ---------------------------------------------------------------- packing --

struct Bucket{W}
    adj::CuMatrix{UInt64}     # (N*W, P)
    wts::CuMatrix{Int32}      # (N, P) exact (quantized) weights, for widths
    planes::CuMatrix{UInt64}  # (W*NPL, P) bit planes of the score weights
    npl::Int32                # number of planes
    nvs::CuVector{Int32}
    nints::CuVector{Int32}
    hnv::Vector{Int32}
    hnint::Vector{Int32}
    P::Int
end

bucketwords(n::Integer) = n <= 64 ? 1 : n <= 128 ? 2 : n <= 256 ? 4 : n <= 512 ? 8 : cld(n, 64)

"""
    pack(graphs, weights, W; nints, sweights) -> Bucket{W}

Pad each graph (n <= 64W) to N = 64W rows of W words. `weights[i]` are exact
Int32 weights (widths are sums of them); `sweights[i]` are small nonnegative
integer weights used only in greedy scores (default: `weights`), stored as
bit planes. `nints[i]` (default n) vertices 1..nint are interior.
"""
function pack(graphs::AbstractVector, weights::AbstractVector, W::Int; nints = nothing, sweights = weights)
    N = 64W; P = length(graphs)
    adj = zeros(UInt64, N * W, P)
    wts = zeros(Int32, N, P)
    hnv = zeros(Int32, P); hnint = zeros(Int32, P)
    smax = maximum(p -> maximum(sweights[p]; init = 0), 1:P; init = 0)
    npl = max(1, 64 - leading_zeros(UInt64(smax)))
    planes = zeros(UInt64, W * npl, P)

    for p in 1:P
        g = graphs[p]; n = nv(g)
        @assert n <= N
        hnv[p] = n; hnint[p] = isnothing(nints) ? n : nints[p]

        for v in 1:n
            u = v - 1
            setbit!(adj, p, N, u, u)
            wts[v, p] = weights[p][v]
            sw = sweights[p][v]
            @assert sw >= 0

            for pl in 0:npl-1
                if (sw >> pl) & 1 == 1
                    planes[u >> 6 + 1 + W * pl, p] |= one(UInt64) << (u & 63)
                end
            end

            for x in neighbors(g, v)
                x == v && continue
                setbit!(adj, p, N, u, x - 1)
            end
        end
    end

    return Bucket{W}(CuArray(adj), CuArray(wts), CuArray(planes), Int32(npl), CuArray(hnv), CuArray(hnint), hnv, hnint, P)
end

@inline function setbit!(adj, p, N, u, x)
    j = x >> 6
    adj[u + j * N + 1, p] |= one(UInt64) << (x & 63)
end

# ------------------------------------------------------------ device bits --

@inline function warpsum(x::Int32)
    x += shfl_xor_sync(0xffffffff, x, 16)
    x += shfl_xor_sync(0xffffffff, x, 8)
    x += shfl_xor_sync(0xffffffff, x, 4)
    x += shfl_xor_sync(0xffffffff, x, 2)
    x += shfl_xor_sync(0xffffffff, x, 1)
    return x
end

@inline function warpmin(x::UInt64)
    for o in (16, 8, 4, 2, 1)
        y = shfl_xor_sync(0xffffffff, x, o)
        x = min(x, y)
    end
    return x
end

# log2(2^a + 2^b) in Float32
@inline function lse2(a::Float32, b::Float32)
    hi = max(a, b); lo = min(a, b)
    return hi + CUDA.log2(1.0f0 + CUDA.exp2(lo - hi))
end

@inline function fmix32(h::UInt32)
    h ⊻= h >> 16; h *= 0x85ebca6b
    h ⊻= h >> 13; h *= 0xc2b2ae35
    h ⊻= h >> 16
    return h
end

@inline hash3(a::UInt32, b::UInt32, c::UInt32) = fmix32(a * 0x9e3779b1 ⊻ fmix32(b + 0x7f4a7c15 ⊻ fmix32(c)))

# (width, log2 cost) as one sortable key; the cost is >= 0 so its float bits sort
@inline jobkey(w::Int32, lc::Float32) = (UInt64(reinterpret(UInt32, w)) << 32) | UInt64(reinterpret(UInt32, max(lc, 0.0f0)))

@inline function loadpiece!(sm, adj, p, ::Val{W}, lane) where {W}
    N = 64W
    i = lane
    while i < N * W
        @inbounds sm[i + 1] = adj[i + 1, p]
        i += 32
    end
    return
end

# closures must not capture variables that are reassigned (they get boxed), so
# the per-step tuple updates live in small functions.

# row v is read by lanes 0..W-1 (one word each) and broadcast by shuffles
@inline function getbag(sm, v::Int32, alive::NTuple{W, UInt64}, lane) where {W}
    x = lane < W ? @inbounds(sm[v + lane * 64W + 1]) : zero(UInt64)
    return ntuple(j -> shfl_sync(0xffffffff, x, j) & alive[j], Val(W))
end

# NB: `x & ~(1 << (v & 63))` is canonicalised by LLVM into a rotate whose
# NVPTX lowering feeds the unmasked amount v to shl/shr (PTX clamps amounts
# >= 64), which zeroes the whole word for v >= 64. Any and-not form gets
# canonicalised the same way, so the bit, which is known to be set, is
# subtracted instead.
@inline bitof(v::Int32) = one(UInt64) << (UInt32(v) & 0x3f)

@inline function clearbit(x::NTuple{W, UInt64}, v::Int32) where {W}
    jv = v >> 6 + 1; bv = bitof(v)
    return ntuple(j -> j == jv ? x[j] - bv : x[j], Val(W))
end

@inline isset(x::NTuple{W, UInt64}, i::Int, lane) where {W} =
    (x[(i - 1) >> 1 + 1] >> (lane + 32 * ((i - 1) & 1))) & one(UInt64) != zero(UInt64)

@inline function firstbits(n, ::Val{W}) where {W}
    return ntuple(j -> 64j <= n ? typemax(UInt64) : 64(j - 1) >= n ? zero(UInt64) : (one(UInt64) << (n - 64(j - 1))) - one(UInt64), Val(W))
end

# fill the bag into the rows of this lane that are in it; returns their weight
@inline function fillbag!(sm, bag::NTuple{W, UInt64}, wl, lane) where {W}
    N = 64W
    s = Int32(0)
    for i in 1:2W
        if isset(bag, i, lane)
            s += wl[i]
            u = lane + 32(i - 1)
            for jj in 1:W
                @inbounds sm[u + (jj - 1) * N + 1] |= bag[jj]
            end
        end
    end
    return s
end

# eliminate the vertices orders[1:nsteps, col]; returns (width, log2 cost, step of the first widest bag)
@inline function scoreloop!(sm, wl, lane, orders, col, nsteps, invq::Float32, ::Val{W}) where {W}
    alive = ntuple(_ -> typemax(UInt64), Val(W))
    wmax = Int32(-1); lc = -Inf32; tmax = Int32(0)
    @inbounds for t in Int32(1):nsteps
        v = Int32(orders[t, col])
        bag = getbag(sm, v, alive, lane)
        s = warpsum(fillbag!(sm, bag, wl, lane))
        if s > wmax
            wmax = s; tmax = t
        end
        lc = lse2(lc, Float32(s) * invq)
        alive = clearbit(alive, v)
        sync_warp()
    end
    return wmax, lc, tmax
end

@inline function warpslot(::Val{W}) where {W}
    tid = threadIdx().x - 1
    lane = tid % 32
    wib = tid ÷ 32
    wpb = blockDim().x ÷ 32
    g = (blockIdx().x - 1) * wpb + wib
    return lane, wib, wpb, g
end

# ------------------------------------------------------------ score kernel --

# orders[t, g]: 0-based local vertex eliminated by job g at step t (interior only)
function score_kernel!(::Val{W}, adj, wts, nints, pidx, orders, width, lcost, tmaxs, njobs::Int32, invq::Float32) where {W}
    N = 64W
    lane, wib, wpb, g = warpslot(Val(W))
    g >= njobs && return
    p = @inbounds pidx[g + 1]
    sm = CuDynamicSharedArray(UInt64, N * W, wib * N * W * 8)
    loadpiece!(sm, adj, p, Val(W), lane)
    wl = ntuple(i -> @inbounds(wts[lane + 32(i - 1) + 1, p]), Val(2W))
    sync_warp()
    wmax, lc, tmax = scoreloop!(sm, wl, lane, orders, g + 1, @inbounds(nints[p]), invq, Val(W))
    if lane == 0
        @inbounds width[g + 1] = wmax
        @inbounds lcost[g + 1] = lc
        @inbounds tmaxs[g + 1] = tmax
    end
    return
end

# ----------------------------------------------------------- mutate kernel --

"""
Local-search moves applied to the incumbent ordering of a piece before
scoring: `nmoves` moves, each a *move* (take the vertex at position i, insert
it at position j) with i the step of the widest bag with probability
`ptarget`, else uniform.
"""
struct MutateParams
    nmoves::Int32
    ptarget::Float32
    span::Int32       # |i - j| <= span (0: unbounded)
    seed::UInt32
end

function mutate_kernel!(::Val{W}, adj, wts, nints, pidx, best, besttmax, params, cand, width, lcost, tmaxs, njobs::Int32, invq::Float32) where {W}
    N = 64W
    lane, wib, wpb, g = warpslot(Val(W))
    g >= njobs && return
    p = @inbounds pidx[g + 1]
    n = @inbounds nints[p]
    prm = @inbounds params[g + 1]
    # copy the incumbent, then lane 0 applies the moves
    i = lane
    while i < n
        @inbounds cand[i + 1, g + 1] = best[i + 1, p]
        i += 32
    end
    sync_warp()
    if lane == 0 && n > 1
        tm = @inbounds besttmax[p]
        for r in Int32(1):prm.nmoves
            h1 = hash3(prm.seed, UInt32(r), 0x00000001)
            h2 = hash3(prm.seed, UInt32(r), 0x00000002)
            a = Float32(h1 >> 8) * 5.9604645f-8 < prm.ptarget && r == 1 ? tm : Int32(h1 % UInt32(n)) + Int32(1)
            a = clamp(a, Int32(1), n)
            b = if prm.span > 0
                d = Int32(h2 % UInt32(2prm.span + 1)) - prm.span
                clamp(a + d, Int32(1), n)
            else
                Int32(h2 % UInt32(n)) + Int32(1)
            end
            @inbounds x = cand[a, g + 1]
            if a < b
                for q in a:b-Int32(1)
                    @inbounds cand[q, g + 1] = cand[q + 1, g + 1]
                end
            else
                for q in a:Int32(-1):b+Int32(1)
                    @inbounds cand[q, g + 1] = cand[q - 1, g + 1]
                end
            end
            @inbounds cand[b, g + 1] = x
        end
    end
    sm = CuDynamicSharedArray(UInt64, N * W, wib * N * W * 8)
    loadpiece!(sm, adj, p, Val(W), lane)
    wl = ntuple(i -> @inbounds(wts[lane + 32(i - 1) + 1, p]), Val(2W))
    threadfence_block()
    sync_warp()
    wmax, lc, tmax = scoreloop!(sm, wl, lane, cand, g + 1, n, invq, Val(W))
    if lane == 0
        @inbounds width[g + 1] = wmax
        @inbounds lcost[g + 1] = lc
        @inbounds tmaxs[g + 1] = tmax
    end
    return
end

# ------------------------------------------------------------ select best --

# block b covers jobs bstart[b] .. bstart[b+1]-1, all on piece bpiece[b]; the
# least (width, cost) job replaces the incumbent of that piece if smaller.
function select_kernel!(bstart, bpiece, cand, width, lcost, tmaxs, best, bestw, bestc, besttmax, nints, nblocks::Int32)
    # NB keep every index Int32: a Union{Int32, Int64} job index gets
    # union-split into lane-divergent paths with different shuffles (garbage)
    lane = Int32(threadIdx().x) - Int32(1)
    b = blockIdx().x
    b > nblocks && return
    j0 = @inbounds bstart[b]; j1 = @inbounds bstart[b + 1] - Int32(1)
    p = @inbounds bpiece[b]
    key = typemax(UInt64)
    j = j0 + lane
    while j <= j1
        key = min(key, jobkey(@inbounds(width[j]), @inbounds(lcost[j])))
        j += Int32(32)
    end
    key = warpmin(key)
    # the first job with the least key
    jb = typemax(Int32)
    j = j0 + lane
    while j <= j1
        if jobkey(@inbounds(width[j]), @inbounds(lcost[j])) == key
            jb = min(jb, j)
        end
        j += Int32(32)
    end
    for o in (16, 8, 4, 2, 1)
        jb = min(jb, shfl_xor_sync(0xffffffff, jb, o))
    end
    jbest = jb
    wb = @inbounds width[jbest]; cb = @inbounds lcost[jbest]
    better = jobkey(wb, cb) < jobkey(@inbounds(bestw[p]), @inbounds(bestc[p]))
    sync_warp()   # every lane has read the incumbent before lane 0 overwrites it
    if better
        n = @inbounds nints[p]
        i = lane
        while i < n
            @inbounds best[i + 1, p] = cand[i + 1, jbest]
            i += 32
        end
        if lane == 0
            @inbounds bestw[p] = wb; @inbounds bestc[p] = cb; @inbounds besttmax[p] = tmaxs[jbest]
        end
    end
    return
end

# ----------------------------------------------------------- greedy kernel --

"""
Per-ordering greedy parameters: score(u) = cdeg*w(N[u]) + cfill*fill(u) +
cavg*fill(u)/w(N[u]), multiplied by (1 + noise*U) with U ~ U[0,1) drawn per
(vertex, step); ties broken by a random key. fill(u) = sum over missing pairs
{a,b} of N(u) of w(a)*w(b). Scores use the score weights (bit planes).
"""
struct GreedyParams
    cdeg::Float32
    cfill::Float32
    cavg::Float32
    noise::Float32
    seed::UInt32
end

# score weight of a set: sum over planes of popcounts
@inline function swsum(x::NTuple{W, UInt64}, pl, npl::Int32) where {W}
    s = Int32(0)
    for q in Int32(0):npl - Int32(1)
        c = Int32(0)
        for j in 1:W
            c += Int32(count_ones(x[j] & @inbounds(pl[j + W * q])))
        end
        s += c << q
    end
    return s
end

@inline function swone(pl, a::Int32, npl::Int32, ::Val{W}) where {W}
    ja = a >> 6 + 1; sh = UInt32(a) & 0x3f
    s = Int32(0)
    for q in Int32(0):npl - Int32(1)
        s += Int32((@inbounds(pl[ja + W * q]) >> sh) & one(UInt64)) << q
    end
    return s
end

# weighted degree w(N[u] ∩ alive) and weighted fill of row u
@inline function rowscores(sm, pl, npl::Int32, u::Int32, alive::NTuple{W, UInt64}, dofill::Bool) where {W}
    N = Int32(64W)
    row = ntuple(j -> @inbounds(sm[u + (j - 1) * N + 1]) & alive[j], Val(W))
    d = swsum(row, pl, npl)
    f = 0.0f0
    if dofill
        nu = clearbit(row, u)   # open neighbourhood
        for j in 1:W
            x = nu[j]
            while x != zero(UInt64)
                b = Int32(trailing_zeros(x)); x &= x - one(UInt64)
                a = Int32(64(j - 1)) + b
                miss = ntuple(jj -> nu[jj] & ~@inbounds(sm[a + (jj - 1) * N + 1]), Val(W))
                f += Float32(swone(pl, a, npl, Val(W))) * Float32(swsum(miss, pl, npl))
            end
        end
        f *= 0.5f0
    end
    return d, f
end

@inline function combine(d::Int32, f::Float32, prm::GreedyParams)
    df = Float32(d)
    return prm.cdeg * df + prm.cfill * f + (prm.cavg != 0.0f0 ? prm.cavg * f / max(df, 1.0f0) : 0.0f0)
end

@inline function pickkey(sc::NTuple{R, Float32}, sel::NTuple{W, UInt64}, lane, prm::GreedyParams, t::UInt32) where {R, W}
    best = typemax(UInt64)
    for i in 1:R
        if isset(sel, i, lane)
            u = UInt32(lane + 32(i - 1))
            h = hash3(prm.seed, t, u)
            s = sc[i]
            if prm.noise != 0.0f0
                s *= 1.0f0 + prm.noise * Float32(h >> 8) * 5.9604645f-8
            end
            key = (UInt64(reinterpret(UInt32, max(s, 0.0f0))) << 32) | (UInt64(h & 0x003fffff) << 10) | UInt64(u)
            best = min(best, key)
        end
    end
    return best
end

@inline function warpsumf(x::Float32)
    for o in (16, 8, 4, 2, 1)
        x += shfl_xor_sync(0xffffffff, x, o)
    end
    return x
end

# weighted fill of row u, warp-cooperatively: lane l handles the neighbours
# a of u among its own rows (a = l, l+32, ...), so the work is spread evenly
@inline function coopfill(sm, pl, npl::Int32, u::Int32, alive::NTuple{W, UInt64}, swl, lane) where {W}
    N = Int32(64W)
    row = ntuple(j -> @inbounds(sm[u + (j - 1) * N + 1]) & alive[j], Val(W))
    nu = clearbit(row, u)
    part = 0.0f0
    for i in 1:2W
        if isset(nu, i, lane)
            a = Int32(lane + 32(i - 1))
            miss = ntuple(jj -> nu[jj] & ~@inbounds(sm[a + (jj - 1) * N + 1]), Val(W))
            part += Float32(swl[i]) * Float32(swsum(miss, pl, npl))
        end
    end
    return 0.5f0 * warpsumf(part), swsum(row, pl, npl)
end

# recompute the scores of the rows in `todo` (one bool per row of this lane):
# rows are visited warp-uniformly (ballots), each fill computed cooperatively
@inline function updatescores(sm, pl, npl, sc::NTuple{R, Float32}, todo::NTuple{R, Bool}, alive::NTuple{W, UInt64}, swl, lane, prm::GreedyParams) where {R, W}
    for i in 1:R
        m = vote_ballot_sync(0xffffffff, todo[i])
        while m != zero(UInt32)
            l = Int32(trailing_zeros(m)); m &= m - one(UInt32)
            u = l + Int32(32(i - 1))
            f, d = coopfill(sm, pl, npl, u, alive, swl, lane)
            v = combine(d, f, prm)
            sc = lane == l ? Base.setindex(sc, v, i) : sc
        end
    end
    return sc
end

@inline function deltodo(sc::NTuple{R, Float32}, sel::NTuple{W, UInt64}, lane) where {R, W}
    return ntuple(i -> isset(sel, i, lane) ? sc[i] : Inf32, Val(R))
end

# rescore affected rows: degree changes in the bag; fill changes for rows
# meeting the bag in at least two vertices
@inline function rescore(sm, pl, npl, sc::NTuple{R, Float32}, alive::NTuple{W, UInt64}, sel::NTuple{W, UInt64}, bag::NTuple{W, UInt64}, swl, lane, prm::GreedyParams, dofill::Bool) where {R, W}
    N = 64W
    if dofill
        todo = ntuple(Val(R)) do i
            u = lane + 32(i - 1)
            c = 0
            for jj in 1:W
                c += count_ones(@inbounds(sm[u + (jj - 1) * N + 1]) & bag[jj])
            end
            isset(sel, i, lane) && c >= 2
        end
        return deltodo(updatescores(sm, pl, npl, sc, todo, alive, swl, lane, prm), sel, lane)
    end
    return ntuple(Val(R)) do i
        u = Int32(lane + 32(i - 1))
        if !isset(sel, i, lane)
            Inf32
        elseif isset(bag, i, lane)
            combine(rowscores(sm, pl, npl, u, alive, false)..., prm)
        else
            sc[i]
        end
    end
end

@inline function initscores(sm, pl, npl, alive::NTuple{W, UInt64}, sel::NTuple{W, UInt64}, swl, lane, prm::GreedyParams, dofill::Bool) where {W}
    if dofill
        sc0 = ntuple(_ -> Inf32, Val(2W))
        todo = ntuple(i -> isset(sel, i, lane), Val(2W))
        return updatescores(sm, pl, npl, sc0, todo, alive, swl, lane, prm)
    end
    return ntuple(Val(2W)) do i
        u = Int32(lane + 32(i - 1))
        isset(sel, i, lane) ? combine(rowscores(sm, pl, npl, u, alive, false)..., prm) : Inf32
    end
end

function greedy_kernel!(::Val{W}, adj, wts, planes, npl::Int32, nvs, nints, pidx, params, order, width, lcost, tmaxs, njobs::Int32, invq::Float32) where {W}
    N = 64W
    lane, wib, wpb, g = warpslot(Val(W))
    g >= njobs && return
    p = @inbounds pidx[g + 1]
    prm = @inbounds params[g + 1]
    dofill = prm.cfill != 0.0f0 || prm.cavg != 0.0f0

    sm = CuDynamicSharedArray(UInt64, N * W, wib * N * W * 8)
    pl = CuDynamicSharedArray(UInt64, W * npl, wpb * N * W * 8 + wib * W * npl * 8)
    loadpiece!(sm, adj, p, Val(W), lane)
    i = lane
    while i < W * npl
        @inbounds pl[i + 1] = planes[i + 1, p]
        i += 32
    end
    n = @inbounds nvs[p]; nint = @inbounds nints[p]
    wl = ntuple(i -> @inbounds(wts[lane + 32(i - 1) + 1, p]), Val(2W))
    alive = firstbits(n, Val(W))
    sel = firstbits(nint, Val(W))
    sync_warp()
    swl = ntuple(i -> swone(pl, Int32(lane + 32(i - 1)), npl, Val(W)), Val(2W))
    sc = initscores(sm, pl, npl, alive, sel, swl, lane, prm, dofill)

    wmax = Int32(-1); lc = -Inf32; tmax = Int32(0)
    @inbounds for t in Int32(1):nint
        best = warpmin(pickkey(sc, sel, lane, prm, UInt32(t)))
        v = Int32(best & 0x3ff)
        lane == 0 && (order[t, g + 1] = Int16(v))
        bag = getbag(sm, v, alive, lane)
        s = warpsum(fillbag!(sm, bag, wl, lane))
        if s > wmax
            wmax = s; tmax = t
        end
        lc = lse2(lc, Float32(s) * invq)
        alive = clearbit(alive, v)
        sel = clearbit(sel, v)
        sync_warp()
        sc = rescore(sm, pl, npl, sc, alive, sel, bag, swl, lane, prm, dofill)
        sync_warp()
    end

    if lane == 0
        @inbounds width[g + 1] = wmax
        @inbounds lcost[g + 1] = lc
        @inbounds tmaxs[g + 1] = tmax
    end
    return
end

# ------------------------------------------------------------ host driver --

function devinfo(dev = CUDA.device())
    return (
        name = CUDA.name(dev),
        sms = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        smem_block_optin = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN),
        smem_sm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR),
        threads_sm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR),
        cc = CUDA.capability(dev),
    )
end

# Warps per block from device attributes only: the per-warp shared slice
# fixes how many warps fit on an SM; a block takes up to a quarter of that so
# several blocks can be resident (and tails are short), capped at 8 warps.
function launchcfg(kernel, bytes_per_warp::Int, njobs::Int)
    dev = CUDA.device()
    smsm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
    optin = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    maxthr = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    kthr = CUDA.maxthreads(kernel)
    wsm = max(1, min(smsm ÷ max(bytes_per_warp, 1), maxthr ÷ 32))
    wpb = clamp(wsm ÷ 4, 1, 8)
    wpb = max(1, min(wpb, optin ÷ max(bytes_per_warp, 1), kthr ÷ 32, max(njobs, 1)))
    shmem = wpb * bytes_per_warp
    shmem > 48 * 1024 && (CUDA.attributes(kernel.fun)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem)
    return wpb, shmem
end

function score!(width, lcost, tmaxs, b::Bucket{W}, pidx::CuVector{Int32}, orders::CuMatrix{Int16}; invq::Real = 1.0) where {W}
    njobs = length(pidx); N = 64W
    njobs == 0 && return
    args = (Val(W), b.adj, b.wts, b.nints, pidx, orders, width, lcost, tmaxs, Int32(njobs), Float32(invq))
    kernel = @cuda launch = false score_kernel!(args...)
    wpb, shmem = launchcfg(kernel, N * W * 8, njobs)
    kernel(args...; threads = 32wpb, blocks = cld(njobs, wpb), shmem)
    return
end

function greedy!(order, width, lcost, tmaxs, b::Bucket{W}, pidx::CuVector{Int32}, params::CuVector{GreedyParams}; invq::Real = 1.0) where {W}
    njobs = length(pidx); N = 64W
    njobs == 0 && return
    args = (Val(W), b.adj, b.wts, b.planes, b.npl, b.nvs, b.nints, pidx, params, order, width, lcost, tmaxs, Int32(njobs), Float32(invq))
    kernel = @cuda launch = false greedy_kernel!(args...)
    wpb, shmem = launchcfg(kernel, N * W * 8 + W * Int(b.npl) * 8, njobs)
    kernel(args...; threads = 32wpb, blocks = cld(njobs, wpb), shmem)
    return
end

function mutate!(cand, width, lcost, tmaxs, b::Bucket{W}, pidx::CuVector{Int32}, best, besttmax, params::CuVector{MutateParams}; invq::Real = 1.0) where {W}
    njobs = length(pidx); N = 64W
    njobs == 0 && return
    args = (Val(W), b.adj, b.wts, b.nints, pidx, best, besttmax, params, cand, width, lcost, tmaxs, Int32(njobs), Float32(invq))
    kernel = @cuda launch = false mutate_kernel!(args...)
    wpb, shmem = launchcfg(kernel, N * W * 8, njobs)
    kernel(args...; threads = 32wpb, blocks = cld(njobs, wpb), shmem)
    return
end

function selectbest!(bstart::CuVector{Int32}, bpiece::CuVector{Int32}, cand, width, lcost, tmaxs, best, bestw, bestc, besttmax, b::Bucket)
    nb = length(bpiece)
    nb == 0 && return
    @cuda threads = 32 blocks = nb select_kernel!(bstart, bpiece, cand, width, lcost, tmaxs, best, bestw, bestc, besttmax, b.nints, Int32(nb))
    return
end

# ---------------------------------------------------------- CPU reference --

# The same algorithm on the CPU (bitsets, one ordering at a time): (width, log2 cost)
function cpu_score_bitset(rows0::Matrix{UInt64}, w::Vector{Int32}, order::AbstractVector, invq::Float64 = 1.0)
    W, n = size(rows0)
    rows = copy(rows0)
    alive = zeros(UInt64, W)
    for v in 0:n-1
        alive[v >> 6 + 1] |= one(UInt64) << (v & 63)
    end
    bag = zeros(UInt64, W)
    wmax = Int32(0); lc = -Inf
    @inbounds for v1 in order
        v = v1 - 1
        for j in 1:W
            bag[j] = rows[j, v1] & alive[j]
        end
        s = Int32(0)
        for j in 1:W
            x = bag[j]
            while x != 0
                b = trailing_zeros(x); x &= x - 1
                u = 64(j - 1) + b + 1
                s += w[u]
                for jj in 1:W
                    rows[jj, u] |= bag[jj]
                end
            end
        end
        wmax = max(wmax, s)
        x = s * invq
        lc = lc == -Inf ? x : max(lc, x) + log2(1 + exp2(-abs(lc - x)))
        alive[v >> 6 + 1] &= ~(one(UInt64) << (v & 63))
    end
    return wmax, lc
end

function cpurows(g, W)
    n = nv(g)
    rows = zeros(UInt64, W, n)
    for v in 1:n
        rows[(v - 1) >> 6 + 1, v] |= one(UInt64) << ((v - 1) & 63)
        for x in neighbors(g, v)
            rows[(x - 1) >> 6 + 1, v] |= one(UInt64) << ((x - 1) & 63)
        end
    end
    return rows
end

end
