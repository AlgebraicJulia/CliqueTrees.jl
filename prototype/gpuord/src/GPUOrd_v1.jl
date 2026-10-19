module GPUOrd

# Batched, bit-parallel elimination on padded bitset adjacency matrices.
#
# A *bucket* holds P pieces with at most N = 64W vertices each. Piece p is
# stored word-major: word j (0-based) of row u is adj[u + j*N + 1, p], so the
# lanes of a warp, which own consecutive rows, touch consecutive words. The
# diagonal is set, so row v masked by the alive set is the closed bag N^+[v].
#
# One warp runs one (piece, ordering) job; lane l owns the rows l, l+32, ...
# (2W of them). Eliminating v: bag = row v & alive; every lane ORs the bag
# into those of its rows that are in the bag (fill); the bag weight is a warp
# sum of the weights of the set bits.

using CUDA
using Graphs

export Bucket, pack, score!, greedy!, cpu_score, cpu_score_bitset, devinfo,
    GreedyParams, launchcfg

const LOG2E = 1.4426950408889634

# ---------------------------------------------------------------- packing --

struct Bucket{W}
    adj::CuMatrix{UInt64}     # (N*W, P)
    wts::CuMatrix{Int32}      # (N, P)
    nvs::CuVector{Int32}      # (P,)
    hnv::Vector{Int32}
    P::Int
end

nrows(::Bucket{W}) where {W} = 64W
bucketwords(n::Integer) = n <= 64 ? 1 : n <= 128 ? 2 : n <= 256 ? 4 : n <= 512 ? 8 : cld(n, 64)

"""
    pack(graphs, weights, W) -> Bucket{W}

Pad each graph (n <= 64W) to N = 64W rows of W words. `weights[i]` are the
Int32 vertex weights of `graphs[i]`.
"""
function pack(graphs::AbstractVector, weights::AbstractVector, W::Int)
    N = 64W; P = length(graphs)
    adj = zeros(UInt64, N * W, P)
    wts = zeros(Int32, N, P)
    hnv = zeros(Int32, P)

    for p in 1:P
        g = graphs[p]; n = nv(g)
        @assert n <= N
        hnv[p] = n

        for v in 1:n
            u = v - 1
            setbit!(adj, p, N, u, u)
            wts[v, p] = weights[p][v]

            for x in neighbors(g, v)
                x == v && continue
                setbit!(adj, p, N, u, x - 1)
            end
        end
    end

    return Bucket{W}(CuArray(adj), CuArray(wts), CuArray(hnv), hnv, P)
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

# copy the piece into this warp's shared slice
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
# the per-step tuple updates live in small functions
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

@inline function clearbit(alive::NTuple{W, UInt64}, v::Int32) where {W}
    jv = v >> 6 + 1; bv = bitof(v)
    return ntuple(j -> j == jv ? alive[j] - bv : alive[j], Val(W))
end

@inline isset(x::NTuple{W, UInt64}, i::Int, lane) where {W} =
    (x[(i - 1) >> 1 + 1] >> (lane + 32 * ((i - 1) & 1))) & one(UInt64) != zero(UInt64)

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

# ------------------------------------------------------------ score kernel --

# jobs g = (p - 1) * K + k; orders[t, g] is the 0-based local vertex eliminated
# at step t; width[g] (Int32, exact); lcost[g] = log2 sum_t 2^(w(bag_t) * invq)
function score_kernel!(::Val{W}, adj, wts, nvs, orders, K::Int32, width, lcost, njobs::Int32, invq::Float32) where {W}
    N = 64W
    tid = threadIdx().x - 1
    lane = tid % 32
    wib = tid ÷ 32
    wpb = blockDim().x ÷ 32
    g = (blockIdx().x - 1) * wpb + wib
    g >= njobs && return
    p = g ÷ K + 1

    sm = CuDynamicSharedArray(UInt64, N * W, wib * N * W * 8)
    loadpiece!(sm, adj, p, Val(W), lane)
    wl = ntuple(i -> @inbounds(wts[lane + 32(i - 1) + 1, p]), Val(2W))
    n = @inbounds nvs[p]
    alive = ntuple(_ -> typemax(UInt64), Val(W))
    sync_warp()

    wmax = Int32(0)
    lc = -Inf32
    @inbounds for t in 1:n
        v = Int32(orders[t, g + 1])
        bag = getbag(sm, v, alive, lane)
        s = warpsum(fillbag!(sm, bag, wl, lane))
        wmax = max(wmax, s)
        lc = lse2(lc, Float32(s) * invq)
        alive = clearbit(alive, v)
        sync_warp()
    end

    if lane == 0
        @inbounds width[g + 1] = wmax
        @inbounds lcost[g + 1] = lc
    end
    return
end

# ----------------------------------------------------------- greedy kernel --

"""
Per-ordering greedy parameters: score(u) = cdeg*w(N[u]) + cfill*fill(u) +
cavg*fill(u)/w(N[u]), multiplied by (1 + noise*U) with U ~ U[0,1) drawn per
(vertex, step); ties broken by a random key. fill(u) = sum over missing pairs
{a,b} of N(u) of w(a)*w(b).
"""
struct GreedyParams
    cdeg::Float32
    cfill::Float32
    cavg::Float32
    noise::Float32
    seed::UInt32
end

@inline function wsum_bits(x::UInt64, wsm, base::Int32)
    s = Int32(0)
    while x != zero(UInt64)
        b = Int32(trailing_zeros(x))
        @inbounds s += wsm[base + b + 1]
        x &= x - one(UInt64)
    end
    return s
end

# weighted degree w(N[u] ∩ alive) and weighted fill of row u
@inline function rowscores(sm, wsm, u::Int32, alive::NTuple{W, UInt64}, dofill::Bool) where {W}
    N = Int32(64W)
    row = ntuple(j -> @inbounds(sm[u + (j - 1) * N + 1]) & alive[j], Val(W))
    d = Int32(0)
    for j in 1:W
        d += wsum_bits(row[j], wsm, Int32(64(j - 1)))
    end
    f = 0.0f0
    if dofill
        ju = u >> 6 + 1; bu = bitof(u)
        nu = ntuple(j -> j == ju ? row[j] - bu : row[j], Val(W))   # open neighbourhood
        for j in 1:W
            x = nu[j]
            while x != zero(UInt64)
                b = Int32(trailing_zeros(x)); x &= x - one(UInt64)
                a = Int32(64(j - 1)) + b
                fa = Int32(0)
                for jj in 1:W
                    miss = nu[jj] & ~@inbounds(sm[a + (jj - 1) * N + 1])
                    fa += wsum_bits(miss, wsm, Int32(64(jj - 1)))
                end
                f += Float32(@inbounds(wsm[a + 1])) * Float32(fa)
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

@inline function pickkey(sc::NTuple{R, Float32}, alive::NTuple{W, UInt64}, lane, prm::GreedyParams, t::UInt32) where {R, W}
    best = typemax(UInt64)
    for i in 1:R
        if isset(alive, i, lane)
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

# rescore affected rows: degree changes in the bag; fill changes for rows
# meeting the bag in at least two vertices
@inline function rescore(sm, wsm, sc::NTuple{R, Float32}, alive::NTuple{W, UInt64}, bag::NTuple{W, UInt64}, lane, prm::GreedyParams, dofill::Bool) where {R, W}
    N = 64W
    return ntuple(Val(R)) do i
        u = Int32(lane + 32(i - 1))
        if !isset(alive, i, lane)
            Inf32
        else
            hit = if dofill
                c = 0
                for jj in 1:W
                    c += count_ones(@inbounds(sm[u + (jj - 1) * N + 1]) & bag[jj])
                end
                c >= 2
            else
                isset(bag, i, lane)
            end
            hit ? combine(rowscores(sm, wsm, u, alive, dofill)..., prm) : sc[i]
        end
    end
end

@inline function initscores(sm, wsm, alive::NTuple{W, UInt64}, lane, n, prm::GreedyParams, dofill::Bool) where {W}
    return ntuple(Val(2W)) do i
        u = Int32(lane + 32(i - 1))
        u < n ? combine(rowscores(sm, wsm, u, alive, dofill)..., prm) : Inf32
    end
end

@inline function initalive(n, ::Val{W}) where {W}
    return ntuple(j -> 64j <= n ? typemax(UInt64) : 64(j - 1) >= n ? zero(UInt64) : (one(UInt64) << (n - 64(j - 1))) - one(UInt64), Val(W))
end

# jobs g = (p - 1) * K + k, params[g]. Writes order[t, g], width[g], lcost[g].
function greedy_kernel!(::Val{W}, adj, wts, nvs, params, K::Int32, order, width, lcost, njobs::Int32, invq::Float32) where {W}
    N = 64W
    tid = threadIdx().x - 1
    lane = tid % 32
    wib = tid ÷ 32
    wpb = blockDim().x ÷ 32
    g = (blockIdx().x - 1) * wpb + wib
    g >= njobs && return
    p = g ÷ K + 1
    prm = @inbounds params[g + 1]
    dofill = prm.cfill != 0.0f0 || prm.cavg != 0.0f0

    sm = CuDynamicSharedArray(UInt64, N * W, wib * N * W * 8)
    wsm = CuDynamicSharedArray(Int32, N, wpb * N * W * 8 + wib * N * 4)
    loadpiece!(sm, adj, p, Val(W), lane)
    i = lane
    while i < N
        @inbounds wsm[i + 1] = wts[i + 1, p]
        i += 32
    end
    n = @inbounds nvs[p]
    wl = ntuple(i -> @inbounds(wts[lane + 32(i - 1) + 1, p]), Val(2W))
    alive = initalive(n, Val(W))
    sync_warp()
    sc = initscores(sm, wsm, alive, lane, n, prm, dofill)

    wmax = Int32(0)
    lc = -Inf32
    @inbounds for t in Int32(1):n
        # choose v: least perturbed score, random tie-break, among alive rows
        best = pickkey(sc, alive, lane, prm, UInt32(t))
        best = warpmin(best)
        v = Int32(best & 0x3ff)
        lane == 0 && (order[t, g + 1] = Int16(v))

        bag = getbag(sm, v, alive, lane)
        s = warpsum(fillbag!(sm, bag, wl, lane))
        wmax = max(wmax, s)
        lc = lse2(lc, Float32(s) * invq)
        alive = clearbit(alive, v)
        sync_warp()
        sc = rescore(sm, wsm, sc, alive, bag, lane, prm, dofill)
        sync_warp()
    end

    if lane == 0
        @inbounds width[g + 1] = wmax
        @inbounds lcost[g + 1] = lc
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
    wpb = min(wpb, optin ÷ max(bytes_per_warp, 1), kthr ÷ 32, max(njobs, 1))
    wpb = max(wpb, 1)
    shmem = wpb * bytes_per_warp
    shmem > 48 * 1024 && CUDA.cufuncattribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem)
    return wpb, shmem
end

"""
    score!(width, lcost, b::Bucket{W}, orders::CuMatrix{Int16}, K; invq)

Score K orderings per piece: `orders[:, (p-1)*K + k]` (0-based local vertices,
first nv(p) entries used).
"""
function score!(width::CuVector{Int32}, lcost::CuVector{Float32}, b::Bucket{W}, orders::CuMatrix{Int16}, K::Integer; invq::Real = 1.0) where {W}
    njobs = b.P * K
    N = 64W
    kernel = @cuda launch = false score_kernel!(Val(W), b.adj, b.wts, b.nvs, orders, Int32(K), width, lcost, Int32(njobs), Float32(invq))
    wpb, shmem = launchcfg(kernel, N * W * 8, njobs)
    kernel(Val(W), b.adj, b.wts, b.nvs, orders, Int32(K), width, lcost, Int32(njobs), Float32(invq);
        threads = 32wpb, blocks = cld(njobs, wpb), shmem)
    return width, lcost
end

function greedy!(order::CuMatrix{Int16}, width::CuVector{Int32}, lcost::CuVector{Float32}, b::Bucket{W}, params::CuVector{GreedyParams}, K::Integer; invq::Real = 1.0) where {W}
    njobs = b.P * K
    N = 64W
    kernel = @cuda launch = false greedy_kernel!(Val(W), b.adj, b.wts, b.nvs, params, Int32(K), order, width, lcost, Int32(njobs), Float32(invq))
    wpb, shmem = launchcfg(kernel, N * W * 8 + N * 4, njobs)
    kernel(Val(W), b.adj, b.wts, b.nvs, params, Int32(K), order, width, lcost, Int32(njobs), Float32(invq);
        threads = 32wpb, blocks = cld(njobs, wpb), shmem)
    return order, width, lcost
end

# ---------------------------------------------------------- CPU reference --

# Same algorithm on the CPU (bitsets, one ordering at a time): (width, log2 cost)
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
