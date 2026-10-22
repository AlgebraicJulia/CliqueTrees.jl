# Batched exact subtree reconfiguration on the GPU (cotengra's subtree_reconfigure, many subtrees at once).
#
# A chunk is a connected set I of internal nodes of a contraction tree, rooted at r, with frontier F
# (k = |I| + 1 <= 12 nodes whose subtrees stay as they are). Re-contracting F optimally is a DP over
# subsets of F. Local indices u (the legs of the frontier nodes) are described by
#     mask_u = frontier nodes having u as a leg,   inout_u = u is a leg of r (it survives the chunk)
# and for S subset of F (all exponents log2):
#     legs(S)          = {u : mask_u meets S and (mask_u not inside S or inout_u)}
#     involved(S1, S2) = {u : mask_u meets S} minus internal(S1) minus internal(S2),
#                        internal(X) = {u : mask_u inside X and not inout_u}
# so with Z(X) = w{u : mask_u inside X} and Zi(X) = w{internal(X)} (zeta transforms of point masses):
#     flops(S1, S2) = 2^( Zall(F) - Zall(F \ S) - Zi(S1) - Zi(S2) ),
#     best(S) = min over splits {S1, S2} of best(S1) + best(S2) + flops(S1, S2),  best({f}) = 0.
# One thread block per chunk; subsets processed by popcount level, one thread per subset, all its
# splits enumerated (the split containing the lowest bit, 2^(|S|-1) - 1 of them). Costs are Float32
# relative to 2^off (off = the largest flops exponent of the current chunk); the host re-costs exactly.
module GPUReconf

using CUDA

export DPBatch, dp_solve!

const KMAX = 12

# subsets of {0..k-1} ordered by popcount, for every k; LV[k][j] = range of popcount-j subsets
function levels()
    sets = Int32[]; kofs = zeros(Int32, KMAX + 1); lptr = zeros(Int32, KMAX + 2, KMAX + 1)
    for k in 1:KMAX
        kofs[k] = length(sets)
        for j in 0:k
            lptr[j+1, k] = length(sets) - kofs[k]
            for S in 0:(1<<k)-1
                count_ones(S) == j && push!(sets, Int32(S))
            end
        end
        lptr[k+2, k] = length(sets) - kofs[k]
    end
    sets, kofs, lptr
end

const HOSTLV = levels()

function dp_kernel!(best_out, arg, kk, uptr, umask, uin, uw, off, sets, kofs, lptr, K::Int32)
    c = Int32(blockIdx().x)
    tid = Int32(threadIdx().x) - Int32(1); nth = Int32(blockDim().x)
    k = kk[c]
    M = Int32(1) << k; full = M - Int32(1)
    sh = CuDynamicSharedArray(Float32, (Int32(3) << K,))
    Za = view(sh, 1:(1<<K)); Zi = view(sh, (1<<K)+1:(2<<K)); best = view(sh, (2<<K)+1:(3<<K))
    @inbounds begin
        x = tid
        while x < M
            Za[x+1] = 0.0f0; Zi[x+1] = 0.0f0; best[x+1] = 0.0f0
            x += nth
        end
        sync_threads()
        e = uptr[c] + tid
        while e < uptr[c+1]
            m = Int32(umask[e+1]); wv = uw[e+1]
            CUDA.@atomic Za[m+1] += wv
            uin[e+1] == 0x00 && (CUDA.@atomic Zi[m+1] += wv)
            e += nth
        end
        sync_threads()
        for b in Int32(0):k-Int32(1)
            bit = Int32(1) << b
            x = tid
            while x < M
                if x & bit != Int32(0)
                    Za[x+1] += Za[(x⊻bit)+1]; Zi[x+1] += Zi[(x⊻bit)+1]
                end
                x += nth
            end
            sync_threads()
        end
        tot = Za[full+1]; o = off[c]
        base = (c - Int32(1)) * (Int32(1) << K)
        ko = kofs[k]
        for j in Int32(2):k
            lo = lptr[j+1, k]; hi = lptr[j+2, k]
            q = lo + tid
            while q < hi
                S = sets[ko+q+1]
                ts = tot - Za[(full⊻S)+1] - o
                low = S & (-S); R = S ⊻ low
                s = (R - Int32(1)) & R
                bb = Inf32; ba = Int32(0)
                while true
                    S1 = s | low; S2 = S ⊻ S1
                    v = best[S1+1] + best[S2+1] + exp2(ts - Zi[S1+1] - Zi[S2+1])
                    if v < bb
                        bb = v; ba = S1
                    end
                    s == Int32(0) && break
                    s = (s - Int32(1)) & R
                end
                best[S+1] = bb; arg[base+S+1] = UInt16(ba)
                q += nth
            end
            sync_threads()
        end
        tid == Int32(0) && (best_out[c] = best[full+1])
    end
    return
end

struct DPBatch
    kk::Vector{Int32}
    uptr::Vector{Int32}
    umask::Vector{UInt16}
    uin::Vector{UInt8}
    uw::Vector{Float32}
    off::Vector{Float32}
end
DPBatch() = DPBatch(Int32[], Int32[0], UInt16[], UInt8[], Float32[], Float32[])
Base.length(b::DPBatch) = length(b.kk)

const DEVLV = Ref{Any}(nothing)

"""Solve all chunks of `b`; returns (best::Vector{Float32} (relative to 2^off), arg::Matrix{UInt16}
(2^K x nchunks; arg[S+1, c] = the part S1 of the optimal split of S, containing S's lowest bit))."""
function dp_solve!(b::DPBatch; K::Int = Int(maximum(b.kk; init = Int32(2))), threads::Int = 256)
    C = length(b); C == 0 && return Float32[], zeros(UInt16, 0, 0)
    if DEVLV[] === nothing
        s, ko, lp = HOSTLV
        DEVLV[] = (CuArray(s), CuArray(ko), CuArray(lp))
    end
    sets, kofs, lptr = DEVLV[]
    dkk = CuArray(b.kk); duptr = CuArray(b.uptr); dm = CuArray(b.umask); din = CuArray(b.uin); dw = CuArray(b.uw); doff = CuArray(b.off)
    best = CUDA.zeros(Float32, C); arg = CUDA.zeros(UInt16, (1 << K) * C)
    shmem = 3 * 4 * (1 << K)
    @cuda threads = threads blocks = C shmem = shmem dp_kernel!(best, arg, dkk, duptr, dm, din, dw, doff, sets, kofs, lptr, Int32(K))
    Array(best), reshape(Array(arg), 1 << K, C)
end

end
