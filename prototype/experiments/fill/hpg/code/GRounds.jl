# GPU rounds: the CPU compresses the graph and computes the initial scores,
# the GPU runs the bulk-synchronous rounds (score, distance-2 Luby
# selection, elimination with Wing-Huang deficiency updates, supervariable
# detection), and the CPU finishes the small dense tail with the serial
# algorithm.
#
# Adjacency lists live in one pool and stay sorted; an elimination writes the
# new list of each clique vertex to fresh pool space (the old space is
# garbage, reclaimed by compaction when the pool fills up).
module GRounds

using CUDA, CliqueTrees, Graphs
using ..Rounds
using ..Rounds: WSCALE, State, Work, compress!, initdef!, serialalg, mix

export grounds, GPUHybrid

const ALIVE = 0x00
const ABSORBED = 0x01
const DEAD = 0x02

@inline function hash32(v::Int32, seed::UInt32)
    x = reinterpret(UInt32, v) * 0x9e3779b1 + seed * 0x85ebca6b
    x ⊻= x >> 16; x *= 0x7feb352d; x ⊻= x >> 15; x *= 0x846ca68b; x ⊻= x >> 16
    return x
end

@inline function dmix(v::Int32)
    x = UInt64(reinterpret(UInt32, v)) * 0x9e3779b97f4a7c15
    x ⊻= x >> 32
    return x * 0xd6e8feb86659fd93
end

# is c in the sorted list pool[s:s+l-1]?
@inline function bsearch(pool, s::Int64, l::Int32, c::Int32)
    lo = s; hi = s + l - 1
    @inbounds while lo <= hi
        mid = (lo + hi) >> 1
        x = pool[mid]
        if x == c
            return true
        elseif x < c
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return false
end

@inline function warpsum(x::Int64)
    off = 16
    while off > 0
        x += shfl_down_sync(0xffffffff, x, off)
        off >>= 1
    end
    return shfl_sync(0xffffffff, x, 1)
end

@inline warpid() = ((blockIdx().x - 1) * (blockDim().x >> 5)) + ((threadIdx().x - 1) >> 5) + 1
@inline laneid() = (threadIdx().x - 1) & 31

# ---------------------------------------------------------------- kernels

function k_score!(keyv, alist, cnts, def, wdeg, w, sc, kmin32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds nal = cnts[3]
    kb = typemax(UInt32)
    if i <= nal
        @inbounds v = alist[i]
        @inbounds if sc == 0
            k = Float32(wdeg[v])
        elseif sc == 1
            k = Float32(def[v])
        else
            k = Float32(def[v]) / Float32(w[v] + wdeg[v])
        end
        @inbounds keyv[i] = k
        kb = reinterpret(UInt32, k)
    end
    # warp minimum, one atomic per warp
    off = 16
    while off > 0
        kb = min(kb, shfl_down_sync(0xffffffff, kb, off))
        off >>= 1
    end
    if laneid() == 0 && kb != typemax(UInt32)
        CUDA.atomic_min!(pointer(kmin32, 1), kb)
    end
    return nothing
end

# counters: cnts = [ncand, nsel, nalive, nalive after compaction]
function k_newround!(cnts, kmin32)
    @inbounds cnts[3] = cnts[4]
    @inbounds cnts[1] = 0; @inbounds cnts[2] = 0; @inbounds cnts[4] = 0
    @inbounds kmin32[1] = typemax(UInt32)
    return nothing
end

function k_gather!(info, cnts, koff, boff, pbound, ptop, nc)
    @inbounds ns = cnts[2]
    @inbounds info[1] = ns
    @inbounds info[2] = ns > 0 ? koff[ns] : 0
    @inbounds info[3] = ns > 0 ? boff[ns] : 0
    @inbounds info[4] = pbound[1] + ptop[1]
    @inbounds info[5] = ptop[1]
    return nothing
end

function k_cand!(cand, cpri, cnts, alist, keyv, kmin32, relax, delta, seed)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds nal = cnts[3]
    @inbounds kmin = reinterpret(Float32, kmin32[1])
    thr = isinf(relax) ? Inf32 : max(kmin, kmin * (1f0 + relax) + delta)
    if i <= nal
        @inbounds k = keyv[i]
        if k <= thr
            @inbounds v = alist[i]
            j = CUDA.atomic_add!(pointer(cnts, 1), Int32(1)) + Int32(1)
            @inbounds cand[j] = v
            @inbounds cpri[j] = (UInt64(reinterpret(UInt32, k)) << 32) | UInt64(hash32(v, seed))
        end
    end
    return nothing
end

# warp per candidate: best[x] = min priority over the candidates near x
function k_luby1!(best, cand, cpri, nc, pool, astart, alen)
    wi = warpid(); lane = laneid()
    if wi <= nc
        @inbounds v = cand[wi]; @inbounds p = cpri[wi]
        lane == 0 && CUDA.atomic_min!(pointer(best, v), p)
        @inbounds s = astart[v]; @inbounds l = alen[v]
        k = lane
        while k < l
            @inbounds x = pool[s + k]
            CUDA.atomic_min!(pointer(best, x), p)
            k += 32
        end
    end
    return nothing
end

function k_luby2!(sel, cnts, best, cand, cpri, nc, pool, astart, alen)
    wi = warpid(); lane = laneid()
    if wi <= nc
        @inbounds v = cand[wi]; @inbounds p = cpri[wi]
        @inbounds ok = best[v] == p
        @inbounds s = astart[v]; @inbounds l = alen[v]
        k = lane
        while k < l
            @inbounds x = pool[s + k]
            @inbounds ok &= best[x] == p
            k += 32
        end
        allok = vote_all_sync(0xffffffff, ok)
        if allok && lane == 0
            j = CUDA.atomic_add!(pointer(cnts, 2), Int32(1)) + Int32(1)
            @inbounds sel[j] = v
        end
    end
    return nothing
end

function k_reset!(best, cand, nc, pool, astart, alen)
    wi = warpid(); lane = laneid()
    if wi <= nc
        @inbounds v = cand[wi]
        lane == 0 && (@inbounds best[v] = typemax(UInt64))
        @inbounds s = astart[v]; @inbounds l = alen[v]
        k = lane
        while k < l
            @inbounds best[pool[s + k]] = typemax(UInt64)
            k += 32
        end
    end
    return nothing
end

# warp per selected vertex: arena sizes and the pool bound
function k_sizes!(ksz, bsz, pbound, sel, cnts, nc, pool, astart, alen)
    wi = warpid(); lane = laneid()
    @inbounds ns = cnts[2]
    if ns < wi <= nc && lane == 0
        @inbounds ksz[wi] = 0; @inbounds bsz[wi] = 0
    end
    if wi <= ns
        @inbounds v = sel[wi]; @inbounds s = astart[v]; @inbounds l = alen[v]
        acc = Int64(0)
        k = lane
        while k < l
            @inbounds x = pool[s + k]
            @inbounds acc += Int64(alen[x]) + Int64(l)
            k += 32
        end
        acc = warpsum(acc)
        if lane == 0
            @inbounds ksz[wi] = Int64(l)
            @inbounds bsz[wi] = Int64(l) * Int64(cld(l, 32))
            CUDA.atomic_add!(pointer(pbound, 1), acc)
        end
    end
    return nothing
end

# block per selected vertex
function k_elim!(pool, astart, alen, stt, w, wdeg, def, kown, kidx, sel, koff, kar, swar, dsel,
        boff, bar, ptop, needdef)
    bi = blockIdx().x; tid = threadIdx().x; nth = blockDim().x
    nw = nth >> 5; wl = (tid - 1) >> 5 + 1; lane = (tid - 1) & 31
    sh_d = CuStaticSharedArray(Int32, 1)
    sh_WK = CuStaticSharedArray(Int64, 1)
    @inbounds v = sel[bi]
    @inbounds sv = astart[v]; @inbounds L = alen[v]
    @inbounds ko = bi == 1 ? Int64(0) : koff[bi - 1]
    @inbounds wv = w[v]
    # phase 0: K = live neighbours of v (warp 1, order kept)
    if wl == 1
        cnt = Int32(0)
        base = Int32(0)
        while base < L
            idx = base + Int32(lane)
            x = idx < L ? (@inbounds pool[sv + idx]) : Int32(0)
            ok = idx < L && (@inbounds stt[x]) == ALIVE
            mask = vote_ballot_sync(0xffffffff, ok)
            if ok
                p = cnt + Int32(count_ones(mask & ((UInt32(1) << lane) - UInt32(1))))
                @inbounds kar[ko + p + 1] = x
            end
            cnt += Int32(count_ones(mask))
            base += Int32(32)
        end
        if lane == 0
            sh_d[1] = cnt; sh_WK[1] = 0
        end
    end
    sync_threads()
    d = sh_d[1]
    i = tid
    while i <= d
        @inbounds x = kar[ko + i]
        @inbounds kown[x] = v
        @inbounds kidx[x] = i
        CUDA.atomic_add!(pointer(sh_WK, 1), @inbounds w[x])
        i += nth
    end
    tid == 1 && (@inbounds dsel[bi] = d)
    W = cld(d, Int32(32))
    @inbounds bo = bi == 1 ? Int64(0) : boff[bi - 1]
    k = tid
    while k <= Int64(d) * W
        @inbounds bar[bo + k] = UInt32(0)
        k += nth
    end
    sync_threads()
    WK = sh_WK[1]
    # phase 1: adjacency inside K as a bit matrix, and K-neighbour weights
    i = wl
    while i <= d
        @inbounds a = kar[ko + i]
        @inbounds sa = astart[a]; @inbounds la = alen[a]
        acc = Int64(0)
        k = lane
        while k < la
            @inbounds b = pool[sa + k]
            if (@inbounds kown[b]) == v
                @inbounds j = kidx[b]
                CUDA.atomic_or!(pointer(bar, bo + Int64(i - 1) * W + (j - 1) >> 5 + 1), UInt32(1) << ((j - 1) & 31))
                @inbounds acc += w[b]
            end
            k += 32
        end
        acc = warpsum(acc)
        lane == 0 && (@inbounds swar[ko + i] = acc)
        i += nw
    end
    sync_threads()
    # phase 2: Wing-Huang updates over the missing pairs {a, b}, i < j
    if needdef
        i = wl
        while i <= d
            @inbounds a = kar[ko + i]
            @inbounds sa = astart[a]; @inbounds la = alen[a]
            @inbounds wa = w[a]
            for kw in 1:W
                @inbounds bits = ~bar[bo + Int64(i - 1) * W + kw]
                lo = (kw - 1) * 32          # bit t is j = lo + t + 1
                # keep j > i and j <= d
                if lo + 32 <= i
                    bits = UInt32(0)
                elseif lo < i
                    bits &= ~((UInt32(1) << (i - lo)) - UInt32(1))
                end
                if lo + 32 > d
                    bits &= (UInt32(1) << (d - lo)) - UInt32(1)
                end
                while bits != 0
                    t = trailing_zeros(bits)
                    bits &= bits - UInt32(1)
                    j = lo + t + 1
                    @inbounds b = kar[ko + j]
                    @inbounds wb = w[b]
                    wab = wa * wb
                    @inbounds sb = astart[b]; @inbounds lb = alen[b]
                    wout = Int64(0)
                    k = lane
                    while k < lb
                        @inbounds c = pool[sb + k]
                        if c != v && (@inbounds stt[c]) == ALIVE && bsearch(pool, sa, la, c)
                            CUDA.atomic_sub!(pointer(def, c), wab)
                            if (@inbounds kown[c]) != v
                                @inbounds wout += w[c]
                            end
                        end
                        k += 32
                    end
                    wout = warpsum(wout)
                    if lane == 0 && wout != 0
                        CUDA.atomic_sub!(pointer(def, a), wb * wout)
                        CUDA.atomic_sub!(pointer(def, b), wa * wout)
                    end
                end
            end
            i += nw
        end
    end
    sync_threads()
    # phase 3: removal of v and the new neighbours
    i = tid
    while i <= d
        @inbounds x = kar[ko + i]; @inbounds swx = swar[ko + i]
        @inbounds wx = w[x]
        Wn = WK - wx - swx
        @inbounds Wo = wdeg[x] - wv - swx
        needdef && CUDA.atomic_add!(pointer(def, x), Wo * (Wn - wv))
        @inbounds wdeg[x] += Wn - wv
        i += nth
    end
    sync_threads()
    # phase 4: new sorted adjacency list of each x: (N(x) \ K \ v, live) ∪ (K \ x)
    i = tid
    while i <= d
        @inbounds x = kar[ko + i]
        @inbounds sx = astart[x]; @inbounds lx = alen[x]
        cnt = Int32(0)
        for k in 0:(lx - 1)
            @inbounds e = pool[sx + k]
            if e != v && (@inbounds stt[e]) == ALIVE && (@inbounds kown[e]) != v
                cnt += Int32(1)
            end
        end
        newlen = cnt + d - Int32(1)
        pos = CUDA.atomic_add!(pointer(ptop, 1), Int64(newlen)) + 1
        p = 0; q = 1; o = pos
        # merge
        while true
            # next element of the old list
            e = typemax(Int32)
            while p < lx
                @inbounds e2 = pool[sx + p]
                if e2 != v && (@inbounds stt[e2]) == ALIVE && (@inbounds kown[e2]) != v
                    e = e2; break
                end
                p += 1
            end
            f = typemax(Int32)
            while q <= d
                @inbounds f2 = kar[ko + q]
                if f2 != x
                    f = f2; break
                end
                q += 1
            end
            (e == typemax(Int32) && f == typemax(Int32)) && break
            if e < f
                @inbounds pool[o] = e; p += 1
            else
                @inbounds pool[o] = f; q += 1
            end
            o += 1
        end
        @inbounds astart[x] = pos
        @inbounds alen[x] = newlen
        i += nth
    end
    return nothing
end

# block per selected vertex: K vertices with equal neighbourhoods outside K
function k_detect!(par, hsh, cnt, pool, astart, alen, stt, kown, sel, koff, kar, dsel)
    bi = blockIdx().x; tid = threadIdx().x; nth = blockDim().x
    @inbounds v = sel[bi]; @inbounds ko = bi == 1 ? Int64(0) : koff[bi - 1]; @inbounds d = dsel[bi]
    i = tid
    while i <= d
        @inbounds x = kar[ko + i]
        @inbounds sx = astart[x]; @inbounds lx = alen[x]
        h = UInt64(0); c = Int32(0)
        for k in 0:(lx - 1)
            @inbounds o = pool[sx + k]
            if (@inbounds kown[o]) != v && (@inbounds stt[o]) == ALIVE
                h += dmix(o); c += Int32(1)
            end
        end
        @inbounds hsh[ko + i] = h; @inbounds cnt[ko + i] = c
        i += nth
    end
    sync_threads()
    i = tid
    while i <= d
        @inbounds h = hsh[ko + i]; @inbounds c = cnt[ko + i]
        r = Int32(0)
        for j in 1:(i - 1)
            if (@inbounds hsh[ko + j]) == h && (@inbounds cnt[ko + j]) == c
                r = Int32(j); break
            end
        end
        rep = Int32(0)
        if r != 0
            # verify: every outside neighbour of x is a neighbour of K[r]
            @inbounds x = kar[ko + i]; @inbounds y = kar[ko + r]
            @inbounds sx = astart[x]; @inbounds lx = alen[x]
            @inbounds sy = astart[y]; @inbounds ly = alen[y]
            ok = true
            for k in 0:(lx - 1)
                @inbounds o = pool[sx + k]
                if (@inbounds kown[o]) != v && (@inbounds stt[o]) == ALIVE && !bsearch(pool, sy, ly, o)
                    ok = false; break
                end
            end
            ok && (rep = y)
        end
        @inbounds par[ko + i] = rep
        i += nth
    end
    return nothing
end

function k_apply!(parent, par, stt, w, wdeg, koff, kar, dsel)
    bi = blockIdx().x; tid = threadIdx().x; nth = blockDim().x
    @inbounds ko = bi == 1 ? Int64(0) : koff[bi - 1]; @inbounds d = dsel[bi]
    i = tid
    while i <= d
        @inbounds r = par[ko + i]
        if r != 0
            @inbounds x = kar[ko + i]
            @inbounds stt[x] = ABSORBED
            @inbounds parent[x] = r
            @inbounds wx = w[x]
            CUDA.atomic_add!(pointer(w, r), wx)
            CUDA.atomic_sub!(pointer(wdeg, r), wx)
        end
        i += nth
    end
    return nothing
end

function k_finish!(order, stt, sel, ns, nord)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= ns
        @inbounds v = sel[i]
        @inbounds stt[v] = DEAD
        @inbounds order[nord + i] = v
    end
    return nothing
end

function k_compact!(alist2, cnts, alist, stt)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds nal = cnts[3]
    if i <= nal
        @inbounds v = alist[i]
        if (@inbounds stt[v]) == ALIVE
            j = CUDA.atomic_add!(pointer(cnts, 4), Int32(1)) + Int32(1)
            @inbounds alist2[j] = v
        end
    end
    return nothing
end

# copy the live lists to a fresh pool
function k_repack!(newpool, newstart, pool, astart, alen, alist, nal, off)
    wi = warpid(); lane = laneid()
    if wi <= nal
        @inbounds v = alist[wi]; @inbounds s = astart[v]; @inbounds l = alen[v]
        @inbounds o = off[wi]
        k = lane
        while k < l
            @inbounds newpool[o + k + 1] = pool[s + k]
            k += 32
        end
        lane == 0 && (@inbounds newstart[v] = o + 1)
    end
    return nothing
end

# ---------------------------------------------------------------- driver

const TPB = 256

nblk(n, t = TPB) = max(1, cld(n, t))

function ensure!(a::CuVector{T}, n) where {T}
    if length(a) < n
        return CuVector{T}(undef, max(n, 2 * length(a)))
    end
    return a
end

"""
    grounds(weights, graph, alg::RoundsAlg; stats=nothing, timing=nothing)

GPU version of `Rounds.rounds` (selection is always the one-pass Luby).
"""
function grounds(weights::AbstractVector, graph::AbstractGraph, alg::RoundsAlg; kw...)
    T0 = time()
    n = nv(graph)
    ptr = Vector{Int64}(undef, n + 1); ptr[1] = 1
    for v in 1:n
        c = 0
        for u in neighbors(graph, v)
            u != v && (c += 1)
        end
        ptr[v + 1] = ptr[v] + c
    end
    tgt = Vector{Int32}(undef, ptr[n + 1] - 1)
    for v in 1:n
        k = ptr[v]
        for u in neighbors(graph, v)
            u != v && (tgt[k] = u; k += 1)
        end
        seg = view(tgt, ptr[v]:(ptr[v + 1] - 1))
        issorted(seg) || sort!(seg)
    end
    return grounds_csr(weights, ptr, tgt, alg; tcsr = time() - T0, kw...)
end

# initial state on the device -------------------------------------------

function k_hash!(hk, pool, astart, alen, n)
    v = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if v <= n
        @inbounds s = astart[v]; @inbounds l = alen[v]
        h = dmix(Int32(v))
        for k in 0:(l - 1)
            @inbounds h += dmix(pool[s + k])
        end
        @inbounds hk[v] = h
    end
    return nothing
end

function k_runhead!(hd, perm, hk, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds hd[i] = (i == 1 || hk[perm[i]] != hk[perm[i - 1]]) ? Int32(i) : Int32(0)
    end
    return nothing
end

# x joins the run head if their closed neighbourhoods are equal
function k_twins!(par, perm, hd, pool, astart, alen, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds x = perm[i]; @inbounds y = perm[hd[i]]
        r = Int32(0)
        if x != y && (@inbounds alen[x]) == (@inbounds alen[y])
            @inbounds sx = astart[x]; @inbounds lx = alen[x]
            @inbounds sy = astart[y]; @inbounds ly = alen[y]
            ok = bsearch(pool, sx, lx, y)
            k = 0
            while ok && k < lx
                @inbounds u = pool[sx + k]
                ok = u == y || bsearch(pool, sy, ly, u)
                k += 1
            end
            ok && (r = y)
        end
        @inbounds par[x] = r
    end
    return nothing
end

function k_absorb0!(parent, stt, w, par, n)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if x <= n
        @inbounds r = par[x]
        if r != 0
            @inbounds stt[x] = ABSORBED
            @inbounds parent[x] = r
            CUDA.atomic_add!(pointer(w, r), @inbounds w[x])
        end
    end
    return nothing
end

function k_wdeg!(wdeg, pool, astart, alen, stt, w, n)
    v = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if v <= n
        @inbounds s = astart[v]; @inbounds l = alen[v]
        acc = Int64(0)
        for k in 0:(l - 1)
            @inbounds u = pool[s + k]
            (@inbounds stt[u]) == ALIVE && (@inbounds acc += w[u])
        end
        @inbounds wdeg[v] = acc
    end
    return nothing
end

# warp per live vertex: def = ((sum w)^2 - sum w^2)/2 - weight of the edges inside N(v)
function k_initdef!(def, alist, nal, pool, astart, alen, stt, w)
    wi = warpid(); lane = laneid()
    if wi <= nal
        @inbounds v = alist[wi]; @inbounds sv = astart[v]; @inbounds lv = alen[v]
        W = Int64(0); W2 = Int64(0); tri = Int64(0)
        k = lane
        while k < lv
            @inbounds a = pool[sv + k]
            if (@inbounds stt[a]) == ALIVE
                @inbounds wa = w[a]
                W += wa; W2 += wa * wa
                @inbounds sa = astart[a]; @inbounds la = alen[a]
                for q in 0:(la - 1)
                    @inbounds b = pool[sa + q]
                    if (@inbounds stt[b]) == ALIVE && bsearch(pool, sv, lv, b)
                        @inbounds tri += wa * w[b]
                    end
                end
            end
            k += 32
        end
        W = warpsum(W); W2 = warpsum(W2); tri = warpsum(tri)
        lane == 0 && (@inbounds def[v] = ((W * W - W2) >> 1) - (tri >> 1))
    end
    return nothing
end

function k_alive!(alist, acount, stt, n)
    v = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if v <= n && (@inbounds stt[v]) == ALIVE
        j = CUDA.atomic_add!(pointer(acount, 1), Int32(1)) + Int32(1)
        @inbounds alist[j] = Int32(v)
    end
    return nothing
end

"""
    grounds_csr(weights, ptr, tgt, alg; ...)

GPU rounds on a graph given as sorted CSR adjacency (1-based, no self loops).
"""
function grounds_csr(weights::AbstractVector, ptr::AbstractVector, tgt::AbstractVector, alg::RoundsAlg;
        stats = nothing, timing = nothing, poolfactor = 3.0, check::Bool = false, tcsr = 0.0, sortsel::Bool = false)
    T0 = time()
    n = length(ptr) - 1
    sc = alg.score === :md ? 0 : alg.score === :mf ? 1 : 2
    needdef = sc != 0
    tot = Int64(ptr[n + 1] - 1)
    cap = max(Int64(1024), round(Int64, poolfactor * tot))
    pool = CuVector{Int32}(undef, cap)
    copyto!(pool, 1, convert(Vector{Int32}, tgt), 1, tot)
    ptop = CuVector{Int64}([tot])
    astart = CuVector{Int64}(convert(Vector{Int64}, ptr[1:n]))
    alen = CuVector{Int32}(Int32.(diff(ptr)))
    stt = CUDA.fill(ALIVE, n)
    w = CuVector{Int64}([max(1, round(Int, WSCALE * weights[v])) for v in 1:n])
    parent = CUDA.zeros(Int32, n)
    if alg.compress
        hk = CuVector{UInt64}(undef, n)
        @cuda threads = TPB blocks = nblk(n) k_hash!(hk, pool, astart, alen, Int32(n))
        perm = Int32.(sortperm(hk))
        hd = CuVector{Int32}(undef, n)
        @cuda threads = TPB blocks = nblk(n) k_runhead!(hd, perm, hk, Int32(n))
        hd = accumulate(max, hd)
        par0 = CuVector{Int32}(undef, n)
        @cuda threads = TPB blocks = nblk(n) k_twins!(par0, perm, hd, pool, astart, alen, Int32(n))
        @cuda threads = TPB blocks = nblk(n) k_absorb0!(parent, stt, w, par0, Int32(n))
        CUDA.unsafe_free!(hk); CUDA.unsafe_free!(hd); CUDA.unsafe_free!(par0)
    end
    wdeg = CuVector{Int64}(undef, n)
    @cuda threads = TPB blocks = nblk(n) k_wdeg!(wdeg, pool, astart, alen, stt, w, Int32(n))
    alist = CuVector{Int32}(undef, n)
    acount = CUDA.zeros(Int32, 1)
    @cuda threads = TPB blocks = nblk(n) k_alive!(alist, acount, stt, Int32(n))
    nal = Int(Array(acount)[1])
    def = CUDA.zeros(Int64, n)
    needdef && @cuda threads = TPB blocks = nblk(32nal) k_initdef!(def, alist, Int32(nal), pool, astart, alen, stt, w)
    CUDA.synchronize()
    T1 = time()

    kown = CUDA.zeros(Int32, n)
    kidx = CUDA.zeros(Int32, n)
    best = CUDA.fill(typemax(UInt64), n)
    keyv = CuVector{Float32}(undef, n)
    alist2 = CuVector{Int32}(undef, n)
    cand = CuVector{Int32}(undef, n)
    cpri = CuVector{UInt64}(undef, n)
    sel = CuVector{Int32}(undef, n)
    order = CuVector{Int32}(undef, n)
    cnts = CuVector{Int32}([0, 0, 0, nal])
    kmin32 = CuVector{UInt32}([typemax(UInt32)])
    info = CuVector{Int64}(undef, 5)
    pbound = CUDA.zeros(Int64, 1)
    ksz = CuVector{Int64}(undef, n); bsz = CuVector{Int64}(undef, n)
    koff = CuVector{Int64}(undef, n); boff = CuVector{Int64}(undef, n)
    dsel = CuVector{Int32}(undef, n)
    kar = CuVector{Int32}(undef, 1 << 16); swar = CuVector{Int64}(undef, 1 << 16)
    hsh = CuVector{UInt64}(undef, 1 << 16); cnt = CuVector{Int32}(undef, 1 << 16)
    par = CuVector{Int32}(undef, 1 << 16)
    bar = CuVector{UInt32}(undef, 1 << 16)
    seed = UInt32(alg.seed % UInt32)
    relax = Float32(alg.relax); delta = Float32(alg.delta * Rounds.scoreunit(alg.score))
    nord = 0
    nround = 0
    nrepack = 0
    small = 0
    tsel = 0.0; telim = 0.0
    while nal > 0
        ta = time()
        @cuda threads = 1 blocks = 1 k_newround!(cnts, kmin32)
        @cuda threads = TPB blocks = nblk(nal) k_score!(keyv, alist, cnts, def, wdeg, w, Int32(sc), kmin32)
        @cuda threads = TPB blocks = nblk(nal) k_cand!(cand, cpri, cnts, alist, keyv, kmin32, relax, delta, seed)
        c = Array(cnts)                       # sync 1
        nc = Int(c[1]); nal = Int(c[3])
        (nal == 0 || nal <= alg.tail) && break
        nround += 1
        @cuda threads = TPB blocks = nblk(32nc) k_luby1!(best, cand, cpri, Int32(nc), pool, astart, alen)
        @cuda threads = TPB blocks = nblk(32nc) k_luby2!(sel, cnts, best, cand, cpri, Int32(nc), pool, astart, alen)
        @cuda threads = TPB blocks = nblk(32nc) k_reset!(best, cand, Int32(nc), pool, astart, alen)
        fill!(pbound, 0)
        @cuda threads = TPB blocks = nblk(32nc) k_sizes!(ksz, bsz, pbound, sel, cnts, Int32(nc), pool, astart, alen)
        accumulate!(+, view(koff, 1:nc), view(ksz, 1:nc))
        accumulate!(+, view(boff, 1:nc), view(bsz, 1:nc))
        @cuda threads = 1 blocks = 1 k_gather!(info, cnts, koff, boff, pbound, ptop, Int32(nc))
        ns, ktot, btot, topb, top = Array(info)   # sync 2
        @assert ns > 0
        # hand the rest to the CPU once the rounds are small and the cliques
        # large (the dense end, where one block per elimination starves the GPU)
        small = (ns <= alg.tailsel && ktot >= 32 * ns) ? small + 1 : 0
        if small >= 2 && nal <= alg.tailmax
            break
        end
        stats === nothing || push!(stats, (nal, nc, ns))
        sortsel && sort!(view(sel, 1:ns))
        tb = time(); tsel += tb - ta
        if length(kar) < ktot
            m = max(ktot, 2 * length(kar))
            kar = CuVector{Int32}(undef, m); swar = CuVector{Int64}(undef, m)
            hsh = CuVector{UInt64}(undef, m); cnt = CuVector{Int32}(undef, m); par = CuVector{Int32}(undef, m)
        end
        if length(bar) < btot
            bar = CuVector{UInt32}(undef, max(btot, 2 * length(bar)))
        end
        if topb > cap
            # repack the live lists, growing the pool if they fill most of it
            pb = topb - top
            lv = CuVector{Int64}(undef, nal)
            lv .= Int64.(view(alen, view(alist, 1:nal)))
            used = Int64(sum(lv))
            newcap = max(cap, round(Int64, 2 * (used + pb)))
            off = similar(lv)
            accumulate!(+, off, lv); off .-= lv
            newpool = CuVector{Int32}(undef, newcap)
            @cuda threads = TPB blocks = nblk(32nal) k_repack!(newpool, astart, pool, astart, alen, alist, Int32(nal), off)
            CUDA.unsafe_free!(pool)
            pool = newpool; cap = newcap
            fill!(ptop, used)
            nrepack += 1
        end
        @cuda threads = TPB blocks = ns k_elim!(pool, astart, alen, stt, w, wdeg, def, kown, kidx, sel, koff,
            kar, swar, dsel, boff, bar, ptop, needdef)
        if alg.compress
            @cuda threads = TPB blocks = ns k_detect!(par, hsh, cnt, pool, astart, alen, stt, kown, sel, koff, kar, dsel)
            @cuda threads = TPB blocks = ns k_apply!(parent, par, stt, w, wdeg, koff, kar, dsel)
        end
        @cuda threads = TPB blocks = nblk(ns) k_finish!(order, stt, sel, Int32(ns), Int32(nord))
        nord += ns
        @cuda threads = TPB blocks = nblk(nal) k_compact!(alist2, cnts, alist, stt)
        alist, alist2 = alist2, alist
        telim += time() - tb
        if check && needdef
            nal2 = Int(Array(cnts)[4])
            check_state(pool, astart, alen, stt, w, def, alist, nal2)
        end
    end
    T2 = time()
    # back to the CPU: order so far, absorption parents, remaining graph
    orderH = Array(view(order, 1:nord))
    parentH = Array(parent)
    final = Int32[]
    append!(final, orderH)
    if nal > 0
        al = Array(view(alist, 1:nal))
        stH = Array(astart); lnH = Array(alen); wH = Array(w)
        poolH = Array(view(pool, 1:Int(Array(ptop)[1])))
        loc = zeros(Int, n)
        for (i, v) in enumerate(al)
            loc[v] = i
        end
        fadj = Vector{Vector{Int}}(undef, nal); ne2 = 0
        for (i, v) in enumerate(al)
            lst = Int[]
            for k in stH[v]:(stH[v] + lnH[v] - 1)
                j = loc[poolH[k]]
                j > 0 && push!(lst, j)
            end
            sort!(lst); fadj[i] = lst; ne2 += length(lst)
        end
        sub = SimpleGraph(ne2 ÷ 2, fadj)
        wsub = [wH[v] / WSCALE for v in al]
        so, _ = permutation(wsub, sub, serialalg(alg))
        for i in so
            push!(final, al[i])
        end
    end
    # expand supervariables: each vertex is followed by the vertices absorbed into it
    head = zeros(Int32, n); nxt = zeros(Int32, n)
    for x in 1:n
        r = parentH[x]
        if r != 0
            nxt[x] = head[r]; head[r] = x
        end
    end
    out = Vector{Int}(undef, n); no = 0
    stack = Int32[]
    for v in final
        push!(stack, v)
        while !isempty(stack)
            x = pop!(stack)
            out[no += 1] = x
            c = head[x]
            while c != 0
                push!(stack, c); c = nxt[c]
            end
        end
    end
    @assert no == n "emitted $no of $n"
    for a in (kown, kidx, best, keyv, alist2, cand, cpri, sel, ksz, bsz, koff, boff, dsel,
            kar, swar, hsh, cnt, par, bar, wdeg, def, stt, pool, astart, alen, w, parent, alist, order)
        CUDA.unsafe_free!(a)
    end
    T3 = time()
    if timing !== nothing
        timing[:pre] = T1 - T0; timing[:csr] = tcsr; timing[:gpu] = T2 - T1; timing[:cpu_post] = T3 - T2
        timing[:sel] = tsel; timing[:elim] = telim; timing[:rounds] = nround; timing[:repack] = nrepack
        timing[:tail] = nal
    end
    return out, invperm(out)
end

# debug: compare the device deficiencies with a from-scratch CPU computation
function check_state(pool, astart, alen, stt, w, def, alist, nal)
    P = Array(pool); S = Array(astart); L = Array(alen); St = Array(stt); W = Array(w); D = Array(def)
    al = Array(view(alist, 1:nal))
    nb(v) = [u for u in P[S[v]:(S[v] + L[v] - 1)] if St[u] == ALIVE]
    for v in al
        N = nb(v); set = Set(N)
        @assert issorted(P[S[v]:(S[v] + L[v] - 1)])
        Wt = 0; W2 = 0; tri = 0
        for a in N
            Wt += W[a]; W2 += W[a]^2
            for b in nb(a)
                b in set && (tri += W[a] * W[b])
            end
        end
        df = ((Wt * Wt - W2) >> 1) - (tri >> 1)
        @assert D[v] == df "device deficiency mismatch at $v: $(D[v]) vs $df"
    end
end

"""
    GPUHybrid(alg::RoundsAlg, small; nmin=20000)

GPU rounds for graphs with at least `nmin` vertices, the elimination
algorithm `small` (on the CPU) for the others; for use inside ND or BestWidth.
"""
struct GPUHybrid{A <: CliqueTrees.EliminationAlgorithm} <: CliqueTrees.EliminationAlgorithm
    alg::RoundsAlg
    small::A
    nmin::Int
end

GPUHybrid(alg::RoundsAlg, small; nmin::Integer = 20000) = GPUHybrid(alg, small, Int(nmin))

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::GPUHybrid; verbose::Bool = false) where {V}
    if nv(graph) < alg.nmin
        return permutation(weights, graph, alg.small)
    end
    order, index = grounds(weights, graph, alg.alg)
    return convert(Vector{V}, order), convert(Vector{V}, index)
end

end # module
