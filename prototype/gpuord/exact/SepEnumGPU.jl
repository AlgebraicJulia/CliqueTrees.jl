module SepEnumGPU

# Batched GPU port of CliqueTrees' sepenum (Tamaki's MinSepsGenerator: the
# minimal separators S with w(S) < maxk), one-word graphs (n <= 64).
#
# One thread runs the search from one start vertex of one graph: the same
# DFS as sepenum_one!/sepenum_from!/sepenum_branch!/sepenum_ste (one-word
# methods), made iterative with an explicit stack of frames in global
# memory. Every separator is listed with a key (thread, sequence number);
# sorting by key gives the list of sepenum_st exactly (the ∅ separator of a
# disconnected graph, which comes first, is added on the host). Weights are
# Float64 and are summed in the CPU's order, so every comparison is the same.

using CUDA
using CliqueTrees
using Graphs

export SepBatch, prepare, gpu_sepenum, cpu_sepenum_lists

struct SepBatch
    adj::CuMatrix{UInt64}      # (64, G)
    wts::CuMatrix{Float64}     # (64, G)
    nvs::CuVector{Int32}
    maxk::CuVector{Float64}
    smax::CuVector{Int32}
    order::CuMatrix{Int32}     # (64, G) start vertices
    tg::CuVector{Int32}        # thread -> graph
    ti::CuVector{Int32}        # thread -> start index
    nthr::Int
    emp::Vector{Bool}          # ∅ is listed first (disconnected graph)
    G::Int
end

# host side: the same adj, start order and smax as sepenum_adj!
function prepare(graphs, weights, maxks)
    G = length(graphs)
    adj = zeros(UInt64, 64, G); wts = zeros(Float64, 64, G); nvs = zeros(Int32, G)
    order = zeros(Int32, 64, G); smax = zeros(Int32, G); emp = falses(G)
    tg = Int32[]; ti = Int32[]
    for g in 1:G
        gr = graphs[g]; n = nv(gr); @assert n <= 64
        nvs[g] = n
        a = zeros(UInt64, n); o = zeros(Int, n); key = zeros(Int, n); cnt = zeros(Int, n)
        smax[g] = CliqueTrees.sepenum_adj!(a, o, key, cnt, weights[g], gr, maxks[g])
        adj[1:n, g] .= a; wts[1:n, g] .= weights[g][1:n]; order[1:n, g] .= o
        # ∅ if disconnected (sepenum_emp!)
        if n > 0 && zero(Float64) < maxks[g]
            m = n == 64 ? typemax(UInt64) : (one(UInt64) << n) - one(UInt64)
            c = cmp1(a, m, 1)
            emp[g] = c != m
        end
        for i in 1:n
            push!(tg, g); push!(ti, i)
        end
    end
    return SepBatch(CuArray(adj), CuArray(wts), CuArray(nvs), CuArray(Float64.(maxks)), CuArray(smax), CuArray(order),
        CuArray(tg), CuArray(ti), length(tg), emp, G)
end

# ------------------------------------------------------------- device --

@inline bit(v) = one(UInt64) << ((v - 1) & 63)
@inline vmin(w::UInt64) = Int32(trailing_zeros(w) + 1)
@inline popc(w::UInt64) = Int32(count_ones(w))

# the component of v in the set m
@inline function cmp1(adj, m::UInt64, v)
    c = t = bit(v)
    while true
        u = zero(UInt64)
        while t != zero(UInt64)
            u |= adj[trailing_zeros(t) + 1]; t &= t - one(UInt64)
        end
        t = u & m & ~c
        t == zero(UInt64) && break
        c |= t
    end
    return c
end

@inline function nbr1(adj, b::UInt64)
    u = zero(UInt64); word = b
    while word != zero(UInt64)
        u |= adj[trailing_zeros(word) + 1]; word &= word - one(UInt64)
    end
    return u & ~b
end

@inline function wgt(w::Float64, word::UInt64, wts)
    while word != zero(UInt64)
        w += wts[trailing_zeros(word) + 1]; word &= word - one(UInt64)
    end
    return w
end

@inline sml(nA, nS, smax, n) = (nS <= smax && 2nA <= n - nS) || 2nA + 2nS - n <= min(smax, nS - Int32(1))

# a view of the columns of a matrix as 1-based vectors, for the graph g
struct Col{T, A}
    a::A
    g::Int32
end
@inline Base.getindex(c::Col, i) = @inbounds c.a[i, c.g]

# the stack region of one thread
struct Stk{A}
    a::A
    off::Int64
end
@inline Base.getindex(s::Stk, i) = @inbounds s.a[s.off + i]
@inline Base.setindex!(s::Stk, x, i) = (@inbounds s.a[s.off + i] = x)

# stack frames, in words of stk (thread-local region):
#   from:   [hdr, A, S, F, ws, path, B1, N1, B2, N2, ...]  hdr = 1 | ncomp<<8 | idx<<16 | lst<<24 | level<<32
#   branch: [hdr, A, B, S, ws, G, wg, path, keys...]      hdr = 2 | nk<<8 | idx<<16 | stage<<24 | nA<<32 | level<<40
# A listing is keyed (root thread << 42 | path, seq): path holds, for the
# levels 1..3 (most significant first), comp index + 1 and key index + 1 (7
# bits each), so sorting the keys gives the DFS preorder of one thread.
const HDRF = 6
const HDRB = 8
const PATHBITS = 42

@inline hdr_tag(h) = Int32(h & 0xff)
@inline field(h, s) = Int32((h >> s) & 0xff)
@inline compshift(l) = PATHBITS - 14l + 7
@inline keyshift(l) = PATHBITS - 14l

# the hanging-tree test (sepenum_ste, one word); scratch at stk[sp+1 : sp+64]
@inline function ste(stk, sp, adj, wts, B, S, F, nA, maxk, n)
    nF = popc(F); nU = popc(S) - nF; nB = popc(B)
    wmin = typemax(Float64); word = (S & ~F) | B
    while word != zero(UInt64)
        wmin = min(wmin, wts[trailing_zeros(word) + 1]); word &= word - one(UInt64)
    end
    jmax = Int32(0); w = wgt(0.0, F, wts)
    while jmax < nU && w + wmin < maxk
        jmax += Int32(1); w += wmin
    end
    m = nU - jmax
    (m <= 0 || 2(nA + m + nB) <= n - nF - jmax) && return false
    R = B; t = Int32(0); word = S & ~F
    while word != zero(UInt64)
        stk[sp + t + 1] = adj[trailing_zeros(word) + 1] & R; word &= word - one(UInt64); t += Int32(1)
    end
    r = Int32(0); nlive = nU; nfrz = Int32(0); sfrz = Int32(0); nrst = nB; res = false
    while true
        sm = sfrz + max(Int32(0), m - nfrz) * (r + Int32(1))
        if 2(nA + sm) > n - nF - jmax
            res = true; break
        end
        (nlive == 0 || 2(nA + sm + nrst) <= n - nF - jmax) && break
        t = Int32(0); stop = Int32(0)
        while t < nlive
            f = stk[sp + t + 1] & R
            if f == zero(UInt64)
                stop += Int32(1); nlive -= Int32(1); stk[sp + t + 1] = stk[sp + nlive + 1]
            else
                x = trailing_zeros(f) + 1; nrst -= Int32(1); R &= ~bit(x)
                stk[sp + t + 1] = (f | adj[x]) & R; t += Int32(1)
            end
        end
        sfrz += (r + Int32(1)) * min(stop, max(Int32(0), m - nfrz))
        nfrz += stop; r += Int32(1)
    end
    return res
end

# sepenum_from!: push a from-frame (the components of R next to f)
@inline function push_from(stk, sp, adj, A, R, S, F, ws, maxk, n, path, level)
    f = F == zero(UInt64) ? Int32(0) : vmin(F)
    Rf = f == 0 ? R : adj[f] & R
    nc = Int32(0); base = sp + HDRF
    while Rf != zero(UInt64)
        v = vmin(Rf)
        B = cmp1(adj, R, v)
        Rf &= ~B
        stk[base + 2nc + 1] = B
        stk[base + 2nc + 2] = nbr1(adj, B)
        nc += Int32(1)
    end
    nS = popc(S)
    lst = nS != 0 && 2popc(A) <= n - nS && ws < maxk
    stk[sp + 1] = UInt64(1) | (UInt64(nc) << 8) | (UInt64(lst) << 24) | (UInt64(level) << 32)
    stk[sp + 2] = A; stk[sp + 3] = S; stk[sp + 4] = F
    stk[sp + 5] = reinterpret(UInt64, ws); stk[sp + 6] = path
    return sp + HDRF + 2nc
end

@inline function putrule(stk, base, nc, j, nn, X, n)
    Nj = stk[base + 2j + 2]
    for q in Int32(0):nc - Int32(1)
        if stk[base + 2q + 2] == Nj
            q < j && return false
            Bq = stk[base + 2q + 1]
            if 2popc(Bq) <= n - nn && (Bq & X) != zero(UInt64)
                return false
            end
        end
    end
    return true
end

@inline function emit!(outkey, outsep, cnt, cap, k1::UInt64, k2::UInt64, S::UInt64)
    slot = CUDA.atomic_add!(pointer(cnt), Int32(1)) + Int32(1)
    if slot <= cap
        @inbounds outkey[2slot - 1] = k1
        @inbounds outkey[2slot] = k2
        @inbounds outsep[slot] = S
    end
    return
end

# sepenum_branch! entry: checks, listing, then push a branch-frame (or not)
@inline function enter_branch(stk, sp, adj, wts, X, A, B, S, F, ws, put, maxk, smax, n, outkey, outsep, cnt, cap, k1, seq, path, level)
    nA = popc(A); nS = popc(S)
    sml(nA, nS, smax, n) || return sp, seq, false
    nS == 0 && return sp, seq, false
    if ws >= maxk
        ste(stk, sp, adj, wts, B, S, F, nA, maxk, n) && return sp, seq, false
    end
    if put
        emit!(outkey, outsep, cnt, cap, k1 | path, UInt64(seq), S)
        seq += Int32(1)
    end
    G = F; wg = wgt(0.0, F, wts)
    base = sp + HDRB; nk = Int32(0); word = S & ~F
    while word != zero(UInt64)
        u = vmin(word); word &= word - one(UInt64)
        key = (UInt64(popc(adj[u] & B)) << 32) | (UInt64(typemax(UInt32)) - UInt64(u))
        s = nk
        while s > 0 && stk[base + s] < key
            stk[base + s + 1] = stk[base + s]; s -= Int32(1)
        end
        stk[base + s + 1] = key
        nk += Int32(1)
    end
    nk == 0 && return sp, seq, false
    stk[sp + 1] = UInt64(2) | (UInt64(nk) << 8) | (UInt64(nA) << 32) | (UInt64(level) << 40)
    stk[sp + 2] = A; stk[sp + 3] = B; stk[sp + 4] = S
    stk[sp + 5] = reinterpret(UInt64, ws); stk[sp + 6] = G
    stk[sp + 7] = reinterpret(UInt64, wg); stk[sp + 8] = path
    return sp + HDRB + nk, seq, true
end

@inline keyvtx(key::UInt64) = Int32(UInt64(typemax(UInt32)) - (key & UInt64(typemax(UInt32))))

# The DFS from one from() call. split > 0: from-frames at level split + 1
# are not searched but written as tasks. pathfix: paths are not extended
# (inside a task, the seq orders the listings).
@inline function rundfs!(st, stkw, adj, wts, X, a, maxk, smax, n, mask, A, R, S, F, ws, path0, level0, k1,
        outkey, outsep, cnt, cap, split, tasks, tcnt, tcap, root, ovf, tid, maxsteps)
    seq = Int32(0); sp = Int32(0); steps = Int64(0); top = Int32(0)
    nsp = push_from(st, sp, adj, A, R, S, F, ws, maxk, n, path0, level0)
    top += Int32(1); st[stkw - top + 1] = UInt64(sp); sp = nsp
    fixed = split == 0
    while top > 0
        steps += 1
        if steps > maxsteps || sp + 300 > stkw - top
            @inbounds ovf[tid] = Int32(1); break
        end
        fp = Int32(st[stkw - top + 1])
        h = st[fp + 1]
        if hdr_tag(h) == 1
            nc = field(h, 8); idx = field(h, 16); lst = field(h, 24) != 0; level = Int32((h >> 32) & 0xff)
            if idx >= nc
                sp = fp; top -= Int32(1); continue
            end
            st[fp + 1] = (h & ~(UInt64(0xff) << 16)) | (UInt64(idx + 1) << 16)
            fA = st[fp + 2]; fS = st[fp + 3]; fF = st[fp + 4]
            fws = reinterpret(Float64, st[fp + 5]); fpath = st[fp + 6]
            bpath = fixed || level > 3 ? fpath : fpath | (UInt64(idx + 1) << compshift(level))
            base = fp + HDRF
            B = st[base + 2idx + 1]; N = st[base + 2idx + 2]
            pushed = false; nsp = sp
            if N == fS
                put = lst && putrule(st, base, nc, idx, popc(fS), X, n)
                nsp, seq, pushed = enter_branch(st, sp, adj, wts, X, fA, B, fS, fF, fws, put, maxk, smax, n, outkey, outsep, cnt, cap, k1, seq, bpath, level)
            elseif (fF & ~N) == zero(UInt64)
                T = ~(B | N) & mask
                C = cmp1(adj, T, a)
                if (C & X) == zero(UInt64)
                    nN = popc(N); wN = wgt(0.0, N, wts)
                    put = nN != 0 && 2popc(C) <= n - nN && wN < maxk && putrule(st, base, nc, idx, nN, X, n)
                    nsp, seq, pushed = enter_branch(st, sp, adj, wts, X, C, B, N, fF, wN, put, maxk, smax, n, outkey, outsep, cnt, cap, k1, seq, bpath, level)
                end
            end
            if pushed
                top += Int32(1); st[stkw - top + 1] = UInt64(sp); sp = nsp
            end
        else
            nk = field(h, 8); idx = field(h, 16); stage = field(h, 24); nA = Int32((h >> 32) & 0xff); level = Int32((h >> 40) & 0xff)
            bA = st[fp + 2]; bB = st[fp + 3]; bS = st[fp + 4]
            bws = reinterpret(Float64, st[fp + 5]); G = st[fp + 6]
            wg = reinterpret(Float64, st[fp + 7]); bpath = st[fp + 8]
            v = keyvtx(st[fp + HDRB + idx + 1])
            if stage == 1
                if !(wg + wts[v] < maxk) || idx + 1 >= nk
                    sp = fp; top -= Int32(1); continue
                end
                wg += wts[v]; G |= bit(v)
                st[fp + 6] = G; st[fp + 7] = reinterpret(UInt64, wg)
                st[fp + 1] = (h & ~(UInt64(0xffff) << 16)) | (UInt64(idx + 1) << 16)
                continue
            end
            st[fp + 1] = h | (UInt64(1) << 24)
            bv = bit(v); nbv = adj[v]
            nb = nbv & ~bS & ~bA; fx = nb & X; S1 = (bS | nb) & ~bv
            wS = wgt(bws - wts[v], nb, wts); wF = wgt(wg, fx, wts)
            if wF < maxk && sml(nA + Int32(1), popc(S1), smax, n)
                cpath = fixed || level > 3 ? bpath : bpath | (UInt64(idx + 1) << keyshift(level))
                if !fixed && level == split
                    j = CUDA.atomic_add!(pointer(tcnt), Int32(1)) + Int32(1)
                    if j <= tcap
                        @inbounds tasks[1, j] = bA | bv; @inbounds tasks[2, j] = bB & ~nbv; @inbounds tasks[3, j] = S1
                        @inbounds tasks[4, j] = G | fx; @inbounds tasks[5, j] = reinterpret(UInt64, wS)
                        @inbounds tasks[6, j] = cpath; @inbounds tasks[7, j] = UInt64(root)
                    else
                        @inbounds ovf[tid] = Int32(2)
                    end
                else
                    nsp = push_from(st, sp, adj, bA | bv, bB & ~nbv, S1, G | fx, wS, maxk, n, cpath, level + Int32(1))
                    top += Int32(1); st[stkw - top + 1] = UInt64(sp); sp = nsp
                end
            end
        end
    end
    return
end

@inline function threadsetup(tg, ti, nvs, maxks, smaxs, orders, adj0, wts0, root)
    g = @inbounds tg[root]; i = @inbounds ti[root]
    adj = Col{UInt64, typeof(adj0)}(adj0, g); wts = Col{Float64, typeof(wts0)}(wts0, g)
    n = @inbounds nvs[g]; maxk = @inbounds maxks[g]; smax = @inbounds smaxs[g]
    X = zero(UInt64)
    for j in Int32(1):i - Int32(1)
        X |= bit(@inbounds orders[j, g])
    end
    a = @inbounds orders[i, g]
    mask = n == 64 ? typemax(UInt64) : (one(UInt64) << n) - one(UInt64)
    return adj, wts, n, maxk, smax, X, a, mask
end

# mode A: one thread per start (split = 0: the whole search; split = D:
# levels 1..D, the rest written as tasks)
function sepenum_kernel!(adj0, wts0, nvs, maxks, smaxs, orders, tg, ti, nthr::Int32, stk, stkw::Int32, outkey, outsep, cnt, cap::Int32, ovf, maxsteps::Int64,
        split::Int32, tasks, tcnt, tcap::Int32)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    tid > nthr && return
    adj, wts, n, maxk, smax, X, a, mask = threadsetup(tg, ti, nvs, maxks, smaxs, orders, adj0, wts0, tid)
    st = Stk(stk, (tid - Int32(1)) * Int64(stkw))
    A = bit(a); S = adj[a]; F = S & X; R = ~(S | A) & mask
    if wgt(0.0, F, wts) < maxk && sml(Int32(1), popc(S), smax, n)
        rundfs!(st, stkw, adj, wts, X, a, maxk, smax, n, mask, A, R, S, F, wgt(0.0, S, wts), zero(UInt64), Int32(1), UInt64(tid) << PATHBITS,
            outkey, outsep, cnt, cap, split, tasks, tcnt, tcap, tid, ovf, tid, maxsteps)
    end
    return
end

# mode B: persistent threads; each takes the next task from a queue
function task_kernel!(adj0, wts0, nvs, maxks, smaxs, orders, tg, ti, ntask::Int32, next, tasks, stk, stkw::Int32, outkey, outsep, cnt, cap::Int32, ovf, maxsteps::Int64)
    me = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    st = Stk(stk, (me - Int32(1)) * Int64(stkw))
    while true
        j = CUDA.atomic_add!(pointer(next), Int32(1)) + Int32(1)
        j > ntask && break
        root = Int32(@inbounds tasks[7, j])
        adj, wts, n, maxk, smax, X, a, mask = threadsetup(tg, ti, nvs, maxks, smaxs, orders, adj0, wts0, root)
        path = @inbounds tasks[6, j]
        rundfs!(st, stkw, adj, wts, X, a, maxk, smax, n, mask, @inbounds(tasks[1, j]), @inbounds(tasks[2, j]), @inbounds(tasks[3, j]), @inbounds(tasks[4, j]),
            reinterpret(Float64, @inbounds tasks[5, j]), path, Int32(99), (UInt64(root) << PATHBITS) | path,
            outkey, outsep, cnt, cap, Int32(0), tasks, cnt, Int32(0), root, ovf, j, maxsteps)
    end
    return
end

# resident threads of a kernel on this device (occupancy API)
function resident(kernel, threads)
    dev = CUDA.device()
    sms = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    nb = CUDA.active_blocks(kernel.fun, threads)
    return sms * nb * threads
end

"""
    gpu_sepenum(batch; split) -> lists of separators (one Vector{UInt64} per graph), overflow flags, kernel time

split = 0: one thread per start; split = D (1..3): phase 1 runs levels
1..D per start and writes the deeper subtrees as tasks, phase 2 runs one
thread per task.
"""
function gpu_sepenum(b::SepBatch; split::Int = 0, stkw::Int = 1024, cap::Int = 1 << 24, tcap::Int = 1 << 23, maxsteps::Int = 10^9, threads::Int = 64, chunkmax::Int = 1 << 18)
    outkey = CUDA.zeros(UInt64, 2cap); outsep = CUDA.zeros(UInt64, cap)
    cnt = CUDA.zeros(Int32, 1); ovf = CUDA.zeros(Int32, max(b.nthr, split > 0 ? tcap : 0))
    tasks = CUDA.zeros(UInt64, 7, split > 0 ? tcap : 1); tcnt = CUDA.zeros(Int32, 1)
    stk = CUDA.zeros(UInt64, stkw * b.nthr)
    t1 = CUDA.@elapsed @cuda threads = threads blocks = cld(b.nthr, threads) sepenum_kernel!(b.adj, b.wts, b.nvs, b.maxk, b.smax, b.order, b.tg, b.ti, Int32(b.nthr),
        stk, Int32(stkw), outkey, outsep, cnt, Int32(cap), ovf, Int64(maxsteps), Int32(split), tasks, tcnt, Int32(tcap))
    t2 = 0.0; ntask = 0
    if split > 0
        ntask = Int(Array(tcnt)[1]); @assert ntask <= tcap "task capacity exceeded"
        if ntask > 0
            stk = nothing; GC.gc(false)
            next = CUDA.zeros(Int32, 1)
            ovf2 = CUDA.zeros(Int32, ntask)
            args = (b.adj, b.wts, b.nvs, b.maxk, b.smax, b.order, b.tg, b.ti, Int32(ntask), next, tasks,
                CUDA.zeros(UInt64, 1), Int32(stkw), outkey, outsep, cnt, Int32(cap), ovf2, Int64(maxsteps))
            kernel = @cuda launch = false task_kernel!(args...)
            P = min(resident(kernel, threads), cld(ntask, threads) * threads)
            stk2 = CUDA.zeros(UInt64, stkw * P)
            args = Base.setindex(args, stk2, 12)
            t2 = CUDA.@elapsed kernel(args...; threads, blocks = cld(P, threads))
            # task overflow -> flag its root
            ho = Array(ovf2); ht = Array(tasks)
            hov = Array(ovf)
            for j in 1:ntask
                ho[j] != 0 && (hov[Int(ht[7, j])] = 1)
            end
            copyto!(ovf, hov)
        end
    end
    c = Int(Array(cnt)[1]); @assert c <= cap "output capacity exceeded"
    hk = reshape(Array(outkey)[1:2c], 2, c); hs = Array(outsep)[1:c]
    p = sortperm(collect(zip(view(hk, 1, :), view(hk, 2, :))))
    tg = Array(b.tg); ov = Array(ovf)
    lists = [UInt64[] for _ in 1:b.G]
    for g in 1:b.G
        b.emp[g] && push!(lists[g], zero(UInt64))
    end
    for j in p
        tid = Int(hk[1, j] >> PATHBITS)
        push!(lists[tg[tid]], hs[j])
    end
    ovg = falses(b.G)
    for tid in 1:b.nthr
        ov[tid] != 0 && (ovg[tg[tid]] = true)
    end
    return lists, ovg, t1 + t2, (; t1, t2, ntask)
end

# CPU reference (CliqueTrees.sepenum, one thread), as words
function cpu_sepenum_lists(graph, weights, maxk)
    sep, status = CliqueTrees.sepenum(weights, graph, maxk, Inf, 1)
    out = UInt64[]
    for s in 1:nv(sep)
        w = zero(UInt64)
        for x in neighbors(sep, s)
            w |= one(UInt64) << (x - 1)
        end
        push!(out, w)
    end
    return out
end

end
