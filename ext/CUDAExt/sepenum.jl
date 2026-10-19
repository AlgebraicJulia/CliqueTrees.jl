module SepEnumGPU

# GPU sepenum for any word count: sets are UInt64 (n <= 64) or Bits{W}
# (n <= 64W). The same DFS as SepEnumGPU (one-word), written against the
# operations of Bits.jl, following CliqueTrees' multi-word methods
# (sepenum_from!, sepenum_branch!, sepenum_ste): identical results.

using CUDA
using CliqueTrees
using Graphs

using ..GPUBits
export prepare, gpu_sepenum, cpu_sepenum_sets, Bits, settype

settype(n) = n <= 64 ? UInt64 : Bits{cld(n, 64)}

struct Batch{T}
    adj::CuMatrix{UInt64}      # (Nmax * W, G): set of v at words (v-1)W+1 : vW
    wts::CuMatrix{Float64}     # (Nmax, G)
    nvs::CuVector{Int32}
    maxk::CuVector{Float64}
    smax::CuVector{Int32}
    order::CuMatrix{Int32}
    tg::CuVector{Int32}
    ti::CuVector{Int32}
    nthr::Int
    emp::Vector{Bool}
    G::Int
end

function prepare(::Type{T}, graphs, weights, maxks) where {T}
    G = length(graphs); W = nwords(T); NM = 64W
    adj = zeros(UInt64, NM * W, G); wts = zeros(Float64, NM, G); nvs = zeros(Int32, G)
    order = zeros(Int32, NM, G); smax = zeros(Int32, G); emp = falses(G)
    tg = Int32[]; ti = Int32[]
    for g in 1:G
        gr = graphs[g]; n = nv(gr); @assert n <= NM
        nvs[g] = n
        nw = cld(n, 64)
        a = nw == 1 ? zeros(UInt64, n) : zeros(UInt64, nw, n)
        o = zeros(Int, n); key = zeros(Int, n); cnt = zeros(Int, n)
        smax[g] = CliqueTrees.sepenum_adj!(a, o, key, cnt, weights[g], gr, maxks[g])
        for v in 1:n, j in 1:nw
            adj[(v - 1) * W + j, g] = nw == 1 ? a[v] : a[j, v]
        end
        wts[1:n, g] .= weights[g][1:n]; order[1:n, g] .= o
        if n > 0 && zero(Float64) < maxks[g]
            m = maskn(T, n)
            c = cmpset(AdjCol{T, typeof(adj)}(adj, Int32(g)), m, 1)
            emp[g] = c != m
        end
        for i in 1:n
            push!(tg, g); push!(ti, i)
        end
    end
    return Batch{T}(CuArray(adj), CuArray(wts), CuArray(nvs), CuArray(Float64.(maxks)), CuArray(smax), CuArray(order),
        CuArray(tg), CuArray(ti), length(tg), emp, G)
end

# ------------------------------------------------------------- device --

struct AdjCol{T, A}
    a::A
    g::Int32
end
@inline function Base.getindex(c::AdjCol{UInt64}, v)
    return @inbounds c.a[v, c.g]
end
@inline function Base.getindex(c::AdjCol{Bits{W}}, v) where {W}
    return Bits{W}(ntuple(j -> @inbounds(c.a[(v - 1) * W + j, c.g]), Val(W)))
end

struct Col{A}
    a::A
    g::Int32
end
@inline Base.getindex(c::Col, i) = @inbounds c.a[i, c.g]

struct Stk{A}
    a::A
    off::Int64
end
@inline Base.getindex(s::Stk, i) = @inbounds s.a[s.off + i]
@inline Base.setindex!(s::Stk, x, i) = (@inbounds s.a[s.off + i] = x)

@inline function cmpset(adj::AdjCol{T}, m::T, v) where {T}
    c = t = bitof(T, v)
    while true
        u = zero(T)
        while !iszero(t)
            u = u | adj[vmin(t)]; t = clearmin(t)
        end
        t = u & m & ~c
        iszero(t) && break
        c = c | t
    end
    return c
end

@inline function nbrset(adj::AdjCol{T}, b::T) where {T}
    u = zero(T); word = b
    while !iszero(word)
        u = u | adj[vmin(word)]; word = clearmin(word)
    end
    return u & ~b
end

@inline function wgt(w::Float64, word, wts)
    while !iszero(word)
        w += wts[vmin(word)]; word = clearmin(word)
    end
    return w
end

@inline sml(nA, nS, smax, n) = (nS <= smax && 2nA <= n - nS) || 2nA + 2nS - n <= min(smax, nS - Int32(1))

# frames (sets take W words): from = [hdr, ws, path, A, S, F, (B, N)...]; branch = [hdr, ws, wg, path, A, B, S, G, keys...]
@inline fromhdr(::Type{T}) where {T} = 3 + 3nwords(T)
@inline brhdr(::Type{T}) where {T} = 4 + 4nwords(T)
const PATHBITS = 42
@inline compshift(l) = PATHBITS - 14l + 7
@inline keyshift(l) = PATHBITS - 14l
@inline field(h, s) = Int32((h >> s) & 0xff)

@inline function ste(::Type{T}, st, sp, adj, wts, B, S, F, nA, maxk, n) where {T}
    W = nwords(T)
    nF = popc(F); nU = popc(S) - nF; nB = popc(B)
    wmin = typemax(Float64); word = (S & ~F) | B
    while !iszero(word)
        wmin = min(wmin, wts[vmin(word)]); word = clearmin(word)
    end
    jmax = Int32(0); w = wgt(0.0, F, wts)
    while jmax < nU && w + wmin < maxk
        jmax += Int32(1); w += wmin
    end
    m = nU - jmax
    (m <= 0 || 2(nA + m + nB) <= n - nF - jmax) && return false
    R = B; t = Int32(0); word = S & ~F
    while !iszero(word)
        stw!(st, sp + t * W, adj[vmin(word)] & R); word = clearmin(word); t += Int32(1)
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
            f = ldw(T, st, sp + t * W) & R
            if iszero(f)
                stop += Int32(1); nlive -= Int32(1); stw!(st, sp + t * W, ldw(T, st, sp + nlive * W))
            else
                x = vmin(f); nrst -= Int32(1); R = R & ~bitof(T, x)
                stw!(st, sp + t * W, (f | adj[x]) & R); t += Int32(1)
            end
        end
        sfrz += (r + Int32(1)) * min(stop, max(Int32(0), m - nfrz))
        nfrz += stop; r += Int32(1)
    end
    return res
end

@inline function push_from(::Type{T}, st, sp, adj, A, R, S, F, ws, maxk, n, path, level) where {T}
    W = nwords(T); H = fromhdr(T)
    f = vmin(F)
    Rf = f == 0 ? R : adj[f] & R
    nc = Int32(0)
    while !iszero(Rf)
        v = vmin(Rf)
        B = cmpset(adj, R, v)
        Rf = Rf & ~B
        stw!(st, sp + H + 2nc * W, B)
        stw!(st, sp + H + (2nc + 1) * W, nbrset(adj, B))
        nc += Int32(1)
    end
    nS = popc(S)
    lst = nS != 0 && 2popc(A) <= n - nS && ws < maxk
    st[sp + 1] = UInt64(1) | (UInt64(nc) << 8) | (UInt64(lst) << 24) | (UInt64(level) << 32)
    st[sp + 2] = reinterpret(UInt64, ws); st[sp + 3] = path
    stw!(st, sp + 3, A); stw!(st, sp + 3 + W, S); stw!(st, sp + 3 + 2W, F)
    return sp + H + 2nc * W
end

@inline function putrule(::Type{T}, st, base, nc, j, nn, X, n) where {T}
    W = nwords(T)
    Nj = ldw(T, st, base + (2j + 1) * W)
    for q in Int32(0):nc - Int32(1)
        if ldw(T, st, base + (2q + 1) * W) == Nj
            q < j && return false
            Bq = ldw(T, st, base + 2q * W)
            if 2popc(Bq) <= n - nn && !iszero(Bq & X)
                return false
            end
        end
    end
    return true
end

@inline function emit!(outkey, outsep, cnt, cap, k1::UInt64, k2::UInt64, S::T) where {T}
    slot = CUDA.atomic_add!(pointer(cnt), Int32(1)) + Int32(1)
    if slot <= cap
        @inbounds outkey[2slot - 1] = k1
        @inbounds outkey[2slot] = k2
        stset!(outsep, slot, S)
    end
    return
end

@inline function enter_branch(::Type{T}, st, sp, adj, wts, X, A, B, S, F, ws, put, maxk, smax, n, outkey, outsep, cnt, cap, k1, seq, path, level) where {T}
    W = nwords(T); H = brhdr(T)
    nA = popc(A); nS = popc(S)
    sml(nA, nS, smax, n) || return sp, seq, false
    nS == 0 && return sp, seq, false
    if ws >= maxk
        ste(T, st, sp, adj, wts, B, S, F, nA, maxk, n) && return sp, seq, false
    end
    if put
        emit!(outkey, outsep, cnt, cap, k1 | path, UInt64(seq), S)
        seq += Int32(1)
    end
    G = F; wg = wgt(0.0, F, wts)
    base = sp + H; nk = Int32(0); word = S & ~F
    while !iszero(word)
        u = vmin(word); word = clearmin(word)
        key = (UInt64(popc(adj[u] & B)) << 32) | (UInt64(typemax(UInt32)) - UInt64(u))
        s = nk
        while s > 0 && st[base + s] < key
            st[base + s + 1] = st[base + s]; s -= Int32(1)
        end
        st[base + s + 1] = key
        nk += Int32(1)
    end
    nk == 0 && return sp, seq, false
    st[sp + 1] = UInt64(2) | (UInt64(nk) << 8) | (UInt64(nA) << 32) | (UInt64(level) << 48)
    st[sp + 2] = reinterpret(UInt64, ws); st[sp + 3] = reinterpret(UInt64, wg); st[sp + 4] = path
    stw!(st, sp + 4, A); stw!(st, sp + 4 + W, B); stw!(st, sp + 4 + 2W, S); stw!(st, sp + 4 + 3W, G)
    return sp + H + nk, seq, true
end

@inline keyvtx(key::UInt64) = Int32(UInt64(typemax(UInt32)) - (key & UInt64(typemax(UInt32))))

@inline function rundfs!(::Type{T}, st, stkw, adj, wts, X, a, maxk, smax, n, mask, A, R, S, F, ws, path0, level0, k1,
        outkey, outsep, cnt, cap, split, tasks, tcnt, tcap, root, ovf, tid, maxsteps) where {T}
    W = nwords(T); HF = fromhdr(T); HB = brhdr(T)
    seq = Int32(0); sp = Int32(0); steps = Int64(0); top = Int32(0)
    nsp = push_from(T, st, sp, adj, A, R, S, F, ws, maxk, n, path0, level0)
    top += Int32(1); st[stkw - top + 1] = UInt64(sp); sp = nsp
    fixed = split == 0
    margin = 8 + (2 + 2 * 64 * W) * W   # the largest frame, and the ste scratch
    while top > 0
        steps += 1
        if steps > maxsteps || sp + margin > stkw - top
            @inbounds ovf[tid] = Int32(1); break
        end
        # cooperative cancellation: cnt[2] is the abort flag (set by the host at the deadline)
        if (steps & 1023) == 0 && CUDA.atomic_add!(pointer(cnt, 2), Int32(0)) != Int32(0)
            @inbounds ovf[tid] = Int32(3); break
        end
        fp = Int32(st[stkw - top + 1])
        h = st[fp + 1]
        if Int32(h & 0xff) == 1
            nc = field(h, 8); idx = field(h, 16); lst = field(h, 24) != 0; level = Int32((h >> 32) & 0xff)
            if idx >= nc
                sp = fp; top -= Int32(1); continue
            end
            st[fp + 1] = (h & ~(UInt64(0xff) << 16)) | (UInt64(idx + 1) << 16)
            fws = reinterpret(Float64, st[fp + 2]); fpath = st[fp + 3]
            fA = ldw(T, st, fp + 3); fS = ldw(T, st, fp + 3 + W); fF = ldw(T, st, fp + 3 + 2W)
            bpath = fixed || level > 3 ? fpath : fpath | (UInt64(idx + 1) << compshift(level))
            base = fp + HF
            B = ldw(T, st, base + 2idx * W); N = ldw(T, st, base + (2idx + 1) * W)
            pushed = false; nsp = sp
            if N == fS
                put = lst && putrule(T, st, base, nc, idx, popc(fS), X, n)
                nsp, seq, pushed = enter_branch(T, st, sp, adj, wts, X, fA, B, fS, fF, fws, put, maxk, smax, n, outkey, outsep, cnt, cap, k1, seq, bpath, level)
            elseif iszero(fF & ~N)
                Tm = ~(B | N) & mask
                C = cmpset(adj, Tm, a)
                if iszero(C & X)
                    nN = popc(N); wN = wgt(0.0, N, wts)
                    put = nN != 0 && 2popc(C) <= n - nN && wN < maxk && putrule(T, st, base, nc, idx, nN, X, n)
                    nsp, seq, pushed = enter_branch(T, st, sp, adj, wts, X, C, B, N, fF, wN, put, maxk, smax, n, outkey, outsep, cnt, cap, k1, seq, bpath, level)
                end
            end
            if pushed
                top += Int32(1); st[stkw - top + 1] = UInt64(sp); sp = nsp
            end
        else
            nk = field(h, 8); idx = field(h, 16); stage = field(h, 24); nA = Int32((h >> 32) & 0xffff); level = Int32((h >> 48) & 0xff)
            bws = reinterpret(Float64, st[fp + 2]); wg = reinterpret(Float64, st[fp + 3]); bpath = st[fp + 4]
            bA = ldw(T, st, fp + 4); bB = ldw(T, st, fp + 4 + W); bS = ldw(T, st, fp + 4 + 2W); G = ldw(T, st, fp + 4 + 3W)
            v = keyvtx(st[fp + HB + idx + 1])
            if stage == 1
                if !(wg + wts[v] < maxk) || idx + 1 >= nk
                    sp = fp; top -= Int32(1); continue
                end
                wg += wts[v]; G = G | bitof(T, v)
                st[fp + 3] = reinterpret(UInt64, wg); stw!(st, fp + 4 + 3W, G)
                st[fp + 1] = (h & ~(UInt64(0xffff) << 16)) | (UInt64(idx + 1) << 16)
                continue
            end
            st[fp + 1] = h | (UInt64(1) << 24)
            bv = bitof(T, v); nbv = adj[v]
            nb = nbv & ~bS & ~bA; fx = nb & X; S1 = (bS | nb) & ~bv
            wS = wgt(bws - wts[v], nb, wts); wF = wgt(wg, fx, wts)
            if wF < maxk && sml(nA + Int32(1), popc(S1), smax, n)
                cpath = fixed || level > 3 ? bpath : bpath | (UInt64(idx + 1) << keyshift(level))
                if !fixed && level == split
                    j = CUDA.atomic_add!(pointer(tcnt), Int32(1)) + Int32(1)
                    if j <= tcap
                        o = (j - Int32(1)) * (4W + 3)
                        @inbounds tasks[o + 1] = reinterpret(UInt64, wS); @inbounds tasks[o + 2] = cpath; @inbounds tasks[o + 3] = UInt64(root)
                        tv = Stk(tasks, Int64(o) + 3)
                        stw!(tv, 0, bA | bv); stw!(tv, W, bB & ~nbv); stw!(tv, 2W, S1); stw!(tv, 3W, G | fx)
                    else
                        @inbounds ovf[tid] = Int32(2)
                    end
                else
                    nsp = push_from(T, st, sp, adj, bA | bv, bB & ~nbv, S1, G | fx, wS, maxk, n, cpath, level + Int32(1))
                    top += Int32(1); st[stkw - top + 1] = UInt64(sp); sp = nsp
                end
            end
        end
    end
    return
end

@inline function threadsetup(::Type{T}, tg, ti, nvs, maxks, smaxs, orders, adj0, wts0, root) where {T}
    g = @inbounds tg[root]; i = @inbounds ti[root]
    adj = AdjCol{T, typeof(adj0)}(adj0, g); wts = Col(wts0, g)
    n = @inbounds nvs[g]; maxk = @inbounds maxks[g]; smax = @inbounds smaxs[g]
    X = zero(T)
    for j in Int32(1):i - Int32(1)
        X = X | bitof(T, @inbounds orders[j, g])
    end
    a = @inbounds orders[i, g]
    return adj, wts, n, maxk, smax, X, a, maskn(T, n)
end

function start_kernel!(::Type{T}, adj0, wts0, nvs, maxks, smaxs, orders, tg, ti, nthr::Int32, stk, stkw::Int32, outkey, outsep, cnt, cap::Int32, ovf, maxsteps::Int64,
        split::Int32, tasks, tcnt, tcap::Int32) where {T}
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    tid > nthr && return
    adj, wts, n, maxk, smax, X, a, mask = threadsetup(T, tg, ti, nvs, maxks, smaxs, orders, adj0, wts0, tid)
    st = Stk(stk, (tid - Int32(1)) * Int64(stkw))
    A = bitof(T, a); S = adj[a]; F = S & X; R = ~(S | A) & mask
    if wgt(0.0, F, wts) < maxk && sml(Int32(1), popc(S), smax, n)
        rundfs!(T, st, stkw, adj, wts, X, a, maxk, smax, n, mask, A, R, S, F, wgt(0.0, S, wts), zero(UInt64), Int32(1), UInt64(tid) << PATHBITS,
            outkey, outsep, cnt, cap, split, tasks, tcnt, tcap, tid, ovf, tid, maxsteps)
    end
    return
end

function task_kernel!(::Type{T}, adj0, wts0, nvs, maxks, smaxs, orders, tg, ti, ntask::Int32, next, tasks, stk, stkw::Int32, outkey, outsep, cnt, cap::Int32, ovf, maxsteps::Int64) where {T}
    W = nwords(T)
    me = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    st = Stk(stk, (me - Int32(1)) * Int64(stkw))
    while true
        j = CUDA.atomic_add!(pointer(next), Int32(1)) + Int32(1)
        j > ntask && break
        o = (j - Int32(1)) * (4W + 3)
        wS = reinterpret(Float64, @inbounds tasks[o + 1]); path = @inbounds tasks[o + 2]; root = Int32(@inbounds tasks[o + 3])
        tv = Stk(tasks, Int64(o) + 3)
        A = ldw(T, tv, 0); R = ldw(T, tv, W); S = ldw(T, tv, 2W); F = ldw(T, tv, 3W)
        adj, wts, n, maxk, smax, X, a, mask = threadsetup(T, tg, ti, nvs, maxks, smaxs, orders, adj0, wts0, root)
        rundfs!(T, st, stkw, adj, wts, X, a, maxk, smax, n, mask, A, R, S, F, wS, path, Int32(99), (UInt64(root) << PATHBITS) | path,
            outkey, outsep, cnt, cap, Int32(0), tasks, cnt, Int32(0), root, ovf, j, maxsteps)
    end
    return
end

function resident(kernel, threads)
    sms = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    return sms * CUDA.active_blocks(kernel.fun, threads) * threads
end

# Run f() (kernel launches) and wait; if the host clock passes the deadline
# dl, set flag[i] (device memory) from another stream so that the kernels
# stop. Returns (GPU time, aborted).
const ABORTSTREAM = Ref{Any}(nothing)
function timed_dl(f, dl::Float64, flag, i)
    # (the flag is set by a memset on a stream that does not wait for the
    # kernels: a copy from host memory, or a kernel, can wait for them)
    ABORTSTREAM[] === nothing && (ABORTSTREAM[] = CuStream(; flags = CUDA.STREAM_NON_BLOCKING))
    e0 = CuEvent(); e1 = CuEvent()
    CUDA.record(e0); f(); CUDA.record(e1)
    ab = false
    if isfinite(dl)
        while !CUDA.isdone(e1)
            if !ab && time() > dl
                CUDA.memset(reinterpret(CuPtr{UInt32}, pointer(flag, i)), UInt32(1), 1; stream = ABORTSTREAM[])
                ab = true
            end
            Libc.systemsleep(50e-6)
        end
    end
    CUDA.synchronize(e1)
    return CUDA.elapsed(e0, e1), ab
end

# device buffers kept between calls; grown geometrically, never zeroed
# (every slot that is read was written), except the counters
const WS = Dict{Symbol, Any}()
function buf(name::Symbol, ::Type{E}, n::Integer) where {E}
    x = get(WS, name, nothing)
    if x === nothing || length(x) < n
        x = CuArray{E}(undef, max(n, x === nothing ? 0 : 2length(x)))
        WS[name] = x
    end
    return x
end

"""
    gpu_sepenum(b::Batch{T}; split) -> (lists of sets per graph, overflow per graph, kernel time, info)

split = 0: one thread per start vertex; split = D (1..3): phase 1 runs
levels 1..D per start and writes the deeper subtrees as tasks, which
persistent threads take from a queue. Output and task capacity start small
and the call is repeated with more if they overflow.
"""
function gpu_sepenum(b::Batch{T}; split::Int = 3, stkw::Int = 1024 * nwords(T)^2, cap::Int = 1 << 18, tcap::Int = 1 << 16, maxsteps::Int = 10^10, threads::Int = 64, maxresident::Int = 1 << 15, dl::Float64 = Inf) where {T}
    W = nwords(T)
    while true
        outkey = buf(:outkey, UInt64, 2cap); outsep = buf(:outsep, UInt64, cap * W)
        cnt = buf(:cnt, Int32, 2); fill!(cnt, 0)
        ovf = buf(:ovf, Int32, b.nthr); fill!(view(ovf, 1:b.nthr), 0)
        tasks = buf(:tasks, UInt64, split > 0 ? tcap * (4W + 3) : 1); tcnt = buf(:tcnt, Int32, 1); fill!(tcnt, 0)
        stk = buf(:stk, UInt64, stkw * b.nthr)
        t1, ab = timed_dl(dl, cnt, 2) do
            @cuda threads = threads blocks = cld(b.nthr, threads) start_kernel!(T, b.adj, b.wts, b.nvs, b.maxk, b.smax, b.order, b.tg, b.ti, Int32(b.nthr),
                stk, Int32(stkw), outkey, outsep, cnt, Int32(cap), ovf, Int64(maxsteps), Int32(split), tasks, tcnt, Int32(tcap))
        end
        ab && return nothing, nothing, t1, (; t1, t2 = 0.0, ntask = 0, aborted = true)
        t2 = 0.0; ntask = 0
        if split > 0
            ntask = Int(Array(tcnt)[1])
            if ntask > tcap
                tcap = nextpow(2, ntask); continue
            end
            if ntask > 0
                next = buf(:next, Int32, 1); fill!(next, 0)
                ovf2 = buf(:ovf2, Int32, ntask); fill!(view(ovf2, 1:ntask), 0)
                args = (T, b.adj, b.wts, b.nvs, b.maxk, b.smax, b.order, b.tg, b.ti, Int32(ntask), next, tasks,
                    stk, Int32(stkw), outkey, outsep, cnt, Int32(cap), ovf2, Int64(maxsteps))
                kernel = @cuda launch = false task_kernel!(args...)
                # stacks get at most a quarter of the free device memory
                Pmem = max(threads, (CUDA.free_memory() ÷ 4) ÷ (8stkw) ÷ threads * threads)
                P = min(resident(kernel, threads), cld(ntask, threads) * threads, maxresident, Pmem)
                stk2 = buf(:stk, UInt64, stkw * P)
                args = Base.setindex(args, stk2, 13)
                t2, ab = timed_dl(() -> kernel(args...; threads, blocks = cld(P, threads)), dl, cnt, 2)
                ab && return nothing, nothing, t1 + t2, (; t1, t2, ntask, aborted = true)
                ho = Array(view(ovf2, 1:ntask))
                if any(!iszero, ho)
                    ht = Array(view(tasks, 1:ntask * (4W + 3))); hov = Array(view(ovf, 1:b.nthr))
                    for j in 1:ntask
                        ho[j] != 0 && (hov[Int(ht[(j - 1) * (4W + 3) + 3])] = 1)
                    end
                    copyto!(ovf, hov)
                end
            end
        end
        c = Int(Array(cnt)[1])
        if c > cap
            cap = nextpow(2, c); continue
        end
        hk = reshape(Array(view(outkey, 1:2c)), 2, c); hs = Array(view(outsep, 1:c * W))
        key = Vector{UInt128}(undef, c)
        @inbounds for j in 1:c
            key[j] = (UInt128(hk[1, j]) << 64) | UInt128(hk[2, j])
        end
        p = sortperm(key; alg = Base.Sort.DEFAULT_UNSTABLE)
        tg = Array(b.tg); ov = Array(view(ovf, 1:b.nthr))
        # counting pass, then fill each graph's list in key order
        cntg = zeros(Int, b.G)
        for g in 1:b.G
            b.emp[g] && (cntg[g] += 1)
        end
        @inbounds for j in 1:c
            cntg[tg[Int(hk[1, j] >> PATHBITS)]] += 1
        end
        lists = [Vector{T}(undef, cntg[g]) for g in 1:b.G]
        fillg = zeros(Int, b.G)
        for g in 1:b.G
            b.emp[g] && (fillg[g] += 1; lists[g][1] = zero(T))
        end
        @inbounds for j in p
            g = tg[Int(hk[1, j] >> PATHBITS)]
            fillg[g] += 1; lists[g][fillg[g]] = ldset(T, hs, j)
        end
        ovg = falses(b.G)
        for tid in 1:b.nthr
            ov[tid] != 0 && (ovg[tg[tid]] = true)
        end
        return lists, ovg, t1 + t2, (; t1, t2, ntask, aborted = false)
    end
end

# CPU reference: CliqueTrees.sepenum, one thread, as sets
function cpu_sepenum_sets(::Type{T}, graph, weights, maxk) where {T}
    sep, status = CliqueTrees.sepenum(weights, graph, maxk, Inf, 1)
    return [toset(T, neighbors(sep, s)) for s in 1:nv(sep)]
end

# the separators as CliqueTrees expects them (for sepdp)
function sepgraph(sets::Vector{T}, nvtx) where {T}
    m = length(sets); W = nwords(T)
    ptr = Vector{Int}(undef, m + 1); ptr[1] = 1
    @inbounds for i in 1:m
        ptr[i + 1] = ptr[i] + popc(sets[i])
    end
    tgt = Vector{Int}(undef, ptr[m + 1] - 1)
    @inbounds for i in 1:m
        q = ptr[i]; x = sets[i]
        for j in 1:W
            word = T === UInt64 ? x : x.w[j]
            while word != 0
                tgt[q] = 64(j - 1) + trailing_zeros(word) + 1; q += 1; word &= word - one(UInt64)
            end
        end
    end
    return BipartiteGraph{Int, Int}(nvtx, m, length(tgt), ptr, tgt)
end

end
