module SepDPGPU

# Batched GPU port of the block evaluation of CliqueTrees' sepdp (Tamaki's
# separator DP, SEA 2019), one-word graphs (n <= 64).
#
# The CPU runs sepdp's preprocessing (blocks, work numbering, sort by size,
# packed sets, trie: sepdp_blk!, sepdp_lab!, sepdp_rel, sepdp_srt!,
# sepdp_set!, sepdp_tri). The GPU evaluates the blocks: one thread per block,
# one launch per block size (a block's children are smaller), then the root,
# one thread per graph. Each thread runs the search of sepdp_cap!/sepdp_uni!
# (one-word methods) with an explicit stack: sepdp_uni! tail-calls
# sepdp_cap!, so the stack holds cap frames only, each followed by its trie
# entries. Float64 weights are summed in the CPU's order, so every value and
# every cap is the CPU's. The CPU then runs sepdp_ord! on the caps.

using CUDA
using CliqueTrees
using CliqueTrees: bt_mat, bt_adj!, bt_clk, sepdp_blk!, sepdp_lab!, sepdp_rel, sepdp_srt!, sepdp_set!, sepdp_tri, sepdp_ord!, FVector, FScalar
using Graphs

export gpu_sepdp, cpu_sepdp

# ------------------------------------------------------------- device --

@inline bit(v) = one(UInt64) << ((v - 1) & 63)
@inline vmin(w::UInt64) = Int32(trailing_zeros(w) + 1)
@inline popc(w::UInt64) = Int32(count_ones(w))

@inline function cmp1(adj, g, m::UInt64, v)
    c = t = bit(v)
    while true
        u = zero(UInt64)
        while t != zero(UInt64)
            u |= @inbounds(adj[trailing_zeros(t) + 1, g]); t &= t - one(UInt64)
        end
        t = u & m & ~c
        t == zero(UInt64) && break
        c |= t
    end
    return c
end

@inline function nbr1(adj, g, b::UInt64)
    u = zero(UInt64); word = b
    while word != zero(UInt64)
        u |= @inbounds(adj[trailing_zeros(word) + 1, g]); word &= word - one(UInt64)
    end
    return u & ~b
end

# per-graph and per-block data (blocks of graph g at base[g] + 1 : base[g] + nblk[g])
# per-graph and per-block data: a NamedTuple of device arrays (fields as below)

# the block (local id) with smallest vertex ymin, ylen vertices and separator N, or 0
@inline function getblk(d, g, base, ymin, ylen, N)
    lo = @inbounds d.fptr[ymin, g]; last = @inbounds d.fptr[ymin + 1, g] - Int32(1); hi = last + Int32(1)
    while lo < hi
        p = lo + (hi - lo) >> 1
        if @inbounds(d.blklen[base + d.tgt[base + p]]) < ylen
            lo = p + Int32(1)
        else
            hi = p
        end
    end
    p = lo
    while p <= last
        b = @inbounds d.tgt[base + p]
        @inbounds(d.blklen[base + b]) == ylen || break
        @inbounds(d.sepset[base + b]) == N && return b
        p += Int32(1)
    end
    return Int32(0)
end

# stack frame of a cap search (words): 1 hdr (mode | pend<<8 | root<<16 | nbag<<24 | clen<<32 | cmin<<40),
# 2 bag, 3 pc, 4 wbag, 5 m, 6 mink, 7 q | qend<<32, 8 trie top; trie entries (p | lo<<21 | hi<<42) follow
const FR = 8
struct Stk{A}
    a::A
    off::Int64
end
@inline Base.getindex(s::Stk, i) = @inbounds s.a[s.off + i]
@inline Base.setindex!(s::Stk, x, i) = (@inbounds s.a[s.off + i] = x)

@inline packt(p, lo, hi) = UInt64(p) | (UInt64(lo) << 21) | (UInt64(hi) << 42)
@inline unpackt(x) = (Int32(x & 0x1fffff), Int32((x >> 21) & 0x1fffff), Int32((x >> 42) & 0x1fffff))

# start a cap search: either an immediate value (pushed = false), or a frame at sp
@inline function cap_start!(st, sp, d, g, base, maxk, bag, nbag, wbag, pc, clen, cmin, cwgt, m, root, capout, ci)
    k = max(wbag + cwgt, m)
    if k <= maxk
        @inbounds capout[ci] = bag | pc
        return k, sp, false
    end
    mink = k
    first = @inbounds d.fptr[cmin, g]; last = @inbounds d.fptr[cmin + 1, g] - Int32(1)
    lo = first; hi = last + Int32(1)
    while lo < hi
        p = lo + (hi - lo) >> 1
        if @inbounds(d.blklen[base + d.tgt[base + p]]) < clen
            lo = p + Int32(1)
        else
            hi = p
        end
    end
    q = first; qend = lo - Int32(1); lin = qend - q < 256
    top = sp + FR
    if !lin
        st[top + 1] = packt(@inbounds(d.head[cmin, g]), first, last); top += Int32(1)
    end
    st[sp + 1] = UInt64(lin ? 0 : 1) | (UInt64(root) << 16) | (UInt64(nbag) << 24) | (UInt64(clen) << 32) | (UInt64(cmin) << 40)
    st[sp + 2] = bag; st[sp + 3] = pc
    st[sp + 4] = reinterpret(UInt64, wbag); st[sp + 5] = reinterpret(UInt64, m); st[sp + 6] = reinterpret(UInt64, mink)
    st[sp + 7] = UInt64(UInt32(q)) | (UInt64(UInt32(qend)) << 32); st[sp + 8] = UInt64(top)
    return mink, top, true
end

# sepdp_uni!: an immediate value (tail = false) or the arguments of the tail call to cap
@inline function uni(d, g, base, n, maxk, bag, nbag, wbag, pc, x, m, root, capout, ci)
    rest = pc & ~bag
    x > 0 && (rest &= ~@inbounds(d.blkset[base + x]))
    nful = Int32(0); nnon = Int32(0); flen = Int32(0); fmin = Int32(0); mful = m; pf = zero(UInt64)
    while rest != zero(UInt64)
        v = vmin(rest)
        Y = cmp1(d.adj, g, rest, v); rest &= ~Y
        N = nbr1(d.adj, g, Y); ylen = popc(Y)
        y = getblk(d, g, base, v, ylen, N)
        if popc(N) == nbag
            nful += Int32(1)
            if y == 0
                nnon += Int32(1); flen = ylen; fmin = v; pf = Y
            else
                mful = max(mful, @inbounds d.valk[base + y])
            end
        else
            k = y == 0 ? typemax(Float64) : max(wbag, m, @inbounds d.valk[base + y])
            k > maxk && return k, false, zero(UInt64), Int32(0), Int32(0), 0.0, 0.0
            vy = @inbounds d.valk[base + y]
            m = max(m, vy); mful = max(mful, vy)
        end
    end
    if nnon == 0
        k = max(wbag, mful)
        k <= maxk && (@inbounds capout[ci] = bag)
        return k, false, zero(UInt64), Int32(0), Int32(0), 0.0, 0.0
    elseif nnon == 1 && (nful == 1 || root && 2flen > n - nbag)
        fwgt = 0.0; fone = typemax(Float64); word = pf
        while word != zero(UInt64)
            u = trailing_zeros(word) + 1; word &= word - one(UInt64)
            wu = @inbounds d.ww[u, g]; fwgt += wu; fone = min(fone, wu)
        end
        k = max(wbag + fone, mful)
        k <= maxk && return k, true, pf, flen, fmin, fwgt, mful
        return k, false, zero(UInt64), Int32(0), Int32(0), 0.0, 0.0
    end
    return typemax(Float64), false, zero(UInt64), Int32(0), Int32(0), 0.0, 0.0
end

# The whole search of one cap (block or root): returns its value.
# Frame modes: 0 linear scan, 1 trie scan. pend: 0 none, 1 a child for a
# candidate x is running, 2 the child for "cmin in the bag" is running.
@inline function capsearch!(st, stkw, d, g, base, n, maxk, wmin, bag0, nbag0, wbag0, pc0, clen0, cmin0, cwgt0, root0, capout, ci, ovf, oi)
    k0, top0, pushed0 = cap_start!(st, Int32(0), d, g, base, maxk, bag0, nbag0, wbag0, pc0, clen0, cmin0, cwgt0, 0.0, root0, capout, ci)
    pushed0 || return k0
    nfr = Int32(1); st[stkw] = UInt64(0)       # frame bases at the end of the stack, top first
    have = false; result = 0.0
    while true
        fp = Int32(st[stkw - nfr + 1])
        h = st[fp + 1]
        mode = Int32(h & 0xff); pend = Int32((h >> 8) & 0xff); root = ((h >> 16) & 0xff) != 0
        nbag = Int32((h >> 24) & 0xff); clen = Int32((h >> 32) & 0xff); cmin = Int32((h >> 40) & 0xff)
        bag = st[fp + 2]; pc = st[fp + 3]
        wbag = reinterpret(Float64, st[fp + 4]); m = reinterpret(Float64, st[fp + 5]); mink = reinterpret(Float64, st[fp + 6])
        ttop = Int32(st[fp + 8])
        ret = false; r = 0.0
        if have
            have = false
            if pend == 1
                if result <= maxk
                    ret = true; r = result
                else
                    mink = min(mink, result); st[fp + 6] = reinterpret(UInt64, mink)
                    st[fp + 1] = h & ~(UInt64(0xff) << 8)
                end
            else
                ret = true; r = min(mink, result)
            end
        end
        if !ret
            if ttop + 200 > stkw - nfr
                @inbounds ovf[oi] = Int32(1); return typemax(Float64)
            end
            # the next candidate x
            x = Int32(0)
            if mode == 0
                qq = st[fp + 7]; q = Int32(qq & 0xffffffff); qend = Int32(qq >> 32)
                if q <= qend
                    x = @inbounds d.tgt[base + q]; q += Int32(1)
                    st[fp + 7] = UInt64(UInt32(q)) | (UInt64(UInt32(qend)) << 32)
                end
            else
                while ttop > fp + FR
                    p, lo, hi = unpackt(st[ttop]); ttop -= Int32(1)
                    if p == 0
                        x = @inbounds d.tord[base + lo]; break
                    end
                    @inbounds(d.nodn[base + p]) < clen || continue
                    c = popc(@inbounds(d.nodall[base + p]) & ~bag); dd = popc(bag & ~@inbounds(d.nodany[base + p]))
                    max(m, wbag + c * wmin, @inbounds(d.nodw[base + p]) + dd * wmin) < mink || continue
                    st[ttop + 1] = packt(@inbounds(d.nodr[base + p]), p + Int32(1), hi)
                    st[ttop + 2] = packt(@inbounds(d.nodl[base + p]), lo, p); ttop += Int32(2)
                end
                st[fp + 8] = UInt64(ttop)
                if x != 0 && !(@inbounds(d.blklen[base + x]) < clen)
                    continue
                end
            end
            if x != 0
                kx = max(m, @inbounds d.valk[base + x])
                (kx <= maxk || kx < mink) || continue
                (@inbounds(d.blkset[base + x]) & bag) == zero(UInt64) || continue
                if kx > maxk
                    mink = kx; st[fp + 6] = reinterpret(UInt64, mink); continue
                end
                sx = @inbounds(d.sepset[base + x]) & ~bag; w = wbag; word = sx
                while word != zero(UInt64)
                    w += @inbounds(d.ww[trailing_zeros(word) + 1, g]); word &= word - one(UInt64)
                end
                kk = max(w, kx)
                if kk > maxk
                    mink = min(mink, kk); st[fp + 6] = reinterpret(UInt64, mink); continue
                end
                kc, tail, pf, flen, fmin, fwgt, mful = uni(d, g, base, n, maxk, bag | sx, nbag + popc(sx), w, pc, x, kx, root, capout, ci)
                if tail
                    kc, ntop, pushed = cap_start!(st, ttop, d, g, base, maxk, bag | sx, nbag + popc(sx), w, pf, flen, fmin, fwgt, mful, root, capout, ci)
                    if pushed
                        st[fp + 1] = (h & ~(UInt64(0xff) << 8)) | (UInt64(1) << 8)
                        nfr += Int32(1); st[stkw - nfr + 1] = UInt64(ttop)
                        continue
                    end
                end
                if kc <= maxk
                    ret = true; r = kc
                else
                    mink = min(mink, kc); st[fp + 6] = reinterpret(UInt64, mink); continue
                end
            else
                # cmin in the bag
                w = wbag + @inbounds(d.ww[cmin, g]); kk = max(w, m)
                if kk <= maxk
                    kc, tail, pf, flen, fmin, fwgt, mful = uni(d, g, base, n, maxk, bag | bit(cmin), nbag + Int32(1), w, pc, Int32(0), m, root, capout, ci)
                    if tail
                        kc, ntop, pushed = cap_start!(st, ttop, d, g, base, maxk, bag | bit(cmin), nbag + Int32(1), w, pf, flen, fmin, fwgt, mful, root, capout, ci)
                        if pushed
                            st[fp + 1] = (h & ~(UInt64(0xff) << 8)) | (UInt64(2) << 8)
                            nfr += Int32(1); st[stkw - nfr + 1] = UInt64(ttop)
                            continue
                        end
                    end
                    kk = kc
                end
                ret = true; r = min(mink, kk)
            end
        end
        # this frame returns r
        nfr -= Int32(1)
        nfr == 0 && return r
        have = true; result = r
    end
end

# jobs: (graph, local block) of one size class; root jobs: block = 0
function val_kernel!(d, jg, jb, njob::Int32, nbag0, wbag0, blkmin, blkwgt, stk, stkw::Int32, capout, ovf, next)
    me = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    st = Stk(stk, (me - Int32(1)) * Int64(stkw))
    while true
    j = CUDA.atomic_add!(pointer(next), Int32(1)) + Int32(1)
    j > njob && break
    g = @inbounds jg[j]; b = @inbounds jb[j]
    base = @inbounds d.base[g]; n = @inbounds d.nvs[g]; maxk = @inbounds d.maxk[g]; wmin = @inbounds d.wmin[g]
    if b > 0
        gb = base + b
        v = capsearch!(st, stkw, d, g, base, n, maxk, wmin, @inbounds(d.sepset[gb]), @inbounds(nbag0[gb]), @inbounds(wbag0[gb]),
            @inbounds(d.blkset[gb]), @inbounds(d.blklen[gb]), @inbounds(blkmin[gb]), @inbounds(blkwgt[gb]), false, capout, gb, ovf, j)
        @inbounds d.valk[gb] = v
    else
        # the root: capout and valk slots after the blocks of all graphs
        mask = n == 64 ? typemax(UInt64) : (one(UInt64) << n) - one(UInt64)
        wall = 0.0
        for v in Int32(1):n
            wall += @inbounds d.ww[v, g]
        end
        ri = @inbounds(d.base[length(d.base)]) + g   # base has G + 1 entries
        v = capsearch!(st, stkw, d, g, base, n, maxk, wmin, zero(UInt64), Int32(0), 0.0, mask, n, Int32(1), wall, true, capout, ri, ovf, j)
        @inbounds d.valk[ri] = v
    end
    end
    return
end

# --------------------------------------------------------------- host --

# CPU preprocessing of sepdp (as in sepdp_run!, one thread)
function prep(weights, graph, sep)
    nvtx = nv(graph); V = Int; W = Float64
    adj = bt_adj!(bt_mat(1, nvtx), graph); clk = bt_clk(Inf)
    nsep = nv(sep); sepwgt = FVector{W}(undef, nsep); work1 = FVector{V}(undef, nvtx)
    lstmin, lstlen, lstwgt, lstsep, lstset = sepdp_blk!(sepwgt, adj, weights, graph, sep, clk)
    nblk = length(lstmin)
    new2old, old2new, moved = sepdp_lab!(lstmin, lstset, nvtx, 1)
    ww, gg, ss, aa = moved ? sepdp_rel(weights, graph, sep, adj, new2old, old2new) : (weights, graph, sep, adj)
    blkmin = FVector{V}(undef, nblk); blklen = FVector{V}(undef, nblk); blkwgt = FVector{W}(undef, nblk)
    blksep = FVector{V}(undef, nblk); order = FVector{V}(undef, nblk)
    fst = sepdp_srt!(blkmin, blklen, blkwgt, blksep, order, work1, lstmin, lstlen, lstwgt, lstsep, nvtx)
    blkset, sepset = sepdp_set!(aa, ss, blksep, order, lstset, nvtx)
    trie = sepdp_tri(fst, blklen, sepset, ww)
    return (; nvtx, nblk, new2old, moved, ww = collect(ww[1:nvtx]), gg, ss, aa = collect(aa), fst, blkmin = collect(blkmin), blklen = collect(blklen),
        blkwgt = collect(blkwgt), blksep = collect(blksep), blkset = collect(blkset), sepset = collect(sepset), trie,
        nbag0 = Int32[length(neighbors(ss, blksep[b])) for b in 1:nblk], wbag0 = Float64[sepwgt[blksep[b]] for b in 1:nblk])
end

# the ordering from the caps (sepdp_ord!), back in the graph's numbering
function finish(P, valk, capw, rootval, rootcap, maxk)
    nvtx = P.nvtx; V = Int
    perm = collect(1:nvtx); invp = collect(1:nvtx)
    mink = rootval
    if mink <= maxk
        ptr = Int[1]; tgt = Int[]
        for w in vcat(capw, rootcap)
            word = w
            while word != 0
                push!(tgt, trailing_zeros(word) + 1); word &= word - 1
            end
            push!(ptr, length(tgt) + 1)
        end
        cap = BipartiteGraph{V, V}(nvtx, P.nblk + 1, length(tgt), ptr, tgt)
        tag = FScalar{V}(undef); tag[] = 0
        p = FVector{V}(undef, nvtx); ip = FVector{V}(undef, nvtx)
        sepdp_ord!(p, ip, FVector{V}(undef, nvtx + 1), FVector{V}(undef, nvtx), FVector{V}(undef, nvtx), tag, P.gg, P.ss, P.fst, P.blklen, P.blksep, cap)
        perm = collect(p)
        if P.moved
            perm = [P.new2old[perm[i]] for i in 1:nvtx]
        end
        invp = invperm(perm)
    end
    return perm, invp, mink, mink <= maxk ? :found : :above
end

"""
    gpu_sepdp(weights, graphs, seps, maxks) -> results [(perm, invp, mink, status)], timings

Batched sepdp: CPU preprocessing, GPU block values and roots, CPU ordering.
"""
function gpu_sepdp(weights, graphs, seps, maxks; stkw::Int = 4096, threads::Int = 64)
    isempty(graphs) && return [], (; tprep = 0.0, tupload = 0.0, tgpu = 0.0, tfinish = 0.0, novf = 0, nblk = 0), Float64[], UInt64[], Int32[]
    G = length(graphs)
    tp = @elapsed Ps = [prep(weights[i], graphs[i], seps[i]) for i in 1:G]
    nb = [P.nblk for P in Ps]; base = Int32.(cumsum(vcat(0, nb)))   # G + 1 entries
    NB = Int(base[end])
    cat(f, T) = convert(Vector{T}, reduce(vcat, [f(P) for P in Ps]))
    adj = zeros(UInt64, 64, G); ww = zeros(Float64, 64, G); head = zeros(Int32, 65, G); fptr = zeros(Int32, 66, G)
    for (g, P) in enumerate(Ps)
        n = P.nvtx
        adj[1:n, g] .= P.aa; ww[1:n, g] .= P.ww
        head[1:n, g] .= P.trie.head
        fp = CliqueTrees.pointers(P.fst); fptr[1:n+1, g] .= fp
    end
    tr(f) = cat(P -> f(P.trie), Int32)
    t0 = time()
    d = (; adj = CuArray(adj), ww = CuArray(ww), head = CuArray(head), fptr = CuArray(fptr), nvs = CuArray(Int32[P.nvtx for P in Ps]),
        maxk = CuArray(Float64.(maxks)), wmin = CuArray(Float64[P.trie.wmin for P in Ps]), base = CuArray(base),
        blkset = CuArray(cat(P -> P.blkset, UInt64)), sepset = CuArray(cat(P -> P.sepset, UInt64)), blklen = CuArray(cat(P -> P.blklen, Int32)),
        tgt = CuArray(cat(P -> CliqueTrees.targets(P.fst), Int32)), tord = CuArray(tr(t -> t.tord)), nodl = CuArray(tr(t -> t.nodl)), nodr = CuArray(tr(t -> t.nodr)),
        nodn = CuArray(tr(t -> t.nodn)), nodw = CuArray(cat(P -> P.trie.nodw, Float64)), nodall = CuArray(cat(P -> P.trie.nodall, UInt64)),
        nodany = CuArray(cat(P -> P.trie.nodany, UInt64)), valk = CUDA.zeros(Float64, NB + G))
    nbag0 = CuArray(cat(P -> P.nbag0, Int32)); wbag0 = CuArray(cat(P -> P.wbag0, Float64))
    blkmin = CuArray(cat(P -> P.blkmin, Int32)); blkwgt = CuArray(cat(P -> P.blkwgt, Float64))
    capout = CUDA.zeros(UInt64, NB + G)
    # jobs by size class
    maxlen = maximum(P -> isempty(P.blklen) ? 0 : maximum(P.blklen), Ps; init = 0)
    classes = [(Int32[], Int32[]) for _ in 1:maxlen]
    for (g, P) in enumerate(Ps), b in 1:P.nblk
        push!(classes[P.blklen[b]][1], g); push!(classes[P.blklen[b]][2], b)
    end
    maxjobs = max(G, maximum(c -> length(c[1]), classes; init = 0))
    ovf = CUDA.zeros(Int32, maxjobs); next = CUDA.zeros(Int32, 1)
    dummy = (CuArray(Int32[1]), CuArray(Int32[1]))
    kern = @cuda launch = false val_kernel!(d, dummy[1], dummy[2], Int32(1), nbag0, wbag0, blkmin, blkwgt, CUDA.zeros(UInt64, 1), Int32(stkw), capout, ovf, next)
    sms = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    P = sms * CUDA.active_blocks(kern.fun, threads) * threads      # resident threads (occupancy API)
    stk = CUDA.zeros(UInt64, stkw * P)
    tup = time() - t0
    novf = 0
    run(jg, jb) = begin
        nj = length(jg); next .= 0
        kern(d, CuArray(jg), CuArray(jb), Int32(nj), nbag0, wbag0, blkmin, blkwgt, stk, Int32(stkw), capout, ovf, next;
            threads, blocks = cld(min(nj, P), threads))
    end
    tg = CUDA.@elapsed begin
        for (jg, jb) in classes
            isempty(jg) || run(jg, jb)
        end
        run(Int32.(1:G), zeros(Int32, G))
    end
    novf = count(!iszero, Array(ovf))
    valk = Array(d.valk); capw = Array(capout)
    tf = @elapsed res = [finish(Ps[g], valk[base[g]+1:base[g+1]], capw[base[g]+1:base[g+1]], valk[NB + g], capw[NB + g], maxks[g]) for g in 1:G]
    return res, (; tprep = tp, tupload = tup, tgpu = tg, tfinish = tf, novf, nblk = NB), valk, capw, base
end

# CPU reference: CliqueTrees.sepdp, one thread
function cpu_sepdp(weights, graph, sep, maxk)
    perm, invp, mink, status = CliqueTrees.sepdp(weights, graph, sep, maxk, Inf, 1)
    return collect(perm), collect(invp), mink, status == CliqueTrees.BT_FOUND ? :found : :above
end

end
