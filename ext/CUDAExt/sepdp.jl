module SepDPGPU

# GPU block evaluation of CliqueTrees' sepdp for any word count (sets are
# UInt64 or Bits{W}); the W-generic form of SepDPGPU, following the
# multi-word methods of sepdp_cap!/sepdp_uni!. CPU: preprocessing and the
# final ordering; GPU: block values (one thread per block, persistent
# threads per size class) and roots.

using CUDA
using CliqueTrees
using CliqueTrees: bt_mat, bt_adj!, bt_clk, sepdp_blk!, sepdp_lab!, sepdp_rel, sepdp_srt!, sepdp_set!, sepdp_tri, sepdp_ord!, FVector, FScalar
using Graphs

using ..GPUBits
export gpu_sepdp, cpu_sepdp, settype

# (the abort flags of the graphs, as in SepEnumGPU)
using ..SepEnumGPU: ABORTSTREAM, cutflags, upload

const WS = Dict{Symbol, Any}()
function buf(name::Symbol, ::Type{E}, n::Integer) where {E}
    x = get(WS, name, nothing)
    if x === nothing || length(x) < n
        x = CuArray{E}(undef, max(n, x === nothing ? 0 : 2length(x)))
        WS[name] = x
    end
    return x
end

settype(n) = n <= 64 ? UInt64 : Bits{cld(n, 64)}

struct Stk{A}
    a::A
    off::Int64
end
@inline Base.getindex(s::Stk, i) = @inbounds s.a[s.off + i]
@inline Base.setindex!(s::Stk, x, i) = (@inbounds s.a[s.off + i] = x)

@inline adjset(::Type{UInt64}, d, g, v) = @inbounds d.adj[v, g]
@inline adjset(::Type{Bits{W}}, d, g, v) where {W} = Bits{W}(ntuple(j -> @inbounds(d.adj[(v - 1) * W + j, g]), Val(W)))
@inline gset(::Type{T}, a, i) where {T} = ldset(T, a, i)    # set number i of a flat array

@inline function cmpset(::Type{T}, d, g, m::T, v) where {T}
    c = t = bitof(T, v)
    while true
        u = zero(T)
        while !iszero(t)
            u = u | adjset(T, d, g, vmin(t)); t = clearmin(t)
        end
        t = u & m & ~c
        iszero(t) && break
        c = c | t
    end
    return c
end

@inline function nbrset(::Type{T}, d, g, b::T) where {T}
    u = zero(T); word = b
    while !iszero(word)
        u = u | adjset(T, d, g, vmin(word)); word = clearmin(word)
    end
    return u & ~b
end

@inline function getblk(::Type{T}, d, g, base, ymin, ylen, N) where {T}
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
        gset(T, d.sepset, base + b) == N && return b
        p += Int32(1)
    end
    return Int32(0)
end

# frame: 1 hdr (mode | pend<<8 | root<<16 | nbag<<24 (10b) | clen<<34 (10b) | cmin<<44 (10b)),
# 2 wbag, 3 m, 4 mink, 5 q | qend<<32, 6 trie top, then bag (W words), pc (W words); trie entries follow
@inline frsize(::Type{T}) where {T} = 6 + 2nwords(T)
@inline packt(p, lo, hi) = UInt64(p) | (UInt64(lo) << 21) | (UInt64(hi) << 42)
@inline unpackt(x) = (Int32(x & 0x1fffff), Int32((x >> 21) & 0x1fffff), Int32((x >> 42) & 0x1fffff))

@inline function cap_start!(::Type{T}, st, sp, d, g, base, maxk, bag, nbag, wbag, pc, clen, cmin, cwgt, m, root, capout, ci) where {T}
    W = nwords(T)
    k = max(wbag + cwgt, m)
    if k <= maxk
        stset!(capout, ci, bag | pc)
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
    top = sp + frsize(T)
    if !lin
        st[top + 1] = packt(@inbounds(d.head[cmin, g]), first, last); top += Int32(1)
    end
    st[sp + 1] = UInt64(lin ? 0 : 1) | (UInt64(root) << 16) | (UInt64(nbag) << 24) | (UInt64(clen) << 34) | (UInt64(cmin) << 44)
    st[sp + 2] = reinterpret(UInt64, wbag); st[sp + 3] = reinterpret(UInt64, m); st[sp + 4] = reinterpret(UInt64, mink)
    st[sp + 5] = UInt64(UInt32(q)) | (UInt64(UInt32(qend)) << 32); st[sp + 6] = UInt64(top)
    stw!(st, sp + 6, bag); stw!(st, sp + 6 + W, pc)
    return mink, top, true
end

@inline function uni(::Type{T}, d, g, base, n, maxk, bag, nbag, wbag, pc, x, m, root, capout, ci) where {T}
    rest = pc & ~bag
    x > 0 && (rest = rest & ~gset(T, d.blkset, base + x))
    nful = Int32(0); nnon = Int32(0); flen = Int32(0); fmin = Int32(0); mful = m; pf = zero(T)
    while !iszero(rest)
        v = vmin(rest)
        Y = cmpset(T, d, g, rest, v); rest = rest & ~Y
        N = nbrset(T, d, g, Y); ylen = popc(Y)
        y = getblk(T, d, g, base, v, ylen, N)
        if popc(N) == nbag
            nful += Int32(1)
            if y == 0
                nnon += Int32(1); flen = ylen; fmin = v; pf = Y
            else
                mful = max(mful, @inbounds d.valk[base + y])
            end
        else
            k = y == 0 ? typemax(Float64) : max(wbag, m, @inbounds d.valk[base + y])
            k > maxk && return k, false, zero(T), Int32(0), Int32(0), 0.0, 0.0
            vy = @inbounds d.valk[base + y]
            m = max(m, vy); mful = max(mful, vy)
        end
    end
    if nnon == 0
        k = max(wbag, mful)
        k <= maxk && stset!(capout, ci, bag)
        return k, false, zero(T), Int32(0), Int32(0), 0.0, 0.0
    elseif nnon == 1 && (nful == 1 || root && 2flen > n - nbag)
        fwgt = 0.0; fone = typemax(Float64); word = pf
        while !iszero(word)
            u = vmin(word); word = clearmin(word)
            wu = @inbounds d.ww[u, g]; fwgt += wu; fone = min(fone, wu)
        end
        k = max(wbag + fone, mful)
        k <= maxk && return k, true, pf, flen, fmin, fwgt, mful
        return k, false, zero(T), Int32(0), Int32(0), 0.0, 0.0
    end
    return typemax(Float64), false, zero(T), Int32(0), Int32(0), 0.0, 0.0
end

@inline function capsearch!(::Type{T}, st, stkw, d, g, base, n, maxk, wmin, bag0, nbag0, wbag0, pc0, clen0, cmin0, cwgt0, root0, capout, ci, ovf, oi, gab) where {T}
    W = nwords(T); FR = frsize(T)
    k0, top0, pushed0 = cap_start!(T, st, Int32(0), d, g, base, maxk, bag0, nbag0, wbag0, pc0, clen0, cmin0, cwgt0, 0.0, root0, capout, ci)
    pushed0 || return k0
    nfr = Int32(1); st[stkw] = UInt64(0)
    have = false; result = 0.0; steps = Int64(0)
    while true
        steps += 1
        # cooperative cancellation: the abort flag of the graph
        if (steps & 1023) == 0 && CUDA.atomic_add!(pointer(gab, g), Int32(0)) != Int32(0)
            @inbounds ovf[oi] = Int32(3); return typemax(Float64)
        end
        fp = Int32(st[stkw - nfr + 1])
        h = st[fp + 1]
        mode = Int32(h & 0xff); pend = Int32((h >> 8) & 0xff); root = ((h >> 16) & 0xff) != 0
        nbag = Int32((h >> 24) & 0x3ff); clen = Int32((h >> 34) & 0x3ff); cmin = Int32((h >> 44) & 0x3ff)
        wbag = reinterpret(Float64, st[fp + 2]); m = reinterpret(Float64, st[fp + 3]); mink = reinterpret(Float64, st[fp + 4])
        bag = ldw(T, st, fp + 6); pc = ldw(T, st, fp + 6 + W)
        ttop = Int32(st[fp + 6])
        ret = false; r = 0.0
        if have
            have = false
            if pend == 1
                if result <= maxk
                    ret = true; r = result
                else
                    mink = min(mink, result); st[fp + 4] = reinterpret(UInt64, mink)
                    st[fp + 1] = h & ~(UInt64(0xff) << 8)
                end
            else
                ret = true; r = min(mink, result)
            end
        end
        if !ret
            if ttop + 4FR + 2 * 64W > stkw - nfr
                @inbounds ovf[oi] = Int32(1); return typemax(Float64)
            end
            x = Int32(0)
            if mode == 0
                qq = st[fp + 5]; q = Int32(qq & 0xffffffff); qend = Int32(qq >> 32)
                if q <= qend
                    x = @inbounds d.tgt[base + q]; q += Int32(1)
                    st[fp + 5] = UInt64(UInt32(q)) | (UInt64(UInt32(qend)) << 32)
                end
            else
                while ttop > fp + FR
                    p, lo, hi = unpackt(st[ttop]); ttop -= Int32(1)
                    if p == 0
                        x = @inbounds d.tord[base + lo]; break
                    end
                    @inbounds(d.nodn[base + p]) < clen || continue
                    c = popc(gset(T, d.nodall, base + p) & ~bag); dd = popc(bag & ~gset(T, d.nodany, base + p))
                    max(m, wbag + c * wmin, @inbounds(d.nodw[base + p]) + dd * wmin) < mink || continue
                    st[ttop + 1] = packt(@inbounds(d.nodr[base + p]), p + Int32(1), hi)
                    st[ttop + 2] = packt(@inbounds(d.nodl[base + p]), lo, p); ttop += Int32(2)
                end
                st[fp + 6] = UInt64(ttop)
                if x != 0 && !(@inbounds(d.blklen[base + x]) < clen)
                    continue
                end
            end
            if x != 0
                kx = max(m, @inbounds d.valk[base + x])
                (kx <= maxk || kx < mink) || continue
                iszero(gset(T, d.blkset, base + x) & bag) || continue
                if kx > maxk
                    mink = kx; st[fp + 4] = reinterpret(UInt64, mink); continue
                end
                sx = gset(T, d.sepset, base + x) & ~bag; w = wbag; word = sx
                while !iszero(word)
                    w += @inbounds(d.ww[vmin(word), g]); word = clearmin(word)
                end
                kk = max(w, kx)
                if kk > maxk
                    mink = min(mink, kk); st[fp + 4] = reinterpret(UInt64, mink); continue
                end
                kc, tail, pf, flen, fmin, fwgt, mful = uni(T, d, g, base, n, maxk, bag | sx, nbag + popc(sx), w, pc, x, kx, root, capout, ci)
                if tail
                    kc, ntop, pushed = cap_start!(T, st, ttop, d, g, base, maxk, bag | sx, nbag + popc(sx), w, pf, flen, fmin, fwgt, mful, root, capout, ci)
                    if pushed
                        st[fp + 1] = (h & ~(UInt64(0xff) << 8)) | (UInt64(1) << 8)
                        nfr += Int32(1); st[stkw - nfr + 1] = UInt64(ttop)
                        continue
                    end
                end
                if kc <= maxk
                    ret = true; r = kc
                else
                    mink = min(mink, kc); st[fp + 4] = reinterpret(UInt64, mink); continue
                end
            else
                w = wbag + @inbounds(d.ww[cmin, g]); kk = max(w, m)
                if kk <= maxk
                    bc = bag | bitof(T, cmin)
                    kc, tail, pf, flen, fmin, fwgt, mful = uni(T, d, g, base, n, maxk, bc, nbag + Int32(1), w, pc, Int32(0), m, root, capout, ci)
                    if tail
                        kc, ntop, pushed = cap_start!(T, st, ttop, d, g, base, maxk, bc, nbag + Int32(1), w, pf, flen, fmin, fwgt, mful, root, capout, ci)
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
        nfr -= Int32(1)
        nfr == 0 && return r
        have = true; result = r
    end
end

function val_kernel!(::Type{T}, d, jg, jb, joff::Int32, njob::Int32, nbag0, wbag0, blkmin, blkwgt, stk, stkw::Int32, capout, ovf, next, gab) where {T}
    me = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    st = Stk(stk, (me - Int32(1)) * Int64(stkw))
    while true
        j = CUDA.atomic_add!(pointer(next), Int32(1)) + Int32(1)
        j > njob && break
        g = @inbounds jg[joff + j]; b = @inbounds jb[joff + j]
        base = @inbounds d.base[g]; n = @inbounds d.nvs[g]; maxk = @inbounds d.maxk[g]; wmin = @inbounds d.wmin[g]
        if b > 0
            gb = base + b
            v = capsearch!(T, st, stkw, d, g, base, n, maxk, wmin, gset(T, d.sepset, gb), @inbounds(nbag0[gb]), @inbounds(wbag0[gb]),
                gset(T, d.blkset, gb), @inbounds(d.blklen[gb]), @inbounds(blkmin[gb]), @inbounds(blkwgt[gb]), false, capout, gb, ovf, joff + j, gab)
            @inbounds d.valk[gb] = v
        else
            wall = 0.0
            for v in Int32(1):n
                wall += @inbounds d.ww[v, g]
            end
            ri = @inbounds(d.base[length(d.base)]) + g
            v = capsearch!(T, st, stkw, d, g, base, n, maxk, wmin, zero(T), Int32(0), 0.0, maskn(T, n), n, Int32(1), wall, true, capout, ri, ovf, joff + j, gab)
            @inbounds d.valk[ri] = v
        end
    end
    return
end

# --------------------------------------------------------------- host --

# (nothing if the blocks are not listed before the deadline dl)
function prep(weights, graph, sep, dl = Inf)
    nvtx = nv(graph); V = Int; Wt = Float64; nwrd = cld(max(nvtx, 1), 64)
    adj = bt_adj!(bt_mat(nwrd, nvtx), graph); clk = bt_clk(dl)
    clk.out[] && return nothing
    nsep = nv(sep); sepwgt = FVector{Wt}(undef, nsep); work1 = FVector{V}(undef, nvtx)
    lstmin, lstlen, lstwgt, lstsep, lstset = sepdp_blk!(sepwgt, adj, weights, graph, sep, clk)
    clk.out[] && return nothing
    nblk = length(lstmin)
    new2old, old2new, moved = sepdp_lab!(lstmin, lstset, nvtx, nwrd)
    ww, gg, ss, aa = moved ? sepdp_rel(weights, graph, sep, adj, new2old, old2new) : (weights, graph, sep, adj)
    blkmin = FVector{V}(undef, nblk); blklen = FVector{V}(undef, nblk); blkwgt = FVector{Wt}(undef, nblk)
    blksep = FVector{V}(undef, nblk); order = FVector{V}(undef, nblk)
    fst = sepdp_srt!(blkmin, blklen, blkwgt, blksep, order, work1, lstmin, lstlen, lstwgt, lstsep, nvtx)
    blkset, sepset = sepdp_set!(aa, ss, blksep, order, lstset, nvtx)
    trie = sepdp_tri(fst, blklen, sepset, ww)
    return (; nvtx, nwrd, nblk, new2old, moved, ww = collect(ww[1:nvtx]), gg, ss, aa = collect(aa), fst, blkmin = collect(blkmin), blklen = collect(blklen),
        blkwgt = collect(blkwgt), blksep = collect(blksep), blkset = collect(blkset), sepset = collect(sepset), trie,
        nbag0 = Int32[length(neighbors(ss, blksep[b])) for b in 1:nblk], wbag0 = Float64[sepwgt[blksep[b]] for b in 1:nblk])
end

# a list of nwrd-word sets (vector for one word, nwrd x m matrix otherwise) as W words each
function widen(x, nwrd, W)
    m = nwrd == 1 ? length(x) : size(x, 2)
    out = zeros(UInt64, W * m)
    for i in 1:m, j in 1:nwrd
        out[(i - 1) * W + j] = nwrd == 1 ? x[i] : x[j, i]
    end
    return out
end

# the ordering from the caps (sepdp_ord!); caps are read as words from capw
# (blocks base+1 .. base+nblk, then the root at slot ri)
function finish(P, capw::Vector{UInt64}, W::Int, base::Int, ri::Int, rootval, maxk)
    nvtx = P.nvtx; V = Int
    perm = collect(1:nvtx); mink = rootval
    if mink <= maxk
        nb = P.nblk
        slot(b) = b <= nb ? base + b : ri
        ptr = Vector{Int}(undef, nb + 2); ptr[1] = 1
        @inbounds for b in 1:nb + 1
            o = (slot(b) - 1) * W; c = 0
            for j in 1:W
                c += count_ones(capw[o + j])
            end
            ptr[b + 1] = ptr[b] + c
        end
        tgt = Vector{Int}(undef, ptr[end] - 1)
        @inbounds for b in 1:nb + 1
            o = (slot(b) - 1) * W; q = ptr[b]
            for j in 1:W
                word = capw[o + j]
                while word != 0
                    tgt[q] = 64(j - 1) + trailing_zeros(word) + 1; q += 1; word &= word - one(UInt64)
                end
            end
        end
        cap = BipartiteGraph{V, V}(nvtx, nb + 1, length(tgt), ptr, tgt)
        tag = FScalar{V}(undef); tag[] = 0
        p = FVector{V}(undef, nvtx); ip = FVector{V}(undef, nvtx)
        sepdp_ord!(p, ip, FVector{V}(undef, nvtx + 1), FVector{V}(undef, nvtx), FVector{V}(undef, nvtx), tag, P.gg, P.ss, P.fst, P.blklen, P.blksep, cap)
        perm = collect(p)
        P.moved && (perm = [P.new2old[perm[i]] for i in 1:nvtx])
    end
    return perm, invperm(perm), mink, mink <= maxk ? :found : :above
end

# jobs (graph, block) grouped by block size (class c = size c), then the
# roots (block 0); coff[c] is the offset of class c, the roots are the last class
function buildjobs(blklens::Vector{Vector{Int}}, maxlen::Int, G::Int)
    cnt = zeros(Int, maxlen + 1)
    for bl in blklens, x in bl
        cnt[x] += 1
    end
    cnt[maxlen + 1] = G
    coff = zeros(Int, maxlen + 2)
    for c in 1:maxlen + 1
        coff[c + 1] = coff[c] + cnt[c]
    end
    JG = Vector{Int32}(undef, coff[end]); JB = Vector{Int32}(undef, coff[end])
    pos = copy(coff)
    for (g, bl) in enumerate(blklens), b in eachindex(bl)
        c = bl[b]; pos[c] += 1; JG[pos[c]] = g; JB[pos[c]] = b
    end
    for g in 1:G
        pos[maxlen + 1] += 1; JG[pos[maxlen + 1]] = g; JB[pos[maxlen + 1]] = 0
    end
    return JG, JB, coff
end

# The kernels of a part of a batch: the prepared graphs Ps at the widths
# ks, on the stream s; the stacks take a share 1/nshare of a quarter of
# the free memory. Returns the state of the launch (see collect!).
function launch(::Type{T}, Ps, ks, s::CuStream, part::Int, nshare::Int; stkw::Int, threads::Int, maxresident::Int) where {T}
    W = nwords(T); NM = 64W; L = length(Ps)
    nb = [P.nblk for P in Ps]; base = Int32.(cumsum(vcat(0, nb))); NB = Int(base[end])
    cat(f, Tt) = convert(Vector{Tt}, reduce(vcat, [f(P) for P in Ps]))
    tr(f) = cat(P -> f(P.trie), Int32)
    adj = zeros(UInt64, NM * W, L); ww = zeros(Float64, NM, L); head = zeros(Int32, NM + 1, L); fptr = zeros(Int32, NM + 2, L)
    for (g, P) in enumerate(Ps)
        n = P.nvtx
        a = widen(P.aa, P.nwrd, W); adj[1:length(a), g] .= a
        ww[1:n, g] .= P.ww; head[1:n, g] .= P.trie.head
        fp = CliqueTrees.pointers(P.fst); fptr[1:n+1, g] .= fp
    end
    maxlen = maximum(P -> isempty(P.blklen) ? 0 : maximum(P.blklen), Ps; init = 0)
    JG, JB, coff = buildjobs([P.blklen for P in Ps], maxlen, L)   # jobs sorted by class, then the roots
    out = falses(L)
    CUDA.stream!(s) do
        # (three copies, one for each element type)
        adj_, blkset, sepset, nodall, nodany = upload([adj, cat(P -> widen(P.blkset, P.nwrd, W), UInt64), cat(P -> widen(P.sepset, P.nwrd, W), UInt64),
            cat(P -> widen(P.trie.nodall, P.nwrd, W), UInt64), cat(P -> widen(P.trie.nodany, P.nwrd, W), UInt64)])
        ww_, maxk, wmin, nodw, wbag0, blkwgt = upload([ww, Float64[k for k in ks], Float64[P.trie.wmin for P in Ps], cat(P -> P.trie.nodw, Float64),
            cat(P -> P.wbag0, Float64), cat(P -> P.blkwgt, Float64)])
        head_, fptr_, nvs, base_, blklen, tgt, tord, nodl, nodr, nodn, nbag0, blkmin, djg, djb = upload([head, fptr, Int32[P.nvtx for P in Ps], base,
            cat(P -> P.blklen, Int32), cat(P -> CliqueTrees.targets(P.fst), Int32), tr(t -> t.tord), tr(t -> t.nodl), tr(t -> t.nodr), tr(t -> t.nodn),
            cat(P -> P.nbag0, Int32), cat(P -> P.blkmin, Int32), JG, JB])
        d = (; adj = adj_, ww = ww_, head = head_, fptr = fptr_, nvs, maxk, wmin, base = base_, blkset, sepset, blklen, tgt, tord, nodl, nodr, nodn,
            nodw, nodall, nodany, valk = CUDA.zeros(Float64, NB + L))
        capout = CUDA.zeros(UInt64, (NB + L) * W)
        ovf = buf(Symbol(:ovf, part), Int32, length(JG)); fill!(view(ovf, 1:length(JG)), 0); next = buf(Symbol(:next, part), Int32, 1)
        gab = cutflags(Symbol(:gabdp, part), out)
        # (the address of the flags, taken here, on the stream s: taken later,
        # on the stream of the task, it would wait for the kernels)
        gabp = reinterpret(CuPtr{UInt32}, pointer(gab))
        kern = @cuda launch = false val_kernel!(T, d, djg, djb, Int32(0), Int32(1), nbag0, wbag0, blkmin, blkwgt, CUDA.zeros(UInt64, 1), Int32(stkw), capout, ovf, next, gab)
        sms = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        Pmem = max(threads, (CUDA.free_memory() ÷ (4nshare)) ÷ (8stkw) ÷ threads * threads)
        P = min(sms * CUDA.active_blocks(kern.fun, threads) * threads, maxresident, Pmem)
        stk = buf(Symbol(:stk, part), UInt64, stkw * P)
        e0 = CuEvent(); e1 = CuEvent(); CUDA.record(e0)
        for c in 1:length(coff) - 1
            nj = coff[c + 1] - coff[c]
            nj == 0 && continue
            fill!(next, 0)
            kern(T, d, djg, djb, Int32(coff[c]), Int32(nj), nbag0, wbag0, blkmin, blkwgt, stk, Int32(stkw), capout, ovf, next, gab;
                threads, blocks = cld(min(nj, P), threads))
        end
        CUDA.record(e1)
        return (; Ps, base, NB, JG, d, capout, ovf, gab, gabp, out, e0, e1, s, collected = Ref(false))
    end
end

# The block values of the graphs on the GPU, then their orderings: res[g] =
# (perm, invp, mink, :found | :above), or nothing if the graph was stopped
# (cut(g), by default its deadline dl; or its preprocessing ran past the
# deadline of its clock clks[g]), or :failed if its search overflowed a
# stack (ovf[g]). done(g, res[g]) is called once the result of g is known.
#
# The batch goes to the GPU in two parts, on two streams: the host prepares
# the second part while the GPU searches the first, and makes the orderings
# of the first while the GPU searches the second.
function gpu_sepdp(::Type{T}, weights, graphs, seps, maxks; stkw::Int = 4096 * nwords(T), threads::Int = 64, maxresident::Int = 1 << 15, dl::Float64 = Inf,
        cut = g -> time() > dl, clks = [bt_clk(dl) for _ in graphs], done = (g, r) -> nothing) where {T}
    W = nwords(T); G = length(graphs)
    res = Vector{Any}(nothing, G); ovg = falses(G); tp = tup = tg = tf = 0.0; NB = 0
    info(aborted) = (; tprep = tp, tupload = tup, tgpu = tg, tfinish = tf, novf = count(ovg), ovf = ovg, nblk = NB, aborted)
    G == 0 && return res, info(false)
    np = G >= 2 ? 2 : 1; parts = [i:np:G for i in 1:np]
    ABORTSTREAM[] === nothing && (ABORTSTREAM[] = CuStream(; flags = CUDA.STREAM_NON_BLOCKING))
    while length(STREAMS) < np
        push!(STREAMS, CuStream(; flags = CUDA.STREAM_NON_BLOCKING))
    end
    launched = []; ans = 0

    # the results of a part, once its kernels are done
    function collect!(x, idx)
        ho = Array(view(x.ovf, 1:length(x.JG)))
        for j in eachindex(x.JG)
            ho[j] != 0 && !x.out[x.JG[j]] && (ovg[idx[x.JG[j]]] = true)
        end
        valk = Array(x.d.valk); capw = Array(x.capout)
        tf += @elapsed for (j, g) in enumerate(idx)
            if ovg[g]
                res[g] = :failed
            elseif !x.out[j] && !cut(g)
                res[g] = finish(x.Ps[j], capw, W, Int(x.base[j]), x.NB + j, valk[x.NB + j], maxks[g])
            end
            done(g, res[g]); ans += 1
        end
        tg += CUDA.elapsed(x.e0, x.e1); x.collected[] = true
        return
    end

    for (p, part) in enumerate(parts)
        # (a graph whose preprocessing is stopped leaves the batch)
        t = time()
        Ps = [cut(i) ? nothing : prep(weights[i], graphs[i], seps[i], clks[i]) for i in part]
        idx = [i for (i, P) in zip(part, Ps) if !isnothing(P) && !cut(i)]
        for i in part
            i in idx || (done(i, nothing); ans += 1)
        end
        tp += time() - t
        isempty(idx) && continue
        t = time()
        x = launch(T, [P for (i, P) in zip(part, Ps) if i in idx], maxks[idx], STREAMS[p], p, np; stkw, threads, maxresident)
        tup += time() - t; NB += x.NB
        push!(launched, (x, idx))
    end

    # wait for the parts, and stop the graphs that are cut (see timed_cut)
    while any(x -> !x.collected[], first.(launched))
        for (x, idx) in launched
            x.collected[] && continue
            if CUDA.isdone(x.e1)
                CUDA.synchronize(x.e1); collect!(x, idx)
            else
                for j in eachindex(idx)
                    if !x.out[j] && cut(idx[j])
                        CUDA.memset(x.gabp + 4(j - 1), UInt32(1), 1; stream = ABORTSTREAM[])
                        x.out[j] = true
                    end
                end
            end
        end
        all(x -> x.collected[], first.(launched)) || Libc.systemsleep(50e-6)
    end

    return res, info(all(isnothing, res))
end

const STREAMS = CuStream[]

function cpu_sepdp(weights, graph, sep, maxk; nt = 1)
    perm, invp, mink, status = CliqueTrees.sepdp(weights, graph, sep, maxk, Inf, nt)
    return collect(perm), collect(invp), mink, status == CliqueTrees.BT_FOUND ? :found : :above
end

end
