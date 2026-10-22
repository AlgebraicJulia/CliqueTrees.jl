# GPU minimum fill-in: gpupid's level-synchronous PID-BT (hash sets, append buffers, warp-tiled
# join, occupancy-sized grid-stride kernels; included read-only from tn-work/gpupid/src) with the
# fill objective of fillcore.jl. Same rules, rounds and phases as the CPU reference fill_decide,
# so the per-round counts (t, |I|, |O|, |P|, pending) must match it exactly.
const GPUPID_SRC = joinpath(@__DIR__, "..", "..", "..", "gpupid", "src")
include(joinpath(GPUPID_SRC, "gpupid.jl"))     # pidref, devfuns, sortjoin, DSet, DBuf, DVec, launch!
include("fillcore.jl")

@inline function rejmini!(rej, x::Int64)
    CUDA.atomic_min!(pointer(rej), x); return
end
# index of key in the set, 0 if absent
@inline function k_hidx(tab, keys, key)
    mask = UInt64(length(tab) - 1)
    h = bs_hash(key) & mask
    while true
        @inbounds s = tab[Int(h) + 1]
        s == Int32(0) && return Int32(0)
        s > 0 && (@inbounds keys[s]) === key && return s
        h = (h + one(UInt64)) & mask
    end
end

function k_fr0!(out, ctr, cap, adj, n, K, rej)
    v = gtid()
    while v <= n
        @inbounds X = bs_or(adj[v], bs_bit(bsval(adj[v]), v))
        f = bs_fill(adj, X)
        f <= K ? push!_(out, ctr, cap, X) : rejmini!(rej, f)
        v += gstride()
    end
    return
end

function k_fclassify!(oA, oN, oO, octr, ocap, pout, pctr, pcap, big, bctr, X, m, adj, all, K, rej, scr, pmc_only, look, Icost, t, useolb)
    i = gtid(); st = gstride(); sbase = gtid()
    while i <= m
        @inbounds x = X[i]
        kind, A = dv_classify(adj, all, x, scr, sbase, st, pmc_only)
        if kind == Int32(1)
            f, f0 = useolb ? olb(adj, all, x, A, look, Icost, t) : (ofill(adj, x, A), ofill(adj, x, A))
            if f <= K
                j = push!_(oA, octr, ocap, A)
                if j <= ocap
                    base = ofill(adj, x, A)
                    @inbounds oN[j] = x; @inbounds oO[j] = ((f - base) << 32) | (f0 - base)
                end
            else
                rejmini!(rej, f0)
            end
        elseif kind == Int32(2)
            push!_(pout, pctr, pcap, x)
        elseif kind == Int32(3)
            push!_(big, bctr, length(big), x)
        end
        i += st
    end
    return
end

function k_fr1!(oA, oN, oO, octr, ocap, IN, lo, hi, adj, all, K, rej, look, Icost, t, useolb)
    i = gtid() + lo - 1
    while i <= hi
        @inbounds S = IN[i]
        A = dv_full(adj, all, S)
        if !bs_isempty(A)
            f, f0 = useolb ? olb(adj, all, S, A, look, Icost, t) : (ofill(adj, S, A), ofill(adj, S, A))
            if f <= K
                j = push!_(oA, octr, ocap, A)
                if j <= ocap
                    base = ofill(adj, S, A)
                    @inbounds oN[j] = S; @inbounds oO[j] = ((f - base) << 32) | (f0 - base)
                end
            else
                rejmini!(rej, f0)
            end
        end
        i += gstride()
    end
    return
end

# O-tile summary: union of the A's of TO consecutive O-blocks.
function k_osumf!(Aor, OA, tlo, thi, n, TO)
    t = gtid() + tlo - 1
    while t <= thi
        lo = (t - 1) * TO + 1; hi = min(t * TO, n)
        @inbounds ao = OA[lo]
        for i in (lo + 1):hi
            @inbounds ao = bs_or(ao, OA[i])
        end
        @inbounds Aor[t] = ao
        t += gstride()
    end
    return
end

# R2 join with warp tiles (as gpupid's k_join3!): lane l tests O-tile base + l against the warp's
# I-tile intersection ∩C ⊆ ∪A, a ballot gives the survivors, then lanes (I-blocks) test their own
# C ⊆ ∪A and the pairs C ⊆ A; a pair yields X = N(A) ∪ N(C), kept iff fill(X) <= K.
function k_fjoin!(out, ctr, cap, IC, IN, ilo, ihi, ItC, OA, ON, olo, ohi, OtAor, adj, K, rej, TO, Icost, Oo, jlb)
    ws = Int(warpsize())
    lane = (Int(threadIdx().x) - 1) % ws + 1
    wib = (Int(threadIdx().x) - 1) ÷ ws
    nwb = Int(blockDim().x) ÷ ws
    tilo = (ilo - 1) ÷ ws + 1; tihi = (ihi - 1) ÷ ws + 1
    tolo = (olo - 1) ÷ TO + 1; tohi = (ohi - 1) ÷ TO + 1
    nto = tohi - tolo + 1
    chunk = cld(cld(nto, Int(gridDim().y)), ws) * ws
    jlo = tolo + (Int(blockIdx().y) - 1) * chunk; jhi = min(tohi, jlo + chunk - 1)
    best = typemax(Int64)
    it = tilo + (Int(blockIdx().x) - 1) * nwb + wib
    while it <= tihi
        i = (it - 1) * ws + lane
        valid = ilo <= i <= ihi
        ii = valid ? i : max(ilo, (it - 1) * ws + 1)
        @inbounds C = IC[ii]; @inbounds NC = IN[ii]; @inbounds cC = Icost[ii]
        @inbounds tC = ItC[it]
        base = jlo
        while base <= jhi
            j = base + lane - 1
            pass = j <= jhi && bs_subset(tC, @inbounds OtAor[j])
            mask = CUDA.vote_ballot_sync(CUDA.FULL_MASK, pass)
            while mask != zero(mask)
                q = trailing_zeros(mask); mask &= mask - one(mask)
                j2 = base + q
                if valid && bs_subset(C, @inbounds OtAor[j2])
                    blo = max(olo, (j2 - 1) * TO + 1); bhi = min(ohi, j2 * TO)
                    for b in blo:bhi
                        @inbounds A = OA[b]
                        bs_subset(C, A) || continue
                        @inbounds X = bs_or(ON[b], NC)
                        f = bs_fill(adj, X); f0 = f
                        if jlb   # C and the components outside A ∪ N(A) stay components of G - Ω
                            @inbounds o = Oo[b]
                            f0 = f + cC + (o & 0xffffffff); f = f + cC + (o >> 32)
                        end
                        if f <= K
                            push!_(out, ctr, cap, X)
                        else
                            best = min(best, f0)
                        end
                    end
                end
            end
            base += ws
        end
        it += Int(gridDim().x) * nwb
    end
    best < typemax(Int64) && rejmini!(rej, best)
    return
end

function k_fr3!(out, ctr, cap, OA, ON, lo, hi, off, total, adj, K, rej)
    t = gtid()
    while t <= total
        l = lo; h = hi
        while l < h
            mid = (l + h + 1) >> 1
            @inbounds off[mid - lo + 1] < t ? (l = mid) : (h = mid - 1)
        end
        b = l
        @inbounds S = ON[b]; @inbounds A = OA[b]
        r = t - (@inbounds off[b - lo + 1])
        v = 0; cnt = 0
        for i in 1:length(S)
            c = count_ones(S[i])
            if cnt + c >= r
                word = S[i]
                for _ in 1:(r - cnt - 1)
                    word &= word - one(UInt64)
                end
                v = 64 * (i - 1) + trailing_zeros(word) + 1
                break
            end
            cnt += c
        end
        X = bs_or(S, bs_and(@inbounds(adj[v]), A))
        f = bs_fill(adj, X)
        f <= K ? push!_(out, ctr, cap, X) : rejmini!(rej, f)
        t += gstride()
    end
    return
end

# R4 with costs. Pc[p] = -1 until the support of p is admitted; then c (target cost) or the root total.
# Emits I-candidates (crib, outlet, p, c) for c <= t; keeps the rest pending; drops totals > K.
function k_fr4!(icC, icN, icP, icc, ictr, icap, still, sctr, rootpk, late, pend, np, P, blk, Pc, adj, all, Itab, Ikeys, Icost, K, t, rej)
    i = gtid()
    while i <= np
        @inbounds p = pend[i]
        @inbounds Ω = P[p]
        @inbounds c = Pc[p]
        keep = true
        if c < 0
            @inbounds bk = blk[p]
            if bs_isempty(bk) || k_hidx(Itab, Ikeys, bk) > Int32(0)
                outlet, has = dv_outlet(adj, all, Ω)
                rest = bs_andnot(all, Ω); tot = Int64(0); miss = false
                while !bs_isempty(rest)
                    v = bs_min(rest)
                    D = bs_comp(adj, rest, v)
                    rest = bs_andnot(rest, D)
                    N = bs_open(adj, D)
                    if !has || !bs_subset(N, outlet)
                        j = k_hidx(Itab, Ikeys, D)
                        if j == Int32(0)
                            miss = true; @inbounds blk[p] = D; break
                        end
                        @inbounds tot += Icost[j]
                    end
                end
                if !miss
                    tot += bs_fill(adj, Ω)
                    if tot > K
                        rejmini!(rej, tot); keep = false
                    else
                        c = has ? tot - bs_fill(adj, outlet) : tot
                        @inbounds Pc[p] = c
                    end
                end
            end
        end
        if keep && c >= 0 && c <= t
            keep = false
            outlet, has = dv_outlet(adj, all, Ω)
            if !has
                CUDA.atomic_min!(pointer(rootpk), (UInt64(c) << 32) | UInt64(p))
            else
                crib = bs_andnot(Ω, outlet); rest = bs_andnot(all, Ω)
                while !bs_isempty(rest)
                    v = bs_min(rest)
                    D = bs_comp(adj, rest, v)
                    rest = bs_andnot(rest, D)
                    bs_subset(bs_open(adj, D), outlet) || (crib = bs_or(crib, D))
                end
                if k_hidx(Itab, Ikeys, crib) == Int32(0)
                    j = push!_(icC, ictr, icap, crib)
                    if j <= icap
                        @inbounds icN[j] = outlet; @inbounds icP[j] = Int32(p); @inbounds icc[j] = c
                    end
                end
            end
        end
        keep && push!_(still, sctr, length(still), p)
        i += gstride()
    end
    return
end

# After inserting the candidates: the least (c, p) per new I-block (index > oldn).
function k_icost!(pk, cand, cc, cp, m, tab, keys, oldn)
    i = gtid()
    while i <= m
        @inbounds j = k_hidx(tab, keys, cand[i])
        if j > oldn
            @inbounds CUDA.atomic_min!(pointer(pk, Int(j - oldn)), (UInt64(cc[i]) << 32) | UInt64(cp[i]))
        end
        i += gstride()
    end
    return
end
function k_icostset!(Icost, Ibp, pk, oldn, nnew, t, late)
    i = gtid()
    while i <= nnew
        @inbounds x = pk[i]
        c = Int64(x >> 32)
        @inbounds Icost[oldn + i] = c; @inbounds Ibp[oldn + i] = Int32(x & 0xffffffff)
        c < t && CUDA.atomic_add!(pointer(late), Int64(1))
        i += gstride()
    end
    return
end
function k_minpc!(out, pend, np, Pc)
    i = gtid()
    while i <= np
        @inbounds c = Pc[pend[i]]
        c >= 0 && CUDA.atomic_min!(pointer(out), c)
        i += gstride()
    end
    return
end

# I-block lookup for olb on the device
struct DLook{T, K}
    tab::T
    keys::K
end
@inline (l::DLook)(D) = k_hidx(l.tab, l.keys, D)

struct FGPU{W}
    pg::PGraph{W}
    adj::CuVector{BS{W}}
end
FGPU(pg::PGraph{W}) where {W} = FGPU{W}(pg, CuVector(pg.adj))

function gpu_fill_decide(gg::FGPU{W}, K::Int; deadline = Inf, verbose = false, TO = 8) where {W}
    pg = gg.pg; n = pg.n; adj = gg.adj; all = pg.all; K = Int64(K)
    rej = CuVector{Int64}([typemax(Int64)])
    Iset = DSet{W}(); IN = DVec{BS{W}}(); Ibp = DVec{Int32}(); Icost = DVec{Int64}()
    Oset = DSet{W}(); ON = DVec{BS{W}}()
    Pset = DSet{W}(); blk = DVec{BS{W}}(); Pc = DVec{Int64}()
    X2 = DSet{W}(); X3 = DSet{W}()
    bufX = DBuf{BS{W}}(); bufOA = DBuf{BS{W}}(); bufON = CuVector{BS{W}}(undef, INITCAP[]); bufOo = CuVector{Int64}(undef, INITCAP[]); Oo = DVec{Int64}()
    bufP = DBuf{BS{W}}(); bufBig = DBuf{BS{W}}(64)
    bufIC = DBuf{BS{W}}(); bufIN = CuVector{BS{W}}(undef, INITCAP[]); bufIP = CuVector{Int32}(undef, INITCAP[]); bufIc = CuVector{Int64}(undef, INITCAP[])
    pend = DVec{Int32}(); still = DBuf{Int32}()
    rootpk = CuVector{UInt64}([typemax(UInt64)]); late = CUDA.zeros(Int64, 1)
    ws = CUDA.warpsize(CUDA.device())
    ItC = DVec{BS{W}}(); ItN = DVec{BS{W}}(); OtAor = DVec{BS{W}}()
    function isum!(oldn)
        nt = cld(Iset.n, ws); nt == 0 && return
        ensure!(ItC, nt); ensure!(ItN, nt); tlo = oldn ÷ ws + 1
        launch!(k_isum!, nt - tlo + 1, ItC.v, ItN.v, Iset.keys, IN.v, tlo, nt, Iset.n, ws)
        ItC.n = ItN.n = nt
    end
    function osum!(oldn)
        nt = cld(Oset.n, TO); nt == 0 && return
        ensure!(OtAor, nt); tlo = oldn ÷ TO + 1
        launch!(k_osumf!, nt - tlo + 1, OtAor.v, Oset.keys, tlo, nt, Oset.n, TO)
        OtAor.n = nt
    end
    st = Dict{Symbol, Any}(:rounds => 0, :phases => 0, :pairs => 0, :xcand2 => 0, :xcand3 => 0, :pmc_tests => 0, :t => Dict{Symbol, Float64}(), :log => Tuple{Int, Int, Int, Int, Int}[])
    tt = st[:t]
    tick(f, key) = (t0 = time(); r = f(); CUDA.synchronize(); tt[key] = get(tt, key, 0.0) + time() - t0; r)
    scr = CuVector{UInt64}(undef, 1)
    maxbuf = haskey(ENV, "GPUPID_MAXBUF") ? parse(Int, ENV["GPUPID_MAXBUF"]) : max(1 << 16, Int(floor(0.06 * CUDA.total_memory() / sizeof(BS{W}))))

    function classify!(Xkeys, lo, m, pmc_only)
        m == 0 && return
        Xv = view(Xkeys, lo:(lo + m - 1))
        while true
            oc = bcount(bufOA); pc = bcount(bufP)
            k = @cuda launch = false k_fclassify!(bufOA.data, bufON, bufOo, bufOA.ctr, length(bufOA.data), bufP.data, bufP.ctr, length(bufP.data),
                bufBig.data, bufBig.ctr, Xv, m, adj, all, K, rej, scr, pmc_only, DLook(Iset.tab, Iset.keys), Icost.v, t, OLB[])
            cfg = launch_configuration(k.fun)
            blocks = min(cld(m, cfg.threads), cfg.blocks * 4); T = blocks * cfg.threads
            length(scr) < 256 * T && (scr = CuVector{UInt64}(undef, 256 * T))
            k(bufOA.data, bufON, bufOo, bufOA.ctr, length(bufOA.data), bufP.data, bufP.ctr, length(bufP.data),
                bufBig.data, bufBig.ctr, Xv, m, adj, all, K, rej, scr, pmc_only, DLook(Iset.tab, Iset.keys), Icost.v, t, OLB[]; threads = cfg.threads, blocks)
            oc2 = bcount(bufOA); pc2 = bcount(bufP)
            if oc2 <= length(bufOA.data) && pc2 <= length(bufP.data)
                bcount(bufBig) > 0 && error("bag of more than 128 vertices: not supported")
                st[:pmc_tests] += m
                return
            end
            growto = nextpow(2, 2 * max(oc2, pc2, 1024))
            nOA = CuVector{BS{W}}(undef, max(growto, length(bufOA.data))); copyto!(nOA, 1, bufOA.data, 1, oc); bufOA.data = nOA
            nON = CuVector{BS{W}}(undef, length(nOA)); copyto!(nON, 1, bufON, 1, oc); bufON = nON
            nOo = CuVector{Int64}(undef, length(nOA)); copyto!(nOo, 1, bufOo, 1, oc); bufOo = nOo
            nP = CuVector{BS{W}}(undef, max(growto, length(bufP.data))); copyto!(nP, 1, bufP.data, 1, pc); bufP.data = nP
            copyto!(bufOA.ctr, Int64[oc]); copyto!(bufP.ctr, Int64[pc])
        end
    end
    function flushP!()
        m = bcount(bufP); m == 0 && return 0
        pos, nnew = dinsert!(Pset, bufP.data, m)
        ensure!(blk, Pset.n); fill!(view(blk.v, (blk.n + 1):Pset.n), bs_zero(Val(W))); blk.n = Pset.n
        ensure!(Pc, Pset.n); fill!(view(Pc.v, (Pc.n + 1):Pset.n), Int64(-1)); Pc.n = Pset.n
        ensure!(pend, pend.n + nnew)
        c = CuVector{Int32}([pend.n]); launch!(k_newlist!, m, view(pend.v, 1:length(pend.v)), pos, m, c)
        pend.n += nnew; reset!(bufP)
        return nnew
    end
    function flushO!()
        m = bcount(bufOA); m == 0 && return 0
        idx = sortidx(bufON, m)
        cA = permute(bufOA.data, idx, m); cN = permute(bufON, idx, m); cO = permute(bufOo, idx, m)
        oldn = Oset.n
        pos, nnew = dinsert!(Oset, cA, m)
        ensure!(ON, Oset.n)
        ensure!(Oo, Oset.n)
        launch!(k_gather!, m, ON.v, cN, pos, m); launch!(k_gather!, m, Oo.v, cO, pos, m)
        ON.n = Oo.n = Oset.n
        osum!(oldn)
        reset!(bufOA)
        return nnew
    end
    function dedupX!(S::DSet, cnt)
        lo = S.n + 1
        dinsert!(S, bufX.data, cnt)
        return lo, S.n - lo + 1
    end

    c = fill_buf!(bufX) do d, ctr, cap
        launch!(k_fr0!, n, d, ctr, cap, adj, n, K, rej)
    end
    lo3, m3 = dedupX!(X3, c); classify!(X3.keys, lo3, m3, true); flushP!()

    nIold = 0; nOold = 0; rootv = typemax(UInt64); t = Int64(0)
    while true
        time() > deadline && (st[:timeout] = true; break)
        npend = 0
        tick(:r4) do
            np = pend.n
            if np > 0
                still.data = length(still.data) < np ? CuVector{Int32}(undef, np) : still.data
                c = fill_buf!(bufIC) do d, ctr, cap
                    if length(bufIN) < length(d)
                        bufIN = CuVector{BS{W}}(undef, length(d)); bufIP = CuVector{Int32}(undef, length(d)); bufIc = CuVector{Int64}(undef, length(d))
                    end
                    reset!(still)
                    launch!(k_fr4!, np, d, bufIN, bufIP, bufIc, ctr, cap, still.data, still.ctr, rootpk, late, pend.v, np, Pset.keys, blk.v, Pc.v,
                        adj, all, Iset.tab, Iset.keys, Icost.v, K, t, rej)
                end
                ns = bcount(still)
                copyto!(pend.v, 1, still.data, 1, ns); pend.n = ns
                rootv = Array(rootpk)[1]
                if rootv == typemax(UInt64) && c > 0
                    idx = sortidx(bufIC.data, c)
                    cC = permute(bufIC.data, idx, c); cN = permute(bufIN, idx, c); cP = permute(bufIP, idx, c); cc = permute(bufIc, idx, c)
                    oldn = Iset.n
                    pos, nnew = dinsert!(Iset, cC, c)
                    ensure!(IN, Iset.n); ensure!(Ibp, Iset.n); ensure!(Icost, Iset.n)
                    launch!(k_gather!, c, IN.v, cN, pos, c)
                    pk = CUDA.fill(typemax(UInt64), max(nnew, 1))
                    launch!(k_icost!, c, pk, cC, cc, cP, c, Iset.tab, Iset.keys, Int32(oldn))
                    launch!(k_icostset!, nnew, Icost.v, Ibp.v, pk, oldn, nnew, t, late)
                    IN.n = Ibp.n = Icost.n = Iset.n
                    isum!(oldn)
                end
            end
            npend = pend.n
        end
        rootv != typemax(UInt64) && break
        dIlo = nIold + 1; dIhi = Iset.n; dOlo = nOold + 1; dOhi = Oset.n
        if dIhi < dIlo && dOhi < dOlo
            m = CuVector{Int64}([typemax(Int64)])
            launch!(k_minpc!, pend.n, m, pend.v, pend.n, Pc.v)
            tn = Array(m)[1]
            tn == typemax(Int64) && break
            t = tn; st[:phases] += 1
            continue
        end
        st[:rounds] += 1
        tick(:r1) do
            if dIhi >= dIlo
                fill_buf!(bufOA) do d, ctr, cap
                    length(bufON) < length(d) && (bufON = CuVector{BS{W}}(undef, length(d)); bufOo = CuVector{Int64}(undef, length(d)))
                    launch!(k_fr1!, dIhi - dIlo + 1, d, bufON, bufOo, ctr, cap, IN.v, dIlo, dIhi, adj, all, K, rej, DLook(Iset.tab, Iset.keys), Icost.v, t, OLB[])
                end
            end
        end
        clear!(X2); clear!(X3)
        stack = [(ilo, ihi, olo, ohi) for (ilo, ihi, olo, ohi) in ((dIlo, dIhi, 1, dOhi), (1, nIold, dOlo, dOhi)) if ihi >= ilo && ohi >= olo]
        timedout = false
        while !isempty(stack)
            (ilo, ihi, olo, ohi) = pop!(stack)
            cX, fits = tick(:join) do
                fill_buf_cap!(bufX, maxbuf) do d, ctr, cap
                    jargs = (d, ctr, cap, Iset.keys, IN.v, ilo, ihi, ItC.v, Oset.keys, ON.v, olo, ohi, OtAor.v, adj, K, rej, TO, Icost.v, Oo.v, JLB[])
                    nti = (ihi - 1) ÷ ws - (ilo - 1) ÷ ws + 1; nto = (ohi - 1) ÷ TO - (olo - 1) ÷ TO + 1
                    launch_join!(k_fjoin!, jargs, ws, nti, cld(nto, ws))
                end
            end
            if !fits
                if ihi - ilo >= ohi - olo && ihi > ilo
                    mid = (ilo + ihi) ÷ 2; push!(stack, (ilo, mid, olo, ohi), (mid + 1, ihi, olo, ohi))
                elseif ohi > olo
                    mid = (olo + ohi) ÷ 2; push!(stack, (ilo, ihi, olo, mid), (ilo, ihi, mid + 1, ohi))
                else
                    error("a single pair overflows the candidate buffer")
                end
                continue
            end
            st[:pairs] += (ihi - ilo + 1) * (ohi - olo + 1); st[:xcand2] += cX
            tick(:classify2) do
                lo2, m2 = dedupX!(X2, cX)
                classify!(X2.keys, lo2, m2, false)
            end
            time() > deadline && (timedout = true; break)
        end
        timedout && (st[:timeout] = true; break)
        tick(:r3) do
            if dOhi >= dOlo
                nb = dOhi - dOlo + 1
                off = CuVector{Int32}(undef, nb)
                launch!(k_popc!, nb, off, ON.v, dOlo, dOhi)
                cnt = Array(off)
                lo = 1
                while lo <= nb
                    hi = lo; tot = Int(cnt[lo])
                    while hi < nb && tot + cnt[hi + 1] <= maxbuf
                        hi += 1; tot += cnt[hi]
                    end
                    c64 = CuVector{Int64}(Int64.(cnt[lo:hi])); offx = cumsum(c64) .- c64
                    blo = dOlo + lo - 1; bhi = dOlo + hi - 1
                    c3 = fill_buf!(bufX) do d, ctr, cap
                        launch!(k_fr3!, tot, d, ctr, cap, Oset.keys, ON.v, blo, bhi, offx, tot, adj, K, rej)
                    end
                    st[:xcand3] += c3
                    lo3, m3 = dedupX!(X3, c3)
                    classify!(X3.keys, lo3, m3, true)
                    lo = hi + 1
                end
            end
        end
        nIold = dIhi; nOold = dOhi
        tick(:flush) do
            flushO!(); flushP!()
        end
        push!(st[:log], (Int(t), Iset.n, Oset.n, Pset.n, npend))
        verbose && println("round $(st[:rounds]) t=$t: |I|=$(Iset.n) |O|=$(Oset.n) |P|=$(Pset.n) pending=$npend pairs=$(st[:pairs])")
    end
    st[:I] = Iset.n; st[:O] = Oset.n; st[:P] = Pset.n; st[:late] = Int(Array(late)[1])
    nextK = max(K + 1, Array(rej)[1])
    if rootv != typemax(UInt64)
        Ih = Array(view(Iset.keys, 1:Iset.n)); Ibh = Int.(Array(view(Ibp.v, 1:Iset.n))); Ph = Array(view(Pset.keys, 1:Pset.n))
        return (yes = true, opt = Int(rootv >> 32), root = Int(rootv & 0xffffffff), nextK = 0, I = Ih, Ibp = Ibh, P = Ph, st)
    end
    return (yes = false, opt = -1, root = 0, nextK = Int(nextK), I = BS{W}[], Ibp = Int[], P = BS{W}[], st)
end

# decide interface for solve_atom (same as cpu_decide)
function gpu_decide_fill(pg::PGraph{W}, K::Int; deadline = Inf, verbose = false) where {W}
    r = gpu_fill_decide(FGPU(pg), K; deadline, verbose)
    to = get(r.st, :timeout, false)
    stats = Dict(k => v for (k, v) in r.st if k != :log)
    if r.yes
        Iset = Dict(C => i for (i, C) in enumerate(r.I))
        f = bags_fill(pg, fill_bags(pg, r.P, r.root, Iset, r.Ibp))
        length(f) == r.opt || error("reconstruction: $(length(f)) fill edges, cost $(r.opt)")
        return (yes = true, opt = r.opt, nextK = 0, fill = f, stats, timeout = false, log = r.st[:log])
    end
    return (yes = false, opt = -1, nextK = r.nextK, fill = Tuple{Int, Int}[], stats, timeout = to, log = r.st[:log])
end
