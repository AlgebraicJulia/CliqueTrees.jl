# Sorting of candidate sets (bitonic sort of an index array, keys compared with bs_less) and
# the pruned join: I-blocks in tiles of one warp, O-blocks in tiles of TO, each tile with a
# summary; pairs are tested only where the tile bounds allow them.

# ------------------------------------------------------------------ bitonic sort of indices
@inline function idx_less(keys, a::Int32, b::Int32)
    a == Int32(0) && return false
    b == Int32(0) && return true
    @inbounds return bs_less(keys[a], keys[b])
end
function k_bitonic!(idx, keys, j::Int32, k::Int32, N::Int32)
    t = gtid() - 1
    while t < N
        l = t ⊻ j
        if l > t
            @inbounds a = idx[t + 1]; @inbounds b = idx[l + 1]
            up = (t & k) == 0
            if up ? idx_less(keys, b, a) : idx_less(keys, a, b)
                @inbounds idx[t + 1] = b; @inbounds idx[l + 1] = a
            end
        end
        t += gstride()
    end
    return
end
function k_iota!(idx, m, N)
    t = gtid()
    while t <= N
        @inbounds idx[t] = t <= m ? Int32(t) : Int32(0)
        t += gstride()
    end
    return
end
# The permutation that sorts keys[1:m].
function sortidx(keys::CuVector, m::Int)
    N = nextpow(2, max(m, 2))
    idx = CuVector{Int32}(undef, N)
    launch!(k_iota!, N, idx, m, N)
    k = 2
    while k <= N
        j = k >> 1
        while j >= 1
            launch!(k_bitonic!, N, idx, keys, Int32(j), Int32(k), Int32(N))
            j >>= 1
        end
        k <<= 1
    end
    return view(idx, 1:m)
end
function k_permute!(dst, src, idx, m)
    t = gtid()
    while t <= m
        @inbounds dst[t] = src[idx[t]]
        t += gstride()
    end
    return
end
function permute(src::CuVector{T}, idx, m::Int) where {T}
    dst = CuVector{T}(undef, max(m, 1))
    launch!(k_permute!, m, dst, src, idx, m)
    return dst
end

# ------------------------------------------------------------------ tile summaries
# I-tiles of TI (the warp size): Cand = ∩ C, NCand = ∩ N(C), over the blocks of the tile.
function k_isum!(Cand, NCand, IC, IN, tlo, thi, n, TI)
    t = gtid() + tlo - 1
    while t <= thi
        lo = (t - 1) * TI + 1; hi = min(t * TI, n)
        @inbounds c = IC[lo]; @inbounds nc = IN[lo]
        for i in (lo + 1):hi
            @inbounds c = bs_and(c, IC[i]); @inbounds nc = bs_and(nc, IN[i])
        end
        @inbounds Cand[t] = c; @inbounds NCand[t] = nc
        t += gstride()
    end
    return
end
# O-tiles of TO: Aor, Aand, NAor and the least w(N(A)) over the tile.
function k_osum!(Aor, Aand, NAor, Wmin, OA, ON, Ow, tlo, thi, n, TO)
    t = gtid() + tlo - 1
    while t <= thi
        lo = (t - 1) * TO + 1; hi = min(t * TO, n)
        @inbounds ao = OA[lo]; ad = ao; @inbounds no = ON[lo]; @inbounds wm = Ow[lo]
        for i in (lo + 1):hi
            @inbounds a = OA[i]
            ao = bs_or(ao, a); ad = bs_and(ad, a)
            @inbounds no = bs_or(no, ON[i]); @inbounds wm = min(wm, Ow[i])
        end
        @inbounds Aor[t] = ao; @inbounds Aand[t] = ad; @inbounds NAor[t] = no; @inbounds Wmin[t] = wm
        t += gstride()
    end
    return
end

# ------------------------------------------------------------------ the pruned join
# One warp per I-tile (lane = I-block), blockIdx().y splits the O-tiles. A pair (C, A) passes iff
# C ⊆ A and w(N(A)) + w(N(C) ∩ A) <= K; since C ⊆ A, N(C) ∩ A = N(C) \ N(A), so over an O-tile
#   w(N(C) ∩ A) >= max(w(N(C) \ NAor), w(N(C) ∩ Aand)),
# and the same with the I-tile intersections for a whole tile pair. Pruned pairs fail with a
# value of at least Wmin + bound, which feeds the next width.
function k_join2!(out, ctr, cap, IC, IN, ilo, ihi, ItC, ItN, OA, ON, Ow, olo, ohi, OtAor, OtAand, OtNAor, OtW,
        masks, wcls, ncls, K, tol, rej, TO)
    ws = Int(warpsize())
    lane = (Int(threadIdx().x) - 1) % ws + 1
    wib = (Int(threadIdx().x) - 1) ÷ ws
    nwb = Int(blockDim().x) ÷ ws
    tilo = (ilo - 1) ÷ ws + 1; tihi = (ihi - 1) ÷ ws + 1
    tolo = (olo - 1) ÷ TO + 1; tohi = (ohi - 1) ÷ TO + 1
    nto = tohi - tolo + 1
    chunk = cld(nto, Int(gridDim().y))
    jlo = tolo + (Int(blockIdx().y) - 1) * chunk; jhi = min(tohi, jlo + chunk - 1)
    best = Inf
    it = tilo + (Int(blockIdx().x) - 1) * nwb + wib
    while it <= tihi
        i = (it - 1) * ws + lane
        valid = ilo <= i <= ihi
        ii = valid ? i : max(ilo, (it - 1) * ws + 1)
        @inbounds C = IC[ii]; @inbounds NC = IN[ii]
        @inbounds tC = ItC[it]; @inbounds tN = ItN[it]
        for j in jlo:jhi
            @inbounds Aor = OtAor[j]
            bs_subset(tC, Aor) || continue
            @inbounds NAor = OtNAor[j]; @inbounds Aand = OtAand[j]; @inbounds wm = OtW[j]
            lbt = max(bs_weight(bs_andnot(tN, NAor), masks, wcls, ncls), bs_weight(bs_and(tN, Aand), masks, wcls, ncls))
            if !okw(wm + lbt, K, tol)
                best = min(best, wm + lbt); continue
            end
            (valid && bs_subset(C, Aor)) || continue
            lb = max(bs_weight(bs_andnot(NC, NAor), masks, wcls, ncls), bs_weight(bs_and(NC, Aand), masks, wcls, ncls))
            if !okw(wm + lb, K, tol)
                best = min(best, wm + lb); continue
            end
            blo = max(olo, (j - 1) * TO + 1); bhi = min(ohi, j * TO)
            for b in blo:bhi
                @inbounds A = OA[b]
                bs_subset(C, A) || continue
                @inbounds x = Ow[b] + bs_weight(bs_and(NC, A), masks, wcls, ncls)
                if okw(x, K, tol)
                    @inbounds push!_(out, ctr, cap, bs_or(ON[b], NC))
                else
                    best = min(best, x)
                end
            end
        end
        it += Int(gridDim().x) * nwb
    end
    best < Inf && rejmin!(rej, best)
    return
end

# The pruned join with the tile tests spread over the lanes: lane l tests O-tile base + l against
# the warp's I-tile summary (coalesced loads), a ballot gives the surviving O-tiles, and for each of
# them the lanes (now I-blocks) test their own block against the tile, then the pairs. A warp
# stops when the host sets abort[1] (read every 64 steps, the same answer for the whole warp).
function k_join3!(out, ctr, cap, IC, IN, ilo, ihi, ItC, ItN, OA, ON, Ow, olo, ohi, OtAor, OtAand, OtNAor, OtW,
        masks, wcls, ncls, K, tol, rej, TO, abort)
    ws = Int(warpsize())
    lane = (Int(threadIdx().x) - 1) % ws + 1
    wib = (Int(threadIdx().x) - 1) ÷ ws
    nwb = Int(blockDim().x) ÷ ws
    tilo = (ilo - 1) ÷ ws + 1; tihi = (ihi - 1) ÷ ws + 1
    tolo = (olo - 1) ÷ TO + 1; tohi = (ohi - 1) ÷ TO + 1
    nto = tohi - tolo + 1
    chunk = cld(cld(nto, Int(gridDim().y)), ws) * ws
    jlo = tolo + (Int(blockIdx().y) - 1) * chunk; jhi = min(tohi, jlo + chunk - 1)
    best = Inf; stopped = false; step = 0
    it = tilo + (Int(blockIdx().x) - 1) * nwb + wib
    while it <= tihi && !stopped
        i = (it - 1) * ws + lane
        valid = ilo <= i <= ihi
        ii = valid ? i : max(ilo, (it - 1) * ws + 1)
        @inbounds C = IC[ii]; @inbounds NC = IN[ii]
        @inbounds tC = ItC[it]; @inbounds tN = ItN[it]
        base = jlo
        while base <= jhi
            step += 1
            if (step & 63) == 1
                ab = CUDA.atomic_add!(pointer(abort), UInt32(0)) != UInt32(0)
                CUDA.vote_any_sync(CUDA.FULL_MASK, ab) && (stopped = true; break)
            end
            # lane-parallel tile tests
            j = base + lane - 1
            pass = false
            if j <= jhi
                @inbounds Aor = OtAor[j]
                if bs_subset(tC, Aor)
                    @inbounds NAor = OtNAor[j]; @inbounds Aand = OtAand[j]; @inbounds wm = OtW[j]
                    lbt = max(bs_weight(bs_andnot(tN, NAor), masks, wcls, ncls), bs_weight(bs_and(tN, Aand), masks, wcls, ncls))
                    if okw(wm + lbt, K, tol)
                        pass = true
                    else
                        best = min(best, wm + lbt)
                    end
                end
            end
            mask = CUDA.vote_ballot_sync(CUDA.FULL_MASK, pass)
            while mask != zero(mask)
                q = trailing_zeros(mask); mask &= mask - one(mask)
                j2 = base + q
                @inbounds Aor = OtAor[j2]; @inbounds NAor = OtNAor[j2]; @inbounds Aand = OtAand[j2]; @inbounds wm = OtW[j2]
                lp = false
                if valid && bs_subset(C, Aor)
                    lb = max(bs_weight(bs_andnot(NC, NAor), masks, wcls, ncls), bs_weight(bs_and(NC, Aand), masks, wcls, ncls))
                    if okw(wm + lb, K, tol)
                        lp = true
                    else
                        best = min(best, wm + lb)
                    end
                end
                if lp
                    blo = max(olo, (j2 - 1) * TO + 1); bhi = min(ohi, j2 * TO)
                    for b in blo:bhi
                        @inbounds A = OA[b]
                        bs_subset(C, A) || continue
                        @inbounds x = Ow[b] + bs_weight(bs_and(NC, A), masks, wcls, ncls)
                        if okw(x, K, tol)
                            @inbounds push!_(out, ctr, cap, bs_or(ON[b], NC))
                        else
                            best = min(best, x)
                        end
                    end
                end
            end
            base += ws
        end
        it += Int(gridDim().x) * nwb
    end
    best < Inf && rejmin!(rej, best)
    return
end
