# GSmall.jl: GPU rounds for small graphs (SMALL_NOTES.md). Included into module GRounds.
#
# The replayed CUDA graph of GRounds.jl launches ~10 grid-wide kernels per round,
# each sized to the whole device, so below ~100k vertices a round costs its
# launch and synchronisation latency (B200: 70-100 µs per round). Here the whole
# round loop runs inside ONE kernel launch:
#   :block  one thread block per graph (BlockCx): the phases of a round are
#           separated by sync_threads, the control block lives in shared memory;
#   :grid   one cooperative grid of G resident blocks per graph (GridCx): the
#           phases are separated by a software grid barrier;
#   batch   many graphs in one launch, one block per graph (grounds_batch).
# The round code is the code of GRounds.jl (same functions, run under another
# execution context), except that the twin detection and the absorption of
# small cliques run one warp per selected vertex (detect_warp!, apply_warp!:
# same results as the block versions, without serialising the selected
# vertices of a round on one block). The state evolution, so the fill, is the
# same as the replayed version for the same configuration.

const DWARP = 64             # cliques up to this size: detection and absorption by one warp
const VPS = 128              # auto mode: the persistent grid up to this many vertices per SM
const NEED_OB = 8            # C_NEED bit set by the persistent kernels: grow the outside-mask arena

# ---------------------------------------------------------------- device side

# volatile load (polling the barrier word)
@inline function vload(p::Core.LLVMPtr{UInt32, 1})
    return Base.llvmcall("""
        %val = load volatile i32, ptr addrspace(1) %0
        ret i32 %val
        """, UInt32, Tuple{Core.LLVMPtr{UInt32, 1}}, p)
end

using CUDA.CUDACore.LLVM.Interop: @asmcall

@inline atom_add_release(p::Core.LLVMPtr{UInt32, 1}, v::UInt32) =
    @asmcall("atom.add.release.gpu.u32 \$0,[\$1],\$2;", "=r,l,r,~{memory}", true, UInt32, Tuple{Core.LLVMPtr{UInt32, 1}, UInt32}, p, v)
@inline ld_acquire(p::Core.LLVMPtr{UInt32, 1}) =
    @asmcall("ld.acquire.gpu.u32 \$0,[\$1];", "=r,l,~{memory}", true, UInt32, Tuple{Core.LLVMPtr{UInt32, 1}}, p)

# barrier between the phases of a round
@inline gbar!(::BlockCx, gb) = sync_threads()
@inline function gbar!(::GridCx, gb)
    if gridDim().x == 1
        sync_threads()
        return nothing
    end
    # all blocks resident (cooperative launch); the top bit of gb[1] flips once
    # every block has arrived (the scheme of CUDA's cooperative groups)
    sync_threads()
    if threadIdx().x == 1
        p = pointer(gb, 1)
        nb = blockIdx().x == 1 ? (0x80000000 - (UInt32(gridDim().x) - 0x00000001)) : 0x00000001
        if CUDA.CUDACore.sm_major() >= UInt32(7)
            # arrive with release, poll with acquire
            old = atom_add_release(p, nb)
            while ((old ⊻ ld_acquire(p)) & 0x80000000) == 0x00000000
            end
        else
            threadfence()
            old = CUDA.atomic_add!(p, nb)
            while ((old ⊻ vload(p)) & 0x80000000) == 0x00000000
            end
            threadfence()
        end
    end
    sync_threads()
    return nothing
end

# twin detection of selected vertex wi (clique of at most DWARP) by one warp: the
# same candidates (smallest earlier equal (hash, count)) and verification as detect_one!
@inline function detect_warp!(par, hsh, cnt, pool, astart, alen, stt, kown, kar, sel, kst, dsel, wi)
    lane = laneid()
    @inbounds v = sel[wi]; @inbounds ko = kst[wi]; @inbounds d = dsel[wi]
    i = lane + 1
    while i <= d
        @inbounds h = hsh[ko + i]; @inbounds c = cnt[ko + i]
        r = Int32(0)
        for j in 1:(i - 1)
            if (@inbounds hsh[ko + j]) == h && (@inbounds cnt[ko + j]) == c
                r = Int32(j); break
            end
        end
        @inbounds par[ko + i] = r
        i += 32
    end
    sync_warp()
    for i in 1:Int(d)
        @inbounds r = par[ko + i]
        if r != 0
            @inbounds x = kar[ko + i]; @inbounds y = kar[ko + r]
            @inbounds sx = astart[x]; @inbounds lx = alen[x]
            @inbounds sy = astart[y]; @inbounds ly = alen[y]
            bad = false
            k = lane
            while k < lx
                @inbounds o = pool[sx + k]
                if (@inbounds kown[o]) != v && (@inbounds stt[o]) == ALIVE && !bsearch(pool, sy, ly, o)
                    bad = true
                end
                k += 32
            end
            anybad = vote_any_sync(FULL, bad)
            lane == 0 && (@inbounds par[ko + i] = anybad ? Int32(0) : y)
        end
    end
    sync_warp()
    return nothing
end

@inline function k_detectw!(cx, par, hsh, cnt, htab, pool, astart, alen, stt, kown, kar, sel, kst, dsel, ctl)
    @inbounds (ctl[C_STATUS] != 0) && return nothing
    @inbounds ns = ctl[C_NS]
    wi = Int64(gwarp(cx))
    while wi <= ns
        (@inbounds dsel[wi]) <= DWARP && detect_warp!(par, hsh, cnt, pool, astart, alen, stt, kown, kar, sel, kst, dsel, wi)
        wi += nwarps(cx)
    end
    bi = Int64(vblock(cx))
    while bi <= ns
        (@inbounds dsel[bi]) > DWARP && detect_one!(par, hsh, cnt, htab, pool, astart, alen, stt, kown, kar, sel, kst, dsel, bi)
        bi += vgrid(cx)
    end
    return nothing
end

# absorption, retirement and re-keying of selected vertex wi by one warp (as apply_one!)
@inline function apply_warp!(Lout, cout, inL, order, parent, par, stt, w, wdeg, def, kar, sel, kst, dsel, ctl, kv, sc, compress,
        tch, nm, cks, wi, T)
    lane = laneid()
    @inbounds v = sel[wi]; @inbounds ko = kst[wi]; @inbounds d = dsel[wi]
    na = Int64(0)
    if compress
        i = lane + 1
        while i <= d
            @inbounds r = par[ko + i]
            if r != 0
                @inbounds x = kar[ko + i]
                @inbounds stt[x] = ABSORBED
                @inbounds kv[x] = Inf32
                @inbounds parent[x] = r
                @inbounds wx = w[x]
                CUDA.atomic_add!(pointer(w, r), wx)
                CUDA.atomic_sub!(pointer(wdeg, r), wx)
                CUDA.atomic_add!(pointer(nm, r), @inbounds nm[x])
                na += 1
            end
            i += 32
        end
    end
    na = warpsum(na)
    sync_warp()
    if lane == 0
        @inbounds stt[v] = DEAD
        @inbounds kv[v] = Inf32
        pos = CUDA.atomic_add!(pointer(ctl, C_NORD), Int64(1)) + 1
        @inbounds order[pos] = v
        q = @inbounds nm[v]; c = @inbounds cks[wi]
        CUDA.atomic_add!(pointer(ctl, C_NNZ), q * c + (q * (q + 1)) >> 1)
        CUDA.atomic_add!(pointer(ctl, C_FLOP), q * c * c + c * q * (q + 1) + (q * (q + 1) * (2q + 1)) ÷ 6)
        CUDA.atomic_max!(pointer(ctl, C_TW), c + q)
        CUDA.atomic_sub!(pointer(ctl, C_NAL), Int64(1) + na)
    end
    i = lane + 1
    while i <= d
        @inbounds x = kar[ko + i]
        @inbounds tch[x] = 0x01
        if (@inbounds stt[x]) == ALIVE
            k = vkey(def, wdeg, w, x, sc)
            @inbounds kv[x] = k
            if k <= T && (@inbounds inL[x]) == UInt32(0)
                if CUDA.atomic_xchg!(pointer(inL, x), UInt32(1)) == UInt32(0)
                    j = CUDA.atomic_add!(pointer(ctl, cout), Int64(1)) + 1
                    @inbounds Lout[j] = x
                end
            end
        end
        i += 32
    end
    return nothing
end

@inline function k_applyw!(cx, Lout, cout, inL, order, parent, par, stt, w, wdeg, def, kar, sel, kst, dsel, ctl, kv, sc, compress, tch, nm, cks)
    @inbounds (ctl[C_STATUS] != 0) && return nothing
    sh_n = CuStaticSharedArray(Int64, 1)
    @inbounds ns = ctl[C_NS]
    @inbounds T = i2f(ctl[C_T])
    wi = Int64(gwarp(cx))
    while wi <= ns
        (@inbounds dsel[wi]) <= DWARP && apply_warp!(Lout, cout, inL, order, parent, par, stt, w, wdeg, def, kar, sel, kst, dsel, ctl,
            kv, sc, compress, tch, nm, cks, wi, T)
        wi += nwarps(cx)
    end
    bi = Int64(vblock(cx))
    while bi <= ns
        (@inbounds dsel[bi]) > DWARP && apply_one!(Lout, cout, inL, order, parent, par, stt, w, wdeg, def, kar, sel, kst, dsel, ctl, kv,
            sc, compress, tch, nm, cks, bi, T, sh_n)
        bi += vgrid(cx)
    end
    return nothing
end

@inline lbufs(dv, cur) = cur == 1 ? (dv.L1, dv.key1, C_CNT1, dv.L2, dv.key2, C_CNT2) : (dv.L2, dv.key2, C_CNT2, dv.L1, dv.key1, C_CNT1)

# the round loop of one graph (dv: its arrays and parameters; ctl, kmin: its control
# block, in shared memory under BlockCx). Runs until the status is not 0 (done, tail,
# capacity) or maxr rounds have been made. The same kernels as capture_rounds, in the
# same order; dense mode (k_ob!, k_whg!) in the rounds that have a clique >= dgroup.
const NPROF = 16
const PROF_NAMES = (:scan, :reb, :cand, :luby2, :sizes, :elim12, :dense, :elim34, :detect, :apply, :rounds, :total, :ktot, :ns, :nc, :dmax)

# phase profile (p.prof): cycles of block 1 between the barriers, added to dv.prof[k]
@inline function tick!(cx, descs, g, prof::Bool, k::Int, t::UInt64)
    if prof && vblock(cx) == 1 && threadIdx().x == 1
        now = clock(UInt64)
        @inbounds DRef(descs, Int32(g)).prof[k] += reinterpret(Int64, now - t)
        return now
    end
    return t
end

# Reference to the descriptor of graph g in descs (device arrays and
# parameters): every field access is its own load from global memory, so only
# the arrays of the current phase are live in registers (holding the whole
# descriptor across the round loop costs ~250 registers per thread and spills).
struct DRef{D}
    descs::D
    g::Int32
end
@inline function Base.getproperty(r::DRef{D}, s::Symbol) where {D}
    descs = getfield(r, :descs)
    DT = eltype(D)
    i = Base.fieldindex(DT, s)
    FT = fieldtype(DT, i)
    p = reinterpret(Core.LLVMPtr{FT, 1}, pointer(descs, getfield(r, :g)) + fieldoffset(DT, i))
    return unsafe_load(p, 1, Val(Base.datatype_alignment(FT)))
end

# ---- device-side growth (batch): a graph whose round needs more arena space takes it
# from a spare arena (one bump counter per launch) and redoes the round, instead of
# exiting and waiting for a relaunch. Same new sizes and the same repacked pool layout
# as pgrow! on the host, so the run is the same as with the relaunch; a graph whose
# request does not fit in the spare arena exits with NEED as before.

# store a new device array into field s of descriptor g (in global memory)
@inline function dset!(descs, g::Int32, ::Val{s}, ptr::Core.LLVMPtr{T, 1}, len::Int64) where {s, T}
    DT = eltype(descs)
    i = Base.fieldindex(DT, s)
    FT = fieldtype(DT, i)
    p = reinterpret(Core.LLVMPtr{FT, 1}, pointer(descs, g) + fieldoffset(DT, i))
    unsafe_store!(p, FT(ptr, (Int(len),)), 1, Val(Base.datatype_alignment(FT)))
    return nothing
end
@inline sptr(::Type{T}, spare, off::Int64) where {T} = reinterpret(Core.LLVMPtr{T, 1}, pointer(spare) + off)
al16(x::Int64) = (x + 15) & ~Int64(15)

@inline function dgrow!(cx::BlockCx{NW}, descs, g, ctl, wbuf, spare, sofs, need::Int64) where {NW}
    sh = CuDeviceArray{Int64, 1, CUDA.AS.Shared}(reinterpret(Core.LLVMPtr{Int64, 3}, pointer(wbuf)), (64,))
    gi = Int32(g)
    t = threadIdx().x; lane = laneid(); wid = (t - 1) ÷ 32 + 1
    dv = DRef(descs, gi)
    na = Int(@inbounds ctl[C_NA])
    S = cld(na, NW); lo = (wid - 1) * S + 1; hi = min(na, wid * S)
    # live list lengths of each warp's contiguous piece of the live list (pool repack)
    if need & 4 != 0
        alist = dv.alist; stt = dv.stt; alen = dv.alen
        acc = Int64(0); i = lo + lane
        while i <= hi
            @inbounds v = alist[i]
            (@inbounds stt[v]) == ALIVE && (acc += Int64(@inbounds alen[v]))
            i += 32
        end
        acc = warpsum(acc)
        lane == 0 && (@inbounds sh[16 + wid] = acc)
    end
    sync_threads()
    if t == 1
        @inbounds begin
            # exclusive prefix of the warp totals = each warp's base in the new pool
            used = Int64(0)
            if need & 4 != 0
                for w in 1:NW
                    x = sh[16 + w]; sh[16 + w] = used; used += x
                end
            end
            mk = need & 1 != 0 ? max(ctl[C_KTOT], 2 * Int64(length(dv.kar))) : Int64(0)
            mb = need & 2 != 0 ? max(ctl[C_BTOT], 2 * Int64(length(dv.bar))) : Int64(0)
            cap = Int64(length(dv.pool))
            mp = need & 4 != 0 ? max(cap, 2 * (used + ctl[C_PB]), min(2cap, Int64(1) << 26)) : Int64(0)
            mo = ctl[C_OWANT] > ctl[C_OCAP] ? 2 * ctl[C_OWANT] : Int64(0)
            bytes = (mk > 0 ? al16(4mk) * 4 + al16(8mk) * 3 + al16(16mk) : Int64(0)) + al16(4mb) + al16(4mp) + al16(4mo)
            off = CUDA.atomic_add!(pointer(sofs, 1), bytes)
            sh[1] = off + bytes <= length(spare) ? off : Int64(-1)
            sh[2] = mk; sh[3] = mb; sh[4] = mp; sh[5] = mo; sh[6] = used
        end
    end
    sync_threads()
    @inbounds off = sh[1]
    off < 0 && return false
    @inbounds mk = sh[2]; @inbounds mb = sh[3]; @inbounds mp = sh[4]; @inbounds mo = sh[5]; @inbounds used = sh[6]
    # offsets of the new arrays (same order as the byte count above)
    okar = off; okblk = okar + al16(4mk); ocnt = okblk + al16(4mk); opar = ocnt + al16(4mk); ohtab = opar + al16(4mk)
    oswar = ohtab + al16(16mk); okwt = oswar + al16(8mk); ohsh = okwt + al16(8mk)
    obar = ohsh + al16(8mk)
    opool = obar + al16(4mb); oob = opool + al16(4mp)
    if mp > 0
        # repack: each warp copies its piece of the live lists, in list order, from its base
        oldpool = dv.pool; alist = dv.alist; stt = dv.stt; alen = dv.alen; astart = dv.astart
        np = sptr(Int32, spare, opool)
        @inbounds base = sh[16 + wid]
        for i in lo:hi
            @inbounds v = alist[i]
            if (@inbounds stt[v]) == ALIVE
                @inbounds l = Int64(alen[v]); @inbounds s0 = astart[v]
                k = lane
                while k < l
                    @inbounds unsafe_store!(np, oldpool[s0 + k], base + k + 1)
                    k += 32
                end
                CUDA.sync_warp()
                lane == 0 && (@inbounds astart[v] = base + 1)
                base += l
            end
        end
    end
    if mo > 0
        nob = sptr(UInt32, spare, oob)
        k = t
        while k <= mo
            unsafe_store!(nob, UInt32(0), k)
            k += blockDim().x
        end
    end
    sync_threads()
    if t == 1
        if mk > 0
            dset!(descs, gi, Val(:kar), sptr(Int32, spare, okar), mk); dset!(descs, gi, Val(:kblk), sptr(Int32, spare, okblk), mk)
            dset!(descs, gi, Val(:cnt), sptr(Int32, spare, ocnt), mk); dset!(descs, gi, Val(:par), sptr(Int32, spare, opar), mk)
            dset!(descs, gi, Val(:htab), sptr(Int32, spare, ohtab), 4mk); dset!(descs, gi, Val(:swar), sptr(Int64, spare, oswar), mk)
            dset!(descs, gi, Val(:kwt), sptr(Int64, spare, okwt), mk); dset!(descs, gi, Val(:hsh), sptr(UInt64, spare, ohsh), mk)
            @inbounds ctl[C_KCAP] = mk
        end
        if mb > 0
            dset!(descs, gi, Val(:bar), sptr(UInt32, spare, obar), mb); @inbounds ctl[C_BCAP] = mb
        end
        if mp > 0
            dset!(descs, gi, Val(:pool), sptr(Int32, spare, opool), mp)
            @inbounds ctl[C_PTOP] = used; @inbounds ctl[C_PCAP] = mp
        end
        if mo > 0
            dset!(descs, gi, Val(:ob), sptr(UInt32, spare, oob), mo); @inbounds ctl[C_OCAP] = mo
        end
        @inbounds ctl[C_STATUS] = 0; @inbounds ctl[C_NEED] = 0
        CUDA.atomic_add!(pointer(sofs, 2), Int64(1))
        threadfence_block()
    end
    sync_threads()
    return true
end

@inline function prounds!(cx, descs, g, ctl, kmin, gb, maxr::Int64, wbuf, grow = nothing)
    dv = DRef(descs, Int32(g))
    prof = dv.p.prof
    t = prof ? clock(UInt64) : UInt64(0)
    t00 = t
    @inbounds r0 = ctl[C_NROUND]
    while true
        dv = DRef(descs, Int32(g)); p = dv.p
        @inbounds begin
            ctl[C_STATUS] != 0 && break
            ctl[C_NROUND] - r0 >= maxr && break
            if p.needdef && ctl[C_OWANT] > ctl[C_OCAP]
                # the outside-mask arena is too small: grow it here (batch) or let the host do it
                (grow !== nothing && length(grow[1]) > 0 && dgrow!(cx, descs, g, ctl, wbuf, grow[1], grow[2], Int64(NEED_OB))) && continue
                if vblock(cx) == 1 && threadIdx().x == 1
                    ctl[C_NEED] = NEED_OB; ctl[C_STATUS] = NEED
                end
                break
            end
            cur = ctl[C_CUR]
        end
        Lin, kin, cin, Lout, kout, cout = lbufs(dv, cur)
        k_scan!(cx, Lout, kout, Lin, cin, cout, dv.inL, dv.kv, ctl, kmin, p.relax, p.delta, p.tail, p.urelax, p.udelta)
        gbar!(cx, gb)
        t = tick!(cx, descs, g, prof, 1, t)
        @inbounds st = ctl[C_STATUS]
        @inbounds reb = ctl[C_REB]
        st != 0 && break
        if reb != 0
            dv = DRef(descs, Int32(g)); p = dv.p
            k_reb1!(cx, dv.alist2, dv.alist, dv.stt, dv.kv, ctl, kmin, dv.hist, p.relax, p.delta, p.ltarget, p.urelax, p.udelta)
            gbar!(cx, gb)
            dv = DRef(descs, Int32(g))
            Lin, kin, cin, Lout, kout, cout = lbufs(dv, cur)
            k_reb2!(cx, Lout, kout, cout, dv.alist, dv.alist2, dv.inL, dv.kv, ctl)
            gbar!(cx, gb)
            t = tick!(cx, descs, g, prof, 2, t)
        end
        dv = DRef(descs, Int32(g)); p = dv.p
        Lin, kin, cin, Lout, kout, cout = lbufs(dv, cur)
        k_cand!(cx, dv.cand, dv.cpri, dv.best, Lout, kout, cout, ctl, dv.pool, dv.astart, dv.alen, p.seed, dv.tch, p.tidx, p.fcap, p.voff)
        gbar!(cx, gb)
        t = tick!(cx, descs, g, prof, 3, t)
        dv = DRef(descs, Int32(g))
        k_luby2!(cx, dv.sel, dv.best, dv.cand, dv.cpri, ctl, cin, dv.pool, dv.astart, dv.alen)
        gbar!(cx, gb)
        t = tick!(cx, descs, g, prof, 4, t)
        dv = DRef(descs, Int32(g)); p = dv.p
        k_sizes!(cx, dv.kst, dv.bst, dv.best, dv.cand, dv.sel, ctl, dv.pool, dv.astart, dv.alen, dv.slog, p.tailsel, p.tailmax,
            p.logstats, dv.kar, dv.kblk, dv.dsel, dv.wks, dv.kown, dv.kidx, dv.kstamp, dv.stt, dv.w, dv.nm, dv.cks, p.fcap, dv.kwt, dv.wun)
        gbar!(cx, gb)
        t = tick!(cx, descs, g, prof, 5, t)
        @inbounds st = ctl[C_STATUS]
        if st != 0
            # out of arena space: grow on the device (batch) and redo the round
            st == NEED && grow !== nothing && length(grow[1]) > 0 && dgrow!(cx, descs, g, ctl, wbuf, grow[1], grow[2], @inbounds(ctl[C_NEED])) &&
                continue
            break
        end
        if prof && vblock(cx) == 1 && threadIdx().x == 1
            pr = DRef(descs, Int32(g)).prof
            @inbounds pr[13] += ctl[C_KTOP]; @inbounds pr[14] += ctl[C_NS]; @inbounds pr[15] += ctl[C_NC]; @inbounds pr[16] += ctl[C_DLAST]
        end
        dv = DRef(descs, Int32(g)); p = dv.p
        dense = p.needdef && (@inbounds ctl[C_DLAST]) >= p.dgroup
        k_elim12!(cx, dv.bar, dv.swar, dv.def, dv.kar, dv.kblk, dv.dsel, dv.kown, dv.kidx, dv.sel, dv.kst, dv.bst, ctl, dv.pool,
            dv.astart, dv.alen, dv.stt, dv.w, p.needdef, p.dgroup, dv.lown, dv.lid, wbuf)
        gbar!(cx, gb)
        t = tick!(cx, descs, g, prof, 6, t)
        if dense
            dv = DRef(descs, Int32(g)); p = dv.p
            k_ob!(cx, dv.ob, dv.lown, dv.lid, dv.kar, dv.kblk, dv.dsel, dv.kown, dv.sel, dv.kst, ctl, dv.pool, dv.astart, dv.alen, dv.stt,
                p.dgroup, p.needdef)
            gbar!(cx, gb)
            dv = DRef(descs, Int32(g)); p = dv.p
            k_whg!(cx, dv.def, dv.bar, dv.ob, dv.lown, dv.lid, dv.kar, dv.kwt, dv.wun, dv.kblk, dv.dsel, dv.kown, dv.sel, dv.kst, dv.bst,
                ctl, dv.pool, dv.astart, dv.alen, dv.stt, dv.w, p.dgroup, p.needdef, wbuf)
            gbar!(cx, gb)
            t = tick!(cx, descs, g, prof, 7, t)
        end
        dv = DRef(descs, Int32(g)); p = dv.p
        Lin, kin, cin, Lout, kout, cout = lbufs(dv, cur)
        k_elim34!(cx, dv.pool, dv.astart, dv.alen, dv.wdeg, dv.def, dv.hsh, dv.cnt, dv.kar, dv.swar, dv.kblk, dv.dsel, dv.wks, dv.kown,
            dv.sel, dv.kst, ctl, dv.stt, dv.w, p.needdef, dv.kv, dv.inL, Lout, cout, dv.kstamp, p.sc, dv.ob, wbuf)
        gbar!(cx, gb)
        t = tick!(cx, descs, g, prof, 8, t)
        if p.compress
            dv = DRef(descs, Int32(g))
            k_detectw!(cx, dv.par, dv.hsh, dv.cnt, dv.htab, dv.pool, dv.astart, dv.alen, dv.stt, dv.kown, dv.kar, dv.sel, dv.kst,
                dv.dsel, ctl)
            gbar!(cx, gb)
            t = tick!(cx, descs, g, prof, 9, t)
        end
        dv = DRef(descs, Int32(g)); p = dv.p
        Lin, kin, cin, Lout, kout, cout = lbufs(dv, cur)
        k_applyw!(cx, Lout, cout, dv.inL, dv.order, dv.parent, dv.par, dv.stt, dv.w, dv.wdeg, dv.def, dv.kar, dv.sel, dv.kst, dv.dsel,
            ctl, dv.kv, p.sc, p.compress, dv.tch, dv.nm, dv.cks)
        gbar!(cx, gb)
        t = tick!(cx, descs, g, prof, 10, t)
        if prof && vblock(cx) == 1 && threadIdx().x == 1
            @inbounds DRef(descs, Int32(g)).prof[11] += 1
        end
    end
    if prof && vblock(cx) == 1 && threadIdx().x == 1
        @inbounds DRef(descs, Int32(g)).prof[12] += reinterpret(Int64, clock(UInt64) - t00)
    end
    return nothing
end

# one block per graph (descs[b] = arrays and parameters of graph b)
function k_pblock!(cx::BlockCx, descs, maxr::Int64, spare, sofs)
    dv = DRef(descs, Int32(blockIdx().x))
    ctl = CuStaticSharedArray(Int64, NCTL)
    kmin = CuStaticSharedArray(UInt32, 2)
    gctl = dv.ctl; gkmin = dv.kmin
    t = threadIdx().x
    while t <= NCTL
        @inbounds ctl[t] = gctl[t]
        t += blockDim().x
    end
    threadIdx().x <= 2 && (@inbounds kmin[threadIdx().x] = gkmin[threadIdx().x])
    sync_threads()
    # (always the tuple: a Union here would compile the round loop twice, shared memory included)
    prounds!(cx, descs, blockIdx().x, ctl, kmin, nothing, maxr, CuStaticSharedArray(Int32, OSHT), (spare, sofs))
    sync_threads()
    t = threadIdx().x
    while t <= NCTL
        @inbounds gctl[t] = ctl[t]
        t += blockDim().x
    end
    threadIdx().x <= 2 && (@inbounds gkmin[threadIdx().x] = kmin[threadIdx().x])
    return nothing
end

# the whole (cooperative) grid on one graph
function k_pgrid!(cx::GridCx, descs, gb, maxr::Int64)
    dv = DRef(descs, Int32(1))
    prounds!(cx, descs, 1, dv.ctl, dv.kmin, gb, maxr, CuStaticSharedArray(Int32, OSHT))
    return nothing
end

# copy the input lists (compact, global vertex ids) to their pool regions
function k_scatter!(pool, ctgt, cptr, astart, alen, n)
    v = gthread()
    while v <= n
        @inbounds s = cptr[v]; @inbounds d = astart[v]; @inbounds l = alen[v]
        for k in 0:(l - 1)
            @inbounds pool[d + k] = ctgt[s + k]
        end
        v += nthreads()
    end
    return nothing
end

# Twin representatives without a sort (batch setup): rep(x) = the smallest vertex with
# x's closed-neighbourhood hash, which is the run head of the stable sortperm that
# grounds_csr uses (k_runhead!), so par0 is the same. Open addressing on the hash.
@inline hslot(h::UInt64, mask::Int64) = Int64((h ⊻ (h >> 29)) & UInt64(mask))
function k_hins!(tab, hk, mask::Int64, n)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if x <= n
        @inbounds h = hk[x]
        s = hslot(h, mask)
        while true
            @inbounds cur = tab[s + 1]
            if cur == Int32(0)
                old = CUDA.atomic_cas!(pointer(tab, s + 1), Int32(0), Int32(x))
                old == Int32(0) && break
                cur = old
            end
            if (@inbounds hk[cur]) == h
                CUDA.atomic_min!(pointer(tab, s + 1), Int32(x))
                break
            end
            s = (s + 1) & mask
        end
    end
    return nothing
end
function k_twinsh!(par, tab, hk, mask::Int64, pool, astart, alen, n)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if x <= n
        @inbounds h = hk[x]
        s = hslot(h, mask)
        y = @inbounds tab[s + 1]
        while (@inbounds hk[y]) != h
            s = (s + 1) & mask
            y = @inbounds tab[s + 1]
        end
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

# ---------------------------------------------------------------- host side

# a device vector for elements off+1 : off+len of a
@inline function dsub(a::CUDA.CuDeviceArray{T, 1, A}, off::Integer, len::Integer) where {T, A}
    return CUDA.CuDeviceArray{T, 1, A}(a.ptr + Int(off) * sizeof(T), (Int(len),))
end

"""
    PCfg(kind, G, nw)

Persistent configuration: `kind` is `:block` (one block of `32nw` threads),
`:grid` (cooperative grid of `G` blocks of `TPB` threads) or `:replay` (the
captured-graph driver of `grounds_csr`). `nw = 0` and `G = 0` mean automatic.
"""
struct PCfg
    kind::Symbol
    G::Int
    nw::Int
end

# warps per block of the one-block mode: the largest power of two (<= 32) that
# leaves at least 128 registers per thread
function auto_nw(dev = CUDA.device())
    regs = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)
    tmax = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    nw = 32
    while nw > 1 && (32nw > tmax || regs ÷ (32nw) < 128)
        nw >>= 1
    end
    return nw
end

"""
    pick_mode(n) -> PCfg

Mode for a graph with `n` vertices, from the device attributes: one block
(`auto_nw` warps) while n is at most twice its thread count; a cooperative grid
of one block per SM while n is at most VPS vertices per SM; else the replayed
CUDA graph (whose full-device grids win once the rounds have enough work).
"""
function pick_mode(n::Integer; dev = CUDA.device())
    nsm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    coop = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH) != 0
    nw = auto_nw(dev)
    n <= 2 * 32 * nw && return PCfg(:block, 1, nw)
    (coop && n <= VPS * nsm) && return PCfg(:grid, nsm, TPB ÷ 32)
    return PCfg(:replay, 0, 0)
end

# state of a persistent run over B graphs: vertex arrays are shared (global
# vertex ids, graph b owns vo[b]+1 : vo[b]+ns[b]); everything else is a
# per-graph slice; descs[b] holds graph b's device views and parameters
mutable struct PState
    B::Int
    ns::Vector{Int}
    vo::Vector{Int}
    U::NamedTuple
    descs::Vector
    keep::Vector{Vector{Any}}
    poolref::Vector{Tuple{Any, Int64}}
    ctlH::Matrix{Int64}
    ntail::Vector{Int}
    pb::Any            # pinned staging buffer (checked out; returned by pfree!)
end

const VFIELDS = (:astart, :alen, :stt, :w, :wdeg, :def, :parent, :inL, :kown, :kidx, :kstamp, :kv, :best, :tch, :nm, :lown, :lid)
const NFIELDS = (:order, :alist, :alist2, :L1, :L2, :key1, :key2, :cand, :cpri, :sel, :kst, :bst, :dsel, :wks, :cks, :wun)
const KFIELDS = (:kar, :kblk, :swar, :hsh, :cnt, :par, :kwt)
const KTYPES = (Int32, Int32, Int64, UInt64, Int32, Int32, Int64)

function mkparams(alg::RoundsAlg; ltarget, urelax = NaN, udelta = NaN, fcap = 0, dgroup = DGROUP, prof = false)
    sc = alg.score === :md ? 0 : alg.score === :mf ? 1 : 2
    return (sc = Int32(sc), relax = Float32(alg.relax), delta = Float32(alg.delta * Rounds.scoreunit(alg.score)),
        tail = Int64(alg.tail), tailsel = Int64(alg.tailsel), tailmax = Int64(alg.tailmax),
        seed = UInt32(alg.seed % UInt32), ltarget = Int64(ltarget), needdef = sc != 0,
        compress = alg.compress, logstats = false, tidx = alg.ties === :index, fcap = Int64(fcap),
        urelax = Float32(isnan(urelax) ? alg.relax : urelax),
        udelta = Float32((isnan(udelta) ? alg.delta : udelta) * Rounds.scoreunit(alg.score)),
        dgroup = Int32(min(dgroup, typemax(Int32))), voff = Int32(0), prof = prof)
end

# ---- host staging and input packing (batch end to end, SMALL_NOTES.md "End to end")

# Pinned host buffers: a pool of buffers, each checked out by one batch (or chunk)
# at a time (so concurrent calls from several tasks are safe), grown on demand and
# kept across calls (allocating pinned memory costs ~0.1 ms per MB, so the first
# large call pays it once). Uploads from them are asynchronous DMA copies.
const PINFREE = Any[]
const PINLOCK = ReentrantLock()
function pin_take(nbytes::Integer)
    lock(PINLOCK) do
        # the smallest free buffer that is large enough, else replace the largest free one
        i = 0
        for (j, b) in enumerate(PINFREE)
            length(b) >= nbytes && (i == 0 || length(b) < length(PINFREE[i])) && (i = j)
        end
        if i > 0
            return popat!(PINFREE, i)
        end
        if !isempty(PINFREE)
            j = argmax(length.(PINFREE))
            CUDA.unsafe_free!(popat!(PINFREE, j))
        end
        return CuVector{UInt8, CUDA.HostMemory}(undef, max(Int(nbytes) + Int(nbytes) ÷ 4, 1 << 20))
    end
end
pin_give(b) = (b === nothing || lock(() -> push!(PINFREE, b), PINLOCK); nothing)
# typed host view of bytes off+1 : off+n*sizeof(T) of a pinned buffer
hwrap(b, ::Type{T}, off::Integer, n::Integer) where {T} =
    unsafe_wrap(Array, Ptr{T}(UInt(pointer(b; type = CUDA.HostMemory)) + UInt(off)), Int(n))
al8(x) = (x + 7) & ~7

# adjacency sources: a SimpleGraph without self loops (its lists are sorted) or a sorted CSR
struct CSRSrc{P, T}
    ptr::P
    tgt::T
end
srcof(g::SimpleGraph) = g
srcof(g::AbstractGraph) = CSRSrc(csr(g)...)
srcof(c::Tuple) = CSRSrc(c[1], c[2])
srcof(c::CSRSrc) = c
srcn(g::SimpleGraph) = length(g.fadjlist)
srcn(c::CSRSrc) = length(c.ptr) - 1
srct(g::SimpleGraph) = Int64(2 * ne(g))
srct(c::CSRSrc) = length(c.ptr) <= 1 ? Int64(0) : Int64(c.ptr[end] - c.ptr[1])

@inline function packw(ws, v)
    ws === nothing && return Int64(WSCALE)
    return Int64(max(1, round(Int, WSCALE * @inbounds(ws[v]))))
end

# contiguous ranges of 1:B with about equal volume (for the threaded host loops)
function vranges(vol::AbstractVector, parts::Int)
    B = length(vol)
    (B == 0 || parts <= 1) && return [1:B]
    tot = sum(vol); tgt = max(tot / parts, 1); rs = UnitRange{Int}[]; a = 1; acc = 0.0
    for b in 1:B
        acc += vol[b]
        if acc >= tgt * (length(rs) + 1) || b == B
            push!(rs, a:b); a = b + 1
        end
    end
    return rs
end
function tforeach(f, rs)
    if Threads.nthreads() == 1 || length(rs) <= 1
        foreach(f, rs)
    else
        @sync for r in rs
            Threads.@spawn f(r)
        end
    end
    return nothing
end

# stage timer: synchronises the stream only when timing is asked for
@inline function tmark!(tm, key::Symbol, t0::Float64; sync::Bool = false)
    tm === nothing && return t0
    sync && CUDA.synchronize()
    t = time()
    tm[key] = get(tm, key, 0.0) + (t - t0)
    return t
end

# device memory budget of the arenas of one call: a quarter of the free memory
arena_budget() = CUDA.free_memory() ÷ 4

# Host side of the setup (no device work): sizes, arena sizes, and the lists and
# per-vertex arrays of the B graphs packed straight into a pinned buffer
# (threaded over graphs). graphs: SimpleGraph / AbstractGraph / (ptr, tgt) CSR /
# CSRSrc; wts: nothing (unit) or per-graph weights.
# Arena sizes: arena = :min keeps the growth-on-demand start sizes (kmin0, poolfactor/poolsmall);
# :auto sizes each graph's arenas up front, so that NEED relaunches are rare: pool = pfac × its lists
# (pfac from the device memory budget, 8-32; observed totals of the ND leaves: 11-16×, rarely 38×),
# clique arena = max(kmin0, 2 maxdeg, kfrac × n) (observed peaks: up to 0.34 n).
struct PPack
    B::Int
    ns::Vector{Int}
    tots::Vector{Int64}
    vo::Vector{Int}
    caps::Vector{Int64}
    base::Vector{Int64}
    kcaps::Vector{Int64}
    N::Int
    TOT::Int64
    pb::Any
    offs::NTuple{8, Int}
end

function ppack(graphs::AbstractVector, wts; poolfactor = 2.0, poolmin = 1 << 16, kmin0 = 1 << 16, poolsmall = 8,
        arena::Symbol = :min, kfrac = 0.5, pfmax = 32.0, tm = nothing)
    t0 = time()
    B = length(graphs)
    nth = Threads.nthreads()
    srcs = Any[g isa SimpleGraph ? g : srcof(g) for g in graphs]
    local N, TOT, vo, ns, tots, maxd, offs, caps
    pb = nothing
    while true
        ns = Int[srcn(s) for s in srcs]; tots = Int64[srct(s) for s in srcs]
        vo = zeros(Int, B); for b in 2:B; vo[b] = vo[b - 1] + ns[b - 1]; end
        toff = zeros(Int64, B); for b in 2:B; toff[b] = toff[b - 1] + tots[b - 1]; end
        N = sum(ns; init = 0); TOT = sum(tots; init = Int64(0))
        if arena === :auto
            pfac = clamp(arena_budget() / (4 * max(TOT, 1)), 8.0, Float64(pfmax))
            caps = Int64[max(Int64(poolmin), ceil(Int64, pfac * t)) for t in tots]
        else
            caps = Int64[max(Int64(poolmin), round(Int64, poolfactor * t), min(poolsmall * t, Int64(1) << 26)) for t in tots]
        end
        base = zeros(Int64, B); for b in 2:B; base[b] = base[b - 1] + caps[b - 1]; end
        # pinned staging: ctgt Int32[TOT] | cptr, sabs, srel, w Int64[N] | alen Int32[N] | stt UInt8[N] | alist, live Int32[N]
        M = max(N, 1)
        o1 = al8(4 * max(TOT, 1)); o2 = o1 + 8M; o3 = o2 + 8M; o4 = o3 + 8M; o5 = o4 + 8M; o6 = al8(o5 + 4M)
        o7 = al8(o6 + M); o8 = o7 + 4M; nb = o8 + 4M
        offs = (o1, o2, o3, o4, o5, o6, o7, o8)
        pb === nothing || pin_give(pb)
        pb = pin_take(nb)
        ctgt = hwrap(pb, Int32, 0, max(TOT, 1)); cptr = hwrap(pb, Int64, o1, M); sabs = hwrap(pb, Int64, o2, M)
        srel = hwrap(pb, Int64, o3, M); wH = hwrap(pb, Int64, o4, M); alenH = hwrap(pb, Int32, o5, M)
        maxd = zeros(Int, B)
        t0 = tmark!(tm, :pre_size, t0)
        tforeach(vranges(Float64[tots[b] + ns[b] for b in 1:B], 4nth)) do r
            for b in r
                maxd[b] = packg!(ctgt, cptr, sabs, srel, alenH, wH, srcs[b], vo[b], toff[b], base[b], wts === nothing ? nothing : wts[b])
            end
        end
        t0 = tmark!(tm, :pre_pack, t0)
        # a SimpleGraph with self loops (packg! returned -1): drop them (csr_generic) and pack again
        bad = findall(<(0), maxd)
        isempty(bad) && break
        for b in bad
            srcs[b] = CSRSrc(csr_generic(graphs[b])...)
        end
    end
    kcaps = arena === :auto ?
        Int64[max(Int64(kmin0), 2 * Int64(maxd[b]), cld(Int64(ns[b]) * round(Int64, 1024kfrac), 1024)) for b in 1:B] :
        Int64[max(Int64(kmin0), 2 * Int64(maxd[b])) for b in 1:B]
    base = zeros(Int64, B); for b in 2:B; base[b] = base[b - 1] + caps[b - 1]; end
    return PPack(B, ns, tots, vo, caps, base, kcaps, N, TOT, pb, offs)
end

# write graph b's lists (global ids) at ctgt[toff+1 ...] and its per-vertex arrays;
# returns its largest degree (-1: a SimpleGraph with a self loop)
function packg!(ctgt, cptr, sabs, srel, alenH, wH, g::SimpleGraph, o::Int, toff::Int64, base::Int64, ws)
    fl = g.fadjlist; k = toff; s = Int64(1); md = 0; loop = false
    @inbounds for v in eachindex(fl)
        lst = fl[v]; l = length(lst); md = max(md, l)
        cptr[o + v] = k + 1; srel[o + v] = s; sabs[o + v] = base + s; alenH[o + v] = l % Int32
        for u in lst
            k += 1; ctgt[k] = (u + o) % Int32; loop |= u == v
        end
        s += l
        wH[o + v] = packw(ws, v)
    end
    return loop ? -1 : md
end
function packg!(ctgt, cptr, sabs, srel, alenH, wH, c::CSRSrc, o::Int, toff::Int64, base::Int64, ws)
    ptr = c.ptr; tgt = c.tgt; k = toff; p1 = Int64(ptr[1]); md = 0
    @inbounds for v in 1:srcn(c)
        a = Int64(ptr[v]); l = Int64(ptr[v + 1]) - a; s = a - p1 + 1; md = max(md, Int(l))
        cptr[o + v] = k + 1; srel[o + v] = s; sabs[o + v] = base + s; alenH[o + v] = l % Int32
        for q in a:(a + l - 1)
            k += 1; ctgt[k] = (tgt[q] + o) % Int32
        end
        wH[o + v] = packw(ws, v)
    end
    return md
end

# Device side of the setup of a packed batch (on the task's stream): upload, and the
# initial state (twins, weighted degrees, deficiencies, keys) on the disjoint union
# of the graphs, with the same kernels as grounds_csr.
function pdev(P::PPack, p0; ocap0 = 1 << 20, compress = true, twins::Symbol = :hash, tm = nothing)
    t0 = time()
    B = P.B; N = P.N; TOT = P.TOT; ns = P.ns; vo = P.vo; M = max(N, 1)
    caps = P.caps; base = P.base; kcaps = P.kcaps
    kof = zeros(Int64, B); for b in 2:B; kof[b] = kof[b - 1] + kcaps[b - 1]; end
    KC = sum(kcaps; init = Int64(0))
    o1, o2, o3, o4, o5, o6, o7, o8 = P.offs; pb = P.pb
    ctgt = hwrap(pb, Int32, 0, max(TOT, 1)); cptr = hwrap(pb, Int64, o1, M); sabs = hwrap(pb, Int64, o2, M)
    srel = hwrap(pb, Int64, o3, M); wH = hwrap(pb, Int64, o4, M); alenH = hwrap(pb, Int32, o5, M)
    sttH = hwrap(pb, UInt8, o6, M); alistH = hwrap(pb, Int32, o7, M); live = hwrap(pb, Int32, o8, M)
    nth = Threads.nthreads()
    pool = CuVector{Int32}(undef, max(sum(caps; init = Int64(0)), 1))
    alen = CuVector{Int32}(undef, M); astart = CuVector{Int64}(undef, M); w = CuVector{Int64}(undef, M)
    if N > 0
        copyto!(alen, 1, alenH, 1, N); copyto!(astart, 1, sabs, 1, N); copyto!(w, 1, wH, 1, N)
    else
        fill!(w, 1)
    end
    if TOT > 0
        cg = CuVector{Int32}(undef, TOT); cp = CuVector{Int64}(undef, N)
        copyto!(cg, 1, ctgt, 1, TOT); copyto!(cp, 1, cptr, 1, N)
        @cuda threads = TPB blocks = nblk(N) k_scatter!(pool, cg, cp, astart, alen, Int32(N))
        CUDA.unsafe_free!(cg); CUDA.unsafe_free!(cp)
    end
    t0 = tmark!(tm, :pre_h2d, t0; sync = true)
    stt = CUDA.fill(ALIVE, M)
    parent = CUDA.zeros(Int32, M)
    nm = CUDA.ones(Int64, M)
    if compress && N > 0
        hk = CuVector{UInt64}(undef, N)
        @cuda threads = TPB blocks = nblk(N) k_hash!(hk, pool, astart, alen, Int32(N))
        par0 = CuVector{Int32}(undef, N)
        if twins === :hash
            mask = Int64(nextpow(2, 2N)) - 1
            tab = CUDA.zeros(Int32, mask + 1)
            @cuda threads = TPB blocks = nblk(N) k_hins!(tab, hk, mask, Int32(N))
            @cuda threads = TPB blocks = nblk(N) k_twinsh!(par0, tab, hk, mask, pool, astart, alen, Int32(N))
            CUDA.unsafe_free!(tab)
        else
            perm = Int32.(sortperm(hk))
            hd = CuVector{Int32}(undef, N)
            @cuda threads = TPB blocks = nblk(N) k_runhead!(hd, perm, hk, Int32(N))
            accumulate!(max, hd, hd)
            @cuda threads = TPB blocks = nblk(N) k_twins!(par0, perm, hd, pool, astart, alen, Int32(N))
            foreach(CUDA.unsafe_free!, (hd, perm))
        end
        @cuda threads = TPB blocks = nblk(N) k_absorb0!(parent, stt, w, par0, Int32(N), nm)
        foreach(CUDA.unsafe_free!, (hk, par0))
    end
    t0 = tmark!(tm, :pre_twins, t0; sync = true)
    wdeg = CuVector{Int64}(undef, M)
    N > 0 && @cuda threads = TPB blocks = nblk(N) k_wdeg!(wdeg, pool, astart, alen, stt, w, Int32(N))
    N > 0 && copyto!(sttH, 1, stt, 1, N)
    # live lists: graph b's live vertices (sorted) at o+1 : o+nals[b]; live = all of them, compact
    nals = zeros(Int, B)
    rs = vranges(Float64[ns[b] for b in 1:B], 4nth)
    tforeach(rs) do r
        for b in r
            local o = vo[b]; local c = 0
            @inbounds for v in (o + 1):(o + ns[b])
                sttH[v] == ALIVE && (c += 1; alistH[o + c] = v)
            end
            nals[b] = c
        end
    end
    lo = zeros(Int, B); for b in 2:B; lo[b] = lo[b - 1] + nals[b - 1]; end
    nl = sum(nals; init = 0)
    tforeach(rs) do r
        for b in r
            copyto!(live, lo[b] + 1, alistH, vo[b] + 1, nals[b])
        end
    end
    alist = CuVector{Int32}(undef, M)
    N > 0 && copyto!(alist, 1, alistH, 1, N)
    def = CUDA.zeros(Int64, M)
    if p0.needdef && nl > 0
        lv = CuVector{Int32}(undef, nl); copyto!(lv, 1, live, 1, nl)
        @cuda threads = TPB blocks = nblk(32nl) k_initdef!(def, lv, Int32(nl), pool, astart, alen, stt, w)
        CUDA.unsafe_free!(lv)
    end
    kv = CuVector{Float32}(undef, M)
    N > 0 && @cuda threads = TPB blocks = nblk(N) k_initkey!(kv, stt, def, wdeg, w, Int32(p0.sc), Int32(N))
    N > 0 && copyto!(astart, 1, srel, 1, N)      # the rounds address each graph's own pool region
    t0 = tmark!(tm, :pre_live, t0; sync = true)
    U = (pool = pool, astart = astart, alen = alen, stt = stt, w = w, wdeg = wdeg, def = def, parent = parent,
        inL = CUDA.zeros(UInt32, M), kown = CUDA.zeros(Int32, M), kidx = CUDA.zeros(Int32, M), kstamp = CUDA.zeros(Int32, M),
        kv = kv, best = CUDA.fill(typemax(UInt64), M), tch = CUDA.zeros(UInt8, M), nm = nm,
        lown = CUDA.zeros(Int64, M), lid = CuVector{Int64}(undef, M),
        order = CuVector{Int32}(undef, M), alist = alist, alist2 = CuVector{Int32}(undef, M),
        L1 = CuVector{Int32}(undef, M), L2 = CuVector{Int32}(undef, M), key1 = CuVector{Float32}(undef, M),
        key2 = CuVector{Float32}(undef, M), cand = CuVector{Int32}(undef, M), cpri = CuVector{UInt64}(undef, M),
        sel = CuVector{Int32}(undef, M), kst = CuVector{Int64}(undef, M), bst = CuVector{Int64}(undef, M),
        dsel = CuVector{Int32}(undef, M), wks = CuVector{Int64}(undef, M), cks = CuVector{Int64}(undef, M),
        wun = CuVector{Int64}(undef, M),
        kar = CuVector{Int32}(undef, KC), kblk = CuVector{Int32}(undef, KC), swar = CuVector{Int64}(undef, KC),
        hsh = CuVector{UInt64}(undef, KC), cnt = CuVector{Int32}(undef, KC), par = CuVector{Int32}(undef, KC),
        kwt = CuVector{Int64}(undef, KC), htab = CuVector{Int32}(undef, 4KC), bar = CuVector{UInt32}(undef, KC),
        ob = CUDA.zeros(UInt32, B * ocap0), ctl = CUDA.zeros(Int64, NCTL * B), kmin = CUDA.fill(typemax(UInt32), 2B),
        hist = CUDA.zeros(UInt32, NBIN * B), slog = CuVector{Int32}(undef, 3B), prof = CUDA.zeros(Int64, NPROF * B))
    ctlH = zeros(Int64, NCTL, B)
    for b in 1:B
        c = view(ctlH, :, b)
        c[C_NAL] = nals[b]; c[C_NA] = nals[b]; c[C_PTOP] = P.tots[b]; c[C_PCAP] = caps[b]
        c[C_KCAP] = kcaps[b]; c[C_BCAP] = kcaps[b]; c[C_OCAP] = ocap0; c[C_CUR] = 1
        c[C_FH] = p0.fcap == 0 ? Int64(typemax(UInt32)) : (Int64(typemax(UInt32)) * p0.fcap) ÷ max(p0.fcap, nals[b])
        ns[b] == 0 && (c[C_STATUS] = DONE)
    end
    copyto!(U.ctl, vec(ctlH))
    Ud = map(cudaconvert, U)
    descs = mkdescs(Ud, B, vo, ns, base, caps, kof, kcaps, ocap0, p0)
    tmark!(tm, :pre_alloc, t0; sync = true)
    return PState(B, ns, vo, U, descs, [Any[] for _ in 1:B], Tuple{Any, Int64}[(pool, base[b]) for b in 1:B], ctlH,
        zeros(Int, B), P.pb)
end

function pinit(graphs::AbstractVector, wts, p0; poolfactor = 2.0, poolmin = 1 << 16, kmin0 = 1 << 16,
        ocap0 = 1 << 20, compress = true, poolsmall = 8, arena::Symbol = :min, kfrac = 0.5, pfmax = 32.0, twins::Symbol = :hash,
        tm = nothing)
    P = ppack(graphs, wts; poolfactor, poolmin, kmin0, poolsmall, arena, kfrac, pfmax, tm)
    return pdev(P, p0; ocap0, compress, twins, tm)
end

# device descriptors of the B graphs: views of the union arrays
function mkdescs(Ud, B, vo, ns, base, caps, kof, kcaps, ocap0, p0)
    mk(b) = (o = vo[b]; n = max(ns[b], 1); ko = kof[b]; kc = kcaps[b]; (
        pool = dsub(Ud.pool, base[b], caps[b]),
        astart = Ud.astart, alen = Ud.alen, stt = Ud.stt, w = Ud.w, wdeg = Ud.wdeg, def = Ud.def, parent = Ud.parent,
        inL = Ud.inL, kown = Ud.kown, kidx = Ud.kidx, kstamp = Ud.kstamp, kv = Ud.kv, best = Ud.best, tch = Ud.tch,
        nm = Ud.nm, lown = Ud.lown, lid = Ud.lid,
        order = dsub(Ud.order, o, n), alist = dsub(Ud.alist, o, n), alist2 = dsub(Ud.alist2, o, n),
        L1 = dsub(Ud.L1, o, n), L2 = dsub(Ud.L2, o, n), key1 = dsub(Ud.key1, o, n), key2 = dsub(Ud.key2, o, n),
        cand = dsub(Ud.cand, o, n), cpri = dsub(Ud.cpri, o, n), sel = dsub(Ud.sel, o, n), kst = dsub(Ud.kst, o, n),
        bst = dsub(Ud.bst, o, n), dsel = dsub(Ud.dsel, o, n), wks = dsub(Ud.wks, o, n), cks = dsub(Ud.cks, o, n),
        wun = dsub(Ud.wun, o, n),
        kar = dsub(Ud.kar, ko, kc), kblk = dsub(Ud.kblk, ko, kc), swar = dsub(Ud.swar, ko, kc), hsh = dsub(Ud.hsh, ko, kc),
        cnt = dsub(Ud.cnt, ko, kc), par = dsub(Ud.par, ko, kc), kwt = dsub(Ud.kwt, ko, kc),
        htab = dsub(Ud.htab, 4ko, 4kc), bar = dsub(Ud.bar, ko, kc), ob = dsub(Ud.ob, (b - 1) * ocap0, ocap0),
        ctl = dsub(Ud.ctl, (b - 1) * NCTL, NCTL), kmin = dsub(Ud.kmin, 2(b - 1), 2),
        hist = dsub(Ud.hist, (b - 1) * NBIN, NBIN), slog = dsub(Ud.slog, 3(b - 1), 3),
        prof = dsub(Ud.prof, NPROF * (b - 1), NPROF), p = merge(p0, (voff = Int32(o),))))
    return [mk(b) for b in 1:B]
end

# capacity growth of graph b after a NEED exit (the round is redone: same
# rules as the replay driver)
function pgrow!(S::PState, b::Int)
    c = view(S.ctlH, :, b); d = S.descs[b]; ks = S.keep[b]
    need = c[C_NEED]
    why = (need & 7) | (c[C_OWANT] > c[C_OCAP] ? 8 : 0)
    upd = Pair{Symbol, Any}[]
    if need & 1 != 0
        m = max(c[C_KTOT], 2 * Int64(length(d.kar)))
        for (s, T) in zip(KFIELDS, KTYPES)
            a = CuVector{T}(undef, m); push!(ks, a); push!(upd, s => cudaconvert(a))
        end
        a = CuVector{Int32}(undef, 4m); push!(ks, a); push!(upd, :htab => cudaconvert(a))
        c[C_KCAP] = m
    end
    if need & 2 != 0
        m = max(c[C_BTOT], 2 * Int64(length(d.bar)))
        a = CuVector{UInt32}(undef, m); push!(ks, a); push!(upd, :bar => cudaconvert(a))
        c[C_BCAP] = m
    end
    if need & 4 != 0
        # repack the live lists into a new pool of the graph's own
        na = c[C_NA]; pb = c[C_PB]
        lv = CuVector{Int64}(undef, max(na, 1))
        na > 0 && @cuda threads = TPB blocks = nblk(na) k_livelen!(lv, d.alist, d.stt, d.alen, Int32(na))
        off = similar(lv)
        accumulate!(+, off, lv)
        used = na > 0 ? Int64(Array(view(off, na:na))[1]) : Int64(0)
        off .-= lv
        cap = Int64(length(d.pool))
        newcap = max(cap, round(Int64, 2 * (used + pb)), min(2cap, Int64(1) << 26))
        newpool = CuVector{Int32}(undef, newcap)
        na > 0 && @cuda threads = TPB blocks = nblk(32na) k_repack!(newpool, d.astart, d.pool, d.astart, d.alen, d.alist, d.stt,
            Int32(na), off)
        old = S.poolref[b][1]
        CUDA.synchronize()
        if old !== S.U.pool
            filter!(a -> a !== old, ks); CUDA.unsafe_free!(old)
        end
        CUDA.unsafe_free!(lv); CUDA.unsafe_free!(off)
        push!(ks, newpool); push!(upd, :pool => cudaconvert(newpool)); S.poolref[b] = (newpool, 0)
        c[C_PTOP] = used; c[C_PCAP] = newcap
    end
    if c[C_OWANT] > c[C_OCAP]
        a = CUDA.zeros(UInt32, 2 * c[C_OWANT]); push!(ks, a); push!(upd, :ob => cudaconvert(a))
        c[C_OCAP] = length(a)
    end
    c[C_STATUS] = 0; c[C_NEED] = 0
    isempty(upd) || (S.descs[b] = merge(d, (; upd...)))
    return why
end

# one-block kernel with nw warps; maxregs = 0: as many registers as one block of 32nw threads can have
pkernel_block(nw, descs, maxregs, spare, sofs) = @cuda launch = false maxregs = (maxregs > 0 ? maxregs :
    min(255, CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK) ÷ (32nw))) k_pblock!(BlockCx{nw}(), descs,
    Int64(0), spare, sofs)
pkernel_grid(descs, gb) = @cuda launch = false k_pgrid!(GCX, descs, gb, Int64(0))

# largest cooperative grid of k_pgrid! on this device
function pgrid_max(descs, gb)
    k = pkernel_grid(descs, gb)
    nsm = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    return nsm * CUDA.active_blocks(k.fun, TPB), k
end

# run the round loops of all graphs to the end (or to the CPU tail), relaunching
# after capacity growth and tail switches
function prun!(S::PState, kind::Symbol; G::Int = 1, nw::Int = 8, gputail = true, tailrelax = 0.0, cpualpha = 1.1e-9,
        stats = nothing, maxregs::Int = 0, spare::Integer = 0, tm = nothing)
    act = [b for b in 1:S.B if S.ctlH[C_STATUS, b] == 0]
    tk = time()
    # spare arena for the device-side growth (one block per graph only); kept until pfree!
    sp = CuVector{UInt8}(undef, kind === :block ? Int(spare) : 0); so = CUDA.zeros(Int64, 2)
    S.B > 0 && push!(S.keep[1], sp, so)
    ndg = 0
    nlaunch = 0; ngrow = 0; gwhy = zeros(Int, 4)
    gb = kind === :grid ? CUDA.zeros(UInt32, 1) : nothing
    kb = nothing
    tlast = time(); rlast = 0; tround = 0.0
    while !isempty(act)
        sort!(act; by = b -> -S.ns[b])
        # single graph in the GPU tail: launch 32 rounds at a time and hand the rest
        # to the CPU once the serial dense algorithm is cheaper (as grounds_csr)
        intail = S.B == 1 && S.ntail[1] > 0
        maxr = intail ? Int64(32) : typemax(Int64)
        if kind === :grid
            descs = CuVector([S.descs[act[1]]])
            Gmax, k = pgrid_max(descs, gb)
            k(GCX, descs, gb, maxr; threads = TPB, blocks = max(1, min(G, Gmax)), cooperative = true)
        else
            descs = CuVector([S.descs[b] for b in act])
            if kb === nothing
                kb = pkernel_block(nw, descs, maxregs, sp, so)
                if tm !== nothing
                    tm[:occ] = CUDA.active_blocks(kb.fun, 32nw); tm[:regs] = CUDA.registers(kb)
                    tm[:lmem] = CUDA.memory(kb).local
                end
            end
            kb(BlockCx{nw}(), descs, maxr, sp, so; threads = 32nw, blocks = length(act))
        end
        nlaunch += 1
        copyto!(vec(S.ctlH), S.U.ctl)     # sync
        if length(sp) > 0
            # graphs grown on the device: their descriptors (new arena views) come back
            n2 = Int(Array(view(so, 2:2))[1])
            if n2 > ndg
                ndg = n2
                dh = Array(descs)
                for (j, b) in enumerate(act)
                    S.descs[b] = dh[j]
                end
            end
        end
        tk = tmark!(tm, nlaunch == 1 ? :kernel : :relaunch, tk)
        nxt = Int[]; changed = false
        for b in act
            st = S.ctlH[C_STATUS, b]
            if st == 0
                push!(nxt, b)
            elseif st == TAIL && gputail
                p = S.descs[b].p
                p = merge(p, (relax = Float32(tailrelax), tailsel = Int64(0), urelax = min(p.urelax, Float32(tailrelax))))
                S.descs[b] = merge(S.descs[b], (p = p,))
                S.ntail[b] = Int(S.ctlH[C_NAL, b])
                S.ctlH[C_STATUS, b] = 0
                push!(nxt, b); changed = true
                S.B == 1 && (tlast = time(); rlast = S.ctlH[C_NROUND, b])
            elseif st == NEED
                why = pgrow!(S, b); ngrow += 1
                for i in 1:4; (why >> (i - 1)) & 1 != 0 && (gwhy[i] += 1); end
                push!(nxt, b); changed = true
            end
        end
        if intail && !isempty(nxt) && S.ctlH[C_STATUS, 1] == 0
            nr = S.ctlH[C_NROUND, 1]
            if nr >= rlast + 16
                tnow = time()
                tround = (tnow - tlast) / (nr - rlast)
                tlast = tnow; rlast = nr
                tnal = S.ctlH[C_NAL, 1]
                if cpualpha * Float64(tnal)^3 < 0.5 * tnal * tround
                    S.ctlH[C_STATUS, 1] = TAIL     # the CPU finishes
                    break
                end
            end
        end
        changed && copyto!(S.U.ctl, vec(S.ctlH))
        act = nxt
        tk = tmark!(tm, :grow, tk)
    end
    return (launches = nlaunch, grows = ngrow, gk = gwhy[1], gb = gwhy[2], gp = gwhy[3], go = gwhy[4], dgrows = ndg)
end

# orders (CPU tail for the graphs that stopped with live vertices) and fill counters
# out[no+1:...] = v and the vertices absorbed into it (local numbers), as expand
@inline function emit!(out, no::Int, stack, head, nxt, v::Int32, o::Int)
    sp = 1; @inbounds stack[1] = v
    @inbounds while sp > 0
        x = stack[sp]; sp -= 1
        out[no += 1] = x - o
        ch = head[x]
        while ch != 0
            stack[sp += 1] = ch; ch = nxt[ch]
        end
    end
    return no
end

# host copy of a device array view (a descriptor field)
function dl(d::CUDA.CuDeviceArray{T, 1}) where {T}
    h = Vector{T}(undef, length(d))
    length(h) > 0 && CUDA.unsafe_copyto!(pointer(h), reinterpret(CuPtr{T}, d.ptr), length(h))
    return h
end

# orders of graphs 1:B into out[goff[b]+1 : goff[b]+ns[b]] (local vertex numbers; a
# representative followed by the vertices absorbed into it, as expand), the CPU tail
# for the graphs that stopped with live vertices, and the fill counters.
function pfinish(S::PState, alg::RoundsAlg; out::Vector{Int} = Vector{Int}(undef, sum(S.ns; init = 0)), goff = S.vo,
        tm = nothing)
    t0 = time()
    N = sum(S.ns; init = 0); M = max(N, 1)
    pb = S.pb
    orderH = hwrap(pb, Int32, 0, M); parentH = hwrap(pb, Int32, 4M, M)
    if N > 0
        copyto!(orderH, 1, S.U.order, 1, N); copyto!(parentH, 1, S.U.parent, 1, N)
    end
    t0 = tmark!(tm, :post_d2h, t0)
    tails = falses(S.B)
    for b in 1:S.B
        tails[b] = S.ns[b] > 0 && S.ctlH[C_NAL, b] > 0
    end
    nnz = zeros(Int64, S.B); flops = zeros(Int64, S.B); tw = zeros(Int64, S.B)
    # the absorbed vertices of each representative (union numbering, inside its own graph)
    head = Vector{Int32}(undef, M); nxt = Vector{Int32}(undef, M)
    nemit = zeros(Int, S.B)
    tforeach(vranges(Float64[S.ns[b] for b in 1:S.B], 4Threads.nthreads())) do r
        local nmax = maximum(b -> S.ns[b], r; init = 0)
        local stack = Vector{Int32}(undef, max(nmax, 1))
        for b in r
            local n = S.ns[b]; local o = S.vo[b]; local g = goff[b]
            @inbounds for x in (o + 1):(o + n)
                head[x] = 0; nxt[x] = 0
            end
            @inbounds for x in (o + 1):(o + n)
                local p = parentH[x]
                if p != 0
                    nxt[x] = head[p]; head[p] = x
                end
            end
            local nord = Int(S.ctlH[C_NORD, b]); local no = g
            @inbounds for i in 1:nord
                no = emit!(out, no, stack, head, nxt, orderH[o + i], o)
            end
            nemit[b] = no - g
            nnz[b] = S.ctlH[C_NNZ, b]; flops[b] = S.ctlH[C_FLOP, b]; tw[b] = S.ctlH[C_TW, b]
        end
    end
    t0 = tmark!(tm, :post_unpack, t0)
    if any(tails)
        sttH = Array(S.U.stt); astartH = Array(S.U.astart); alenH = Array(S.U.alen); wH = Array(S.U.w); nmH = Array(S.U.nm)
        tstack = Vector{Int32}(undef, M)
        for b in 1:S.B
            tails[b] || continue
            n = S.ns[b]; o = S.vo[b]; no = goff[b] + nemit[b]
            fnnz = nnz[b]; fflop = flops[b]; ftw = tw[b]
            poolH = dl(S.descs[b].pool)
            tv = Int32[v for v in (o + 1):(o + n) if sttH[v] == ALIVE]
            nt = length(tv)
            loc = Dict{Int32, Int}(v => i for (i, v) in enumerate(tv))
            fadj = Vector{Vector{Int}}(undef, nt); m2 = 0
            for (i, v) in enumerate(tv)
                s = astartH[v]; l = alenH[v]
                a = Int[]
                for q in s:(s + l - 1)
                    j = get(loc, poolH[q], 0)
                    j > 0 && push!(a, j)
                end
                fadj[i] = sort!(a); m2 += length(a)
            end
            sub = SimpleGraph(m2 ÷ 2, fadj)
            wsub = [wH[v] / WSCALE for v in tv]
            so, _ = permutation(wsub, sub, serialalg(alg))
            for i in so
                no = emit!(out, no, tstack, head, nxt, tv[i], o)
            end
            qs = [nmH[v] for v in tv]
            cw = tailcw(qs, sub, so)
            for (k, i) in enumerate(so)
                q = qs[i]; cc = cw[k] - q
                fnnz += q * cc + (q * (q + 1)) >> 1
                fflop += q * cc * cc + cc * q * (q + 1) + (q * (q + 1) * (2q + 1)) ÷ 6
                ftw = max(ftw, cc + q)
            end
            nemit[b] = no - goff[b]
            nnz[b] = fnnz; flops[b] = fflop; tw[b] = ftw
        end
        t0 = tmark!(tm, :post_tail, t0)
    end
    for b in 1:S.B
        @assert nemit[b] == S.ns[b] "emitted $(nemit[b]) of $(S.ns[b])"
        tw[b] -= 1
    end
    orders = [view(out, (goff[b] + 1):(goff[b] + S.ns[b])) for b in 1:S.B]
    return orders, (nnz = nnz, flops = flops, tw = tw, rounds = Int[S.ctlH[C_NROUND, b] for b in 1:S.B],
        tail = Int[tails[b] ? S.ctlH[C_NAL, b] : 0 for b in 1:S.B])
end

function pfree!(S::PState)
    pin_give(S.pb); S.pb = nothing
    for a in values(S.U)
        CUDA.unsafe_free!(a)
    end
    for ks in S.keep, a in ks
        CUDA.unsafe_free!(a)
    end
    return nothing
end

# single graph in a persistent mode (called by grounds_csr for mode != :replay)
function grounds_pcsr(weights::AbstractVector, ptr::AbstractVector, tgt::AbstractVector, alg::RoundsAlg, pc::PCfg;
        timing = nothing, tcsr = 0.0, poolfactor = 2.0, gputail::Bool = alg.tailalg === nothing, tailrelax::Real = 0.0,
        cpualpha::Real = 1.1e-9, urelax::Real = NaN, udelta::Real = NaN, fcap::Integer = 0, dgroup::Integer = DGROUP,
        ocap::Integer = 1 << 20, prof::Bool = false)
    T0 = time()
    n = length(ptr) - 1
    n == 0 && return Int[], Int[]
    kind = pc.kind
    nw = kind === :block ? (pc.nw > 0 ? pc.nw : auto_nw()) : TPB ÷ 32
    nthr = kind === :block ? 32nw : TPB * max(pc.G, 1)
    p0 = mkparams(alg; ltarget = max(32, nthr ÷ 4), urelax, udelta, fcap, dgroup, prof)
    S = pinit([(ptr, tgt)], [weights], p0; poolfactor, ocap0 = Int(ocap), compress = alg.compress)
    CUDA.synchronize()
    T1 = time()
    G = pc.G
    if kind === :grid && G <= 0
        G = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    end
    r = prun!(S, kind; G, nw, gputail, tailrelax, cpualpha)
    T2 = time()
    orders, info = pfinish(S, alg)
    if prof && timing !== nothing
        pr = Array(S.U.prof)
        for (k, nm) in enumerate(PROF_NAMES)
            timing[Symbol("prof_", nm)] = pr[k]
        end
    end
    pfree!(S)
    T3 = time()
    if timing !== nothing
        timing[:pre] = T1 - T0; timing[:csr] = tcsr; timing[:gpu] = T2 - T1; timing[:cpu_post] = T3 - T2
        timing[:rounds] = info.rounds[1]; timing[:launches] = r.launches; timing[:grows] = r.grows
        timing[:tail] = info.tail[1]; timing[:gtail] = S.ntail[1]
        timing[:nnz] = info.nnz[1]; timing[:flops] = info.flops[1]; timing[:tw] = info.tw[1]
        timing[:mode] = kind === :block ? "block$(32nw)" : "grid$(G)"
    end
    o = parent(orders[1])       # the whole buffer: one graph
    return o, invperm(o)
end

"""
    grounds_small(weights, graph, alg; mode=:auto, G=0, nw=0, kw...)

GPU rounds with the whole ordering in one kernel launch (`:block`, `:grid`; see
`pick_mode` for `:auto`). Same results as `grounds` for the same configuration.
"""
function grounds_small(weights::AbstractVector, graph::AbstractGraph, alg::RoundsAlg; mode::Symbol = :auto, G::Integer = 0,
        nw::Integer = 0, kw...)
    return grounds(weights, graph, alg; mode, G, nw, kw...)
end

"""
    grounds_batch(graphs, alg; weights=nothing, nw=0, maxregs=0, timing=nothing, arena=:auto, chunks=0, spare=-1, kw...)
        -> (orders, info)

Orders many graphs, one block of `32nw` threads per graph, largest first (`nw = 0`:
from the batch shape and the device, see `batch_cfg`; `maxregs` caps the registers). `graphs`
are `SimpleGraph`s (any `AbstractGraph`; `(ptr, tgt)` CSRs through
`grounds_batch_csr`); `weights` is `nothing` (unit) or one vector per graph.
`orders[b]` is a view into one `Vector{Int}` buffer (local vertex numbers of graph b;
`collect` it to own it). `info` has per-graph `nnz`, `flops`, `tw` (unit counts, as
`timing` of `grounds`), `rounds`, `tail`, and the totals `launches`, `grows`.

End to end (SMALL_NOTES.md, "End to end"):
- the lists are packed straight from the graphs into pinned host memory (threaded
  over graphs when Julia has threads) and uploaded with asynchronous copies;
- `arena = :auto` sizes each graph's arenas up front (see `ppack`), so that a graph
  rarely runs out of space and needs a relaunch (`:min`: the old start sizes);
- the graphs (sorted by size) are cut into `chunks` pieces of growing volume
  (0 = from the volume, 1-4); chunk k is set up and launched on its own stream
  while the host packs chunk k+1, and unpacked while later chunks run;
- the orders are unpacked threaded into one buffer.
Graphs that reach a dense tail stay on the GPU (serial rule) unless `alg.tailalg` is set.
`timing` (a Dict) receives the stage times (s): pre_size, pre_pack, pre_h2d, pre_twins,
pre_live, pre_alloc, kernel, grow, relaunch, post_d2h, post_unpack, post_tail (summed over
chunks; asking for them syncs the GPU between stages), total, gc (GC time), alloc (bytes),
chunks, launches, grows and the growth reasons gk/gb/gp/go (clique, bit-row, pool,
outside-mask arenas).
"""
function grounds_batch(graphs::AbstractVector, alg::RoundsAlg; weights = nothing, kw...)
    return grounds_batch_csr(graphs, weights, alg; kw...)
end

# chunk boundaries of the size-sorted graphs: volume fractions 1/2^(K-1), 1/2^(K-1), 1/2^(K-2), ..., 1/2
function bchunks(vol::AbstractVector, K::Int)
    B = length(vol)
    K = clamp(K, 1, max(B, 1))
    K == 1 && return [1:B]
    tot = sum(vol); cuts = [tot / 2.0^(K - k) for k in 1:(K - 1)]
    rs = UnitRange{Int}[]; a = 1; acc = 0.0; k = 1
    for b in 1:B
        acc += vol[b]
        if k <= K - 1 && acc >= cuts[k] && b < B
            push!(rs, a:b); a = b + 1
            while k <= K - 1 && acc >= cuts[k]; k += 1; end
        end
    end
    push!(rs, a:B)
    return filter(!isempty, rs)
end

# Block configuration of a batch (nw = 0): resident blocks per SM from the device attributes
# (threads, registers, blocks per SM; registers = maxregs, or all a block can have), and
#  - graphs of >= 16k vertices (median): 32 warps (the widest block);
#  - else the widest of 16 warps, 8 warps at 128 registers, 4 warps at 128 registers whose
#    resident blocks (SMs x blocks per SM) are at least a third of the graphs, keeping at
#    least nmed/32 threads per block; the narrowest allowed one if none is.
# (B200, SMALL_NOTES "End to end": 1k-2k graphs x 500-3000 run 1.3-1.8x faster at 2-4 blocks
# per SM than at one block of 16 warps; 10k graphs x 100-200 and 30-50k graphs want the widest.)
function batch_cfg(B::Integer, nmed::Integer; dev = CUDA.device())
    nsm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    tsm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    rsm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)
    rbl = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)
    bsm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR)
    tmax = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    regs(nw, mr) = mr > 0 ? mr : min(255, rbl ÷ (32nw))
    occ(nw, mr) = min(tsm ÷ (32nw), rsm ÷ (32nw * regs(nw, mr)), bsm)
    nmed >= 16384 && 32 * 32 <= tmax && return (32, 0)
    cands = [(16, 0), (8, 128), (4, 128)]
    filter!(c -> 32c[1] <= tmax, cands)
    allowed = filter(c -> 32c[1] >= nmed / 32, cands)
    isempty(allowed) && (allowed = cands[1:1])
    for c in allowed
        nsm * occ(c...) >= B / 3 && return c
    end
    return allowed[end]
end

function grounds_batch_csr(csrs::AbstractVector, wts, alg::RoundsAlg; nw::Integer = 0, maxregs::Integer = 0,
        timing = nothing, tcsr = 0.0, arena::Symbol = :auto, chunks::Integer = 0, twins::Symbol = :hash, spare::Real = -1,
        poolfactor = 2.0, poolmin = 1024, kmin0 = 2048, ocap0 = 4096, poolsmall = 8, gputail::Bool = alg.tailalg === nothing,
        tailrelax::Real = 0.0, urelax::Real = NaN, udelta::Real = NaN, fcap::Integer = 0, dgroup::Integer = DGROUP)
    T0 = time(); g0 = Base.gc_num()
    B = length(csrs)
    srcs = Any[g isa SimpleGraph ? g : srcof(g) for g in csrs]
    ns = Int[srcn(s) for s in srcs]; vol = Float64[srcn(s) + srct(s) for s in srcs]
    if nw == 0
        nw, mr = batch_cfg(B, B == 0 ? 0 : sort(ns)[cld(B, 2)])
        maxregs == 0 && (maxregs = mr)
    end
    p0 = mkparams(alg; ltarget = max(32, 8nw), urelax, udelta, fcap, dgroup)
    goff = zeros(Int, B); for b in 2:B; goff[b] = goff[b - 1] + ns[b - 1]; end
    out = Vector{Int}(undef, sum(ns; init = 0))
    ord = sortperm(ns; rev = true)
    V = sum(vol; init = 0.0)
    K = chunks > 0 ? Int(chunks) : clamp(floor(Int, log2(max(V, 1.0) / 1e6)) + 1, 1, 4)
    rs = bchunks(vol[ord], K)
    K = length(rs)
    tms = [timing === nothing ? nothing : Dict{Symbol, Any}() for _ in 1:K]
    res = Vector{Any}(undef, K)
    function run(k, P)
        local ix = ord[rs[k]]; local tm = tms[k]
        local S = pdev(P, p0; ocap0, compress = alg.compress, twins, tm)
        local sb = spare >= 0 ? Int(spare) : Int(min(CUDA.free_memory() ÷ 8, max(Int64(256) << 20, 128 * P.TOT)))
        local r = prun!(S, :block; nw = Int(nw), gputail, tailrelax, maxregs = Int(maxregs), spare = sb, tm)
        local inf = pfinish(S, alg; out, goff = goff[ix], tm)[2]
        pfree!(S)
        return (r, inf)
    end
    tasks = Task[]
    for k in 1:K
        ix = ord[rs[k]]
        P = ppack(view(srcs, ix), wts === nothing ? nothing : view(wts, ix); poolfactor, poolmin, kmin0, poolsmall, arena,
            tm = tms[k])
        if K == 1
            res[1] = run(1, P)
        else
            # own task = own stream: chunk k's device work overlaps the packing of chunk k+1
            push!(tasks, Threads.@spawn run($k, $P))
        end
    end
    for k in eachindex(tasks)
        res[k] = fetch(tasks[k])
    end
    nnz = zeros(Int64, B); flops = zeros(Int64, B); tw = zeros(Int64, B); rounds = zeros(Int, B); tail = zeros(Int, B)
    nl = 0; ng = 0; gwhy = zeros(Int, 4); ndg = 0
    for k in 1:K
        r, inf = res[k]; ix = ord[rs[k]]
        nnz[ix] = inf.nnz; flops[ix] = inf.flops; tw[ix] = inf.tw; rounds[ix] = inf.rounds; tail[ix] = inf.tail
        nl += r.launches; ng += r.grows; gwhy .+= (r.gk, r.gb, r.gp, r.go); ndg += r.dgrows
    end
    orders = [view(out, (goff[b] + 1):(goff[b] + ns[b])) for b in 1:B]
    info = (nnz = nnz, flops = flops, tw = tw, rounds = rounds, tail = tail, launches = nl, grows = ng, dgrows = ndg)
    if timing !== nothing
        for tm in tms, (key, v) in tm
            timing[key] = key in (:occ, :regs, :lmem) ? max(get(timing, key, 0), v) : get(timing, key, 0.0) + v
        end
        gd = Base.GC_Diff(Base.gc_num(), g0)
        timing[:csr] = tcsr; timing[:total] = time() - T0 + tcsr; timing[:gc] = gd.total_time / 1e9; timing[:alloc] = gd.allocd
        timing[:chunks] = K; timing[:launches] = nl; timing[:grows] = ng; timing[:dgrows] = ndg
        timing[:nw] = nw; timing[:maxregs] = maxregs
        timing[:gk], timing[:gb], timing[:gp], timing[:go] = gwhy
        timing[:pre] = sum(get(timing, s, 0.0) for s in (:pre_size, :pre_pack, :pre_h2d, :pre_twins, :pre_live, :pre_alloc))
        timing[:gpu] = sum(get(timing, s, 0.0) for s in (:kernel, :grow, :relaunch))
        timing[:cpu_post] = sum(get(timing, s, 0.0) for s in (:post_d2h, :post_unpack, :post_tail))
    end
    return orders, info
end

"""
    grounds_race(weights, graph, alg; serial=MF(; strategy=1), cap=Inf, metric=:nnz, gpu=nothing, kw...)
        -> (order, index, info)

Runs the serial algorithm on a CPU thread and the GPU (`gpu = nothing`: one
`grounds` run with `kw`, e.g. `mode=:auto`; else a vector of `(alg, kw)`
members for `grounds_ens`) at the same time. Returns the order with the smaller
`metric` (`:nnz`, `:flops`, `:tw`, unit counts) when both have finished, or the
first one to finish once `cap` seconds have passed. The loser is not stopped
(it finishes in the background). Needs Julia with at least 2 threads to overlap.
"""
function grounds_race(weights::AbstractVector, graph::AbstractGraph, alg::RoundsAlg; serial = CliqueTrees.MF(; strategy = 1),
        cap::Real = Inf, metric::Symbol = :nnz, gpu = nothing, kw...)
    T0 = time()
    n = nv(graph)
    cpu = Threads.@spawn begin
        t = time()
        o, _ = permutation(weights, graph, serial)
        tc = time() - t
        cw = tailcw(ones(Int64, n), graph, o)
        (order = collect(Int, o), t = tc, nnz = sum(cw), flops = sum(c * c for c in cw; init = 0), tw = maximum(cw; init = 1) - 1)
    end
    gtask = Threads.@spawn begin
        t = time()
        if gpu === nothing
            tm = Dict{Symbol, Any}()
            o, _ = grounds(weights, graph, alg; timing = tm, kw...)
            (order = o, t = time() - t, nnz = tm[:nnz], flops = tm[:flops], tw = tm[:tw])
        else
            o, _, inf = grounds_ens(weights, graph, gpu; metric)
            b = inf.best
            (order = o, t = time() - t, nnz = inf.nnz[b], flops = inf.flops[b], tw = inf.tw[b])
        end
    end
    while true
        dc = istaskdone(cpu); dg = istaskdone(gtask)
        (dc && dg) && break
        (dc || dg) && time() - T0 >= cap && break
        sleep(0.0005)
    end
    rc = istaskdone(cpu) ? fetch(cpu) : nothing
    rg = istaskdone(gtask) ? fetch(gtask) : nothing
    win = rc === nothing ? :gpu : rg === nothing ? :cpu : (getfield(rg, metric) < getfield(rc, metric) ? :gpu : :cpu)
    r = win === :gpu ? rg : rc
    info = (winner = win, total = time() - T0, tcpu = rc === nothing ? NaN : rc.t, tgpu = rg === nothing ? NaN : rg.t,
        cpu = rc === nothing ? nothing : (nnz = rc.nnz, flops = rc.flops, tw = rc.tw),
        gpu = rg === nothing ? nothing : (nnz = rg.nnz, flops = rg.flops, tw = rg.tw))
    return r.order, invperm(r.order), info
end

# ---- device-input / device-output batch (ND leaves that never leave the device; ND_NOTES "Closing the AMD gap")
#
# grounds_batch_dev(uptr, uadj, loffH, alg) orders B graphs given as one union CSR already on the device:
# uptr (NU+1, 1-based positions into uadj), uadj (neighbours as 1-based ids local to their own graph,
# sorted per row, no self loops), loffH (host, B+1): graph b = union vertices loffH[b] : loffH[b+1]-1.
# Unit weights. Returns (dord, info): dord[loffH[b] + i - 1] = local id of the i-th vertex of graph b's
# order, on the device. Same setup as grounds_batch_csr (arena = :auto sizes, twins = :hash, same kernels,
# graphs in the given order, one chunk); the host packing, the live-list download and the host unpacking
# are replaced by device kernels. Graphs that end in a CPU tail (rare: gputail is the default) fall back to
# the host unpacking for the whole batch.

# per graph: adjacency total and largest degree
function k_dgstats!(st, uptr, loff, B)
    b = gthread()
    if b <= B
        @inbounds begin
            a = loff[b]; e = loff[b + 1] - Int32(1)
            md = Int32(0)
            for v in a:e
                md = max(md, uptr[v + 1] - uptr[v])
            end
            st[b] = Int64(e >= a ? uptr[e + 1] - uptr[a] : Int32(0)); st[B + b] = Int64(md)
        end
    end
    return nothing
end
function k_dgid!(gid, loff, B)
    b = gthread()
    if b <= B
        @inbounds for v in loff[b]:(loff[b + 1] - Int32(1))
            gid[v] = b
        end
    end
    return nothing
end
# pool lists (global ids = local + graph offset), alen, sabs (absolute pool start), srel (graph-relative), weights
function k_dscatter!(pool, alen, sabs, srel, w, uptr, uadj, loff, gid, base, N, wunit::Int64)
    v = gthread()
    while v <= N
        @inbounds begin
            b = gid[v]; a = loff[b]
            s0 = uptr[a]; s = uptr[v]; l = uptr[v + 1] - s
            r = Int64(s - s0) + 1
            alen[v] = l; srel[v] = r; d = base[b] + r; sabs[v] = d; w[v] = wunit
            o = a - Int32(1)
            for k in Int32(0):(l - Int32(1))
                pool[d + k] = uadj[s + k] + o
            end
        end
        v += nthreads()
    end
    return nothing
end
function k_dalive!(fl, stt, N)
    v = gthread()
    while v <= N
        @inbounds fl[v] = stt[v] == ALIVE ? Int32(1) : Int32(0)
        v += nthreads()
    end
    return nothing
end
# alist (graph-local compaction at the graph's offset), live (global compaction), nals per graph
function k_dlive!(alist, live, nals, cs, stt, gid, loff, N, B)
    v = gthread()
    while v <= N
        @inbounds begin
            b = gid[v]; a = loff[b]
            c0 = a == Int32(1) ? Int32(0) : cs[a - Int32(1)]
            if stt[v] == ALIVE
                alist[(a - Int32(1)) + (cs[v] - c0)] = v
                live[cs[v]] = v
            end
            if v == loff[b + 1] - Int32(1)
                nals[b] = Int64(cs[v] - c0)
            end
        end
        v += nthreads()
    end
    return nothing
end
# order expansion (as pfinish/emit!): per graph one thread, representatives in elimination order, each
# followed by the vertices absorbed into it (depth first, children in the same order as on the host)
function k_demit!(dord, order, parent, head, nxt, stack, ctl, loff, B)
    b = gthread()
    if b <= B
        @inbounds begin
            a = loff[b]; e = loff[b + 1] - Int32(1); o = a - Int32(1)
            for x in a:e
                head[x] = Int32(0); nxt[x] = Int32(0)
            end
            for x in a:e
                p = parent[x]
                if p != Int32(0)
                    nxt[x] = head[p]; head[p] = x
                end
            end
            nord = Int32(ctl[(b - 1) * NCTL + C_NORD])
            no = o
            for i in Int32(1):nord
                sp = o + Int32(1); stack[sp] = order[o + i]
                while sp > o
                    x = stack[sp]; sp -= Int32(1)
                    no += Int32(1); dord[no] = x - o
                    ch = head[x]
                    while ch != Int32(0)
                        sp += Int32(1); stack[sp] = ch; ch = nxt[ch]
                    end
                end
            end
        end
    end
    return nothing
end

function pdev_dev(uptr, uadj, loffH, loffD, gid, ns, tots, caps, base, kcaps, p0; ocap0 = 4096, compress = true, tm = nothing)
    t0 = time()
    B = length(ns); N = sum(ns; init = 0); M = max(N, 1)
    kof = zeros(Int64, B); for b in 2:B; kof[b] = kof[b - 1] + kcaps[b - 1]; end
    KC = sum(kcaps; init = Int64(0))
    vo = Int[loffH[b] - 1 for b in 1:B]
    pool = CuVector{Int32}(undef, max(sum(caps; init = Int64(0)), 1))
    alen = CuVector{Int32}(undef, M); astart = CuVector{Int64}(undef, M); w = CuVector{Int64}(undef, M)
    srel = CuVector{Int64}(undef, M)
    if N > 0
        baseD = CuVector{Int64}(base)
        @cuda threads = TPB blocks = min(nblk(N), 4096) k_dscatter!(pool, alen, astart, srel, w, uptr, uadj, loffD, gid, baseD, Int32(N), Int64(WSCALE))
        CUDA.unsafe_free!(baseD)
    else
        fill!(w, 1)
    end
    t0 = tmark!(tm, :pre_h2d, t0; sync = true)
    stt = CUDA.fill(ALIVE, M)
    parent = CUDA.zeros(Int32, M)
    nm = CUDA.ones(Int64, M)
    if compress && N > 0
        hk = CuVector{UInt64}(undef, N)
        @cuda threads = TPB blocks = nblk(N) k_hash!(hk, pool, astart, alen, Int32(N))
        par0 = CuVector{Int32}(undef, N)
        mask = Int64(nextpow(2, 2N)) - 1
        tab = CUDA.zeros(Int32, mask + 1)
        @cuda threads = TPB blocks = nblk(N) k_hins!(tab, hk, mask, Int32(N))
        @cuda threads = TPB blocks = nblk(N) k_twinsh!(par0, tab, hk, mask, pool, astart, alen, Int32(N))
        CUDA.unsafe_free!(tab)
        @cuda threads = TPB blocks = nblk(N) k_absorb0!(parent, stt, w, par0, Int32(N), nm)
        foreach(CUDA.unsafe_free!, (hk, par0))
    end
    t0 = tmark!(tm, :pre_twins, t0; sync = true)
    wdeg = CuVector{Int64}(undef, M)
    N > 0 && @cuda threads = TPB blocks = nblk(N) k_wdeg!(wdeg, pool, astart, alen, stt, w, Int32(N))
    alist = CuVector{Int32}(undef, M); live = CuVector{Int32}(undef, M)
    nalsD = CUDA.zeros(Int64, max(B, 1))
    if N > 0
        fl = CuVector{Int32}(undef, N)
        @cuda threads = TPB blocks = min(nblk(N), 4096) k_dalive!(fl, stt, Int32(N))
        cs = similar(fl); accumulate!(+, cs, fl)
        @cuda threads = TPB blocks = min(nblk(N), 4096) k_dlive!(alist, live, nalsD, cs, stt, gid, loffD, Int32(N), Int32(B))
        CUDA.unsafe_free!(fl); CUDA.unsafe_free!(cs)
    end
    nals = Array(nalsD); CUDA.unsafe_free!(nalsD)
    nl = sum(nals; init = 0)
    def = CUDA.zeros(Int64, M)
    if p0.needdef && nl > 0
        @cuda threads = TPB blocks = nblk(32nl) k_initdef!(def, live, Int32(nl), pool, astart, alen, stt, w)
    end
    CUDA.unsafe_free!(live)
    kv = CuVector{Float32}(undef, M)
    N > 0 && @cuda threads = TPB blocks = nblk(N) k_initkey!(kv, stt, def, wdeg, w, Int32(p0.sc), Int32(N))
    N > 0 && copyto!(astart, 1, srel, 1, N)      # the rounds address each graph's own pool region
    CUDA.unsafe_free!(srel)
    t0 = tmark!(tm, :pre_live, t0; sync = true)
    U = (pool = pool, astart = astart, alen = alen, stt = stt, w = w, wdeg = wdeg, def = def, parent = parent,
        inL = CUDA.zeros(UInt32, M), kown = CUDA.zeros(Int32, M), kidx = CUDA.zeros(Int32, M), kstamp = CUDA.zeros(Int32, M),
        kv = kv, best = CUDA.fill(typemax(UInt64), M), tch = CUDA.zeros(UInt8, M), nm = nm,
        lown = CUDA.zeros(Int64, M), lid = CuVector{Int64}(undef, M),
        order = CuVector{Int32}(undef, M), alist = alist, alist2 = CuVector{Int32}(undef, M),
        L1 = CuVector{Int32}(undef, M), L2 = CuVector{Int32}(undef, M), key1 = CuVector{Float32}(undef, M),
        key2 = CuVector{Float32}(undef, M), cand = CuVector{Int32}(undef, M), cpri = CuVector{UInt64}(undef, M),
        sel = CuVector{Int32}(undef, M), kst = CuVector{Int64}(undef, M), bst = CuVector{Int64}(undef, M),
        dsel = CuVector{Int32}(undef, M), wks = CuVector{Int64}(undef, M), cks = CuVector{Int64}(undef, M),
        wun = CuVector{Int64}(undef, M),
        kar = CuVector{Int32}(undef, KC), kblk = CuVector{Int32}(undef, KC), swar = CuVector{Int64}(undef, KC),
        hsh = CuVector{UInt64}(undef, KC), cnt = CuVector{Int32}(undef, KC), par = CuVector{Int32}(undef, KC),
        kwt = CuVector{Int64}(undef, KC), htab = CuVector{Int32}(undef, 4KC), bar = CuVector{UInt32}(undef, KC),
        ob = CUDA.zeros(UInt32, B * ocap0), ctl = CUDA.zeros(Int64, NCTL * B), kmin = CUDA.fill(typemax(UInt32), 2B),
        hist = CUDA.zeros(UInt32, NBIN * B), slog = CuVector{Int32}(undef, 3B), prof = CUDA.zeros(Int64, NPROF * B))
    ctlH = zeros(Int64, NCTL, B)
    for b in 1:B
        c = view(ctlH, :, b)
        c[C_NAL] = nals[b]; c[C_NA] = nals[b]; c[C_PTOP] = tots[b]; c[C_PCAP] = caps[b]
        c[C_KCAP] = kcaps[b]; c[C_BCAP] = kcaps[b]; c[C_OCAP] = ocap0; c[C_CUR] = 1
        c[C_FH] = p0.fcap == 0 ? Int64(typemax(UInt32)) : (Int64(typemax(UInt32)) * p0.fcap) ÷ max(p0.fcap, nals[b])
        ns[b] == 0 && (c[C_STATUS] = DONE)
    end
    copyto!(U.ctl, vec(ctlH))
    Ud = map(cudaconvert, U)
    descs = mkdescs(Ud, B, vo, ns, base, caps, kof, kcaps, ocap0, p0)
    tmark!(tm, :pre_alloc, t0; sync = true)
    return PState(B, ns, vo, U, descs, [Any[] for _ in 1:B], Tuple{Any, Int64}[(pool, base[b]) for b in 1:B], ctlH,
        zeros(Int, B), nothing)
end

"""
    grounds_batch_dev(uptr, uadj, loffH, alg; nw=0, maxregs=0, timing=nothing, kw...) -> (dord, info)

Device-input, device-output batch of grounds_batch_csr (see the comment above): the union CSR of the B
graphs is on the device, the orders come back on the device (local ids, each graph in its own range).
"""
function grounds_batch_dev(uptr::CuVector{Int32}, uadj::CuVector{Int32}, loffH::AbstractVector{<:Integer}, alg::RoundsAlg;
        nw::Integer = 0, maxregs::Integer = 0, timing = nothing, kfrac = 0.5, pfmax = 32.0, poolmin = 1024, kmin0 = 2048,
        ocap0 = 4096, spare::Real = -1, gputail::Bool = alg.tailalg === nothing, tailrelax::Real = 0.0, urelax::Real = NaN,
        udelta::Real = NaN, fcap::Integer = 0, dgroup::Integer = DGROUP)
    T0 = time()
    tm = timing === nothing ? nothing : Dict{Symbol, Any}()
    B = length(loffH) - 1
    ns = Int[loffH[b + 1] - loffH[b] for b in 1:B]
    N = sum(ns; init = 0)
    dord = CuVector{Int32}(undef, max(N, 1))
    B == 0 && return dord, (nnz = Int64[], flops = Int64[], tw = Int64[], rounds = Int[], tail = Int[], launches = 0, grows = 0, dgrows = 0)
    loffD = CuVector{Int32}(loffH)
    sD = CuVector{Int64}(undef, 2B)
    @cuda threads = TPB blocks = nblk(B) k_dgstats!(sD, uptr, loffD, Int32(B))
    gid = CuVector{Int32}(undef, max(N, 1))
    @cuda threads = TPB blocks = nblk(B) k_dgid!(gid, loffD, Int32(B))
    sH = Array(sD); CUDA.unsafe_free!(sD)
    tots = sH[1:B]; maxd = sH[(B + 1):(2B)]
    TOT = sum(tots; init = Int64(0))
    if nw == 0
        nw, mr = batch_cfg(B, sort(ns)[cld(B, 2)])
        maxregs == 0 && (maxregs = mr)
    end
    p0 = mkparams(alg; ltarget = max(32, 8nw), urelax, udelta, fcap, dgroup)
    # arena sizes as ppack(arena = :auto)
    pfac = clamp(arena_budget() / (4 * max(TOT, 1)), 8.0, Float64(pfmax))
    caps = Int64[max(Int64(poolmin), ceil(Int64, pfac * t)) for t in tots]
    base = zeros(Int64, B); for b in 2:B; base[b] = base[b - 1] + caps[b - 1]; end
    kcaps = Int64[max(Int64(kmin0), 2 * Int64(maxd[b]), cld(Int64(ns[b]) * round(Int64, 1024kfrac), 1024)) for b in 1:B]
    tm === nothing || (tm[:pre_size] = time() - T0)
    S = pdev_dev(uptr, uadj, loffH, loffD, gid, ns, tots, caps, base, kcaps, p0; ocap0, compress = alg.compress, tm)
    sb = spare >= 0 ? Int(spare) : Int(min(CUDA.free_memory() ÷ 8, max(Int64(256) << 20, 128 * TOT)))
    r = prun!(S, :block; nw = Int(nw), gputail, tailrelax, maxregs = Int(maxregs), spare = sb, tm)
    t0 = time()
    tails = any(b -> S.ns[b] > 0 && S.ctlH[C_NAL, b] > 0, 1:B)
    if tails
        # CPU tail somewhere: host unpacking (as grounds_batch_csr) for the whole batch
        S.pb = pin_take(8 * max(N, 1))
        out = Vector{Int}(undef, N)
        _, inf = pfinish(S, alg; out, goff = S.vo, tm)
        copyto!(dord, Int32.(out))
    else
        head = CuVector{Int32}(undef, max(N, 1)); nxt = CuVector{Int32}(undef, max(N, 1)); stack = CuVector{Int32}(undef, max(N, 1))
        @cuda threads = 64 blocks = cld(B, 64) k_demit!(dord, S.U.order, S.U.parent, head, nxt, stack, S.U.ctl, loffD, Int32(B))
        foreach(CUDA.unsafe_free!, (head, nxt, stack))
        tw = Int64[S.ctlH[C_TW, b] - 1 for b in 1:B]
        inf = (nnz = Int64[S.ctlH[C_NNZ, b] for b in 1:B], flops = Int64[S.ctlH[C_FLOP, b] for b in 1:B], tw = tw,
            rounds = Int[S.ctlH[C_NROUND, b] for b in 1:B], tail = zeros(Int, B))
    end
    tm === nothing || (tm[:post_emit] = time() - t0)
    pfree!(S)
    foreach(CUDA.unsafe_free!, (loffD, gid))
    info = (nnz = inf.nnz, flops = inf.flops, tw = inf.tw, rounds = inf.rounds, tail = inf.tail, launches = r.launches,
        grows = r.grows, dgrows = r.dgrows)
    if timing !== nothing
        for (key, v) in tm
            timing[key] = v
        end
        timing[:total] = time() - T0; timing[:nw] = nw; timing[:maxregs] = maxregs; timing[:launches] = r.launches
        timing[:grows] = r.grows; timing[:dgrows] = r.dgrows; timing[:tails] = tails
        timing[:pre] = sum(get(timing, s, 0.0) for s in (:pre_size, :pre_h2d, :pre_twins, :pre_live, :pre_alloc))
        timing[:gpu] = sum(get(timing, s, 0.0) for s in (:kernel, :grow, :relaunch))
    end
    return dord, info
end
