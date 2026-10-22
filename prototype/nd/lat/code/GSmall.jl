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

@inline function prounds!(cx, descs, g, ctl, kmin, gb, maxr::Int64, wbuf)
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
                # the outside-mask arena is too small: let the host grow it
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
        @inbounds (ctl[C_STATUS] != 0) && break
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
function k_pblock!(cx::BlockCx, descs, maxr::Int64)
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
    prounds!(cx, descs, blockIdx().x, ctl, kmin, nothing, maxr, CuStaticSharedArray(Int32, OSHT))
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

# upload B graphs (sorted CSR each) and compute the initial state (twins,
# weighted degrees, deficiencies, keys) on their disjoint union, with the same
# kernels as grounds_csr
function pinit(csrs::AbstractVector, wts::AbstractVector, p0; poolfactor = 2.0, poolmin = 1 << 16, kmin0 = 1 << 16,
        ocap0 = 1 << 20, compress = true, poolsmall = 8)
    B = length(csrs)
    ns = Int[length(c[1]) - 1 for c in csrs]
    tots = Int64[ns[b] == 0 ? 0 : Int64(csrs[b][1][ns[b] + 1] - 1) for b in 1:B]
    vo = zeros(Int, B); for b in 2:B; vo[b] = vo[b - 1] + ns[b - 1]; end
    N = sum(ns); TOT = sum(tots)
    caps = Int64[max(Int64(poolmin), round(Int64, poolfactor * t), min(poolsmall * t, Int64(1) << 26)) for t in tots]
    base = zeros(Int64, B); for b in 2:B; base[b] = base[b - 1] + caps[b - 1]; end
    # host CSR of the union: compact targets (global ids), list starts in the
    # compact array, in the pool (absolute) and in the graph's region (relative)
    ctgt = Vector{Int32}(undef, max(TOT, 1))
    cptr = Vector{Int64}(undef, N); sabs = Vector{Int64}(undef, N); srel = Vector{Int64}(undef, N)
    alenH = Vector{Int32}(undef, N); wH = Vector{Int64}(undef, N)
    kcaps = zeros(Int64, B)
    k = 1
    for b in 1:B
        ptr, tgt = csrs[b]; o = vo[b]; ws = wts[b]
        maxd = 0
        @inbounds for v in 1:ns[b]
            s = ptr[v]; l = ptr[v + 1] - s
            cptr[o + v] = k; srel[o + v] = s; sabs[o + v] = base[b] + s; alenH[o + v] = l
            maxd = max(maxd, l)
            for q in s:(s + l - 1)
                ctgt[k] = tgt[q] + o; k += 1
            end
            wH[o + v] = max(1, round(Int, WSCALE * ws[v]))
        end
        kcaps[b] = max(Int64(kmin0), 2 * Int64(maxd))
    end
    kof = zeros(Int64, B); for b in 2:B; kof[b] = kof[b - 1] + kcaps[b - 1]; end
    KC = sum(kcaps)
    pool = CuVector{Int32}(undef, max(sum(caps), 1))
    alen = CuVector{Int32}(alenH)
    astart = CuVector{Int64}(sabs)
    if N > 0
        cg = CuVector{Int32}(ctgt); cp = CuVector{Int64}(cptr)
        @cuda threads = TPB blocks = nblk(N) k_scatter!(pool, cg, cp, astart, alen, Int32(N))
        CUDA.unsafe_free!(cg); CUDA.unsafe_free!(cp)
    end
    stt = CUDA.fill(ALIVE, max(N, 1))
    w = CuVector{Int64}(N > 0 ? wH : Int64[1])
    parent = CUDA.zeros(Int32, max(N, 1))
    nm = CUDA.ones(Int64, max(N, 1))
    if compress && N > 0
        hk = CuVector{UInt64}(undef, N)
        @cuda threads = TPB blocks = nblk(N) k_hash!(hk, pool, astart, alen, Int32(N))
        perm = Int32.(sortperm(hk))
        hd = CuVector{Int32}(undef, N)
        @cuda threads = TPB blocks = nblk(N) k_runhead!(hd, perm, hk, Int32(N))
        accumulate!(max, hd, hd)
        par0 = CuVector{Int32}(undef, N)
        @cuda threads = TPB blocks = nblk(N) k_twins!(par0, perm, hd, pool, astart, alen, Int32(N))
        @cuda threads = TPB blocks = nblk(N) k_absorb0!(parent, stt, w, par0, Int32(N), nm)
        foreach(CUDA.unsafe_free!, (hk, hd, par0, perm))
    end
    wdeg = CuVector{Int64}(undef, max(N, 1))
    N > 0 && @cuda threads = TPB blocks = nblk(N) k_wdeg!(wdeg, pool, astart, alen, stt, w, Int32(N))
    sttH = N > 0 ? Array(stt) : UInt8[]
    alistH = Vector{Int32}(undef, max(N, 1)); live = Vector{Int32}(undef, max(N, 1))
    nals = zeros(Int, B); nl = 0
    for b in 1:B
        o = vo[b]; c = 0
        @inbounds for v in (o + 1):(o + ns[b])
            if sttH[v] == ALIVE
                c += 1; alistH[o + c] = v; nl += 1; live[nl] = v
            end
        end
        nals[b] = c
    end
    alist = CuVector{Int32}(alistH)
    def = CUDA.zeros(Int64, max(N, 1))
    if p0.needdef && nl > 0
        lv = CuVector{Int32}(live[1:nl])
        @cuda threads = TPB blocks = nblk(32nl) k_initdef!(def, lv, Int32(nl), pool, astart, alen, stt, w)
        CUDA.unsafe_free!(lv)
    end
    kv = CuVector{Float32}(undef, max(N, 1))
    N > 0 && @cuda threads = TPB blocks = nblk(N) k_initkey!(kv, stt, def, wdeg, w, Int32(p0.sc), Int32(N))
    N > 0 && copyto!(astart, srel)      # the rounds address each graph's own pool region
    M = max(N, 1)
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
    return PState(B, ns, vo, U, descs, [Any[] for _ in 1:B], Tuple{Any, Int64}[(pool, base[b]) for b in 1:B], ctlH, zeros(Int, B))
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
    return nothing
end

# one-block kernel with nw warps; maxregs = 0: as many registers as one block of 32nw threads can have
pkernel_block(nw, descs, maxregs = 0) = @cuda launch = false maxregs = (maxregs > 0 ? maxregs : min(255, 65536 ÷ (32nw))) k_pblock!(BlockCx{nw}(), descs, Int64(0))
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
        stats = nothing, maxregs::Int = 0)
    act = [b for b in 1:S.B if S.ctlH[C_STATUS, b] == 0]
    nlaunch = 0; ngrow = 0
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
            kb === nothing && (kb = pkernel_block(nw, descs, maxregs))
            kb(BlockCx{nw}(), descs, maxr; threads = 32nw, blocks = length(act))
        end
        nlaunch += 1
        copyto!(vec(S.ctlH), S.U.ctl)     # sync
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
                pgrow!(S, b); ngrow += 1
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
    end
    return (launches = nlaunch, grows = ngrow)
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

function pfinish(S::PState, alg::RoundsAlg)
    N = sum(S.ns)
    orderH::Vector{Int32} = N > 0 ? Array(S.U.order) : Int32[]
    parentH::Vector{Int32} = N > 0 ? Array(S.U.parent) : Int32[]
    tails = falses(S.B)
    for b in 1:S.B
        tails[b] = S.ns[b] > 0 && S.ctlH[C_NAL, b] > 0
    end
    sttH::Vector{UInt8} = UInt8[]; astartH::Vector{Int64} = Int64[]; alenH::Vector{Int32} = Int32[]
    wH::Vector{Int64} = Int64[]; nmH::Vector{Int64} = Int64[]
    if any(tails)
        sttH = Array(S.U.stt); astartH = Array(S.U.astart); alenH = Array(S.U.alen); wH = Array(S.U.w); nmH = Array(S.U.nm)
    end
    # the absorbed vertices of each representative (union numbering), as in expand
    head = zeros(Int32, N); nxt = zeros(Int32, N)
    @inbounds for x in 1:N
        r = parentH[x]
        if r != 0
            nxt[x] = head[r]; head[r] = x
        end
    end
    stack = Vector{Int32}(undef, max(N, 1))
    orders = Vector{Vector{Int}}(undef, S.B)
    nnz = zeros(Int64, S.B); flops = zeros(Int64, S.B); tw = zeros(Int64, S.B)
    for b in 1:S.B
        n = S.ns[b]; o = S.vo[b]; c = view(S.ctlH, :, b)
        nord = Int(c[C_NORD])
        fnnz = c[C_NNZ]; fflop = c[C_FLOP]; ftw = c[C_TW]
        out = Vector{Int}(undef, n); no = 0
        @inbounds for i in 1:nord
            no = emit!(out, no, stack, head, nxt, orderH[o + i], o)
        end
        if tails[b]
            arr, poff = S.poolref[b]
            cap = Int(length(S.descs[b].pool))
            poolH = Array(view(arr, (poff + 1):(poff + cap)))
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
                no = emit!(out, no, stack, head, nxt, tv[i], o)
            end
            qs = [nmH[v] for v in tv]
            cw = tailcw(qs, sub, so)
            for (k, i) in enumerate(so)
                q = qs[i]; cc = cw[k] - q
                fnnz += q * cc + (q * (q + 1)) >> 1
                fflop += q * cc * cc + cc * q * (q + 1) + (q * (q + 1) * (2q + 1)) ÷ 6
                ftw = max(ftw, cc + q)
            end
        end
        @assert no == n "emitted $no of $n"
        orders[b] = out
        nnz[b] = fnnz; flops[b] = fflop; tw[b] = ftw - 1
    end
    return orders, (nnz = nnz, flops = flops, tw = tw, rounds = Int[S.ctlH[C_NROUND, b] for b in 1:S.B],
        tail = Int[tails[b] ? S.ctlH[C_NAL, b] : 0 for b in 1:S.B])
end

function pfree!(S::PState)
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
    o = orders[1]
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
    grounds_batch(graphs, alg; weights=nothing, nw=8, timing=nothing, kw...) -> (orders, info)

Orders many graphs in one kernel launch, one block of `32nw` threads per graph
(launched largest first; relaunched only for graphs that need more arena
space). `info` has per-graph `nnz`, `flops`, `tw` (unit counts, as `timing` of
`grounds`), `rounds` and `tail`. Graphs that reach a dense tail stay on the GPU
(serial rule) unless `alg.tailalg` is set.
"""
function grounds_batch(graphs::AbstractVector, alg::RoundsAlg; weights = nothing, nw::Integer = 8, maxregs::Integer = 0, timing = nothing,
        poolfactor = 2.0, poolmin = 1024, kmin0 = 2048, ocap0 = 4096, poolsmall = 8, gputail::Bool = alg.tailalg === nothing,
        tailrelax::Real = 0.0, urelax::Real = NaN, udelta::Real = NaN, fcap::Integer = 0, dgroup::Integer = DGROUP)
    T0 = time()
    csrs = [csr(g) for g in graphs]
    wts = weights === nothing ? [ones(nv(g)) for g in graphs] : weights
    T1 = time()
    return grounds_batch_csr(csrs, wts, alg; nw, maxregs, timing, tcsr = T1 - T0, poolfactor, poolmin, kmin0, ocap0, poolsmall, gputail, tailrelax,
        urelax, udelta, fcap, dgroup)
end

function grounds_batch_csr(csrs::AbstractVector, wts::AbstractVector, alg::RoundsAlg; nw::Integer = 8, maxregs::Integer = 0,
        timing = nothing, tcsr = 0.0,
        poolfactor = 2.0, poolmin = 1024, kmin0 = 2048, ocap0 = 4096, poolsmall = 8, gputail::Bool = alg.tailalg === nothing,
        tailrelax::Real = 0.0, urelax::Real = NaN, udelta::Real = NaN, fcap::Integer = 0, dgroup::Integer = DGROUP)
    T0 = time()
    p0 = mkparams(alg; ltarget = max(32, 8nw), urelax, udelta, fcap, dgroup)
    S = pinit(csrs, wts, p0; poolfactor, poolmin, kmin0, ocap0, poolsmall, compress = alg.compress)
    CUDA.synchronize()
    T1 = time()
    r = prun!(S, :block; nw = Int(nw), gputail, tailrelax, maxregs = Int(maxregs))
    T2 = time()
    orders, info = pfinish(S, alg)
    pfree!(S)
    T3 = time()
    info = merge(info, (launches = r.launches, grows = r.grows))
    if timing !== nothing
        timing[:csr] = tcsr; timing[:pre] = T1 - T0; timing[:gpu] = T2 - T1; timing[:cpu_post] = T3 - T2
        timing[:launches] = r.launches; timing[:grows] = r.grows
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
