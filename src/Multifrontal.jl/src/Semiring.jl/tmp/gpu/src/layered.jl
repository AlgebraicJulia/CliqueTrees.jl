# ===== layered L sweep =====
#
# rowmajor_down_kernel! gives each row one thread, which walks every front below the top of the tree:
# a serial chain of ~nf dependent gathers per thread, and only k threads. On a large GPU that is a
# few percent of the resident thread slots (22.5k rows against 303k slots on a B200), so the sweep is
# bound by the latency of that chain, not by bandwidth.
#
# The rows are independent, and so are disjoint subtrees within a row. This cuts the fronts below the
# top into layers of regions: the bottom layer holds the maximal subtrees with fewer than m fronts, the
# next layer the maximal such subtrees of what is left, and so on. A launch per layer, top layer first,
# gives every (row, region) pair a thread, which walks its region parents first. Every front's
# ancestors are in its own region (earlier in the walk) or in a layer above, so they are final when it
# runs. With m ≈ √(2 nf), a balanced tree needs two or three layers and each thread walks O(√nf)
# fronts instead of nf. Same operations per front as rowmajor_down_kernel!.
#
const LAYER_MAXREG = 65535           # gridDim.y

struct LayerPlan{I}
    layers::Vector{Tuple{CuVector{I}, CuVector{I}, Int}}   # top-down: (region pointers, fronts, number of regions)
    m::Int
    maxchain::Int                                          # longest walk of one thread, in fronts
end

function layer_plan(G::GPUSLU{<:Any, <:Any, I}) where {I}
    m = config().layer_size                    # region size in fronts (0: √(2 nf))
    @step "layer plan" get!(G.cache, Symbol(:layers, m)) do
        istop = Array(G.istop); pnt = Array(G.pnt)
        nf = G.nf
        alive = .!istop
        m = m > 0 ? max(2, m) : max(16, isqrt(2 * count(alive)))   # m = 1 would leave no region
        #
        # full subtree sizes, to find each subtree's postorder range [f - fsz[f] + 1, f]
        #
        fsz = ones(Int, nf)

        for f in 1:nf
            p = pnt[f]
            iszero(p) || (fsz[p] += fsz[f])
        end

        sz = zeros(Int, nf)
        layers = Tuple{Vector{I}, Vector{I}}[]

        while any(alive)
            fill!(sz, 0)

            for f in 1:nf
                alive[f] || continue
                sz[f] += 1
                p = pnt[f]
                (!iszero(p) && alive[p]) && (sz[p] += sz[f])
            end

            # region roots: alive, below m (or a root of what is left), parent not a region member
            final = all(f -> !alive[f] || sz[f] < m, 1:nf)
            isroot(f) = alive[f] && (final || sz[f] < m) && (iszero(pnt[f]) || !alive[pnt[f]] || (!final && sz[pnt[f]] >= m))
            regptr = I[1]; regfronts = I[]

            for r in 1:nf
                isroot(r) || continue

                for f in r:-1:(r - fsz[r] + 1)          # reverse postorder: parents first
                    alive[f] && push!(regfronts, f)
                end

                push!(regptr, length(regfronts) + 1)
            end

            for f in regfronts
                alive[f] = false
            end

            @assert !isempty(regfronts)
            push!(layers, (regptr, regfronts))
        end

        reverse!(layers)                                # execute top-down
        out = Tuple{CuVector{I}, CuVector{I}, Int}[]
        maxchain = 0

        for (regptr, regfronts) in layers
            nreg = length(regptr) - 1
            maxchain += maximum(diff(regptr))
            #
            # gridDim.y is at most 65535: merge neighbouring regions of a layer if needed
            #
            if nreg > LAYER_MAXREG
                step = cld(nreg, LAYER_MAXREG)
                regptr = regptr[1:step:end]
                regptr[end] == length(regfronts) + 1 || push!(regptr, length(regfronts) + 1)
                nreg = length(regptr) - 1
            end

            push!(out, (upload(regptr), upload(regfronts), nreg))
        end

        LayerPlan{I}(out, m, maxchain)
    end
end

function layered_down_kernel!(s::AbstractSemiring, trans::Val, C::AbstractMatrix{T}, regptr, regfronts,
        Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, zr::Val{SKIP} = Val(false), sources = nothing, cinvp = nothing, idx = nothing, fd = nothing) where {T, SKIP}
    t = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    g = blockIdx().y

    if t <= size(C, 1)
        ft = SKIP ? (@inbounds idx[cinvp[sources[t]]]) : 0

        @inbounds for i in regptr[g]:(regptr[g + 1] - 1)
            f = regfronts[i]
            zi = SKIP && !(fd[f] <= ft <= f)                 # see skip_fill in sgetrs.jl
            downward_front_reg!(s, trans, C, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, zi, Val(false))
        end
    end

    return
end


# ===== slot-cached layered walk =====
#
# layered_down_kernel! spends ~16 instructions per min-plus multiply-add (ncu, B200 grid2d-180: IPC
# 3.3 of 4, DRAM at 2–10%; RTX PRO 6000: long-scoreboard stalls): every front reloads Rptr…Lptr,
# gathers its na separator values from C with 64-bit address arithmetic and one index load each,
# though most of them were written a few fronts earlier by the same thread, loads one L value per
# operation, and a front wider than 8 columns re-gathers its separator for every column.
#
# Here the walk of a region follows a plan of chunks made on the host, and each row keeps a small
# cache of S values in shared memory (slots):
#
#   X[s, row]   s = 1:S,   at X[(s - 1) R + row], R = rows per block
#
# The rows of a thread are private (no other thread reads them), so there is no synchronization. The
# walk of a region is the same for every row, so which column sits in which slot is decided once, by
# simulating the walk with Belady's replacement (keep the values that are read again soonest). Every
# value is still written to C as soon as it is final, so the slots are only a read cache, and a value
# that is not in one is read from C. A chunk computes w ≤ SLOT_W columns:
#
#   x[j] ← x[j] ⊕ ⊕ᵢ vᵢ ⊗ c[i, j]          vᵢ: its inputs, each from a slot or from C
#   x ← x K*  (the solve within the chunk, in registers),   C[t, col_j] ← x[j], and to a slot if read again
#
# Per chunk: one 48-byte header; inputs in groups of 4 (one 16-byte load of their slot offsets or
# columns, one of their coefficients per output column, shared by the SLOT_Q rows of a thread); two
# inputs read from C are loaded one chunk ahead. The coefficients are gathered from the factor into
# plan order on every sweep (one small gather). Chunks are packed (see build_slot_plan), so most
# loaded values feed several columns. The ⊕ of a chunk visits its terms in another order than
# downward_front_reg! and adds zero terms: exact for idempotent ⊕ (min-plus, max-plus, max-min),
# a rounding change otherwise.
#
# Result, L_layered per closure (layered_down_kernel! → this), min-plus results bit-identical:
#
#                  B200            RTX PRO 6000    RTX 5060 Laptop
#   grid2d-180     8.15 → 3.89 ms  8.12 → 4.98 ms  51.5 → 23.4 ms
#   delaunay_n15  10.23 → 4.46     8.42 → 5.18     64.9 → 25.5
#   grid2d-150     4.26 → 2.09     4.33 → 2.64     23.9 → 10.9
#   grid3d-30      8.46 → 4.22     8.20 → 5.15     55.5 → 28.1
#
# On the B200 (ncu, grid2d-180) this is 3.7× fewer instructions (2.0e9 warp instructions, 4.4 per
# multiply-add); it is now latency-bound at 16 warps per SM. On the laptop, writing the result
# (n Σnn values, 3.9 GB for grid2d-180) alone takes ~15 ms. The plan costs 4–6 ms on 4 host threads.
#
const SLOT_W = 8                     # columns per chunk (register-blocked)
const SLOT_PACK = 2.0                # merge fronts into a chunk while it costs ≤ this many times their operations
const SLOT_HDR = 12                  # header words per chunk (three 16-byte loads)
const SLOT_PF = 2                    # inputs read from C that are loaded one chunk ahead
const SLOT_TB = 32                   # threads per block
const SLOT_Q = 4                     # rows per thread (rows t, t + SLOT_TB, ...): uniform loads and addresses shared by 4 rows
const SLOT_MAX = 64                  # at most this many slots per row
const SLOT_MAXREGS = 128             # (112 spills a little; 4 rows × 8 columns of accumulators)
const SLOT_TASKS = 4                 # host tasks building the plan (regions are independent; more gain little)

struct SlotPlan{T}
    layers::Vector{Tuple{CuVector{Int32}, Int}}    # top-down: (first chunk of each region and one past the last, number of regions)
    hdr::CuVector{Int32}            # SLOT_HDR words per chunk, see build_slot_plan
    ent::CuVector{Int32}            # per chunk: output columns, inputs (slot offsets or columns), see build_slot_plan
    cmap::CuVector{Int32}           # source of each coefficient: > 0 an index into LLval, < 0 minus an index into LDval, 0 the semiring zero
    coef::CuVector{T}               # the coefficients in plan order (gathered by slot_coefficients!)
    slots::Int                      # S
    rows::Int                       # R: rows per block, the stride between slots
    sread::Int                      # inputs per row read from a slot (statistics)
    gread::Int                      # inputs per row read from C
end

#
# Slots per row: shared memory is what limits the resident blocks (the walk is latency-bound, so
# occupancy matters more than the last misses), so take as many slots as leave the blocks that the
# kernel's registers allow resident (the driver reserves 1 KB per block), at most SLOT_MAX and 48 KB
# per block. RTX 5060 Laptop / RTX PRO 6000 / L4: 10 slots, B200: 26.
#
function slot_count(::Type{T}, regs::Integer) where {T}
    S = config().layer_slots
    S > 0 && return S
    p = device_profile()
    R = SLOT_TB * SLOT_Q
    nb = max(1, resident_warps(p, regs, SLOT_TB) ÷ cld(SLOT_TB, 32))
    return clamp((p.shmem_sm - nb * 1024) ÷ (nb * R * sizeof(T)), 1, min(SLOT_MAX, 48 * 1024 ÷ (R * sizeof(T))))
end

# mapped: the plan's columns are storage columns of a ColMapped work matrix, ColMapped(·, G.rperm)
function slot_plan(G::GPUSLU{<:Any, T}, S::Int, R::Int; mapped::Bool = false) where {T}
    lp = layer_plan(G)
    @step "slot plan" get!(G.cache, slot_key(lp, S, R, mapped)) do
        build_slot_plan(G, lp, S, R; wcol = mapped ? host_rperm(G) : nothing)
    end
end

slot_key(lp, S, R, mapped::Bool) = Symbol(:slots, lp.m, :_, S, :_, R, mapped ? :_m : :_)

pack8(a, b, c, d) = (a | (b << 8) | (c << 16) | (d << 24)) % Int32

#
# Header of a chunk (SLOT_HDR Int32 words):
#   1 ent pointer of the (fd[a], a) pairs of its outputs (for the rows whose source is below the chunk)
#   2 w | npf << 4 | ng << 6   3 groups of cached inputs   4 ent pointer   5 coefficient pointer
#   6 fd[r]   7 r (r: the highest front of the chunk)   8–9 slots of the outputs (a byte each, 255: none)
#   10 slots of the npf inputs prefetched for this chunk (a byte each)
#   11–12 columns of the inputs to prefetch for the next chunk (0: none)
# ent from the pointer: the w output columns (padded to 4); 4 slot byte offsets ((s - 1) R sizeof(T))
# per group of cached inputs; 4 columns and 4 slot byte offsets (-1: not kept) per group of the other
# inputs, read from C; then the (fd[a], a) pairs. Coefficients from the pointer: [group][output j][4],
# w per prefetched input (then padding to a multiple of 4), [group][output j][4] of the groups read
# from C, then the w(w - 1)/2 of the solve in down_reg!'s order (j = w:-1:1, k = j+1:w).
#
# A chunk is either SLOT_W columns of a wide front, or a connected piece of the region (a front and
# some of its descendants, ≤ SLOT_W columns in all, merged while that costs at most SLOT_PACK times
# the operations of walking them one by one). Its outputs are the columns of its fronts, descendants
# first, so that within the chunk an ancestor's columns are solved before the columns that read them;
# its inputs are the separator values its fronts read from outside it. The coefficient of input (or
# output) column i for output j of front a is L₂₁[i, j] of a when i is in a's separator, (L₁₁)[k, j]
# of a when i is a's residual column k > j, and the semiring zero otherwise: the same products as the
# front-by-front walk plus zero terms, exact for idempotent ⊕.
#
function build_slot_plan(G::GPUSLU{<:Any, T, I}, lp::LayerPlan, S::Int, R::Int; wcol = nothing) where {T, I}
    S < 255 || return nothing                   # slot indices must fit a byte
    st = SlotStructure(G.hRptr, G.hSptr, G.hDptr, G.hLptr, Array(G.Stgt), Array(G.pnt), Array(first_descendants(G)), G.n, G.nf, S, R * sizeof(T),
        isnothing(wcol) ? I[] : Vector{I}(wcol))
    max(length(G.LLval), length(G.LDval)) < typemax(Int32) || return nothing    # coefficient sources must fit 32 bits
    sread = 0; gread = 0
    pool = SlotScratch[]; plock = ReentrantLock()
    outs = Vector{Vector{Tuple{Vector{Int32}, Vector{Int32}, Vector{Int32}, Int, Int}}}()

    for (regptr, regfronts, nreg) in lp.layers
        rp = Array(regptr); rf = Array(regfronts)
        out = Vector{Tuple{Vector{Int32}, Vector{Int32}, Vector{Int32}, Int, Int}}(undef, nreg)
        next = Threads.Atomic{Int}(1)
        #
        # regions are independent: a few tasks take them in turn (dynamic, so that this also behaves
        # inside concurrent tasks), each with its own scratch, kept for the next layer
        #
        worker() = begin
            sc = lock(() -> isempty(pool) ? SlotScratch(st) : pop!(pool), plock)

            while (g = Threads.atomic_add!(next, 1)) <= nreg
                ok = slot_region!(sc, st, view(rf, rp[g]:(rp[g + 1] - 1)))
                out[g] = ok ? (copy(sc.hdr), copy(sc.ent), Vector{Int32}(sc.cmap), sc.sread, sc.gread) : (Int32[], Int32[], Int32[], -1, 0)
            end

            lock(() -> push!(pool, sc), plock)
        end

        ntask = min(Threads.nthreads(), SLOT_TASKS, nreg)
        ntask <= 1 ? worker() : foreach(wait, [Threads.@spawn(worker()) for _ in 1:ntask])
        any(o -> o[4] < 0, out) && return nothing
        push!(outs, out)
    end
    #
    # concatenate the regions (pointers made absolute), into arrays of the final size
    #
    lens = [sum(o -> length(o[k]), out; init = 0) for out in outs, k in 1:3]
    max(sum(lens[:, 2]), sum(lens[:, 3])) < typemax(Int32) - 2^20 || return nothing
    hdr = Vector{Int32}(undef, sum(lens[:, 1])); ent = Vector{Int32}(undef, sum(lens[:, 2])); cmap = Vector{Int32}(undef, sum(lens[:, 3]))
    nh = 0; ne = 0; nk = 0
    layers = Tuple{CuVector{Int32}, Int}[]

    for out in outs
        rchunk = Int32[nh ÷ SLOT_HDR + 1]

        for (h, e, k, sr, gr) in out
            copyto!(hdr, nh + 1, h, 1, length(h))

            for c in (nh + 1):SLOT_HDR:(nh + length(h))
                hdr[c] += ne; hdr[c + 3] += ne; hdr[c + 4] += nk
            end

            copyto!(ent, ne + 1, e, 1, length(e)); copyto!(cmap, nk + 1, k, 1, length(k))
            nh += length(h); ne += length(e); nk += length(k)
            push!(rchunk, nh ÷ SLOT_HDR + 1)
            sread += sr; gread += gr
        end

        push!(layers, (upload(rchunk), length(out)))
    end

    return SlotPlan{T}(layers, upload(hdr), upload(ent), upload(cmap), CuVector{T}(undef, length(cmap)), S, R, sread, gread)
end

# the symbolic structure the plan reads (host)
struct SlotStructure{I}
    Rptr::Vector{I}; Sptr::Vector{I}; Dptr::Vector{I}; Lptr::Vector{I}
    Stgt::Vector{I}; pnt::Vector{I}; fd::Vector{I}
    n::Int; nf::Int; S::Int
    RB::Int                                     # bytes per slot (R sizeof(T))
    wcol::Vector{I}                             # by column: where it is stored (empty: there)
end

# the storage column of a column, in the plan's entries
@inline wcol(st::SlotStructure, col) = isempty(st.wcol) ? col : oftype(col, st.wcol[col])

# scratch of one task, reused from region to region (no allocation per region once grown)
mutable struct SlotScratch
    nxt::Vector{Int32}                          # by column: reverse scan, the next chunk that reads it (0: none)
    lastw::Vector{Int32}                        # by column: the chunk that wrote it (0: none yet)
    cslot::Vector{Int32}                        # by column: its slot (0: none)
    spos::Vector{Int32}                         # by column: its position in the separator of the current front (0: not in it)
    mark::Vector{Int32}                         # by column: stamp of the last set it was put in
    stamp::Int32
    root::Vector{Int32}                         # by front: the highest front of its piece
    head::Vector{Int32}                         # by front (a piece's root): its lowest front
    nextm::Vector{Int32}                        # by front: the next front of its piece (ascending, 0 at the end)
    wsum::Vector{Int32}                         # by front (a root): columns of the piece
    wops::Vector{Int64}                         # by front (a root): operations of walking the piece
    inreg::BitVector                            # by front: in the current region
    scol::Vector{Int32}; snext::Vector{Int32}   # by slot: its column (0: free) and that column's next read
    cout::Vector{Int32}; cfr::Vector{Int32}; cj::Vector{Int32}; coptr::Vector{Int32}    # chunk outputs: column, front, index in it
    cin::Vector{Int32}; ciptr::Vector{Int32}    # chunk inputs
    innext::Vector{Int32}; outnext::Vector{Int32}
    hits::Vector{Int32}; miss::Vector{Int32}; pf::Vector{Int32}
    K::Vector{Int64}                            # coefficient sources of a chunk: [input, output] and the solve [j, k]
    Kd::Vector{Int64}
    hdr::Vector{Int32}; ent::Vector{Int32}; cmap::Vector{Int64}                        # the plan of the region
    sread::Int; gread::Int
end

SlotScratch(st::SlotStructure) = SlotScratch(zeros(Int32, st.n), zeros(Int32, st.n), zeros(Int32, st.n), zeros(Int32, st.n), zeros(Int32, st.n), 0,
    zeros(Int32, st.nf), zeros(Int32, st.nf), zeros(Int32, st.nf), zeros(Int32, st.nf), zeros(Int64, st.nf), falses(st.nf),
    zeros(Int32, st.S), zeros(Int32, st.S), Int32[], Int32[], Int32[], Int32[], Int32[], Int32[], Int32[], Int32[], Int32[], Int32[], Int32[],
    Int64[], zeros(Int64, SLOT_W * SLOT_W), Int32[], Int32[], Int64[], 0, 0)

@inline slot_nn(st, f) = Int(st.Rptr[f + 1] - st.Rptr[f])
@inline slot_na(st, f) = Int(st.Sptr[f + 1] - st.Sptr[f])

# the inputs of the piece whose fronts are listed from `first` (ascending) and outputs [lo, end] of cout
function piece_inputs!(sc::SlotScratch, st::SlotStructure, first, lo)
    sc.stamp += Int32(1); stamp = sc.stamp

    for o in lo:length(sc.cout)
        sc.mark[sc.cout[o]] = stamp
    end

    f = first

    while !iszero(f)
        for r in st.Sptr[f]:(st.Sptr[f + 1] - 1)
            c = st.Stgt[r]
            sc.mark[c] == stamp || (push!(sc.cin, c); sc.mark[c] = stamp)
        end

        f = sc.nextm[f]
    end
end

# the number of inputs and of columns of the union of the pieces listed from a and from b
function piece_size(sc::SlotScratch, st::SlotStructure, a, b)
    sc.stamp += Int32(1); stamp = sc.stamp; w = 0; ni = 0

    for h in (a, b)
        f = h

        while !iszero(f)
            for c in st.Rptr[f]:(st.Rptr[f + 1] - 1)
                sc.mark[c] = stamp; w += 1
            end

            f = sc.nextm[f]
        end
    end

    for h in (a, b)
        f = h

        while !iszero(f)
            for r in st.Sptr[f]:(st.Sptr[f + 1] - 1)
                c = st.Stgt[r]
                sc.mark[c] == stamp || (ni += 1; sc.mark[c] = stamp)
            end

            f = sc.nextm[f]
        end
    end

    return ni, w
end

# the factor entry (> 0 LLval, < 0 LDval, 0 zero) multiplying column i into column j of front a
# (sc.spos holds a's separator positions)
@inline function slot_coef(sc::SlotScratch, st::SlotStructure, i, a, j)
    Rp = st.Rptr[a]; nn = st.Rptr[a + 1] - Rp

    if Rp <= i < Rp + nn
        k = i - Rp + 1
        return k > j ? -Int64(st.Dptr[a] + (j - 1) * nn + k - 1) : Int64(0)
    elseif ispositive(sc.spos[i])
        return Int64(st.Lptr[a] + (j - 1) * (st.Sptr[a + 1] - st.Sptr[a]) + sc.spos[i] - 1)
    else
        return Int64(0)
    end
end

function slot_setsep!(sc::SlotScratch, st::SlotStructure, a, v::Bool)
    for r in st.Sptr[a]:(st.Sptr[a + 1] - 1)
        sc.spos[st.Stgt[r]] = v ? r - st.Sptr[a] + 1 : 0
    end
end

#
# a slot for a value read next at chunk `next` (0: never): a free one, else the one whose value is
# read latest, if later than this one (Belady); 0 when the value is not kept
#
function take_slot!(sc::SlotScratch, S, col, next)
    iszero(next) && return 0
    best = 0; far = next

    for s in 1:S
        if iszero(sc.scol[s])
            best = s; break
        elseif sc.snext[s] > far
            best = s; far = sc.snext[s]
        end
    end

    if ispositive(best)
        iszero(sc.scol[best]) || (sc.cslot[sc.scol[best]] = 0)
        sc.scol[best] = col; sc.snext[best] = next; sc.cslot[col] = best
    end

    return best
end

function slot_align!(v::Vector, x)
    while !iszero(length(v) % 4)
        push!(v, x)
    end
end

# the plan of one region (fronts fs in walk order: descending) into sc.hdr, sc.ent, sc.cmap, with
# pointers relative to the region; false when it does not fit the encoding
function slot_region!(sc::SlotScratch, st::SlotStructure, fs)
    S = st.S; RB = st.RB
    empty!(sc.hdr); empty!(sc.ent); empty!(sc.cmap); sc.sread = 0; sc.gread = 0
    walkops(f) = slot_na(st, f) * slot_nn(st, f) + slot_nn(st, f) * (slot_nn(st, f) - 1) ÷ 2
    #
    # pieces: merge a front's piece into its parent's, children first, while it pays
    #
    for f in fs
        sc.inreg[f] = true; sc.root[f] = f; sc.head[f] = f; sc.nextm[f] = 0; sc.wsum[f] = slot_nn(st, f); sc.wops[f] = walkops(f)
    end

    for i in length(fs):-1:1
        f = fs[i]; p = st.pnt[f]
        (!iszero(p) && sc.inreg[p] && slot_nn(st, f) <= SLOT_W && slot_nn(st, p) <= SLOT_W && sc.wsum[f] + sc.wsum[p] <= SLOT_W) || continue
        ni, w = piece_size(sc, st, sc.head[f], sc.head[p])
        ni * w + w * (w - 1) ÷ 2 <= SLOT_PACK * (sc.wops[f] + sc.wops[p]) || continue
        # merge the two ascending lists
        a = sc.head[f]; b = sc.head[p]; h = 0; t = 0

        while !iszero(a) || !iszero(b)
            if iszero(b) || (!iszero(a) && a < b)
                x = a; a = sc.nextm[a]
            else
                x = b; b = sc.nextm[b]
            end

            iszero(t) ? (h = x) : (sc.nextm[t] = x)
            t = x; sc.root[x] = p
        end

        sc.nextm[t] = 0; sc.head[p] = h; sc.wsum[p] += sc.wsum[f]; sc.wops[p] += sc.wops[f]
    end
    #
    # the chunks, in walk order (a piece where its highest front is)
    #
    empty!(sc.cout); empty!(sc.cfr); empty!(sc.cj); empty!(sc.cin)
    empty!(sc.coptr); empty!(sc.ciptr); push!(sc.coptr, 1); push!(sc.ciptr, 1)

    for f in fs
        sc.root[f] == f || continue
        Rp = st.Rptr[f]; nn = slot_nn(st, f)

        if sc.head[f] == f && nn > SLOT_W
            j1 = nn

            while j1 >= 1                           # a wide front, from its last columns (the solve with L₁₁ runs backward)
                j0 = max(1, j1 - SLOT_W + 1)
                lo = length(sc.cout) + 1

                for j in j0:j1
                    push!(sc.cout, Rp + j - 1); push!(sc.cfr, f); push!(sc.cj, j)
                end

                for r in st.Sptr[f]:(st.Sptr[f + 1] - 1)
                    push!(sc.cin, st.Stgt[r])
                end

                for k in (j1 + 1):nn
                    push!(sc.cin, Rp + k - 1)
                end

                push!(sc.coptr, length(sc.cout) + 1); push!(sc.ciptr, length(sc.cin) + 1)
                j1 = j0 - 1
            end
        else
            lo = length(sc.cout) + 1
            a = sc.head[f]

            while !iszero(a)
                for j in 1:slot_nn(st, a)
                    push!(sc.cout, st.Rptr[a] + j - 1); push!(sc.cfr, a); push!(sc.cj, j)
                end

                a = sc.nextm[a]
            end

            piece_inputs!(sc, st, sc.head[f], lo)
            push!(sc.coptr, length(sc.cout) + 1); push!(sc.ciptr, length(sc.cin) + 1)
        end
    end

    for f in fs
        sc.inreg[f] = false
    end

    nc = length(sc.coptr) - 1
    #
    # next reads, scanning backward: at chunk c, nxt[col] is the first chunk after c that reads col
    #
    resize!(sc.innext, length(sc.cin)); resize!(sc.outnext, length(sc.cout))

    for c in nc:-1:1
        for o in sc.coptr[c]:(sc.coptr[c + 1] - 1)
            sc.outnext[o] = sc.nxt[sc.cout[o]]
        end

        for i in sc.ciptr[c]:(sc.ciptr[c + 1] - 1)
            sc.innext[i] = sc.nxt[sc.cin[i]]; sc.nxt[sc.cin[i]] = c
        end
    end

    for col in sc.cin
        sc.nxt[col] = 0
    end
    #
    # forward: cached inputs are read from their slots, the others from C (and kept if read again soon
    # enough), the outputs written to C (and kept likewise)
    #
    hdr = sc.hdr; ent = sc.ent; cmap = sc.cmap; K = sc.K; Kd = sc.Kd
    okeep = zeros(Int, 8)                       # (one small allocation per region)

    for c in 1:nc
        o0 = sc.coptr[c]; w = sc.coptr[c + 1] - o0; i0 = sc.ciptr[c]; ni = sc.ciptr[c + 1] - i0
        empty!(sc.hits); empty!(sc.miss); empty!(sc.pf)

        for i in 1:ni
            col = sc.cin[i0 + i - 1]

            if ispositive(sc.cslot[col])
                push!(sc.hits, i)
            elseif c > 1 && length(sc.pf) < SLOT_PF && sc.lastw[col] < c - 1      # loaded before chunk c - 1 runs
                push!(sc.pf, i)
            else
                push!(sc.miss, i)
            end
        end
        #
        # coefficients: K[i, j] of the inputs, Kd[j, k] of the solve within the chunk
        #
        length(K) < ni * w && resize!(K, ni * w)

        for j in 1:w
            a = sc.cfr[o0 + j - 1]; jj = sc.cj[o0 + j - 1]
            (j == 1 || sc.cfr[o0 + j - 2] != a) && slot_setsep!(sc, st, a, true)

            for i in 1:ni
                K[(j - 1) * ni + i] = slot_coef(sc, st, sc.cin[i0 + i - 1], a, jj)
            end

            for k in (j + 1):w
                Kd[(k - 1) * w + j] = slot_coef(sc, st, sc.cout[o0 + k - 1], a, jj)
            end

            (j == w || sc.cfr[o0 + j] != a) && slot_setsep!(sc, st, a, false)
        end

        nh = length(sc.hits); ngrp = cld(nh, 4)
        ep = length(ent) + 1; cp = length(cmap) + 1

        for o in o0:(o0 + w - 1)
            push!(ent, wcol(st, sc.cout[o]))
        end

        slot_align!(ent, Int32(0))
        # cached inputs, in groups of 4 (a short group repeats its first slot, with coefficient zero)
        for q in 1:ngrp, k in 1:4
            i = 4q - 4 + k <= nh ? sc.hits[4q - 4 + k] : sc.hits[4q - 3]
            push!(ent, (sc.cslot[sc.cin[i0 + i - 1]] - 1) * RB)
        end

        for q in 1:ngrp, j in 1:w, k in 1:4
            push!(cmap, 4q - 4 + k <= nh ? K[(j - 1) * ni + sc.hits[4q - 4 + k]] : Int64(0))
        end

        for i in sc.hits                        # last read: free the slot
            col = sc.cin[i0 + i - 1]; s = sc.cslot[col]
            sc.snext[s] = sc.innext[i0 + i - 1]

            if iszero(sc.innext[i0 + i - 1])
                sc.scol[s] = 0; sc.cslot[col] = 0
            end
        end

        pcol1 = 0; pcol2 = 0; pk1 = 255; pk2 = 255

        for (k, i) in enumerate(sc.pf)
            col = sc.cin[i0 + i - 1]; s = take_slot!(sc, S, col, sc.innext[i0 + i - 1])
            k == 1 ? (pcol1 = col; pk1 = ispositive(s) ? s - 1 : 255) : (pcol2 = col; pk2 = ispositive(s) ? s - 1 : 255)

            for j in 1:w
                push!(cmap, K[(j - 1) * ni + i])
            end
        end

        slot_align!(cmap, Int64(0))
        nm = length(sc.miss)

        for q in 1:cld(nm, 4)                   # the other inputs read from C, in groups of 4 (padded like the cached ones)
            for k in 1:4
                push!(ent, wcol(st, sc.cin[i0 + sc.miss[min(4q - 4 + k, nm)] - 1]))
            end

            for k in 1:4
                if 4q - 4 + k <= nm
                    col = sc.cin[i0 + sc.miss[4q - 4 + k] - 1]; s = take_slot!(sc, S, col, sc.innext[i0 + sc.miss[4q - 4 + k] - 1])
                    push!(ent, ispositive(s) ? (s - 1) * RB : -1)
                else
                    push!(ent, -1)
                end
            end

            for j in 1:w, k in 1:4
                push!(cmap, 4q - 4 + k <= nm ? K[(j - 1) * ni + sc.miss[4q - 4 + k]] : Int64(0))
            end
        end

        for j in w:-1:1, k in (j + 1):w         # the solve within the chunk, in down_reg!'s order
            push!(cmap, Kd[(k - 1) * w + j])
        end

        zp = length(ent) + 1
        fill!(okeep, 255); r = 0

        for j in 1:w
            a = sc.cfr[o0 + j - 1]; col = sc.cout[o0 + j - 1]
            push!(ent, st.fd[a], a)
            s = take_slot!(sc, S, col, sc.outnext[o0 + j - 1])
            okeep[j] = ispositive(s) ? s - 1 : 255
            sc.lastw[col] = c; r = max(r, a)
        end

        push!(hdr, zp, w | (length(sc.pf) << 4) | (cld(nm, 4) << 6), ngrp, ep, cp, st.fd[r], r)
        push!(hdr, pack8(okeep[1], okeep[2], okeep[3], okeep[4]), pack8(okeep[5], okeep[6], okeep[7], okeep[8]), pack8(pk1, pk2, 255, 255), 0, 0)
        c > 1 && (hdr[end - 2SLOT_HDR + 11] = ispositive(pcol1) ? wcol(st, pcol1) : 0; hdr[end - 2SLOT_HDR + 12] = ispositive(pcol2) ? wcol(st, pcol2) : 0)   # the chunk before loads them
        slot_align!(ent, Int32(0)); slot_align!(cmap, Int64(0))
        sc.sread += nh; sc.gread += length(sc.pf) + nm
        (length(ent) < 2^30 && length(cmap) < 2^30 && nm < 2^24) || return false
    end

    for col in sc.cout
        sc.lastw[col] = 0
    end

    for s in 1:S                                # the next region starts with an empty cache
        iszero(sc.scol[s]) || (sc.cslot[sc.scol[s]] = 0)
        sc.scol[s] = 0
    end

    return true
end

function layered_down_coef_kernel!(coef, cmap, LLval, LDval, z)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds while i <= length(coef)
        k = cmap[i]
        coef[i] = k > 0 ? LLval[k] : k < 0 ? LDval[-k] : z
        i += blockDim().x * gridDim().x
    end

    return
end

function slot_coefficients!(P::SlotPlan{T}, G::GPUSLU) where {T}
    n = length(P.coef)
    ispositive(n) && @cuda threads = 256 blocks = min(cld(n, 256), 4096) layered_down_coef_kernel!(P.coef, P.cmap, G.LLval, G.LDval, szero(G.s, T, Val(:N)))
    return P
end

# four consecutive elements A[i:i+3] in one load (i - 1 a multiple of 4: the plan aligns them; an
# element type that is not a primitive type is loaded one by one)
@inline function vload4(A::CuDeviceVector{T}, i) where {T}
    if isprimitivetype(T)
        p = reinterpret(Core.LLVMPtr{NTuple{4, VecElement{T}}, CUDA.AS.Global}, pointer(A, i))
        v = unsafe_load(p, 1, Val(4 * sizeof(T)))
        return (v[1].value, v[2].value, v[3].value, v[4].value)
    else
        return @inbounds (A[i], A[i + 1], A[i + 2], A[i + 3])
    end
end

@inline slot_header(hdr, c) = (b = (c - Int32(1)) * Int32(SLOT_HDR) + Int32(1); (vload4(hdr, b)..., vload4(hdr, b + Int32(4))..., vload4(hdr, b + Int32(8))...))

@inline ld(p::Core.LLVMPtr{T}) where {T} = unsafe_load(p, 1, Val(sizeof(T)))

@inline st!(p::Core.LLVMPtr{T}, x) where {T} = unsafe_store!(p, x, 1, Val(sizeof(T)))

# the inputs of the next chunk that the header H of this one lists for prefetching (zero where none)
@inline function slot_prefetch(cb::Core.LLVMPtr{T}, msz, H, valid::NTuple{Q}, z, ::Val{RS}) where {T, Q, RS}
    p1 = H[11] > Int32(0); p2 = H[12] > Int32(0)
    a1 = cb + (H[11] - Int32(1)) * msz; a2 = cb + (H[12] - Int32(1)) * msz
    v1 = ntuple(k -> p1 & valid[k] ? ld(a1 + (k - 1) * RS) : z, Val(Q))
    v2 = ntuple(k -> p2 & valid[k] ? ld(a2 + (k - 1) * RS) : z, Val(Q))
    return (v1, v2)
end

#
# The chunks of region blockIdx().y for the Q rows t₀ + tx + (k - 1) TB of this thread. C, the slots
# and the plan are addressed through pointers: the Q rows of a thread share one address computation
# per column or slot, plus compile-time offsets (k - 1) TB.
#
function layered_down_slot_kernel!(s::AbstractSemiring, trans::Val, C::CuDeviceMatrix{T}, ::Val{Q}, ::Val{TB}, nslot::Int32, rchunk, hdr, ent, coef,
        skip::Val{SKIP}, sources, cinvp, idx) where {T, Q, TB, SKIP}
    tx = threadIdx().x % Int32
    R = Int32(TB * Q)
    X = @inbounds CuDynamicSharedArray(T, nslot * R, 0)
    g = blockIdx().y
    row1 = (blockIdx().x - Int32(1)) * R + tx
    m = size(C, 1)
    valid = ntuple(k -> row1 + Int32((k - 1) * TB) <= m, Val(Q))
    fts = ntuple(k -> SKIP && valid[k] ? (@inbounds idx[cinvp[sources[row1 + Int32((k - 1) * TB)]]] % Int32) : Int32(0), Val(Q))
    z = szero(s, T, trans)
    cb = pointer(C) + (row1 - Int32(1)) * sizeof(T)          # C[row1, col] at cb + (col - 1) msz
    msz = m * sizeof(T)
    xb = pointer(X) + (tx - Int32(1)) * sizeof(T)            # slot s of row k at xb + (s - 1) R sizeof(T) + (k - 1) TB sizeof(T)
    RS = Val(TB * sizeof(T))

    @inbounds begin
        c = rchunk[g]; ce = rchunk[g + 1] - Int32(1)
        c > ce && return
        H = slot_header(hdr, c)
        PF = (ntuple(_ -> z, Val(Q)), ntuple(_ -> z, Val(Q)))       # the first chunk of a region prefetches nothing

        while true
            PFn = slot_prefetch(cb, msz, H, valid, z, RS)               # for the next chunk, issued before this one runs
            slot_dispatch(H[2] & Int32(15), (s, trans, cb, msz, xb, ent, coef, valid, skip, fts, H, PF, z, RS, Val(R * sizeof(T))), Val(SLOT_W))
            c == ce && break
            c += Int32(1)
            H = slot_header(hdr, c)
            PF = PFn
        end
    end

    return
end

# slot_chunk! for a chunk of width w ≤ WMAX (a branch per width, uniform across the block)
@generated function slot_dispatch(w, args, ::Val{WMAX}) where {WMAX}
    ex = :(slot_chunk!(args..., Val($WMAX)))

    for k in (WMAX - 1):-1:1
        ex = :(w == $k ? slot_chunk!(args..., Val($k)) : $ex)
    end

    return quote
        $(Expr(:meta, :inline))
        $ex
        return
    end
end

@generated function slot_chunk!(s, trans, cb, msz, xb, ent, coef, valid::NTuple{Q}, ::Val{SKIP}, fts, H, PF, z, ::Val{RS}, ::Val{RB}, ::Val{W}) where {Q, SKIP, RS, RB, W}
    x(j, k) = Symbol(:x, j, :_, k)
    v(i, k) = Symbol(:v, i, :_, k)
    oc(j) = :(cb + ($(Symbol(:o, cld(j, 4)))[$(mod1(j, 4))] - Int32(1)) * msz)    # C[row1, column of output j]
    upd(i, k, j, c) = :($(x(j, k)) = smuladd(s, $(v(i, k)), $c, $(x(j, k)), Val(:N), trans))
    W4 = 4 * cld(W, 4)
    # the output columns (loaded where needed, not kept in registers through the chunk)
    cols = [:($(Symbol(:o, q)) = vload4(ent, H[4] + Int32($(4q - 4)))) for q in 1:cld(W, 4)]
    #
    # the residual entries before the sweep: zero (not read) unless the row's source lies below the
    # output's front (skip_fill in sgetrs.jl), which the header tests for the whole chunk first
    #
    if SKIP
        init = Expr[]

        for k in 1:Q
            append!(init, [:($(x(j, k)) = z) for j in 1:W])
            ld_k = [:(if ent[zp + Int32($(2j - 2))] <= fts[$k] <= ent[zp + Int32($(2j - 1))]; $(x(j, k)) = ld($(oc(j)) + $((k - 1) * RS)); end) for j in 1:W]
            push!(init, :(if valid[$k] & (H[6] <= fts[$k] <= H[7]); zp = H[1]; $(cols...); $(ld_k...); end))
        end
    else
        init = [cols; [:($(x(j, k)) = valid[$k] ? ld($(oc(j)) + $((k - 1) * RS)) : z) for j in 1:W for k in 1:Q]]
    end
    # a group of 4 cached inputs
    grp = Expr[:(o = vload4(ent, e)); :(e += Int32(4))]
    append!(grp, [:($(v(i, k)) = ld(xb + o[$i] + $((k - 1) * RS))) for i in 1:4 for k in 1:Q])

    for j in 1:W
        push!(grp, :(cc = vload4(coef, p)), :(p += Int32(4)))
        append!(grp, [upd(i, k, j, :(cc[$i])) for i in 1:4 for k in 1:Q])
    end
    # one input with value v(1, ·) and coefficients coef[p:p+W-1]
    one = Expr[]
    append!(one, [upd(1, k, j, :(coef[p + Int32($(j - 1))])) for j in 1:W for k in 1:Q])
    push!(one, :(p += Int32($W)))
    keep(off) = :(if $off >= Int32(0); $([:(st!(xb + $off + $((k - 1) * RS), $(v(1, k)))) for k in 1:Q]...); end)
    slotoff(word, b) = :((($word >> Int32($(8b))) & Int32(255)) == Int32(255) ? Int32(-1) : (($word >> Int32($(8b))) & Int32(255)) * Int32($RB))
    pf1 = [[:($(v(1, k)) = PF[1][$k]) for k in 1:Q]; :(ko = $(slotoff(:(H[10]), 0))); keep(:ko); one]
    pf2 = [[:($(v(1, k)) = PF[2][$k]) for k in 1:Q]; :(ko = $(slotoff(:(H[10]), 1))); keep(:ko); one]
    # a group of 4 inputs read from C: their columns, then their slot byte offsets (-1: not kept)
    rest = Expr[:(rc = vload4(ent, e)); :(rk = vload4(ent, e + Int32(4))); :(e += Int32(8))]
    append!(rest, [:($(v(i, k)) = valid[$k] ? ld(cb + (rc[$i] - Int32(1)) * msz + $((k - 1) * RS)) : z) for i in 1:4 for k in 1:Q])
    append!(rest, [:(if rk[$i] >= Int32(0); $([:(st!(xb + rk[$i] + $((k - 1) * RS), $(v(i, k)))) for k in 1:Q]...); end) for i in 1:4])

    for j in 1:W
        push!(rest, :(cc = vload4(coef, p)), :(p += Int32(4)))
        append!(rest, [upd(i, k, j, :(cc[$i])) for i in 1:4 for k in 1:Q])
    end

    solve = Expr[]

    for j in W:-1:1, k in (j + 1):W
        push!(solve, :(d = coef[p]), :(p += Int32(1)))
        append!(solve, [:($(x(j, q)) = smuladd(s, $(x(k, q)), d, $(x(j, q)), Val(:N), trans)) for q in 1:Q])
    end

    stores = copy(cols)

    for j in 1:W
        push!(stores, :(os = $(slotoff(j <= 4 ? :(H[8]) : :(H[9]), (j - 1) % 4))))
        append!(stores, [:(valid[$k] && (st!($(oc(j)) + $((k - 1) * RS), $(x(j, k))); true)) for k in 1:Q])
        push!(stores, :(if os >= Int32(0); $([:(st!(xb + os + $((k - 1) * RS), $(x(j, k)))) for k in 1:Q]...); end))
    end

    return quote
        $(Expr(:meta, :inline))

        @inbounds begin
            npf = (H[2] >> Int32(4)) & Int32(3); ng = H[2] >> Int32(6); e = H[4] + Int32($W4); p = H[5]
            $(init...)

            for _ in Int32(1):H[3]
                $(grp...)
            end

            if npf >= Int32(1)
                $(pf1...)
            end

            if npf >= Int32(2)
                $(pf2...)
            end

            p = ((p + Int32(2)) & ~Int32(3)) + Int32(1)                # the coefficients of the groups read from C are aligned
            for _ in Int32(1):ng
                $(rest...)
            end

            $(solve...)
            $(stores...)
        end

        return
    end
end

#
# The L sweep below the top of the tree with the slot-cached walk; nothing when the plan does not
# apply (the caller then runs layered_down_kernel!).
#
function layered_slot_sweep!(G::GPUSLU, W::Union{CuMatrix{T}, ColMapped{T}}, trans::Val, timer, zr) where {T}
    s = G.s
    mapped = W isa ColMapped                    # (then the plan's columns are W.p's: see slot_plan)
    W = storage_matrix(W)
    zargs = isnothing(zr) ? (Val(false), nothing, nothing, nothing) : (Val(true), zr, G.cinvp, G.idx)
    R = SLOT_TB * SLOT_Q
    i32 = get!(() -> CuVector{Int32}(undef, 4), G.cache, :int32)::CuVector{Int32}      # stands for the plan's arrays, to compile first
    kernel = @cuda launch = false maxregs = SLOT_MAXREGS layered_down_slot_kernel!(s, trans, W, Val(SLOT_Q), Val(SLOT_TB), Int32(1),
        i32, i32, i32, G.LLval, zargs...)
    P = slot_plan(G, slot_count(T, CUDA.registers(kernel)), R; mapped)
    isnothing(P) && return nothing
    @phase timer :L_layered slot_coefficients!(P, G)
    nb = cld(size(W, 1), R)
    shmem = P.slots * R * sizeof(T)

    for (rchunk, nreg) in P.layers
        @phase timer :L_layered kernel(s, trans, W, Val(SLOT_Q), Val(SLOT_TB), Int32(P.slots), rchunk, P.hdr, P.ent, P.coef, zargs...;
            threads = SLOT_TB, blocks = (nb, nreg), shmem)
    end

    return W
end
