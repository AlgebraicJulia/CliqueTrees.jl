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
    get!(G.cache, Symbol(:layers, m)) do
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
            downward_front_reg!(s, trans, C, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, zi)
        end
    end

    return
end

# ===== slot-cached layered walk =====
#
# layered_down_kernel! is bound by instruction issue on the big GPUs (B200: IPC 3.3 of 4, ~16
# instructions per min-plus multiply-add, DRAM at 2–10%) and by DRAM on the small ones (laptop, L4):
# every front gathers its na separator values from global memory with 64-bit address arithmetic,
# though most of them were written a few fronts earlier by the same thread, and a front wider than
# 8 columns re-gathers them for every column.
#
# Here each (row, region) walk keeps a small cache of S values per row in shared memory (slots):
#
#   X[s, row]   s = 1:S,   stored as X[(s - 1) R + row], R = rows per block
#
# The rows of a thread are private (no other thread reads them), so there is no synchronization. The
# walk of a region is the same for every row, so which column sits in which slot is decided once, on
# the host, by simulating the walk with Belady's replacement (keep the values read again soonest).
# The kernel then follows a plan of chunks: a chunk computes ≤ SLOT_W residual columns of one front,
#
#   x[j] ← x[j] ⊕ ⊕ᵢ vᵢ ⊗ c[i, j]          vᵢ: its inputs, each from a slot or from C
#   x ← x (L₁₁)* (in registers),  C[t, col₀ + j - 1] ← x[j],  and to a slot if read again
#
# The inputs of a chunk of front f with columns j₀:j₁ are its separator values (c = L₂₁) and the
# residual columns j₁+1:nn of f (c = L₁₁, from the chunks of f before it: wide fronts are cut into
# chunks of SLOT_W columns, from the last one, so the separator is read once per chunk instead of
# once per column). The coefficients are gathered from the factor into plan order on every sweep
# (one small gather), so that a chunk reads them sequentially. Every value is still written to C as
# soon as it is final, so the slots are only a read cache, and a column that does not fit is read
# from C. The ⊕ of a chunk visits its terms in another order than downward_front_reg! (cached
# inputs first): exact for idempotent ⊕ (min-plus, max-plus, max-min), a rounding change otherwise.
#
# Global traffic per row: a column of the region is read from C only when it is not in a slot; with
# 32–64 slots this is close to its first read (grid2d-180: 20.5 → 5.5 GB per closure, of which
# 3.9 GB are the writes of the result). Instructions: a cached input is one shared load per row, and
# an input's coefficients and plan entry are uniform loads shared by the Q rows of a thread.
#
const SLOT_W = 8                     # residual columns per chunk (register-blocked)
const SLOT_HDR = 8                   # header words per chunk
const SLOT_TB = 128                  # threads per block
const SLOT_Q = 1                     # rows per thread (rows t, t + SLOT_TB, ...)
const SLOT_MAX = 64                  # at most this many slots per row (more gains < 3% traffic on 2D/3D meshes)
const SLOT_MAXREGS = 64

struct SlotPlan{T}
    layers::Vector{Tuple{CuVector{Int32}, Int}}    # top-down: (first chunk of each region and one past the last, number of regions)
    hdr::CuVector{Int32}            # per chunk: col₀, width, cached inputs, uncached inputs, entry pointer, coefficient pointer, fd[f], f
    ent::CuVector{Int32}            # per chunk: slot offsets of the cached inputs; (column, slot offset or -1) of the others; slot offset or -1 of each output
    cmap::CuVector{Int64}           # source of each coefficient: > 0 an index into LLval, < 0 minus an index into LDval
    coef::CuVector{T}               # the coefficients in plan order (gathered by slot_coefficients!)
    slots::Int                      # S
    rows::Int                       # R: rows per block, the stride between slots
    sread::Int                      # inputs per row read from a slot (statistics)
    gread::Int                      # inputs per row read from C
end

# slots per row: as many as keep 4 blocks resident per SM, at most SLOT_MAX
function slot_count(::Type{T}, R::Int) where {T}
    S = config().layer_slots
    S > 0 && return S
    p = device_profile()
    return clamp((p.shmem_sm - 4 * 1024) ÷ (4 * R * sizeof(T)), 1, SLOT_MAX)
end

function slot_plan(G::GPUSLU{<:Any, T}, S::Int, R::Int) where {T}
    lp = layer_plan(G)
    get!(G.cache, Symbol(:slots, lp.m, :_, S, :_, R)) do
        build_slot_plan(G, lp, S, R)
    end
end

function build_slot_plan(G::GPUSLU{<:Any, T}, lp::LayerPlan, S::Int, R::Int) where {T}
    Rptr = G.hRptr; Sptr = G.hSptr; Dptr = G.hDptr; Lptr = G.hLptr
    Stgt = Array(G.Stgt); fd = Array(first_descendants(G))
    hdr = Int32[]; ent = Int32[]; cmap = Int64[]
    layers = Tuple{CuVector{Int32}, Int}[]
    nxt = zeros(Int, G.n)                       # reverse scan: the next chunk that reads a column (0: none)
    cslot = zeros(Int, G.n)                     # slot holding a column (0: none)
    scol = zeros(Int, S); snext = zeros(Int, S) # column in each slot (0: free) and its next read
    sread = 0; gread = 0
    # the chunks of a region: front, first and last column; and their inputs, flattened
    chf = Int[]; chj0 = Int[]; chj1 = Int[]; inptr = Int[]; incol = Int[]; inco = Int[]; innext = Int[]; outnext = Int[]
    hits = Int[]; miss = Int[]
    #
    # inputs of chunk (f, j₀, j₁): the separator, then the residual columns j₁+1:nn of f; inco is the
    # position of the first coefficient (> 0: LLval, < 0: LDval), the next ones are a stride apart
    #
    function chunk_inputs!(f, j0, j1)
        Rp = Rptr[f]; nn = Rptr[f + 1] - Rp; Sp = Sptr[f]; na = Sptr[f + 1] - Sp

        for r in 1:na
            push!(incol, Stgt[Sp + r - 1]); push!(inco, Lptr[f] + (j0 - 1) * na + r - 1)
        end

        for k in (j1 + 1):nn
            push!(incol, Rp + k - 1); push!(inco, -(Dptr[f] + (j0 - 1) * nn + k - 1))
        end
    end
    # the coefficients of input i of chunk c, for its columns j₀:j₁
    function push_coefs!(c, i)
        f = chf[c]; w = chj1[c] - chj0[c] + 1
        stride = inco[i] > 0 ? Sptr[f + 1] - Sptr[f] : -(Rptr[f + 1] - Rptr[f])
        for j in 1:w
            push!(cmap, inco[i] + (j - 1) * stride)
        end
    end
    #
    # a slot for a value read next at chunk `next` (0: never): a free one, else the one whose value is
    # read latest, if later than this one (Belady); 0 when the value is not kept
    #
    function take_slot!(col, next)
        iszero(next) && return 0
        best = 0; far = next

        for s in 1:S
            if iszero(scol[s])
                best = s; break
            elseif snext[s] > far
                best = s; far = snext[s]
            end
        end

        if ispositive(best)
            iszero(scol[best]) || (cslot[scol[best]] = 0)
            scol[best] = col; snext[best] = next; cslot[col] = best
        end

        return best
    end

    for (regptr, regfronts, nreg) in lp.layers
        rp = Array(regptr); rf = Array(regfronts)
        rchunk = Int32[length(hdr) ÷ SLOT_HDR + 1]

        for g in 1:nreg
            empty!(chf); empty!(chj0); empty!(chj1); empty!(inptr); empty!(incol); empty!(inco); empty!(innext); empty!(outnext)
            push!(inptr, 1)

            for i in rp[g]:(rp[g + 1] - 1)
                f = rf[i]; j1 = Rptr[f + 1] - Rptr[f]

                while j1 >= 1                       # chunks from the last columns (the solve with L₁₁ runs backward)
                    j0 = max(1, j1 - SLOT_W + 1)
                    push!(chf, f); push!(chj0, j0); push!(chj1, j1)
                    chunk_inputs!(f, j0, j1)
                    push!(inptr, length(incol) + 1)
                    j1 = j0 - 1
                end
            end

            nc = length(chf)
            resize!(innext, length(incol))
            #
            # next reads, scanning backward: at chunk c, nxt[col] is the first chunk after c that reads col
            #
            outptr = cumsum([1; [chj1[c] - chj0[c] + 1 for c in 1:nc]])
            resize!(outnext, outptr[end] - 1)

            for c in nc:-1:1
                Rp = Rptr[chf[c]]

                for j in chj0[c]:chj1[c]
                    outnext[outptr[c] + j - chj0[c]] = nxt[Rp + j - 1]
                end

                for i in inptr[c]:(inptr[c + 1] - 1)
                    innext[i] = nxt[incol[i]]; nxt[incol[i]] = c
                end
            end

            for col in incol
                nxt[col] = 0
            end
            #
            # forward: cached inputs are read from their slots, the others from C (and kept if read again
            # soon enough), the outputs written to C (and kept likewise)
            #
            for c in 1:nc
                f = chf[c]; j0 = chj0[c]; j1 = chj1[c]; w = j1 - j0 + 1
                empty!(hits); empty!(miss)

                for i in inptr[c]:(inptr[c + 1] - 1)
                    push!(ispositive(cslot[incol[i]]) ? hits : miss, i)
                end

                append!(hdr, Int32.((Rptr[f] + j0 - 1, w, length(hits), length(miss), length(ent) + 1, length(cmap) + 1, fd[f], f)))

                for i in hits
                    s = cslot[incol[i]]
                    push!(ent, (s - 1) * R)
                    push_coefs!(c, i)
                    snext[s] = innext[i]

                    if iszero(innext[i])            # last read: free the slot
                        scol[s] = 0; cslot[incol[i]] = 0
                    end
                end

                for i in miss
                    s = take_slot!(incol[i], innext[i])
                    push!(ent, incol[i], ispositive(s) ? (s - 1) * R : -1)
                    push_coefs!(c, i)
                end

                Dp = Dptr[f]; nn = Rptr[f + 1] - Rptr[f]

                for j in j1:-1:j0, k in (j + 1):j1  # the solve with L₁₁ within the chunk, in down_reg!'s order
                    push!(cmap, -(Dp + (j - 1) * nn + k - 1))
                end

                for j in j0:j1
                    s = take_slot!(Rptr[f] + j - 1, outnext[outptr[c] + j - j0])
                    push!(ent, ispositive(s) ? (s - 1) * R : -1)
                end

                sread += length(hits); gread += length(miss)
            end

            for s in 1:S                            # the next region starts with an empty cache
                iszero(scol[s]) || (cslot[scol[s]] = 0)
                scol[s] = 0
            end

            push!(rchunk, length(hdr) ÷ SLOT_HDR + 1)
        end

        push!(layers, (upload(rchunk), nreg))
    end

    max(length(ent), length(cmap)) < typemax(Int32) || return nothing
    return SlotPlan{T}(layers, upload(hdr), upload(ent), upload(cmap), CuVector{T}(undef, length(cmap)), S, R, sread, gread)
end

function layered_down_coef_kernel!(coef, cmap, LLval, LDval)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds while i <= length(coef)
        k = cmap[i]
        coef[i] = k > 0 ? LLval[k] : LDval[-k]
        i += blockDim().x * gridDim().x
    end

    return
end

function slot_coefficients!(P::SlotPlan, G::GPUSLU)
    n = length(P.coef)
    ispositive(n) && @cuda threads = 256 blocks = min(cld(n, 256), 4096) layered_down_coef_kernel!(P.coef, P.cmap, G.LLval, G.LDval)
    return P
end

#
# The chunks of region blockIdx().y for the Q rows t₀ + tx + (k - 1) blockDim().x of this thread.
#
function layered_down_slot_kernel!(s::AbstractSemiring, trans::Val, C::AbstractMatrix{T}, ::Val{Q}, nslot::Int32, rchunk, hdr, ent, coef,
        ::Val{SKIP}, sources, cinvp, idx) where {T, Q, SKIP}
    nt = blockDim().x % Int32
    tx = threadIdx().x % Int32
    R = nt * Int32(Q)
    X = CuDynamicSharedArray(T, nslot * R)
    g = blockIdx().y
    t0 = (blockIdx().x - 1) * R
    m = size(C, 1)
    rows = ntuple(k -> t0 + tx + (k - 1) * nt, Val(Q))
    lis = ntuple(k -> tx + Int32(k - 1) * nt, Val(Q))
    valid = ntuple(k -> rows[k] <= m, Val(Q))
    fts = ntuple(k -> SKIP && valid[k] ? (@inbounds idx[cinvp[sources[rows[k]]]] % Int32) : Int32(0), Val(Q))
    hdr = CUDA.CUDACore.Const(hdr); ent = CUDA.CUDACore.Const(ent); coef = CUDA.CUDACore.Const(coef)

    @inbounds for c in rchunk[g]:(rchunk[g + 1] - Int32(1))
        h = (c - Int32(1)) * Int32(SLOT_HDR)
        col0 = hdr[h + 1]; w = hdr[h + 2]; ns = hdr[h + 3]; ng = hdr[h + 4]; ep = hdr[h + 5]; cp = hdr[h + 6]
        fdf = hdr[h + 7]; f = hdr[h + 8]
        zis = ntuple(k -> SKIP && !(fdf <= fts[k] <= f), Val(Q))      # see skip_fill in sgetrs.jl
        args = (s, trans, C, X, ent, coef, rows, lis, valid, zis, col0, ns, ng, ep, cp)

        if w == 1
            slot_chunk!(args..., Val(1))
        elseif w == 2
            slot_chunk!(args..., Val(2))
        elseif w <= 4
            w == 3 ? slot_chunk!(args..., Val(3)) : slot_chunk!(args..., Val(4))
        else
            w == 5 ? slot_chunk!(args..., Val(5)) :
            w == 6 ? slot_chunk!(args..., Val(6)) :
            w == 7 ? slot_chunk!(args..., Val(7)) :
                     slot_chunk!(args..., Val(8))
        end
    end

    return
end

@generated function slot_chunk!(s, trans, C, X, ent, coef, rows::NTuple{Q}, lis, valid, zis, col0, ns, ng, ep, cp, ::Val{W}) where {Q, W}
    x(j, k) = Symbol(:x, j, :_, k)
    v(k) = Symbol(:v, k)
    init = [:($(x(j, k)) = zis[$k] | !valid[$k] ? z : C[rows[$k], col0 + $(j - 1)]) for j in 1:W for k in 1:Q]
    loadc = [:($(Symbol(:l, j)) = coef[p + $(j - 1)]) for j in 1:W]
    upd = [:($(x(j, k)) = smuladd(s, $(v(k)), $(Symbol(:l, j)), $(x(j, k)), Val(:N), trans)) for j in 1:W for k in 1:Q]
    sload = [:($(v(k)) = X[so + lis[$k]]) for k in 1:Q]
    gload = [:($(v(k)) = valid[$k] ? C[rows[$k], col] : z) for k in 1:Q]
    gkeep = [:(X[st + lis[$k]] = $(v(k))) for k in 1:Q]
    solve = Expr[]

    for j in W:-1:1, k in (j + 1):W
        push!(solve, :(d = coef[p]; p += Int32(1)))
        append!(solve, [:($(x(j, q)) = smuladd(s, $(x(k, q)), d, $(x(j, q)), Val(:N), trans)) for q in 1:Q])
    end

    stores = Expr[]

    for j in 1:W
        push!(stores, :(os = ent[e + $(j - 1)]))
        append!(stores, [:(valid[$k] && (C[rows[$k], col0 + $(j - 1)] = $(x(j, k)))) for k in 1:Q])
        push!(stores, :(if os >= 0; $([:(X[os + lis[$k]] = $(x(j, k))) for k in 1:Q]...); end))
    end

    return quote
        $(Expr(:meta, :inline))
        z = szero(s, eltype(C), trans)

        @inbounds begin
            $(init...)
            e = ep; p = cp

            for _ in Int32(1):ns
                so = ent[e]; e += Int32(1)
                $(sload...)
                $(loadc...)
                $(upd...)
                p += Int32($W)
            end

            for _ in Int32(1):ng
                col = ent[e]; st = ent[e + Int32(1)]; e += Int32(2)
                $(gload...)
                if st >= 0
                    $(gkeep...)
                end
                $(loadc...)
                $(upd...)
                p += Int32($W)
            end

            $(solve...)
            $(stores...)
        end

        return
    end
end

function layered_slot_sweep!(G::GPUSLU, W::CuMatrix{T}, P::SlotPlan, trans::Val, timer, zr) where {T}
    s = G.s
    zargs = isnothing(zr) ? (Val(false), nothing, nothing, nothing) : (Val(true), zr, G.cinvp, G.idx)
    @phase timer :L_layered slot_coefficients!(P, G)
    nb = cld(size(W, 1), P.rows)
    shmem = P.slots * P.rows * sizeof(T)

    for (rchunk, nreg) in P.layers
        @phase timer :L_layered @cuda threads = SLOT_TB blocks = (nb, nreg) shmem = shmem maxregs = SLOT_MAXREGS layered_down_slot_kernel!(s, trans, W, Val(SLOT_Q), Int32(P.slots),
            rchunk, P.hdr, P.ent, P.coef, zargs...)
    end

    return W
end
