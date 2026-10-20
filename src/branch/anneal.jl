# ----------------------------------------------------------------------------------------
# simulated annealing
# ----------------------------------------------------------------------------------------
#
# Simulated annealing of a whole contraction tree by tree rotations (Kalachev, Panteleev,
# Yung 2021; TreeSA in OMEinsumContractionOrders.jl). An internal node P with children X
# and Y = (B, C) becomes P = (B, Y'), Y' = (X, C). Only the legs of Y and the contractions
# at Y and P change:
#
#     L(Y')   = (L(X) | L(C)) & (L(B) | L(P))
#     inv(Y') =  L(X) | L(C)
#     inv(P') =  L(B) | L(Y')          w(inv(P')) = w(L(B)) + w(L(Y') \ L(B))
#
# so a proposal is a few word operations on bitset legs. The legs are stored in one of two
# ways (chosen by the number of words of a bitset):
#
#   - AnnealDense: a word matrix, one column per node (fast for small networks; the memory
#     is nodes x words);
#   - AnnealSparse: per node, the sorted list of its nonzero words (word index, bits) in an
#     arena, so that a proposal costs O(words in the four lists) and the memory is O(sum of
#     the leg sizes). The vertices are numbered in the order they first occur in a preorder
#     of the initial tree, so the legs of a subtree tend to share words.
#
# Each class of equal weights starts on a word boundary, so w(word) = its class weight times
# its popcount; with more than ANNEAL_CLASSES classes, set bits are summed one by one.
#
# A proposal changes the energy by
#
#     dE = (1 - mix) * log2(C' / C) + mix * (lse(tc'(Y), tc'(P)) - lse(tc(Y), tc(P)))
#        + lambda * (max(0, w(L(Y')) - W) - max(0, w(L(Y)) - W))
#
# where C is the total cost (sum over contractions of 2^tc) and lse(a, b) = log2(2^a + 2^b):
# mix = 0 is the true objective; mix = 1 is TreeSA's local criterion, which also rewards
# improvements to cheap subtrees (the true objective cannot see them, so they drift). The
# tree returned is the cheapest one of width at most W seen at the end of a sweep (or, if
# there is none, the narrowest).
#
# The run is one or more annealing cycles, each from the best tree so far. By default, a
# cold cycle refines the initial tree under the penalty, and a hotter one then walks
# without it (the cap only selects the trees kept): a hot start alone can lose a good
# initial tree (under the penalty, the walk can settle among trees of the same width that
# are worse than it), a cold start alone cannot leave a poor one, and some cheaper trees
# within the cap are reached only through wider ones.
#
const ANNEAL_CLASSES = 8
const ANNEAL_DENSE = 48     # dense legs for at most this many words (and a small matrix)
const ANNEAL_TABLE = 64     # entries per unit of the table of log2(1 + 2^-d)
const ANNEAL_RANGE = 48     # beyond this, log2(1 + 2^-d) is 0

# dense legs: column x of L is L(x); aligned class k fills words kptr[k]:kptr[k + 1] - 1
struct AnnealDense
    L::FMatrix{UInt64}
    kptr::FVector{Int}
    kwt::FVector{Float64}
    wbit::FVector{Float64}
    nk::Int                     # number of classes (0: not aligned, use wbit)
    nw::Int
end

# sparse legs: L(x) is the list aw[i], ab[i] for i in lptr[x]:lptr[x] + llen[x] - 1 (room for
# lcap[x]) in an arena of which atop entries are used, live of them by lists
mutable struct AnnealSparse
    aw::Vector{Int32}
    ab::Vector{UInt64}
    aw2::Vector{Int32}
    ab2::Vector{UInt64}
    lptr::FVector{Int}
    llen::FVector{Int}
    lcap::FVector{Int}
    wword::FVector{Float64}
    wbit::FVector{Float64}
    mb::FVector{UInt64}         # zeroed scratch words
    mp::FVector{UInt64}
    aligned::Bool
    atop::Int
    live::Int
end

"""
    branchanneal(weights, hypergraph, point, parent; sweeps=nothing, beta0=(5.0, 1.0),
        beta1=30.0, mix=1.0, W=nothing, lambda=nothing, time=Inf, refine=true, seed=1,
        info=false)

Simulated annealing of a contraction tree by tree rotations, as in TreeSA (see the comment at
the top of `anneal.jl`). Each sweep proposes one rotation at every internal node. The run is
one annealing cycle per entry of `beta0`, each with an equal share of the sweeps (or of the
time) and starting from the best tree so far; in a cycle, the inverse temperature grows
geometrically from its `beta0` to `beta1`. Returns a parent vector (same convention as
`branchtree`: children are numbered before their parents): the cheapest tree of width at
most `W` seen at the end of a sweep, or, if there was none, the narrowest tree seen, then
improved by `branchrefine` (never wider). Widths are those of the intermediates, as in
`branchstats` (the input tensors are not counted).

  - `sweeps`: number of sweeps (by default 1000, or no limit if `time` is finite)
  - `beta0`: initial inverse temperature (per unit of log2 cost) of each cycle, a number or
    a tuple; a small one forgets the tree, a large one refines it. The default refines the
    initial tree, then searches from the result
  - `beta1`: final inverse temperature
  - `mix`: weight of TreeSA's local acceptance criterion against the total cost (0 ≤ mix ≤ 1)
  - `W`: width cap (by default the width of `parent`, so the tree returned is never wider;
    `Inf` for none)
  - `lambda`: penalty per unit of width above `W`, which steers the walk towards trees
    within the cap: a number, or one per cycle. By default 1, except 0 in the last of
    several cycles once a tree within the cap is known: that cycle walks freely and keeps
    only the trees within the cap
  - `time`: time limit of the annealing in seconds; if finite, the schedule follows the
    elapsed time (whichever of `sweeps` and `time` runs out first ends the run)
  - `refine`: finish with `branchrefine` (it runs after the time limit)
  - `seed`: random seed
  - `info`: if true, return `(parent, info)`, where `info` is a named tuple with the number
    of `sweeps`, the `time`, the numbers of `proposals` and `accepted` rotations, the
    `cost` and `width` of the tree returned, and whether it is `feasible` (within `W`)
  - `dense`: store the legs as a dense word matrix (`true`) or as sparse word lists
    (`false`); by default dense for at most $(ANNEAL_DENSE) words per leg set
"""
function branchanneal(
        weights::AbstractVector,
        hypergraph::BipartiteGraph,
        point::Integer,
        parent::AbstractVector{<:Integer};
        sweeps::Union{Nothing, Integer} = nothing,
        beta0::Union{Real, Tuple{Vararg{Real}}} = (5.0, 1.0),
        beta1::Real = 30.0,
        mix::Real = 1.0,
        W = nothing,
        lambda::Union{Nothing, Real, Tuple{Vararg{Real}}} = nothing,
        time::Real = Inf,
        refine::Bool = true,
        seed::Integer = 1,
        info::Bool = false,
        dense::Union{Nothing, Bool} = nothing,
    )
    t0 = Base.time()
    H = simplehypergraph(hypergraph)
    m = nv(H); n = nov(H); nn = length(parent)
    @assert nn == 2m - 2 && parent[point] == 0
    @assert length(weights) == n
    @assert 0 <= mix <= 1
    @assert all(>(0), beta0) && 0 < beta1 && !isempty(beta0)
    @assert isnothing(lambda) || all(>=(0), lambda)
    @assert !(lambda isa Tuple) || length(lambda) == length(beta0) "one lambda per cycle"

    if isnothing(sweeps)
        nsweep = isfinite(time) ? typemax(Int) : 1000
    else
        nsweep = Int(sweeps)
    end

    if m <= 3
        result = collect(Int, parent)
        stats = branchstats(weights, H, point, result)
        Wcap = isnothing(W) ? stats.width : Float64(W)
        res = (; sweeps = 0, time = Base.time() - t0, proposals = 0, accepted = 0,
            cost = stats.cost, width = stats.width, feasible = stats.width <= Wcap + EPS)
        return info ? (result, res) : result
    end

    # weight classes (sorted) and the number of vertices in each
    deg  = FVector{Int}(undef, n)
    bit  = FVector{Int}(undef, n)
    cls  = FVector{Float64}(undef, n)
    ccnt = FVector{Int}(undef, n)
    fill!(deg, 0)

    @inbounds for i in arcs(H)
        deg[targets(H)[i]] += 1
    end

    nc = 0

    @inbounds for v in oneto(n)
        if deg[v] > 1 && weights[v] > 0
            nc += 1; cls[nc] = weights[v]
        end
    end

    sort!(view(cls, oneto(nc)))
    nk = 0

    @inbounds for i in oneto(nc)
        if nk == 0 || cls[i] != cls[nk]
            nk += 1; cls[nk] = cls[i]; ccnt[nk] = 0
        end

        ccnt[nk] += 1
    end

    aligned = nk <= ANNEAL_CLASSES

    # first bit of each class (on a word boundary if aligned)
    cbase = FVector{Int}(undef, nk + 1)
    nb = 0

    @inbounds for k in oneto(nk)
        if aligned
            nb = 64 * cld(nb, 64)
        end

        cbase[k] = nb; nb += ccnt[k]
    end

    @inbounds cbase[nk + 1] = nb
    nw = max(cld(nb, 64), 1)

    if isnothing(dense)
        dense = nw <= ANNEAL_DENSE && nw * nn <= 1 << 22
    end

    # table of log2(1 + 2^-d) for 0 ≤ d ≤ ANNEAL_RANGE
    ntab = ANNEAL_RANGE * ANNEAL_TABLE + 2
    gtab = FVector{Float64}(undef, ntab)

    @inbounds for i in oneto(ntab)
        gtab[i] = log2(1.0 + exp2(-(i - 1) / ANNEAL_TABLE))
    end

    wword  = FVector{Float64}(undef, nw)
    wbit   = FVector{Float64}(undef, 64nw)
    mask   = FVector{UInt64}(undef, nw)
    touch  = FVector{Int32}(undef, nw)
    nyw    = FVector{Int32}(undef, nw + 1)
    nyb    = FVector{UInt64}(undef, nw + 1)
    left   = FVector{Int}(undef, nn)
    right  = FVector{Int}(undef, nn)
    par    = FVector{Int}(undef, nn)
    tc     = FVector{Float64}(undef, nn)
    sc     = FVector{Float64}(undef, nn)
    best   = FVector{Int}(undef, nn)
    pre    = FVector{Int}(undef, nn)
    preidx = FVector{Int}(undef, nn)
    stack  = FVector{Int}(undef, nn)
    uf     = FVector{Int}(undef, nn)
    head   = FVector{Int}(undef, nn)
    prev   = FVector{Int}(undef, n)
    lca    = FVector{Int}(undef, n)
    next   = FVector{Int}(undef, n)
    betas  = FVector{Float64}(undef, length(beta0))
    lams   = FVector{Float64}(undef, length(beta0))

    @inbounds for x in oneto(nn)
        par[x] = parent[x]
    end

    # by default, a penalty of 1, except in the last of several cycles if the best tree so
    # far is within the cap: that cycle walks freely (the cap only selects the trees kept)
    freelast = isnothing(lambda) && length(betas) > 1

    @inbounds for i in eachindex(betas)
        betas[i] = beta0[i]
        lams[i] = isnothing(lambda) ? 1.0 : lambda[min(i, length(lambda))]
    end

    # the bit positions, from a preorder of the initial tree
    anneal_tree!(left, right, pre, preidx, stack, par, m, Int(point))
    anneal_label!(bit, next, wword, wbit, pre, deg, cls, cbase, weights, H, m, nn, nk, aligned)

    if isnothing(W)
        Wcap = NaN      # the width of parent
    else
        Wcap = Float64(W)
    end

    # the two branches call the same code on the two kinds of legs (a function barrier)
    if dense
        nkd = aligned ? nk : 0
        kptr = FVector{Int}(undef, nkd + 1)
        kwt  = FVector{Float64}(undef, max(nkd, 1))

        @inbounds kptr[1] = 1

        @inbounds for k in oneto(nkd)
            kwt[k] = cls[k]; kptr[k + 1] = cbase[k + 1] >> 6 + 1
        end

        @inbounds kptr[nkd + 1] = nw + 1
        dlegs = AnnealDense(FMatrix{UInt64}(undef, nw, nn), kptr, kwt, wbit, nkd, nw)

        res = branchanneal_cycles!(dlegs, gtab, mask, touch, nyw, nyb, left, right, par, tc, sc,
            best, pre, preidx, stack, uf, head, prev, lca, next, bit, H, m, Int(point), betas,
            lams, freelast, nsweep, Float64(beta1), Float64(mix), Wcap, Float64(time), t0,
            Xoshiro(seed))
    else
        lptr = FVector{Int}(undef, nn)
        llen = FVector{Int}(undef, nn)
        lcap = FVector{Int}(undef, nn)
        mb   = FVector{UInt64}(undef, nw)
        mp   = FVector{UInt64}(undef, nw)
        fill!(mb, zero(UInt64))
        fill!(mp, zero(UInt64))
        slegs = AnnealSparse(Vector{Int32}(undef, 4nn + 64), Vector{UInt64}(undef, 4nn + 64),
            Int32[], UInt64[], lptr, llen, lcap, wword, wbit, mb, mp, aligned, 0, 0)

        res = branchanneal_cycles!(slegs, gtab, mask, touch, nyw, nyb, left, right, par, tc, sc,
            best, pre, preidx, stack, uf, head, prev, lca, next, bit, H, m, Int(point), betas,
            lams, freelast, nsweep, Float64(beta1), Float64(mix), Wcap, Float64(time), t0,
            Xoshiro(seed))
    end

    result = convert(Vector{Int}, best)

    if refine
        result = branchrefine(weights, H, point, result)
    end

    if info
        nswp, nacc, Wcap = res
        stats = branchstats(weights, H, point, result)
        res = (; sweeps = nswp, time = Base.time() - t0, proposals = nswp * (m - 2),
            accepted = nacc, cost = stats.cost, width = stats.width,
            feasible = stats.width <= Wcap + EPS)
        return result, res
    else
        return result
    end
end

# One annealing cycle per entry of betas, each from the best tree so far (in par), with an
# equal share of the remaining sweeps and time. Returns (sweeps, accepted, cap).
function branchanneal_cycles!(
        legs::Union{AnnealDense, AnnealSparse},
        gtab::AbstractVector{Float64},
        mask::AbstractVector{UInt64},
        touch::AbstractVector{Int32},
        nyw::AbstractVector{Int32},
        nyb::AbstractVector{UInt64},
        left::AbstractVector{Int},
        right::AbstractVector{Int},
        par::AbstractVector{Int},
        tc::AbstractVector{Float64},
        sc::AbstractVector{Float64},
        best::AbstractVector{Int},
        pre::AbstractVector{Int},
        preidx::AbstractVector{Int},
        stack::AbstractVector{Int},
        uf::AbstractVector{Int},
        head::AbstractVector{Int},
        prev::AbstractVector{Int},
        lca::AbstractVector{Int},
        next::AbstractVector{Int},
        bit::AbstractVector{Int},
        H::BipartiteGraph,
        m::Int,
        point::Int,
        betas::AbstractVector{Float64},
        lams::AbstractVector{Float64},
        freelast::Bool,
        sweeps::Int,
        beta1::Float64,
        mix::Float64,
        Wcap::Float64,
        time::Float64,
        t0::Float64,
        rng::AbstractRNG,
    )
    nn = 2m - 2; nc = length(betas); total = 0; accepted = 0; feasible = false

    for c in oneto(nc)
        if c > 1
            @inbounds for x in oneto(nn)
                par[x] = best[x]
            end
        end

        # the tree, the vertices closed at each node, and the legs
        root = anneal_tree!(left, right, pre, preidx, stack, par, m, point)
        anneal_lca!(head, next, prev, lca, uf, pre, preidx, par, bit, H, m, nn)
        anneal_build!(legs, mask, touch, head, next, left, right, pre, bit, H, m, point)

        # this cycle's share of what is left
        k = nc - c + 1
        ns = sweeps == typemax(Int) ? sweeps : (sweeps - total) ÷ k
        tc0 = Base.time(); tm = (time - (tc0 - t0)) / k

        lam = freelast && c == nc && feasible ? 0.0 : @inbounds(lams[c])

        ns1, acc, Wcap, feasible = branchanneal_impl!(legs, gtab, nyw, nyb, left, right, par,
            tc, sc, best, pre, preidx, stack, root, m, point, ns, @inbounds(betas[c]), beta1,
            mix, Wcap, lam, tm, tc0, rng)

        total += ns1; accepted += acc
    end

    return total, accepted, Wcap
end

function branchanneal_impl!(
        legs::Union{AnnealDense, AnnealSparse},
        gtab::AbstractVector{Float64},
        nyw::AbstractVector{Int32},
        nyb::AbstractVector{UInt64},
        left::AbstractVector{Int},
        right::AbstractVector{Int},
        par::AbstractVector{Int},
        tc::AbstractVector{Float64},
        sc::AbstractVector{Float64},
        best::AbstractVector{Int},
        pre::AbstractVector{Int},
        preidx::AbstractVector{Int},
        stack::AbstractVector{Int},
        root::Int,
        m::Int,
        point::Int,
        sweeps::Int,
        beta0::Float64,
        beta1::Float64,
        mix::Float64,
        Wcap::Float64,
        lambda::Float64,
        time::Float64,
        t0::Float64,
        rng::AbstractRNG,
    )
    nn = 2m - 2; ni = nn - m

    # contraction exponents and leg weights
    width = 0.0

    @inbounds for x in oneto(nn)
        if x != point
            sc[x] = anneal_union(legs, x, x)
        end
    end

    @inbounds for x in m + 1:nn
        tc[x] = anneal_union(legs, left[x], right[x])
        width = max(width, sc[x])
    end

    if isnan(Wcap)
        Wcap = width
    end

    track = isfinite(Wcap) || legs isa AnnealSparse

    # the initial tree is the first candidate
    ref, C = anneal_total(tc, m, nn)
    bestE = ref + log2(C); bestX = max(0.0, width - Wcap)

    @inbounds for x in oneto(nn)
        best[x] = par[x]
    end

    lb = log(beta1 / beta0)
    sweep = 0; accepted = 0

    while sweep < sweeps
        f = sweep / max(sweeps - 1, 1)

        if isfinite(time)
            el = Base.time() - t0

            if el >= time
                break
            end

            f = max(f, el / time)
        end

        f = min(f, 1.0)
        beta = beta0 * exp(lb * f)
        sweep += 1

        @inbounds for P in m + 1:nn
            r = rand(rng, UInt64)
            ok, ref, C = anneal_propose!(legs, gtab, nyw, nyb, left, right, par, tc, sc, P, r,
                beta, mix, Wcap, lambda, track, ref, C, m, nn)
            accepted += ok
        end

        # the exact total (the running one loses precision when a dominant term leaves)
        ref, C = anneal_total(tc, m, nn)
        E = ref + log2(C)
        wd = 0.0

        if track
            @inbounds for x in m + 1:nn
                wd = max(wd, sc[x])
            end
        end

        # the cheapest tree within the cap, else the narrowest
        ex = max(0.0, wd - Wcap)

        if ex < bestX - EPS || (ex <= bestX + EPS && E < bestE - EPS)
            bestE = E; bestX = ex
            anneal_parent!(best, preidx, pre, stack, left, right, root, point, m, nn)
        end
    end

    return sweep, accepted, Wcap, bestX <= EPS
end

# left and right children from par; the preorder from the point (pre, preidx); returns the
# child of the point
function anneal_tree!(
        left::AbstractVector{Int},
        right::AbstractVector{Int},
        pre::AbstractVector{Int},
        preidx::AbstractVector{Int},
        stack::AbstractVector{Int},
        par::AbstractVector{Int},
        m::Int,
        point::Int,
    )
    nn = 2m - 2; root = 0

    @inbounds for x in oneto(nn)
        left[x] = 0; right[x] = 0
    end

    @inbounds for x in oneto(nn)
        p = par[x]

        if p != 0
            if p == point
                root = x
            end

            if left[p] == 0
                left[p] = x
            else
                right[p] = x
            end
        end
    end

    npre = 0; sp = 1

    @inbounds stack[1] = point

    @inbounds while sp > 0
        x = stack[sp]; sp -= 1
        npre += 1; pre[npre] = x; preidx[x] = npre

        if right[x] != 0
            sp += 1; stack[sp] = right[x]
        end

        if left[x] != 0
            sp += 1; stack[sp] = left[x]
        end
    end

    @assert npre == nn "the parent vector is not a tree rooted at the point"
    return root
end

# Bit positions: the vertices of each class in the order they first occur in the leaves,
# in preorder, so that the legs of a subtree tend to share words. Also the weight of the
# bits of each word (aligned) and of each bit.
function anneal_label!(
        bit::AbstractVector{Int},
        next::AbstractVector{Int},
        wword::AbstractVector{Float64},
        wbit::AbstractVector{Float64},
        pre::AbstractVector{Int},
        deg::AbstractVector{Int},
        cls::AbstractVector{Float64},
        cbase::AbstractVector{Int},
        weights::AbstractVector,
        H::BipartiteGraph,
        m::Int,
        nn::Int,
        nk::Int,
        aligned::Bool,
    )
    n = nov(H)

    # the next bit of each class
    @inbounds for k in oneto(nk)
        next[k] = cbase[k]
    end

    @inbounds for v in oneto(n)
        bit[v] = -1
    end

    @inbounds for i in oneto(nn)
        x = pre[i]

        if x <= m
            for v in neighbors(H, x)
                if bit[v] < 0 && deg[v] > 1 && weights[v] > 0
                    k = searchsortedfirst(view(cls, oneto(nk)), Float64(weights[v]))
                    bit[v] = next[k]; next[k] += 1
                end
            end
        end
    end

    fill!(wbit, 0.0)
    fill!(wword, 0.0)

    @inbounds for v in oneto(n)
        b = bit[v]

        if b >= 0
            wbit[b + 1] = weights[v]; wword[b >> 6 + 1] = weights[v]
        end
    end

    return bit
end

# The vertices closed at each internal node (whose occurrences all lie below it), as linked
# lists head/next: the LCA of each vertex's occurrences, by union-find as in branchstats.
# The vertices of the point never close.
function anneal_lca!(
        head::AbstractVector{Int},
        next::AbstractVector{Int},
        prev::AbstractVector{Int},
        lca::AbstractVector{Int},
        uf::AbstractVector{Int},
        pre::AbstractVector{Int},
        preidx::AbstractVector{Int},
        par::AbstractVector{Int},
        bit::AbstractVector{Int},
        H::BipartiteGraph,
        m::Int,
        nn::Int,
    )
    n = nov(H)

    @inbounds for x in oneto(nn)
        head[x] = 0; uf[x] = x
    end

    @inbounds for v in oneto(n)
        prev[v] = 0; lca[v] = 0
    end

    # children before parents
    @inbounds for i in nn:-1:1
        x = pre[i]

        if x <= m
            for v in neighbors(H, x)
                if bit[v] >= 0
                    if prev[v] != 0
                        q = anneal_find!(uf, prev[v])

                        if lca[v] == 0 || preidx[q] < preidx[lca[v]]
                            lca[v] = q
                        end
                    end

                    prev[v] = x
                end
            end
        end

        if par[x] != 0
            uf[x] = par[x]
        end
    end

    @inbounds for v in oneto(n)
        q = lca[v]

        if bit[v] >= 0 && q > m
            next[v] = head[q]; head[q] = v
        end
    end

    return head
end

function anneal_find!(uf::AbstractVector{Int}, x::Int)
    @inbounds while uf[x] != x
        uf[x] = uf[uf[x]]; x = uf[x]
    end

    return x
end

# The legs of every node, bottom-up: a leaf's are its vertices; an internal node's are its
# children's minus the vertices closed at it (marked in the zeroed scratch words mask).
function anneal_build!(
        legs::AnnealDense,
        mask::AbstractVector{UInt64},
        touch::AbstractVector{Int32},
        head::AbstractVector{Int},
        next::AbstractVector{Int},
        left::AbstractVector{Int},
        right::AbstractVector{Int},
        pre::AbstractVector{Int},
        bit::AbstractVector{Int},
        H::BipartiteGraph,
        m::Int,
        point::Int,
    )
    L = legs.L; nw = legs.nw; nn = 2m - 2

    @inbounds for i in nn:-1:1
        x = pre[i]

        if x == point
            continue
        elseif x <= m
            for w in oneto(nw)
                L[w, x] = zero(UInt64)
            end

            for v in neighbors(H, x)
                b = bit[v]

                if b >= 0
                    L[b >> 6 + 1, x] |= one(UInt64) << (b & 63)
                end
            end
        else
            l = left[x]; r = right[x]

            for w in oneto(nw)
                L[w, x] = L[w, l] | L[w, r]
            end

            v = head[x]

            while v != 0
                b = bit[v]; L[b >> 6 + 1, x] &= ~(one(UInt64) << (b & 63)); v = next[v]
            end
        end
    end

    return legs
end

function anneal_build!(
        legs::AnnealSparse,
        mask::AbstractVector{UInt64},
        touch::AbstractVector{Int32},
        head::AbstractVector{Int},
        next::AbstractVector{Int},
        left::AbstractVector{Int},
        right::AbstractVector{Int},
        pre::AbstractVector{Int},
        bit::AbstractVector{Int},
        H::BipartiteGraph,
        m::Int,
        point::Int,
    )
    aw = legs.aw; ab = legs.ab; lptr = legs.lptr; llen = legs.llen; lcap = legs.lcap
    nn = 2m - 2; atop = 0; live = 0
    fill!(mask, zero(UInt64))

    @inbounds for i in nn:-1:1
        x = pre[i]

        if x == point
            lptr[x] = atop + 1; llen[x] = 0; lcap[x] = 0
            continue
        end

        # room for the longest possible list
        if x <= m
            k = 0

            for v in neighbors(H, x)
                if bit[v] >= 0
                    k += 1
                end
            end
        else
            k = llen[left[x]] + llen[right[x]]
        end

        if atop + k > length(aw)
            resize!(aw, 2 * length(aw) + k); resize!(ab, length(aw))
        end

        p = atop; len = 0

        if x <= m
            nt = 0

            for v in neighbors(H, x)
                b = bit[v]

                if b >= 0
                    w = b >> 6 + 1

                    if iszero(mask[w])
                        nt += 1; touch[nt] = w
                    end

                    mask[w] |= one(UInt64) << (b & 63)
                end
            end

            sort!(view(touch, oneto(nt)))

            for j in oneto(nt)
                w = touch[j]
                len += 1; aw[p + len] = w; ab[p + len] = mask[w]; mask[w] = zero(UInt64)
            end
        else
            v = head[x]

            while v != 0
                b = bit[v]; mask[b >> 6 + 1] |= one(UInt64) << (b & 63); v = next[v]
            end

            l = left[x]; r = right[x]
            il = lptr[l]; el = il + llen[l]; ir = lptr[r]; er = ir + llen[r]

            while il < el || ir < er
                wl = il < el ? aw[il] : typemax(Int32)
                wr = ir < er ? aw[ir] : typemax(Int32)
                w = min(wl, wr); bits = zero(UInt64)

                if wl == w
                    bits |= ab[il]; il += 1
                end

                if wr == w
                    bits |= ab[ir]; ir += 1
                end

                bits &= ~mask[w]

                if !iszero(bits)
                    len += 1; aw[p + len] = w; ab[p + len] = bits
                end
            end

            v = head[x]

            while v != 0
                mask[bit[v] >> 6 + 1] = zero(UInt64); v = next[v]
            end
        end

        lptr[x] = p + 1; llen[x] = len; lcap[x] = k; atop += k; live += k
    end

    legs.atop = atop; legs.live = live
    return legs
end

# w(set) for the bits x of word w, one bit at a time
@inline function anneal_bits(x::UInt64, w::Integer, wbit::AbstractVector{Float64})
    s = 0.0; o = 64 * (Int(w) - 1)

    @inbounds while !iszero(x)
        s += wbit[o + trailing_zeros(x) + 1]; x &= x - one(UInt64)
    end

    return s
end

# w(L(x) | L(y))
function anneal_union(legs::AnnealDense, x::Int, y::Int)
    L = legs.L; s = 0.0

    @inbounds if ispositive(legs.nk)
        for k in oneto(legs.nk)
            c = 0

            for i in legs.kptr[k]:legs.kptr[k + 1] - 1
                c += count_ones(L[i, x] | L[i, y])
            end

            s += legs.kwt[k] * c
        end
    else
        for i in oneto(legs.nw)
            s += anneal_bits(L[i, x] | L[i, y], i, legs.wbit)
        end
    end

    return s
end

function anneal_union(legs::AnnealSparse, x::Int, y::Int)
    aw = legs.aw; ab = legs.ab; s = 0.0

    @inbounds begin
        ix = legs.lptr[x]; ex = ix + legs.llen[x]; iy = legs.lptr[y]; ey = iy + legs.llen[y]

        while ix < ex || iy < ey
            wx = ix < ex ? aw[ix] : typemax(Int32)
            wy = iy < ey ? aw[iy] : typemax(Int32)
            w = min(wx, wy); bits = zero(UInt64)

            if wx == w
                bits |= ab[ix]; ix += 1
            end

            if wy == w
                bits |= ab[iy]; iy += 1
            end

            if legs.aligned
                s += legs.wword[w] * count_ones(bits)
            else
                s += anneal_bits(bits, w, legs.wbit)
            end
        end
    end

    return s
end

# the total cost as (ref, C): sum over contractions of 2^tc = C * 2^ref
function anneal_total(tc::AbstractVector{Float64}, m::Int, nn::Int)
    ref = -Inf

    @inbounds for x in m + 1:nn
        ref = max(ref, tc[x])
    end

    C = 0.0

    @inbounds for x in m + 1:nn
        C += exp2(tc[x] - ref)
    end

    return ref, C
end

# lse(a, b) = log2(2^a + 2^b), by linear interpolation in the table of log2(1 + 2^-d)
@inline function anneal_lse(a::Float64, b::Float64, gtab::AbstractVector{Float64})
    if a < b
        a, b = b, a
    end

    d = (a - b) * ANNEAL_TABLE

    if d >= ANNEAL_RANGE * ANNEAL_TABLE
        return a
    end

    i = unsafe_trunc(Int, d); f = d - i
    @inbounds g0 = gtab[i + 1]; @inbounds g1 = gtab[i + 2]
    return a + g0 + f * (g1 - g0)
end

# The weights of inv(Y') = L(X) | L(C), of L(Y') and of L(Y') \ L(B). Returns
# (ta, ty, tn, k); the sparse legs also put the k words of L(Y') in (nyw, nyb). The sparse
# legs are always tracked (their leg weights kept).
@inline function anneal_sums(
        legs::AnnealDense,
        nyw::AbstractVector{Int32},
        nyb::AbstractVector{UInt64},
        X::Int,
        B::Int,
        Cc::Int,
        P::Int,
        track::Bool,
    )
    L = legs.L; ta = 0.0; ty = 0.0; tn = 0.0

    # untracked (the leg weights are not kept): tn = w(inv(P')) and ty = 0
    @inbounds if ispositive(legs.nk)
        for k in oneto(legs.nk)
            ca = 0; cy = 0; cn = 0

            for i in legs.kptr[k]:legs.kptr[k + 1] - 1
                lb = L[i, B]; a = L[i, X] | L[i, Cc]; ny = a & (lb | L[i, P])
                ca += count_ones(a)

                if track
                    cy += count_ones(ny); cn += count_ones(ny & ~lb)
                else
                    cn += count_ones(ny | lb)
                end
            end

            c = legs.kwt[k]; ta += c * ca; ty += c * cy; tn += c * cn
        end
    else
        for i in oneto(legs.nw)
            lb = L[i, B]; a = L[i, X] | L[i, Cc]; ny = a & (lb | L[i, P])
            ta += anneal_bits(a, i, legs.wbit)

            if track
                ty += anneal_bits(ny, i, legs.wbit)
                tn += anneal_bits(ny & ~lb, i, legs.wbit)
            else
                tn += anneal_bits(ny | lb, i, legs.wbit)
            end
        end
    end

    return ta, ty, tn, 0
end

# Sparse: one pass over the words of L(X) | L(C) (L(Y') is a subset), merged without
# branches; L(B) and L(P) are scattered into the zeroed scratch words mb and mp first (and
# cleared after).
@inline function anneal_sums(
        legs::AnnealSparse,
        nyw::AbstractVector{Int32},
        nyb::AbstractVector{UInt64},
        X::Int,
        B::Int,
        Cc::Int,
        P::Int,
        track::Bool,
    )
    aw = legs.aw; ab = legs.ab; lptr = legs.lptr; llen = legs.llen; mb = legs.mb; mp = legs.mp
    ta = 0.0; ty = 0.0; tn = 0.0; k = 0

    @inbounds begin
        ib = lptr[B]; eb = ib + llen[B]; ip = lptr[P]; ep = ip + llen[P]

        for j in ib:eb - 1
            mb[aw[j]] = ab[j]
        end

        for j in ip:ep - 1
            mp[aw[j]] = ab[j]
        end

        ix = lptr[X]; ex = ix + llen[X]; ic = lptr[Cc]; ec = ic + llen[Cc]

        while ix < ex && ic < ec
            wx = aw[ix]; wc = aw[ic]; tx = wx <= wc; tcc = wc <= wx
            w = min(wx, wc)
            a = ifelse(tx, ab[ix], zero(UInt64)) | ifelse(tcc, ab[ic], zero(UInt64))
            ix += tx; ic += tcc
            ta, ty, tn, k = anneal_step(legs, w, a, nyw, nyb, ta, ty, tn, k)
        end

        while ix < ex
            ta, ty, tn, k = anneal_step(legs, aw[ix], ab[ix], nyw, nyb, ta, ty, tn, k)
            ix += 1
        end

        while ic < ec
            ta, ty, tn, k = anneal_step(legs, aw[ic], ab[ic], nyw, nyb, ta, ty, tn, k)
            ic += 1
        end

        for j in ib:eb - 1
            mb[aw[j]] = zero(UInt64)
        end

        for j in ip:ep - 1
            mp[aw[j]] = zero(UInt64)
        end
    end

    return ta, ty, tn, k
end

# one word w of L(X) | L(C), with bits a
@inline function anneal_step(
        legs::AnnealSparse,
        w::Int32,
        a::UInt64,
        nyw::AbstractVector{Int32},
        nyb::AbstractVector{UInt64},
        ta::Float64,
        ty::Float64,
        tn::Float64,
        k::Int,
    )
    @inbounds begin
        lb = legs.mb[w]; ny = a & (lb | legs.mp[w])

        if legs.aligned
            c = legs.wword[w]
            ta += c * count_ones(a); ty += c * count_ones(ny); tn += c * count_ones(ny & ~lb)
        else
            ta += anneal_bits(a, w, legs.wbit)
            ty += anneal_bits(ny, w, legs.wbit)
            tn += anneal_bits(ny & ~lb, w, legs.wbit)
        end

        nyw[k + 1] = w; nyb[k + 1] = ny; k += !iszero(ny)
    end

    return ta, ty, tn, k
end

# One proposal at the internal node P from the random word r; returns (accepted, ref, C).
@inline function anneal_propose!(
        legs::Union{AnnealDense, AnnealSparse},
        gtab::AbstractVector{Float64},
        nyw::AbstractVector{Int32},
        nyb::AbstractVector{UInt64},
        left::AbstractVector{Int},
        right::AbstractVector{Int},
        par::AbstractVector{Int},
        tc::AbstractVector{Float64},
        sc::AbstractVector{Float64},
        P::Int,
        r::UInt64,
        beta::Float64,
        mix::Float64,
        Wcap::Float64,
        lambda::Float64,
        track::Bool,
        ref::Float64,
        C::Float64,
        m::Int,
        nn::Int,
    )
    @inbounds begin
        # P = (X, Y), Y = (B, Cc): the side of Y and the grandchild B from two random bits
        sd = isodd(r); t = isodd(r >> 1)

        if sd
            Y = right[P]; X = left[P]
        else
            Y = left[P]; X = right[P]
        end

        if Y <= m
            X, Y = Y, X; sd = !sd

            if Y <= m
                return false, ref, C
            end
        end

        if t
            B = right[Y]; Cc = left[Y]
        else
            B = left[Y]; Cc = right[Y]
        end

        ta, ty, tn, k = anneal_sums(legs, nyw, nyb, X, B, Cc, P, track)
        tb = track ? sc[B] + tn : tn
        dE = 0.0; dC = 0.0

        if mix < 1.0
            dC = exp2(ta - ref) + exp2(tb - ref) - exp2(tc[Y] - ref) - exp2(tc[P] - ref)

            if C + dC > 0.0
                dE += (1.0 - mix) * log2((C + dC) / C)
            else
                dE = -Inf
            end
        end

        if mix > 0.0
            dE += mix * (anneal_lse(ta, tb, gtab) - anneal_lse(tc[Y], tc[P], gtab))
        end

        if track && lambda > 0.0
            dE += lambda * (max(0.0, ty - Wcap) - max(0.0, sc[Y] - Wcap))
        end

        if dE > 0.0 && (r >> 11) * 0x1.0p-53 >= exp(-beta * dE)
            return false, ref, C
        end

        # accept
        anneal_store!(legs, nyw, nyb, k, Y, X, B, Cc, P)

        if sd
            left[P] = B; right[P] = Y
        else
            right[P] = B; left[P] = Y
        end

        left[Y] = X; right[Y] = Cc; par[B] = P; par[X] = Y
        tc[Y] = ta; tc[P] = tb

        if track
            sc[Y] = ty
        end

        if mix < 1.0
            Cn = C + dC

            if !(Cn > 1e-3 * C) || Cn > 1e12
                ref, C = anneal_total(tc, m, nn)
            else
                C = Cn
            end
        end
    end

    return true, ref, C
end

# the new legs of Y: L(Y') = (L(X) | L(C)) & (L(B) | L(P))
@inline function anneal_store!(
        legs::AnnealDense,
        nyw::AbstractVector{Int32},
        nyb::AbstractVector{UInt64},
        k::Int,
        Y::Int,
        X::Int,
        B::Int,
        Cc::Int,
        P::Int,
    )
    L = legs.L

    @inbounds for i in oneto(legs.nw)
        L[i, Y] = (L[i, X] | L[i, Cc]) & (L[i, B] | L[i, P])
    end

    return legs
end

# Sparse: the k words in (nyw, nyb), in place if they fit, else at the top of the arena
# (compacted first when it is mostly garbage, else grown).
function anneal_store!(
        legs::AnnealSparse,
        nyw::AbstractVector{Int32},
        nyb::AbstractVector{UInt64},
        k::Int,
        Y::Int,
        X::Int,
        B::Int,
        Cc::Int,
        P::Int,
    )
    lptr = legs.lptr; lcap = legs.lcap

    @inbounds if k > lcap[Y]
        cap = k + k >> 1 + 2

        if legs.atop + cap > length(legs.aw)
            if 2legs.live < legs.atop
                anneal_compact!(legs)
            end

            if legs.atop + cap > length(legs.aw)
                resize!(legs.aw, 2 * length(legs.aw) + cap); resize!(legs.ab, length(legs.aw))
            end
        end

        legs.live += cap - lcap[Y]
        lptr[Y] = legs.atop + 1; lcap[Y] = cap; legs.atop += cap
    end

    aw = legs.aw; ab = legs.ab

    @inbounds begin
        p = lptr[Y] - 1

        for j in oneto(k)
            aw[p + j] = nyw[j]; ab[p + j] = nyb[j]
        end

        legs.llen[Y] = k
    end

    return legs
end

# Copy every list to the front of the arena (room: its length plus a quarter).
function anneal_compact!(legs::AnnealSparse)
    aw = legs.aw; ab = legs.ab; aw2 = legs.aw2; ab2 = legs.ab2
    lptr = legs.lptr; llen = legs.llen; lcap = legs.lcap; nn = length(lptr)
    total = 0

    @inbounds for x in oneto(nn)
        total += llen[x] + llen[x] >> 2 + 1
    end

    if length(aw2) < total
        resize!(aw2, total); resize!(ab2, total)
    end

    top = 0

    @inbounds for x in oneto(nn)
        p = lptr[x] - 1; len = llen[x]

        for j in oneto(len)
            aw2[top + j] = aw[p + j]; ab2[top + j] = ab[p + j]
        end

        cap = len + len >> 2 + 1
        lptr[x] = top + 1; lcap[x] = cap; top += cap
    end

    if length(aw) < top
        resize!(aw, top); resize!(ab, top)
    end

    copyto!(aw, 1, aw2, 1, top)
    copyto!(ab, 1, ab2, 1, top)
    legs.atop = top; legs.live = top
    return legs
end

# The parent vector of the current tree. Its internal nodes are renumbered in reverse
# preorder, so that children come before their parents, as in branchtree (the rotations
# move nodes between parents and break that order).
function anneal_parent!(
        best::AbstractVector{Int},
        newid::AbstractVector{Int},
        pre::AbstractVector{Int},
        stack::AbstractVector{Int},
        left::AbstractVector{Int},
        right::AbstractVector{Int},
        root::Int,
        point::Int,
        m::Int,
        nn::Int,
    )
    npre = 0; sp = 1

    @inbounds stack[1] = root

    @inbounds while sp > 0
        x = stack[sp]; sp -= 1

        if x > m
            npre += 1; pre[npre] = x
            sp += 1; stack[sp] = right[x]
            sp += 1; stack[sp] = left[x]
        end
    end

    t = m

    @inbounds for i in npre:-1:1
        t += 1; newid[pre[i]] = t
    end

    @inbounds for i in oneto(npre)
        x = pre[i]; y = newid[x]; l = left[x]; r = right[x]

        if l > m
            l = newid[l]
        end

        if r > m
            r = newid[r]
        end

        best[l] = y; best[r] = y
    end

    @inbounds best[newid[root]] = point
    @inbounds best[point] = 0
    return best
end
