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
#     inv(P') =  L(B) | L(Y')
#
# so a proposal costs O(words) on bitset legs. The vertices are renumbered so that each
# class of equal weights fills a range of words: w(set) = sum_k kwt[k] * popcount(set[range k]).
# With more than ANNEAL_CLASSES classes, set bits are summed one by one instead.
#
# A proposal changes the energy by
#
#     dE = (1 - mix) * log2(C' / C) + mix * (lse(tc'(Y), tc'(P)) - lse(tc(Y), tc(P)))
#        + lambda * (max(0, w(L(Y')) - W) - max(0, w(L(Y)) - W))
#
# where C is the total cost (sum over contractions of 2^tc) and lse(a, b) = log2(2^a + 2^b):
# mix = 0 is the true objective; mix = 1 is TreeSA's local criterion, which also rewards
# improvements to cheap subtrees (the true objective cannot see them, so they drift). The
# best tree whose width is at most W, seen at the end of a sweep, is returned.
#
const ANNEAL_CLASSES = 8
const ANNEAL_TABLE = 64     # entries per unit of the table of log2(1 + 2^-d)
const ANNEAL_RANGE = 48     # beyond this, log2(1 + 2^-d) is 0

"""
    branchanneal(weights, hypergraph, point, parent; sweeps=1000, beta0=1.0, beta1=30.0,
        mix=1.0, W=nothing, lambda=1.0, time=Inf, seed=1) -> parent

Simulated annealing of a contraction tree by tree rotations, as in TreeSA (see the comment at
the top of `anneal.jl`). Each sweep proposes one rotation at every internal node; the
inverse temperature grows geometrically from `beta0` to `beta1`. Returns the cheapest tree
of width at most `W` seen at the end of a sweep, as a parent vector (same convention as
`branchtree`).

  - `sweeps`: number of sweeps
  - `beta0`, `beta1`: initial and final inverse temperatures (per unit of log2 cost);
    a small `beta0` forgets the initial tree, a large one refines it
  - `mix`: weight of TreeSA's local acceptance criterion against the total cost (0 ≤ mix ≤ 1)
  - `W`: width cap (by default the width of `parent`; `Inf` for none), enforced as a
    penalty of `lambda` per unit of excess width; only trees within it are returned
  - `time`: time limit in seconds; if finite, the schedule follows the elapsed time
    (whichever of `sweeps` and `time` runs out first ends the run)
  - `seed`: random seed
"""
function branchanneal(
        weights::AbstractVector,
        hypergraph::BipartiteGraph,
        point::Integer,
        parent::AbstractVector{<:Integer};
        sweeps::Integer = 1000,
        beta0::Real = 1.0,
        beta1::Real = 30.0,
        mix::Real = 1.0,
        W = nothing,
        lambda::Real = 1.0,
        time::Real = Inf,
        seed::Integer = 1,
    )
    H = simplehypergraph(hypergraph)
    m = nv(H); n = nov(H); nn = length(parent)
    @assert nn == 2m - 2 && parent[point] == 0
    @assert length(weights) == n
    @assert 0 <= mix <= 1
    @assert 0 < beta0 && 0 < beta1

    if m <= 3
        return collect(Int, parent)
    end

    # bit positions: each weight class starts on a word boundary
    deg = FVector{Int}(undef, n)
    bit = FVector{Int}(undef, n)
    cls = FVector{Float64}(undef, n)
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
            nk += 1; cls[nk] = cls[i]
        end
    end

    aligned = nk <= ANNEAL_CLASSES
    nb = 0

    @inbounds for v in oneto(n)
        bit[v] = -1
    end

    @inbounds if aligned
        for k in oneto(nk)
            nb = 64 * cld(nb, 64)

            for v in oneto(n)
                if deg[v] > 1 && weights[v] == cls[k]
                    bit[v] = nb; nb += 1
                end
            end
        end
    else
        for v in oneto(n)
            if deg[v] > 1 && weights[v] > 0
                bit[v] = nb; nb += 1
            end
        end
    end

    nw = max(cld(nb, 64), 1)

    # aligned: class k fills words kptr[k]:kptr[k + 1] - 1, of weight kwt[k] per bit;
    # otherwise bit b weighs wbit[b + 1]
    if !aligned
        nk = 0
    end

    kptr = FVector{Int}(undef, nk + 1)
    kwt  = FVector{Float64}(undef, max(nk, 1))
    wbit = FVector{Float64}(undef, 64nw)
    fill!(wbit, 0.0)

    @inbounds for v in oneto(n)
        b = bit[v]

        if b >= 0
            wbit[b + 1] = weights[v]
        end
    end

    @inbounds if aligned
        kptr[1] = 1

        for k in oneto(nk)
            kwt[k] = cls[k]; i = kptr[k]

            while i <= nw && wbit[64i - 63] == cls[k]
                i += 1
            end

            kptr[k + 1] = i
        end
    end

    # table of log2(1 + 2^-d) for 0 ≤ d ≤ ANNEAL_RANGE
    ntab = ANNEAL_RANGE * ANNEAL_TABLE + 2
    gtab = FVector{Float64}(undef, ntab)

    @inbounds for i in oneto(ntab)
        gtab[i] = log2(1.0 + exp2(-(i - 1) / ANNEAL_TABLE))
    end

    L      = FMatrix{UInt64}(undef, nw, nn)
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

    @inbounds for x in oneto(nn)
        par[x] = parent[x]
    end

    if W === nothing
        Wcap = -1.0     # the width of parent, once its legs are known
    else
        Wcap = Float64(W)
    end

    branchanneal_impl!(L, kptr, kwt, wbit, gtab, left, right, par, tc, sc, best, pre, preidx,
        stack, uf, head, prev, lca, next, bit, H, m, Int(point), nw, nk, Int(sweeps),
        Float64(beta0), Float64(beta1), Float64(mix), Wcap, Float64(lambda), Float64(time),
        Xoshiro(seed))

    return convert(Vector{Int}, best)
end

function branchanneal_impl!(
        L::AbstractMatrix{UInt64},
        kptr::AbstractVector{Int},
        kwt::AbstractVector{Float64},
        wbit::AbstractVector{Float64},
        gtab::AbstractVector{Float64},
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
        nw::Int,
        nk::Int,
        sweeps::Int,
        beta0::Float64,
        beta1::Float64,
        mix::Float64,
        Wcap::Float64,
        lambda::Float64,
        time::Float64,
        rng::AbstractRNG,
    )
    nn = 2m - 2
    root = anneal_legs!(L, left, right, par, pre, preidx, stack, uf, head, prev, lca, next, bit, H, m, point, nw)

    # contraction exponents and leg weights
    width = 0.0

    @inbounds for x in m + 1:nn
        tc[x] = anneal_union(L, kptr, kwt, wbit, left[x], right[x], nw, nk)
        sc[x] = anneal_union(L, kptr, kwt, wbit, x, x, nw, nk)
        width = max(width, sc[x])
    end

    if Wcap < 0.0
        Wcap = width
    end

    capped = isfinite(Wcap)
    exc = 0.0

    if capped
        @inbounds for x in m + 1:nn
            exc += max(0.0, sc[x] - Wcap)
        end
    end

    # the total cost, as C * 2^ref
    ref, C = anneal_total(tc, m, nn)
    bestE = Inf

    # the initial tree is a candidate (if within the cap)
    @inbounds for x in oneto(nn)
        best[x] = par[x]
    end

    if exc <= EPS
        bestE = ref + log2(C)
    end

    t0 = Base.time()
    lb = log(beta1 / beta0)
    sweep = 0

    while sweep < sweeps
        f = sweep / max(sweeps - 1, 1)

        if isfinite(time)
            el = Base.time() - t0

            if el >= time
                break
            end

            f = max(f, el / time)
        end

        beta = beta0 * exp(lb * min(f, 1.0))
        sweep += 1

        @inbounds for P in m + 1:nn
            r = rand(rng, UInt64)
            exc, ref, C = anneal_propose!(L, kptr, kwt, wbit, gtab, left, right, par, tc, sc, P, r,
                beta, mix, Wcap, lambda, capped, exc, ref, C, m, nn, nw, nk)
        end

        # the exact total (the running one loses precision when a dominant term leaves)
        ref, C = anneal_total(tc, m, nn)
        E = ref + log2(C)

        if capped
            exc = 0.0

            @inbounds for x in m + 1:nn
                exc += max(0.0, sc[x] - Wcap)
            end
        end

        if exc <= EPS && E < bestE - EPS
            bestE = E
            anneal_parent!(best, left, right, root, point, m, nn)
        end
    end

    return best
end

# The legs of every node of the tree par, built bottom-up: L(x) = L(l) | L(r) minus the
# vertices whose occurrences all lie below x (x is the LCA of their occurrences, found by
# union-find as in branchstats). Fills left and right; returns the child of the point.
function anneal_legs!(
        L::AbstractMatrix{UInt64},
        left::AbstractVector{Int},
        right::AbstractVector{Int},
        par::AbstractVector{Int},
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
        nw::Int,
    )
    nn = 2m - 2; n = nov(H); root = 0

    @inbounds for x in oneto(nn)
        left[x] = 0; right[x] = 0; head[x] = 0; uf[x] = x
    end

    @inbounds for v in oneto(n)
        prev[v] = 0; lca[v] = 0
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

    # preorder from the point
    npre = 0; sp = 1; stack[1] = point

    @inbounds while sp > 0
        x = stack[sp]; sp -= 1
        npre += 1; pre[npre] = x; preidx[x] = npre

        if left[x] != 0
            sp += 1; stack[sp] = left[x]
        end

        if right[x] != 0
            sp += 1; stack[sp] = right[x]
        end
    end

    @assert npre == nn "the parent vector is not a tree rooted at the point"

    # children before parents: the LCA of each vertex's occurrences
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

    # the vertices closed at each internal node (not those of the point: they never close)
    @inbounds for v in oneto(n)
        q = lca[v]

        if bit[v] >= 0 && q > m
            next[v] = head[q]; head[q] = v
        end
    end

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
                b = bit[v]
                L[b >> 6 + 1, x] &= ~(one(UInt64) << (b & 63))
                v = next[v]
            end
        end
    end

    return root
end

function anneal_find!(uf::AbstractVector{Int}, x::Int)
    @inbounds while uf[x] != x
        uf[x] = uf[uf[x]]; x = uf[x]
    end

    return x
end

# w(set) for the bits of one word (unaligned weights)
@inline function anneal_bits(x::UInt64, i::Int, wbit::AbstractVector{Float64})
    s = 0.0; o = 64 * (i - 1)

    @inbounds while x != zero(UInt64)
        s += wbit[o + trailing_zeros(x) + 1]; x &= x - one(UInt64)
    end

    return s
end

# w(L(x) | L(y))
function anneal_union(
        L::AbstractMatrix{UInt64},
        kptr::AbstractVector{Int},
        kwt::AbstractVector{Float64},
        wbit::AbstractVector{Float64},
        x::Int,
        y::Int,
        nw::Int,
        nk::Int,
    )
    s = 0.0

    @inbounds if ispositive(nk)
        for k in oneto(nk)
            c = 0

            for i in kptr[k]:kptr[k + 1] - 1
                c += count_ones(L[i, x] | L[i, y])
            end

            s += kwt[k] * c
        end
    else
        for i in oneto(nw)
            s += anneal_bits(L[i, x] | L[i, y], i, wbit)
        end
    end

    return s
end

# the weights of inv(Y') = L(X) | L(C), inv(P') = L(B) | L(Y') and L(Y') (if capped)
@inline function anneal_sums(
        L::AbstractMatrix{UInt64},
        kptr::AbstractVector{Int},
        kwt::AbstractVector{Float64},
        wbit::AbstractVector{Float64},
        X::Int,
        B::Int,
        Cc::Int,
        P::Int,
        nw::Int,
        nk::Int,
        capped::Bool,
    )
    ta = 0.0; tb = 0.0; ty = 0.0

    @inbounds if ispositive(nk)
        for k in oneto(nk)
            ca = 0; cb = 0; cy = 0

            for i in kptr[k]:kptr[k + 1] - 1
                lx = L[i, X]; lb = L[i, B]; lc = L[i, Cc]; lp = L[i, P]
                a = lx | lc; ny = a & (lb | lp)
                ca += count_ones(a); cb += count_ones(lb | ny)

                if capped
                    cy += count_ones(ny)
                end
            end

            ta += kwt[k] * ca; tb += kwt[k] * cb; ty += kwt[k] * cy
        end
    else
        for i in oneto(nw)
            lx = L[i, X]; lb = L[i, B]; lc = L[i, Cc]; lp = L[i, P]
            a = lx | lc; ny = a & (lb | lp)
            ta += anneal_bits(a, i, wbit); tb += anneal_bits(lb | ny, i, wbit)

            if capped
                ty += anneal_bits(ny, i, wbit)
            end
        end
    end

    return ta, tb, ty
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

# one proposal at the internal node P from the random word r; returns (exc, ref, C)
@inline function anneal_propose!(
        L::AbstractMatrix{UInt64},
        kptr::AbstractVector{Int},
        kwt::AbstractVector{Float64},
        wbit::AbstractVector{Float64},
        gtab::AbstractVector{Float64},
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
        capped::Bool,
        exc::Float64,
        ref::Float64,
        C::Float64,
        m::Int,
        nn::Int,
        nw::Int,
        nk::Int,
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
                return exc, ref, C
            end
        end

        if t
            B = right[Y]; Cc = left[Y]
        else
            B = left[Y]; Cc = right[Y]
        end

        ta, tb, ty = anneal_sums(L, kptr, kwt, wbit, X, B, Cc, P, nw, nk, capped)

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

        dp = 0.0

        if capped
            dp = max(0.0, ty - Wcap) - max(0.0, sc[Y] - Wcap)
            dE += lambda * dp
        end

        if dE > 0.0 && (r >> 11) * 0x1.0p-53 >= exp(-beta * dE)
            return exc, ref, C
        end

        # accept
        for i in oneto(nw)
            L[i, Y] = (L[i, X] | L[i, Cc]) & (L[i, B] | L[i, P])
        end

        if sd
            left[P] = B; right[P] = Y
        else
            right[P] = B; left[P] = Y
        end

        left[Y] = X; right[Y] = Cc; par[B] = P; par[X] = Y
        tc[Y] = ta; tc[P] = tb

        if capped
            sc[Y] = ty; exc += dp
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

    return exc, ref, C
end

# the parent vector of the current tree
function anneal_parent!(best::AbstractVector{Int}, left::AbstractVector{Int}, right::AbstractVector{Int}, root::Int, point::Int, m::Int, nn::Int)
    @inbounds for x in m + 1:nn
        best[left[x]] = x; best[right[x]] = x
    end

    @inbounds best[root] = point
    @inbounds best[point] = 0
    return best
end
