#
# Cost-aware selection inside the PMC dynamic program (evaluated into the
# CliqueTrees module: Base.include(CliqueTrees, "pmccost.jl")).
#
# pmcdp finds, over the minimal triangulations built from a list of
# potential maximal cliques, one of least width (ties: fewest heaviest
# bags). pmccost finds, among those whose bags all weigh at most maxk,
# one of least
#
#     cost = sum over its maximal cliques Ω of 2^w(Ω)
#
# (w = sum of log2 dims, so 2^w(Ω) is the size of the bag's dense
# table, the cost of eliminating it). The cost is a sum over the bags,
# so the recurrence of pmcdp holds with (min, +) in place of (min, max):
#
#     cost(S, C) = min over caps Ω of 2^w(Ω) + sum of cost(children)
#
# restricted to caps with w(Ω) <= maxk. With maxk = the treewidth this
# is the cheapest optimal-width decomposition among those the PMC list
# can build; maxk = tw + 1, tw + 2 trade a wider bag for a cheaper
# total. A block with no feasible cap falls back to the bag S ∪ C,
# as in pmcdp. Returns (perm, invp, log2 cost), log2 cost = Inf if no
# decomposition with bags <= maxk was found. With mult, the term of the
# PMC p is mult[p] 2^w(p) (random multipliers give diverse cheap TDs).
#
function pmccost(weights::AbstractVector{W}, graph::AbstractGraph{V}, pmc::BipartiteGraph{V}, maxk::W; mult::Union{Nothing, AbstractVector{Float64}} = nothing) where {W, V}
    nvtx = nv(graph); npmc = nv(pmc)

    perm = FVector{V}(undef, nvtx)
    invp = FVector{V}(undef, nvtx)

    if iszero(npmc)
        mink = zero(W)

        for v in oneto(nvtx)
            perm[v] = invp[v] = v
            mink += weights[v]
        end

        return perm, invp, mink <= maxk ? mink : convert(Float64, Inf)
    end

    work1  = FVector{V}(undef, nvtx)
    cmpptr = FVector{V}(undef, npmc + 1)

    cmp, owner, sep, cmpmin, cmplen, cmpwgt = pmcdp_cmp!(
        cmpptr, perm, work1, invp, weights, graph, pmc)

    ncmp = nv(sep)

    new    = FScalar{V}(undef)
    svar   = FVector{V}(undef, max(ncmp, npmc))
    flag   = FVector{V}(undef, ncmp)
    size   = FVector{V}(undef, ncmp)
    sub    = FVector{V}(undef, ncmp)
    captgt = FVector{V}(undef, ncmp)
    outwgt = FVector{W}(undef, max(ncmp, npmc))
    var    = FVector{V}(undef, ncmp + 1)
    keymin = FVector{V}(undef, 2ncmp)
    keylen = FVector{V}(undef, 2ncmp)
    keysep = FVector{V}(undef, 2ncmp)
    blkmin = FVector{V}(undef, 2ncmp)
    blkcmp = FVector{V}(undef, max(2ncmp, nvtx + 1))
    blkwgt = FVector{W}(undef, max(2ncmp, npmc))
    capptr = FVector{V}(undef, 2ncmp + 1)

    nsep = pmcdp_sep!(new, var, svar, flag, size, sep)

    chd = pmcdp_chd!(var, flag, size, outwgt, work1, weights,
        pmc, cmp, sep, cmpmin, cmplen, cmpwgt)

    cap = pmcdp_blk!(keymin, keylen, keysep, sub, captgt,
        blkmin, blkwgt, blkcmp, capptr, weights, sep, svar, nsep,
        cmpmin, cmplen, cmpwgt, flag, size, outwgt)

    pmcwgt = outwgt
    choice = keymin
    valc   = Vector{Float64}(undef, nv(cap))
    rootc  = Vector{Float64}(undef, npmc)
    rootf  = ones(V, npmc)

    pmccost_val!(pmcwgt, valc, choice, rootc, weights, pmc, cmp, owner, chd, sub, cap, blkwgt, maxk, mult)

    pmcdp_ord!(perm, invp, keylen, graph, pmc, cmp, owner,
        sep, chd, sub, blkmin, blkcmp, choice, rootc, rootf)

    return perm, invp, log2(minimum(rootc))
end

function pmccost_val!(
        pmcwgt::AbstractVector{W},
        valc::AbstractVector{Float64},
        choice::AbstractVector{V},
        rootc::AbstractVector{Float64},
        weights::AbstractVector{W},
        pmc::BipartiteGraph{V},
        cmp::BipartiteGraph{V},
        owner::AbstractVector{V},
        chd::BipartiteGraph{V},
        sub::AbstractVector{V},
        cap::BipartiteGraph{V},
        blkwgt::AbstractVector{W},
        maxk::W,
        mult::Union{Nothing, AbstractVector{Float64}},
    ) where {W, V}
    tol = 1.0e-9 * max(one(W), abs(maxk))

    @inbounds for p in vertices(pmc)
        w = zero(W)

        for x in neighbors(pmc, p)
            w += weights[x]
        end

        pmcwgt[p] = w
    end

    @inbounds for b in vertices(cap)
        best = blkwgt[b] <= maxk + tol ? exp2(convert(Float64, blkwgt[b])) : Inf; minc = zero(V)

        for c in neighbors(cap, b)
            k = pmcwgt[owner[c]]
            k > maxk + tol && continue
            s = exp2(convert(Float64, k))
            isnothing(mult) || (s *= mult[owner[c]])

            for d in neighbors(chd, c)
                s < best || break
                s += valc[sub[d]]
            end

            if s < best
                best = s; minc = c
            end
        end

        valc[b] = best; choice[b] = minc
    end

    @inbounds for p in vertices(pmc)
        k = pmcwgt[p]

        if k > maxk + tol
            rootc[p] = Inf
        else
            s = exp2(convert(Float64, k))
            isnothing(mult) || (s *= mult[p])

            for c in neighbors(cmp, p)
                s += valc[sub[c]]
            end

            rootc[p] = s
        end
    end

    return
end
