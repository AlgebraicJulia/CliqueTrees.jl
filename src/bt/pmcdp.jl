#
# Compute an elimination ordering from a collection
# C of potential maximal cliques. The time complexity
# is
#
#   O(|C| (|V| + |E|))
#
function pmcdp(weights::AbstractVector{W}, graph::AbstractGraph{V}, pmc::BipartiteGraph{V}) where {W, V}
    nvtx = nv(graph); npmc = nv(pmc)

    perm = FVector{V}(undef, nvtx)
    invp = FVector{V}(undef, nvtx)

    if iszero(npmc)
        mink = zero(W)

        for v in oneto(nvtx)
            perm[v] = invp[v] = v
            mink += weights[v]
        end

        minf = one(V)
    else
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
        valk   = FVector{W}(undef, 2ncmp)
        capptr = FVector{V}(undef, 2ncmp + 1)

        nsep = pmcdp_sep!(new, var, svar, flag, size, sep)

        chd = pmcdp_chd!(var, flag, size, outwgt, work1, weights,
            pmc, cmp, sep, cmpmin, cmplen, cmpwgt)

        cap = pmcdp_blk!(keymin, keylen, keysep, sub, captgt,
            blkmin, blkwgt, blkcmp, capptr, weights, sep, svar, nsep,
            cmpmin, cmplen, cmpwgt, flag, size, outwgt)

        pmcwgt = outwgt
        rootk  = blkwgt
        rootf  = svar
        valf   = keysep
        choice = keymin

        pmcdp_val!(pmcwgt, valk, valf, choice, rootk, rootf, weights,
            pmc, cmp, owner, chd, sub, cap, blkwgt)

        mink, minf = pmcdp_ord!(perm, invp, keylen, graph, pmc, cmp, owner,
            sep, chd, sub, blkmin, blkcmp, choice, rootk, rootf)
    end

    return perm, invp, mink, minf
end

function pmcdp_cmp!(
        cmpptr::AbstractVector{V},
        inpmc::AbstractVector{V},
        insep::AbstractVector{V},
        queue::AbstractVector{V},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        pmc::BipartiteGraph{V},
    ) where {W, V}
    nvtx = nv(graph); npmc = nv(pmc)

    @assert npmc < length(cmpptr)
    @assert nvtx <= length(inpmc)
    @assert nvtx <= length(insep)
    @assert nvtx <= length(queue)
    @assert nvtx <= length(weights)

    @inbounds for v in vertices(graph)
        inpmc[v] = insep[v] = zero(V)
    end

    c = zero(V)
    a = zero(V)

    cmpptr[1] = one(V)
    owner = V[]
    cmpmin = V[]
    cmplen = V[]
    cmpwgt = W[]
    sepptr = V[one(V)]
    septgt = V[]

    @inbounds for p in vertices(pmc)
        for x in neighbors(pmc, p)
            inpmc[x] = p
        end

        for v in vertices(graph)
            if abs(inpmc[v]) != p
                c += one(V)
                h = one(V)
                t = one(V)
                w = zero(W)

                inpmc[v] = -p
                queue[t] = v

                while h <= t
                    u = queue[h]; h += one(V); w += weights[u]

                    for x in neighbors(graph, u)
                        if abs(inpmc[x]) != p
                            inpmc[x] = -p; t += one(V); queue[t] = x
                        elseif inpmc[x] == p && insep[x] != c
                            insep[x] = c; a += one(V); push!(septgt, x)
                        end
                    end
                end

                push!(owner, p)
                push!(cmpmin, v)
                push!(cmplen, t)
                push!(cmpwgt, w)
                push!(sepptr, a + one(V))
            end
        end

        cmpptr[p + one(V)] = c + one(V)
    end

    ncmp = c
    narc = a
    cmp = BipartiteGraph{V, V}(ncmp, npmc, ncmp, cmpptr, oneto(ncmp))
    sep = BipartiteGraph{V, V}(nvtx, ncmp, narc, sepptr, septgt)
    return cmp, owner, sep, cmpmin, cmplen, cmpwgt
end

function pmcdp_sep!(
        new::AbstractScalar,
        var::AbstractVector,
        svar::AbstractVector,
        flag::AbstractVector,
        size::AbstractVector,
        sep::BipartiteGraph,
    )
    partition = twins_impl!(new, var, svar, flag, size, reverse(sep), Val(:N))
    return nv(partition)
end

function pmcdp_chd!(
        chdptr::AbstractVector{V},
        outmin::AbstractVector{V},
        outlen::AbstractVector{V},
        outwgt::AbstractVector{W},
        insep::AbstractVector{V},
        weights::AbstractVector{W},
        pmc::BipartiteGraph{V},
        cmp::BipartiteGraph,
        sep::BipartiteGraph,
        cmpmin::AbstractVector{V},
        cmplen::AbstractVector{V},
        cmpwgt::AbstractVector{W},
    ) where {W, V}
    nvtx = nov(sep); ncmp = nv(sep)

    @assert ncmp < length(chdptr)
    @assert ncmp <= length(outmin)
    @assert ncmp <= length(outlen)
    @assert ncmp <= length(outwgt)
    @assert ncmp <= length(cmpmin)
    @assert ncmp <= length(cmplen)
    @assert ncmp <= length(cmpwgt)
    @assert nvtx <= length(insep)
    @assert nvtx <= length(weights)

    @inbounds for v in outvertices(sep)
        insep[v] = zero(V)
    end

    chdtgt = V[]; a = zero(V)

    @inbounds for p in vertices(pmc)
        for c in neighbors(cmp, p)
            chdptr[c] = a + one(V); m = typemax(V); s = zero(V); w = zero(W)

            for x in neighbors(sep, c)
                insep[x] = c
            end

            for x in neighbors(pmc, p)
                if insep[x] != c
                    m = min(m, x); s += one(V); w += weights[x]
                end
            end

            for d in neighbors(cmp, p)
                for x in neighbors(sep, d)
                    if insep[x] != c
                        push!(chdtgt, d); a += one(V); m = min(m, cmpmin[d]); s += cmplen[d]; w += cmpwgt[d]
                        break
                    end
                end
            end

            outmin[c] = m; outlen[c] = s; outwgt[c] = w
        end
    end

    nchd = a
    chdptr[ncmp + one(V)] = nchd + one(V)
    chd = BipartiteGraph{V, V}(ncmp, ncmp, nchd, chdptr, chdtgt)
    return chd
end

function pmcdp_blk!(
        keymin::AbstractVector{V},
        keylen::AbstractVector{V},
        keysep::AbstractVector{V},
        sub::AbstractVector{V},
        captgt::AbstractVector{V},
        blkmin::AbstractVector{V},
        blkwgt::AbstractVector{W},
        blkcmp::AbstractVector{V},
        capptr::AbstractVector{V},
        weights::AbstractVector{W},
        sep::BipartiteGraph,
        sepid::AbstractVector{V},
        nsep::Integer,
        cmpmin::AbstractVector{V},
        cmplen::AbstractVector{V},
        cmpwgt::AbstractVector{W},
        outmin::AbstractVector{V},
        outlen::AbstractVector{V},
        outwgt::AbstractVector{W},
    ) where {W, V}
    nvtx = nov(sep); ncmp = nv(sep)

    @assert twice(ncmp) <= length(keymin)
    @assert twice(ncmp) <= length(keylen)
    @assert twice(ncmp) <= length(keysep)
    @assert ncmp <= length(sub)
    @assert ncmp <= length(captgt)
    @assert twice(ncmp) <= length(blkmin)
    @assert twice(ncmp) <= length(blkwgt)
    @assert twice(ncmp) <= length(blkcmp)
    @assert twice(ncmp) <  length(capptr)
    @assert ncmp <= length(sepid)
    @assert ncmp <= length(cmpmin)
    @assert ncmp <= length(cmplen)
    @assert ncmp <= length(cmpwgt)
    @assert ncmp <= length(outmin)
    @assert ncmp <= length(outlen)
    @assert ncmp <= length(outwgt)
    @assert max(nvtx, nsep) < length(blkcmp)
    @assert nvtx <= length(weights)

    nsrc = zero(V)

    @inbounds for c in vertices(sep)
        d = twice(c)

        keymin[d - one(V)] = cmpmin[c]; keymin[d] = outmin[c]
        keylen[d - one(V)] = cmplen[c]; keylen[d] = outlen[c]
        keysep[d - one(V)] = sepid[c];  keysep[d] = sepid[c]

        nsrc += one(V); blkmin[nsrc] = d - one(V)
        nsrc += one(V); blkmin[nsrc] = d
    end

    pmcdp_sort!(capptr, blkmin, keymin, blkcmp, nsrc, nvtx)
    pmcdp_sort!(blkmin, capptr, keysep, blkcmp, nsrc, nsep)
    pmcdp_sort!(capptr, blkmin, keylen, blkcmp, nsrc, nvtx)

    b = q = j = zero(V)

    @inbounds for i in oneto(nsrc)
        k = capptr[i]; c = cld(k, two(V))

        if isone(i) || keymin[k] != keymin[j] || keysep[k] != keysep[j] || keylen[k] != keylen[j]
            if isodd(k)
                w = cmpwgt[c]
            else
                w = outwgt[c]
            end

            for x in neighbors(sep, c)
                w += weights[x]
            end

            b += one(V)
            blkmin[b] = keymin[k]
            blkwgt[b] = w
            blkcmp[b] = c
            capptr[b] = q + one(V)
        end

        if isodd(k)
            sub[c] = b
        else
            q += one(V); captgt[q] = c
        end

        j = k
    end

    nblk = b
    ncap = q
    capptr[nblk + one(V)] = ncap + one(V)
    cap = BipartiteGraph{V, V}(ncmp, nblk, ncap, capptr, captgt)
    return cap
end

function pmcdp_sort!(
        dst::AbstractVector{V},
        src::AbstractVector{V},
        key::AbstractVector{V},
        cnt::AbstractVector{V},
        nsrc::V,
        nkey::V,
    ) where {V}
    @assert nkey <= length(cnt)
    @assert nsrc <= length(src)
    @assert nsrc <= length(dst)

    @inbounds for j in oneto(nkey)
        cnt[j] = zero(V)
    end

    @inbounds for i in oneto(nsrc)
        j = key[src[i]]; cnt[j] += one(V)
    end

    c = zero(V)

    @inbounds for j in oneto(nkey)
        cnt[j] = c += cnt[j]
    end

    @inbounds for i in reverse(oneto(nsrc))
        j = key[src[i]]; dst[cnt[j]] = src[i]; cnt[j] -= one(V)
    end

    return dst
end

function pmcdp_val!(
        pmcwgt::AbstractVector{W},
        valk::AbstractVector{W},
        valf::AbstractVector{V},
        choice::AbstractVector{V},
        rootk::AbstractVector{W},
        rootf::AbstractVector{V},
        weights::AbstractVector{W},
        pmc::BipartiteGraph{V},
        cmp::BipartiteGraph{V},
        owner::AbstractVector{V},
        chd::BipartiteGraph{V},
        sub::AbstractVector{V},
        cap::BipartiteGraph{V},
        blkwgt::AbstractVector{W},
    ) where {W, V}
    npmc = nv(pmc); nblk = nv(cap); ncmp = nov(cap); nvtx = nov(pmc)

    @assert npmc <= length(pmcwgt)
    @assert npmc <= length(rootk)
    @assert npmc <= length(rootf)
    @assert nblk <= length(valk)
    @assert nblk <= length(valf)
    @assert nblk <= length(choice)
    @assert nblk <= length(blkwgt)
    @assert ncmp <= length(owner)
    @assert ncmp <= length(sub)
    @assert nvtx <= length(weights)

    @inbounds for p in vertices(pmc)
        w = zero(W)

        for x in neighbors(pmc, p)
            w += weights[x]
        end

        pmcwgt[p] = w
    end

    @inbounds for b in vertices(cap)
        mink = blkwgt[b]; minf = one(V); minc = zero(V)

        for c in neighbors(cap, b)
            k = pmcwgt[owner[c]]; f = one(V)

            for d in neighbors(chd, c)
                (k, f) < (mink, minf) || break
                e = sub[d]; k, f = pmcdp_plus(k, f, valk[e], valf[e])
            end

            if (k, f) < (mink, minf)
                mink = k; minf = f; minc = c
            end
        end

        valk[b] = mink; valf[b] = minf; choice[b] = minc
    end

    @inbounds for p in vertices(pmc)
        k = pmcwgt[p]; f = one(V)

        for c in neighbors(cmp, p)
            b = sub[c]; k, f = pmcdp_plus(k, f, valk[b], valf[b])
        end

        rootk[p] = k; rootf[p] = f
    end

    return
end

function pmcdp_ord!(
        perm::AbstractVector{V},
        invp::AbstractVector{V},
        stack::AbstractVector{V},
        graph::AbstractGraph{V},
        pmc::BipartiteGraph{V},
        cmp::BipartiteGraph,
        owner::AbstractVector{V},
        sep::BipartiteGraph,
        chd::BipartiteGraph,
        sub::AbstractVector{V},
        blkmin::AbstractVector{V},
        blkcmp::AbstractVector{V},
        choice::AbstractVector{V},
        rootk::AbstractVector{W},
        rootf::AbstractVector{V},
    ) where {W, V}
    nvtx = nv(graph); ncmp = nv(sep); npmc = nv(pmc)

    @assert nvtx <= length(perm)
    @assert nvtx <= length(invp)
    @assert ncmp <= length(stack)
    @assert ncmp <= length(sub)
    @assert ncmp <= length(owner)
    @assert npmc <= length(rootk)
    @assert npmc <= length(rootf)

    r = one(V)
    mink = rootk[r]
    minf = rootf[r]

    @inbounds for p in vertices(pmc)
        if (rootk[p], rootf[p]) < (mink, minf)
            r = p
            mink = rootk[p]
            minf = rootf[p]
        end
    end

    i = nvtx; t = zero(V)

    @inbounds for v in vertices(graph)
        invp[v] = nvtx + one(V)
    end

    @inbounds for x in neighbors(pmc, r)
        perm[i] = x; invp[x] = i; i -= one(V)
    end

    @inbounds for c in neighbors(cmp, r)
        t += one(V); stack[t] = sub[c]
    end

    @inbounds while ispositive(t)
        b = stack[t]; t -= one(V); c = choice[b]; j = nvtx + one(V) + i

        if iszero(c)
            for x in neighbors(sep, blkcmp[b])
                if invp[x] > nvtx
                    invp[x] = j
                end
            end

            v = blkmin[b]
            perm[i] = v; invp[v] = i

            h = i
            i -= one(V)

            while h > i
                u = perm[h]; h -= one(V)

                for x in neighbors(graph, u)
                    if invp[x] > nvtx && invp[x] != j
                        perm[i] = x; invp[x] = i; i -= one(V)
                    end
                end
            end
        else
            for x in neighbors(sep, c)
                if invp[x] > nvtx
                    invp[x] = j
                end
            end

            for x in neighbors(pmc, owner[c])
                if invp[x] > nvtx && invp[x] != j
                    perm[i] = x; invp[x] = i; i -= one(V)
                end
            end

            for d in neighbors(chd, c)
                t += one(V); stack[t] = sub[d]
            end
        end
    end

    return mink, minf
end

function pmcdp_plus(ak::W, af, bk::W, bf) where {W}
    if ak == bk
        ck = ak; cf = af + bf
    elseif ak < bk
        ck = bk; cf = bf
    else
        ck = ak; cf = af
    end

    return ck, cf
end
