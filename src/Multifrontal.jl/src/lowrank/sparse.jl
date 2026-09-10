const LOWRANK_BLOCK = 32
const LOWRANK_LARGE = 4096

struct HyperSymbolic{I}
    col::BipartiteGraph{I, I, FVector{I}, FVector{I}}
    Aptr::FVector{I}
    nFval::I
end

function HyperSymbolic(S::ChordalSymbolic{I}, B::SparseMatrixCSC{T}) where {T, I <: Integer}
    res = S.res
    sep = S.sep
    chd = S.chd
    idx = S.idx

    n = convert(I, size(B, 2))
    m = convert(I, nnz(B))
    t = nv(res)

    col = FBipartiteGraph{I, I}(S.nFval, n, m)
    Aptr = FVector{I}(undef, t + one(I))
    Mptr = FVector{I}(undef, S.nMptr)
    #
    # copy the column pointers of B into col
    #
    for c in oneto(n + one(I))
        pointers(col)[c] = B.colptr[c]
    end
    #
    # initialize Aptr
    #
    #   Aptr ← 0
    #
    fill!(Aptr, zero(I))
    #
    # for each node j, count the number
    # of columns c whose first row is in
    # the residual at j.
    #
    #   Aptr[j + 1] ← # columns
    #
    strt = one(I)

    for c in oneto(n)
        stop = B.colptr[c + one(I)]

        if strt < stop
            r = rowvals(B)[strt]
            j = idx[r]
            Aptr[j + one(I)] += one(I)
        end

        strt = stop
    end
    #
    # perform a cumulative sum so that
    # for all nodes j,
    #
    #   Aptr[j + 1] - 1
    #
    # contains the number of columns
    # with first row in the subtree
    # at j.
    #
    c = one(I)

    for j in oneto(t)
        Aptr[j] = c += Aptr[j]
    end

    Aptr[t + one(I)] += c
    #
    # for all arcs p = (r, c), let j be the
    # unique node whose residual contains
    # the first row r* ≤ r in c
    #
    #   r* ∈ res(j);
    #
    # assign to tgt[p] the *relative* index of r in
    # bag(j).
    #
    sp = zero(I); nFval = one(I)

    for j in oneto(t)
        nn = eltypedegree(res, j)
        na = eltypedegree(sep, j)
        nj = nn + na

        jsep = neighbors(sep, j)
        jres = neighbors(res, j)

        strt = Aptr[j]
        stop = Aptr[j + one(I)]

        @inbounds for c in strt:stop - one(I)
            rloc = one(I)
            s = jres[rloc]

            for p in nzrange(B, c)
                r = rowvals(B)[p]

                while s < r && rloc < nn
                    rloc += one(I)
                    s = jres[rloc]
                end

                while s < r && rloc < nj
                    rloc += one(I)
                    s = jsep[rloc - nn]
                end

                if s != r
                    throw(ArgumentError("index ($r, $c) not in symbolic pattern"))
                end

                targets(col)[p] = rloc
            end
        end

        nb = stop - strt

        for i in neighbors(chd, j)
            nb += Mptr[sp]
            sp -= one(I)
        end

        nFval = max(nFval, nj * nb)

        if ispositive(na)
            sp += one(I)
            Mptr[sp] = min(nb, na)
        end
    end

    return HyperSymbolic{I}(col, Aptr, nFval)
end

function HyperSymbolic(L::ChordalTriangular, B::SparseMatrixCSC)
    return HyperSymbolic(L.S, B)
end

struct LowrankWorkspace{T, I}
    Fval::FVector{T}
    Wval::FVector{T}
    Sptr::FVector{I}
    Mptr::FVector{I}
    Mval::FVector{T}
end

function LowrankWorkspace{T}(S::ChordalSymbolic{I}, H::HyperSymbolic{I}) where {T, I <: Integer}
    nWval = convert(I, LOWRANK_BLOCK) * S.nFval
    Fval = FVector{T}(undef, H.nFval)
    Wval = FVector{T}(undef, nWval + max(H.nFval, nWval))
    Sptr = FVector{I}(undef, S.nFval + one(I))
    Mptr = FVector{I}(undef, S.nMptr)
    Mval = FVector{T}(undef, S.nMval)
    return LowrankWorkspace{T, I}(Fval, Wval, Sptr, Mptr, Mval)
end

function LowrankWorkspace(L::ChordalTriangular{DIAG, UPLO, T}, H::HyperSymbolic) where {DIAG, UPLO, T}
    return LowrankWorkspace{T}(L.S, H)
end

function LinearAlgebra.lowrankupdate!(F::ChordalCholesky{UPLO, T}, B::SparseMatrixCSC; check::Bool=true) where {UPLO, T}
    return lowrankupdate!(F, convert(SparseMatrixCSC{T}, B); check)
end

function LinearAlgebra.lowrankupdate!(F::ChordalCholesky{UPLO, T}, B::SparseMatrixCSC{T}; check::Bool=true) where {UPLO, T}
    L = triangular(F)

    rowperm = F.perm
    B = rowpermute(B, rowperm)

    colperm = lowrank_sparse_permutation(B)
    B = colpermute(B, colperm)

    H = HyperSymbolic(L, B)
    W = LowrankWorkspace(L, H)
    info = lowrankupdate!(W, L, H, nonzeros(B))

    if ispositive(info)
        F.info[] = rowperm[info]
    else
        F.info[] = info
    end

    checkinfo(info, L.diag, check)
    return F
end

function LinearAlgebra.lowrankupdate!(F::ChordalCholesky, B::AbstractMatrix; check::Bool=true)
    return lowrankupdate!(F, sparse(B); check)
end

function lowrank_sparse_permutation(B::SparseMatrixCSC{T, I}) where {T, I <: Integer}
    m = convert(I, size(B, 1))
    n = convert(I, size(B, 2))

    ptr = FVector{I}(undef, m + one(I))
    tgt = FVector{I}(undef, n)
    #
    # initialize ptr
    #
    #   ptr ← 0
    #
    for r in oneto(m + one(I))
        ptr[r] = zero(I)
    end
    #
    # for each row r, count the number
    # of columns c whose first row is r
    #
    #   ptr[r + 1] ← # columns
    #
    strt = one(I)

    for c in oneto(n)
        stop = B.colptr[c + one(I)]

        if strt < stop
            r = rowvals(B)[strt]
            ptr[r + one(I)] += one(I)
        end

        strt = stop
    end
    #
    # perform a cumulative sum so that
    # for all rows r,
    #
    #   ptr[r + 1] - 1
    #
    # contains the number of columns
    # with first row less-than-or-equal-to
    # r.
    #
    c = one(I)

    for r in oneto(m)
        ptr[r] = c += ptr[r]
    end

    ptr[m + one(I)] += c
    #
    # use a bucket sort to compute
    # a permutation tgt
    #
    # for all rows r, this permutation
    # assigns the columns with first
    # row r to the interval
    #
    #   ptr[r] ... ptr[r + 1] - 1,
    #
    # and it assigns empty columns
    # to the interval
    #
    #   ptr[m + 1] ... n
    #
    strt = one(I)

    for c in oneto(n)
        stop = B.colptr[c + one(I)]

        if strt < stop
            r = rowvals(B)[strt]
            tgt[ptr[r]] = c
            ptr[r] += one(I)
        else
            tgt[ptr[m + one(I)]] = c
            ptr[m + one(I)] += one(I)
        end

        strt = stop
    end

    return tgt
end

function LinearAlgebra.lowrankupdate!(
        W::LowrankWorkspace{T, I},
        L::ChordalTriangular{:N, UPLO, T, I},
        H::HyperSymbolic{I},
        Bval::AbstractVector{T},
    ) where {UPLO, T, I <: Integer}
    return lowrank_sparse_impl!(
        W.Mptr, W.Mval, L.S.Dptr, L.Dval, L.S.Lptr, L.Lval, W.Fval, W.Wval, W.Sptr, H.Aptr,
        Bval, L.S.res, L.S.sep, L.S.rel, H.col, L.S.chd, L.uplo,
    )
end

function lowrank_sparse_impl!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Fval::AbstractVector{T},
        Wval::AbstractVector{T},
        Sptr::AbstractVector{I},
        Aptr::AbstractVector{I},
        Bval::AbstractVector{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        rel::AbstractGraph{I},
        col::AbstractGraph{I},
        chd::AbstractGraph{I},
        uplo::Val{UPLO},
    ) where {UPLO, T, I <: Integer}
    ns = zero(I); Mptr[one(I)] = one(I)
    info = zero(I)

    for j in vertices(res)
        ns, localinfo = lowrank_sparse_loop!(
            Mptr, Mval, Dptr, Dval, Lptr, Lval, Fval, Wval, Sptr, Aptr,
            Bval, res, sep, rel, col, chd, ns, j, uplo,
        )

        if ispositive(localinfo) && iszero(info)
            info = localinfo + pointers(res)[j] - one(I)
        end
    end

    return info
end

function lowrank_sparse_iszero(D₁₁::AbstractMatrix{T}, L₂₁::AbstractMatrix{T}, ::Val{:L}) where {T}
    m = size(D₁₁, 1)
    n = size(L₂₁, 1)

    @inbounds for c in oneto(m)
        for r in c:m
            iszero(D₁₁[r, c]) || return false
        end

        for r in oneto(n)
            iszero(L₂₁[r, c]) || return false
        end
    end

    return true
end

function lowrank_sparse_iszero(D₁₁::AbstractMatrix{T}, U₁₂::AbstractMatrix{T}, ::Val{:U}) where {T}
    m = size(D₁₁, 1)
    n = size(U₁₂, 2)

    @inbounds for c in oneto(m)
        for r in oneto(c)
            iszero(D₁₁[r, c]) || return false
        end
    end

    @inbounds for c in oneto(n)
        for r in oneto(m)
            iszero(U₁₂[r, c]) || return false
        end
    end

    return true
end

function lowrank_sparse_loop!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Fval::AbstractVector{T},
        Wval::AbstractVector{T},
        Sptr::AbstractVector{I},
        Aptr::AbstractVector{I},
        Bval::AbstractVector{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        rel::AbstractGraph{I},
        col::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        uplo::Val{UPLO},
    ) where {UPLO, T, I <: Integer}
    #
    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    nn = eltypedegree(res, j)
    #
    # na is the size of the separator at node j
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(sep, j)
    #
    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na
    #
    # L is part of the Cholesky factor:
    #
    #        res(j)
    #   L = [ D₁₁ ] res(j)
    #       [ L₂₁ ] sep(j)
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end

    for r in oneto(nj + one(I))
        Sptr[r] = zero(I)
    end
    #
    # construct the function
    #
    #   ncl: new(j) → bag(j)
    #
    # mapping each "new" column to its
    # least vertex
    #
    ns0 = ns; nb = lowrank_sparse_send_1A!(Sptr, Aptr, col, j)

    for i in Iterators.reverse(neighbors(chd, j))
        nb += lowrank_sparse_send_1B!(Sptr, Mptr, rel, ns, i)
        ns -= one(I)
    end

    ns = ns0; p = zero(I)

    for r in oneto(nj + one(I))
        p = Sptr[r] += p
    end
    #
    # bn is the number of new columns owned by res(j)
    #
    #   bn = | ncl⁻¹ res(j) |
    #
    bn = Sptr[nn + one(I)]
    #
    # F is the frontal matrix at node j
    #
    #            nn  nb
    #     F = [ D₁₁ F₁₂ ] nn
    #         [ L₂₁ F₂₂ ] na
    #
    #            nn  bn  ba
    #       = [ D₁₁ G₁₂     ] nn
    #         [ L₂₁ G₂₂ G₂₃ ] na
    #
    if UPLO === :L
        F₂ = reshape(view(Fval, one(I):nj * nb), nj, nb)
        F₁₂ = view(F₂,      one(I):nn, one(I):nb)
        F₂₂ = view(F₂, nn + one(I):nj, one(I):nb)
        G₁₂ = view(F₁₂,     one(I):nn, one(I):bn)
        G₂₂ = view(F₂₂,     one(I):na, one(I):bn)
    else
        F₂ = reshape(view(Fval, one(I):nb * nj), nb, nj)
        F₁₂ = view(F₂, one(I):nb,      one(I):nn)
        F₂₂ = view(F₂, one(I):nb, nn + one(I):nj)
        G₁₂ = view(F₁₂, one(I):bn, one(I):nn)
        G₂₂ = view(F₂₂, one(I):bn, one(I):na)
    end
    #
    #     F₂ ← 0
    #
    fill!(F₂, false)
    #
    # write B to F
    #
    #     F₂ ← B
    #
    lowrank_sparse_send_2A!(F₂, Sptr, Aptr, Bval, col, j, uplo)

    for i in Iterators.reverse(neighbors(chd, j))
        #
        # write the update matrix for child i to F
        #
        #   F₂ ← [ ... ] TODO
        #
        lowrank_sparse_send_2B!(F₂, Sptr, Mptr, Mval, rel, ns, i, uplo)
        ns -= one(I)
    end

    if lowrank_sparse_iszero(D₁₁, L₂₁, uplo)
        #
        # factorize F
        #
        #     [ G₁₂     ] = [ G₁₂'       ] [ R₂₂ R₂₃ ]
        #     [ G₂₂ G₂₃ ]   [ G₂₂'  G₂₃' ] [ R₃₂ R₃₃ ]
        #
        if ispositive(nb)
            lowrank_sparse_factor_2!(F₂, Sptr, Wval, zero(I), uplo)
        end
        #
        # copy F₁₂ and F₂₂ into D₁₁ and L₂₁
        #
        #   D₁₁ ← F₁₂
        #   L₂₁ ← F₂₂
        #
        nd = min(nn, nb)

        if UPLO === :L
            @inbounds for c in oneto(nd)
                for r in c:nn
                    D₁₁[r, c] = F₁₂[r, c]
                end

                for r in oneto(na)
                    L₂₁[r, c] = F₂₂[r, c]
                end
            end

            @inbounds for c in nd + one(I):nn
                for r in c:nn
                    D₁₁[r, c] = zero(T)
                end

                for r in oneto(na)
                    L₂₁[r, c] = zero(T)
                end
            end
        else
            @inbounds for c in oneto(nd)
                for r in oneto(c)
                    D₁₁[r, c] = F₁₂[r, c]
                end
            end

            @inbounds for c in nd + one(I):nn
                for r in oneto(nd)
                    D₁₁[r, c] = F₁₂[r, c]
                end

                for r in nd + one(I):c
                    D₁₁[r, c] = zero(T)
                end
            end

            @inbounds for c in oneto(na)
                for r in oneto(nd)
                    L₂₁[r, c] = F₂₂[r, c]
                end

                for r in nd + one(I):nn
                    L₂₁[r, c] = zero(T)
                end
            end
        end
        #
        # nc is the number of new columns at node j
        #
        #     nc = | new(j) |
        #
        nc = min(na, nb - nd); c0 = nn
    else
        #
        # factorize F
        #
        #     [ D₁₁ G₁₂ ] = [ D₁₁'       ] [ Q₁₁ Q₁₂ ]
        #     [ L₂₁ G₂₂ ]   [ L₂₁'  G₂₂' ] [ Q₂₁ Q₂₂ ]
        #
        # and write
        #
        #   D₁₁ ← D₁₁'
        #   L₂₁ ← L₂₁'
        #   G₂₂ ← G₂₂'
        #
        if ispositive(bn)
            lowrank_sparse_factor_1!(D₁₁, L₂₁, G₁₂, G₂₂, Wval, uplo)
        end
        #
        # factorize F₂₂
        #
        #   [ G₂₂ G₂₃ ] = [ G₂₂' 0 ] [ R₂₂ R₂₃ ]
        #                            [ R₃₂ R₃₃ ]
        #
        # and write
        #
        #   G₂₂ ← G₂₂'
        #
        if ispositive(na) && ispositive(nb)
            lowrank_sparse_factor_2!(F₂₂, Sptr, Wval, nn, uplo)
        end
        #
        # nc is the number of new columns at node j
        #
        #     nc = | new(j) |
        #
        nc = min(na, nb); c0 = zero(I)
    end
    #
    # ensure that L has a non-negative diagonal
    #
    info = zero(I)

    @inbounds for k in oneto(nn)
        Lkk = D₁₁[k, k]

        if isnegative(real(Lkk))
            if UPLO === :L
                for r in k:nn
                    D₁₁[r, k] = -D₁₁[r, k]
                end

                for r in oneto(na)
                    L₂₁[r, k] = -L₂₁[r, k]
                end
            else
                for c in k:nn
                    D₁₁[k, c] = -D₁₁[k, c]
                end

                for c in oneto(na)
                    L₂₁[k, c] = -L₂₁[k, c]
                end
            end
        elseif iszero(Lkk) && iszero(info)
            info = k
        end
    end

    @inbounds for k in oneto(nc)
        if UPLO === :L
            Fkk = F₂₂[k, c0 + k]

            if isnegative(real(Fkk))
                for r in k:na
                    F₂₂[r, c0 + k] = -F₂₂[r, c0 + k]
                end
            end
        else
            Fkk = F₂₂[c0 + k, k]

            if isnegative(real(Fkk))
                for c in k:na
                    F₂₂[c0 + k, c] = -F₂₂[c0 + k, c]
                end
            end
        end
    end

    if ispositive(na)
        #
        # M₂₂ is the update matrix for node j
        #
        #           nc  nb - nc
        #   F₂₂ = [ M₂₂         ] na
        #
        ns += one(I)
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * nc

        if UPLO === :L
            M₂₂ = reshape(view(Mval, strt:stop - one(I)), na, nc)

            @inbounds for c in oneto(nc)
                for s in oneto(c - one(I))
                    M₂₂[s, c] = zero(T)
                end

                for s in c:na
                    M₂₂[s, c] = F₂₂[s, c0 + c]
                end
            end
        else
            M₂₂ = reshape(view(Mval, strt:stop - one(I)), nc, na)

            @inbounds for c in oneto(na)
                for r in oneto(min(c, nc))
                    M₂₂[r, c] = F₂₂[c0 + r, c]
                end

                for r in c + one(I):nc
                    M₂₂[r, c] = zero(T)
                end
            end
        end
    end

    return ns, info
end

function lowrank_sparse_send_1A!(
        Sptr::AbstractVector{I},
        Aptr::AbstractVector{I},
        col::AbstractGraph{I},
        j::I,
    ) where {I <: Integer}

    strt = Aptr[j]
    stop = Aptr[j + one(I)]

    @inbounds for c in strt:stop - one(I)
        p = pointers(col)[c]
        r = targets(col)[p]
        Sptr[r + one(I)] += one(I)
    end

    return stop - strt
end

function lowrank_sparse_send_1B!(
        Sptr::AbstractVector{I},
        Mptr::AbstractVector{I},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
    ) where {I <: Integer}
    #
    # na is the size of the separator at node j
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(rel, i)
    #
    # inj is the subset inclusion
    #
    #     inj: sep(i) → bag(parent(i))
    #
    inj = neighbors(rel, i)

    strt = Mptr[ns]
    stop = Mptr[ns + one(I)]
    #
    # nc is the number of new columns at node i
    #
    #     nc = | new(i) |
    #
    nc = div(stop - strt, na)

    @inbounds for c in oneto(nc)
        d = inj[c]
        Sptr[d + one(I)] += one(I)
    end

    return nc
end

function lowrank_sparse_send_2A!(
        F::AbstractMatrix{T},
        Sptr::AbstractVector{I},
        Aptr::AbstractVector{I},
        Bval::AbstractVector{T},
        col::AbstractGraph{I},
        j::I,
        ::Val{UPLO},
    ) where {UPLO, T, I <: Integer}
    strt = Aptr[j]
    stop = Aptr[j + one(I)]

    for e in strt:stop - one(I)
        c = zero(I)

        for p in incident(col, e)
            r = targets(col)[p]

            if iszero(c)
                c = Sptr[r] += one(I)
            end

            if UPLO === :L
                F[r, c] = Bval[p]
            else
                F[c, r] = conj(Bval[p])
            end
        end
    end

    return
end

function lowrank_sparse_send_2B!(
        F::AbstractMatrix{T},
        Sptr::AbstractVector{I},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
        ::Val{UPLO},
    ) where {UPLO, T, I <: Integer}

    strt = Mptr[ns]
    stop = Mptr[ns + one(I)]
    #
    # na is the size of the separator at node i
    #
    #     na = | sep(i) |
    #
    na = eltypedegree(rel, i)
    #
    # nc is the number of new columns at node i
    #
    #     nc = | new(i) |
    #
    nc = div(stop - strt, na)
    #
    # inj is the subset inclusion
    #
    #     inj: sep(i) → bag(parent(i))
    #
    inj = neighbors(rel, i)
    #
    # M₂₂ is the update matrix from child i
    #
    if UPLO === :L
        M₂₂ = reshape(view(Mval, strt:stop - one(I)), na, nc)
    else
        M₂₂ = reshape(view(Mval, strt:stop - one(I)), nc, na)
    end

    @inbounds for c in oneto(nc)
        d = Sptr[inj[c]] += one(I)

        for s in c:na
            if UPLO === :L
                F[inj[s], d] = M₂₂[s, c]
            else
                F[d, inj[s]] = M₂₂[c, s]
            end
        end
    end

    return
end

function lowrank_sparse_factor_1!(
        D₁₁::AbstractMatrix{T},
        L₂₁::AbstractMatrix{T},
        G₁₂::AbstractMatrix{T},
        G₂₂::AbstractMatrix{T},
        Wval::AbstractVector{T},
        ::Val{UPLO},
    ) where {UPLO, T}
    m = size(D₁₁, 1)
    n = UPLO === :L ? size(L₂₁, 1) : size(L₂₁, 2)
    k = min(m, LOWRANK_BLOCK)

    wsize = k * max(m, n)

    work1 = view(Wval,         1:k * m)
    work2 = view(Wval, k * m + 1:k * m + wsize)

    W = reshape(work1, k, m)
    #
    # factorize the first row
    #
    #   [ D₁₁ G₁₂ ] = [ D₁₁' 0 ] [ Q₁₁ Q₁₂ ]
    #                            [ Q₂₁ Q₂₂ ]
    # and write
    #
    #   D₁₁ ← D₁₁'
    #
    if UPLO === :L
        tplqt!(D₁₁, G₁₂, W, work2)
    else
        tpqrt!(D₁₁, G₁₂, W, work2)
    end

    if !isempty(G₂₂)
        #
        # update the second row
        #
        #   [ L₂₁ G₂₂ ] ← [ L₂₁ G₂₂ ] [ Q₁₁ᴴ Q₂₁ᴴ ]
        #                             [ Q₁₂ᴴ Q₂₂ᴴ ]
        #
        if UPLO === :L
            tpmlqt!(G₁₂, W, L₂₁, G₂₂, work2)
        else
            tpmqrt!(G₁₂, W, L₂₁, G₂₂, work2)
        end
    end

    return
end

#
# Given a matrix
#
#           r ⋯
#   F = r [  fᵣᴴ ]
#       ⋮ [  Fₙ  ]
#
# construct the Householder matrix
#
#   H = I - τ v vᴴ
#
# such that fᵣᴴ H = [ β 0 ⋯ 0 ] and
# apply it to Fₙ
#
#   fᵣᴴ ← fᵣᴴ H
#   Fₙ  ← Fₙ  H
#
function lowrank_sparse_block!(F::AbstractMatrix{T}, w::AbstractVector{T}, r::Int, t::Int, n::Int, ::Val{:L}) where {T}
    if t >= r
        σ = zero(real(T))

        @inbounds for c in r:t
            σ = max(σ, abs(F[r, c]))
        end

        if !iszero(σ)
            Frr = @inbounds F[r, r]

            if sqrt(floatmin(real(T))) <= σ <= sqrt(floatmax(real(T)) / (t - r + 1)) / 2
                σ = one(real(T))
            end

            ν = zero(real(T))

            @inbounds for c in r + 1:t
                ν += abs2(F[r, c] / σ)
            end

            if !iszero(ν) || !isreal(Frr) || isnegative(real(Frr))
                #
                # apply H to fᵣᴴ
                #
                #   fᵣᴴ ← fᵣᴴ H
                #
                @inbounds F[r, r] = β = -copysign(σ * sqrt(abs2(Frr / σ) + ν), real(Frr))
                #
                # compute the Householder scalar τ
                #
                τ = (β - conj(Frr)) / β
                #
                # compute the Householder vector v
                #
                δ = inv(Frr - β)

                @inbounds for c in r + 1:t
                    F[r, c] *= δ
                end
                #
                # apply H to Fₙ
                #
                #   Fₙ  ← Fₙ H
                #
                @inbounds for s in r + 1:n
                    w[s] = F[s, r]
                end

                @inbounds for c in r + 1:t
                    cFrc = conj(F[r, c])

                    for s in r + 1:n
                        w[s] = muladd(F[s, c], cFrc, w[s])
                    end
                end

                @inbounds for s in r + 1:n
                    F[s, r] -= w[s] *= τ
                end

                @inbounds for c in r + 1:t
                    Frc = F[r, c]

                    for s in r + 1:n
                        F[s, c] = muladd(-w[s], Frc, F[s, c])
                    end
                end
            end
        end
    end

    return
end

#
# Given a matrix
#
#         c
#   F = [ fᶜ Fₙ ]
#
# construct the Householder matrix
#
#   H = I - τ v vᴴ
#
# such that H fᶜ = [ β 0 ⋯ 0 ]ᵀ and
# apply it to the remaining columns
#
#   fᶜ ← H fᶜ
#   Fₙ ← H Fₙ
#
function lowrank_sparse_block!(F::AbstractMatrix{T}, c::Int, t::Int, n::Int, ::Val{:U}) where {T}
    if t >= c
        σ = zero(real(T))

        @inbounds for r in c:t
            σ = max(σ, abs(F[r, c]))
        end

        if !iszero(σ)
            Fcc = @inbounds F[c, c]

            if sqrt(floatmin(real(T))) <= σ <= sqrt(floatmax(real(T)) / (t - c + 1)) / 2
                σ = one(real(T))
            end

            ν = zero(real(T))

            @inbounds for r in c + 1:t
                ν += abs2(F[r, c] / σ)
            end

            if !iszero(ν) || !isreal(Fcc) || isnegative(real(Fcc))
                #
                # apply H to fᶜ
                #
                #   fᶜ ← H fᶜ
                #
                @inbounds F[c, c] = β = -copysign(σ * sqrt(abs2(Fcc / σ) + ν), real(Fcc))
                #
                # compute the Householder scalar τ
                #
                τ = (β - conj(Fcc)) / β
                #
                # compute the Householder vector v
                #
                δ = inv(Fcc - β)

                @inbounds for r in c + 1:t
                    F[r, c] *= δ
                end
                #
                # apply H to Fₙ
                #
                #   Fₙ ← H Fₙ
                #
                @inbounds for u in c + 1:n
                    α = F[c, u]

                    for r in c + 1:t
                        α = muladd(conj(F[r, c]), F[r, u], α)
                    end

                    α *= τ
                    F[c, u] -= α

                    for r in c + 1:t
                        F[r, u] = muladd(-α, F[r, c], F[r, u])
                    end
                end
            end
        end
    end

    return
end

function lowrank_sparse_factor_2!(
        F₂₂::AbstractMatrix{T},
        Sptr::AbstractVector{I},
        Wval::AbstractVector{T},
        r0::I,
        uplo::Val{UPLO},
    ) where {UPLO, T, I <: Integer}
    n = size(F₂₂, UPLO === :L ? 1 : 2)

    rstrt = 1

    @inbounds while rstrt <= n
        rstop = min(rstrt + LOWRANK_BLOCK - 1, n)
        cstop = Sptr[r0 + rstop]

        if cstop >= rstrt
            if (n - rstop) * (cstop - rstrt + 1) >= LOWRANK_LARGE
                rsize = rstop - rstrt + 1
                csize = cstop - rstrt + 1
                τsize = min(rsize, csize)
                wsize = τsize * max(rsize, csize, n - rstop)

                work1 = view(Wval,                 1:τsize * τsize)
                work2 = view(Wval, τsize * τsize + 1:τsize * τsize + wsize)

                W = reshape(work1, τsize, τsize)

                if UPLO === :L
                    A = view(F₂₂, rstrt:rstop, rstrt:cstop)
                    C = view(F₂₂, rstop + 1:n, rstrt:cstop)

                    gelqt!(A, W, work2)
                    gemlqt!(Val(:R), Val(:C), view(A, 1:τsize, :), W, C, work2)
                else
                    A = view(F₂₂, rstrt:cstop, rstrt:rstop)
                    C = view(F₂₂, rstrt:cstop, rstop + 1:n)

                    geqrt!(A, W, work2)
                    gemqrt!(Val(:L), Val(:C), view(A, :, 1:τsize), W, C, work2)
                end
            else
                for r in rstrt:rstop
                    if UPLO === :L
                        lowrank_sparse_block!(F₂₂, Wval, r, Sptr[r0 + r], n, uplo)
                    else
                        lowrank_sparse_block!(F₂₂, r, Sptr[r0 + r], n, uplo)
                    end
                end
            end
        end

        rstrt = rstop + 1
    end

    return
end
