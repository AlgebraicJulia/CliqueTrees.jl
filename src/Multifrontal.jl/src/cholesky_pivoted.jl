function chol!(F::ChordalFactorization{DIAG, UPLO, T, I}, ::RowMaximum, S::AbstractVector{T}, reg::AbstractRegularization, check::Bool, tol::Real, diag::Val{DIAG}) where {DIAG, UPLO, T, I <: Integer}
    @assert checksigns(S, reg)

    Mptr = FVector{I}(undef, F.S.nMptr)
    Mval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)
    piv  = FVector{BlasInt}(undef, F.S.nFval)
    mval = FVector{I}(undef, F.S.nNval)
    fval = FVector{I}(undef, F.S.nFval)

    if reg isa NoRegularization
        R = convert(real(T), tol)
    else
        R = reg
    end

    info = chol_piv_fwd!(
        Mptr, Mval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, F.d, Fval,
        F.S.res, F.S.rel, F.S.chd, piv, F.perm, S, R, Val{UPLO}(), diag
    )

    if isnegative(info)
        throw(ArgumentError(info))
    end

    F.info[] = zero(I)

    chol_piv_bwd!(Mptr, mval, fval, F.S.Dptr, F.S.Lptr, F.Lval, F.S.res, F.S.rel, F.S.sep, F.S.chd, F.perm, Fval, Val{UPLO}())
    chol_piv_rel!(F.S.res, F.S.sep, F.S.rel, F.S.chd)

    @inbounds for i in eachindex(F.invp)
        F.invp[i] = F.perm[F.invp[i]]
    end

    @inbounds for i in eachindex(F.invp)
        F.perm[F.invp[i]] = i
    end

    return F
end

# ============================= chol_piv_fwd! =============================
#
# Unified pivoted forward pass for Cholesky (Val{:N}) and LDLt (Val{:U})
#

function chol_piv_fwd!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        d::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        piv::AbstractVector{<:Integer},
        invp::AbstractVector{I},
        S::AbstractVector{T},
        R::Union{AbstractRegularization, Real},
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {T, I <: Integer, UPLO, DIAG}

    ns = zero(I); Mptr[one(I)] = one(I)

    for j in vertices(res)
        ns, info = chol_piv_fwd_loop!(
            Mptr, Mval, Dptr, Dval, Lptr, Lval, d, Fval,
            res, rel, chd, ns, j, piv, invp, S, R, uplo, diag
        )

        if isnegative(info)
            return info
        end
    end

    return zero(I)
end

function chol_piv_fwd_loop!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        d::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        piv::AbstractVector{<:Integer},
        invp::AbstractVector{I},
        S::AbstractVector{T},
        R::Union{AbstractRegularization, Real},
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {T, I <: Integer, UPLO, DIAG}
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
    na = eltypedegree(rel, j)
    #
    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na
    #
    # F is the frontal matrix at node j
    #
    #           nn  na
    #     F = [ F₁₁     ] nn
    #         [ F₂₁ F₂₂ ] na
    #
    F = reshape(view(Fval, oneto(nj * nj)), nj, nj)

    F₁₁ = view(F, oneto(nn),      oneto(nn))
    F₂₂ = view(F, nn + one(I):nj, nn + one(I):nj)

    if UPLO === :L
        F₂₁ = view(F, nn + one(I):nj, oneto(nn))
    else
        F₂₁ = view(F, oneto(nn), nn + one(I):nj)
    end
    #
    # L is part of the factor
    #
    #          res(j)
    #     L = [ D₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    # d₁₁ is the diagonal for the vertices in res(j)
    #
    d₁₁ = view(d, neighbors(res, j))
    #
    #     F ← 0
    #
    zerotri!(F, uplo)

    for i in Iterators.reverse(neighbors(chd, j))
        #
        # add the update matrix for child i to F
        #
        #   F ← F + Rᵢ Sᵢ Rᵢᵀ
        #
        chol_send!(F, Mptr, Mval, rel, ns, i, uplo)
        ns -= one(I)
    end
    #
    # add F to L
    #
    #     L ← L + F
    #
    addtri!(D₁₁, F₁₁, uplo)
    addrec!(L₂₁, F₂₁)
    #
    # M₂₂ is the update matrix for node j
    #
    if ispositive(na)
        ns += one(I)
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * na
        M₂₂ = reshape(view(Mval, strt:stop - one(I)), na, na)
        #
        #     M₂₂ ← F₂₂
        #
        copytri!(M₂₂, F₂₂, uplo)
    else
        M₂₂ = reshape(view(Mval, one(I):zero(I)), zero(I), zero(I))
    end
    #
    # S₁₁ is the signs for the vertices in res(j)
    #
    S₁₁ = view(S, neighbors(res, j))
    #
    # pivoted factorization
    #
    info = chol_piv_kernel!(D₁₁, L₂₁, M₂₂, Fval, d₁₁, piv, S₁₁, R, uplo, diag)
    #
    # update invp with local pivot permutation
    # invp maps P-indices to Q-indices
    # piv[k] = p means vertex (offset + p) gets Q-index (offset + k)
    #
    offset = first(neighbors(res, j)) - one(I)

    @inbounds for v in oneto(nn)
        invp[offset + piv[v]] = offset + v
    end

    return ns, info
end

# ============================= chol_piv_kernel! =============================
#
# Unified pivoted factorization kernel for Cholesky (Val{:N}) and LDLt (Val{:U})
#

function chol_piv_kernel!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        M::AbstractMatrix{T},
        W::AbstractVector{T},
        d::AbstractFill{T},
        P::AbstractVector,
        S::AbstractVector{T},
        R::Union{AbstractRegularization, Real},
        uplo::Val{UPLO},
        diag::Val{:N},
    ) where {T, UPLO}
    return chol_piv_kernel!(D, L, M, W, W, P, S, R, uplo, diag)
end

function chol_piv_kernel!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        M::AbstractMatrix{T},
        W::AbstractVector{T},
        d::AbstractVector{T},
        P::AbstractVector,
        S::AbstractVector{T},
        R::Union{AbstractRegularization, Real},
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {T, UPLO, DIAG}
    @assert size(D, 1) == size(D, 2)
    @assert size(M, 1) == size(M, 2)

    if UPLO === :L
        @assert size(L, 1) == size(M, 1)
        @assert size(L, 2) == size(D, 1)
    else
        @assert size(L, 1) == size(D, 1)
        @assert size(L, 2) == size(M, 1)
    end

    info, rank = chol_piv_factor!(D, L, W, d, P, S, R, uplo, diag)

    if iszero(info) && !isempty(M) && ispositive(rank)
        #
        # Use only the first `rank` columns/rows for the Schur complement
        #
        rd = view(d, 1:rank)

        if UPLO === :L
            rL = view(L, :, 1:rank)
            trans = Val(:N)
        else
            rL = view(L, 1:rank, :)
            trans = Val(:C)
        end

        syrk!(uplo, trans, -one(real(T)), W, rL, rd, one(real(T)), M, diag)
    end

    return info
end

function chol_piv_factor!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        W::AbstractVector{T},
        d::AbstractVector{T},
        P::AbstractVector,
        S::AbstractVector{T},
        tol::Real,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {T, UPLO, DIAG}
    @assert size(D, 1) == size(D, 2)

    if UPLO === :L
        @assert size(L, 2) == size(D, 1)
    else
        @assert size(L, 1) == size(D, 1)
    end
    #
    # factorize D with pivoting
    #
    #     Pᵀ D P = L Lᵀ
    #
    info, rank = pstrf!(uplo, W, D, d, P, S, NoRegularization(), tol, diag)
    #
    # zero out the rank-deficient part of D and L
    #
    zerotri!(D, uplo, rank + 1:size(D, 1))

    if UPLO === :L
        zerorec!(L, axes(L, 1), rank + 1:size(D, 1))
    else
        zerorec!(L, rank + 1:size(D, 1), axes(L, 2))
    end

    if iszero(info) && !isempty(L) && ispositive(rank)
        #
        # permute L by pivot
        #
        M = reshape(view(W, 1:length(L)), size(L))
        copyrec!(M, L)

        if UPLO === :L
            copyrec!(L, M, axes(L, 1), view(P, axes(D, 1)))
        else
            copyrec!(L, M, view(P, axes(D, 1)), axes(L, 2))
        end
        #
        # Use only the first `rank` columns/rows for the solve
        #
        rD = view(D, 1:rank, 1:rank)
        rd = view(d, 1:rank)

        if UPLO === :L
            rL = view(L, :, 1:rank)
            side = Val(:R)
        else
            rL = view(L, 1:rank, :)
            side = Val(:L)
        end
        #
        #     rL ← rL rD⁻ᴴ       (Cholesky)
        #     rL ← rL rD⁻ᴴ rd⁻¹  (LDLt)
        #
        trsm!(side, uplo, Val(:C), diag, one(T), rD, rL)
        cdiv!(side, diag, rL, rd)
    end

    return info, rank
end

# ============================= chol_piv_bwd! =============================

function chol_piv_bwd!(
        mptr::AbstractVector{I},
        mval::AbstractVector{I},
        fval::AbstractVector{I},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        sep::AbstractGraph{I},
        chd::AbstractGraph{I},
        invp::AbstractVector{I},
        Fval::AbstractVector{T},
        uplo::Val{UPLO},
    ) where {T, I <: Integer, UPLO}

    ns = zero(I); mptr[one(I)] = one(I)

    for j in reverse(vertices(res))
        ns = chol_piv_bwd_loop!(
            mptr, mval, fval, Dptr, Lptr, Lval,
            res, rel, sep, chd, ns, j, invp, Fval, uplo
        )
    end

    return
end

function chol_piv_bwd_loop!(
        mptr::AbstractVector{I},
        mval::AbstractVector{I},
        fval::AbstractVector{I},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        sep::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        invp::AbstractVector{I},
        Fval::AbstractVector{T},
        uplo::Val{UPLO},
    ) where {T, I <: Integer, UPLO}
    #
    # nn is the size of the residual at node j
    #
    #     nn = |res(j)|
    #
    nn = eltypedegree(res, j)
    #
    # na is the size of the separator at node j
    #
    #     na = |sep(j)|
    #
    na = eltypedegree(rel, j)
    #
    # nj is the size of the bag at node j
    #
    #     nj = |bag(j)|
    #
    nj = nn + na
    #
    # f is the frontal Q-index vector at node j
    #
    #   f = [ f₁ ] nn
    #       [ f₂ ] na
    #
    f  = view(fval, oneto(nj))
    f₁ = view(f,    oneto(nn))
    f₂ = view(f,    nn + one(I):nj)
    #
    #     f₁ ← invp[res(j)]
    #
    copyrec!(f₁, view(invp, neighbors(res, j)))
    #
    # pull Q-indices from ancestor's update matrix
    #
    #     f₂ ← m₂
    #
    if ispositive(na)
        strt = mptr[ns]
        m₂ = view(mval, strt:strt + na - one(I))
        ns -= one(I)
        copyrec!(f₂, m₂)
    end

    for i in neighbors(chd, j)
        #
        # push f restricted to sep(i) to child i
        #
        #     mᵢ ← Rᵢᵀ f
        #
        ns += one(I)
        chol_piv_bwd_update!(f, mptr, mval, rel, ns, i)
    end
    #
    # s₂ is the sorting permutation for sep(j) by Q-index
    #
    s₂ = neighbors(sep, j)
    sortperm!(s₂, f₂; alg=QuickSort, initialized=false)
    #
    # L₂₁ is the off-diagonal block of the Cholesky factor
    #
    #        nn
    #   L = [ L₂₁ ] na
    #
    Lp = Lptr[j]

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
        F₂₁ = reshape(view(Fval, oneto(na * nn)),           na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
        F₂₁ = reshape(view(Fval, oneto(nn * na)),           nn, na)
    end
    #
    # permute rows/cols of L₂₁ by s₂
    #
    copyrec!(F₂₁, L₂₁)

    if UPLO === :L
        @inbounds for k in oneto(nn)
            for i in oneto(na)
                L₂₁[i, k] = F₂₁[s₂[i], k]
            end
        end
    else
        @inbounds for i in oneto(na)
            for k in oneto(nn)
                L₂₁[k, i] = F₂₁[k, s₂[i]]
            end
        end
    end
    #
    # update sep(j) targets with sorted Q-indices
    #
    #     sep(j) ← f₂[s₂]
    #
    @inbounds for i in oneto(na)
        s₂[i] = f₂[s₂[i]]
    end

    return ns
end

function chol_piv_bwd_update!(
        f::AbstractVector{I},
        ptr::AbstractVector{I},
        val::AbstractVector{I},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
    ) where {I <: Integer}
    #
    # na is the size of the separator at node i
    #
    #     na = |sep(i)|
    #
    na = eltypedegree(rel, i)
    #
    # inj: sep(i) → bag(parent(i))
    #
    inj = neighbors(rel, i)
    #
    # mᵢ ← Rᵢᵀ f
    #
    strt = ptr[ns]
    stop = ptr[ns + one(I)] = strt + na
    m = view(val, strt:stop - one(I))

    copyrec!(m, f, inj)

    return
end

# ============================= chol_piv_rel! =============================

function chol_piv_rel!(
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
    ) where {I <: Integer}

    for j in vertices(res)
        chol_piv_rel_loop!(res, sep, rel, chd, j)
    end

    return
end

function chol_piv_rel_loop!(
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        j::I,
    ) where {I <: Integer}
    #
    # nn is the size of the residual at node j
    #
    #     nn = |res(j)|
    #
    nn = eltypedegree(res, j)
    #
    # na is the size of the separator at node j
    #
    #     na = |sep(j)|
    #
    na = eltypedegree(sep, j)
    #
    # bag(j) = res(j) ∪ sep(j), both sorted by Q-index
    #
    res_j = neighbors(res, j)
    sep_j = neighbors(sep, j)
    #
    # for each child i, recompute rel(i)
    #
    #     rel(i) : sep(i) → bag(j)
    #
    for i in neighbors(chd, j)
        #
        # sep(i) ⊆ bag(j), both sorted by Q-index
        # use merge-style traversal to find positions
        #
        q = one(I)

        for p in incident(sep, i)
            v = targets(sep)[p]
            #
            # find v in bag(j)
            # bag(j)[q] = res_j[q] if q ≤ nn, else sep_j[q - nn]
            #
            while true
                if q <= nn
                    w = res_j[q]
                else
                    w = sep_j[q - nn]
                end

                if w >= v
                    break
                end
                q += one(I)
            end

            targets(rel)[p] = q
        end
    end

    return
end

function chol_piv_factor!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        W::AbstractVector{T},
        d::AbstractVector{T},
        P::AbstractVector,
        S::AbstractVector{T},
        R::DynamicRegularization,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {T, UPLO, DIAG}
    @assert size(D, 1) == size(D, 2)
    @assert length(D) <= length(W)

    if DIAG === :U
        @assert length(d) == size(D, 1)
    end

    if UPLO === :L
        @assert size(L, 2) == size(D, 1)
    else
        @assert size(L, 1) == size(D, 1)
    end

    pstrf!(uplo, W, D, d, P, S, R, -one(real(T)), diag)

    if !isempty(L)
        M = reshape(view(W, 1:length(L)), size(L))
        copyrec!(M, L)

        if UPLO === :L
            copyrec!(L, M, axes(L, 1), view(P, axes(D, 1)))
            side = Val(:R)
        else
            copyrec!(L, M, view(P, axes(D, 1)), axes(L, 2))
            side = Val(:L)
        end

        trsm!(side, uplo, Val(:C), diag, one(T), D, L)
        cdiv!(side, diag, L, d)
    end

    return 0, size(D, 1)
end

function chol_piv_factor!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        W::AbstractVector{T},
        d::AbstractVector{T},
        P::AbstractVector,
        S::AbstractVector{T},
        R::GMW81,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {T, UPLO, DIAG}
    if UPLO === :L
        @assert size(D, 1) == size(D, 2) == size(L, 2)
    else
        @assert size(D, 1) == size(D, 2) == size(L, 1)
    end

    @inbounds for j in axes(D, 1)
        P[j] = j

        if UPLO === :L
            d[j] = zero(real(T))
        else
            d[j] = real(D[j, j])
        end
    end

    @inbounds for bstrt in 1:THRESHOLD:size(D, 1)
        bstop = min(bstrt + THRESHOLD - 1, size(D, 1))

        chol_piv_factor_block!(D, L, d, P, S, R, bstrt, bstop, uplo, diag)

        if bstop < size(D, 1)
            Drr = view(D, bstop + 1:size(D, 1), bstop + 1:size(D, 1))
            dbb = view(d, bstrt:bstop)

            if UPLO === :L
                Drb = view(D, bstop + 1:size(D, 1), bstrt:bstop)
                syrk!(uplo, Val(:N), -one(real(T)), W, Drb, dbb, one(real(T)), Drr, diag)

                if !isempty(L)
                    Lnb = view(L, :, bstrt:bstop)
                    Lnr = view(L, :, bstop + 1:size(D, 1))
                    gemm!(Val(:N), Val(:C), -one(T), W, Lnb, Drb, dbb, one(T), Lnr, diag)
                end

                @inbounds for j in bstop + 1:size(D, 1)
                    d[j] = zero(real(T))
                end
            else
                Drb = view(D, bstrt:bstop, bstop + 1:size(D, 1))
                syrk!(uplo, Val(:C), -one(real(T)), W, Drb, dbb, one(real(T)), Drr, diag)

                if !isempty(L)
                    Lnb = view(L, bstrt:bstop, :)
                    Lnr = view(L, bstop + 1:size(D, 1), :)
                    gemm!(Val(:C), Val(:N), -one(T), W, Drb, Lnb, dbb, one(T), Lnr, diag)
                end
            end
        end
    end

    return 0, size(D, 1)
end

function chol_piv_factor_block!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        d::AbstractVector{T},
        P::AbstractVector,
        S::AbstractVector{T},
        R::GMW81,
        bstrt::Int,
        bstop::Int,
        ::Val{:L},
        ::Val{DIAG},
    ) where {T, DIAG}

    @inbounds for j in bstrt:bstop
        maxval = abs(real(D[j, j]) - d[j])
        maxind = j

        for i in j + 1:size(D, 1)
            Aii = abs(real(D[i, i]) - d[i])

            if Aii > maxval
                maxval = Aii
                maxind = i
            end
        end

        if maxind != j
            swaptri!(D, j, maxind, Val(:L))
            swapcol!(L, j, maxind)
            swaprec!(P, j, maxind)
            swaprec!(d, j, maxind)
            swaprec!(S, j, maxind)
        end

        for k in bstrt:j - 1
            if DIAG === :U
                cDkj = d[k] * conj(D[j, k])
            else
                cDkj = conj(D[j, k])
            end

            for i in j + 1:size(D, 1)
                D[i, j] -= D[i, k] * cDkj
            end
        end

        for k in bstrt:j - 1
            if DIAG === :U
                cDkj = d[k] * conj(D[j, k])
            else
                cDkj = conj(D[j, k])
            end

            for i in axes(L, 1)
                L[i, j] -= L[i, k] * cDkj
            end
        end

        Djj = real(D[j, j]) - real(d[j])

        Djj = regularize(R, S, D, L, Djj, j, Val(:L))

        if DIAG === :U
            d[j] = Djj
        else
            D[j, j] = Djj = sqrt(Djj)
        end

        iDjj = inv(Djj)

        for i in j + 1:size(D, 1)
            D[i, j] *= iDjj
        end

        for i in axes(L, 1)
            L[i, j] *= iDjj
        end

        for i in j + 1:size(D, 1)
            if DIAG === :N
                d[i] += abs2(D[i, j])
            else
                d[i] += Djj * abs2(D[i, j])
            end
        end
    end

    return
end

function chol_piv_factor_block!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        d::AbstractVector{T},
        P::AbstractVector,
        S::AbstractVector{T},
        R::GMW81,
        bstrt::Int,
        bstop::Int,
        ::Val{:U},
        ::Val{DIAG},
    ) where {T, DIAG}

    @inbounds for j in bstrt:bstop
        maxval = abs(d[j])
        maxind = j

        for i in j + 1:size(D, 1)
            Aii = abs(d[i])

            if Aii > maxval
                maxval = Aii
                maxind = i
            end
        end

        if maxind != j
            swaptri!(D, j, maxind, Val(:U))
            swaprow!(L, j, maxind)
            swaprec!(P, j, maxind)
            swaprec!(d, j, maxind)
            swaprec!(S, j, maxind)
        end

        for i in j + 1:size(D, 1)
            for k in bstrt:j - 1
                if DIAG === :U
                    D[j, i] -= D[k, i] * d[k] * conj(D[k, j])
                else
                    D[j, i] -= D[k, i] * conj(D[k, j])
                end
            end
        end

        for i in axes(L, 2)
            for k in bstrt:j - 1
                if DIAG === :U
                    L[j, i] -= L[k, i] * d[k] * conj(D[k, j])
                else
                    L[j, i] -= L[k, i] * conj(D[k, j])
                end
            end
        end

        Djj = real(d[j])

        Djj = regularize(R, S, D, L, Djj, j, Val(:U))

        if DIAG === :U
            d[j] = Djj
        else
            D[j, j] = Djj = sqrt(Djj)
        end

        iDjj = inv(Djj)

        for i in j + 1:size(D, 1)
            D[j, i] *= iDjj

            if DIAG === :N
                d[i] -= abs2(D[j, i])
            else
                d[i] -= Djj * abs2(D[j, i])
            end
        end

        for i in axes(L, 2)
            L[j, i] *= iDjj
        end
    end

    return
end
