# ===== Pivoted LDLt Factorization with Dynamic Regularization =====

function _ldlt!(F::ChordalLDLt{UPLO, T, I}, ::RowMaximum, signs::AbstractVector, reg::AbstractRegularization, check::Bool, tol::Real) where {UPLO, T, I <: Integer}
    Mptr = FVector{I}(undef, F.S.nMptr)
    Mval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)
    piv  = FVector{I}(undef, F.S.nFval)
    mval = FVector{I}(undef, F.S.nNval)
    fval = FVector{I}(undef, F.S.nFval)
    S = FVector{I}(undef, length(signs))

    @inbounds for i in eachindex(F.perm)
        S[i] = signs[F.perm[i]]
    end

    ldlt_piv_reg_fwd!(
        Mptr, Mval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, F.d, Fval,
        F.S.res, F.S.rel, F.S.chd, piv, F.perm, Val{UPLO}(), S, reg
    )

    F.info[] = zero(I)

    ldlt_piv_bwd!(Mptr, mval, fval, F.S.Dptr, F.S.Lptr, F.Lval, F.S.res, F.S.rel, F.S.sep, F.S.chd, F.perm, Fval, Val{UPLO}())
    cholpiv_rel!(F.S.res, F.S.sep, F.S.rel, F.S.chd)

    @inbounds for i in eachindex(F.invp)
        F.invp[i] = F.perm[F.invp[i]]
    end

    @inbounds for i in eachindex(F.invp)
        F.perm[F.invp[i]] = i
    end

    return F
end

# ============================= ldlt_piv_reg_fwd! =============================

function ldlt_piv_reg_fwd!(
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
        piv::AbstractVector{I},
        invp::AbstractVector{I},
        uplo::Val{UPLO},
        S::AbstractVector,
        R::AbstractRegularization,
    ) where {T, I <: Integer, UPLO}

    ns = zero(I); Mptr[one(I)] = one(I)

    for j in vertices(res)
        ns = ldlt_piv_reg_fwd_loop!(
            Mptr, Mval, Dptr, Dval, Lptr, Lval, d, Fval,
            res, rel, chd, ns, j, piv, invp, uplo, S, R
        )
    end

    return
end

function ldlt_piv_reg_fwd_loop!(
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
        piv::AbstractVector{I},
        invp::AbstractVector{I},
        uplo::Val{UPLO},
        S::AbstractVector,
        R::AbstractRegularization,
    ) where {T, I <: Integer, UPLO}
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
    # L is part of the LDLt factor
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
        M₂₂ = reshape(view(Mval, oneto(zero(I))), zero(I), zero(I))
    end
    #
    # pivoted factorization with regularization
    #
    S₁₁ = view(S, neighbors(res, j))
    ldlt_piv_reg_kernel!(D₁₁, L₂₁, M₂₂, Fval, d₁₁, piv, uplo, S₁₁, R)
    #
    # update invp with local pivot permutation
    # invp maps P-indices to Q-indices
    # piv[k] = p means vertex (offset + p) gets Q-index (offset + k)
    #
    offset = first(neighbors(res, j)) - one(I)

    @inbounds for v in oneto(nn)
        invp[offset + piv[v]] = offset + v
    end

    return ns
end

# ============================= ldlt_piv_reg_kernel! =============================

function ldlt_piv_reg_kernel!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        M::AbstractMatrix{T},
        Wval::AbstractVector{T},
        d::AbstractVector{T},
        P::AbstractVector{<:Integer},
        uplo::Val{UPLO},
        S::AbstractVector,
        R::AbstractRegularization,
    ) where {T, UPLO}
    @assert size(D, 1) == size(D, 1) == length(d)
    @assert size(M, 1) == size(M, 2)
    @assert length(L) <= length(Wval)

    if UPLO === :L
        @assert size(L, 1) == size(M, 1)
        @assert size(L, 2) == size(D, 1)
    else
        @assert size(L, 1) == size(D, 1)
        @assert size(L, 2) == size(M, 1)
    end

    ldlt_piv_reg_factor!(D, L, Wval, d, P, uplo, S, R)

    if !isempty(M) && !isempty(D)
        #
        #     M ← M - L d Lᴴ
        #
        W = view(Wval, eachindex(L))
        W₂₁ = reshape(W, size(L))

        if UPLO === :L
            syrk!(uplo, Val(:N), -one(real(T)), W₂₁, L, d, one(real(T)), M)
        else
            syrk!(uplo, Val(:C), -one(real(T)), W₂₁, L, d, one(real(T)), M)
        end
    end

    return
end

function ldlt_piv_reg_factor!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        Wval::AbstractVector{T},
        d::AbstractVector{T},
        P::AbstractVector{<:Integer},
        uplo::Val{UPLO},
        S::AbstractVector,
        R::DynamicRegularization,
    ) where {T, UPLO}
    @assert size(D, 1) == size(D, 2) == length(d)
    @assert length(D) <= length(Wval)

    if UPLO === :L
        @assert size(L, 2) == size(D, 1)
    else
        @assert size(L, 1) == size(D, 1)
    end

    W = reshape(view(Wval, eachindex(D)), size(D))
    qstrf!(uplo, W, D, d, P, S, R)

    if !isempty(L)
        W = reshape(view(Wval, eachindex(L)), size(L))
        copyrec!(W, L)

        if UPLO === :L
            copyrec!(L, W, axes(L, 1), view(P, axes(D, 1)))
            trsm!(Val(:R), Val(:L), Val(:C), Val(:U), one(T), D, L)

            @inbounds for k in axes(L, 2)
                idk = inv(d[k])

                for i in axes(L, 1)
                    L[i, k] *= idk
                end
            end
        else
            copyrec!(L, W, view(P, axes(D, 1)), axes(L, 2))
            trsm!(Val(:L), Val(:U), Val(:C), Val(:U), one(T), D, L)

            @inbounds for i in axes(L, 2)
                for k in axes(L, 1)
                    L[k, i] *= inv(d[k])
                end
            end
        end
    end

    return
end

function ldlt_piv_reg_factor!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        Wval::AbstractVector{T},
        d::AbstractVector{T},
        P::AbstractVector{<:Integer},
        uplo::Val{UPLO},
        S::AbstractVector,
        R::GMW81,
    ) where {T, UPLO}
    if UPLO === :L
        @assert size(D, 1) == size(D, 2) == size(L, 2) == length(d)
    else
        @assert size(D, 1) == size(D, 2) == size(L, 1) == length(d)
    end

    @inbounds for j in axes(D, 1)
        P[j] = j
    end

    @inbounds for bstrt in 1:THRESHOLD:size(D, 1)
        bstop = min(bstrt + THRESHOLD - 1, size(D, 1))
        bsize = bstop - bstrt + 1

        if UPLO === :L
            Dbb = view(D, bstrt:size(D, 1), bstrt:bstop)
            Lbb = view(L, :, bstrt:bstop)
        else
            Dbb = view(D, bstrt:bstop, bstrt:size(D, 1))
            Lbb = view(L, bstrt:bstop, :)
        end

        dbb = view(d, bstrt:bstop)
        Pbb = view(P, bstrt:size(D, 1))
        Sbb = view(S, bstrt:bstop)

        ldlt_piv_reg_factor_block!(Dbb, Lbb, dbb, Pbb, Sbb, R, uplo)

        if bstop < size(D, 1)
            brest = size(D, 1) - bstop
            Drr = view(D, bstop + 1:size(D, 1), bstop + 1:size(D, 1))

            if UPLO === :L
                Drb = view(D, bstop + 1:size(D, 1), bstrt:bstop)
                Lrb = view(L, :, bstop + 1:size(D, 1))
                W = reshape(view(Wval, 1:brest * bsize), brest, bsize)

                syrk!(uplo, Val(:N), -one(real(T)), W, Drb, dbb, one(real(T)), Drr)
                gemm!(Val(:N), Val(:C), -one(T), Lbb, Drb, one(T), Lrb)
            else
                Dbr = view(D, bstrt:bstop, bstop + 1:size(D, 1))
                Lbr = view(L, bstop + 1:size(D, 1), :)
                W = reshape(view(Wval, 1:bsize * brest), bsize, brest)

                syrk!(uplo, Val(:C), -one(real(T)), W, Dbr, dbb, one(real(T)), Drr)
                gemm!(Val(:C), Val(:N), -one(T), Dbr, Lbb, one(T), Lbr)
            end
        end
    end

    return
end

# ============================= ldlt_piv_reg_factor_block! =============================

function ldlt_piv_reg_factor_block!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        d::AbstractVector{T},
        P::AbstractVector,
        S::AbstractVector,
        R::GMW81,
        uplo::Val{:L},
    ) where {T}
    @assert size(D, 2) == size(L, 2) == length(d)

    @inbounds for j in axes(D, 2)
        maxval = abs(real(D[j, j]))
        maxind = j

        for i in j + 1:size(D, 1)
            absAii = abs(real(D[i, i]))

            if absAii > maxval
                maxval = absAii
                maxind = i
            end
        end

        if maxind != j
            swaptri!(D, j, maxind, uplo)
            swapcol!(L, j, maxind)
            swaprec!(P, j, maxind)
            swaprec!(S, j, maxind)
        end

        for k in 1:j - 1
            cDkj = d[k] * conj(D[j, k])

            for i in j + 1:size(D, 1)
                D[i, j] -= D[i, k] * cDkj
            end
        end

        for k in 1:j - 1
            cDkj = d[k] * conj(D[j, k])

            for i in axes(L, 1)
                L[i, j] -= L[i, k] * cDkj
            end
        end

        Djj = real(D[j, j])

        for k in 1:j - 1
            Djj -= d[k] * abs2(D[j, k])
        end

        Djj = d[j] = regularize(R, S, D, L, Djj, j, uplo); iDjj = inv(Djj)

        for i in j + 1:size(D, 1)
            D[i, j] *= iDjj
        end

        for i in axes(L, 1)
            L[i, j] *= iDjj
        end
    end

    return
end

function ldlt_piv_reg_factor_block!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        d::AbstractVector{T},
        P::AbstractVector,
        S::AbstractVector,
        R::GMW81,
        uplo::Val{:U},
    ) where {T}
    @assert size(D, 1) == size(L, 1) == length(d)

    @inbounds for j in axes(D, 1)
        maxval = abs(real(D[j, j]))
        maxind = j

        for i in j + 1:size(D, 2)
            absAii = abs(real(D[i, i]))

            if absAii > maxval
                maxval = absAii
                maxind = i
            end
        end

        if maxind != j
            swaptri!(D, j, maxind, uplo)
            swaprow!(L, j, maxind)
            swaprec!(P, j, maxind)
            swaprec!(S, j, maxind)
        end

        for i in j + 1:size(D, 2)
            for k in 1:j - 1
                D[j, i] -= D[k, i] * d[k] * conj(D[k, j])
            end
        end

        for i in axes(L, 2)
            for k in 1:j - 1
                L[j, i] -= L[k, i] * d[k] * conj(D[k, j])
            end
        end

        Djj = real(D[j, j])

        for k in 1:j - 1
            Djj -= d[k] * abs2(D[k, j])
        end

        Djj = d[j] = regularize(R, S, D, L, Djj, j, uplo); iDjj = inv(Djj)

        for i in j + 1:size(D, 2)
            D[j, i] *= iDjj
        end

        for i in axes(L, 2)
            L[j, i] *= iDjj
        end
    end

    return
end
