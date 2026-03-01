function chol!(F::ChordalFactorization{DIAG, UPLO, T, I}, ::RowMaximum, S::AbstractVector{T}, R::SE99, check::Bool, tol::Real, diag::Val{DIAG}) where {DIAG, UPLO, T, I <: Integer}
    @assert checksigns(S, R)

    Mptr = FVector{I}(undef, F.S.nMptr)
    Mval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)
    Eval = FVector{T}(undef, F.S.nFval)
    piv  = FVector{I}(undef, F.S.nFval)
    mval = FVector{I}(undef, F.S.nNval)
    fval = FVector{I}(undef, F.S.nFval)

    if DIAG === :U
        d = F.d
    else
        d = FVector{T}(undef, size(F, 1))
    end

    foreachfront(ChordalTriangular(F)) do D, L, res, sep
        d[res] .= view(parent(D), diagind(D))
    end

    chol_se99_piv_fwd!(
        Mptr, Mval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, d, Eval, Fval,
        F.S.res, F.S.rel, F.S.sep, F.S.chd, piv, F.perm, S, R, Val{UPLO}(), diag
    )

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

# ============================= chol_se99_piv_fwd! =============================

function chol_se99_piv_fwd!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        d::AbstractVector{T},
        Eval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        sep::AbstractGraph{I},
        chd::AbstractGraph{I},
        piv::AbstractVector{<:Integer},
        invp::AbstractVector{I},
        S::AbstractVector{T},
        R::SE99,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {T, I <: Integer, UPLO, DIAG}

    ns = zero(I); Mptr[one(I)] = one(I)
    phase = true
    delta = zero(real(T))

    for j in vertices(res)
        ns, delta, phase = chol_se99_piv_fwd_loop!(
            Mptr, Mval, Dptr, Dval, Lptr, Lval, d, Eval, Fval,
            res, rel, sep, chd, ns, j, piv, invp, S, R, uplo, diag, phase, delta
        )
    end
end

function chol_se99_piv_fwd_loop!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        d::AbstractVector{T},
        Eval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        sep::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        piv::AbstractVector{<:Integer},
        invp::AbstractVector{I},
        S::AbstractVector{T},
        R::SE99,
        uplo::Val{UPLO},
        diag::Val{DIAG},
        phase::Bool,
        delta::Real,
    ) where {T, I <: Integer, UPLO, DIAG}
    #
    # nn is the size of the residual at node j
    #
    nn = eltypedegree(res, j)
    #
    # na is the size of the separator at node j
    #
    na = eltypedegree(rel, j)
    #
    # nj is the size of the bag at node j
    #
    nj = nn + na
    #
    # F is the frontal matrix at node j
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
    Dp = Dptr[j]
    Lp = Lptr[j]
    d₁₁ = view(d, neighbors(res, j))
    S₁₁ = view(S, neighbors(res, j))
    S₂₂ = view(S, neighbors(sep, j))
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    # g₁₁ is the Gerschgorin bounds vector
    # e₂₂ is the separator diagonal vector
    #
    d₂₂ = view(d, neighbors(sep, j))
    g₁₁ = view(Eval, oneto(nn))
    e₂₂ = view(Eval, nn + one(I):nj)
    copyrec!(e₂₂, d₂₂)
    #
    #     F ← 0
    #
    zerotri!(F, uplo)

    for i in Iterators.reverse(neighbors(chd, j))
        #
        # add the update matrix for child i to F
        #
        chol_send!(F, Mptr, Mval, rel, ns, i, uplo)
        ns -= one(I)
    end
    #
    # add F to L
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
    # pivoted factorization
    #
    delta, phase = chol_se99_piv_kernel!(D₁₁, L₂₁, M₂₂, Fval, d₁₁, e₂₂, g₁₁, piv, S₁₁, S₂₂, R, phase, delta, uplo, diag)
    #
    # copy separator diagonal back to global
    #
    copyrec!(d₂₂, e₂₂)
    #
    # update invp with local pivot permutation
    #
    offset = first(neighbors(res, j)) - one(I)

    @inbounds for v in oneto(nn)
        invp[offset + piv[v]] = offset + v
    end

    return ns, delta, phase
end

# ============================= chol_se99_piv_kernel! =============================

function chol_se99_piv_kernel!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        M::AbstractMatrix{T},
        W::AbstractVector{T},
        d::AbstractVector{T},
        e::AbstractVector{T},
        g::AbstractVector{T},
        P::AbstractVector,
        S::AbstractVector{T},
        Se::AbstractVector{T},
        R::SE99,
        phase::Bool,
        delta::Real,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {T, UPLO, DIAG}
    @assert size(D, 1) == size(D, 2)
    @assert size(M, 1) == size(M, 2)
    #
    # initialize P
    #
    @inbounds for i in axes(D, 1)
        P[i] = i
    end
    #
    # factorize D
    #
    j = 1

    if phase
        j, delta, phase = chol_se99_piv_factor!(D, L, W, d, e, g, P, S, Se, R, delta, j, uplo, diag, Val(true))
    end

    if !phase
        j, delta, phase = chol_se99_piv_factor!(D, L, W, d, e, g, P, S, Se, R, delta, j, uplo, diag, Val(false))
    end

    if !isempty(M)
        if UPLO === :L
            trans = Val(:N)
        else
            trans = Val(:C)
        end
        #
        #     M ← M - L D Lᴴ
        #
        syrk!(uplo, trans, -one(real(T)), W, L, d, one(real(T)), M, diag)
    end

    return delta, phase
end

# ============================= chol_se99_piv_factor! =============================

function chol_se99_piv_factor_loop!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        W::AbstractVector{T},
        d::AbstractVector{T},
        bstrt::Int,
        bstop::Int,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {T, UPLO, DIAG}

    n = size(D, 1)
    bstop >= n && return

    Drr = view(D, bstop + 1:n, bstop + 1:n)
    dbb = view(d, bstrt:bstop)

    if UPLO === :L
        Drb = view(D, bstop + 1:n, bstrt:bstop)
        syrk!(Val(:L), Val(:N), -one(real(T)), W, Drb, dbb, one(real(T)), Drr, diag)
    else
        Drb = view(D, bstrt:bstop, bstop + 1:n)
        syrk!(Val(:U), Val(:C), -one(real(T)), W, Drb, dbb, one(real(T)), Drr, diag)
    end

    if !isempty(L)
        if UPLO === :L
            Lnb = view(L, :, bstrt:bstop)
            Lnr = view(L, :, bstop + 1:n)
            gemm!(Val(:N), Val(:C), -one(T), W, Lnb, Drb, dbb, one(T), Lnr, diag)
        else
            Lnb = view(L, bstrt:bstop, :)
            Lnr = view(L, bstop + 1:n, :)
            gemm!(Val(:C), Val(:N), -one(T), W, Drb, Lnb, dbb, one(T), Lnr, diag)
        end
    end

    return
end

function chol_se99_piv_factor!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        W::AbstractVector{T},
        d::AbstractVector{T},
        e::AbstractVector{T},
        g::AbstractVector{T},
        P::AbstractVector,
        S::AbstractVector{T},
        Se::AbstractVector{T},
        R::SE99,
        delta::Real,
        bstrt::Int,
        uplo::Val{UPLO},
        diag::Val{DIAG},
        phase::Val{PHASE},
    ) where {T, UPLO, DIAG, PHASE}

    n = size(D, 1)
    bstop = min(cld(bstrt, THRESHOLD) * THRESHOLD, n)

    @inbounds while bstrt <= n
        j, delta, newphase = chol_se99_piv_factor_block!(D, L, d, e, g, P, S, Se, R, delta, bstrt, bstop, uplo, diag, phase)

        if PHASE && !newphase
            j, delta = chol_se99_piv_factor_block!(D, L, d, e, g, P, S, Se, R, delta, j, bstop, uplo, diag, Val(false))
            chol_se99_piv_factor_loop!(D, L, W, d, bstrt, bstop, uplo, diag)

            for cstrt in bstop + 1:THRESHOLD:n
                cstop = min(cstrt + THRESHOLD - 1, n)

                j, delta = chol_se99_piv_factor_block!(D, L, d, e, g, P, S, Se, R, delta, cstrt, cstop, uplo, diag, Val(false))
                chol_se99_piv_factor_loop!(D, L, W, d, cstrt, cstop, uplo, diag)
            end

            return n + 1, delta, false
        end

        chol_se99_piv_factor_loop!(D, L, W, d, bstrt, bstop, uplo, diag)

        bstrt = bstop + 1
        bstop = min(bstrt + THRESHOLD - 1, n)
    end

    return n + 1, delta, PHASE
end

# ============================= chol_se99_piv_factor_block! =============================

function chol_se99_piv_factor_block!(
        L₁::AbstractMatrix{T},
        L₂::AbstractMatrix{T},
        d₁::AbstractVector{T},
        d₂::AbstractVector{T},
        g::AbstractVector{T},
        P::AbstractVector,
        S₁::AbstractVector{T},
        S₂::AbstractVector{T},
        R::SE99,
        delta::Real,
        bstrt::Int,
        bstop::Int,
        uplo::Val{:L},
        ::Val{DIAG},
        ::Val{PHASE},
    ) where {T, DIAG, PHASE}

    n = size(L₁, 1)

    @inbounds for j in bstrt:bstop
        #
        # Pivot selection
        #
        if PHASE
            # Phase 1: max signed diagonal
            maxval = real(S₁[j]) * real(d₁[j])
            maxind = j

            for i in j + 1:n
                val = real(S₁[i]) * real(d₁[i])

                if val > maxval
                    maxval = val
                    maxind = i
                end
            end

            # Check A: diagonal acceptability (using pivot value)
            if !checkdiag(R, S₁, S₂, d₁, d₂, maxval, j)
                return j, delta, false
            end
        else
            # Phase 2: initialize Gerschgorin bounds
            if j == bstrt
                for i in bstrt:n
                    g[i] = real(S₁[i]) * real(d₁[i])

                    for k in i+1:n
                        g[i] -= abs(L₁[k, i])
                    end

                    for k in axes(L₂, 1)
                        g[i] -= abs(L₂[k, i])
                    end
                end
            end

            # Phase 2: max Gerschgorin bound
            maxval = real(g[j])
            maxind = j

            for i in j + 1:n
                val = real(g[i])

                if val > maxval
                    maxval = val
                    maxind = i
                end
            end
        end

        if maxind != j
            swaptri!(L₁, j, maxind, uplo)
            swapcol!(L₂, j, maxind)
            swaprec!(P, j, maxind)
            swaprec!(d₁, j, maxind)
            swaprec!(S₁, j, maxind)

            if !PHASE
                swaprec!(g, j, maxind)
            end
        end
        #
        # Left-looking update
        #
        for k in bstrt:j-1
            Ljk = L₁[j, k]
            cLjk = conj(Ljk)

            for i in j+1:n
                if DIAG === :N
                    L₁[i, j] -= L₁[i, k] * cLjk
                else
                    L₁[i, j] -= L₁[i, k] * d₁[k] * cLjk
                end
            end
        end

        for k in bstrt:j-1
            Ljk = L₁[j, k]
            cLjk = conj(Ljk)

            for i in axes(L₂, 1)
                if DIAG === :N
                    L₂[i, j] -= L₂[i, k] * cLjk
                else
                    L₂[i, j] -= L₂[i, k] * d₁[k] * cLjk
                end
            end
        end
        #
        # Regularization
        #
        Ljj = real(d₁[j])

        if PHASE && !lookahead(R, S₁, S₂, L₁, L₂, d₁, d₂, j, uplo)
            return j, delta, false
        elseif !PHASE
            Ljj, delta = regularize(R, S₁, L₁, L₂, Ljj, j, delta, uplo)
        end
        #
        # Update Gerschgorin bounds (before scaling)
        #
        if !PHASE
            bound = zero(real(T))

            for i in j+1:n
                bound += abs(L₁[i, j])
            end

            for i in axes(L₂, 1)
                bound += abs(L₂[i, j])
            end

            if Ljj > bound
                temp = one(real(T)) - bound / Ljj

                for i in j+1:n
                    g[i] += abs(L₁[i, j]) * temp
                end
            end
        end

        if DIAG === :N
            L₁[j, j] = Ljj = sqrt(Ljj)
        else
            d₁[j] = Ljj
        end

        iLjj = inv(Ljj)

        for i in j+1:n
            L₁[i, j] *= iLjj
        end

        for i in axes(L₂, 1)
            L₂[i, j] *= iLjj
        end
        #
        # Diagonal update
        #
        for i in j+1:n
            if DIAG === :N
                d₁[i] -= abs2(L₁[i, j])
            else
                d₁[i] -= Ljj * abs2(L₁[i, j])
            end
        end

        if PHASE
            for i in axes(L₂, 1)
                if DIAG === :N
                    d₂[i] -= abs2(L₂[i, j])
                else
                    d₂[i] -= Ljj * abs2(L₂[i, j])
                end
            end
        end
    end

    return bstop + 1, delta, PHASE
end

function chol_se99_piv_factor_block!(
        L₁::AbstractMatrix{T},
        L₂::AbstractMatrix{T},
        d₁::AbstractVector{T},
        d₂::AbstractVector{T},
        g::AbstractVector{T},
        P::AbstractVector,
        S₁::AbstractVector{T},
        S₂::AbstractVector{T},
        R::SE99,
        delta::Real,
        bstrt::Int,
        bstop::Int,
        uplo::Val{:U},
        ::Val{DIAG},
        ::Val{PHASE},
    ) where {T, DIAG, PHASE}

    n = size(L₁, 1)

    @inbounds for j in bstrt:bstop
        #
        # Pivot selection
        #
        if PHASE
            # Phase 1: max signed diagonal
            maxval = real(S₁[j]) * real(d₁[j])
            maxind = j

            for i in j + 1:n
                val = real(S₁[i]) * real(d₁[i])

                if val > maxval
                    maxval = val
                    maxind = i
                end
            end

            # Check A: diagonal acceptability (using pivot value)
            if !checkdiag(R, S₁, S₂, d₁, d₂, maxval, j)
                return j, delta, false
            end
        else
            # Phase 2: initialize Gerschgorin bounds
            if j == bstrt
                for i in bstrt:n
                    g[i] = real(S₁[i]) * real(d₁[i])

                    for k in i+1:n
                        g[i] -= abs(L₁[i, k])
                    end

                    for k in axes(L₂, 2)
                        g[i] -= abs(L₂[i, k])
                    end
                end
            end

            # Phase 2: max Gerschgorin bound
            maxval = real(g[j])
            maxind = j

            for i in j + 1:n
                val = real(g[i])

                if val > maxval
                    maxval = val
                    maxind = i
                end
            end
        end

        if maxind != j
            swaptri!(L₁, j, maxind, uplo)
            swaprow!(L₂, j, maxind)
            swaprec!(P, j, maxind)
            swaprec!(d₁, j, maxind)
            swaprec!(S₁, j, maxind)

            if !PHASE
                swaprec!(g, j, maxind)
            end
        end
        #
        # Left-looking update
        #
        for i in j+1:n
            for k in bstrt:j-1
                if DIAG === :N
                    L₁[j, i] -= L₁[k, i] * conj(L₁[k, j])
                else
                    L₁[j, i] -= L₁[k, i] * d₁[k] * conj(L₁[k, j])
                end
            end
        end

        for i in axes(L₂, 2)
            for k in bstrt:j-1
                if DIAG === :N
                    L₂[j, i] -= L₂[k, i] * conj(L₁[k, j])
                else
                    L₂[j, i] -= L₂[k, i] * d₁[k] * conj(L₁[k, j])
                end
            end
        end
        #
        # Regularization
        #
        Ljj = real(d₁[j])

        if PHASE && !lookahead(R, S₁, S₂, L₁, L₂, d₁, d₂, j, uplo)
            return j, delta, false
        elseif !PHASE
            Ljj, delta = regularize(R, S₁, L₁, L₂, Ljj, j, delta, uplo)
        end
        #
        # Update Gerschgorin bounds (before scaling)
        #
        if !PHASE
            bound = zero(real(T))

            for i in j + 1:n
                bound += abs(L₁[j, i])
            end

            for i in axes(L₂, 2)
                bound += abs(L₂[j, i])
            end

            if Ljj > bound
                temp = one(real(T)) - bound / Ljj

                for i in j+1:n
                    g[i] += abs(L₁[j, i]) * temp
                end
            end
        end

        if DIAG === :N
            L₁[j, j] = Ljj = sqrt(Ljj)
        else
            d₁[j] = Ljj
        end

        iLjj = inv(Ljj)

        for i in j + 1:n
            L₁[j, i] *= iLjj

            if DIAG === :N
                d₁[i] -= abs2(L₁[j, i])
            else
                d₁[i] -= Ljj * abs2(L₁[j, i])
            end
        end

        for i in axes(L₂, 2)
            L₂[j, i] *= iLjj
        end

        if PHASE
            for i in axes(L₂, 2)
                if DIAG === :N
                    d₂[i] -= abs2(L₂[j, i])
                else
                    d₂[i] -= Ljj * abs2(L₂[j, i])
                end
            end
        end
    end

    return bstop + 1, delta, PHASE
end

