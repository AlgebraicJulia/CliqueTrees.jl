function chol!(F::ChordalFactorization{DIAG, UPLO, T, I}, ::NoPivot, S::AbstractVector{T}, R::SE99, check::Bool, tol::Real, diag::Val{DIAG}) where {DIAG, UPLO, T, I <: Integer}
    @assert checksigns(S, R)

    Mptr = FVector{I}(undef, F.S.nMptr)
    Mval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)
    Eval = FVector{T}(undef, F.S.nFval)

    if DIAG === :U
        d = F.d
    else
        d = FVector{T}(undef, size(F, 1))
    end

    foreachfront(ChordalTriangular(F)) do D, L, res, sep
        d[res] .= view(parent(D), diagind(D))
    end

    chol_se99_impl!(Mptr, Mval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, d, Eval, Fval, F.S.res, F.S.rel, F.S.sep, F.S.chd, Val{UPLO}(), S, R, diag)

    F.info[] = zero(I)
    return F
end

function chol_se99_impl!(
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
        uplo::Val{UPLO},
        S::AbstractVector{T},
        R::SE99,
        diag::Val{DIAG},
    ) where {T, I <: Integer, UPLO, DIAG}

    ns = zero(I); Mptr[one(I)] = one(I)
    phase = true
    delta = zero(real(T))

    for j in vertices(res)
        ns, delta, phase = chol_se99_loop!(
            Mptr, Mval, Dptr, Dval, Lptr, Lval, d, Eval, Fval,
            res, rel, sep, chd, ns, j, uplo, S, R, diag, phase, delta
        )
    end
end

function chol_se99_loop!(
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
        uplo::Val{UPLO},
        S::AbstractVector{T},
        R::SE99,
        diag::Val{DIAG},
        phase::Bool,
        delta::Real,
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
    #     L = [ D₁₁ ] res(j)
    #         [ L₂₁ ] sep(j)
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
    # e₂₂ is the separator diagonal vector
    #
    #     e₂₂ = diag(F₂₂)
    #
    d₂₂ = view(d, neighbors(sep, j))
    e₂₂ = view(Eval, oneto(na))
    copyrec!(e₂₂, d₂₂)
    #
    #     F ← 0
    #
    zerotri!(F, uplo)

    for i in Iterators.reverse(neighbors(chd, j))
        #
        # add the update matrix for child i to F
        #
        #     F ← F + Rᵢ Sᵢ Rᵢᵀ
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
    # factorize D₁₁
    #
    #     D₁₁ ← cholesky(D₁₁)
    #
    delta, phase = chol_se99_kernel!(D₁₁, L₂₁, M₂₂, Fval, d₁₁, e₂₂, S₁₁, S₂₂, R, phase, delta, uplo, diag)
    #
    # copy separator diagonal back to global
    #
    #     d₂₂ ← e₂₂
    #
    copyrec!(d₂₂, e₂₂)

    return ns, delta, phase
end

function chol_se99_kernel!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        M::AbstractMatrix{T},
        W::AbstractVector{T},
        d::AbstractVector{T},
        e::AbstractVector{T},
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
    # factorize D
    #
    j = 1

    if phase
        j, delta, phase = chol_se99_factor!(D, L, W, d, e, S, Se, R, delta, j, uplo, diag, Val(true))
    end

    if !phase
        j, delta, phase = chol_se99_factor!(D, L, W, d, e, S, Se, R, delta, j, uplo, diag, Val(false))
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

function chol_se99_factor_loop!(
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

function chol_se99_factor!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        W::AbstractVector{T},
        d::AbstractVector{T},
        e::AbstractVector{T},
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
        j, delta, newphase = chol_se99_factor_block!(D, L, d, e, S, Se, R, delta, bstrt, bstop, uplo, diag, phase)

        if PHASE && !newphase
            j, delta = chol_se99_factor_block!(D, L, d, e, S, Se, R, delta, j, bstop, uplo, diag, Val(false))
            chol_se99_factor_loop!(D, L, W, d, bstrt, bstop, uplo, diag)

            for cstrt in bstop + 1:THRESHOLD:n
                cstop = min(cstrt + THRESHOLD - 1, n)

                j, delta = chol_se99_factor_block!(D, L, d, e, S, Se, R, delta, cstrt, cstop, uplo, diag, Val(false))
                chol_se99_factor_loop!(D, L, W, d, cstrt, cstop, uplo, diag)
            end

            return n + 1, delta, false
        end

        chol_se99_factor_loop!(D, L, W, d, bstrt, bstop, uplo, diag)

        bstrt = bstop + 1
        bstop = min(bstrt + THRESHOLD - 1, n)
    end

    return n + 1, delta, PHASE
end

function chol_se99_factor_block!(
        L₁::AbstractMatrix{T},
        L₂::AbstractMatrix{T},
        d₁::AbstractVector{T},
        d₂::AbstractVector{T},
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
        # Check A: diagonal acceptability
        #
        if PHASE && !checkdiag(R, S₁, S₂, d₁, d₂, real(S₁[j]) * real(d₁[j]), j)
            return j, delta, false
        end
        #
        # Left-looking update
        #
        for k in bstrt:j - 1
            Ljk = L₁[j, k]; cLjk = conj(Ljk)

            for i in j+1:n
                if DIAG === :N
                    L₁[i, j] -= L₁[i, k] * cLjk
                else
                    L₁[i, j] -= L₁[i, k] * d₁[k] * cLjk
                end
            end
        end

        for k in bstrt:j - 1
            Ljk = L₁[j, k]; cLjk = conj(Ljk)

            for i in axes(L₂, 1)
                if DIAG === :N
                    L₂[i, j] -= L₂[i, k] * cLjk
                else
                    L₂[i, j] -= L₂[i, k] * d₁[k] * cLjk
                end
            end
        end

        Ljj = real(d₁[j])

        if PHASE && !lookahead(R, S₁, S₂, L₁, L₂, d₁, d₂, j, uplo)
            return j, delta, false
        elseif !PHASE
            Ljj, delta = regularize(R, S₁, L₁, L₂, Ljj, j, delta, uplo)
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

function chol_se99_factor_block!(
        L₁::AbstractMatrix{T},
        L₂::AbstractMatrix{T},
        d₁::AbstractVector{T},
        d₂::AbstractVector{T},
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
        # Check A: diagonal acceptability
        #
        if PHASE && !checkdiag(R, S₁, S₂, d₁, d₂, real(S₁[j]) * real(d₁[j]), j)
            return j, delta, false
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

        Ljj = real(d₁[j])

        if PHASE && !lookahead(R, S₁, S₂, L₁, L₂, d₁, d₂, j, uplo)
            return j, delta, false
        elseif !PHASE
            Ljj, delta = regularize(R, S₁, L₁, L₂, Ljj, j, delta, uplo)
        end

        if DIAG === :N
            L₁[j, j] = Ljj = sqrt(Ljj)
        else
            d₁[j] = Ljj
        end

        iLjj = inv(Ljj)

        for i in j+1:n
            L₁[j, i] *= iLjj
        end

        for i in axes(L₂, 2)
            L₂[j, i] *= iLjj
        end

        for i in j+1:n
            if DIAG === :N
                d₁[i] -= abs2(L₁[j, i])
            else
                d₁[i] -= Ljj * abs2(L₁[j, i])
            end
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
