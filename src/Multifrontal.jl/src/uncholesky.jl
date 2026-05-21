function uncholesky!(
        F::AbstractCholesky{UPLO, T},
    ) where {UPLO, T}
    F.info[] = unfactorize!(triangular(F), F.d)
    return F
end

function uncholesky!(
        L::ChordalTriangular{:N, UPLO, T, I},
    ) where {UPLO, T, I <: Integer}
    d = Ones{T}(ncl(L))
    return unfactorize!(L, d)
end

function unldlt!(
        F::AbstractLDLt{UPLO, T},
    ) where {UPLO, T}
    F.info[] = unfactorize!(triangular(F), F.d)
    return F
end

function unfactorize!(
        F::AbstractFactorization{DIAG, UPLO, T},
    ) where {DIAG, UPLO, T}
    F.info[] = unfactorize!(triangular(F), F.d)
    return F
end

function unfactorize!(
        L::ChordalTriangular{DIAG, UPLO, T, I},
        d::AbstractVector{T},
    ) where {DIAG, UPLO, T, I <: Integer}
    Mptr = FVector{I}(undef, L.S.nMptr)
    Mval = FVector{T}(undef, L.S.nMval)
    Fval = FVector{T}(undef, L.S.nFval * L.S.nFval)
    Wval = FVector{T}(undef, L.S.nFval * L.S.nFval)

    info = unchol_impl!(Mptr, Mval, Fval, Wval, L, d)
    return info
end

#
# Convenience wrapper that unpacks ChordalTriangular types (with d vector for LDLt).
#
function unchol_impl!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Fval::AbstractVector{T},
        Wval::AbstractVector{T},
        L::ChordalTriangular{DIAG, UPLO, T, I},
        d::AbstractVector{T},
    ) where {DIAG, UPLO, T, I <: Integer}
    info = unchol_impl!(
        Mptr, Mval,
        L.S.Dptr, L.Dval,
        L.S.Lptr, L.Lval,
        d, Fval, Wval,
        L.S.res, L.S.rel, L.S.chd,
        L.uplo, L.diag)

    return info
end

#
# Convenience wrapper for Cholesky (creates d = Ones internally).
#
function unchol_impl!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Fval::AbstractVector{T},
        Wval::AbstractVector{T},
        L::ChordalTriangular{:N, UPLO, T, I},
    ) where {UPLO, T, I <: Integer}
    d = Ones{T}(ncl(L))
    info = unchol_impl!(Mptr, Mval, Fval, Wval, L, d)
    return info
end

function unchol_impl!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        d::AbstractVector{T},
        Fval::AbstractVector{T},
        Wval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {UPLO, DIAG, T, I <: Integer}

    ns = zero(I); Mptr[one(I)] = one(I)

    for j in vertices(res)
        ns = unchol_loop!(Mptr, Mval, Dptr, Dval, Lptr, Lval, d, Fval, Wval, res, rel, chd, ns, j, uplo, diag)
    end

    return zero(I)
end

function unchol_loop!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        d::AbstractVector{T},
        Fval::AbstractVector{T},
        Wval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {UPLO, DIAG, T, I <: Integer}
    nn = eltypedegree(res, j)

    if isone(nn)
        return unchol_loop_nod!(
            Mptr, Mval,
            Dptr, Dval,
            Lptr, Lval,
            d, Fval, Wval,
            res, rel, chd, ns, j, uplo, diag
        )
    else
        return unchol_loop_snd!(
            Mptr, Mval,
            Dptr, Dval,
            Lptr, Lval,
            d, Fval, Wval,
            res, rel, chd, ns, nn, j, uplo, diag
        )
    end
end

function unchol_loop_snd!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        d::AbstractVector{T},
        Fval::AbstractVector{T},
        Wval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        nn::I,
        j::I,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {UPLO, DIAG, T, I <: Integer}
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

    F₁₁ = view(F, oneto(nn), oneto(nn))
    F₂₂ = view(F, nn + one(I):nj, nn + one(I):nj)

    if UPLO === :L
        F₂₁ = view(F, nn + one(I):nj, oneto(nn))
    else
        F₂₁ = view(F, oneto(nn), nn + one(I):nj)
    end
    #
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ D₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    d₁₁ = view(d, neighbors(res, j))
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
        side = Val(:R)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
        side = Val(:L)
    end
    #
    #     F ← 0
    #
    zerotri!(F, uplo)

    for i in Iterators.reverse(neighbors(chd, j))
        #
        # add the update matrix for child i to F
        #
        #     F ← F + Rᵢ Mᵢ Rᵢᵀ
        #
        unchol_recv!(F, Mptr, Mval, rel, ns, i, uplo)
        ns -= one(I)
    end
    #
    # M₂₂ is the update matrix for node j
    #
    # compute M₂₂ BEFORE modifying L₂₁
    #
    #     M₂₂ ← F₂₂ + L₂₁ D L₂₁ᵀ  (D = I for Cholesky, D = diag(d₁₁) for LDLt)
    #
    if ispositive(na)
        ns += one(I)
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * na
        M₂₂ = reshape(view(Mval, strt:stop - one(I)), na, na)
        copytri!(M₂₂, F₂₂, uplo)

        if UPLO === :L
            syrk!(Val(:L), Val(:N), one(real(T)), Wval, L₂₁, d₁₁, one(real(T)), M₂₂, diag)
        else
            syrk!(Val(:U), Val(:C), one(real(T)), Wval, L₂₁, d₁₁, one(real(T)), M₂₂, diag)
        end
        #
        # compute L D Lᵀ
        #
        #     L₂₁ ← L₂₁ D D₁₁ᵀ  (D = I for Cholesky)
        #
        cmul!(side, diag, L₂₁, d₁₁)
        trmm!(side, uplo, Val(:C), diag, one(T), D₁₁, L₂₁)
    end
    #
    #     D₁₁ ← D₁₁ D D₁₁ᵀ  (D = I for Cholesky, D = diag(d₁₁) for LDLt)
    #
    W₁₁ = reshape(view(Wval, oneto(nn * nn)), nn, nn)
    fill!(W₁₁, zero(T))
    copytri!(W₁₁, D₁₁, uplo)

    if DIAG === :U
        @inbounds for i in diagind(W₁₁)
            W₁₁[i] = one(T)
        end
    end

    cmul!(side, diag, W₁₁, d₁₁)
    trmm!(side, uplo, Val(:C), diag, one(T), D₁₁, W₁₁)

    copytri!(D₁₁, W₁₁, uplo)
    #
    # add contributions from children
    #
    #     D₁₁ ← D₁₁ + F₁₁
    #
    addtri!(D₁₁, F₁₁, uplo)

    if ispositive(na)
        #
        #     L₂₁ ← L₂₁ + F₂₁
        #
        addrec!(L₂₁, F₂₁)
    end

    return ns
end

# Fast path for nn = 1 (residual size is 1)
# In this case, diagonal blocks are scalars and off-diagonal blocks are vectors
function unchol_loop_nod!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        d::AbstractVector{T},
        Fval::AbstractVector{T},
        Wval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {UPLO, DIAG, T, I <: Integer}
    #
    # nn = 1 (the size of the residual at node j)
    #
    nn = one(I)
    #
    # na is the size of the separator at node j
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(rel, j)
    #
    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) | = 1 + na
    #
    nj = nn + na
    #
    # F is the frontal matrix at node j
    #
    #           1   na
    #     F = [ f₁₁     ] 1
    #         [ f₂₁ F₂₂ ] na
    #
    F = reshape(view(Fval, oneto(nj * nj)), nj, nj)

    F₂₂ = view(F, nn + one(I):nj, nn + one(I):nj)

    if UPLO === :L
        f₂₁ = view(F, nn + one(I):nj, one(I))
    else
        f₂₁ = view(F, one(I), nn + one(I):nj)
    end
    #
    # L is part of the Cholesky factor (dj is scalar, D₁₁ is scalar, l₂₁ is vector)
    #
    #          res(j)
    #     L = [ D₁₁  ] res(j)
    #         [ l₂₁  ] sep(j)
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    dj = d[first(neighbors(res, j))]
    d₁₁ = Dval[Dp]
    l₂₁ = view(Lval, Lp:Lp + na - one(I))
    #
    #     F ← 0
    #
    zerotri!(F, uplo)

    for i in Iterators.reverse(neighbors(chd, j))
        #
        # add the update matrix for child i to F
        #
        #     F ← F + Rᵢ Mᵢ Rᵢᵀ
        #
        unchol_recv!(F, Mptr, Mval, rel, ns, i, uplo)
        ns -= one(I)
    end
    #
    # Load f₁₁ after children updates
    #
    f₁₁ = F[one(I)]
    #
    # M₂₂ is the update matrix for node j
    #
    # compute M₂₂ BEFORE modifying l₂₁
    #
    #     M₂₂ ← F₂₂ + l₂₁ D l₂₁ᴴ  (D = 1 for Cholesky, D = dj for LDLt)
    #
    if ispositive(na)
        ns += one(I)
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * na
        M₂₂ = reshape(view(Mval, strt:stop - one(I)), na, na)
        copytri!(M₂₂, F₂₂, uplo)
        #
        # syrk becomes syr! (rank-1 update): M₂₂ ← M₂₂ + dj * l₂₁ l₂₁ᴴ
        #
        syr!(uplo, dj, l₂₁, M₂₂)
        #
        # compute L D Lᵀ
        #
        #     l₂₁ ← l₂₁ * dj * conj(D₁₁)  (for DIAG === :N)
        #     l₂₁ ← l₂₁ * dj              (for DIAG === :U, unit diagonal D₁₁ = 1)
        #
        if DIAG === :N
            rmul!(l₂₁, dj * conj(d₁₁))
        else
            rmul!(l₂₁, dj)
        end
    end
    #
    #     D₁₁ ← D₁₁ D D₁₁ᴴ  (D = 1 for Cholesky, D = dj for LDLt)
    #
    #     For DIAG === :N: w₁₁ = D₁₁ * dj * conj(D₁₁) = dj * |D₁₁|²
    #     For DIAG === :U: w₁₁ = dj (unit diagonal D₁₁ = 1, so 1 * dj * 1 = dj)
    #
    if DIAG === :N
        w₁₁ = d₁₁ * dj * conj(d₁₁)
    else
        w₁₁ = dj
    end
    #
    # add contributions from children
    #
    #     D₁₁ ← w₁₁ + f₁₁
    #
    Dval[Dp] = w₁₁ + f₁₁

    if ispositive(na)
        #
        #     l₂₁ ← l₂₁ + f₂₁
        #
        addrec!(l₂₁, f₂₁)
    end

    return ns
end

function unchol_recv!(
        F::AbstractMatrix{T},
        ptr::AbstractVector{I},
        val::AbstractVector{T},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
        uplo::Val{UPLO},
    ) where {UPLO, T, I <: Integer}
    #
    # na is the size of the separator at node i
    #
    #     na = | sep(i) |
    #
    na = eltypedegree(rel, i)
    #
    # inj is the subset inclusion
    #
    #     inj: sep(i) → bag(parent(i))
    #
    inj = neighbors(rel, i)
    #
    # M is the update matrix from child i
    #
    strt = ptr[ns]
    M = reshape(view(val, strt:strt + na * na - one(I)), na, na)
    #
    # add M to F
    #
    #     F ← F + inj M injᵀ
    #
    addscattertri!(F, M, inj, uplo)
    return
end
