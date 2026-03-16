"""
    uncholesky!(F::AbstractCholesky)

Compute the product L Lᵀ in place, replacing the Cholesky factor with the
original matrix. This is the inverse operation of [`cholesky!`](@ref).

### Parameters

  - `F`: Cholesky factorization (will be overwritten with L Lᵀ)

"""
function uncholesky!(F::ChordalCholesky{UPLO, T, I}) where {UPLO, T, I <: Integer}
    Mptr = FVector{I}(undef, F.S.nMptr)
    Mval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)
    Wval = FVector{T}(undef, F.S.nFval * F.S.nFval)

    unchol_impl!(Mptr, Mval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, F.d, Fval, Wval, F.S.res, F.S.rel, F.S.chd, F.uplo, F.diag)
    return F
end

"""
    unldlt!(F::AbstractLDLt)

Compute the product L D Lᵀ in place, replacing the LDLt factor with the
original matrix. This is the inverse operation of [`ldlt!`](@ref).

### Parameters

  - `F`: LDLt factorization (will be overwritten with L D Lᵀ)

"""
function unldlt!(F::ChordalLDLt{UPLO, T, I}) where {UPLO, T, I <: Integer}
    Mptr = FVector{I}(undef, F.S.nMptr)
    Mval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)
    Wval = FVector{T}(undef, F.S.nFval * F.S.nFval)

    unchol_impl!(Mptr, Mval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, F.d, Fval, Wval, F.S.res, F.S.rel, F.S.chd, F.uplo, F.diag)
    return F
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

    return
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
