function complete!(
        F::AbstractCholesky{UPLO, T};
        check::Bool=true,
    ) where {UPLO, T}
    info = complete!(triangular(F); check)
    F.info[] = info
    return F
end

function complete!(
        L::ChordalTriangular{:N, UPLO, T, I};
        check::Bool=true,
    ) where {UPLO, T, I <: Integer}
    Mptr = FVector{I}(undef, L.S.nMptr)
    Mval = FVector{T}(undef, L.S.nMval)
    Fval = FVector{T}(undef, L.S.nFval * L.S.nFval)

    info = complete_impl!(Mptr, Mval, Fval, L)

    check && checkinfo(info, L.diag)

    return info
end

#
# Convenience wrapper that unpacks ChordalTriangular types.
#
function complete_impl!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Fval::AbstractVector{T},
        L::ChordalTriangular{:N, UPLO, T, I},
    ) where {UPLO, T, I <: Integer}
    info = complete_impl!(
        Mptr, Mval,
        L.S.Dptr, L.Dval,
        L.S.Lptr, L.Lval,
        Fval,
        L.S.res, L.S.rel, L.S.chd,
        L.uplo)

    return info
end

function complete_impl!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        uplo::Val{UPLO},
    ) where {UPLO, T, I <: Integer}

    ns = zero(I); Mptr[one(I)] = one(I)

    for j in reverse(vertices(res))
        ns, info = complete_loop!(Mptr, Mval, Dptr, Dval, Lptr, Lval, Fval, res, rel, chd, ns, j, uplo)
        ispositive(info) && return info
    end

    return zero(I)
end

function complete_loop!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        uplo::Val{UPLO},
    ) where {UPLO, T, I <: Integer}
    nn = eltypedegree(res, j)

    if isone(nn)
        return complete_loop_nod!(
            Mptr, Mval,
            Dptr, Dval,
            Lptr, Lval,
            Fval,
            res, rel, chd, ns, j, uplo
        )
    else
        return complete_loop_snd!(
            Mptr, Mval,
            Dptr, Dval,
            Lptr, Lval,
            Fval,
            res, rel, chd, ns, nn, j, uplo
        )
    end
end

function complete_loop_snd!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        nn::I,
        j::I,
        uplo::Val{UPLO},
    ) where {UPLO, T, I <: Integer}
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
    # L is part of the Cholesky factor
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
    # copy L into F
    #
    #     F₁₁ ← D₁₁
    #     F₂₁ ← L₂₁
    #
    copytri!(F₁₁, D₁₁, uplo)
    copyrec!(F₂₁, L₂₁)

    if ispositive(na)
        #
        # M₂₂ is the update matrix from the parent of node j
        #
        strt = Mptr[ns]
        M₂₂ = reshape(view(Mval, strt:strt + na * na - one(I)), na, na)
        ns -= one(I)
        #
        # copy M into F
        #
        #     F₂₂ ← M₂₂
        #
        copytri!(F₂₂, M₂₂, uplo)
        #
        # factorize M₂₂
        #
        #     M₂₂ ← chol(M₂₂)
        #
        info = potrf!(uplo, M₂₂)
        ispositive(info) && return ns, info
        #
        # solve
        #
        #     L₂₁ ← M₂₂⁻¹ L₂₁
        #
        if UPLO === :L
            trsm!(Val(:L), Val(:L), Val(:N), Val(:N), one(T), M₂₂, L₂₁)
        else
            trsm!(Val(:R), Val(:U), Val(:N), Val(:N), one(T), M₂₂, L₂₁)
        end
        #
        #     D₁₁ ← D₁₁ - L₂₁ᴴ L₂₁
        #
        if UPLO === :L
            syrk!(Val(:L), Val(:C), -one(real(T)), L₂₁, one(real(T)), D₁₁)
        else
            syrk!(Val(:U), Val(:N), -one(real(T)), L₂₁, one(real(T)), D₁₁)
        end
        #
        # solve
        #
        #     L₂₁ ← -M₂₂⁻ᴴ L₂₁
        #
        if UPLO === :L
            trsm!(Val(:L), Val(:L), Val(:C), Val(:N), -one(T), M₂₂, L₂₁)
        else
            trsm!(Val(:R), Val(:U), Val(:C), Val(:N), -one(T), M₂₂, L₂₁)
        end
    end
    #
    #     D₁₁ ← chol(D₁₁⁻¹)
    #
    if UPLO === :L
        loup = Val(:U)
    else
        loup = Val(:L)
    end

    revtri!(D₁₁, uplo)
    info = potrf!(loup, D₁₁)
    ispositive(info) && return ns, info
    revtri!(D₁₁, loup)

    if ispositive(na)
        #
        #     L₂₁ ← L₂₁ D₁₁⁻¹
        #
        if UPLO === :L
            trsm!(Val(:R), Val(:L), Val(:N), Val(:N), one(T), D₁₁, L₂₁)
        else
            trsm!(Val(:L), Val(:U), Val(:N), Val(:N), one(T), D₁₁, L₂₁)
        end
    end

    trtri!(uplo, Val(:N), D₁₁)

    for i in neighbors(chd, j)
        #
        # send update matrix to child i
        #
        #     Mᵢ ← Rᵢᵀ F Rᵢ
        #
        ns += one(I)
        complete_send!(F, Mptr, Mval, rel, ns, i, uplo)
    end

    return ns, zero(I)
end

# Fast path for nn = 1 (residual size is 1)
# In this case, diagonal blocks are scalars and off-diagonal blocks are vectors
function complete_loop_nod!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        uplo::Val{UPLO},
    ) where {UPLO, T, I <: Integer}
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
    # L is part of the Cholesky factor (d₁₁ is scalar, l₂₁ is vector)
    #
    #          res(j)
    #     L = [ d₁₁  ] res(j)
    #         [ l₂₁  ] sep(j)
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    d₁₁ = Dval[Dp]
    l₂₁ = view(Lval, Lp:Lp + na - one(I))
    #
    # copy L into F
    #
    #     f₁₁ ← d₁₁
    #     f₂₁ ← l₂₁
    #
    F[one(I)] = d₁₁
    copyrec!(f₂₁, l₂₁)

    if ispositive(na)
        #
        # M₂₂ is the update matrix from the parent of node j
        #
        strt = Mptr[ns]
        M₂₂ = reshape(view(Mval, strt:strt + na * na - one(I)), na, na)
        ns -= one(I)
        #
        # copy M into F
        #
        #     F₂₂ ← M₂₂
        #
        copytri!(F₂₂, M₂₂, uplo)
        #
        # factorize M₂₂
        #
        #     M₂₂ ← chol(M₂₂)
        #
        info = potrf!(uplo, M₂₂)
        ispositive(info) && return ns, info
        #
        # solve
        #
        #     l₂₁ ← M₂₂⁻¹ l₂₁ (trsv for vectors)
        #
        if UPLO === :L
            trsv!(Val(:L), Val(:N), Val(:N), M₂₂, l₂₁)
        else
            trsv!(Val(:U), Val(:T), Val(:N), M₂₂, l₂₁)
        end
        #
        #     d₁₁ ← d₁₁ - l₂₁ᴴ l₂₁ (dot product)
        #
        d₁₁ -= dot(l₂₁, l₂₁)
        #
        # solve
        #
        #     l₂₁ ← -M₂₂⁻ᴴ l₂₁ (trsv for vectors, then negate)
        #
        if UPLO === :L
            trsv!(Val(:L), Val(:C), Val(:N), M₂₂, l₂₁)
        else
            trsv!(Val(:U), Val(:N), Val(:N), M₂₂, l₂₁)
        end

        rmul!(l₂₁, -one(T))
    end
    #
    #     d₁₁ ← sqrt(d₁₁) (scalar Cholesky, check positive definite)
    #
    if !ispositive(real(d₁₁))
        return ns, one(I)
    end

    d₁₁ = sqrt(d₁₁)

    if ispositive(na)
        #
        #     l₂₁ ← l₂₁ / d₁₁
        #
        rdiv!(l₂₁, d₁₁)
    end
    #
    #     d₁₁ ← 1 / d₁₁
    #
    d₁₁ = inv(d₁₁)
    #
    # Write back scalar
    #
    Dval[Dp] = d₁₁

    for i in neighbors(chd, j)
        #
        # send update matrix to child i
        #
        #     Mᵢ ← Rᵢᵀ F Rᵢ
        #
        ns += one(I)
        complete_send!(F, Mptr, Mval, rel, ns, i, uplo)
    end

    return ns, zero(I)
end

function complete_send!(
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
    # M is the update matrix for child i
    #
    strt = ptr[ns]
    stop = ptr[ns + one(I)] = strt + na * na
    M = reshape(view(val, strt:stop - one(I)), na, na)
    #
    # copy F into M
    #
    #     M ← Rᵢᵀ F Rᵢ
    #
    copygathertri!(M, F, inj, uplo)
    return
end
