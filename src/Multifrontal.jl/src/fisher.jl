function fisher!(
        Y::AbstractCholesky{UPLO, T},
        F::AbstractCholesky{UPLO, T},
        S::AbstractCholesky{UPLO, T};
        inv::Bool=false,
        check::Bool=true,
    ) where {UPLO, T}
    info = fisher!(triangular(Y), triangular(F), triangular(S); inv, check)
    Y.info[] = info
    return Y
end

function fisher!(
        Y::ChordalTriangular{:N, UPLO, T, I},
        F::ChordalTriangular{:N, UPLO, T, I},
        S::ChordalTriangular{:N, UPLO, T, I};
        inv::Bool=false,
        check::Bool=true,
    ) where {UPLO, T, I <: Integer}
    @assert checksymbolic(Y, F, S)

    # Allocate workspace
    Uptr = FVector{I}(undef, F.S.nMptr)
    Uval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)

    if inv
        info = fisher_impl!(Uptr, Uval, Fval, F, S, Y, Val(true))
    else
        info = fisher_impl!(Uptr, Uval, Fval, F, S, Y, Val(false))
    end

    check && checkinfo(info, F.diag)

    return info
end

#
# inv=false: fisherroot_fwd! → fisher_scale! → fisherroot_bwd!
# inv=true:  fisherroot_bwd! → fisher_scale! → fisherroot_fwd!
#
function fisher_impl!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        L::ChordalTriangular{:N, UPLO, T, I},
        S::ChordalTriangular{:N, UPLO, T, I},
        Y::ChordalTriangular{:N, UPLO, T, I},
        inv::Val{INV},
        args...
    ) where {UPLO, T, I <: Integer, INV}
    if INV
        fisherroot_bwd!(Uptr, Uval, Fval, L, Y, inv, args...)
    else
        fisherroot_fwd!(Uptr, Uval, Fval, L, Y, inv, args...)
    end

    info = fisher_scale!(Uptr, Uval, Fval, L, S, Y, inv, args...)
    ispositive(info) && return info

    if INV
        fisherroot_fwd!(Uptr, Uval, Fval, L, Y, inv, args...)
    else
        fisherroot_bwd!(Uptr, Uval, Fval, L, Y, inv, args...)
    end

    return info
end


# fisher_scale!
# =============


function fisher_scale!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        L::ChordalTriangular{:N, UPLO, T, I},
        S::ChordalTriangular{:N, UPLO, T, I},
        Y::ChordalTriangular{:N, UPLO, T, I},
        inv::Val{INV},
        rng::AbstractRange{I} = vertices(L.S.res),
    ) where {UPLO, T, I <: Integer, INV}

    ns = zero(I); Uptr[one(I)] = one(I)

    for j in reverse(rng)
        ns, info = fisher_scale_loop!(
            Uptr, Uval, Fval,
            L.S.Dptr, L.S.Lptr,
            L.Dval, L.Lval,
            S.Dval, S.Lval,
            Y.Dval, Y.Lval,
            L.S.res, L.S.rel, L.S.chd, ns, j, L.uplo, inv
        )
        ispositive(info) && return info
    end

    return zero(I)
end

function fisher_scale_loop!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLval::AbstractVector{T},
        SDval::AbstractVector{T},
        SLval::AbstractVector{T},
        YDval::AbstractVector{T},
        YLval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        uplo::Val{UPLO},
        inv::Val{INV},
    ) where {UPLO, T, I <: Integer, INV}
    nn = eltypedegree(res, j)

    if isone(nn)
        return fisher_scale_loop_nod!(
            Uptr, Uval, Fval,
            Dptr, Lptr,
            LDval, LLval,
            SDval, SLval,
            YDval, YLval,
            res, rel, chd, ns, j, uplo, inv
        )
    else
        return fisher_scale_loop_snd!(
            Uptr, Uval, Fval,
            Dptr, Lptr,
            LDval, LLval,
            SDval, SLval,
            YDval, YLval,
            res, rel, chd, ns, nn, j, uplo, inv
        )
    end
end

function fisher_scale_loop_snd!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLval::AbstractVector{T},
        SDval::AbstractVector{T},
        SLval::AbstractVector{T},
        YDval::AbstractVector{T},
        YLval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        nn::I,
        j::I,
        uplo::Val{UPLO},
        inv::Val{INV},
    ) where {UPLO, T, I <: Integer, INV}
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
    # Dp and Lp are indices into the diagonal and off-diagonal blocks
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    #
    # S is the selected inverse at node j
    #
    #          res(j)
    #     S = [ S₁₁  ] res(j)
    #         [ S₂₁  ] sep(j)
    #
    S₁₁ = reshape(view(SDval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        S₂₁ = reshape(view(SLval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        S₂₁ = reshape(view(SLval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    # Y is the direction matrix at node j
    #
    #          res(j)
    #     Y = [ Y₁₁  ] res(j)
    #         [ Y₂₁  ] sep(j)
    #
    Y₁₁ = reshape(view(YDval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        Y₂₁ = reshape(view(YLval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        Y₂₁ = reshape(view(YLval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    # L is the Cholesky factor at node j
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    L₁₁ = reshape(view(LDval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(LLval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(LLval, Lp:Lp + nn * na - one(I)), nn, na)
    end

    if UPLO === :L
        sdL = Val(:L)
    else
        sdL = Val(:R)
    end
    #
    # S₂₂ is the update matrix from the parent of node j
    #
    if ispositive(na)
        strt = Uptr[ns]
        S₂₂ = reshape(view(Uval, strt:strt + na * na - one(I)), na, na)
        ns -= one(I)
        #
        #     F₂₂ ← S₂₂
        #
        copytri!(F₂₂, S₂₂, uplo)
        #
        #     Y₂₁ ← S₂₂ Y₂₁  or  Y₂₁ ← S₂₂⁻¹ Y₂₁
        #
        if INV
            #
            #     S₂₂ ← chol(S₂₂)
            #
            info = potrf!(uplo, S₂₂)
            ispositive(info) && return ns, info

            trsm!(sdL, uplo, Val(:N), Val(:N), one(T), S₂₂, Y₂₁)
            trsm!(sdL, uplo, Val(:C), Val(:N), one(T), S₂₂, Y₂₁)
        else
            copyrec!(F₂₁, Y₂₁)
            symm!(sdL, uplo, one(T), S₂₂, F₂₁, zero(T), Y₂₁)
        end
    end
    #
    #     F ← S
    #
    copytri!(F₁₁, S₁₁, uplo)
    copyrec!(F₂₁, S₂₁)

    for i in neighbors(chd, j)
        #
        # send update matrix to child i
        #
        #     Uᵢ ← Rᵢᵀ F Rᵢ
        #
        ns += one(I)
        fisherroot_get_update!(F, Uptr, Uval, rel, ns, i, uplo)
    end

    return ns, zero(I)
end

# Fast path for nn = 1 (residual size is 1)
# In this case, diagonal blocks are scalars and off-diagonal blocks are vectors
function fisher_scale_loop_nod!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLval::AbstractVector{T},
        SDval::AbstractVector{T},
        SLval::AbstractVector{T},
        YDval::AbstractVector{T},
        YLval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        uplo::Val{UPLO},
        inv::Val{INV},
    ) where {UPLO, T, I <: Integer, INV}
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
    # Dp and Lp are indices into the diagonal and off-diagonal blocks
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    #
    # S is the selected inverse at node j (s₁₁ is scalar, s₂₁ is vector)
    #
    s₁₁ = SDval[Dp]
    s₂₁ = view(SLval, Lp:Lp + na - one(I))
    #
    # Y is the direction matrix at node j (y₁₁ is scalar ref, y₂₁ is vector)
    #
    y₂₁ = view(YLval, Lp:Lp + na - one(I))

    if UPLO === :L
        sdL = Val(:L)
    else
        sdL = Val(:R)
    end
    #
    # S₂₂ is the update matrix from the parent of node j
    #
    if ispositive(na)
        strt = Uptr[ns]
        S₂₂ = reshape(view(Uval, strt:strt + na * na - one(I)), na, na)
        ns -= one(I)
        #
        #     F₂₂ ← S₂₂
        #
        copytri!(F₂₂, S₂₂, uplo)
        #
        #     y₂₁ ← S₂₂ y₂₁  or  y₂₁ ← S₂₂⁻¹ y₂₁
        #
        if INV
            #
            #     S₂₂ ← chol(S₂₂)
            #
            info = potrf!(uplo, S₂₂)
            ispositive(info) && return ns, info
            #
            #     y₂₁ ← S₂₂⁻¹ y₂₁ (triangular solves become trsv)
            #     For UPLO = :U, snd does right mult: Y = Y * S⁻¹ * S⁻ᴴ
            #     Column equivalent: y = S⁻ᵀ * S⁻¹ * y (note order change)
            #
            if UPLO === :L
                trsv!(uplo, Val(:N), Val(:N), S₂₂, y₂₁)
                trsv!(uplo, Val(:C), Val(:N), S₂₂, y₂₁)
            else
                trsv!(uplo, Val(:T), Val(:N), S₂₂, y₂₁)
                trsv!(uplo, Val(:N), Val(:N), S₂₂, y₂₁)
            end
        else
            copyrec!(f₂₁, y₂₁)
            #
            #     y₂₁ ← S₂₂ f₂₁ (symm becomes symv)
            #
            symv!(uplo, one(T), S₂₂, f₂₁, zero(T), y₂₁)
        end
    end
    #
    #     F ← S (f₁₁ ← s₁₁ is scalar, f₂₁ ← s₂₁ is vector copy)
    #
    F[one(I)] = s₁₁
    copyrec!(f₂₁, s₂₁)

    for i in neighbors(chd, j)
        #
        # send update matrix to child i
        #
        #     Uᵢ ← Rᵢᵀ F Rᵢ
        #
        ns += one(I)
        fisherroot_get_update!(F, Uptr, Uval, rel, ns, i, uplo)
    end

    return ns, zero(I)
end
