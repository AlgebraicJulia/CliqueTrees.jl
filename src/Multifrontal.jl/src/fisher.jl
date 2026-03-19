function fisher!(
        Y::ChordalCholesky{UPLO, T, I},
        F::ChordalCholesky{UPLO, T, I},
        S::ChordalCholesky{UPLO, T, I};
        inv::Bool=false,
        check::Bool=true,
    ) where {UPLO, T, I <: Integer}
    # Allocate workspace
    Uptr = FVector{I}(undef, F.S.nMptr)
    Uval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)
    Wval = FVector{T}(undef, F.S.nFval * F.S.nFval)

    if inv
        info = fisher_impl!(Uptr, Uval, Fval, Wval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, S.S.Dptr, S.Dval, S.S.Lptr, S.Lval, Y.S.Dptr, Y.Dval, Y.S.Lptr, Y.Lval, F.S.res, F.S.rel, F.S.chd, Val(UPLO), Val(true))
    else
        info = fisher_impl!(Uptr, Uval, Fval, Wval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, S.S.Dptr, S.Dval, S.S.Lptr, S.Lval, Y.S.Dptr, Y.Dval, Y.S.Lptr, Y.Lval, F.S.res, F.S.rel, F.S.chd, Val(UPLO), Val(false))
    end

    if ispositive(info) && check
        throw(PosDefException(info))
    end

    Y.info[] = info
    return Y
end

#
# inv=false: fisherhessian
#
#     fisherroot_fwd! → fisher_scale!(inv=false) → fisherroot_bwd!
#
function fisher_impl!(
        Uptr, Uval, Fval, Wval,
        LDptr, LDval, LLptr, LLval,
        SDptr, SDval, SLptr, SLval,
        YDptr, YDval, YLptr, YLval,
        res, rel, chd,
        uplo, ::Val{false},
    )
    fisherroot_fwd!(Uptr, Uval, Fval, Wval, LDptr, LDval, LLptr, LLval, YDptr, YDval, YLptr, YLval, res, rel, chd, uplo, Val(false))
    info = fisher_scale!(Uptr, Uval, Fval, LDptr, LDval, LLptr, LLval, SDptr, SDval, SLptr, SLval, YDptr, YDval, YLptr, YLval, res, rel, chd, uplo, Val(false))
    ispositive(info) && return info
    fisherroot_bwd!(Uptr, Uval, Fval, Wval, LDptr, LDval, LLptr, LLval, YDptr, YDval, YLptr, YLval, res, rel, chd, uplo, Val(false))
    return info
end

#
# inv=true: fisherhessinv
#
#     fisherroot_bwd!(inv=true) → fisher_scale!(inv=true) → fisherroot_fwd!(inv=true)
#
function fisher_impl!(
        Uptr, Uval, Fval, Wval,
        LDptr, LDval, LLptr, LLval,
        SDptr, SDval, SLptr, SLval,
        YDptr, YDval, YLptr, YLval,
        res, rel, chd,
        uplo, ::Val{true},
    )
    fisherroot_bwd!(Uptr, Uval, Fval, Wval, LDptr, LDval, LLptr, LLval, YDptr, YDval, YLptr, YLval, res, rel, chd, uplo, Val(true))
    info = fisher_scale!(Uptr, Uval, Fval, LDptr, LDval, LLptr, LLval, SDptr, SDval, SLptr, SLval, YDptr, YDval, YLptr, YLval, res, rel, chd, uplo, Val(true))
    ispositive(info) && return info
    fisherroot_fwd!(Uptr, Uval, Fval, Wval, LDptr, LDval, LLptr, LLval, YDptr, YDval, YLptr, YLval, res, rel, chd, uplo, Val(true))
    return info
end

function fisher_scale!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        LDptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLptr::AbstractVector{I},
        LLval::AbstractVector{T},
        SDptr::AbstractVector{I},
        SDval::AbstractVector{T},
        SLptr::AbstractVector{I},
        SLval::AbstractVector{T},
        YDptr::AbstractVector{I},
        YDval::AbstractVector{T},
        YLptr::AbstractVector{I},
        YLval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        uplo::Val{UPLO},
        inv::Val{INV},
    ) where {UPLO, T, I <: Integer, INV}

    ns = zero(I); Uptr[one(I)] = one(I)

    for j in reverse(vertices(res))
        ns, info = fisher_scale_loop!(
            Uptr, Uval, Fval,
            LDptr, LDval, LLptr, LLval,
            SDptr, SDval, SLptr, SLval,
            YDptr, YDval, YLptr, YLval,
            res, rel, chd, ns, j, uplo, inv
        )
        ispositive(info) && return info
    end

    return zero(I)
end

function fisher_scale_loop!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        LDptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLptr::AbstractVector{I},
        LLval::AbstractVector{T},
        SDptr::AbstractVector{I},
        SDval::AbstractVector{T},
        SLptr::AbstractVector{I},
        SLval::AbstractVector{T},
        YDptr::AbstractVector{I},
        YDval::AbstractVector{T},
        YLptr::AbstractVector{I},
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
    # S is the selected inverse at node j
    #
    #          res(j)
    #     S = [ S₁₁  ] res(j)
    #         [ S₂₁  ] sep(j)
    #
    SDp = SDptr[j]
    SLp = SLptr[j]
    S₁₁ = reshape(view(SDval, SDp:SDp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        S₂₁ = reshape(view(SLval, SLp:SLp + nn * na - one(I)), na, nn)
    else
        S₂₁ = reshape(view(SLval, SLp:SLp + nn * na - one(I)), nn, na)
    end
    #
    # Y is the direction matrix at node j
    #
    #          res(j)
    #     Y = [ Y₁₁  ] res(j)
    #         [ Y₂₁  ] sep(j)
    #
    YDp = YDptr[j]
    YLp = YLptr[j]
    Y₁₁ = reshape(view(YDval, YDp:YDp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        Y₂₁ = reshape(view(YLval, YLp:YLp + nn * na - one(I)), na, nn)
    else
        Y₂₁ = reshape(view(YLval, YLp:YLp + nn * na - one(I)), nn, na)
    end
    #
    # L is the Cholesky factor at node j
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    LDp = LDptr[j]
    LLp = LLptr[j]
    L₁₁ = reshape(view(LDval, LDp:LDp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(LLval, LLp:LLp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(LLval, LLp:LLp + nn * na - one(I)), nn, na)
    end

    if UPLO === :L
        sdR, sdL = Val(:R), Val(:L)
    else
        sdR, sdL = Val(:L), Val(:R)
    end
    #
    #     Y₁₁ ← Y₁₁ + Y₁₁ᴴ
    #
    symmtri!(Y₁₁, uplo)

    if INV
        #
        # hessinv: Y₁₁ ← D₁₁ (sym Y₁₁) D₁₁
        #
        #     Y₁₁ ← Y₁₁ L₁₁
        #     Y₁₁ ← L₁₁ᴴ Y₁₁
        #     Y₁₁ ← Y₁₁ L₁₁ᴴ
        #     Y₁₁ ← L₁₁ Y₁₁
        #
        trmm!(sdR, uplo, Val(:N), Val(:N), one(T), L₁₁, Y₁₁)
        trmm!(sdL, uplo, Val(:C), Val(:N), one(T), L₁₁, Y₁₁)
        trmm!(sdR, uplo, Val(:C), Val(:N), one(T), L₁₁, Y₁₁)
        trmm!(sdL, uplo, Val(:N), Val(:N), one(T), L₁₁, Y₁₁)
    else
        #
        # hessian: Y₁₁ ← D₁₁⁻¹ (sym Y₁₁) D₁₁⁻¹
        #
        #     Y₁₁ ← Y₁₁ L₁₁⁻ᴴ
        #     Y₁₁ ← L₁₁⁻¹ Y₁₁
        #     Y₁₁ ← Y₁₁ L₁₁⁻¹
        #     Y₁₁ ← L₁₁⁻ᴴ Y₁₁
        #
        trsm!(sdR, uplo, Val(:C), Val(:N), one(T), L₁₁, Y₁₁)
        trsm!(sdL, uplo, Val(:N), Val(:N), one(T), L₁₁, Y₁₁)
        trsm!(sdR, uplo, Val(:N), Val(:N), one(T), L₁₁, Y₁₁)
        trsm!(sdL, uplo, Val(:C), Val(:N), one(T), L₁₁, Y₁₁)
    end
    #
    # V₂₂ is the update matrix from the parent of node j
    #
    if ispositive(na)
        strt = Uptr[ns]
        V₂₂ = reshape(view(Uval, strt:strt + na * na - one(I)), na, na)
        ns -= one(I)
        #
        #     F₂₂ ← V₂₂
        #
        copytri!(F₂₂, V₂₂, uplo)

        if INV
            #
            #     V₂₂ ← chol(V₂₂)
            #
            info = potrf!(uplo, V₂₂)
            ispositive(info) && return ns, info
            #
            # hessinv: Y₂₁ ← V₂₂⁻¹ Y₂₁ D₁₁
            #
            #     Y₂₁ ← Y₂₁ L₁₁
            #     Y₂₁ ← R₂₂⁻¹ Y₂₁
            #     Y₂₁ ← Y₂₁ L₁₁ᴴ
            #     Y₂₁ ← R₂₂⁻ᴴ Y₂₁
            #
            trmm!(sdR, uplo, Val(:N), Val(:N), one(T), L₁₁, Y₂₁)
            trsm!(sdL, uplo, Val(:N), Val(:N), one(T), V₂₂, Y₂₁)
            trmm!(sdR, uplo, Val(:C), Val(:N), one(T), L₁₁, Y₂₁)
            trsm!(sdL, uplo, Val(:C), Val(:N), one(T), V₂₂, Y₂₁)
        else
            #
            # hessian: Y₂₁ ← V₂₂ Y₂₁ D₁₁⁻¹
            #
            #     Y₂₁ ← V₂₂ Y₂₁
            #     Y₂₁ ← Y₂₁ L₁₁⁻ᴴ
            #     Y₂₁ ← Y₂₁ L₁₁⁻¹
            #
            copyrec!(F₂₁, Y₂₁)

            if UPLO === :L
                symm!(Val(:L), uplo, one(T), V₂₂, F₂₁, zero(T), Y₂₁)
            else
                symm!(Val(:R), uplo, one(T), V₂₂, F₂₁, zero(T), Y₂₁)
            end

            trsm!(sdR, uplo, Val(:C), Val(:N), one(T), L₁₁, Y₂₁)
            trsm!(sdR, uplo, Val(:N), Val(:N), one(T), L₁₁, Y₂₁)
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
