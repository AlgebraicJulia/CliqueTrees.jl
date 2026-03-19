function fisherroot!(
        Y::ChordalCholesky{UPLO, T, I},
        F::ChordalCholesky{UPLO, T, I},
        S::ChordalCholesky{UPLO, T, I};
        adj::Bool=false,
        inv::Bool=false,
        check::Bool=true,
    ) where {UPLO, T, I <: Integer}
    # Allocate workspace
    Uptr = FVector{I}(undef, F.S.nMptr)
    Uval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)
    Wval = FVector{T}(undef, F.S.nFval * F.S.nFval)

    if adj && inv
        info = fisherroot_impl!(Uptr, Uval, Fval, Wval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, S.S.Dptr, S.Dval, S.S.Lptr, S.Lval, Y.S.Dptr, Y.Dval, Y.S.Lptr, Y.Lval, F.S.res, F.S.rel, F.S.chd, Val(UPLO), Val(true), Val(true))
    elseif adj
        info = fisherroot_impl!(Uptr, Uval, Fval, Wval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, S.S.Dptr, S.Dval, S.S.Lptr, S.Lval, Y.S.Dptr, Y.Dval, Y.S.Lptr, Y.Lval, F.S.res, F.S.rel, F.S.chd, Val(UPLO), Val(true), Val(false))
    elseif inv
        info = fisherroot_impl!(Uptr, Uval, Fval, Wval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, S.S.Dptr, S.Dval, S.S.Lptr, S.Lval, Y.S.Dptr, Y.Dval, Y.S.Lptr, Y.Lval, F.S.res, F.S.rel, F.S.chd, Val(UPLO), Val(false), Val(true))
    else
        info = fisherroot_impl!(Uptr, Uval, Fval, Wval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, S.S.Dptr, S.Dval, S.S.Lptr, S.Lval, Y.S.Dptr, Y.Dval, Y.S.Lptr, Y.Lval, F.S.res, F.S.rel, F.S.chd, Val(UPLO), Val(false), Val(false))
    end

    if ispositive(info) && check
        throw(PosDefException(info))
    end

    Y.info[] = info
    return Y
end

function fisherroot_impl!(
        Uptr, Uval, Fval, Wval,
        LDptr, LDval, LLptr, LLval,
        SDptr, SDval, SLptr, SLval,
        YDptr, YDval, YLptr, YLval,
        res, rel, chd,
        uplo, ::Val{false}, ::Val{false},
    )
    fisherroot_fwd!(Uptr, Uval, Fval, Wval, LDptr, LDval, LLptr, LLval, YDptr, YDval, YLptr, YLval, res, rel, chd, uplo, Val(false))
    return fisherroot_scale!(Uptr, Uval, Fval, LDptr, LDval, LLptr, LLval, SDptr, SDval, SLptr, SLval, YDptr, YDval, YLptr, YLval, res, rel, chd, uplo, Val(false), Val(false))
end

function fisherroot_impl!(
        Uptr, Uval, Fval, Wval,
        LDptr, LDval, LLptr, LLval,
        SDptr, SDval, SLptr, SLval,
        YDptr, YDval, YLptr, YLval,
        res, rel, chd,
        uplo, ::Val{true}, ::Val{false},
    )
    info = fisherroot_scale!(Uptr, Uval, Fval, LDptr, LDval, LLptr, LLval, SDptr, SDval, SLptr, SLval, YDptr, YDval, YLptr, YLval, res, rel, chd, uplo, Val(true), Val(false))
    ispositive(info) && return info
    fisherroot_bwd!(Uptr, Uval, Fval, Wval, LDptr, LDval, LLptr, LLval, YDptr, YDval, YLptr, YLval, res, rel, chd, uplo, Val(false))
    return info
end

function fisherroot_impl!(
        Uptr, Uval, Fval, Wval,
        LDptr, LDval, LLptr, LLval,
        SDptr, SDval, SLptr, SLval,
        YDptr, YDval, YLptr, YLval,
        res, rel, chd,
        uplo, ::Val{false}, ::Val{true},
    )
    info = fisherroot_scale!(Uptr, Uval, Fval, LDptr, LDval, LLptr, LLval, SDptr, SDval, SLptr, SLval, YDptr, YDval, YLptr, YLval, res, rel, chd, uplo, Val(false), Val(true))
    ispositive(info) && return info
    fisherroot_fwd!(Uptr, Uval, Fval, Wval, LDptr, LDval, LLptr, LLval, YDptr, YDval, YLptr, YLval, res, rel, chd, uplo, Val(true))
    return info
end

function fisherroot_impl!(
        Uptr, Uval, Fval, Wval,
        LDptr, LDval, LLptr, LLval,
        SDptr, SDval, SLptr, SLval,
        YDptr, YDval, YLptr, YLval,
        res, rel, chd,
        uplo, ::Val{true}, ::Val{true},
    )
    fisherroot_bwd!(Uptr, Uval, Fval, Wval, LDptr, LDval, LLptr, LLval, YDptr, YDval, YLptr, YLval, res, rel, chd, uplo, Val(true))
    return fisherroot_scale!(Uptr, Uval, Fval, LDptr, LDval, LLptr, LLval, SDptr, SDval, SLptr, SLval, YDptr, YDval, YLptr, YLval, res, rel, chd, uplo, Val(true), Val(true))
end

function fisherroot_fwd!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        Wval::AbstractVector{T},
        LDptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLptr::AbstractVector{I},
        LLval::AbstractVector{T},
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

    for j in vertices(res)
        ns = fisherroot_fwd_loop!(
            Uptr, Uval, Fval, Wval,
            LDptr, LDval, LLptr, LLval,
            YDptr, YDval, YLptr, YLval,
            res, rel, chd, ns, j, uplo, inv
        )
    end

    return
end

function fisherroot_fwd_loop!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        Wval::AbstractVector{T},
        LDptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLptr::AbstractVector{I},
        LLval::AbstractVector{T},
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
    #
    #     F ← 0
    #
    zerotri!(F₁₁, uplo)
    zerorec!(F₂₁)
    zerotri!(F₂₂, uplo)

    for i in Iterators.reverse(neighbors(chd, j))
        #
        # add the update matrix for child i to F
        #
        #     F ← F + Rᵢ Uᵢ Rᵢᵀ
        #
        fisherroot_add_update!(F, Uptr, Uval, rel, ns, i, uplo)
        ns -= one(I)
    end
    #
    #     Y ← Y + F
    #
    if !INV
        addtri!(Y₁₁, F₁₁, uplo)
        addrec!(Y₂₁, F₂₁)
    end

    if INV
        α = one(real(T))
    else
        α = -one(real(T))
    end
    #
    # U₂₂ is the update matrix for node j
    #
    if ispositive(na)
        ns += one(I)
        strt = Uptr[ns]
        stop = Uptr[ns + one(I)] = strt + na * na
        U₂₂ = reshape(view(Uval, strt:stop - one(I)), na, na)
        #
        #     U₂₂ ← F₂₂
        #
        copytri!(U₂₂, F₂₂, uplo)
        #
        #     Compute W₂₁ = L₂₁ L₁₁⁻¹
        #
        if UPLO === :L
            W₂₁ = reshape(view(Wval, oneto(na * nn)), na, nn)
        else
            W₂₁ = reshape(view(Wval, oneto(nn * na)), nn, na)
        end

        copyrec!(W₂₁, L₂₁)

        if UPLO === :L
            trsm!(Val(:R), uplo, Val(:N), Val(:N), one(T), L₁₁, W₂₁)
        else
            trsm!(Val(:L), uplo, Val(:N), Val(:N), one(T), L₁₁, W₂₁)
        end
        #
        #     U₂₂ ← U₂₂ + α W₂₁ Y₂₁ᴴ
        #     Y₂₁ ← Y₂₁ + α W₂₁ Y₁₁
        #     U₂₂ ← U₂₂ + α Y₂₁ W₂₁ᴴ
        #
        if UPLO === :L
            gemmt!(uplo, Val(:N), Val(:C), α, W₂₁, Y₂₁, one(T), U₂₂)
            symm!(Val(:R), uplo, α, Y₁₁, W₂₁, one(T), Y₂₁)
            gemmt!(uplo, Val(:N), Val(:C), α, Y₂₁, W₂₁, one(T), U₂₂)
        else
            gemmt!(uplo, Val(:C), Val(:N), α, Y₂₁, W₂₁, one(T), U₂₂)
            symm!(Val(:L), uplo, α, Y₁₁, W₂₁, one(T), Y₂₁)
            gemmt!(uplo, Val(:C), Val(:N), α, W₂₁, Y₂₁, one(T), U₂₂)
        end
    end
    #
    #     Y ← Y + F
    #
    if INV
        addtri!(Y₁₁, F₁₁, uplo)
        addrec!(Y₂₁, F₂₁)
    end

    return ns
end

function fisherroot_bwd!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        Wval::AbstractVector{T},
        LDptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLptr::AbstractVector{I},
        LLval::AbstractVector{T},
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
        ns = fisherroot_bwd_loop!(
            Uptr, Uval, Fval, Wval,
            LDptr, LDval, LLptr, LLval,
            YDptr, YDval, YLptr, YLval,
            res, rel, chd, ns, j, uplo, inv
        )
    end

    return
end

function fisherroot_bwd_loop!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        Wval::AbstractVector{T},
        LDptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLptr::AbstractVector{I},
        LLval::AbstractVector{T},
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
    #
    #     F ← Y
    #
    if INV
        copytri!(F₁₁, Y₁₁, uplo)
        copyrec!(F₂₁, Y₂₁)
    end

    if INV
        α = one(real(T))
    else
        α = -one(real(T))
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
        #
        #     Compute W₂₁ = L₂₁ L₁₁⁻¹
        #
        if UPLO === :L
            W₂₁ = reshape(view(Wval, oneto(na * nn)), na, nn)
        else
            W₂₁ = reshape(view(Wval, oneto(nn * na)), nn, na)
        end

        copyrec!(W₂₁, L₂₁)

        if UPLO === :L
            trsm!(Val(:R), uplo, Val(:N), Val(:N), one(T), L₁₁, W₂₁)
        else
            trsm!(Val(:L), uplo, Val(:N), Val(:N), one(T), L₁₁, W₂₁)
        end
        #
        #     Y₁₁ ← Y₁₁ + α Y₂₁ᴴ W₂₁
        #     Y₂₁ ← Y₂₁ + α V₂₂ W₂₁
        #     Y₁₁ ← Y₁₁ + α W₂₁ᴴ Y₂₁
        #
        if UPLO === :L
            gemmt!(uplo, Val(:C), Val(:N), α, Y₂₁, W₂₁, one(T), Y₁₁)
            symm!(Val(:L), uplo, α, V₂₂, W₂₁, one(T), Y₂₁)
            gemmt!(uplo, Val(:C), Val(:N), α, W₂₁, Y₂₁, one(T), Y₁₁)
        else
            gemmt!(uplo, Val(:N), Val(:C), α, W₂₁, Y₂₁, one(T), Y₁₁)
            symm!(Val(:R), uplo, α, V₂₂, W₂₁, one(T), Y₂₁)
            gemmt!(uplo, Val(:N), Val(:C), α, Y₂₁, W₂₁, one(T), Y₁₁)
        end
    end
    #
    #     F ← Y
    #
    if !INV
        copytri!(F₁₁, Y₁₁, uplo)
        copyrec!(F₂₁, Y₂₁)
    end

    for i in neighbors(chd, j)
        #
        # send update matrix to child i
        #
        #     Uᵢ ← Rᵢᵀ F Rᵢ
        #
        ns += one(I)
        fisherroot_get_update!(F, Uptr, Uval, rel, ns, i, uplo)
    end

    return ns
end

function fisherroot_scale!(
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
        adj::Val{ADJ},
        inv::Val{INV},
    ) where {UPLO, T, I <: Integer, ADJ, INV}

    ns = zero(I); Uptr[one(I)] = one(I)

    for j in reverse(vertices(res))
        ns, info = fisherroot_scale_loop!(
            Uptr, Uval, Fval,
            LDptr, LDval, LLptr, LLval,
            SDptr, SDval, SLptr, SLval,
            YDptr, YDval, YLptr, YLval,
            res, rel, chd, ns, j, uplo, adj, inv
        )
        ispositive(info) && return info
    end

    return zero(I)
end

function fisherroot_scale_loop!(
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
        adj::Val{ADJ},
        inv::Val{INV},
    ) where {UPLO, T, I <: Integer, ADJ, INV}
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
    #     F ← S
    #
    copytri!(F₁₁, S₁₁, uplo)
    copyrec!(F₂₁, S₂₁)
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
    #
    #     Y₁₁ ← Y₁₁ + Y₁₁ᴴ
    #
    symmtri!(Y₁₁, uplo)

    if UPLO === :L
        sdR, sdL = Val(:R), Val(:L)
    else
        sdR, sdL = Val(:L), Val(:R)
    end

    if ADJ
        trR, trL = Val(:N), Val(:C)
    else
        trR, trL = Val(:C), Val(:N)
    end
    #
    #     Y₁₁ ← L₁₁⁻ᴴ Y₁₁ L₁₁⁻¹  or  Y₁₁ ← L₁₁ᴴ Y₁₁ L₁₁
    #
    if !INV
        trsm!(sdR, uplo, trR, Val(:N), one(T), L₁₁, Y₁₁)
        trsm!(sdL, uplo, trL, Val(:N), one(T), L₁₁, Y₁₁)
    else
        trmm!(sdR, uplo, trR, Val(:N), one(T), L₁₁, Y₁₁)
        trmm!(sdL, uplo, trL, Val(:N), one(T), L₁₁, Y₁₁)
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
        #
        #     V₂₂ ← chol(V₂₂)
        #
        info = potrf!(uplo, V₂₂)
        ispositive(info) && return ns, info
        #
        #     Y₂₁ ← V₂₂ L₁₁⁻¹ Y₂₁  or  Y₂₁ ← V₂₂⁻¹ L₁₁ Y₂₁
        #
        if !INV
            trsm!(sdR, uplo, trR, Val(:N), one(T), L₁₁, Y₂₁)
            trmm!(sdL, uplo, trR, Val(:N), one(T), V₂₂, Y₂₁)
        else
            trmm!(sdR, uplo, trR, Val(:N), one(T), L₁₁, Y₂₁)
            trsm!(sdL, uplo, trR, Val(:N), one(T), V₂₂, Y₂₁)
        end
    end

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

function fisherroot_add_update!(
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
    # U is the update matrix from child i
    #
    strt = ptr[ns]
    U = reshape(view(val, strt:strt + na * na - one(I)), na, na)
    #
    # add U to F
    #
    #     F ← F + inj U injᵀ
    #
    addscattertri!(F, U, inj, uplo)
    return
end

function fisherroot_get_update!(
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
    # U is the update matrix for child i
    #
    strt = ptr[ns]
    stop = ptr[ns + one(I)] = strt + na * na
    U = reshape(view(val, strt:stop - one(I)), na, na)
    #
    # copy F into U
    #
    #     U ← injᵀ F inj
    #
    copygathertri!(U, F, inj, uplo)
    return
end
