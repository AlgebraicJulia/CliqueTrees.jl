function fisherroot!(
        Y::AbstractCholesky{UPLO, T},
        F::AbstractCholesky{UPLO, T},
        S::AbstractCholesky{UPLO, T};
        adj::Bool=false,
        inv::Bool=false,
        check::Bool=true,
    ) where {UPLO, T}
    info = fisherroot!(triangular(Y), triangular(F), triangular(S); adj, inv, check)
    Y.info[] = info
    return Y
end

function fisherroot!(
        Y::ChordalTriangular{:N, UPLO, T, I},
        F::ChordalTriangular{:N, UPLO, T, I},
        S::ChordalTriangular{:N, UPLO, T, I};
        adj::Bool=false,
        inv::Bool=false,
        check::Bool=true,
    ) where {UPLO, T, I <: Integer}
    @assert checksymbolic(Y, F, S)

    # Allocate workspace
    Uptr = FVector{I}(undef, F.S.nMptr)
    Uval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)

    if adj && inv
        info = fisherroot_impl!(Uptr, Uval, Fval, F, S, Y, Val(true), Val(true))
    elseif adj
        info = fisherroot_impl!(Uptr, Uval, Fval, F, S, Y, Val(true), Val(false))
    elseif inv
        info = fisherroot_impl!(Uptr, Uval, Fval, F, S, Y, Val(false), Val(true))
    else
        info = fisherroot_impl!(Uptr, Uval, Fval, F, S, Y, Val(false), Val(false))
    end

    check && checkinfo(info, F.diag)

    return info
end

#
# adj=false, inv=false: fwd → scale
# adj=true,  inv=false: scale → bwd
# adj=false, inv=true:  scale → fwd
# adj=true,  inv=true:  bwd → scale
#
function fisherroot_impl!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        L::ChordalTriangular{:N, UPLO, T, I},
        S::ChordalTriangular{:N, UPLO, T, I},
        Y::ChordalTriangular{:N, UPLO, T, I},
        adj::Val{ADJ},
        inv::Val{INV},
    ) where {UPLO, T, I <: Integer, ADJ, INV}
    if ADJ == INV
        # adj=false, inv=false or adj=true, inv=true
        if ADJ
            fisherroot_bwd!(Uptr, Uval, Fval, L, Y, inv)
        else
            fisherroot_fwd!(Uptr, Uval, Fval, L, Y, inv)
        end

        return fisherroot_scale!(Uptr, Uval, Fval, L, S, Y, adj, inv)
    else
        # adj=true, inv=false or adj=false, inv=true
        info = fisherroot_scale!(Uptr, Uval, Fval, L, S, Y, adj, inv)
        ispositive(info) && return info

        if ADJ
            fisherroot_bwd!(Uptr, Uval, Fval, L, Y, inv)
        else
            fisherroot_fwd!(Uptr, Uval, Fval, L, Y, inv)
        end

        return info
    end
end

function fisherroot_fwd!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        L::ChordalTriangular{:N, UPLO, T, I},
        Y::ChordalTriangular{:N, UPLO, T, I},
        inv::Val{INV},
        rng::AbstractRange{I} = vertices(L.S.res),
    ) where {UPLO, T, I <: Integer, INV}

    ns = zero(I); Uptr[one(I)] = one(I)

    for j in rng
        ns = fisherroot_fwd_loop!(
            Uptr, Uval, Fval,
            L.S.Dptr, L.S.Lptr,
            L.Dval, L.Lval,
            Y.Dval, Y.Lval,
            L.S.res, L.S.rel, L.S.chd, ns, j, L.uplo, inv
        )
    end

    return
end

function fisherroot_fwd_loop!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLval::AbstractVector{T},
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
        return fisherroot_fwd_loop_nod!(
            Uptr, Uval, Fval,
            Dptr, Lptr,
            LDval, LLval,
            YDval, YLval,
            res, rel, chd, ns, j, uplo, inv
        )
    else
        return fisherroot_fwd_loop_snd!(
            Uptr, Uval, Fval,
            Dptr, Lptr,
            LDval, LLval,
            YDval, YLval,
            res, rel, chd, ns, nn, j, uplo, inv
        )
    end
end

function fisherroot_fwd_loop_snd!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLval::AbstractVector{T},
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
    #
    # U₂₂ is the update matrix for node j
    #
    if UPLO === :L
        sdR, sdL = Val(:R), Val(:L)
        trN, trC = Val(:N), Val(:C)
    else
        sdR, sdL = Val(:L), Val(:R)
        trN, trC = Val(:C), Val(:N)
    end

    if !INV
        α = -one(real(T))
        #
        #     Y ← Y + F
        #
        addtri!(Y₁₁, F₁₁, uplo)
        addrec!(Y₂₁, F₂₁)
        #
        #     Y₁₁ ← L₁₁⁻¹ Y₁₁ L₁₁⁻ᴴ
        #
        symmtri!(Y₁₁, uplo)
        trsm!(sdR, uplo, Val(:C), Val(:N), one(T), L₁₁, Y₁₁)
        trsm!(sdL, uplo, Val(:N), Val(:N), one(T), L₁₁, Y₁₁)
    else
        α = one(real(T))
    end

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
        #     Y₂₁ ← Y₂₁ L₁₁⁻ᴴ
        #
        if !INV
            trsm!(sdR, uplo, Val(:C), Val(:N), one(T), L₁₁, Y₂₁)
        end
        #
        #     U₂₂ ← U₂₂ +   L₂₁ Y₁₁ L₂₁ᴴ + α L₂₁ Y₂₁ᴴ + α Y₂₁ L₂₁ᴴ
        #     Y₂₁ ← Y₂₁ + α L₂₁ Y₁₁
        #
        gemmt!(uplo, trN, trC, α, L₂₁, Y₂₁, one(T), U₂₂)
        symm!(sdR, uplo, α, Y₁₁, L₂₁, one(T), Y₂₁)
        gemmt!(uplo, trN, trC, α, Y₂₁, L₂₁, one(T), U₂₂)
        #
        #     Y₂₁ ← Y₂₁ L₁₁ᴴ
        #
        if INV
            trmm!(sdR, uplo, Val(:C), Val(:N), one(T), L₁₁, Y₂₁)
        end
    end

    if INV
        #
        #     Y₁₁ ← L₁₁ Y₁₁ L₁₁ᴴ
        #
        symmtri!(Y₁₁, uplo)
        trmm!(sdR, uplo, Val(:C), Val(:N), one(T), L₁₁, Y₁₁)
        trmm!(sdL, uplo, Val(:N), Val(:N), one(T), L₁₁, Y₁₁)
        #
        #     Y ← Y + F
        #
        addtri!(Y₁₁, F₁₁, uplo)
        addrec!(Y₂₁, F₂₁)
    end

    return ns
end

# Fast path for nn = 1 (residual size is 1)
# In this case, diagonal blocks are scalars and off-diagonal blocks are vectors
function fisherroot_fwd_loop_nod!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLval::AbstractVector{T},
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
    # Y is the direction matrix at node j (y₁₁ is scalar, y₂₁ is vector)
    #
    y₁₁ = YDval[Dp]
    y₂₁ = view(YLval, Lp:Lp + na - one(I))
    #
    # L is the Cholesky factor at node j (l₁₁ is scalar, l₂₁ is vector)
    #
    l₁₁ = LDval[Dp]
    l₂₁ = view(LLval, Lp:Lp + na - one(I))
    #
    #     F ← 0
    #
    F[one(I)] = zero(T)
    zerorec!(f₂₁)
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
    # Load f₁₁ after children updates
    #
    f₁₁ = F[one(I)]
    #
    #     Y ← Y + F
    #
    #
    # U₂₂ is the update matrix for node j
    #
    if !INV
        α = -one(real(T))
        #
        #     y₁₁ ← y₁₁ + f₁₁
        #
        y₁₁ += f₁₁
        addrec!(y₂₁, f₂₁)
        #
        #     y₁₁ ← l₁₁⁻¹ y₁₁ l₁₁⁻ᴴ = y₁₁ / |l₁₁|²
        #
        y₁₁ /= abs2(l₁₁)
    else
        α = one(real(T))
    end

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
        #     y₂₁ ← y₂₁ / conj(l₁₁)
        #
        if !INV
            rdiv!(y₂₁, conj(l₁₁))
        end
        #
        #     U₂₂ ← U₂₂ + α l₂₁ y₂₁ᴴ
        #
        gert!(uplo, α, l₂₁, y₂₁, U₂₂)
        #
        #     y₂₁ ← y₂₁ + α y₁₁ l₂₁
        #
        axpy!(α * y₁₁, l₂₁, y₂₁)
        #
        #     U₂₂ ← U₂₂ + α y₂₁ l₂₁ᴴ
        #
        gert!(uplo, α, y₂₁, l₂₁, U₂₂)
        #
        #     y₂₁ ← y₂₁ * conj(l₁₁)
        #
        if INV
            rmul!(y₂₁, conj(l₁₁))
        end
    end

    if INV
        #
        #     y₁₁ ← l₁₁ y₁₁ l₁₁ᴴ = y₁₁ * |l₁₁|²
        #
        y₁₁ *= abs2(l₁₁)
        #
        #     y₁₁ ← y₁₁ + f₁₁
        #
        y₁₁ += f₁₁
        addrec!(y₂₁, f₂₁)
    end
    #
    # Write back scalar
    #
    YDval[Dp] = y₁₁

    return ns
end

function fisherroot_bwd!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        L::ChordalTriangular{:N, UPLO, T, I},
        Y::ChordalTriangular{:N, UPLO, T, I},
        inv::Val{INV},
        rng::AbstractRange{I} = vertices(L.S.res),
    ) where {UPLO, T, I <: Integer, INV}

    ns = zero(I); Uptr[one(I)] = one(I)

    for j in reverse(rng)
        ns = fisherroot_bwd_loop!(
            Uptr, Uval, Fval,
            L.S.Dptr, L.S.Lptr,
            L.Dval, L.Lval,
            Y.Dval, Y.Lval,
            L.S.res, L.S.rel, L.S.chd, ns, j, L.uplo, inv
        )
    end

    return
end

function fisherroot_bwd_loop!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLval::AbstractVector{T},
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
        return fisherroot_bwd_loop_nod!(
            Uptr, Uval, Fval,
            Dptr, Lptr,
            LDval, LLval,
            YDval, YLval,
            res, rel, chd, ns, j, uplo, inv
        )
    else
        return fisherroot_bwd_loop_snd!(
            Uptr, Uval, Fval,
            Dptr, Lptr,
            LDval, LLval,
            YDval, YLval,
            res, rel, chd, ns, nn, j, uplo, inv
        )
    end
end

function fisherroot_bwd_loop_snd!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLval::AbstractVector{T},
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
    #
    #     F ← Y
    #
    if UPLO === :L
        sdR, sdL = Val(:R), Val(:L)
        trN, trC = Val(:C), Val(:N)
    else
        sdR, sdL = Val(:L), Val(:R)
        trN, trC = Val(:N), Val(:C)
    end

    if INV
        α = one(real(T))
        #
        #     F ← Y
        #
        copytri!(F₁₁, Y₁₁, uplo)
        copyrec!(F₂₁, Y₂₁)
        #
        #     Y₁₁ ← L₁₁ᴴ Y₁₁ L₁₁
        #
        symmtri!(Y₁₁, uplo)
        trmm!(sdR, uplo, Val(:N), Val(:N), one(T), L₁₁, Y₁₁)
        trmm!(sdL, uplo, Val(:C), Val(:N), one(T), L₁₁, Y₁₁)
    else
        α = -one(real(T))
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
        #     Y₂₁ ← Y₂₁ L₁₁
        #
        if INV
            trmm!(sdR, uplo, Val(:N), Val(:N), one(T), L₁₁, Y₂₁)
        end
        #
        #     Y₁₁ ← Y₁₁ + L₂₁ᴴ S₂₂ L₂₁ + α L₂₁ᴴ Y₂₁ + α Y₂₁ᴴ L₂₁
        #     Y₂₁ ← Y₂₁ +    α S₂₂ L₂₁
        #
        gemmt!(uplo, trN, trC, α, L₂₁, Y₂₁, one(T), Y₁₁)
        symm!(sdL, uplo, α, S₂₂, L₂₁, one(T), Y₂₁)
        gemmt!(uplo, trN, trC, α, Y₂₁, L₂₁, one(T), Y₁₁)
        #
        #     Y₂₁ ← Y₂₁ L₁₁⁻¹
        #
        if !INV
            trsm!(sdR, uplo, Val(:N), Val(:N), one(T), L₁₁, Y₂₁)
        end
    end

    if !INV
        #
        #     Y₁₁ ← L₁₁⁻ᴴ Y₁₁ L₁₁⁻¹
        #
        symmtri!(Y₁₁, uplo)
        trsm!(sdR, uplo, Val(:N), Val(:N), one(T), L₁₁, Y₁₁)
        trsm!(sdL, uplo, Val(:C), Val(:N), one(T), L₁₁, Y₁₁)
        #
        #     F ← Y
        #
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

# Fast path for nn = 1 (residual size is 1)
# In this case, diagonal blocks are scalars and off-diagonal blocks are vectors
function fisherroot_bwd_loop_nod!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{T},
        Fval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLval::AbstractVector{T},
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
    # Y is the direction matrix at node j (y₁₁ is scalar, y₂₁ is vector)
    #
    y₁₁ = YDval[Dp]
    y₂₁ = view(YLval, Lp:Lp + na - one(I))
    #
    # L is the Cholesky factor at node j (l₁₁ is scalar, l₂₁ is vector)
    #
    l₁₁ = LDval[Dp]
    l₂₁ = view(LLval, Lp:Lp + na - one(I))

    if INV
        α = one(real(T))
        #
        #     f₁₁ ← y₁₁, f₂₁ ← y₂₁
        #
        f₁₁ = y₁₁
        copyrec!(f₂₁, y₂₁)
        #
        #     y₁₁ ← l₁₁ᴴ y₁₁ l₁₁ = y₁₁ * |l₁₁|²
        #
        y₁₁ *= abs2(l₁₁)
    else
        α = -one(real(T))
        f₁₁ = zero(T)
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
        #     y₂₁ ← y₂₁ * l₁₁
        #
        if INV
            rmul!(y₂₁, l₁₁)
        end
        #
        #     y₁₁ ← y₁₁ + α l₂₁ᴴ y₂₁ (dot product)
        #
        y₁₁ += α * dot(l₂₁, y₂₁)
        #
        #     y₂₁ ← y₂₁ + α S₂₂ l₂₁ (symv)
        #
        symv!(uplo, α, S₂₂, l₂₁, one(T), y₂₁)
        #
        #     y₁₁ ← y₁₁ + α y₂₁ᴴ l₂₁ (dot product)
        #
        y₁₁ += α * dot(y₂₁, l₂₁)
        #
        #     y₂₁ ← y₂₁ / l₁₁
        #
        if !INV
            rdiv!(y₂₁, l₁₁)
        end
    end

    if !INV
        #
        #     y₁₁ ← l₁₁⁻ᴴ y₁₁ l₁₁⁻¹ = y₁₁ / |l₁₁|²
        #
        y₁₁ /= abs2(l₁₁)
        #
        #     f₁₁ ← y₁₁, f₂₁ ← y₂₁
        #
        f₁₁ = y₁₁
        copyrec!(f₂₁, y₂₁)
    end
    #
    # Write back scalars
    #
    F[one(I)] = f₁₁
    YDval[Dp] = y₁₁

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
        L::ChordalTriangular{:N, UPLO, T, I},
        S::ChordalTriangular{:N, UPLO, T, I},
        Y::ChordalTriangular{:N, UPLO, T, I},
        adj::Val{ADJ},
        inv::Val{INV},
    ) where {UPLO, T, I <: Integer, ADJ, INV}

    ns = zero(I); Uptr[one(I)] = one(I)

    for j in reverse(vertices(L.S.res))
        ns, info = fisherroot_scale_loop!(
            Uptr, Uval, Fval,
            L.S.Dptr, L.S.Lptr,
            L.Dval, L.Lval,
            S.Dval, S.Lval,
            Y.Dval, Y.Lval,
            L.S.res, L.S.rel, L.S.chd, ns, j, L.uplo, adj, inv
        )
        ispositive(info) && return info
    end

    return zero(I)
end

function fisherroot_scale_loop!(
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
        adj::Val{ADJ},
        inv::Val{INV},
    ) where {UPLO, T, I <: Integer, ADJ, INV}
    nn = eltypedegree(res, j)

    if isone(nn)
        return fisherroot_scale_loop_nod!(
            Uptr, Uval, Fval,
            Dptr, Lptr,
            LDval, LLval,
            SDval, SLval,
            YDval, YLval,
            res, rel, chd, ns, j, uplo, adj, inv
        )
    else
        return fisherroot_scale_loop_snd!(
            Uptr, Uval, Fval,
            Dptr, Lptr,
            LDval, LLval,
            SDval, SLval,
            YDval, YLval,
            res, rel, chd, ns, nn, j, uplo, adj, inv
        )
    end
end

function fisherroot_scale_loop_snd!(
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
        adj::Val{ADJ},
        inv::Val{INV},
    ) where {UPLO, T, I <: Integer, ADJ, INV}
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

    if ADJ
        trR = Val(:N)
    else
        trR = Val(:C)
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
        #     S₂₂ ← chol(S₂₂)
        #
        info = potrf!(uplo, S₂₂)
        ispositive(info) && return ns, info
        #
        #     Y₂₁ ← S₂₂⁻ᴴ Y₂₁  or  Y₂₁ ← S₂₂ᴴ Y₂₁
        #
        if INV
            trsm!(sdL, uplo, trR, Val(:N), one(T), S₂₂, Y₂₁)
        else
            trmm!(sdL, uplo, trR, Val(:N), one(T), S₂₂, Y₂₁)
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

# Fast path for nn = 1 (residual size is 1)
# In this case, diagonal blocks are scalars and off-diagonal blocks are vectors
function fisherroot_scale_loop_nod!(
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
        adj::Val{ADJ},
        inv::Val{INV},
    ) where {UPLO, T, I <: Integer, ADJ, INV}
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
    #     F ← S (f₁₁ ← s₁₁ is scalar, f₂₁ ← s₂₁ is vector copy)
    #
    F[one(I)] = s₁₁
    copyrec!(f₂₁, s₂₁)
    #
    # Y is the direction matrix at node j (y₂₁ is vector)
    #
    y₂₁ = view(YLval, Lp:Lp + na - one(I))

    if ADJ
        trR = Val(:N)
    else
        trR = Val(:C)
    end
    #
    # For UPLO === :U, we need to swap transpose for right-multiplication on vectors
    # Right mult by U → left mult by Uᵀ, so :N → :T
    # Right mult by Uᴴ → left mult by U* = Ū, which for real is U, so :C → :N
    #
    if UPLO === :U
        if ADJ
            trV = Val(:T)
        else
            trV = Val(:N)
        end
    else
        trV = trR
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
        #     S₂₂ ← chol(S₂₂)
        #
        info = potrf!(uplo, S₂₂)
        ispositive(info) && return ns, info
        #
        #     y₂₁ ← S₂₂⁻ᴴ y₂₁  or  y₂₁ ← S₂₂ᴴ y₂₁ (trsv/trmv for vectors)
        #
        if INV
            trsv!(uplo, trV, Val(:N), S₂₂, y₂₁)
        else
            trmv!(uplo, trV, Val(:N), S₂₂, y₂₁)
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

