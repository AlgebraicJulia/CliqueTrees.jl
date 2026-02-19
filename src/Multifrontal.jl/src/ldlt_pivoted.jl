# ===== Pivoted LDLt Factorization =====

function LinearAlgebra.ldlt!(F::ChordalLDLt{UPLO, T, I}, ::RowMaximum; check::Bool=true, tol::Real=-one(real(T))) where {UPLO, T, I <: Integer}
    R = real(T)

    Mptr = FVector{I}(undef, F.S.nMptr)
    Mval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)
    piv  = FVector{BlasInt}(undef, F.S.nFval)
    work = FVector{R}(undef, twice(F.S.nFval))
    invp = FVector{I}(undef, nov(F.S.res))
    mval = FVector{I}(undef, F.S.nNval)
    fval = FVector{I}(undef, F.S.nFval)

    info = ldltpiv_fwd!(
        Mptr, Mval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, F.d, Fval,
        F.S.res, F.S.rel, F.S.chd, piv, work, invp, convert(R, tol), Val{UPLO}()
    )

    if isnegative(info)
        throw(ArgumentError(info))
    else
        F.info[] = zero(I)
    end

    ldltpiv_bwd!(Mptr, mval, fval, F.S.Dptr, F.S.Lptr, F.Lval, F.S.res, F.S.rel, F.S.sep, F.S.chd, invp, Fval, Val{UPLO}())
    cholpiv_rel!(F.S.res, F.S.sep, F.S.rel, F.S.chd)
    invpermute!(F.perm, invp)
    return F
end

# ============================= ldltpiv_fwd! =============================

function ldltpiv_fwd!(
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
        piv::AbstractVector{BlasInt},
        work::AbstractVector{<:Real},
        invp::AbstractVector{I},
        tol::Real,
        uplo::Val{UPLO},
    ) where {T, I <: Integer, UPLO}

    ns = zero(I); Mptr[one(I)] = one(I)

    for j in vertices(res)
        ns, info = ldltpiv_fwd_loop!(
            Mptr, Mval, Dptr, Dval, Lptr, Lval, d, Fval,
            res, rel, chd, ns, j, piv, work, invp, tol, uplo
        )

        if isnegative(info)
            return info
        end
    end

    return zero(I)
end

function ldltpiv_fwd_loop!(
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
        piv::AbstractVector{BlasInt},
        work::AbstractVector{<:Real},
        invp::AbstractVector{I},
        tol::Real,
        uplo::Val{UPLO},
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
    # pivoted factorization of D₁₁
    #
    #     invp₁₁' D₁₁ invp₁₁ = L₁₁ d₁₁ L₁₁'
    #
    W₁₁ = reshape(view(Fval, oneto(nn * nn)), nn, nn)
    info, rank = qstrf!(uplo, W₁₁, D₁₁, d₁₁, piv, tol)
    #
    # zero out the rank-deficient part of D₁₁ and L₂₁
    #
    zerotri!(D₁₁, uplo, rank + one(I):nn)

    if UPLO === :L
        zerorec!(L₂₁, oneto(na), rank + one(I):nn)
    else
        zerorec!(L₂₁, rank + one(I):nn, oneto(na))
    end
    #
    # update invp with local pivot permutation
    # invp maps P-indices to Q-indices
    # piv[k] = p means vertex (offset + p) gets Q-index (offset + k)
    #
    offset = first(neighbors(res, j)) - one(I)

    @inbounds for v in oneto(nn)
        invp[offset + piv[v]] = offset + v
    end

    if ispositive(na) && ispositive(rank)
        ns += one(I)
        #
        # S₂₂ is the update matrix for node j
        #
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * na
        S₂₂ = reshape(view(Mval, strt:stop - one(I)), na, na)
        #
        #     S₂₂ ← F₂₂
        #
        copytri!(S₂₂, F₂₂, uplo)
        #
        # permute L₂₁ by pivot
        #
        copyto!(F₂₁, L₂₁)

        if UPLO === :L
            copyrec!(L₂₁, F₂₁, oneto(na), view(piv, oneto(nn)))
        else
            copyrec!(L₂₁, F₂₁, view(piv, oneto(nn)), oneto(na))
        end
        #
        # Use only the first `rank` columns/rows for the solve
        #
        rD₁₁ = view(D₁₁, oneto(rank), oneto(rank))
        rd₁₁ = view(d₁₁, oneto(rank))

        if UPLO === :L
            rL₂₁ = view(L₂₁, oneto(na), oneto(rank))
        else
            rL₂₁ = view(L₂₁, oneto(rank), oneto(na))
        end
        #
        #     rL₂₁ ← rL₂₁ rD₁₁⁻ᴴ
        #
        if UPLO === :L
            trsm!(Val(:R), Val(:L), Val(:C), Val(:U), one(T), rD₁₁, rL₂₁)
        else
            trsm!(Val(:L), Val(:U), Val(:C), Val(:U), one(T), rD₁₁, rL₂₁)
        end
        #
        # W₂₁ ← rL₂₁ (save before scaling by d⁻¹)
        #
        W₂₁ = view(F, oneto(size(rL₂₁, 1)), oneto(size(rL₂₁, 2)))
        copyrec!(W₂₁, rL₂₁)
        #
        #     rL₂₁ ← rL₂₁ rd₁₁⁻¹
        #
        if UPLO === :L
            @inbounds for k in axes(rL₂₁, 2)
                idk = inv(rd₁₁[k])
                for i in axes(rL₂₁, 1)
                    rL₂₁[i, k] *= idk
                end
            end
        else
            @inbounds for i in axes(rL₂₁, 2)
                for k in axes(rL₂₁, 1)
                    rL₂₁[k, i] *= inv(rd₁₁[k])
                end
            end
        end
        #
        #     S₂₂ ← S₂₂ - W₂₁ rL₂₁ᴴ
        #
        if UPLO === :L
            trrk!(uplo, Val(:N), W₂₁, rL₂₁, S₂₂)
        else
            trrk!(uplo, Val(:C), W₂₁, rL₂₁, S₂₂)
        end
    elseif ispositive(na) && iszero(rank)
        # Rank-deficient: just copy F₂₂ to update matrix
        ns += one(I)
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * na
        S₂₂ = reshape(view(Mval, strt:stop - one(I)), na, na)
        copytri!(S₂₂, F₂₂, uplo)
    end

    return ns, info
end

# ============================= ldltpiv_bwd! =============================

function ldltpiv_bwd!(
        mptr::AbstractVector{I},
        mval::AbstractVector{I},
        fval::AbstractVector{I},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        sep::AbstractGraph{I},
        chd::AbstractGraph{I},
        invp::AbstractVector{I},
        Fval::AbstractVector{T},
        uplo::Val{UPLO},
    ) where {T, I <: Integer, UPLO}

    ns = zero(I); mptr[one(I)] = one(I)

    for j in reverse(vertices(res))
        ns = ldltpiv_bwd_loop!(
            mptr, mval, fval, Dptr, Lptr, Lval,
            res, rel, sep, chd, ns, j, invp, Fval, uplo
        )
    end

    return
end

function ldltpiv_bwd_loop!(
        mptr::AbstractVector{I},
        mval::AbstractVector{I},
        fval::AbstractVector{I},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        sep::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        invp::AbstractVector{I},
        Fval::AbstractVector{T},
        uplo::Val{UPLO},
    ) where {T, I <: Integer, UPLO}
    #
    # nn is the size of the residual at node j
    #
    #     nn = |res(j)|
    #
    nn = eltypedegree(res, j)
    #
    # na is the size of the separator at node j
    #
    #     na = |sep(j)|
    #
    na = eltypedegree(rel, j)
    #
    # nj is the size of the bag at node j
    #
    #     nj = |bag(j)|
    #
    nj = nn + na
    #
    # f is the frontal Q-index vector at node j
    #
    #   f = [ f₁ ] nn
    #       [ f₂ ] na
    #
    f  = view(fval, oneto(nj))
    f₁ = view(f,    oneto(nn))
    f₂ = view(f,    nn + one(I):nj)
    #
    #     f₁ ← invp[res(j)]
    #
    copyrec!(f₁, view(invp, neighbors(res, j)))
    #
    # pull Q-indices from ancestor's update matrix
    #
    #     f₂ ← m₂
    #
    if ispositive(na)
        strt = mptr[ns]
        m₂ = view(mval, strt:strt + na - one(I))
        ns -= one(I)
        copyrec!(f₂, m₂)
    end

    for i in neighbors(chd, j)
        #
        # push f restricted to sep(i) to child i
        #
        #     mᵢ ← Rᵢᵀ f
        #
        ns += one(I)
        cholpiv_bwd_update!(f, mptr, mval, rel, ns, i)
    end
    #
    # s₂ is the sorting permutation for sep(j) by Q-index
    #
    s₂ = neighbors(sep, j)
    sortperm!(s₂, f₂; alg=QuickSort, initialized=false)
    #
    # L₂₁ is the off-diagonal block of the LDLt factor
    #
    #        nn
    #   L = [ L₂₁ ] na
    #
    Lp = Lptr[j]

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
        F₂₁ = reshape(view(Fval, oneto(na * nn)),           na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
        F₂₁ = reshape(view(Fval, oneto(nn * na)),           nn, na)
    end
    #
    # permute rows/cols of L₂₁ by s₂
    #
    copyrec!(F₂₁, L₂₁)

    if UPLO === :L
        @inbounds for k in oneto(nn)
            for i in oneto(na)
                L₂₁[i, k] = F₂₁[s₂[i], k]
            end
        end
    else
        @inbounds for i in oneto(na)
            for k in oneto(nn)
                L₂₁[k, i] = F₂₁[k, s₂[i]]
            end
        end
    end
    #
    # update sep(j) targets with sorted Q-indices
    #
    #     sep(j) ← f₂[s₂]
    #
    @inbounds for i in oneto(na)
        s₂[i] = f₂[s₂[i]]
    end

    return ns
end
