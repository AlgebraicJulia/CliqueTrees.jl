# ===== Rank-Revealing Pivoted LDLt Factorization =====

function LinearAlgebra.ldlt!(F::ChordalLDLt{UPLO, T}, ::RowMaximum; signs::MaybeVector=nothing, reg::MaybeRegularization=nothing, check::Bool=true, tol::Real=-one(real(T))) where {UPLO, T}
    @assert !isnothing(signs) || isnothing(reg)
    return _ldlt!(F, RowMaximum(), signs, reg, check, tol)
end 

function _ldlt!(F::ChordalLDLt{UPLO, T, I}, ::RowMaximum, signs::MaybeVector, ::Nothing, check::Bool, tol::Real) where {UPLO, T, I <: Integer}
    Mptr = FVector{I}(undef, F.S.nMptr)
    Mval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)
    piv  = FVector{I}(undef, F.S.nFval)
    invp = FVector{I}(undef, nov(F.S.res))
    mval = FVector{I}(undef, F.S.nNval)
    fval = FVector{I}(undef, F.S.nFval)

    if isnothing(signs)
        S = nothing
    else
        S = FVector{I}(undef, length(signs))

        @inbounds for i in eachindex(F.perm)
            S[i] = signs[F.perm[i]]
        end
    end

    info = ldlt_piv_fwd!(
        Mptr, Mval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, F.d, Fval,
        F.S.res, F.S.rel, F.S.chd, piv, invp, S, convert(real(T), tol), Val{UPLO}()
    )

    if isnegative(info)
        throw(ArgumentError(info))
    else
        F.info[] = zero(I)
    end

    ldlt_piv_bwd!(Mptr, mval, fval, F.S.Dptr, F.S.Lptr, F.Lval, F.S.res, F.S.rel, F.S.sep, F.S.chd, invp, Fval, Val{UPLO}())
    cholpiv_rel!(F.S.res, F.S.sep, F.S.rel, F.S.chd)
    invpermute!(F.perm, invp)
    return F
end

# ============================= ldlt_piv_fwd! =============================

function ldlt_piv_fwd!(
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
        piv::AbstractVector{I},
        invp::AbstractVector{I},
        S::MaybeVector,
        tol::Real,
        uplo::Val{UPLO},
    ) where {T, I <: Integer, UPLO}

    ns = zero(I); Mptr[one(I)] = one(I)

    for j in vertices(res)
        ns, info = ldlt_piv_fwd_loop!(
            Mptr, Mval, Dptr, Dval, Lptr, Lval, d, Fval,
            res, rel, chd, ns, j, piv, invp, S, tol, uplo
        )

        if isnegative(info)
            return info
        end
    end

    return zero(I)
end

function ldlt_piv_fwd_loop!(
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
        piv::AbstractVector{I},
        invp::AbstractVector{I},
        S::MaybeVector,
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
        M₂₂ = reshape(view(Mval, oneto(zero(I))), zero(I), zero(I))
    end

    if isnothing(S)
        S₁₁ = nothing
    else
        S₁₁ = view(S, neighbors(res, j))
    end
    #
    # pivoted factorization
    #
    info, rank = ldlt_piv_kernel!(D₁₁, L₂₁, M₂₂, Fval, d₁₁, piv, S₁₁, tol, uplo)
    #
    # update invp with local pivot permutation
    # invp maps P-indices to Q-indices
    # piv[k] = p means vertex (offset + p) gets Q-index (offset + k)
    #
    offset = first(neighbors(res, j)) - one(I)

    @inbounds for v in oneto(nn)
        invp[offset + piv[v]] = offset + v
    end

    if ispositive(na) && iszero(rank)
        ns -= one(I)
    end

    return ns, info
end

# ============================= ldlt_piv_kernel! =============================

function ldlt_piv_kernel!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        M::AbstractMatrix{T},
        Wval::AbstractVector{T},
        d::AbstractVector{T},
        P::AbstractVector{<:Integer},
        S::MaybeVector,
        tol::Real,
        uplo::Val{UPLO},
    ) where {T, UPLO}
    @assert size(D, 1) == size(D, 1) == length(d)
    @assert size(M, 1) == size(M, 2)
    @assert length(L) <= length(Wval)

    if UPLO === :L
        @assert size(L, 1) == size(M, 1)
        @assert size(L, 2) == size(D, 1)
    else
        @assert size(L, 1) == size(D, 1)
        @assert size(L, 2) == size(M, 1)
    end

    info, rank = ldlt_piv_factor!(D, L, Wval, d, P, S, tol, uplo)

    if iszero(info) && !isempty(M) && ispositive(rank)
        #
        # Use only the first `rank` columns/rows for the Schur complement
        #
        rd = view(d, 1:rank)

        if UPLO === :L
            rL = view(L, :, 1:rank)
            W = view(Wval, 1:size(L, 1) * rank)
            W₂₁ = reshape(W, size(L, 1), rank)
        else
            rL = view(L, 1:rank, :)
            W = view(Wval, 1:rank * size(L, 2))
            W₂₁ = reshape(W, rank, size(L, 2))
        end
        #
        #     M ← M - rL rd rLᴴ
        #
        if UPLO === :L
            syrk!(uplo, Val(:N), -one(real(T)), W₂₁, rL, rd, one(real(T)), M)
        else
            syrk!(uplo, Val(:C), -one(real(T)), W₂₁, rL, rd, one(real(T)), M)
        end
    end

    return info, rank
end

function ldlt_piv_factor!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        Wval::AbstractVector{T},
        d::AbstractVector{T},
        P::AbstractVector{<:Integer},
        S::MaybeVector,
        tol::Real,
        uplo::Val{UPLO},
    ) where {T, UPLO}
    @assert size(D, 1) == size(D, 2) == length(d)
    @assert length(D) <= length(Wval)

    if UPLO === :L
        @assert size(L, 2) == size(D, 1)
    else
        @assert size(L, 1) == size(D, 1)
    end
    #
    # factorize D with pivoting
    #
    #     P' D P = L d L'
    #
    W = reshape(view(Wval, eachindex(D)), size(D))
    info, rank = qstrf!(uplo, W, D, d, P, S, nothing, tol)
    #
    # zero out the rank-deficient part of D and L
    #
    zerotri!(D, uplo, rank + 1:size(D, 1))

    if UPLO === :L
        zerorec!(L, axes(L, 1), rank + 1:size(D, 1))
    else
        zerorec!(L, rank + 1:size(D, 1), axes(L, 2))
    end

    if iszero(info) && !isempty(L) && ispositive(rank)
        #
        # permute L by pivot
        #
        W = reshape(view(Wval, eachindex(L)), size(L))
        copyrec!(W, L)

        if UPLO === :L
            copyrec!(L, W, axes(L, 1), view(P, axes(D, 1)))
        else
            copyrec!(L, W, view(P, axes(D, 1)), axes(L, 2))
        end
        #
        # Use only the first `rank` columns/rows for the solve
        #
        rD = view(D, 1:rank, 1:rank)
        rd = view(d, 1:rank)

        if UPLO === :L
            rL = view(L, :, 1:rank)
        else
            rL = view(L, 1:rank, :)
        end
        #
        #     rL ← rL rD⁻ᴴ
        #
        if UPLO === :L
            trsm!(Val(:R), Val(:L), Val(:C), Val(:U), one(T), rD, rL)
        else
            trsm!(Val(:L), Val(:U), Val(:C), Val(:U), one(T), rD, rL)
        end
        #
        #     rL ← rL rd⁻¹
        #
        if UPLO === :L
            @inbounds for k in axes(rL, 2)
                idk = inv(rd[k])
                for i in axes(rL, 1)
                    rL[i, k] *= idk
                end
            end
        else
            @inbounds for i in axes(rL, 2)
                for k in axes(rL, 1)
                    rL[k, i] *= inv(rd[k])
                end
            end
        end
    end

    return info, rank
end

# ============================= ldlt_piv_bwd! =============================

function ldlt_piv_bwd!(
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
        ns = ldlt_piv_bwd_loop!(
            mptr, mval, fval, Dptr, Lptr, Lval,
            res, rel, sep, chd, ns, j, invp, Fval, uplo
        )
    end

    return
end

function ldlt_piv_bwd_loop!(
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
