# ===== Pivoted Cholesky Factorization =====

"""
    cholesky!(F::ChordalCholesky, RowMaximum(); tol=-1.0)

Perform a pivoted Cholesky factorization of a sparse
positive semi-definite matrix.

### Basic Usage

Use [`ChordalCholesky`](@ref) to construct a factorization object,
and use [`cholesky!`](@ref) with `RowMaximum()` to perform the
pivoted factorization.

```julia-repl
julia> using CliqueTrees.Multifrontal, LinearAlgebra

julia> A = [
           4  2  0  0  2
           2  5  0  0  3
           0  0  4  2  0
           0  0  2  5  2
           2  3  0  2  7
       ];

julia> F = cholesky!(ChordalCholesky(A), RowMaximum())
```

## Parameters

  - `F`: positive semi-definite matrix
  - `tol`: pivot tolerance (default: -1.0, uses LAPACK default)

"""
function LinearAlgebra.cholesky!(F::ChordalCholesky{UPLO, T, I}, ::RowMaximum; check::Bool=true, tol::Real=-one(real(T))) where {UPLO, T, I <: Integer}
    R = real(T)

    Mptr = FVector{I}(undef, F.S.nMptr)
    Mval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)
    piv  = FVector{BlasInt}(undef, F.S.nFval)
    work = FVector{R}(undef, twice(F.S.nFval))
    invp = FVector{I}(undef, nov(F.S.res))
    mval = FVector{I}(undef, F.S.nNval)
    fval = FVector{I}(undef, F.S.nFval)

    info, totalrank = cholpiv_fwd!(
        Mptr, Mval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, Fval,
        F.S.res, F.S.rel, F.S.chd, piv, work, invp, convert(R, tol), Val{UPLO}()
    )

    if isnegative(info)
        throw(ArgumentError(info))
    else
        F.info[] = zero(I)
        F.rank[] = totalrank
    end

    cholpiv_bwd!(Mptr, mval, fval, F.S.Dptr, F.S.Lptr, F.Lval, F.S.res, F.S.rel, F.S.sep, F.S.chd, invp, Fval, Val{UPLO}())
    cholpiv_rel!(F.S.res, F.S.sep, F.S.rel, F.S.chd)
    invpermute!(F.perm, invp)
    return F
end

# ============================= cholpiv_fwd! =============================

function cholpiv_fwd!(
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
        piv::AbstractVector{BlasInt},
        work::AbstractVector{<:Real},
        invp::AbstractVector{I},
        tol::Real,
        uplo::Val{UPLO},
    ) where {T, I <: Integer, UPLO}

    ns = zero(I); Mptr[one(I)] = one(I)
    totalrank = zero(I)

    for j in vertices(res)
        ns, localinfo, localrank = cholpiv_fwd_loop!(
            Mptr, Mval, Dptr, Dval, Lptr, Lval, Fval,
            res, rel, chd, ns, j, piv, work, invp, tol, uplo
        )

        if isnegative(localinfo)
            return localinfo, totalrank
        end

        totalrank += localrank
    end

    return zero(I), totalrank
end

function cholpiv_fwd_loop!(
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
    #     F ← 0
    #
    zerotri!(F, uplo)

    for i in Iterators.reverse(neighbors(chd, j))
        #
        # add the update matrix for child i to F
        #
        #   F ← F + Rᵢ Sᵢ Rᵢᵀ
        #
        chol_update!(F, Mptr, Mval, rel, ns, i, uplo)
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
    #     invp₁₁' D₁₁ invp₁₁ = L₁₁ L₁₁'
    #
    info, localrank = pstrf!(uplo, D₁₁, piv, work, tol)
    #
    # update invp with local pivot permutation
    # invp maps P-indices to Q-indices
    # piv[k] = p means vertex (offset + p) gets Q-index (offset + k)
    #
    offset = first(neighbors(res, j)) - one(I)

    @inbounds for v in oneto(nn)
        invp[offset + piv[v]] = offset + v
    end

    if ispositive(na) && ispositive(localrank)
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
        if UPLO === :L
            permutecols!(L₂₁, view(piv, oneto(nn)))
        else
            permuterows!(L₂₁, view(piv, oneto(nn)))
        end
        #
        # Use only the first `localrank` columns/rows for the solve
        #
        D₁₁ = view(D₁₁, oneto(localrank), oneto(localrank))

        if UPLO === :L
            L₂₁ = view(L₂₁, oneto(na), oneto(localrank))
        else
            L₂₁ = view(L₂₁, oneto(localrank), oneto(na))
        end
        #
        #     L₂₁ ← L₂₁ D₁₁⁻ᴴ
        #
        if UPLO === :L
            trsm!(Val(:R), Val(:L), Val(:C), Val(:N), one(T), D₁₁, L₂₁)
        else
            trsm!(Val(:L), Val(:U), Val(:C), Val(:N), one(T), D₁₁, L₂₁)
        end
        #
        #     S₂₂ ← S₂₂ - L₂₁ L₂₁ᴴ
        #
        if UPLO === :L
            syrk!(Val(:L), Val(:N), -one(real(T)), L₂₁, one(real(T)), S₂₂)
        else
            syrk!(Val(:U), Val(:C), -one(real(T)), L₂₁, one(real(T)), S₂₂)
        end
    elseif ispositive(na) && iszero(localrank)
        # Rank-deficient: just copy F₂₂ to update matrix
        ns += one(I)
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * na
        S₂₂ = reshape(view(Mval, strt:stop - one(I)), na, na)
        copytri!(S₂₂, F₂₂, uplo)
    end

    return ns, info, localrank
end

# ============================= cholpiv_bwd! =============================

function cholpiv_bwd!(
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
        ns = cholpiv_bwd_loop!(
            mptr, mval, fval, Dptr, Lptr, Lval,
            res, rel, sep, chd, ns, j, invp, Fval, uplo
        )
    end

    return
end

function cholpiv_bwd_loop!(
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
    # L₂₁ is the off-diagonal block of the Cholesky factor
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

function cholpiv_bwd_update!(
        f::AbstractVector{I},
        ptr::AbstractVector{I},
        val::AbstractVector{I},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
    ) where {I <: Integer}
    #
    # na is the size of the separator at node i
    #
    #     na = |sep(i)|
    #
    na = eltypedegree(rel, i)
    #
    # inj: sep(i) → bag(parent(i))
    #
    inj = neighbors(rel, i)
    #
    # mᵢ ← Rᵢᵀ f
    #
    strt = ptr[ns]
    stop = ptr[ns + one(I)] = strt + na
    m = view(val, strt:stop - one(I))

    copyrec!(m, f, inj)

    return
end

# ============================= cholpiv_rel! =============================

function cholpiv_rel!(
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
    ) where {I <: Integer}

    for j in vertices(res)
        cholpiv_rel_loop!(res, sep, rel, chd, j)
    end

    return
end

function cholpiv_rel_loop!(
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        j::I,
    ) where {I <: Integer}
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
    na = eltypedegree(sep, j)
    #
    # bag(j) = res(j) ∪ sep(j), both sorted by Q-index
    #
    res_j = neighbors(res, j)
    sep_j = neighbors(sep, j)
    #
    # for each child i, recompute rel(i)
    #
    #     rel(i) : sep(i) → bag(j)
    #
    for i in neighbors(chd, j)
        #
        # sep(i) ⊆ bag(j), both sorted by Q-index
        # use merge-style traversal to find positions
        #
        q = one(I)

        for p in incident(sep, i)
            v = targets(sep)[p]
            #
            # find v in bag(j)
            # bag(j)[q] = res_j[q] if q ≤ nn, else sep_j[q - nn]
            #
            while true
                if q <= nn
                    w = res_j[q]
                else
                    w = sep_j[q - nn]
                end

                if w >= v
                    break
                end
                q += one(I)
            end

            targets(rel)[p] = q
        end
    end

    return
end

