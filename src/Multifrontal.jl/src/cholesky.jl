"""
    cholesky!(F::ChordalCholesky; check=true)

Perform a Cholesky factorization of a sparse
positive-definite matrix.

### Basic Usage

Use [`ChordalCholesky`](@ref) to construct a factorization object,
and use [`cholesky!`](@ref) to perform the factorization.

```julia-repl
julia> using CliqueTrees.Multifrontal, LinearAlgebra

julia> A = [
           4  2  0  0  2
           2  5  0  0  3
           0  0  4  2  0
           0  0  2  5  2
           2  3  0  2  7
       ];

julia> F = cholesky!(ChordalCholesky(A))
5×5 FChordalCholesky{:L, Float64, Int64} with 10 stored entries:
 2.0   ⋅    ⋅    ⋅    ⋅ 
 0.0  2.0   ⋅    ⋅    ⋅ 
 0.0  1.0  2.0   ⋅    ⋅ 
 1.0  0.0  0.0  2.0   ⋅ 
 0.0  1.0  1.0  1.0  2.0
```

## Parameters

  - `F`: positive-definite matrix
  - `check`: if `check = true`, then the function errors if `F`
    is not positive definite

"""
function LinearAlgebra.cholesky!(F::ChordalCholesky{UPLO, T, I}, ::NoPivot = NoPivot(); check::Bool=true) where {UPLO, T, I <: Integer}
    Mptr = FVector{I}(undef, F.S.nMptr)
    Mval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)
    info = chol_impl!(Mptr, Mval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, Fval, F.S.res, F.S.rel, F.S.chd, Val{UPLO}())

    if isnegative(info)
        throw(ArgumentError(info))
    elseif ispositive(info) && check
        throw(PosDefException(F.perm[info]))
    elseif ispositive(info) && !check
        F.info[] = F.perm[info]
    else
        F.info[] = zero(I)
    end

    return F
end

function chol_impl!(
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
    ) where {T, I <: Integer, UPLO}

    ns = zero(I); Mptr[one(I)] = one(I)

    for j in vertices(res)
        ns, localinfo = chol_loop!(Mptr, Mval, Dptr, Dval, Lptr, Lval, Fval, res, rel, chd, ns, j, uplo)

        if isnegative(localinfo)
            return localinfo
        elseif ispositive(localinfo)
            return localinfo + pointers(res)[j] - one(I)
        end
    end

    return zero(I)
end

function chol_loop!(
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
    # factorize D₁₁
    #
    #     D₁₁ ← chol(D₁₁)
    #
    info = convert(I, potrf!(uplo, D₁₁))

    if iszero(info) && ispositive(na)
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
    end

    return ns, info
end

function chol_update!(
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
    #     na = | sep(j) |
    #
    na = eltypedegree(rel, i)
    #
    # inj is the subset inclusion
    #
    #     inj: sep(i) → bag(parent(i))
    #
    inj = neighbors(rel, i)
    #
    # S is the update matrix from child i
    #
    strt = ptr[ns]
    S = reshape(view(val, strt:strt + na * na - one(I)), na, na)
    #
    # add S to F
    #
    #     F ← F + inj S injᵀ
    #
    addtri!(F, S, uplo, inj)
    return
end
