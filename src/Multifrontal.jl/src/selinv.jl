"""
    selinv!(F::ChordalCholesky)

Compute the selected inverse of a sparse positive-definite
matrix. This computes the inverse of the matrix *only* in the
structural nonzeros of the sparsity pattern of the Cholesky
factor `F.L`.

Use [`ChordalCholesky`](@ref) to construct a factorization object,
use [`cholesky!`](@ref) to perform the factorization, and use
[`selinv!`](@ref) to compute the selected inverse.

```julia-repl
julia> using CliqueTrees.Multifrontal, LinearAlgebra

julia> A = [
           4  2  0  0  2
           2  5  0  0  3
           0  0  4  2  0
           0  0  2  5  2
           2  3  0  2  7
       ];

julia> F = selinv!(cholesky!(ChordalCholesky(A)))
5×5 FChordalCholesky{:L, Float64, Int64} with 10 stored entries:
  0.328125    ⋅         ⋅        ⋅       ⋅
  0.0        0.328125   ⋅        ⋅       ⋅
  0.0       -0.09375   0.3125    ⋅       ⋅
 -0.15625    0.0       0.0      0.3125   ⋅
  0.0       -0.0625   -0.125   -0.125   0.25
```

### Parameters

  - `F`: factorized positive definite matrix

"""
function selinv!(F::ChordalCholesky{UPLO, T}) where {UPLO, T}
    F.info[] = selinv!(triangular(F))
    return F
end

function selinv!(
        L::ChordalTriangular{:N, UPLO, T, I},
    ) where {UPLO, T, I <: Integer}
    Mptr = FVector{I}(undef, L.S.nMptr)
    Mval = FVector{T}(undef, L.S.nMval)
    Fval = FVector{T}(undef, L.S.nFval * L.S.nFval)

    info = selinv_impl!(Mptr, Mval, Fval, L)
    return info
end

#
# Convenience wrapper that unpacks ChordalTriangular types.
#
function selinv_impl!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Fval::AbstractVector{T},
        L::ChordalTriangular{:N, UPLO, T, I},
    ) where {UPLO, T, I <: Integer}
    info = selinv_impl!(
        Mptr, Mval,
        L.S.Dptr, L.Dval,
        L.S.Lptr, L.Lval,
        Fval,
        L.S.res, L.S.rel, L.S.chd,
        L.uplo)

    return info
end

function selinv_impl!(
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
        ns = selinv_loop!(Mptr, Mval, Dptr, Dval, Lptr, Lval, Fval, res, rel, chd, ns, j, uplo)
    end

    return zero(I)
end

function selinv_loop!(
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
        return selinv_loop_nod!(
            Mptr, Mval,
            Dptr, Dval,
            Lptr, Lval,
            Fval,
            res, rel, chd, ns, j, uplo
        )
    else
        return selinv_loop_snd!(
            Mptr, Mval,
            Dptr, Dval,
            Lptr, Lval,
            Fval,
            res, rel, chd, ns, nn, j, uplo
        )
    end
end

function selinv_loop_snd!(
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
    #     F = [ F₁₁ F₁₂ ] nn
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
    # L is part of the lower triangular factor
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
    #     F₁₁ ← 0
    #
    zerorec!(F₁₁)

    if ispositive(na)
        #
        # M₂₂ is the update matrix from the parent of node j
        #
        strt = Mptr[ns]
        M₂₂ = reshape(view(Mval, strt:strt + na * na - one(I)), na, na)
        ns -= one(I)
        #
        #     F₂₂ ← M₂₂
        #
        copytri!(F₂₂, M₂₂, uplo)
        #
        #     L₂₁ ← L₂₁ D₁₁⁻¹
        #
        if UPLO === :L
            trsm!(Val(:R), Val(:L), Val(:N), Val(:N), one(T), D₁₁, L₂₁)
        else
            trsm!(Val(:L), Val(:U), Val(:N), Val(:N), one(T), D₁₁, L₂₁)
        end
        #
        #     F₂₁ ← -M₂₂ L₂₁
        #
        if UPLO === :L
            symm!(Val(:L), Val(:L), -one(T), M₂₂, L₂₁, zero(T), F₂₁)
        else
            symm!(Val(:R), Val(:U), -one(T), M₂₂, L₂₁, zero(T), F₂₁)
        end
        #
        #     F₁₁ ← F₁₁ - L₂₁ᴴ F₂₁
        #
        if UPLO === :L
            trrk!(Val(:L), Val(:C), -one(real(T)), L₂₁, F₂₁, one(real(T)), F₁₁)
        else
            trrk!(Val(:U), Val(:N), -one(real(T)), F₂₁, L₂₁, one(real(T)), F₁₁)
        end
    end
    #
    #     D₁₁ ← D₁₁⁻¹
    #
    trtri!(uplo, Val(:N), D₁₁)
    #
    #     F₁₁ ← F₁₁ + D₁₁ᴴ D₁₁
    #
    if UPLO === :L
        tril!(D₁₁)
        syrk!(Val(:L), Val(:C), one(real(T)), D₁₁, one(real(T)), F₁₁)
    else
        triu!(D₁₁)
        syrk!(Val(:U), Val(:N), one(real(T)), D₁₁, one(real(T)), F₁₁)
    end
    #
    #     D₁₁ ← F₁₁
    #
    copyrec!(D₁₁, F₁₁)
    #
    #     L₂₁ ← F₂₁
    #
    copyrec!(L₂₁, F₂₁)

    for i in neighbors(chd, j)
        #
        # send update matrix to child i
        #
        #     Mᵢ ← Rᵢᵀ F Rᵢ
        #
        ns += one(I)
        selinv_send!(F, Mptr, Mval, rel, ns, i, uplo)
    end

    return ns
end

# Fast path for nn = 1 (residual size is 1)
# In this case, diagonal blocks are scalars and off-diagonal blocks are vectors
function selinv_loop_nod!(
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
    # L is part of the lower triangular factor (d₁₁ is scalar, l₂₁ is vector)
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
    #     f₁₁ ← 0
    #
    f₁₁ = zero(T)

    if ispositive(na)
        #
        # M₂₂ is the update matrix from the parent of node j
        #
        strt = Mptr[ns]
        M₂₂ = reshape(view(Mval, strt:strt + na * na - one(I)), na, na)
        ns -= one(I)
        #
        #     F₂₂ ← M₂₂
        #
        copytri!(F₂₂, M₂₂, uplo)
        #
        #     l₂₁ ← l₂₁ / d₁₁
        #
        rdiv!(l₂₁, d₁₁)
        #
        #     f₂₁ ← -M₂₂ l₂₁ (symv)
        #
        symv!(uplo, -one(T), M₂₂, l₂₁, zero(T), f₂₁)
        #
        #     f₁₁ ← f₁₁ - l₂₁ᴴ f₂₁ (dot product)
        #
        f₁₁ -= dot(l₂₁, f₂₁)
    end
    #
    #     d₁₁ ← 1 / d₁₁
    #
    d₁₁ = inv(d₁₁)
    #
    #     f₁₁ ← f₁₁ + |d₁₁|²
    #
    f₁₁ += abs2(d₁₁)
    #
    #     Write back scalars: D₁₁ ← f₁₁, F[1] ← f₁₁
    #
    Dval[Dp] = f₁₁
    F[one(I)] = f₁₁
    #
    #     l₂₁ ← f₂₁
    #
    copyrec!(l₂₁, f₂₁)

    for i in neighbors(chd, j)
        #
        # send update matrix to child i
        #
        #     Mᵢ ← Rᵢᵀ F Rᵢ
        #
        ns += one(I)
        selinv_send!(F, Mptr, Mval, rel, ns, i, uplo)
    end

    return ns
end

function selinv_send!(
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
