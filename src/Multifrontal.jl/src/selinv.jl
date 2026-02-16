"""
    selinv!(F::ChordalCholesky)

Compute the selected inverse of a sparse positive-definite
matrix. This computes the inverse of the matrix *only* in the
structural nonzeros of the sparsity pattern of the Cholesky
factor `F.L`.

Use [`ChordalCholesky`](@ref) to construct a factorization object,
use [`cholesky!`](@ref) to perform the factorization, and use
[`selinv!](@ref) to compute the selected inverse.

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

  - `F` factorized positive definite matrix

"""
function selinv!(F::ChordalCholesky{UPLO, T, I}) where {UPLO, T, I <: Integer}
    Mptr = FVector{I}(undef, F.S.nMptr)
    Mval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)
    selinv_impl!(Mptr, Mval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, Fval, F.S.res, F.S.rel, F.S.chd, Val{UPLO}())
    return F
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

    return
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
    #     F₁₁ ← D₁₁
    #
    zerorec!(F₁₁)
    copytri!(F₁₁, D₁₁, uplo)
    #
    #     F₁₁ ← F₁₁⁻¹
    #
    trtri!(uplo, Val(:N), F₁₁)

    if ispositive(na)
        #
        #     F₂₁ ← L₂₁
        #
        copyrec!(F₂₁, L₂₁)
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
        #     F₂₁ ← F₂₁ F₁₁
        #
        if UPLO === :L
            trmm!(Val(:R), Val(:L), Val(:N), Val(:N), one(T), F₁₁, F₂₁)
        else
            trmm!(Val(:L), Val(:U), Val(:N), Val(:N), one(T), F₁₁, F₂₁)
        end
        #
        #     L₂₁ ← -F₂₂ F₂₁
        #
        if UPLO === :L
            symm!(Val(:L), Val(:L), -one(T), F₂₂, F₂₁, zero(T), L₂₁)
        else
            symm!(Val(:R), Val(:U), -one(T), F₂₂, F₂₁, zero(T), L₂₁)
        end
        #
        #     D₁₁ ← F₁₁ᴴ F₁₁ - F₂₁ᴴ L₂₁
        #
        if UPLO === :L
            syrk!(Val(:L), Val(:C), one(real(T)), F₁₁, zero(real(T)), D₁₁)
            trrk!(Val(:L), Val(:C), F₂₁, L₂₁, D₁₁)
        else
            syrk!(Val(:U), Val(:N), one(real(T)), F₁₁, zero(real(T)), D₁₁)
            trrk!(Val(:U), Val(:N), L₂₁, F₂₁, D₁₁)
        end
    else
        #
        #     D₁₁ ← F₁₁ᴴ F₁₁
        #
        if UPLO === :L
            syrk!(Val(:L), Val(:C), one(real(T)), F₁₁, zero(real(T)), D₁₁)
        else
            syrk!(Val(:U), Val(:N), one(real(T)), F₁₁, zero(real(T)), D₁₁)
        end
    end
    #
    #     F₁₁ ← D₁₁
    #
    copyrec!(F₁₁, D₁₁)
    #
    #     F₂₁ ← L₂₁
    #
    copyrec!(F₂₁, L₂₁)

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
    copytri!(M, F, uplo, inj)
    return
end
