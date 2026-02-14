"""
    complete!(F::ChordalCholesky)

Compute the Cholesky factorization of the inverse of
the maximum-determinant positive-definite completion
of a symmetric matrix.

### Basic Usage

Use [`ChordalCholesky`](@ref) to construct a factorization object,
and use [`complete!`](@ref) to perform the factorization.

```julia-repl
julia> using CliqueTrees.Multifrontal, LinearAlgebra

julia> A = [
           4  2  0  0  2
           2  5  0  0  3
           0  0  4  2  0
           0  0  2  5  2
           2  3  0  2  7
       ];

julia> F = complete!(ChordalCholesky(A))
5×5 FChordalCholesky{:L, Float64, Int64} with 10 stored entries:
  0.5590169943749475     ⋅                   …   ⋅ 
  0.0                   0.5700877125495689       ⋅ 
  0.0                  -0.17541160386140583      ⋅ 
 -0.22360679774997896   0.0                      ⋅ 
  0.0  
```

## Parameters

  - `F`: symmetric matrix
  - `check`: if `check = true`, then the function errors if `F`
    is not positive definite

"""
function complete!(F::ChordalCholesky{UPLO, T, I}) where {UPLO, T, I <: Integer}
    Mptr = FVector{I}(undef, F.S.nMptr)
    Mval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, twice(F.S.nFval * F.S.nFval))
    cmpl_impl!(Mptr, Mval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, Fval, F.S.res, F.S.rel, F.S.chd, Val{UPLO}())
    return F
end

function cmpl_impl!(
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

    for j in reverse(vertices(res))
        ns = cmpl_loop!(Mptr, Mval, Dptr, Dval, Lptr, Lval, Fval, res, rel, chd, ns, j, uplo)
    end

    return
end

function cmpl_loop!(
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
    # W is a workspace
    #
    #           nn
    #     W = [ W₁₁ ] nn
    #
    W₁₁ = reshape(view(Fval, nj * nj + one(I):nj * nj + nn * nn), nn, nn)
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
    # copy L into F
    #
    #     F₁₁ ← D₁₁
    #     F₂₁ ← L₂₁
    #
    copytri!(F₁₁, D₁₁, uplo)
    copyrec!(F₂₁, L₂₁)

    if ispositive(na)
        #
        # M₂₂ is the update matrix from the parent of node j
        #
        strt = Mptr[ns]
        M₂₂ = reshape(view(Mval, strt:strt + na * na - one(I)), na, na)
        ns -= one(I)
        #
        # copy M into F
        #
        #     F₂₂ ← M₂₂
        #
        copytri!(F₂₂, M₂₂, uplo)
        #
        # factorize M₂₂
        #
        #     M₂₂ ← chol(M₂₂)
        #
        info = potrf!(uplo, M₂₂)

        @assert iszero(info)
        #
        # solve
        #
        #     L₂₁ ← M₂₂⁻¹ L₂₁
        #
        if UPLO === :L
            trsm!(Val(:L), Val(:L), Val(:N), Val(:N), one(T), M₂₂, L₂₁)
        else
            trsm!(Val(:R), Val(:U), Val(:N), Val(:N), one(T), M₂₂, L₂₁)
        end
        #
        #     D₁₁ ← D₁₁ - L₂₁ᴴ L₂₁
        #
        if UPLO === :L
            syrk!(Val(:L), Val(:C), -one(real(T)), L₂₁, one(real(T)), D₁₁)
        else
            syrk!(Val(:U), Val(:N), -one(real(T)), L₂₁, one(real(T)), D₁₁)
        end
        #
        # solve
        #
        #     L₂₁ ← -M₂₂⁻ᴴ L₂₁
        #
        if UPLO === :L
            trsm!(Val(:L), Val(:L), Val(:C), Val(:N), -one(T), M₂₂, L₂₁)
        else
            trsm!(Val(:R), Val(:U), Val(:C), Val(:N), -one(T), M₂₂, L₂₁)
        end
    end
    #
    # factorize D₁₁
    #
    #     D₁₁ ← chol(D₁₁)
    #
    info = potrf!(uplo, D₁₁)
    @assert iszero(info)
    #
    # copy L into W
    #
    #     W₁₁ ← D₁₁
    #
    copytri!(W₁₁, D₁₁, uplo)    
    #
    # invert D₁₁
    #
    #     D₁₁ ← D₁₁⁻¹
    #
    trtri!(uplo, Val(:N), D₁₁)
    #
    # solve
    #
    #    D₁₁ ← W₁₁⁻ᴴ D₁₁
    #
    if UPLO === :L
        trsm!(Val(:L), Val(:L), Val(:C), Val(:N), one(T), W₁₁, D₁₁)
    else
        trsm!(Val(:R), Val(:U), Val(:C), Val(:N), one(T), W₁₁, D₁₁)
    end
    #
    # factorize D₁₁ (again)
    #
    #     D₁₁ ← chol(D₁₁)
    #
    info = potrf!(uplo, D₁₁)
    @assert iszero(info)

    if ispositive(na)
        #
        #     L₂₁ ← L₂₁ D₁₁
        #
        if UPLO === :L
            trmm!(Val(:R), Val(:L), Val(:N), Val(:N), one(T), D₁₁, L₂₁)
        else
            trmm!(Val(:L), Val(:U), Val(:N), Val(:N), one(T), D₁₁, L₂₁)
        end
    end

    for i in neighbors(chd, j)
        #
        # send update matrix to child i
        #
        #     Mᵢ ← Rᵢᵀ F Rᵢ
        #
        ns += one(I)
        cmpl_update!(F, Mptr, Mval, rel, ns, i, uplo)
    end

    return ns
end

function cmpl_update!(
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
