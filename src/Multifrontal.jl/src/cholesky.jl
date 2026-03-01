"""
    cholesky!(F::ChordalCholesky[, pivot::PivotingStrategy]; check=true, reg=NoRegularization())

Perform a Cholesky factorization of a sparse
positive-definite matrix.

### Example

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
  - `pivot`: pivoting strategy
  - `check`: if `check = true`, then the function errors if `F`
    is not positive definite
  - `reg`: dynamic regularization strategy

"""
function LinearAlgebra.cholesky!(F::ChordalCholesky{UPLO, T, I}, pivot::PivotingStrategy=NoPivot(); check::Bool=true, reg::AbstractRegularization=NoRegularization(), tol::Real=-one(real(T))) where {UPLO, T, I <: Integer}
    return chol!(F, pivot, Ones{T}(size(F, 1)), reg, check, tol, Val(:N))
end

"""
    ldlt!(F::ChordalLDLt[, pivot::PivotingStrategy]; check=true, signs=Zeros(size(F, 1)), reg=NoRegularization())

Perform an LDLᵀ factorization of a sparse quasi-definite
matrix.

### Example

Use [`ChordalLDLt`](@ref) to construct a factorization object.
Use [`ldlt!`](@ref) to perform the factorization.

```julia-repl
julia> using CliqueTrees.Multifrontal, LinearAlgebra

julia> A = [
           4  2  0  0  2
           2  5  0  0  3
           0  0  4  2  0
           0  0  2  5  2
           2  3  0  2  7
       ];

julia> F = ldlt!(ChordalLDLt(A))
5×5 FChordalLDLt{:L, Float64, Int64} with 10 stored entries:
 1.0   ⋅    ⋅    ⋅    ⋅
 0.0  1.0   ⋅    ⋅    ⋅
 0.0  0.5  1.0   ⋅    ⋅
 0.5  0.0  0.0  1.0   ⋅
 0.0  0.5  0.5  0.5  1.0

 4.0   ⋅    ⋅    ⋅    ⋅
  ⋅   4.0   ⋅    ⋅    ⋅
  ⋅    ⋅   4.0   ⋅    ⋅
  ⋅    ⋅    ⋅   4.0   ⋅
  ⋅    ⋅    ⋅    ⋅   4.0
```

### Parameters

   - `F`: quasi-definite matrix
   - `pivot`: pivoting strategy
   - `check`: if `check = true`, then the function errors
     if the matrix is singular
   - `signs`: pivot signs
   - `reg`: dynamic regularization strategy

"""
function LinearAlgebra.ldlt!(F::ChordalLDLt{UPLO, T, I}, pivot::PivotingStrategy=NoPivot(); signs::AbstractVector=Zeros{T}(size(F, 1)), reg::AbstractRegularization=NoRegularization(), check::Bool=true, tol::Real=-one(real(T))) where {UPLO, T, I <: Integer}
    S = permuteto(T, signs, F.perm)
    return chol!(F, pivot, S, reg, check, tol, Val(:U))
end

function chol!(F::ChordalFactorization{DIAG, UPLO, T, I}, ::NoPivot, S::AbstractVector{T}, reg::AbstractRegularization, check::Bool, tol::Real, diag::Val{DIAG}) where {DIAG, UPLO, T, I <: Integer}
    @assert checksigns(S, reg)

    Mptr = FVector{I}(undef, F.S.nMptr)
    Mval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)

    info = chol_impl!(Mptr, Mval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, F.d, Fval, F.S.res, F.S.rel, F.S.chd, Val{UPLO}(), S, reg, diag)

    if isnegative(info)
        throw(ArgumentError(info))
    elseif ispositive(info) && check
        if DIAG === :N
            throw(PosDefException(F.perm[info]))
        else
            throw(SingularException(F.perm[info]))
        end
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
        d::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        uplo::Val{UPLO},
        S::AbstractVector{T},
        R::AbstractRegularization,
        diag::Val{DIAG},
    ) where {T, I <: Integer, UPLO, DIAG}

    ns = zero(I); Mptr[one(I)] = one(I)

    for j in vertices(res)
        ns, info = chol_loop!(Mptr, Mval, Dptr, Dval, Lptr, Lval, d, Fval, res, rel, chd, ns, j, uplo, S, R, diag)

        if isnegative(info)
            return info
        elseif ispositive(info)
            return info + pointers(res)[j] - one(I)
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
        d::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        uplo::Val{UPLO},
        S::AbstractVector{T},
        R::AbstractRegularization,
        diag::Val{DIAG},
    ) where {T, I <: Integer, UPLO, DIAG}
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
    d₁₁ = view(d, neighbors(res, j))
    S₁₁ = view(S, neighbors(res, j))
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
        M₂₂ = reshape(view(Mval, one(I):zero(I)), zero(I), zero(I))
    end

    info = convert(I, chol_kernel!(D₁₁, L₂₁, M₂₂, Fval, d₁₁, S₁₁, R, uplo, diag))

    if ispositive(na) && ispositive(info)
        ns -= one(I)
    end

    return ns, info
end

@inline function chol_kernel!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        M::AbstractMatrix{T},
        W::AbstractVector{T},
        d::AbstractVector{T},
        S::AbstractVector{T},
        R::AbstractRegularization,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {T, UPLO, DIAG}
    @assert size(D, 1) == size(D, 2)
    @assert size(M, 1) == size(M, 2)

    if UPLO === :L
        @assert size(L, 1) == size(M, 1)
        @assert size(L, 2) == size(D, 1)
    else
        @assert size(L, 1) == size(D, 1)
        @assert size(L, 2) == size(M, 1)
    end

    if DIAG === :U
        @assert size(D, 1) == length(d)
        @assert length(L) <= length(W)
    end
    #
    # factorize D
    #
    #     D ← cholesky(D)
    #
    info = chol_factor!(D, L, W, d, S, R, uplo, diag)

    if iszero(info) && !isempty(M)
        if UPLO === :L
            trans = Val(:N)
        else
            trans = Val(:C)
        end
        #
        #     M ← M - L Lᴴ
        #
        syrk!(uplo, trans, -one(real(T)), W, L, d, one(real(T)), M, diag)
    end

    return info
end

@inline function chol_factor!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        W::AbstractVector{T},
        d::AbstractVector{T},
        S::AbstractVector{T},
        R::AbstractRegularization,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {T, UPLO, DIAG}
    @assert size(D, 1) == size(D, 2)

    if DIAG === :U
        @assert length(d) == size(D, 1)
        @assert length(D) <= length(W)

        if UPLO === :L
            @assert size(L, 2) == size(D, 1)
        else
            @assert size(L, 1) == size(D, 1)
        end
    end
    #
    # factorize D
    #
    #     D ← cholesky(D)
    #
    info = potrf!(uplo, W, D, d, S, R, diag)

    if iszero(info) && !isempty(L)
        if UPLO === :L
            side = Val(:R)
        else
            side = Val(:L)
        end
        #
        #     L ← L D⁻ᴴ
        #
        trsm!(side, uplo, Val(:C), diag, one(T), D, L)
        cdiv!(side, diag, L, d)
    end

    return info
end

# ===== GMW81 Specialization =====
#
# GMW81 requires blocked factorization with access to both D and L
#

function chol_factor!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        W::AbstractVector{T},
        d::AbstractVector{T},
        S::AbstractVector{T},
        R::GMW81,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {T, UPLO, DIAG}
    if UPLO === :L
        @assert size(D, 1) == size(D, 2) == size(L, 2)
    else
        @assert size(D, 1) == size(D, 2) == size(L, 1)
    end

    if DIAG === :U
        @assert length(d) == size(D, 1)
    end

    n = size(D, 1)

    @inbounds for bstrt in 1:THRESHOLD:n
        bstop = min(bstrt + THRESHOLD - 1, n)

        if UPLO === :L
            Dbb = view(D, bstrt:n, bstrt:bstop)
            Lnb = view(L, :, bstrt:bstop)
        else
            Dbb = view(D, bstrt:bstop, bstrt:n)
            Lnb = view(L, bstrt:bstop, :)
        end

        dbb = view(d, bstrt:bstop)
        Sbb = view(S, bstrt:bstop)

        info = chol_factor_block!(Dbb, Lnb, dbb, Sbb, R, uplo, diag)
        !iszero(info) && return bstrt + info - 1

        if bstop < n
            Drr = view(D, bstop + 1:n, bstop + 1:n)

            if UPLO === :L
                Drb = view(D, bstop + 1:n, bstrt:bstop)
                tA = Val(:N)
                tB = Val(:C)
            else
                Drb = view(D, bstrt:bstop, bstop + 1:n)
                tA = Val(:C)
                tB = Val(:N)
            end

            syrk!(uplo, tA, -one(real(T)), W, Drb, dbb, one(real(T)), Drr, diag)

            if UPLO === :L
                Lnr = view(L, :, bstop + 1:n)
                A = Lnb
                B = Drb
            else
                Lnr = view(L, bstop + 1:n, :)
                A = Drb
                B = Lnb
            end

            gemm!(tA, tB, -one(T), W, A, B, dbb, one(T), Lnr, diag)
        end
    end

    return 0
end

function chol_factor_block!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        d::AbstractVector{T},
        S::AbstractVector{T},
        R::GMW81,
        uplo::Val{:L},
        diag::Val{DIAG},
    ) where {T, DIAG}
    @assert size(D, 2) == size(L, 2)

    if DIAG === :U
        @assert length(d) == size(D, 2)
    end

    @inbounds for j in axes(D, 2)
        Djj = regularize(R, S, D, L, real(D[j, j]), j, uplo)

        if DIAG === :U
            d[j] = Djj
        else
            D[j, j] = Djj = sqrt(Djj)
        end

        iDjj = inv(Djj)

        for i in j + 1:size(D, 1)
            D[i, j] *= iDjj
        end

        for i in axes(L, 1)
            L[i, j] *= iDjj
        end

        for k in j + 1:size(D, 2)
            Dkj = D[k, j]
            cDkj = conj(Dkj)

            if DIAG === :N
                D[k, k] -= abs2(Dkj)
            else
                D[k, k] -= Djj * abs2(Dkj)
            end

            for i in k + 1:size(D, 1)
                if DIAG === :N
                    D[i, k] -= D[i, j] * cDkj
                else
                    D[i, k] -= D[i, j] * Djj * cDkj
                end
            end

            for i in axes(L, 1)
                if DIAG === :N
                    L[i, k] -= L[i, j] * cDkj
                else
                    L[i, k] -= L[i, j] * Djj * cDkj
                end
            end
        end
    end

    return 0
end

function chol_factor_block!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        d::AbstractVector{T},
        S::AbstractVector{T},
        R::GMW81,
        uplo::Val{:U},
        diag::Val{DIAG},
    ) where {T, DIAG}
    @assert size(D, 1) == size(L, 1)

    if DIAG === :U
        @assert length(d) == size(D, 1)
    end

    @inbounds for j in axes(D, 1)
        for i in j + 1:size(D, 2)
            for k in 1:j - 1
                if DIAG === :U
                    D[j, i] -= D[k, i] * d[k] * conj(D[k, j])
                else
                    D[j, i] -= D[k, i] * conj(D[k, j])
                end
            end
        end

        for i in axes(L, 2)
            for k in 1:j - 1
                if DIAG === :U
                    L[j, i] -= L[k, i] * d[k] * conj(D[k, j])
                else
                    L[j, i] -= L[k, i] * conj(D[k, j])
                end
            end
        end

        Djj = real(D[j, j])

        for k in 1:j - 1
            if DIAG === :U
                Djj -= d[k] * abs2(D[k, j])
            else
                Djj -= abs2(D[k, j])
            end
        end

        Djj = regularize(R, S, D, L, Djj, j, uplo)

        if DIAG === :U
            d[j] = Djj
        else
            D[j, j] = Djj = sqrt(Djj)
        end

        iDjj = inv(Djj)

        for i in j + 1:size(D, 2)
            D[j, i] *= iDjj
        end

        for i in axes(L, 2)
            L[j, i] *= iDjj
        end
    end

    return 0
end

function chol_send!(
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
    # M is the update matrix from child i
    #
    strt = ptr[ns]
    M = reshape(view(val, strt:strt + na * na - one(I)), na, na)
    #
    # add M to F
    #
    #     F ← F + inj M injᵀ
    #
    addtri!(F, M, uplo, inj)
    return
end
