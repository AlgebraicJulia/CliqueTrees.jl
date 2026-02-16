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
    # S₂₂ is the update matrix for node j
    #
    if ispositive(na)
        ns += one(I)
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * na
        S₂₂ = reshape(view(Mval, strt:stop - one(I)), na, na)
        #
        #     S₂₂ ← F₂₂
        #
        copytri!(S₂₂, F₂₂, uplo)
    else
        S₂₂ = reshape(view(Mval, oneto(zero(I))), zero(I), zero(I))
    end

    if nj <= THRESHOLD
        info = convert(I, chol_kernel!(D₁₁, L₂₁, S₂₂, uplo, Val(:N)))
    else
        info = convert(I, chol_kernel!(D₁₁, L₂₁, S₂₂, uplo, Val(:S)))
    end

    if ispositive(na) && ispositive(info)
        ns -= one(I)
    end

    return ns, info
end

function chol_kernel!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        S::AbstractMatrix{T},
        uplo::Val{UPLO},
        node::Val{NODE},
    ) where {T, UPLO, NODE}
    @assert size(D, 1) == size(D, 2)
    @assert size(S, 1) == size(S, 2)

    if UPLO === :L
        @assert size(L, 1) == size(S, 1)
        @assert size(L, 2) == size(D, 1)
    else
        @assert size(L, 1) == size(D, 1)
        @assert size(L, 2) == size(S, 1)
    end
    #
    # factorize D
    #
    #     D ← chol(D)
    #
    info = potrf!(uplo, D)

    if iszero(info) && !isempty(S)
        #
        #     L ← L D⁻ᴴ
        #
        if UPLO === :L
            trsm!(Val(:R), Val(:L), Val(:C), Val(:N), one(T), D, L)
        else
            trsm!(Val(:L), Val(:U), Val(:C), Val(:N), one(T), D, L)
        end
        #
        #     S ← S - L Lᴴ
        #
        if UPLO === :L
            syrk!(Val(:L), Val(:N), -one(real(T)), L, one(real(T)), S)
        else
            syrk!(Val(:U), Val(:C), -one(real(T)), L, one(real(T)), S)
        end
    end

    return info
end

function chol_kernel!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        S::AbstractMatrix{T},
        uplo::Val{:L},
        node::Val{:N},
    ) where {T}
    @assert size(D, 1) == size(D, 2)
    @assert size(S, 1) == size(S, 2)
    @assert size(L, 1) == size(S, 1)
    @assert size(L, 2) == size(D, 1)

    @inbounds for j in axes(D, 1)
        Djj = real(D[j, j])

        Djj > zero(real(T)) || return j

        D[j, j] = Djj = sqrt(Djj); iDjj = inv(Djj); ciDjj = conj(iDjj)

        for i in j + 1:size(D, 1)
            D[i, j] *= iDjj
        end

        for k in j + 1:size(D, 1)
            Dkj = D[k, j]; cDkj = conj(Dkj)

            D[k, k] -= abs2(Dkj)

            for i in k + 1:size(D, 1)
                D[i, k] -= D[i, j] * cDkj
            end
        end

        for i in axes(S, 1)
            L[i, j] *= ciDjj
        end

        for k in j + 1:size(D, 1)
            cDkj = conj(D[k, j])

            for i in axes(S, 1)
                L[i, k] -= L[i, j] * cDkj
            end
        end

        for k in axes(S, 1)
            Ljk = L[k, j]; cLjk = conj(Ljk)

            S[k, k] -= abs2(Ljk)

            for i in k + 1:size(S, 1)
                S[i, k] -= L[i, j] * cLjk
            end
        end
    end

    return 0
end

function chol_kernel!(
        D::AbstractMatrix{T},
        U::AbstractMatrix{T},
        S::AbstractMatrix{T},
        uplo::Val{:U},
        node::Val{:N},
    ) where {T}
    @assert size(D, 1) == size(D, 2)
    @assert size(S, 1) == size(S, 2)
    @assert size(U, 1) == size(D, 1)
    @assert size(U, 2) == size(S, 1)

    @inbounds for j in axes(D, 1)
        Djj = real(D[j, j])

        for k in 1:j - 1
            Djj -= abs2(D[k, j])
        end

        Djj > zero(real(T)) || return j

        D[j, j] = Djj = sqrt(Djj); iDjj = inv(Djj); ciDjj = conj(iDjj)

        for i in j + 1:size(D, 1)
            Dji = D[j, i]

            for k in 1:j - 1
                Dji -= conj(D[k, i]) * D[k, j]
            end

            D[j, i] = Dji * iDjj
        end

        for i in axes(S, 1)
            Uji = U[j, i]

            for k in 1:j - 1
                Uji -= conj(U[k, i]) * D[k, j]
            end

            U[j, i] = Uji * ciDjj
        end

        for k in axes(S, 1)
            Ujk = U[j, k]; cUjk = conj(Ujk)

            S[k, k] -= abs2(Ujk)

            for i in 1:k - 1
                S[i, k] -= U[j, i] * cUjk
            end
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
