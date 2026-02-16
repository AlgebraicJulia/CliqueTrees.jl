"""
    ldlt!(F::ChordalLDLt; check=true, reg=nothing)

Perform an LDLᵀ factorization of a sparse quasi-definite
matrix.

### Basic Usage

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
   - `check`: if `check = true`, then the function errors
     if the matrix is singular
   - `reg`: triggers dynamic regularization

"""
function LinearAlgebra.ldlt!(F::ChordalLDLt{UPLO, T, I}; check::Bool=true, reg::Union{DynamicRegularization, Nothing}=nothing) where {UPLO, T, I <: Integer}
    Mptr = FVector{I}(undef, F.S.nMptr)
    Mval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)

    if !isnothing(reg)
        prg = DynamicRegularization{T, I}(nov(F.S.res), reg.delta, reg.epsilon)

        for i in outvertices(F.S.res)
            prg.signs[i] = reg.signs[F.perm[i]]
        end
    else
        prg = nothing
    end

    info = ldlt_impl!(Mptr, Mval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, F.d, Fval, F.S.res, F.S.rel, F.S.chd, Val{UPLO}(), prg)

    if isnegative(info)
        throw(ArgumentError(info))
    elseif ispositive(info) && check
        throw(SingularException(F.perm[info]))
    elseif ispositive(info) && !check
        F.info[] = F.perm[info]
    else
        F.info[] = zero(I)
    end

    return F
end

function ldlt_impl!(
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
        reg::Union{DynamicRegularization, Nothing},
    ) where {T, I <: Integer, UPLO}

    ns = zero(I); Mptr[one(I)] = one(I)

    for j in vertices(res)
        ns, localinfo = ldlt_loop!(Mptr, Mval, Dptr, Dval, Lptr, Lval, d, Fval, res, rel, chd, ns, j, uplo, reg)

        if isnegative(localinfo)
            return localinfo
        elseif ispositive(localinfo)
            return localinfo + pointers(res)[j] - one(I)
        end
    end

    return zero(I)
end

function ldlt_loop!(
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
        reg::Union{DynamicRegularization, Nothing},
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
    # L is part of the LDLᵀ factor
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
    if !isnothing(reg)
        regview = view(reg, neighbors(res, j))
    else
        regview = nothing
    end

    if nj <= THRESHOLD
        info = convert(I, ldlt_kernel!(D₁₁, L₂₁, M₂₂, d₁₁, uplo, Val(:N), regview))
    else
        info = convert(I, ldlt_kernel!(D₁₁, L₂₁, M₂₂, Fval, d₁₁, uplo, Val(:S), regview))
    end

    if ispositive(na) && ispositive(info)
        ns -= one(I)
    end

    return ns, info
end

function ldlt_kernel!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        S::AbstractMatrix{T},
        Wval::AbstractVector{T},
        d::AbstractVector{T},
        uplo::Val{UPLO},
        node::Val{:S},
        reg::Union{DynamicRegularization, Nothing},
    ) where {T, UPLO}
    @assert size(D, 1) == size(D, 2)
    @assert size(S, 1) == size(S, 2)

    if UPLO === :L
        @assert size(L) == (size(S, 1), size(D, 1))
    else
        @assert size(L) == (size(D, 1), size(S, 1))
    end
    #
    # factorize D
    #
    #     D, d ← ldlt(D)
    #
    W₁₁ = reshape(view(Wval, eachindex(D)), size(D))
    info = qdtrf!(uplo, W₁₁, D, d, reg)

    if iszero(info) && !isempty(S)
        #
        #     L ← L D⁻ᴴ
        #
        if UPLO === :L
            trsm!(Val(:R), Val(:L), Val(:C), Val(:U), one(T), D, L)
        else
            trsm!(Val(:L), Val(:U), Val(:C), Val(:U), one(T), D, L)
        end
        #
        #     W ← L
        #
        W₂₁ = reshape(view(Wval, eachindex(L)), size(L))
        copyrec!(W₂₁, L)
        #
        #     L ← L d⁻¹
        #
        if UPLO === :L
            @inbounds for k in axes(L, 2)
                idk = inv(d[k])

                for i in axes(L, 1)
                    L[i, k] *= idk
                end
            end
        else
            @inbounds for i in axes(L, 2)
                for k in axes(L, 1)
                    L[k, i] *= inv(d[k])
                end
            end
        end
        #
        #     S ← S - W Lᴴ
        #
        if UPLO === :L
            trrk!(uplo, Val(:N), W₂₁, L, S)
        else
            trrk!(uplo, Val(:C), W₂₁, L, S)
        end
    end

    return info
end

function ldlt_kernel!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        S::AbstractMatrix{T},
        d::AbstractVector{T},
        uplo::Val{:L},
        node::Val{:N},
        reg::Union{DynamicRegularization, Nothing},
    ) where {T}
    @assert size(D, 1) == size(D, 2)
    @assert size(S, 1) == size(S, 2)
    @assert size(L, 1) == size(S, 1)
    @assert size(L, 2) == size(D, 1)

    @inbounds for j in axes(D, 1)
        Djj = real(D[j, j])

        for k in 1:j - 1
            Djj -= abs2(D[j, k]) * d[k]
        end

        if !isnothing(reg)
            if reg.signs[j] * Djj < reg.epsilon
                Djj = reg.delta * reg.signs[j]
            end
        end

        d[j] = Djj
        iszero(Djj) && return j

        iDjj = inv(Djj)

        for i in j + 1:size(D, 1)
            Dij = D[i, j]

            for k in 1:j - 1
                Dij -= D[i, k] * d[k] * conj(D[j, k])
            end

            D[i, j] = Dij * iDjj
        end

        for i in axes(S, 1)
            Lij = L[i, j]

            for k in 1:j - 1
                Lij -= L[i, k] * d[k] * conj(D[j, k])
            end

            L[i, j] = Lij
        end

        for k in axes(S, 1)
            Ljk = L[k, j]; cLjk = conj(Ljk)

            S[k, k] -= iDjj * abs2(Ljk)

            for i in k + 1:size(S, 1)
                S[i, k] -= iDjj * L[i, j] * cLjk
            end
        end

        for i in axes(S, 1)
            L[i, j] *= iDjj
        end
    end

    return 0
end

function ldlt_kernel!(
        D::AbstractMatrix{T},
        U::AbstractMatrix{T},
        S::AbstractMatrix{T},
        d::AbstractVector{T},
        uplo::Val{:U},
        node::Val{:N},
        reg::Union{DynamicRegularization, Nothing},
    ) where {T}
    @assert size(D, 1) == size(D, 2)
    @assert size(S, 1) == size(S, 2)
    @assert size(U, 1) == size(D, 1)
    @assert size(U, 2) == size(S, 1)

    @inbounds for j in axes(D, 1)
        Djj = real(D[j, j])

        for k in 1:j - 1
            Djj -= abs2(D[k, j]) * d[k]
        end

        if !isnothing(reg)
            if reg.signs[j] * Djj < reg.epsilon
                Djj = reg.delta * reg.signs[j]
            end
        end

        d[j] = Djj
        iszero(Djj) && return j

        iDjj = inv(Djj)

        for i in j + 1:size(D, 1)
            Dji = D[j, i]

            for k in 1:j - 1
                Dji -= conj(D[k, i]) * d[k] * D[k, j]
            end

            D[j, i] = Dji * iDjj
        end

        for i in axes(S, 1)
            Uji = U[j, i]

            for k in 1:j - 1
                Uji -= conj(U[k, i]) * d[k] * D[k, j]
            end

            U[j, i] = Uji
        end

        for k in axes(S, 1)
            Ujk = U[j, k]; cUjk = conj(Ujk)

            S[k, k] -= iDjj * abs2(Ujk)

            for i in 1:k - 1
                S[i, k] -= iDjj * U[j, i] * cUjk
            end
        end

        for i in axes(S, 1)
            U[j, i] *= iDjj
        end
    end

    return 0
end
