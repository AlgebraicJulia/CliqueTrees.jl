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
    # factorize D₁₁ (reuse Fval as workspace since F₁₁, F₂₁ are no longer needed)
    #
    #     D₁₁, d₁₁ ← ldlt(D₁₁)
    #
    W₁₁ = reshape(view(Fval, oneto(nn * nn)), nn, nn)

    if !isnothing(reg)
        info = convert(I, qdtrf!(uplo, W₁₁, D₁₁, d₁₁, view(reg, neighbors(res, j))))
    else
        info = convert(I, qdtrf!(uplo, W₁₁, D₁₁, d₁₁, nothing))
    end

    if iszero(info) && ispositive(na)
        ns += one(I)
        #
        # M₂₂ is the update matrix for node j
        #
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * na
        M₂₂ = reshape(view(Mval, strt:stop - one(I)), na, na)
        #
        #     M₂₂ ← F₂₂
        #
        copytri!(M₂₂, F₂₂, uplo)
        #
        #     L₂₁ ← L₂₁ L₁₁⁻ᴴ  (unit triangular solve)
        #
        if UPLO === :L
            trsm!(Val(:R), Val(:L), Val(:C), Val(:U), one(T), D₁₁, L₂₁)
        else
            trsm!(Val(:L), Val(:U), Val(:C), Val(:U), one(T), D₁₁, L₂₁)
        end
        #
        #     W₂₁ ← L₂₁  (reuse Fval since F is no longer needed)
        #
        if UPLO === :L
            W₂₁ = reshape(view(Fval, oneto(nn * na)), na, nn)
        else
            W₂₁ = reshape(view(Fval, oneto(nn * na)), nn, na)
        end

        copyrec!(W₂₁, L₂₁)
        #
        #     L₂₁ ← L₂₁ d₁₁⁻¹
        #
        if UPLO === :L
            @inbounds for k in axes(L₂₁, 2)
                invD = inv(d₁₁[k])

                for i in axes(L₂₁, 1)
                    L₂₁[i, k] *= invD
                end
            end
        else
            @inbounds for i in axes(L₂₁, 2)
                for k in axes(L₂₁, 1)
                    L₂₁[k, i] *= inv(d₁₁[k])
                end
            end
        end
        #
        #     M₂₂ ← M₂₂ - W₂₁ L₂₁ᴴ  (lower)
        #     M₂₂ ← M₂₂ - W₂₁ᴴ L₂₁  (upper)
        #
        if UPLO === :L
            trrk!(uplo, Val(:N), W₂₁, L₂₁, M₂₂)
        else
            trrk!(uplo, Val(:C), W₂₁, L₂₁, M₂₂)
        end
    end

    return ns, info
end
