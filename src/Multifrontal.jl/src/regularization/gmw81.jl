"""
    GMW81{T} <: AbstractRegularization

The Gill-Murray-Wright 1981 modified Cholesky algorithm.

### Example

```julia-repl
julia> using CliqueTrees.Multifrontal, LinearAlgebra

julia> A = [
           1 1 2
           1 1 3
           2 3 1
       ];

julia> isposdef(A)
false

julia> F = cholesky!(ChordalCholesky(A); reg=GMW81());

julia> B = Matrix(F)
3×3 Matrix{Float64}:
 3.77124  1.0      2.0
 1.0      6.01561  3.0
 2.0      3.0      3.24264

julia> isposdef(B)
true
```

### Fields

- `beta`: scaling parameter
- `delta`: minimum pivot
"""
struct GMW81{T} <: AbstractRegularization
    beta::T
    delta::T
end

function GMW81{T}(; beta=-one(T), delta=-one(T)) where {T}
    return GMW81{T}(beta, delta)
end

function GMW81(; kw...)
    return GMW81{Float64}(; kw...)
end

function initialize(F::ChordalFactorization{DIAG, UPLO, T}, S::AbstractVector, R::GMW81) where {DIAG, UPLO, T}
    if isnegative(R.beta)
        beta = gmw81_beta(ChordalTriangular(F))
    else
        beta = convert(real(T), R.beta)
    end

    if isnegative(R.delta)
        delta = gmw81_delta(real(T))
    else
        delta = convert(real(T), R.delta)
    end

    return GMW81{real(T)}(beta, delta)
end

function gmw81_beta(A::ChordalTriangular{DIAG, UPLO, T, I}) where {DIAG, UPLO, T, I}
    n = size(A, 1)
    gamma = zero(real(T))
    xi = zero(real(T))

    @inbounds for j in vertices(A.S.res)
        nn = eltypedegree(A.S.res, j)
        na = eltypedegree(A.S.sep, j)

        Dp = A.S.Dptr[j]
        Lp = A.S.Lptr[j]

        D = reshape(view(A.Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

        if UPLO === :L
            L = reshape(view(A.Lval, Lp:Lp + nn * na - one(I)), na, nn)
        else
            L = reshape(view(A.Lval, Lp:Lp + nn * na - one(I)), nn, na)
        end

        for i in oneto(nn)
            gamma = max(gamma, abs(D[i, i]))
        end

        if UPLO === :L
            for col in oneto(nn)
                for row in col + one(I):nn
                    xi = max(xi, abs(D[row, col]))
                end
            end
        else
            for col in oneto(nn)
                for row in oneto(col - one(I))
                    xi = max(xi, abs(D[row, col]))
                end
            end
        end

        for v in L
            xi = max(xi, abs(v))
        end
    end

    return max(gamma, xi / sqrt(n^2 - 1), eps(real(T)))
end

function gmw81_delta(::Type{T}) where {T}
    return cbrt(eps(real(T)))
end

function regularize(
        R::GMW81,
        S::AbstractVector,
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        Djj::T,
        j::Integer,
        uplo::Val{UPLO},
    ) where {T, UPLO}
    Aimax = zero(real(T))

    if UPLO === :L
        for i in j + 1:size(D, 1)
            Aimax = max(Aimax, abs(D[i, j]))
        end

        for i in axes(L, 1)
            Aimax = max(Aimax, abs(L[i, j]))
        end
    else
        for i in j + 1:size(D, 2)
            Aimax = max(Aimax, abs(D[j, i]))
        end

        for i in axes(L, 2)
            Aimax = max(Aimax, abs(L[j, i]))
        end
    end

    return real(S[j]) * max(abs(Djj), Aimax^2 / R.beta, R.delta)
end
