"""
    AbstractRegularization

A dynamic regularization strategy. It can be passed to [`cholesky!`](@ref)
or [`ldlt!`](@ref) in order to compute a *modified* Cholesky (or LDLt) factorization.

### Modified Factorizations

If a matrix ``A`` is ill-conditioned (or even indefinite!), it is sometimes helpful
to perturb it by a small matrix ``E`` before computing a Cholesky factorization.

```math
    P (A + E) P^\\mathsf{T} = L L^\\mathsf{T}.
```

The diagonal matrix ``E`` can be chosen statically (before the factorization) dynamically
(during the factorization). Each `AbstractRegularization` object is an algorithm for
finding ``E``.

"""
abstract type AbstractRegularization end

struct NoRegularization <: AbstractRegularization end

"""
    DynamicRegularization{T} <: AbstractRegularization

The simplest dynamic regularization strategy. Small pivots are replaced with
a fixed number ``\\delta``.

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

julia> F = ChordalCholesky(A);

julia> cholesky!(F; reg=DynamicRegularization(F));

julia> B = Matrix(F)
3×3 Matrix{Float64}:
 1.0  1.0      2.0
 1.0  1.00001  3.0
 2.0  3.0      1.65144e5

julia> isposdef(B)
true
```

### Fields

- `delta`: minimum pivot
- `epsilon`: pivot tolerance
"""
struct DynamicRegularization{T} <: AbstractRegularization
    delta::T
    epsilon::T
end

function DynamicRegularization{T}(; delta=dynm_delta(T), epsilon=dynm_epsilon(T)) where {T}
    return DynamicRegularization{T}(delta, epsilon)
end

function DynamicRegularization(; kw...)
    return DynamicRegularization{Float64}(; kw...)
end

function DynamicRegularization(F::ChordalFactorization{DIAG, UPLO, T}; kw...) where {DIAG, UPLO, T}
    return DynamicRegularization{real(T)}(; kw...)
end

function dynm_delta(::Type{T}) where {T}
    return cbrt(eps(real(T)))
end

function dynm_epsilon(::Type{T}) where {T}
    return sqrt(eps(real(T)))
end

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

julia> F = ChordalCholesky(A);

julia> cholesky!(F; reg=GMW81(F));

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

function GMW81{T}(; beta, delta=gmw81_delta(T)) where {T}
    return GMW81{T}(beta, delta)
end

function GMW81(; kw...)
    return GMW81{Float64}(; kw...)
end

function GMW81(F::ChordalFactorization{DIAG, UPLO, T}; beta=gmw81_beta(ChordalTriangular(F)), kw...) where {DIAG, UPLO, T}
    return GMW81{real(T)}(; beta, kw...)
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

"""
    SE99{T} <: AbstractRegularization

The Schnabel-Eskow 1999 modified Cholesky algorithm.

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

julia> F = ChordalCholesky(A);

julia> cholesky!(F; reg=SE99(F));

julia> B = Matrix(F)
3×3 Matrix{Float64}:
 3.0  1.0  2.0
 1.0  3.0  3.0
 2.0  3.0  3.375

julia> isposdef(B)
true
```

### Fields

- `gamma`: maximum diagonal element
- `delta`: minimum pivot
- `epsilon`: pivot tolerance
- `mu`: lookahead parameter
"""
struct SE99{T} <: AbstractRegularization
    gamma::T
    delta::T
    epsilon::T
    mu::T
end

function SE99{T}(; gamma, delta=se99_delta(T), epsilon=se99_epsilon(T), mu=se99_mu(T)) where {T}
    return SE99{T}(gamma, delta, epsilon, mu)
end

function SE99(; kw...)
    return SE99{Float64}(; kw...)
end


function SE99(F::ChordalFactorization{DIAG, UPLO, T}; gamma=se99_gamma(ChordalTriangular(F)), kw...) where {DIAG, UPLO, T}
    return SE99{real(T)}(; gamma, kw...)
end

function se99_gamma(A::ChordalTriangular{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    return mapreducefront(max, A; init=zero(real(T))) do D, L
        out = zero(real(T))

        for i in axes(D, 1)
            out = max(out, abs(parent(D)[i, i]))
        end

        return out
    end
end

function se99_delta(::Type{T}) where {T}
    return cbrt(eps(real(T)))
end

function se99_epsilon(::Type{T}) where {T}
    return cbrt(eps(real(T)))^2
end

function se99_mu(::Type{T}) where {T}
    return convert(real(T), 0.1)
end

# ===== checksigns =====

function checksigns(S::AbstractVector, ::NoRegularization)
    for s in S
        if !iszero(s) && !isone(s) && !isone(-s)
            return false
        end
    end

    return true
end

function checksigns(S::AbstractVector, ::AbstractRegularization)
    for s in S
        if !isone(s) && !isone(-s)
            return false
        end
    end

    return true
end

# ===== regularize functions =====

function regularize(::NoRegularization, S::AbstractVector, Djj, j::Integer)
    if !iszero(S[j]) && !ispositive(real(S[j]) * Djj)
        return zero(Djj)
    else
        return Djj
    end
end

function regularize(R::DynamicRegularization, S::AbstractVector, Djj, j::Integer)
    if real(S[j]) * Djj < R.epsilon
        return R.delta * real(S[j])
    else
        return Djj
    end
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

# ===== lookahead functions =====

function checkdiag(
        R::SE99,
        S₁::AbstractVector{T},
        S₂::AbstractVector{T},
        d₁::AbstractVector{T},
        d₂::AbstractVector{T},
        Djj::Real,
        j::Integer,
    ) where {T}
    if Djj < R.epsilon * R.gamma
        return false
    end

    minval = Djj

    for i in j:length(d₁)
        minval = min(minval, real(S₁[i]) * real(d₁[i]))
    end

    for i in eachindex(d₂)
        minval = min(minval, real(S₂[i]) * real(d₂[i]))
    end

    return minval >= -R.mu * Djj
end

function lookahead(
        R::SE99,
        S₁::AbstractVector{T},
        S₂::AbstractVector{T},
        L₁::AbstractMatrix{T},
        L₂::AbstractMatrix{T},
        d₁::AbstractVector{T},
        d₂::AbstractVector{T},
        j::Integer,
        ::Val{UPLO},
    ) where {T, UPLO}

    Djj = real(d₁[j])

    for i in j + 1:length(d₁)
        Sii = real(S₁[i])

        if UPLO === :L
            ζii = real(d₁[i]) - abs2(L₁[i, j]) / Djj
        else
            ζii = real(d₁[i]) - abs2(L₁[j, i]) / Djj
        end

        if Sii * ζii < -R.mu * R.gamma
            return false
        end
    end

    for i in eachindex(d₂)
        Sii = real(S₂[i])

        if UPLO === :L
            ζii = real(d₂[i]) - abs2(L₂[i, j]) / Djj
        else
            ζii = real(d₂[i]) - abs2(L₂[j, i]) / Djj
        end

        if Sii * ζii < -R.mu * R.gamma
            return false
        end
    end

    return true
end

function regularize(
        R::SE99,
        S::AbstractVector,
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        Djj::Real,
        j::Integer,
        delta::Real,
        ::Val{UPLO},
    ) where {T, UPLO}
    bound = zero(real(T))

    if UPLO === :L
        for i in j + 1:size(D, 1)
            bound += abs(D[i, j])
        end

        for i in axes(L, 1)
            bound += abs(L[i, j])
        end
    else
        for i in j + 1:size(D, 1)
            bound += abs(D[j, i])
        end

        for i in axes(L, 2)
            bound += abs(L[j, i])
        end
    end

    δj = max(zero(real(T)), -real(S[j]) * Djj + max(bound, R.epsilon * R.gamma), delta)

    if ispositive(δj)
        Djj = Djj + real(S[j]) * δj
        delta = δj
    end

    return Djj, delta
end
