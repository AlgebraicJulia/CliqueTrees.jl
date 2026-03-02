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

julia> F = cholesky!(ChordalCholesky(A); reg=SE99());

julia> B = Matrix(F)
3×3 Matrix{Float64}:
 3.0  1.0  2.0
 1.0  3.0  3.0
 2.0  3.0  3.375

julia> isposdef(B)
true
```

### Fields

- `gammapos`: maximum diagonal element for positive pivots
- `gammaneg`: maximum diagonal element for negative pivots
- `epsilon`: pivot tolerance
- `mu`: lookahead parameter
"""
struct SE99{T} <: AbstractRegularization
    gammapos::T
    gammaneg::T
    epsilon::T
    mu::T
end

function SE99{T}(; gammapos=-one(T), gammaneg=-one(T), epsilon=-one(T), mu=-one(T)) where {T}
    return SE99{T}(gammapos, gammaneg, epsilon, mu)
end

function SE99(; kw...)
    return SE99{Float64}(; kw...)
end

function initialize(F::ChordalFactorization{DIAG, UPLO, T}, S::AbstractVector, R::SE99) where {DIAG, UPLO, T}
    if isnegative(R.gammapos)
        gammapos = se99_gamma(ChordalTriangular(F), S, 1)
    else
        gammapos = convert(real(T), R.gammapos)
    end

    if isnegative(R.gammaneg)
        gammaneg = se99_gamma(ChordalTriangular(F), S, -1)
    else
        gammaneg = convert(real(T), R.gammaneg)
    end

    if isnegative(R.epsilon)
        epsilon = se99_epsilon(real(T))
    else
        epsilon = convert(real(T), R.epsilon)
    end

    if isnegative(R.mu)
        mu = se99_mu(real(T))
    else
        mu = convert(real(T), R.mu)
    end

    return SE99{real(T)}(gammapos, gammaneg, epsilon, mu)
end

function se99_gamma(A::ChordalTriangular{DIAG, UPLO, T}, S::AbstractVector, s::Real) where {DIAG, UPLO, T}
    init = eps(real(T))

    function maxabsdiag(D, res)
        out = init

        for (i, j) in zip(res, diagind(D))
            if real(S[i]) == s
                out = max(out, abs(parent(D)[j]))
            end
        end     

        return out
    end

    return mapreducefront((D, L, res, sep) -> maxabsdiag(D, res), max, A; init)
end

function se99_gamma(R::SE99, s::Real)
    if isone(s)
        return R.gammapos
    elseif isone(-s)
        return R.gammaneg
    else
        error()
    end
end

function se99_epsilon(::Type{T}) where {T}
    return cbrt(eps(real(T)))^2
end

function se99_mu(::Type{T}) where {T}
    return convert(real(T), 0.1)
end

function checkdiag(
        R::SE99,
        S₁::AbstractVector{T},
        S₂::AbstractVector{T},
        d₁::AbstractVector{T},
        d₂::AbstractVector{T},
        Djj::Real,
        j::Integer,
    ) where {T}
    gamma = se99_gamma(R, real(S₁[j]))

    if Djj < R.epsilon * gamma
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
        gamma = se99_gamma(R, Sii)

        if UPLO === :L
            ζii = real(d₁[i]) - abs2(L₁[i, j]) / Djj
        else
            ζii = real(d₁[i]) - abs2(L₁[j, i]) / Djj
        end

        if Sii * ζii < -R.mu * gamma
            return false
        end
    end

    for i in eachindex(d₂)
        Sii = real(S₂[i])
        gamma = se99_gamma(R, Sii)

        if UPLO === :L
            ζii = real(d₂[i]) - abs2(L₂[i, j]) / Djj
        else
            ζii = real(d₂[i]) - abs2(L₂[j, i]) / Djj
        end

        if Sii * ζii < -R.mu * gamma
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
        deltapos::Real,
        deltaneg::Real,
        ::Val{UPLO},
    ) where {T, UPLO}
    bound = zero(real(T))
    gamma = se99_gamma(R, real(S[j]))

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

    # Select delta based on sign class
    if real(S[j]) > 0
        delta = deltapos
    else
        delta = deltaneg
    end

    δj = max(zero(real(T)), -real(S[j]) * Djj + max(bound, R.epsilon * gamma), delta)

    if ispositive(δj)
        Djj = Djj + real(S[j]) * δj
    end

    # Update only the relevant delta
    if real(S[j]) > 0
        return Djj, δj, deltaneg
    else
        return Djj, deltapos, δj
    end
end
