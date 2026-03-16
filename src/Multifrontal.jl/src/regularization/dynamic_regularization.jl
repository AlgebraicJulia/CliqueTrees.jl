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

julia> F = cholesky!(ChordalCholesky(A); reg=DynamicRegularization());

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

function DynamicRegularization{T}(; delta=-one(T), epsilon=-one(T)) where {T}
    return DynamicRegularization{T}(delta, epsilon)
end

function DynamicRegularization(; kw...)
    return DynamicRegularization{Float64}(; kw...)
end

function initialize(::AbstractMatrix{T}, S::AbstractVector, R::DynamicRegularization) where {T}
    if isnegative(R.delta)
        delta = dynm_delta(real(T))
    else
        delta = convert(real(T), R.delta)
    end

    if isnegative(R.epsilon)
        epsilon = dynm_epsilon(real(T))
    else
        epsilon = convert(real(T), R.epsilon)
    end

    return DynamicRegularization{real(T)}(delta, epsilon)
end

function dynm_delta(::Type{T}) where {T}
    return cbrt(eps(real(T)))
end

function dynm_epsilon(::Type{T}) where {T}
    return sqrt(eps(real(T)))
end

function regularize(R::DynamicRegularization, S::AbstractVector, Djj, j::Integer)
    if real(S[j]) * Djj < R.epsilon
        return R.delta * real(S[j])
    else
        return Djj
    end
end
