"""
    DynamicRegularization
"""
struct DynamicRegularization{T, I, Sgn <: AbstractVector{I}}
    signs::Sgn
    delta::T
    epsilon::T
end

function DynamicRegularization{T, I}(n::Integer, delta::Number, epsilon::Number) where {T, I}
    signs = FVector{I}(undef, n)
    return DynamicRegularization{T, I, FVector{I}}(signs, delta, epsilon)
end

function DynamicRegularization(signs::Sgn; delta::T=1e-6, epsilon::T=1e-12) where {T, I, Sgn <: AbstractVector{I}}
    return DynamicRegularization{T, I, Sgn}(signs, delta, epsilon)
end

function Base.view(reg::DynamicRegularization, inds)
    return DynamicRegularization(view(reg.signs, inds), reg.delta, reg.epsilon)
end
