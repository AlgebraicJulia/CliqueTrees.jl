"""
    DynamicRegularization
"""
struct DynamicRegularization{T, I, Sgn <: AbstractVector{I}}
    signs::Sgn
    delta::T
    epsilon::T
end

const DynReg = DynamicRegularization

function DynReg{T, I}(n::Integer, delta::Number, epsilon::Number) where {T, I}
    signs = FVector{I}(undef, n)
    return DynReg{T, I, FVector{I}}(signs, delta, epsilon)
end

function DynReg(signs::Sgn; delta::T=1e-6, epsilon::T=1e-12) where {T, I, Sgn <: AbstractVector{I}}
    return DynReg{T, I, Sgn}(signs, delta, epsilon)
end

function Base.view(reg::DynReg, inds)
    return DynReg(view(reg.signs, inds), reg.delta, reg.epsilon)
end
