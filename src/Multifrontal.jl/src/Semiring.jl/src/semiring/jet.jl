struct Jet{N, S <: AbstractSemiring} <: AbstractSemiring
    s::S
end

function Jet{N}(s::S) where {N, S <: AbstractSemiring}
    return Jet{N, S}(s)
end

function slte(s::Jet{N}, a, b) where {N}
    return jetlte(s.s, a, b)
end

function jetlte(s::AbstractSemiring, (a,)::NTuple{1}, (b,)::NTuple{1})
    return slte(s, a, b)
end

function jetlte(s::AbstractSemiring, a::NTuple{N}, b::NTuple{N}) where {N}
    a0..., an = a
    b0..., bn = b
    return jetlte(s, a0, b0) && slte(s, an, bn)
end

function stop(s::Jet{N}, ::Type{NTuple{N, V}}) where {N, V}
    return jettop(s.s, Val(N), V)
end

function jettop(s::AbstractSemiring, ::Val{1}, ::Type{V}) where {V}
    return (stop(s, V),)
end

function jettop(s::AbstractSemiring, ::Val{N}, ::Type{V}) where {N, V}
    return (jettop(s, Val(N - 1), V)..., stop(s, V))
end

function szero(s::Jet{N}, ::Type{NTuple{N, V}}) where {N, V}
    return jetzero(s.s, Val(N), V)
end

function jetzero(s::AbstractSemiring, ::Val{1}, ::Type{V}) where {V}
    return (szero(s, V),)
end

function jetzero(s::AbstractSemiring, ::Val{N}, ::Type{V}) where {N, V}
    return (jetzero(s, Val(N - 1), V)..., szero(s, V))
end

function sone(s::Jet{N}, ::Type{NTuple{N, V}}) where {N, V}
    return jetone(s.s, Val(N), V)
end

function jetone(s::AbstractSemiring, ::Val{1}, ::Type{V}) where {V}
    return (sone(s, V),)
end

function jetone(s::AbstractSemiring, ::Val{N}, ::Type{V}) where {N, V}
    return (jetone(s, Val(N - 1), V)..., szero(s, V))
end

function splus(s::Jet{N}, a, b) where {N}
    return jetplus(s.s, a, b)
end

function jetplus(s::AbstractSemiring, (a,)::NTuple{1}, (b,)::NTuple{1})
    return (splus(s, a, b),)
end

function jetplus(s::AbstractSemiring, a::NTuple{N}, b::NTuple{N}) where {N}
    a0..., an = a
    b0..., bn = b
    return (jetplus(s, a0, b0)..., splus(s, an, bn))
end

function sprod(s::Jet{N}, a, b) where {N}
    return jetprod(s.s, a, b)
end

function jetprod(s::AbstractSemiring, (a,)::NTuple{1}, (b,)::NTuple{1})
    return (sprod(s, a, b),)
end

@inline function jetprod(s::AbstractSemiring, a::NTuple{N}, b::NTuple{N}) where {N}
    T = promote_eltype(a, b)

    a0..., an = a
    b0..., bn = b

    c0 = jetprod(s, a0, b0)
    cn = szero(s, T)

    @inbounds for i in 1:N
        ai = a[i]
        bi = b[N - i + 1]
        cn = smuladd(s, ai, bi, cn)
    end

    return (c0..., cn)
end

function sstar(s::Jet{N}, a) where {N}
    return jetstar(s.s, a)
end

function jetstar(s::AbstractSemiring, (a,)::NTuple{1})
    return (sstar(s, a),)
end

function jetstar(s::AbstractSemiring, a::NTuple{N}) where {N}
    T = eltype(a)

    a0..., an = a

    b0 = jetstar(s, a0)
    bn = szero(s, T)

    @inbounds for i in 1:N - 1
        ai = a[i + 1]
        bi = b0[N - i]
        bn = smuladd(s, ai, bi, bn)
    end

    bn = sprod(s, b0[1], bn)

    return (b0..., bn)
end
