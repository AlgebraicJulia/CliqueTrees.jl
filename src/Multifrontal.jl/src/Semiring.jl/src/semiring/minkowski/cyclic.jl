# The cyclic group ℤₙ.
#
#   - elements are integers
#   - addition is addition-modulo-n
#
struct Cyclic{N} <: MinkowskiQuantale end

# ----- semiring -----

@inline function sid(s::Cyclic{N}, a::T, ::Val{:T}) where {N, T}
    return crot(s, bitreverse(a) >> (8sizeof(eltype(T)) - N), 1)
end

function szero(s::Cyclic{N}, ::Type{T}, ::Val{:C}) where {N, T <: Unsigned}
    if N == 8sizeof(T)
        a = typemax(T)
    else
        a = (one(T) << N) - one(T)
    end

    return a
end

function sprod(s::Cyclic{N}, a::T, b::T, ::Val{:N}, ::Val{:N}) where {N, T <: Unsigned}
    tab = gtab(gbase(s, b)...)
    return sprodrec(s, a, tab, Val(cld(N, 4) - 1))
end

function sprodrec(s::Cyclic, a, tab::Tuple, ::Val{0})
    i = gnib(a, 0)
    @inbounds return tab[i]
end

function sprodrec(s::Cyclic, a, tab::Tuple, ::Val{K}) where {K}
    i = gnib(a, K)
    @inbounds return crot(s, tab[i], 4K) | sprodrec(s, a, tab, Val(K - 1))
end

function sstar(s::Cyclic{N}, a::T) where {N, T <: Unsigned}
    ms..., n = cmasks(s, T)

    for m in ms
        iszero(a & ~m) && return m
    end

    return n
end

# ----- minkowski -----

@inline function gbase(s::Cyclic, a)
    w = a
    x = crot(s, a, 1)
    y = crot(s, a, 2)
    z = crot(s, a, 3)

    return w, x, y, z
end

@inline function gshift(s::Cyclic, x)
    return crot(s, x, 4)
end

@inline function gprod(s::Cyclic{N}, T::NTuple{16, V}, U::NTuple{16, V}, b::E) where {N, V <: Vec, E <: Unsigned}
    return gprodrec(s, T, U, b, Val(cld(N, 8) - 1))
end

@inline function gprodrec(s::Cyclic, T::Tuple, U::Tuple, b, ::Val{0})
    return gbyte(T, U, b, 0)
end

@inline function gprodrec(s::Cyclic, T::Tuple, U::Tuple, b, ::Val{K}) where {K}
    return crot(s, gbyte(T, U, b, K), 8K) | gprodrec(s, T, U, b, Val(K - 1))
end

# ----- helpers -----

@generated function cmasks(::Cyclic{N}, ::Type{T}) where {N, T <: Unsigned}
    masks = T[]

    for d in N:-1:1
        if N % d == 0
            m = zero(T)

            for i in 0:d:N - 1
                m |= one(T) << i
            end

            push!(masks, m)
        end
    end

    return :($(Tuple(masks)))
end

@inline function crot(s::Cyclic{N}, x::T, k::Int) where {N, T}
    if N == 8sizeof(eltype(T))
        y = bitrotate(x, k)
    else
        k = mod(k, N)
        y = ((x << k) | (x >> (N - k))) & szero(s, T, Val(:C))
    end

    return y
end
