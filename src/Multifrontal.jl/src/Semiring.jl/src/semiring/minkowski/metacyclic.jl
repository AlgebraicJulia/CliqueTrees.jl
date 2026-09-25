# Groups with a cyclic subgroup of index 2:
#
#   G = ⟨r, s | rⁿ = e, s r s⁻¹ = rᵗ, s² = rᶜ⟩
#
# Sets 2ᴳ are represented as follows:
#
#   - rᵃ  is bit 2a
#   - rᵃs is bit 2b + 1,  b = ta mod N
#
abstract type Metacyclic{N} <: MinkowskiQuantale end

# The dihedral group
#
#   Dₙ  = ⟨r, s | rⁿ = 1, s r s⁻¹ = r⁻¹, s² = 1⟩
#
#   - elements are symmetries of a regular n-gon
#   - multiplication is composition of symmetries
#
struct Dihedral{N} <: Metacyclic{N} end

# The dicyclic group
#
#   Q₂ₙ  = ⟨r, s | rⁿ = 1, s r s⁻¹ = r⁻¹, s² = rᵐ⟩,  m = n/2
#
# When n = 4, this is the quaternion group Q₈:
#
#   - elements are ±1, ±i, ±j, ±k
#   - multiplication is quaternion multiplication
#
struct Dicyclic{N} <: Metacyclic{N} end

# The semi-dihedral group
#
#   SD₂ₙ = ⟨r, s | rⁿ = 1, s r s⁻¹ = rᵐ⁻¹), s² = 1⟩,  m = n/2
#
struct Semidihedral{N} <: Metacyclic{N} end

# The modular maximal-cyclic group
#
#   M₂ₙ  = ⟨r, s | rⁿ = 1, s r s⁻¹ = rᵐ⁺¹, s² = 1⟩,  m = n/2
#
struct Modular{N} <: Metacyclic{N} end

# ----- semiring -----

@inline function sid(s::Dicyclic{N}, x::V, op::Val{:T}) where {N, V}
    T = eltype(x)
    c = Cyclic{2N}()
    e = V(szero(s, T, Val(:C)) & mmask(T, 2))
    return sid(c, x & e, op) | crot(c, sid(c, mconjs(s, crot(c, x & ~e, -1)), op), 1 - N)
end

@inline function sid(s::Union{Dihedral{N}, Semidihedral{N}, Modular{N}}, x::V, op::Val{:T}) where {N, V}
    T = eltype(x)
    c = Cyclic{2N}()
    e = V(szero(s, T, Val(:C)) & mmask(T, 2))
    return sid(c, x & e, op) | crot(c, sid(c, mconjs(s, crot(c, x & ~e, -1)), op), 1)
end

function szero(s::Metacyclic{N}, ::Type{T}, op::Val{:C}) where {N, T <: Unsigned}
    return szero(Cyclic{2N}(), T, op)
end

function sprod(s::Metacyclic{N}, a::T, b::T, ::Val{:N}, ::Val{:N}) where {N, T <: Unsigned}
    tab = gtab(gbase(s, a)...)
    return sprodrec(s, b, tab, Val(cld(2N, 4) - 1))
end

function sprodrec(s::Metacyclic, b, tab::Tuple, ::Val{0})
    i = gnib(b, 0)
    @inbounds return tab[i]
end

function sprodrec(s::Metacyclic{N}, b, tab::Tuple, ::Val{K}) where {N, K}
    i = gnib(b, K)
    @inbounds return crot(Cyclic{2N}(), tab[i], 4K) | sprodrec(s, b, tab, Val(K - 1))
end

function sstar(s::Dihedral{N}, a::T) where {N, T <: Unsigned}
    c = Cyclic{2N}()
    #
    # the set of rotations
    #
    #   R := {rᵃ | a}
    #
    r = szero(s, T, Val(:C)) & mmask(T, 2)
    #
    # the set of reflections
    #
    #   A - R
    #
    # lying in A
    #
    ars = a & ~r

    if iszero(ars)
        #
        # A - R is empty, so ⟨A⟩ is the subgroup it generates in ℤ/2N
        #
        b = sstar(c, a)
    else
        #
        # the set of rotations
        #
        #   A ∩ R
        #
        # lying in A
        #
        ar = a & r
        #
        # g is the least reflection in A
        #
        p = trailing_zeros(ars)
        #
        #   K := ⟨ (A ∩ R) ∪ (A - R)g ⟩
        #
        k = sstar(c, ar | crot(c, ars, -p))
        #
        #   ⟨A⟩ := K ∪ Kg
        #
        b = k | crot(c, k, p)
    end

    return b
end

function sstar(s::Union{Dicyclic{N}, Semidihedral{N}, Modular{N}}, a::T) where {N, T <: Unsigned}
    c = Cyclic{2N}()
    #
    # the set of rotations
    #
    #   R := {rᵃ | a}
    #
    r = szero(s, T, Val(:C)) & mmask(T, 2)
    #
    # the set of reflections
    #
    #   A - R
    #
    # lying in A
    #
    ars = a & ~r

    if iszero(ars)
        #
        # A - R is empty, so ⟨A⟩ is the subgroup it generates in ℤ/2N
        #
        b = sstar(c, a)
    else
        #
        # the set of rotations
        #
        #   A ∩ R
        #
        # lying in A
        #
        ar = a & r
        #
        # g is the least reflection in A
        #
        p = trailing_zeros(ars); g = one(T) << p
        #
        # g⁻¹ is the inverse of g
        #
        q = trailing_zeros(sid(s, g, Val(:T)))
        #
        #   K := ⟨ (A ∩ R) ∪ (A - R)g⁻¹ ∪ {g²} ⟩
        #
        k = sstar(c, ar | crot(c, mprods(s, ars), q - 1) | crot(c, mprods(s, g), p - 1))
        #
        #   ⟨A⟩ := K ∪ Kg
        #
        b = k | crot(c, mprods(s, k), p - 1)
    end

    return b
end

function iscommutative(::Type{<:Metacyclic})
    return false
end

# ----- minkowski -----

@inline function gbase(s::Metacyclic{N}, a) where {N}
    c = Cyclic{2N}()

    w = a
    x = mprods(s, a)
    y = crot(c, a, 2)
    z = crot(c, x, 2)

    return w, x, y, z
end

@inline function gshift(s::Metacyclic{N}, x) where {N}
    return crot(Cyclic{2N}(), x, 4)
end

@inline function gprod(s::Metacyclic{N}, T::NTuple{16, V}, U::NTuple{16, V}, b::Unsigned) where {N, V <: Vec}
    return gprod(Cyclic{2N}(), T, U, b)
end

# ----- helpers -----

function mmask(::Type{T}, p) where {T}
    return typemax(T) ÷ ((one(T) << p) - one(T))
end

@inline function mconjs(s::Union{Dihedral{N}, Dicyclic{N}}, y) where {N}
    return sid(Cyclic{2N}(), y, Val(:T))
end

@inline function mconjs(s::Semidihedral{N}, y::V) where {N, V}
    T = eltype(y)
    c = Cyclic{2N}()
    z = sid(c, y, Val(:T))
    h = V(mmask(T, 4) << 2)
    return (z & ~h) | crot(c, z & h, N)
end

@inline function mconjs(s::Modular{N}, y::V) where {N, V}
    T = eltype(y)
    c = Cyclic{2N}()
    h = V(mmask(T, 4) << 2)
    return (y & ~h) | crot(c, y & h, N)
end

@inline function mprods(s::Dihedral{N}, x) where {N}
    c = Cyclic{2N}()
    return crot(c, sid(c, x, Val(:T)), 1)
end

@inline function mprods(s::Dicyclic{N}, x::V) where {N, V}
    T = eltype(x)
    c = Cyclic{2N}()
    e = V(szero(s, T, Val(:C)) & mmask(T, 2))
    y = sid(c, x, Val(:T))
    return crot(c, (y & e) | crot(c, y & ~e, N), 1)
end

@inline function mprods(s::Union{Semidihedral{N}, Modular{N}}, x::V) where {N, V}
    T = eltype(x)
    c = Cyclic{2N}()
    e = V(szero(s, T, Val(:C)) & mmask(T, 2))
    return crot(c, mconjs(s, x & e), 1) | mconjs(s, crot(c, x & ~e, -1))
end
