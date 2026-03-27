"""
    Permutation{I} <: AbstractMatrix{Bool}

A permutation matrix.

### Fields

  - `P.perm`: permutation vector
  - `P.invp`: inverse permutation vector

"""
struct Permutation{I, Prm <: AbstractVector{I}, Ivp <: AbstractVector{I}} <: AbstractMatrix{Bool}
    perm::Prm
    invp::Ivp

    function Permutation{I, Prm, Ivp}(perm, invp) where {I, Prm <: AbstractVector{I}, Ivp <: AbstractVector{I}}
        @assert length(perm) == length(invp)

        if Prm <: FVector && !(perm isa Prm)
            perm = Prm(perm)
        end

        if Ivp <: FVector && !(invp isa Ivp)
            invp = Ivp(invp)
        end

        return new{I, Prm, Ivp}(perm, invp)
    end
end

const FPermutation{I} = Permutation{I, FVector{I}, FVector{I}}
const DPermutation{I} = Permutation{I, Vector{I}, Vector{I}}
const NaturalPermutation{I} = Permutation{I, OneTo{I}, OneTo{I}}

function Permutation(perm::Prm, invp::Ivp) where {I, Prm <: AbstractVector{I}, Ivp <: AbstractVector{I}}
    return Permutation{I, Prm, Ivp}(perm, invp)
end

function Permutation{I, Prm, Ivp}(n::Integer) where {I, Prm <: AbstractVector{I}, Ivp <: AbstractVector{I}}
    perm = allocate(Prm, n)
    invp = allocate(Ivp, n)
    return Permutation{I, Prm, Ivp}(perm, invp)
end

function Permutation{I}(n::Integer) where {I}
    return FPermutation{I}(n)
end

function (::Type{Prm})(perm::AbstractVector{I}) where {I, Prm <: Permutation}
    return Prm{I}(perm)
end

function (::Type{Prm})(perm::AbstractVector) where {I, Prm <: Permutation{I}}
    return Prm(perm, invperm(perm))
end

# ===== show =====

for Prm in (:FPermutation, :DPermutation, :NaturalPermutation)
    @eval function Base.show(io::IO, ::Type{$Prm{I}}) where {I}
        print(io, $("$Prm{"), I, "}")
    end
end

# ===== Adjoint, Transpose, Inverse =====

function Base.inv(P::Permutation)
    return Permutation(P.invp, P.perm)
end

function Base.adjoint(P::Permutation)
    return inv(P)
end

function Base.transpose(P::Permutation)
    return inv(P)
end

# ===== Abstract Matrix Interface =====

function Base.size(P::Permutation)
    n = length(P.perm)
    return (n, n)
end

function Base.getindex(P::Permutation, i::Integer, j::Integer)
    @boundscheck checkbounds(P, i, j)
    return P.perm[i] == j
end
