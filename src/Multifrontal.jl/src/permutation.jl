"""
    Permutation{I} <: AbstractMatrix{Bool}

A permutation matrix.

### Fields

  - `P.perm`: permutation vector
  - `P.invp`: inverse permutation vector

"""
struct Permutation{I, Prm <: AbstractVector{I}} <: AbstractMatrix{Bool}
    perm::Prm
    invp::Prm
end

function Permutation{I, Prm}(n::Integer) where {I, Prm <: AbstractVector{I}}
    perm = Prm(undef, n)
    invp = Prm(undef, n)
    return Permutation(perm, invp)
end

function Permutation{I}(n::Integer) where {I}
    return FPermutation{I}(n)
end

const FPermutation{I} = Permutation{I, FVector{I}}

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
