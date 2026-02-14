"""
    Permutation{I} <: AbstractMatrix{Bool}

A permutation matrix.

### Fields

  - `P.perm`: permutation vector

"""
struct Permutation{I, Prm <: AbstractVector{I}} <: AbstractMatrix{Bool}
    perm::Prm
end

function Permutation{I, Prm}(n::Integer) where {I, Prm <: AbstractVector{I}}
    perm = Prm(undef, n)
    return Permutation(perm)
end

function Permutation{I}(n::Integer) where {I}
    return FPermutation{I}(n)
end

const FPermutation{I} = Permutation{I, FVector{I}}

const AdjOrTransPerm{I, Prm} = Union{
      Adjoint{Bool, Permutation{I, Prm}},
    Transpose{Bool, Permutation{I, Prm}},
}

const MaybeAdjOrTransPerm{I, Prm} = Union{
       Permutation{I, Prm},
    AdjOrTransPerm{I, Prm},
}

# ===== Abstract Matrix Interface =====

function Base.size(P::Permutation)
    n = length(P.perm)
    return (n, n)
end

function Base.getindex(P::Permutation, i::Integer, j::Integer)
    @boundscheck checkbounds(P, i, j)
    return P.perm[i] == j
end
