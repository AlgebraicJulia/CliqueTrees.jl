struct SemiringLU{
        Sem <: AbstractSemiring,
        T,
        I,
        LDvl <: AbstractVector{T},
        LLvl <: AbstractVector{T},
        UDvl <: AbstractVector{T},
        ULvl <: AbstractVector{T},
        RPrm <: AbstractVector{I},
        RIvp <: AbstractVector{I},
        CPrm <: AbstractVector{I},
        CIvp <: AbstractVector{I},
    } <: Factorization{T}
    s::Sem
    S::ChordalSymbolic{I}
    LDval::LDvl
    LLval::LLvl
    UDval::UDvl
    ULval::ULvl
    rperm::RPrm
    rinvp::RIvp
    cperm::CPrm
    cinvp::CIvp
end

const FSemiringLU{Sem, T, I} = SemiringLU{
    Sem,
    T,
    I,
    FVector{T},
    FVector{T},
    FVector{T},
    FVector{T},
    FVector{I},
    FVector{I},
    FVector{I},
    FVector{I},
}

const DSemiringLU{Sem, T, I} = SemiringLU{
    Sem,
    T,
    I,
    Vector{T},
    Vector{T},
    Vector{T},
    Vector{T},
    Vector{I},
    Vector{I},
    Vector{I},
    Vector{I},
}

function SemiringLU(s::AbstractSemiring, A::SparseMatrixCSC{T}) where {T}
    P, S = symbolic(symmetric(A, 'N'))
    return SemiringLU(s, T, S, P.perm, P.invp, P.perm, P.invp)
end

function SemiringLU(s::AbstractSemiring, ::Type{T}, S::ChordalSymbolic{I}, rperm, rinvp, cperm, cinvp) where {T, I}
    L = FChordalTriangular{:N, :L, T, I}(S)
    U = FChordalTriangular{:N, :U, T, I}(S)
    return SemiringLU(s, S, L.Dval, L.Lval, U.Dval, U.Lval, rperm, rinvp, cperm, cinvp)
end

function SemiringLU{Sem}(F::SemiringLU) where {Sem}
    return SemiringLU(Sem(), F.S, F.LDval, F.LLval, F.UDval, F.ULval, F.rperm, F.rinvp, F.cperm, F.cinvp)
end

function lowertriangular(F::SemiringLU)
    return ChordalTriangular{:N, :L}(F.S, F.LDval, F.LLval)
end

function uppertriangular(F::SemiringLU)
    return ChordalTriangular{:N, :U}(F.S, F.UDval, F.ULval)
end

function Base.size(F::SemiringLU)
    return size(F.S)
end

function Base.size(F::SemiringLU, d::Integer)
    return size(F.S, d)
end

function Base.getproperty(F::SemiringLU, name::Symbol)
    if name === :L
        return lowertriangular(F)
    elseif name === :U
        return uppertriangular(F)
    elseif name === :P
        return Permutation(getfield(F, :rperm), getfield(F, :rinvp))
    elseif name === :Q
        return Permutation(getfield(F, :cperm), getfield(F, :cinvp))
    else
        return getfield(F, name)
    end
end

function Base.copyto!(F::SemiringLU, A::SparseMatrixCSC)
    A = permute(A, F.rperm, F.cperm)
    scopyto!(F.s, F.L, A)
    scopyto!(F.s, F.U, A)
    return F
end

function scopyto!(s::AbstractSemiring, A::ChordalTriangular{<:Any, <:Any, T}, B::SparseMatrixCSC) where {T}
    fill!(A, szero(s, T))
    return copy_scatter!(A, B)
end
