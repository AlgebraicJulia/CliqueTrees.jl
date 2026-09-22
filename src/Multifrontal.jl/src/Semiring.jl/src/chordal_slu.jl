struct ChordalSLU{
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
    } <: AbstractSLU{T}
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

const FChordalSLU{Sem, T, I} = ChordalSLU{
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

const DChordalSLU{Sem, T, I} = ChordalSLU{
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

function ChordalSLU(s::AbstractSemiring, A::SparseMatrixCSC{T}) where {T}
    P, S = symbolic(symmetric(A, 'N'))
    return ChordalSLU(s, T, S, P.perm, P.invp, P.perm, P.invp)
end

function ChordalSLU(s::AbstractSemiring, ::Type{T}, S::ChordalSymbolic{I}, rperm, rinvp, cperm, cinvp) where {T, I}
    L = FChordalTriangular{:N, :L, T, I}(S)
    U = FChordalTriangular{:N, :U, T, I}(S)
    return ChordalSLU(s, S, L.Dval, L.Lval, U.Dval, U.Lval, rperm, rinvp, cperm, cinvp)
end

function ChordalSLU{Sem}(F::ChordalSLU) where {Sem}
    return ChordalSLU(Sem(), F.S, F.LDval, F.LLval, F.UDval, F.ULval, F.rperm, F.rinvp, F.cperm, F.cinvp)
end

function lowertriangular(F::ChordalSLU)
    return ChordalTriangular{:N, :L}(F.S, F.LDval, F.LLval)
end

function uppertriangular(F::ChordalSLU)
    return ChordalTriangular{:N, :U}(F.S, F.UDval, F.ULval)
end

function Base.size(F::ChordalSLU)
    return size(F.S)
end

function Base.size(F::ChordalSLU, d::Integer)
    return size(F.S, d)
end

function Base.getproperty(F::ChordalSLU, name::Symbol)
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

function Base.copyto!(F::ChordalSLU, A::SparseMatrixCSC)
    A = permute(A, F.rperm, F.cperm)
    scopyto!(F.s, F.L, A)
    scopyto!(F.s, F.U, A)
    return F
end

function scopyto!(s::AbstractSemiring, A::ChordalTriangular{<:Any, <:Any, T}, B::SparseMatrixCSC) where {T}
    fill!(A, szero(s, T, Val(:N)))
    return copy_scatter!(A, B)
end

# ===== sgetrf! =====

function sgetrf!(F::ChordalSLU)
    sgetrf!(F.s, F.L, F.U)
    return F
end

# ===== sgetrs! =====

function sgetrs!(F::ChordalSLU{<:Any, T}, side::Val{SIDE}, trans::Val{TRANS}, B::AbstractVecOrMat; nt::Integer = nthreads()) where {T, SIDE, TRANS}
    C = FArray{T}(undef, size(B))

    if SIDE === :L
        if TRANS === :T
            mul!(C, F.Q, B)
        else
            mul!(C, F.P, B)
        end
    else
        if TRANS === :T
            rdiv!(C, B, F.P)
        else
            rdiv!(C, B, F.Q)
        end
    end

    sgetrs!(F.s, side, trans, F.L, F.U, C; nt)

    if SIDE === :L
        if TRANS === :T
            ldiv!(B, F.P, C)
        else
            ldiv!(B, F.Q, C)
        end
    else
        if TRANS === :T
            mul!(B, C, F.Q)
        else
            mul!(B, C, F.P)
        end
    end

    return B
end
