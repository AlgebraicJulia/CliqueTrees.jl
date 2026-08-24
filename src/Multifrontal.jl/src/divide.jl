# ===== DivisionWorkspace =====

abstract type AbstractDivisionWorkspace{T} end

struct DivisionWorkspace{T} <: AbstractDivisionWorkspace{T}
    Mval::FVector{T}
end

struct DenseDivisionWorkspace{T} <: AbstractDivisionWorkspace{T} end

function DivisionWorkspace{T}(S::ChordalSymbolic, nrhs::Integer) where {T}
    Mval = FVector{T}(undef, S.nFval * nrhs)
    return DivisionWorkspace{T}(Mval)
end

function AbstractDivisionWorkspace{T}(L::ChordalTriangular, nrhs::Integer) where {T}
    return DivisionWorkspace{T}(L.S, nrhs)
end

function AbstractDivisionWorkspace{T}(A::AbstractTriangular, nrhs::Integer) where {T}
    return DenseDivisionWorkspace{T}()
end

function AbstractDivisionWorkspace{T}(A::AdjOrTrans, nrhs::Integer) where {T}
    return AbstractDivisionWorkspace{T}(parent(A), nrhs)
end

function AbstractDivisionWorkspace{T}(F::AbstractFactorization, nrhs::Integer) where {T}
    return AbstractDivisionWorkspace{T}(triangular(F), nrhs)
end

# ================================== \ ==================================

# --- Permutation ---

function Base.:\(A::Permutation, B::AbstractVecOrMat)
    return inv(A) * B
end


# --- ChordalTriangular ---

function Base.:\(A::MaybeAdjOrTransTri, B::AbstractVecOrMat)
    T = promote_eltype(A, B)
    return ldiv!(A, copyto!(similar(B, T), B))
end

# ================================== / ==================================

# --- Permutation ---

function Base.:/(A::AbstractMatrix, B::Permutation)
    return A * inv(B)
end

function Base.:/(A::TransVec, B::Permutation)
    return transpose(transpose(B) \ parent(A))
end

function Base.:/(A::AdjVec, B::Permutation)
    return adjoint(adjoint(B) \ parent(A))
end


# --- ChordalTriangular ---

function Base.:/(B::AbstractMatrix, A::MaybeAdjOrTransTri)
    T = promote_eltype(A, B)
    return rdiv!(copyto!(similar(B, T), B), A)
end

function Base.:/(A::TransVec, B::MaybeAdjOrTransTri)
    return transpose(transpose(B) \ parent(A))
end

function Base.:/(A::AdjVec, B::MaybeAdjOrTransTri)
    return adjoint(adjoint(B) \ parent(A))
end

function Base.:/(A::ChordalTriangular, α::Number)
    B = similar(A, promote_eltype(A, α))
    copyto!(B, A)
    rdiv!(B, α)
    return B
end

function Base.:\(α::Number, A::ChordalTriangular)
    return A / α
end

function Base.:\(α::Number, A::HermOrSymTri)
    return A / α
end

function Base.:/(A::HermTri{UPLO}, α::Number) where {UPLO}
    return Hermitian(parent(A) / α, UPLO)
end

function Base.:/(A::HermTri{UPLO}, α::Real) where {UPLO}
    return Hermitian(parent(A) / α, UPLO)
end

function Base.:/(A::HermTri{UPLO}, α::Complex) where {UPLO}
    @assert iszero(imag(α))
    return Hermitian(parent(A) / real(α), UPLO)
end

function Base.:/(A::SymTri{UPLO}, α::Number) where {UPLO}
    return Symmetric(parent(A) / α, UPLO)
end

function Base.:\(α::Number, A::AdjTri)
    return adjoint(conj(α) \ parent(A))
end

function Base.:/(A::AdjTri, α::Number)
    return adjoint(parent(A) / conj(α))
end

function Base.:\(α::Number, A::TransTri)
    return transpose(α \ parent(A))
end

function Base.:/(A::TransTri, α::Number)
    return transpose(parent(A) / α)
end

# ================================ ldiv! ================================

# --- ChordalTriangular ---

function LinearAlgebra.ldiv!(α::Number, C::ChordalTriangular)
    ldiv!(α, C.Dval)
    ldiv!(α, C.Lval)
    return C
end

function LinearAlgebra.ldiv!(α::Number, A::HermTri)
    ldiv!(α, parent(A))
    return A
end

function LinearAlgebra.ldiv!(α::Number, A::SymTri)
    ldiv!(α, parent(A))
    return A
end

function LinearAlgebra.ldiv!(α::Number, A::AdjTri)
    ldiv!(conj(α), parent(A))
    return A
end

function LinearAlgebra.ldiv!(α::Number, A::TransTri)
    ldiv!(α, parent(A))
    return A
end

function LinearAlgebra.ldiv!(A::MaybeAdjOrTransTri, B::AbstractVecOrMat{R}) where {R}
    W = AbstractDivisionWorkspace{R}(A, size(B, 2))
    return ldiv!(W, A, B)
end

function LinearAlgebra.ldiv!(W::DivisionWorkspace, A::MaybeAdjOrTransTri, B::AbstractVecOrMat)
    @assert size(A, 1) == size(B, 1)
    A, tA = unwrap(A)
    B, tB = unwrap(B)
    return div_impl!(B, W, A, tA, tB, Val(:L))
end

# --- AbstractFactorization ---

function LinearAlgebra.ldiv!(α::Number, F::AbstractFactorization{DIAG}) where {DIAG}
    if DIAG === :N
        ldiv!(sqrt(α), triangular(F))
    else
        ldiv!(α, F.D)
    end

    return F
end

function LinearAlgebra.ldiv!(F::NaturalFactorization, B::AbstractVecOrMat{R}) where {R}
    W = AbstractDivisionWorkspace{R}(F, size(B, 2))
    return ldiv!(W, F, B)
end

function LinearAlgebra.ldiv!(W::AbstractDivisionWorkspace, F::NaturalFactorization{DIAG}, B::AbstractVecOrMat) where {DIAG}
    @assert size(F, 1) == size(B, 1)

    if DIAG === :N
        return ldiv!(W, F.U, ldiv!(W, F.L, B))
    else
        return ldiv!(W, F.U, ldiv!(F.D, ldiv!(W, F.L, B)))
    end
end

function LinearAlgebra.ldiv!(F::AbstractFactorization, B::AbstractVecOrMat)
    @assert size(F, 1) == size(B, 1)
    T = promote_eltype(F, B)
    C = FArray{T}(undef, size(B))
    return ldiv!!(C, F, B)
end

# --- Permutation ---

function LinearAlgebra.ldiv!(C::AbstractVecOrMat, A::Permutation, B::AbstractVecOrMat)
    return mul!(C, inv(A), B)
end

function LinearAlgebra.ldiv!(C::Permutation, A::Permutation, B::Permutation)
    return mul!(C, inv(A), B)
end

# ================================ rdiv! ================================

# --- ChordalTriangular ---

function LinearAlgebra.rdiv!(C::ChordalTriangular, α::Number)
    rdiv!(C.Dval, α)
    rdiv!(C.Lval, α)
    return C
end

function LinearAlgebra.rdiv!(A::HermTri, α::Number)
    rdiv!(parent(A), α)
    return A
end

function LinearAlgebra.rdiv!(A::SymTri, α::Number)
    rdiv!(parent(A), α)
    return A
end

function LinearAlgebra.rdiv!(A::AdjTri, α::Number)
    rdiv!(parent(A), conj(α))
    return A
end

function LinearAlgebra.rdiv!(A::TransTri, α::Number)
    rdiv!(parent(A), α)
    return A
end

function LinearAlgebra.rdiv!(B::AbstractMatrix{R}, A::MaybeAdjOrTransTri) where {R}
    W = AbstractDivisionWorkspace{R}(A, size(B, 1))
    return rdiv!(W, B, A)
end

function LinearAlgebra.rdiv!(W::DivisionWorkspace, B::AbstractMatrix, A::MaybeAdjOrTransTri)
    @assert size(A, 1) == size(B, 2)
    A, tA = unwrap(A)
    B, tB = unwrap(B)
    return div_impl!(B, W, A, tA, tB, Val(:R))
end

# --- AbstractFactorization ---

function LinearAlgebra.rdiv!(F::AbstractFactorization{DIAG}, α::Number) where {DIAG}
    if DIAG === :N
        rdiv!(triangular(F), sqrt(α))
    else
        rdiv!(F.D, α)
    end

    return F
end

function LinearAlgebra.rdiv!(B::AbstractMatrix{R}, F::NaturalFactorization) where {R}
    W = AbstractDivisionWorkspace{R}(F, size(B, 1))
    return rdiv!(W, B, F)
end

function LinearAlgebra.rdiv!(W::AbstractDivisionWorkspace, B::AbstractMatrix, F::NaturalFactorization{DIAG}) where {DIAG}
    @assert size(F, 1) == size(B, 2)

    if DIAG === :N
        return rdiv!(W, rdiv!(W, B, F.U), F.L)
    else
        return rdiv!(W, rdiv!(rdiv!(W, B, F.U), F.D), F.L)
    end
end

function LinearAlgebra.rdiv!(B::AbstractMatrix, F::AbstractFactorization)
    @assert size(F, 1) == size(B, 2)
    T = promote_eltype(F, B)
    C = FMatrix{T}(undef, size(B))
    return rdiv!!(C, B, F)
end

# --- Permutation ---

function LinearAlgebra.rdiv!(C::AbstractMatrix, A::AbstractMatrix, B::Permutation)
    return mul!(C, A, inv(B))
end

function LinearAlgebra.rdiv!(C::Permutation, A::Permutation, B::Permutation)
    return mul!(C, A, inv(B))
end

# ================================ lpdiv! ===============================

# --- ChordalTriangular ---

function lpdiv!(A::MaybeAdjOrTransTri, B::AbstractVecOrMat{R}) where {R}
    W = AbstractDivisionWorkspace{R}(A, size(B, 2))
    return lpdiv!(W, A, B)
end

function lpdiv!(W::DivisionWorkspace, A::MaybeAdjOrTransTri, B::AbstractVecOrMat)
    @assert size(A, 1) == size(B, 1)
    A, tA = unwrap(A)
    B, tB = unwrap(B)
    return div_impl!(B, W, A, tA, tB, Val(:L), Val(true))
end

# ================================ rpdiv! ===============================

# --- ChordalTriangular ---

function rpdiv!(B::AbstractMatrix{R}, A::MaybeAdjOrTransTri) where {R}
    W = AbstractDivisionWorkspace{R}(A, size(B, 1))
    return rpdiv!(W, B, A)
end

function rpdiv!(W::DivisionWorkspace, B::AbstractMatrix, A::MaybeAdjOrTransTri)
    @assert size(A, 1) == size(B, 2)
    A, tA = unwrap(A)
    B, tB = unwrap(B)
    return div_impl!(B, W, A, tA, tB, Val(:R), Val(true))
end

# ============================== div_impl! ==============================

function div_impl!(
        B::AbstractVecOrMat{R},
        W::DivisionWorkspace{R},
        L::ChordalTriangular{DIAG, UPLO, T, I},
        tA::Val{TA},
        tB::Val{TB},
        side::Val{SIDE},
        piv::Val{PIV} = Val(false),
    ) where {T, R, I <: Integer, DIAG, UPLO, TA, TB, SIDE, PIV}
    return div_impl!(B, W.Mval, L, tA, tB, side, piv)
end

function div_impl!(
        B::AbstractVecOrMat{R},
        Mval::AbstractVector{R},
        L::ChordalTriangular{DIAG, UPLO, T, I},
        tA::Val{TA},
        tB::Val{TB},
        side::Val{SIDE},
        piv::Val{PIV} = Val(false),
    ) where {T, R, I <: Integer, DIAG, UPLO, TA, TB, SIDE, PIV}
    return div_impl!(
        B, Mval,
        L.S.Dptr, L.Dval,
        L.S.Lptr, L.Lval,
        L.S.res, L.S.sep,
        tA, tB, L.uplo, side, L.diag, piv)
end

function div_impl!(
        B::AbstractVecOrMat{R},
        Mval::AbstractVector{R},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        tA::Val{TA},
        tB::Val{TB},
        uplo::Val{UPLO},
        side::Val{SIDE},
        diag::Val{DIAG},
        piv::Val{PIV} = Val(false),
    ) where {T, R, I <: Integer, TA, TB, UPLO, SIDE, DIAG, PIV}

    if B isa AbstractVector
        nrhs = one(I)
    elseif SIDE === :L
        nrhs = convert(I, size(B, 2))
    else
        nrhs = convert(I, size(B, 1))
    end

    if isforward(UPLO, TA, SIDE)
        for j in vertices(res)
            div_fwd_loop!(B, Mval, Dptr, Dval, Lptr, Lval, res, sep, nrhs, j, tA, tB, uplo, side, diag, piv)
        end
    else
        for j in Iterators.reverse(vertices(res))
            div_bwd_loop!(B, Mval, Dptr, Dval, Lptr, Lval, res, sep, nrhs, j, tA, tB, uplo, side, diag, piv)
        end
    end

    return B
end

function div_fwd_loop!(
        C::AbstractVecOrMat{R},
        Mval::AbstractVector{R},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nrhs::I,
        j::I,
        tA::Val{TA},
        tB::Val{TB},
        uplo::Val{UPLO},
        side::Val{SIDE},
        diag::Val{DIAG},
        piv::Val{PIV} = Val(false),
    ) where {T, R, I <: Integer, TA, TB, UPLO, SIDE, DIAG, PIV}

    nn = eltypedegree(res, j)

    if T <: Real && isone(nn)
        return div_fwd_loop_nod!(
            C, Dptr, Dval,
            Lptr, Lval,
            res, sep, nrhs, j, tA, tB, uplo, side, diag, piv
        )
    else
        return div_fwd_loop_snd!(
            C, Mval, Dptr, Dval,
            Lptr, Lval,
            res, sep, nrhs, nn, j, tA, tB, uplo, side, diag, piv
        )
    end
end

function div_fwd_loop_snd!(
        C::AbstractVecOrMat{R},
        Mval::AbstractVector{R},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nrhs::I,
        nn::I,
        j::I,
        tA::Val{TA},
        tB::Val{TB},
        uplo::Val{UPLO},
        side::Val{SIDE},
        diag::Val{DIAG},
        piv::Val{PIV} = Val(false),
    ) where {T, R, I <: Integer, TA, TB, UPLO, SIDE, DIAG, PIV}
    #
    # na is the size of the separator at node j
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(sep, j)
    Dp = Dptr[j]
    Lp = Lptr[j]
    #
    # L is part of the factor
    #
    #          res(j)
    #     L = [ D₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    # determine the rank of D₁₁
    #
    if PIV
        rank = one(I)

        @inbounds while rank ≤ nn && ispositive(D₁₁[rank, rank])
            rank += one(I)
        end

        rank -= one(I)
    else
        rank = nn
    end
    #
    # extract the full-rank submatrix of D₁₁:
    #
    #   D₁₁ = [ rD₁₁   ]
    #         [      0 ]
    #
    rD₁₁ = view(D₁₁, oneto(rank), oneto(rank))
    #
    # extract the full-rank submatrix of L₂₁:
    #
    #   L₂₁ = [ rL₂₁ 0 ]
    #
    if UPLO === :L
        rL₂₁ = view(L₂₁, oneto(na), oneto(rank))
    else
        rL₂₁ = view(L₂₁, oneto(rank), oneto(na))
    end
    #
    # C₁ is part of the right-hand side
    #
    #   C = [ C₁  ] res(j)
    #       [ C₂  ] sep(j)
    #
    #     = [ rC₁ ]
    #       [ nC₁ ]
    #       [  C₂ ]
    #
    if C isa AbstractVector
        C₁  = view(C, neighbors(res, j))
        rC₁ = view(C₁,      oneto(rank))
        nC₁ = view(C₁, rank + one(I):nn)
    elseif SIDE === :L
        C₁  = view(C,  neighbors(res, j), oneto(nrhs))
        rC₁ = view(C₁,  oneto(rank),      oneto(nrhs))
        nC₁ = view(C₁,  rank + one(I):nn, oneto(nrhs))
    else
        C₁  = view(C,  oneto(nrhs), neighbors(res, j))
        rC₁ = view(C₁, oneto(nrhs),       oneto(rank))
        nC₁ = view(C₁, oneto(nrhs),  rank + one(I):nn)
    end
    #
    #     nC₁ ← 0
    #
    PIV && zerorec!(nC₁)
    #
    #     rC₁ ← rD₁₁⁻¹ rC₁
    #
    if ispositive(rank)
        if C isa AbstractVector
            trsv!(uplo,       tA, diag,         rD₁₁, rC₁)
        else
            trsm!(side, uplo, tA, diag, one(R), rD₁₁, rC₁)
        end
    end

    if ispositive(na) && ispositive(rank)
        if C isa AbstractVector
            M₂ = view(Mval, oneto(na))
        elseif SIDE === :L
            M₂ = reshape(view(Mval, oneto(na * nrhs)), na, nrhs)
        else
            M₂ = reshape(view(Mval, oneto(na * nrhs)), nrhs, na)
        end
        #
        #     M₂ ← -rL₂₁ rC₁
        #
        if C isa AbstractVector
            if UPLO === :L
                gemv!(Val(:N), -one(R), rL₂₁, rC₁, zero(R), M₂)
            else
                gemv!(tA,      -one(R), rL₂₁, rC₁, zero(R), M₂)
            end
        elseif SIDE === :L
            if UPLO === :L
                gemm!(Val(:N), tB, -one(R), rL₂₁, rC₁, zero(R), M₂)
            else
                gemm!(tA, tB,      -one(R), rL₂₁, rC₁, zero(R), M₂)
            end
        else
            if UPLO === :L
                gemm!(tB, tA,      -one(R), rC₁, rL₂₁, zero(R), M₂)
            else
                gemm!(tB, Val(:N), -one(R), rC₁, rL₂₁, zero(R), M₂)
            end
        end
        #
        #     C₂ ← C₂ + M₂
        #
        if C isa AbstractVector
            addscatterrec!(C, M₂, neighbors(sep, j))
        else
            addscatterrec!(C, M₂, neighbors(sep, j), side)
        end
    end

    return
end

function div_fwd_loop_nod!(
        C::AbstractVecOrMat{R},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nrhs::I,
        j::I,
        tA::Val{TA},
        tB::Val{TB},
        uplo::Val{UPLO},
        side::Val{SIDE},
        diag::Val{DIAG},
        piv::Val{PIV} = Val(false),
    ) where {T, R, I <: Integer, TA, TB, UPLO, SIDE, DIAG, PIV}
    #
    # na is the size of the separator at node j
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(sep, j)
    #
    # L is part of the factor:
    #
    #          res(j)
    #     L = [ d₁₁  ] res(j)
    #         [ l₂₁  ] sep(j)
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    Rp = pointers(res)[j]
    Sp = pointers(sep)[j]
    d₁₁ = Dval[Dp]
    #
    #     c₁ ←      d₁₁⁻¹ c₁
    #     c₂ ← c₂ - l₂₁   c₁
    #
    if C isa AbstractVector
        if PIV && !ispositive(d₁₁)
            c₁ = zero(R)
        else
            c₁ = C[Rp]

            if DIAG === :N
                c₁ /= d₁₁
            end
        end

        C[Rp] = c₁

        @inbounds for i in oneto(na)
            l =         Lval[Lp + i - one(I)]
            s = targets(sep)[Sp + i - one(I)]

            C[s] -= l * c₁
        end
    elseif SIDE === :L
        if PIV && !ispositive(d₁₁)
            @inbounds for k in oneto(nrhs)
                C[Rp, k] = zero(R)
            end
        elseif DIAG === :N
            @inbounds for k in oneto(nrhs)
                C[Rp, k] /= d₁₁
            end
        end

        @inbounds for i in oneto(na)
            l =         Lval[Lp + i - one(I)]
            s = targets(sep)[Sp + i - one(I)]

            for k in oneto(nrhs)
                C[s, k] -= l * C[Rp, k]
            end
        end
    else
        if PIV && !ispositive(d₁₁)
            @inbounds for k in oneto(nrhs)
                C[k, Rp] = zero(R)
            end
        elseif DIAG === :N
            @inbounds for k in oneto(nrhs)
                C[k, Rp] /= d₁₁
            end
        end

        @inbounds for i in oneto(na)
            l =         Lval[Lp + i - one(I)]
            s = targets(sep)[Sp + i - one(I)]

            for k in oneto(nrhs)
                C[k, s] -= l * C[k, Rp]
            end
        end
    end

    return
end

function div_bwd_loop!(
        C::AbstractVecOrMat{R},
        Mval::AbstractVector{R},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nrhs::I,
        j::I,
        tA::Val{TA},
        tB::Val{TB},
        uplo::Val{UPLO},
        side::Val{SIDE},
        diag::Val{DIAG},
        piv::Val{PIV} = Val(false),
    ) where {T, R, I <: Integer, TA, TB, UPLO, SIDE, DIAG, PIV}
    nn = eltypedegree(res, j)

    if T <: Real && isone(nn)
        return div_bwd_loop_nod!(
            C, Dptr, Dval,
            Lptr, Lval,
            res, sep, nrhs, j, tA, tB, uplo, side, diag, piv
        )
    else
        return div_bwd_loop_snd!(
            C, Mval, Dptr, Dval,
            Lptr, Lval,
            res, sep, nrhs, nn, j, tA, tB, uplo, side, diag, piv
        )
    end
end

function div_bwd_loop_snd!(
        C::AbstractVecOrMat{R},
        Mval::AbstractVector{R},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nrhs::I,
        nn::I,
        j::I,
        tA::Val{TA},
        tB::Val{TB},
        uplo::Val{UPLO},
        side::Val{SIDE},
        diag::Val{DIAG},
        piv::Val{PIV} = Val(false),
    ) where {T, R, I <: Integer, TA, TB, UPLO, SIDE, DIAG, PIV}
    #
    # na is the size of the separator at node j
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(sep, j)
    Dp = Dptr[j]
    Lp = Lptr[j]
    #
    # L is part of the factor
    #
    #          res(j)
    #     L = [ D₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    # determine the rank of D₁₁
    #
    if PIV
        rank = one(I)

        @inbounds while rank ≤ nn && ispositive(D₁₁[rank, rank])
            rank += one(I)
        end

        rank -= one(I)
    else
        rank = nn
    end
    #
    # extract the full-rank submatrix of D₁₁:
    #
    #   D₁₁ = [ rD₁₁   ]
    #         [      0 ]
    #
    rD₁₁ = view(D₁₁, oneto(rank), oneto(rank))
    #
    # extract the full-rank submatrix of L₂₁:
    #
    #   L₂₁ = [ rL₂₁ 0 ]
    #
    if UPLO === :L
        rL₂₁ = view(L₂₁, oneto(na), oneto(rank))
    else
        rL₂₁ = view(L₂₁, oneto(rank), oneto(na))
    end
    #
    # C₁ is part of the right-hand side
    #
    #        nrhs
    #   C = [ C₁  ] res(j)
    #       [ C₂  ] sep(j)
    #
    #     = [ rC₁ ]
    #       [ nC₁ ]
    #       [  C₂ ]
    #
    if C isa AbstractVector
        C₁  = view(C,  neighbors(res, j))
        rC₁ = view(C₁,       oneto(rank))
        nC₁ = view(C₁,  rank + one(I):nn)
    elseif SIDE === :L
        C₁  = view(C,  neighbors(res, j), oneto(nrhs))
        rC₁ = view(C₁,  oneto(rank),      oneto(nrhs))
        nC₁ = view(C₁,  rank + one(I):nn, oneto(nrhs))
    else
        C₁  = view(C,  oneto(nrhs), neighbors(res, j))
        rC₁ = view(C₁, oneto(nrhs),       oneto(rank))
        nC₁ = view(C₁, oneto(nrhs),  rank + one(I):nn)
    end

    if ispositive(na) && ispositive(rank)
        if C isa AbstractVector
            M₂ = view(Mval, oneto(na))
        elseif SIDE === :L
            M₂ = reshape(view(Mval, oneto(na * nrhs)), na, nrhs)
        else
            M₂ = reshape(view(Mval, oneto(na * nrhs)), nrhs, na)
        end
        #
        #     M₂ ← C₂
        #
        if C isa AbstractVector
            copygatherrec!(M₂, C, neighbors(sep, j))
        else
            copygatherrec!(M₂, C, neighbors(sep, j), side)
        end
        #
        #     rC₁ ← rC₁ - rL₂₁ᴴ M₂
        #
        if C isa AbstractVector
            if UPLO === :L
                gemv!(tA,      -one(R), rL₂₁, M₂, one(R), rC₁)
            else
                gemv!(Val(:N), -one(R), rL₂₁, M₂, one(R), rC₁)
            end
        elseif SIDE === :L
            if UPLO === :L
                gemm!(tA,      Val(:N), -one(R), rL₂₁, M₂, one(R), rC₁)
            else
                gemm!(Val(:N), Val(:N), -one(R), rL₂₁, M₂, one(R), rC₁)
            end
        else
            if UPLO === :L
                gemm!(Val(:N), Val(:N), -one(R), M₂, rL₂₁, one(R), rC₁)
            else
                gemm!(Val(:N), tA,      -one(R), M₂, rL₂₁, one(R), rC₁)
            end
        end
    end
    #
    #     nC₁ ← 0
    #
    PIV && zerorec!(nC₁)
    #
    #     rC₁ ← rD₁₁⁻ᴴ rC₁
    #
    if ispositive(rank)
        if C isa AbstractVector
            trsv!(uplo,       tA, diag,         rD₁₁, rC₁)
        else
            trsm!(side, uplo, tA, diag, one(R), rD₁₁, rC₁)
        end
    end

    return
end

function div_bwd_loop_nod!(
        C::AbstractVecOrMat{R},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nrhs::I,
        j::I,
        tA::Val{TA},
        tB::Val{TB},
        uplo::Val{UPLO},
        side::Val{SIDE},
        diag::Val{DIAG},
        piv::Val{PIV} = Val(false),
    ) where {T, R, I <: Integer, TA, TB, UPLO, SIDE, DIAG, PIV}
    #
    # na is the size of the separator at node j
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(sep, j)
    #
    # L is part of the factor:
    #
    #          res(j)
    #     L = [ d₁₁  ] res(j)
    #         [ l₂₁  ] sep(j)
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    Rp = pointers(res)[j]
    Sp = pointers(sep)[j]
    d₁₁ = Dval[Dp]
    #
    #     c₁ ←       c₁ - l₂₁ᴴ c₂
    #     c₁ ← d₁₁⁻¹ c₁
    #
    if C isa AbstractVector
        if PIV && !ispositive(d₁₁)
            C[Rp] = zero(R)
        else
            Δ = zero(R)

            @inbounds for i in oneto(na)
                l =         Lval[Lp + i - one(I)]
                s = targets(sep)[Sp + i - one(I)]

                Δ += l * C[s]
            end

            c₁ = C[Rp] - Δ

            if DIAG === :N
                c₁ /= d₁₁
            end

            C[Rp] = c₁
        end
    elseif SIDE === :L
        if PIV && !ispositive(d₁₁)
            @inbounds for k in oneto(nrhs)
                C[Rp, k] = zero(R)
            end
        else
            @inbounds for i in oneto(na)
                l =         Lval[Lp + i - one(I)]
                s = targets(sep)[Sp + i - one(I)]

                for k in oneto(nrhs)
                    C[Rp, k] -= l * C[s, k]
                end
            end

            if DIAG === :N
                @inbounds for k in oneto(nrhs)
                    C[Rp, k] /= d₁₁
                end
            end
        end
    else
        if PIV && !ispositive(d₁₁)
            @inbounds for k in oneto(nrhs)
                C[k, Rp] = zero(R)
            end
        else
            @inbounds for i in oneto(na)
                l =         Lval[Lp + i - one(I)]
                s = targets(sep)[Sp + i - one(I)]

                for k in oneto(nrhs)
                    C[k, Rp] -= l * C[k, s]
                end
            end

            if DIAG === :N
                @inbounds for k in oneto(nrhs)
                    C[k, Rp] /= d₁₁
                end
            end
        end
    end

    return
end

# ================================ ldiv!! ================================

# C is a workspace. Returns B.
function ldiv!!(C, F::AbstractFactorization, B)
    ldiv!(B, F.P, ldiv!(NaturalFactorization(F), mul!(C, F.P, B)))
end

# ================================ rdiv!! ================================

# C is a workspace. Returns B.
function rdiv!!(C, B, F::AbstractFactorization)
    mul!(B, rdiv!(rdiv!(C, B, F.P), NaturalFactorization(F)), F.P)
end
