const MultiplicationWorkspace = DivisionWorkspace
const AbstractMultiplicationWorkspace = AbstractDivisionWorkspace

# ================================== dot ==================================

function LinearAlgebra.dot(A::AbstractVecOrMat, F::NaturalFactorization{DIAG}, B::AbstractVecOrMat) where {DIAG}
    @assert size(F, 1) == size(A, 1) == size(B, 1)
    @assert size(A) == size(B)
    C = F.U * A
    E = F.U * B

    if DIAG === :U
        lmul!(F.D, E)
    end

    return dot(C, E)
end

function LinearAlgebra.dot(A::AbstractVecOrMat, F::AbstractFactorization, B::AbstractVecOrMat)
    @assert size(F, 1) == size(A, 1) == size(B, 1)
    @assert size(A) == size(B)
    return dot(F.P * A, NaturalFactorization(F), F.P * B)
end

# ================================= cong =================================

function cong(A, B)
    return B' * A * B
end

function cong(A::SparseMatrixCSC, B::Permutation)
    return permute(A, B.invp, B.invp)
end

function cong(A::Hermitian, B::Permutation)
    return Hermitian(sympermute(parent(A), B.perm, A.uplo, A.uplo), Symbol(A.uplo))
end

function cong(A::Symmetric{T}, B::Permutation) where {T <: Real}
    return Symmetric(sympermute(parent(A), B.perm, A.uplo, A.uplo), Symbol(A.uplo))
end

function cong(A::Diagonal, B::Permutation)
    return Diagonal(B \ A.diag)
end

# ================================== * ==================================

# --- Permutation ---

function Base.:*(A::Permutation{I}, B::Permutation{I}) where {I}
    @assert size(A, 1) == size(B, 1)
    C = Permutation{I}(size(A, 1))
    return mul!(C, A, B)
end

function Base.:*(A::Permutation, B::SparseMatrixCSC)
    return rowpermute(B, A.perm)
end

function Base.:*(A::SparseMatrixCSC, B::Permutation)
    return colpermute(A, B.invp)
end

function Base.:*(A::Permutation, B::Diagonal)
    C = Diagonal(similar(B.diag, promote_eltype(A, B)))
    return mul!(C, A, B)
end

# --- AbstractFactorization ---

function Base.:*(F::AbstractFactorization, B::AbstractVecOrMat)
    T = promote_eltype(F, B)
    return lmul!(F, copyto!(similar(B, T), B))
end

function Base.:*(B::AbstractMatrix, F::AbstractFactorization)
    T = promote_eltype(F, B)
    return rmul!(copyto!(similar(B, T), B), F)
end

# --- ChordalTriangular ---

function Base.:*(A::ChordalTriangular, α::Number)
    B = similar(A, promote_eltype(A, α))
    copyto!(B, A)
    rmul!(B, α)
    return B
end

function Base.:*(α::Number, A::MaybeHermOrSymTri)
    return A * α
end

function Base.:*(α::Real, A::HermTri{UPLO}) where {UPLO}
    return A * α
end

function Base.:*(A::HermTri{UPLO}, α::Number) where {UPLO}
    return Hermitian(parent(A) * α, UPLO)
end

function Base.:*(A::HermTri{UPLO}, α::Real) where {UPLO}
    return Hermitian(parent(A) * α, UPLO)
end

function Base.:*(A::HermTri{UPLO}, α::Complex) where {UPLO}
    @assert iszero(imag(α))
    return Hermitian(parent(A) * real(α), UPLO)
end

function Base.:*(A::SymTri{UPLO}, α::Number) where {UPLO}
    return Symmetric(parent(A) * α, UPLO)
end

function Base.:*(α::Number, A::AdjTri)
    return adjoint(conj(α) * parent(A))
end

function Base.:*(A::AdjTri, α::Number)
    return adjoint(parent(A) * conj(α))
end

function Base.:*(α::Number, A::TransTri)
    return transpose(α * parent(A))
end

function Base.:*(A::TransTri, α::Number)
    return transpose(parent(A) * α)
end

# ================================ lmul! ================================

# --- AbstractFactorization ---

function LinearAlgebra.lmul!(α::Number, F::AbstractFactorization{DIAG}) where {DIAG}
    if DIAG === :N
        lmul!(sqrt(α), triangular(F))
    else
        lmul!(α, F.D)
    end

    return F
end

function LinearAlgebra.lmul!(F::NaturalFactorization{DIAG}, B::AbstractVecOrMat) where {DIAG}
    @assert size(F, 1) == size(B, 1)

    if DIAG === :N
        return lmul!(F.L, lmul!(F.U, B))
    else
        return lmul!(F.L, lmul!(F.D, lmul!(F.U, B)))
    end
end

function LinearAlgebra.lmul!(F::AbstractFactorization, B::AbstractVecOrMat)
    @assert size(F, 1) == size(B, 1)
    T = promote_eltype(F, B)
    C = FArray{T}(undef, size(B))
    return mul!!(C, F, B)
end

# --- ChordalTriangular ---

function LinearAlgebra.lmul!(α::Number, C::ChordalTriangular)
    lmul!(α, C.Dval)
    lmul!(α, C.Lval)
    return C
end

function LinearAlgebra.lmul!(α::Number, A::HermTri)
    lmul!(α, parent(A))
    return A
end

function LinearAlgebra.lmul!(α::Number, A::SymTri)
    lmul!(α, parent(A))
    return A
end

function LinearAlgebra.lmul!(α::Number, A::AdjTri)
    lmul!(conj(α), parent(A))
    return A
end

function LinearAlgebra.lmul!(α::Number, A::TransTri)
    lmul!(α, parent(A))
    return A
end

function LinearAlgebra.lmul!(A::MaybeAdjOrTransTri, B::AbstractVecOrMat{R}) where {R}
    W = AbstractMultiplicationWorkspace{R}(A, size(B, 2))
    return lmul!(W, A, B)
end

function LinearAlgebra.lmul!(W::MultiplicationWorkspace, A::MaybeAdjOrTransTri, B::AbstractVecOrMat)
    @assert size(A, 1) == size(B, 1)
    A, tA = unwrap(A)
    B, tB = unwrap(B)
    return mul_impl!(B, W, A, tA, tB, Val(:L))
end

# ================================ rmul! ================================

# --- AbstractFactorization ---

function LinearAlgebra.rmul!(F::AbstractFactorization{DIAG}, α::Number) where {DIAG}
    if DIAG === :N
        rmul!(triangular(F), sqrt(α))
    else
        rmul!(F.D, α)
    end

    return F
end

function LinearAlgebra.rmul!(B::AbstractMatrix, F::NaturalFactorization{DIAG}) where {DIAG}
    @assert size(F, 1) == size(B, 2)

    if DIAG === :N
        return rmul!(rmul!(B, F.L), F.U)
    else
        return rmul!(rmul!(rmul!(B, F.L), F.D), F.U)
    end
end

function LinearAlgebra.rmul!(B::AbstractMatrix, F::AbstractFactorization)
    @assert size(F, 1) == size(B, 2)
    T = promote_eltype(F, B)
    C = FMatrix{T}(undef, size(B))
    return mul!!(C, B, F)
end

# --- ChordalTriangular ---

function LinearAlgebra.rmul!(C::ChordalTriangular, α::Number)
    rmul!(C.Dval, α)
    rmul!(C.Lval, α)
    return C
end

function LinearAlgebra.rmul!(A::HermTri, α::Number)
    rmul!(parent(A), α)
    return A
end

function LinearAlgebra.rmul!(A::SymTri, α::Number)
    rmul!(parent(A), α)
    return A
end

function LinearAlgebra.rmul!(A::AdjTri, α::Number)
    rmul!(parent(A), conj(α))
    return A
end

function LinearAlgebra.rmul!(A::TransTri, α::Number)
    rmul!(parent(A), α)
    return A
end

function LinearAlgebra.rmul!(B::AbstractMatrix{R}, A::MaybeAdjOrTransTri) where {R}
    W = AbstractMultiplicationWorkspace{R}(A, size(B, 1))
    return rmul!(W, B, A)
end

function LinearAlgebra.rmul!(W::MultiplicationWorkspace, B::AbstractMatrix, A::MaybeAdjOrTransTri)
    @assert size(A, 1) == size(B, 2)
    A, tA = unwrap(A)
    B, tB = unwrap(B)
    return mul_impl!(B, W, A, tA, tB, Val(:R))
end

# ================================= mul! ================================

# --- AbstractFactorization ---

function LinearAlgebra.mul!(C::AbstractVecOrMat, F::AbstractFactorization, B::AbstractVecOrMat)
    lmul!(F, copyrec!(C, B))
end

# --- ChordalTriangular ---

function LinearAlgebra.mul!(C::AbstractVecOrMat, A::MaybeAdjOrTransTri, B::AbstractVecOrMat)
    lmul!(A, copyrec!(C, B))
end

# --- Permutation ---

function LinearAlgebra.mul!(C::AbstractVecOrMat, A::Permutation, B::AbstractVecOrMat)
    @boundscheck size(C, 1) == size(B, 1) == size(A, 1) || throw(DimensionMismatch())
    return copyscatterrec!(C, B, A.invp, Val(:L))
end

function LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, B::Permutation)
    @boundscheck size(C, 2) == size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    return copyscatterrec!(C, A, B.perm, Val(:R))
end

function LinearAlgebra.mul!(C::Diagonal, A::Permutation, B::Diagonal)
    @boundscheck length(C.diag) == length(B.diag) == size(A, 1) || throw(DimensionMismatch())
    copyscatterrec!(C.diag, B.diag, A.invp)
    return C
end

# ambiguity
function LinearAlgebra.mul!(::AbstractMatrix, ::MaybeAdjOrTransTri, ::Permutation)
    error()
end

# ambiguity
function LinearAlgebra.mul!(::AbstractMatrix, ::HermOrSymTri, ::Permutation)
    error()
end

# ambiguity
function LinearAlgebra.mul!(::AbstractMatrix, ::Permutation, ::HermOrSymTri)
    error()
end

# ambiguity
function LinearAlgebra.mul!(::AbstractMatrix, ::MaybeAdjOrTransTri, ::HermOrSymTri)
    error()
end

# ambiguity
function LinearAlgebra.mul!(::AbstractMatrix, ::HermOrSymTri, ::HermOrSymTri)
    error()
end

function LinearAlgebra.mul!(C::Permutation, A::Permutation, B::Permutation)
    @boundscheck size(C, 2) == size(A, 2) == size(B, 1) || throw(DimensionMismatch())

    @inbounds for i in axes(C, 1)
        j = C.perm[i] = B.perm[A.perm[i]]
        C.invp[j] = i
    end

    return C
end

function LinearAlgebra.mul!(C::AbstractMatrix{T}, A::Permutation, B::Permutation) where {T}
    @boundscheck size(C, 1) == size(C, 2) == size(A, 1) == size(B, 1) || throw(DimensionMismatch())
    fill!(C, zero(T))

    @inbounds for i in axes(C, 1)
        C[i, B.perm[A.perm[i]]] = one(T)
    end

    return C
end
# ============================== mul_impl! ==============================
#
# The multiply is a blocked right-looking triangular product, the dual of the
# division solve. Applying the factor scatters L₂₁ C₁ onto the separator and is
# traversed root → leaves (so a separator slot is written only after it has been
# consumed); applying its adjoint gathers from the separator and is traversed
# leaves → root. As in the solve, `piv` restricts each front to its full-rank
# part and projects the null residual to zero.
#

function mul_impl!(
        B::AbstractVecOrMat{R},
        W::MultiplicationWorkspace{R},
        L::ChordalTriangular{DIAG, UPLO, T, I},
        tA::Val{TA},
        tB::Val{TB},
        side::Val{SIDE},
        piv::Val{PIV} = Val(false),
    ) where {T, R, I <: Integer, DIAG, UPLO, TA, TB, SIDE, PIV}
    return mul_impl!(B, W.Mval, L, tA, tB, side, piv)
end

function mul_impl!(
        B::AbstractVecOrMat{R},
        Mval::AbstractVector{R},
        L::ChordalTriangular{DIAG, UPLO, T, I},
        tA::Val{TA},
        tB::Val{TB},
        side::Val{SIDE},
        piv::Val{PIV} = Val(false),
    ) where {T, R, I <: Integer, DIAG, UPLO, TA, TB, SIDE, PIV}
    return mul_impl!(
        B, Mval,
        L.S.Dptr, L.Dval,
        L.S.Lptr, L.Lval,
        L.S.res, L.S.sep,
        tA, tB, L.uplo, side, L.diag, piv)
end

function mul_impl!(
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
        for j in Iterators.reverse(vertices(res))
            mul_fwd_loop!(B, Mval, Dptr, Dval, Lptr, Lval, res, sep, nrhs, j, tA, tB, uplo, side, diag, piv)
        end
    else
        for j in vertices(res)
            mul_bwd_loop!(B, Mval, Dptr, Dval, Lptr, Lval, res, sep, nrhs, j, tA, tB, uplo, side, diag, piv)
        end
    end

    return B
end

function mul_fwd_loop!(
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
        return mul_fwd_loop_nod!(
            C, Dptr, Dval,
            Lptr, Lval,
            res, sep, nrhs, j, tA, tB, uplo, side, diag, piv
        )
    else
        return mul_fwd_loop_snd!(
            C, Mval, Dptr, Dval,
            Lptr, Lval,
            res, sep, nrhs, nn, j, tA, tB, uplo, side, diag, piv
        )
    end
end

function mul_fwd_loop_snd!(
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
        C₁ = view(C, neighbors(res, j))
        rC₁ = view(C₁,      oneto(rank))
        nC₁ = view(C₁, rank + one(I):nn)
    elseif SIDE === :L
        C₁ = view(C, neighbors(res, j), oneto(nrhs))
        rC₁ = view(C₁, oneto(rank),      oneto(nrhs))
        nC₁ = view(C₁, rank + one(I):nn, oneto(nrhs))
    else
        C₁ = view(C, oneto(nrhs), neighbors(res, j))
        rC₁ = view(C₁, oneto(nrhs),      oneto(rank))
        nC₁ = view(C₁, oneto(nrhs), rank + one(I):nn)
    end

    if ispositive(na) && ispositive(rank)
        #
        # M₂ is the update to the separator part of the right-hand side
        #
        if C isa AbstractVector
            M₂ = view(Mval, oneto(na))
        elseif SIDE === :L
            M₂ = reshape(view(Mval, oneto(na * nrhs)), na, nrhs)
        else
            M₂ = reshape(view(Mval, oneto(na * nrhs)), nrhs, na)
        end
        #
        #     M₂ ← rL₂₁ rC₁
        #
        if C isa AbstractVector
            if UPLO === :L
                gemv!(Val(:N), one(R), rL₂₁, rC₁, zero(R), M₂)
            else
                gemv!(tA, one(R), rL₂₁, rC₁, zero(R), M₂)
            end
        elseif SIDE === :L
            if UPLO === :L
                gemm!(Val(:N), tB, one(R), rL₂₁, rC₁, zero(R), M₂)
            else
                gemm!(tA, tB, one(R), rL₂₁, rC₁, zero(R), M₂)
            end
        else
            if UPLO === :L
                gemm!(tB, tA, one(R), rC₁, rL₂₁, zero(R), M₂)
            else
                gemm!(tB, Val(:N), one(R), rC₁, rL₂₁, zero(R), M₂)
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
    #
    #     rC₁ ← rD₁₁ rC₁
    #
    if ispositive(rank)
        if C isa AbstractVector
            trmv!(uplo,       tA, diag,         rD₁₁, rC₁)
        else
            trmm!(side, uplo, tA, diag, one(R), rD₁₁, rC₁)
        end
    end
    #
    #     nC₁ ← 0
    #
    PIV && zerorec!(nC₁)

    return
end

# Fast path for nn = 1 (residual size is 1)
# In this case, diagonal blocks are scalars and off-diagonal blocks are vectors
function mul_fwd_loop_nod!(
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
    # L is part of the factor (d₁₁ is scalar, l₂₁ is vector)
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
    #     c₂ ← c₂    + l₂₁ c₁
    #     c₁ ←      d₁₁    c₁   (c₁ ← 0 if d₁₁ is not positive)
    #
    if PIV && !ispositive(d₁₁)
        if C isa AbstractVector
            C[Rp] = zero(R)
        elseif SIDE === :L
            @inbounds for k in oneto(nrhs)
                C[Rp, k] = zero(R)
            end
        else
            @inbounds for k in oneto(nrhs)
                C[k, Rp] = zero(R)
            end
        end

        return
    end

    if C isa AbstractVector
        c₁ = C[Rp]

        @inbounds for i in oneto(na)
            l =         Lval[Lp + i - one(I)]
            s = targets(sep)[Sp + i - one(I)]

            C[s] += l * c₁
        end

        DIAG === :N && (C[Rp] = d₁₁ * c₁)
    elseif SIDE === :L
        @inbounds for i in oneto(na)
            l =         Lval[Lp + i - one(I)]
            s = targets(sep)[Sp + i - one(I)]

            for k in oneto(nrhs)
                C[s, k] += l * C[Rp, k]
            end
        end

        if DIAG === :N
            @inbounds for k in oneto(nrhs)
                C[Rp, k] *= d₁₁
            end
        end
    else
        @inbounds for i in oneto(na)
            l =         Lval[Lp + i - one(I)]
            s = targets(sep)[Sp + i - one(I)]

            @simd for k in oneto(nrhs)
                C[k, s] += l * C[k, Rp]
            end
        end

        if DIAG === :N
            @inbounds for k in oneto(nrhs)
                C[k, Rp] *= d₁₁
            end
        end
    end

    return
end

function mul_bwd_loop!(
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
        return mul_bwd_loop_nod!(
            C, Dptr, Dval,
            Lptr, Lval,
            res, sep, nrhs, j, tA, tB, uplo, side, diag, piv
        )
    else
        return mul_bwd_loop_snd!(
            C, Mval, Dptr, Dval,
            Lptr, Lval,
            res, sep, nrhs, nn, j, tA, tB, uplo, side, diag, piv
        )
    end
end

function mul_bwd_loop_snd!(
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
        C₁ = view(C, neighbors(res, j))
        rC₁ = view(C₁,      oneto(rank))
        nC₁ = view(C₁, rank + one(I):nn)
    elseif SIDE === :L
        C₁ = view(C, neighbors(res, j), oneto(nrhs))
        rC₁ = view(C₁, oneto(rank),      oneto(nrhs))
        nC₁ = view(C₁, rank + one(I):nn, oneto(nrhs))
    else
        C₁ = view(C, oneto(nrhs), neighbors(res, j))
        rC₁ = view(C₁, oneto(nrhs),      oneto(rank))
        nC₁ = view(C₁, oneto(nrhs), rank + one(I):nn)
    end
    #
    #     rC₁ ← rD₁₁ᴴ rC₁
    #
    if ispositive(rank)
        if C isa AbstractVector
            trmv!(uplo,       tA, diag,         rD₁₁, rC₁)
        else
            trmm!(side, uplo, tA, diag, one(R), rD₁₁, rC₁)
        end
    end
    #
    #     nC₁ ← 0
    #
    PIV && zerorec!(nC₁)

    if ispositive(na) && ispositive(rank)
        #
        # M₂ is the separator part of the right-hand side
        #
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
        #     rC₁ ← rC₁ + rL₂₁ᴴ M₂
        #
        if C isa AbstractVector
            if UPLO === :L
                gemv!(tA, one(R), rL₂₁, M₂, one(R), rC₁)
            else
                gemv!(Val(:N), one(R), rL₂₁, M₂, one(R), rC₁)
            end
        elseif SIDE === :L
            if UPLO === :L
                gemm!(tA, Val(:N), one(R), rL₂₁, M₂, one(R), rC₁)
            else
                gemm!(Val(:N), Val(:N), one(R), rL₂₁, M₂, one(R), rC₁)
            end
        else
            if UPLO === :L
                gemm!(Val(:N), Val(:N), one(R), M₂, rL₂₁, one(R), rC₁)
            else
                gemm!(Val(:N), tA, one(R), M₂, rL₂₁, one(R), rC₁)
            end
        end
    end

    return
end

function mul_bwd_loop_nod!(
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
    # L is part of the factor (d₁₁ is scalar, l₂₁ is vector)
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
    #     c₁ ←      d₁₁ᴴ  c₁   (c₁ ← 0 if d₁₁ is not positive)
    #     c₁ ← c₁    + l₂₁ᴴ c₂
    #
    if PIV && !ispositive(d₁₁)
        if C isa AbstractVector
            C[Rp] = zero(R)
        elseif SIDE === :L
            @inbounds for k in oneto(nrhs)
                C[Rp, k] = zero(R)
            end
        else
            @inbounds for k in oneto(nrhs)
                C[k, Rp] = zero(R)
            end
        end

        return
    end

    if C isa AbstractVector
        DIAG === :N && (C[Rp] *= d₁₁)

        Δ = zero(R)

        @inbounds for i in oneto(na)
            l =         Lval[Lp + i - one(I)]
            s = targets(sep)[Sp + i - one(I)]

            Δ += l * C[s]
        end

        C[Rp] += Δ
    elseif SIDE === :L
        if DIAG === :N
            @inbounds for k in oneto(nrhs)
                C[Rp, k] *= d₁₁
            end
        end

        @inbounds for i in oneto(na)
            l =         Lval[Lp + i - one(I)]
            s = targets(sep)[Sp + i - one(I)]

            for k in oneto(nrhs)
                C[Rp, k] += l * C[s, k]
            end
        end
    else
        if DIAG === :N
            @inbounds for k in oneto(nrhs)
                C[k, Rp] *= d₁₁
            end
        end

        @inbounds for i in oneto(na)
            l =         Lval[Lp + i - one(I)]
            s = targets(sep)[Sp + i - one(I)]

            @simd for k in oneto(nrhs)
                C[k, Rp] += l * C[k, s]
            end
        end
    end

    return
end

# ================================ mul!! =================================

# C is a workspace. Returns B.
function mul!!(C::AbstractVecOrMat, F::AbstractFactorization, B::AbstractVecOrMat)
    ldiv!(B, F.P, lmul!(NaturalFactorization(F), mul!(C, F.P, B)))
end

# C is a workspace. Returns B.
function mul!!(C::AbstractVecOrMat, B::AbstractVecOrMat, F::AbstractFactorization)
    mul!(B, rmul!(rdiv!(C, B, F.P), NaturalFactorization(F)), F.P)
end
