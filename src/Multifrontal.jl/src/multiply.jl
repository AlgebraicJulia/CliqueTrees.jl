# ================================== * ==================================

# --- Permutation ---

function Base.:*(A::MaybeAdjOrTransPerm{I}, B::MaybeAdjOrTransPerm{I}) where {I}
    @assert size(A, 1) == size(B, 1)
    C = Permutation{I}(size(A, 1))
    return mul!(C, A, B)
end

function Base.:*(A::Permutation, B::SparseMatrixCSC)
    return permute(B, A.perm, axes(B, 2))
end

function Base.:*(A::SparseMatrixCSC, B::Permutation)
    return permute(A, axes(A, 1), invperm(B.perm))
end

function Base.:*(A::AdjOrTransPerm, B::SparseMatrixCSC)
    return parent(A) \ B
end

function Base.:*(A::SparseMatrixCSC, B::AdjOrTransPerm)
    return A / parent(B)
end

# --- ChordalFactorization ---

function Base.:*(F::ChordalFactorization{DIAG, UPLO, T}, B::AbstractVecOrMat) where {DIAG, UPLO, T}
    return lmul!(F, Array{T}(B))
end

function Base.:*(B::AbstractMatrix, F::ChordalFactorization{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    return rmul!(Matrix{T}(B), F)
end

# --- ChordalTriangular ---

function Base.:*(A::MaybeAdjOrTransTri{DIAG, UPLO, T}, B::AbstractVector) where {DIAG, UPLO, T}
    return lmul!(A, Vector{T}(B))
end

function Base.:*(A::MaybeAdjOrTransTri{DIAG, UPLO, T}, B::AbstractZerosVector) where {DIAG, UPLO, T}
    return Zeros{T}(size(A, 1))
end

function Base.:*(A::Transpose{T, ChordalTriangular{DIAG, UPLO, T, I, Val}}, B::AbstractZerosVector{T}) where {DIAG, UPLO, T <: Real, I, Val}
    return Zeros{T}(size(A, 1))
end

function Base.:*(A::MaybeAdjOrTransTri{DIAG, UPLO, T}, B::AbstractMatrix) where {DIAG, UPLO, T}
    return lmul!(A, Matrix{T}(B))
end

function Base.:*(A::MaybeAdjOrTransTri{DIAG, UPLO, T}, B::AbstractZerosMatrix) where {DIAG, UPLO, T}
    return Zeros{T}(size(A, 1), size(B, 2))
end

function Base.:*(A::MaybeAdjOrTransTri{DIAG, UPLO, T}, B::AdjOrTrans{<:Any, <:AbstractZerosVector}) where {DIAG, UPLO, T}
    return Zeros{T}(size(A, 1), size(B, 2))
end

function Base.:*(B::AbstractMatrix, A::MaybeAdjOrTransTri{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    return rmul!(Matrix{T}(B), A)
end

function Base.:*(B::AbstractZerosMatrix, A::MaybeAdjOrTransTri{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    return Zeros{T}(size(B, 1), size(A, 2))
end

function Base.:*(B::Adjoint{<:Any, <:AbstractVector}, A::MaybeAdjOrTransTri{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    return rmul!(Matrix{T}(B), A)
end

function Base.:*(B::Adjoint{<:Any, <:AbstractZerosVector}, A::MaybeAdjOrTransTri{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    return Zeros{T}(1, size(A, 2))
end

function Base.:*(B::Transpose{<:Any, <:AbstractVector}, A::MaybeAdjOrTransTri{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    return rmul!(Matrix{T}(B), A)
end

function Base.:*(B::Transpose{<:Any, <:AbstractZerosVector}, A::MaybeAdjOrTransTri{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    return Zeros{T}(1, size(A, 2))
end

# ================================ lmul! ================================

# --- ChordalFactorization ---

function LinearAlgebra.lmul!(F::ChordalFactorization{DIAG, UPLO, T}, B::AbstractVecOrMat) where {DIAG, UPLO, T}
    @assert size(F, 1) == size(B, 1)
    C = FArray{T}(undef, size(B))

    if DIAG === :N
        return ldiv!(B, F.P, lmul!(F.L, lmul!(F.U, mul!(C, F.P, B))))
    else
        return ldiv!(B, F.P, lmul!(F.L, lmul!(F.D, lmul!(F.U, mul!(C, F.P, B)))))
    end
end

# --- ChordalTriangular ---

function LinearAlgebra.lmul!(A::MaybeAdjOrTransTri{DIAG, UPLO}, B::AbstractVecOrMat) where {DIAG, UPLO}
    @assert size(A, 1) == size(B, 1)
    A, tA = unwrap(A)
    B, tB = unwrap(B)
    return mul_impl!(A.S, A.S.Dptr, A.Dval, A.S.Lptr, A.Lval, B, tA, tB, Val(UPLO), Val(:L), Val(DIAG))
end

# ================================ rmul! ================================

# --- ChordalFactorization ---

function LinearAlgebra.rmul!(B::AbstractMatrix, F::ChordalFactorization{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    @assert size(F, 1) == size(B, 2)
    C = FMatrix{T}(undef, size(B))

    if DIAG === :N
        return mul!(B, rmul!(rmul!(rdiv!(C, B, F.P), F.L), F.U), F.P)
    else
        return mul!(B, rmul!(rmul!(rmul!(rdiv!(C, B, F.P), F.L), F.D), F.U), F.P)
    end
end

# --- ChordalTriangular ---

function LinearAlgebra.rmul!(B::AbstractMatrix, A::MaybeAdjOrTransTri{DIAG, UPLO}) where {DIAG, UPLO}
    @assert size(A, 1) == size(B, 2)
    A, tA = unwrap(A)
    B, tB = unwrap(B)
    return mul_impl!(A.S, A.S.Dptr, A.Dval, A.S.Lptr, A.Lval, B, tA, tB, Val(UPLO), Val(:R), Val(DIAG))
end

# ================================= mul! ================================

# --- ChordalFactorization ---

function LinearAlgebra.mul!(C::AbstractVecOrMat, F::ChordalFactorization, B::AbstractVecOrMat)
    lmul!(F, copyrec!(C, B))
end

# --- ChordalTriangular ---

function LinearAlgebra.mul!(C::AbstractVecOrMat, A::MaybeAdjOrTransTri, B::AbstractVecOrMat)
    lmul!(A, copyrec!(C, B))
end

# --- Permutation ---

function LinearAlgebra.mul!(C::AbstractVecOrMat, A::Permutation, B::AbstractVecOrMat)
    @boundscheck size(C, 1) == size(B, 1) == size(A, 1) || throw(DimensionMismatch())

    @inbounds for j in axes(C, 2)
        for i in axes(C, 1)
            C[i, j] = B[A.perm[i], j]
        end
    end

    return C
end

function LinearAlgebra.mul!(C::AbstractVecOrMat, A::AdjOrTransPerm, B::AbstractVecOrMat)
    return ldiv!(C, parent(A), B)
end

function LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, B::Permutation)
    @boundscheck size(C, 2) == size(A, 2) == size(B, 1) || throw(DimensionMismatch())

    @inbounds for j in axes(C, 2)
        for i in axes(C, 1)
            C[i, B.perm[j]] = A[i, j]
        end
    end

    return C
end

function LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, B::AdjOrTransPerm)
    return rdiv!(C, A, parent(B))
end

function LinearAlgebra.mul!(C::Permutation, A::Permutation, B::Permutation)
    @boundscheck size(C, 2) == size(A, 2) == size(B, 1) || throw(DimensionMismatch())

    @inbounds for i in axes(C, 1)
        C.perm[i] = A.perm[B.perm[i]]
    end

    return C
end

function LinearAlgebra.mul!(C::Permutation, A::AdjOrTransPerm, B::AdjOrTransPerm)
    @boundscheck size(C, 2) == size(A, 2) == size(B, 1) || throw(DimensionMismatch())

    @inbounds for i in axes(C, 1)
        C.perm[parent(B).perm[parent(A).perm[i]]] = i
    end

    return C
end

function LinearAlgebra.mul!(C::AbstractMatrix, A::AdjOrTransPerm, B::Permutation)
    return ldiv!(C, parent(A), B)
end

function LinearAlgebra.mul!(C::AbstractMatrix, A::Permutation, B::AdjOrTransPerm)
    return rdiv!(C, A, parent(B))
end

function LinearAlgebra.mul!(C::AbstractMatrix{T}, A::Permutation, B::Permutation) where {T}
    @boundscheck size(C, 1) == size(C, 2) == size(A, 1) == size(B, 1) || throw(DimensionMismatch())
    fill!(C, zero(T))

    @inbounds for i in axes(C, 1)
        C[i, A.perm[B.perm[i]]] = one(T)
    end

    return C
end

function LinearAlgebra.mul!(C::AbstractMatrix{T}, A::AdjOrTransPerm, B::AdjOrTransPerm) where {T}
    @boundscheck size(C, 1) == size(C, 2) == size(A, 1) == size(B, 1) || throw(DimensionMismatch())
    fill!(C, zero(T))

    @inbounds for i in axes(C, 1)
        C[parent(B).perm[parent(A).perm[i]], i] = one(T)
    end

    return C
end

# ============================== mul_impl! ==============================

function mul_impl!(
        S::ChordalSymbolic{I},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        B::AbstractVecOrMat,
        tA::Val{TA},
        tB::Val{TB},
        uplo::Val{UPLO},
        side::Val{SIDE},
        diag::Val{DIAG},
    ) where {T, I <: Integer, TA, TB, UPLO, SIDE, DIAG}

    res = S.res
    rel = S.rel
    chd = S.chd

    nMptr = S.nMptr
    nNval = S.nNval
    nFval = S.nFval

    if B isa AbstractVector
        nrhs = one(I)
    elseif SIDE === :L
        nrhs = convert(I, size(B, 2))
    else
        nrhs = convert(I, size(B, 1))
    end

    Mptr = FVector{I}(undef, nMptr)
    Mval = FVector{T}(undef, nNval * nrhs)
    Fval = FVector{T}(undef, nFval * nrhs)

    ns = zero(I); Mptr[one(I)] = one(I)

    # forward: L from left, or L' from right (for lower)
    #          U' from left, or U from right (for upper)
    # backward: L' from left, or L from right (for lower)
    #           U from left, or U' from right (for upper)
    if isforward(UPLO, TA, SIDE)
        for j in vertices(res)
            ns = mul_fwd_loop!(B, Mptr, Mval, Dptr, Dval, Lptr, Lval, Fval, res, rel, chd, ns, j, tA, tB, uplo, side, diag)
        end
    else
        for j in reverse(vertices(res))
            ns = mul_bwd_loop!(B, Mptr, Mval, Dptr, Dval, Lptr, Lval, Fval, res, rel, chd, ns, j, tA, tB, uplo, side, diag)
        end
    end

    return B
end

function mul_fwd_loop!(
        C::AbstractVecOrMat{T},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        tA::Val{TA},
        tB::Val{TB},
        uplo::Val{UPLO},
        side::Val{SIDE},
        diag::Val{DIAG},
    ) where {T, I <: Integer, TA, TB, UPLO, SIDE, DIAG}
    #
    # nrhs is the number of right-hand sides
    #
    if C isa AbstractVector
        nrhs = one(I)
    elseif SIDE === :L
        nrhs = convert(I, size(C, 2))
    else
        nrhs = convert(I, size(C, 1))
    end
    #
    # nn is the size of the residual at node j
    #
    #     nn = |res(j)|
    #
    nn = eltypedegree(res, j)
    #
    # na is the size of the separator at node j
    #
    #     na = |sep(j)|
    #
    na = eltypedegree(rel, j)
    #
    # nj is the size of the bag at node j
    #
    #     nj = |bag(j)|
    #
    nj = nn + na
    #
    # F is the frontal matrix at node j
    #
    #        nrhs
    #   F = [ F₁ ] nn
    #       [ F₂ ] na
    #
    if C isa AbstractVector
        F = view(Fval, oneto(nj))
        F₁ = view(F, oneto(nn))
        F₂ = view(F, nn + one(I):nj)
    elseif SIDE === :L
        F = reshape(view(Fval, oneto(nj * nrhs)), nj, nrhs)
        F₁ = view(F, oneto(nn), oneto(nrhs))
        F₂ = view(F, nn + one(I):nj, oneto(nrhs))
    else
        F = reshape(view(Fval, oneto(nj * nrhs)), nrhs, nj)
        F₁ = view(F, oneto(nrhs), oneto(nn))
        F₂ = view(F, oneto(nrhs), nn + one(I):nj)
    end
    #
    # B is part of the L factor
    #
    #        res(j)
    #   B = [ D₁₁ ] res(j)
    #       [ L₂₁ ] sep(j)
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    # C₁ is part of the right-hand side
    #
    #        nrhs
    #   C = [ C₁ ] res(j)
    #
    if C isa AbstractVector
        C₁ = view(C, neighbors(res, j))
    elseif SIDE === :L
        C₁ = view(C, neighbors(res, j), oneto(nrhs))
    else
        C₁ = view(C, oneto(nrhs), neighbors(res, j))
    end
    #
    #     F ← 0
    #
    zerorec!(F)

    for i in Iterators.reverse(neighbors(chd, j))
        #
        # add the update matrix for child i to F
        #
        #     F ← F + Rᵢ Mᵢ
        #
        div_fwd_update!(F, Mptr, Mval, rel, ns, i, side)
        ns -= one(I)
    end

    if ispositive(na)
        ns += one(I)
        #
        # M₂ is the update matrix for node j
        #
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * nrhs

        if C isa AbstractVector
            M₂ = view(Mval, strt:stop - one(I))
        elseif SIDE === :L
            M₂ = reshape(view(Mval, strt:stop - one(I)), na, nrhs)
        else
            M₂ = reshape(view(Mval, strt:stop - one(I)), nrhs, na)
        end
        #
        #     M₂ ← L₂₁ C₁ + F₂
        #
        copyrec!(M₂, F₂)

        if C isa AbstractVector
            if UPLO === :L
                gemv!(Val(:N), one(T), L₂₁, C₁, one(T), M₂)
            else
                gemv!(tA, one(T), L₂₁, C₁, one(T), M₂)
            end
        elseif SIDE === :L
            if UPLO === :L
                gemm!(Val(:N), tB, one(T), L₂₁, C₁, one(T), M₂)
            else
                gemm!(tA, tB, one(T), L₂₁, C₁, one(T), M₂)
            end
        else
            if UPLO === :L
                gemm!(tB, tA, one(T), C₁, L₂₁, one(T), M₂)
            else
                gemm!(tB, Val(:N), one(T), C₁, L₂₁, one(T), M₂)
            end
        end
    end
    #
    #     C₁ ← D₁₁ C₁
    #
    if C isa AbstractVector
        trmv!(uplo, tA, diag, D₁₁, C₁)
    else
        trmm!(side, uplo, tA, diag, one(T), D₁₁, C₁)
    end
    #
    #     C₁ ← C₁ + F₁
    #
    addrec!(C₁, F₁)

    return ns
end

function mul_bwd_loop!(
        C::AbstractVecOrMat{T},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        tA::Val{TA},
        tB::Val{TB},
        uplo::Val{UPLO},
        side::Val{SIDE},
        diag::Val{DIAG},
    ) where {T, I <: Integer, TA, TB, UPLO, SIDE, DIAG}
    #
    # nrhs is the number of right-hand sides
    #
    if C isa AbstractVector
        nrhs = one(I)
    elseif SIDE === :L
        nrhs = convert(I, size(C, 2))
    else
        nrhs = convert(I, size(C, 1))
    end
    #
    # nn is the size of the residual at node j
    #
    #     nn = |res(j)|
    #
    nn = eltypedegree(res, j)
    #
    # na is the size of the separator at node j
    #
    #     na = |sep(j)|
    #
    na = eltypedegree(rel, j)
    #
    # nj is the size of the bag at node j
    #
    #     nj = |bag(j)|
    #
    nj = nn + na
    #
    # B is part of the L factor
    #
    #        res(j)
    #   B = [ D₁₁ ] res(j)
    #       [ L₂₁ ] sep(j)
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    # C₁ is part of the right-hand side
    #
    #        nrhs
    #   C = [ C₁ ] res(j)
    #
    if C isa AbstractVector
        C₁ = view(C, neighbors(res, j))
    elseif SIDE === :L
        C₁ = view(C, neighbors(res, j), oneto(nrhs))
    else
        C₁ = view(C, oneto(nrhs), neighbors(res, j))
    end
    #
    # F is the frontal matrix at node j
    #
    #        nrhs
    #   F = [ F₁ ] nn
    #       [ F₂ ] na
    #
    if C isa AbstractVector
        F = view(Fval, oneto(nj))
        F₁ = view(F, oneto(nn))
        F₂ = view(F, nn + one(I):nj)
    elseif SIDE === :L
        F = reshape(view(Fval, oneto(nj * nrhs)), nj, nrhs)
        F₁ = view(F, oneto(nn), oneto(nrhs))
        F₂ = view(F, nn + one(I):nj, oneto(nrhs))
    else
        F = reshape(view(Fval, oneto(nj * nrhs)), nrhs, nj)
        F₁ = view(F, oneto(nrhs), oneto(nn))
        F₂ = view(F, oneto(nrhs), nn + one(I):nj)
    end
    #
    #     F₁ ← C₁
    #
    copyrec!(F₁, C₁)
    #
    #     C₁ ← D₁₁ᴴ C₁
    #
    if C isa AbstractVector
        trmv!(uplo, tA, diag, D₁₁, C₁)
    else
        trmm!(side, uplo, tA, diag, one(T), D₁₁, C₁)
    end
    #
    # add the update matrix from ancestor to C₁
    #
    #     C₁ ← C₁ + L₂₁ᴴ M₂
    #
    if ispositive(na)
        strt = Mptr[ns]

        if C isa AbstractVector
            M₂ = view(Mval, strt:strt + na - one(I))
        elseif SIDE === :L
            M₂ = reshape(view(Mval, strt:strt + na * nrhs - one(I)), na, nrhs)
        else
            M₂ = reshape(view(Mval, strt:strt + na * nrhs - one(I)), nrhs, na)
        end

        ns -= one(I)

        if C isa AbstractVector
            if UPLO === :L
                gemv!(tA, one(T), L₂₁, M₂, one(T), C₁)
            else
                gemv!(Val(:N), one(T), L₂₁, M₂, one(T), C₁)
            end
        elseif SIDE === :L
            if UPLO === :L
                gemm!(tA, Val(:N), one(T), L₂₁, M₂, one(T), C₁)
            else
                gemm!(Val(:N), Val(:N), one(T), L₂₁, M₂, one(T), C₁)
            end
        else
            if UPLO === :L
                gemm!(Val(:N), Val(:N), one(T), M₂, L₂₁, one(T), C₁)
            else
                gemm!(Val(:N), tA, one(T), M₂, L₂₁, one(T), C₁)
            end
        end
        #
        #     F₂ ← M₂
        #
        copyrec!(F₂, M₂)
    end

    for i in neighbors(chd, j)
        #
        # push F restricted to sep(i) to child i
        #
        #     Mᵢ ← Rᵢᵀ F
        #
        ns += one(I)
        div_bwd_update!(F, Mptr, Mval, rel, ns, i, side)
    end

    return ns
end
