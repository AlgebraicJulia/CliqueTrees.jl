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

# --- AbstractFactorization ---

function Base.:*(F::AbstractFactorization{DIAG, UPLO, T}, B::AbstractVecOrMat) where {DIAG, UPLO, T}
    return lmul!(F, Array{T}(B))
end

function Base.:*(B::AbstractMatrix, F::AbstractFactorization{DIAG, UPLO, T}) where {DIAG, UPLO, T}
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

function Base.:*(B::AdjVec, A::MaybeAdjOrTransTri{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    return rmul!(Matrix{T}(B), A)
end

function Base.:*(B::Adjoint{<:Any, <:AbstractZerosVector}, A::MaybeAdjOrTransTri{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    return Zeros{T}(1, size(A, 2))
end

function Base.:*(B::TransVec, A::MaybeAdjOrTransTri{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    return rmul!(Matrix{T}(B), A)
end

function Base.:*(B::Transpose{<:Any, <:AbstractZerosVector}, A::MaybeAdjOrTransTri{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    return Zeros{T}(1, size(A, 2))
end

# ================================ lmul! ================================

# --- AbstractFactorization ---

function LinearAlgebra.lmul!(F::NaturalFactorization{DIAG}, B::AbstractVecOrMat) where {DIAG}
    @assert size(F, 1) == size(B, 1)

    if DIAG === :N
        return lmul!(F.L, lmul!(F.U, B))
    else
        return lmul!(F.L, lmul!(F.D, lmul!(F.U, B)))
    end
end

function LinearAlgebra.lmul!(F::AbstractFactorization{DIAG, UPLO, T}, B::AbstractVecOrMat) where {DIAG, UPLO, T}
    @assert size(F, 1) == size(B, 1)
    C = FArray{T}(undef, size(B))
    return mul!!(C, F, B)
end

# --- ChordalTriangular ---

function LinearAlgebra.lmul!(A::MaybeAdjOrTransTri, B::AbstractVecOrMat)
    @assert size(A, 1) == size(B, 1)
    A, tA = unwrap(A)
    B, tB = unwrap(B)
    return mul_impl!(A.S, A.S.Dptr, A.Dval, A.S.Lptr, A.Lval, B, tA, tB, A.uplo, Val(:L), A.diag)
end

# ================================ rmul! ================================

# --- AbstractFactorization ---

function LinearAlgebra.rmul!(B::AbstractMatrix, F::NaturalFactorization{DIAG}) where {DIAG}
    @assert size(F, 1) == size(B, 2)

    if DIAG === :N
        return rmul!(rmul!(B, F.L), F.U)
    else
        return rmul!(rmul!(rmul!(B, F.L), F.D), F.U)
    end
end

function LinearAlgebra.rmul!(B::AbstractMatrix, F::AbstractFactorization{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    @assert size(F, 1) == size(B, 2)
    C = FMatrix{T}(undef, size(B))
    return mul!!(C, B, F)
end

# --- ChordalTriangular ---

function LinearAlgebra.rmul!(B::AbstractMatrix, A::MaybeAdjOrTransTri)
    @assert size(A, 1) == size(B, 2)
    A, tA = unwrap(A)
    B, tB = unwrap(B)
    return mul_impl!(A.S, A.S.Dptr, A.Dval, A.S.Lptr, A.Lval, B, tA, tB, A.uplo, Val(:R), A.diag)
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

# --- HermOrSymTri ---

function LinearAlgebra.mul!(Y::AbstractVecOrMat, A::HermOrSymTri{UPLO, T}, X::AbstractVecOrMat) where {UPLO, T}
    @assert size(A, 1) == size(X, 1) == size(Y, 1)
    @assert !(T <: Complex) || A isa Hermitian
    @assert A.uplo == char(parent(A).uplo)
    P = parent(A)
    X, tX = unwrap(X)
    return symm_impl!(P.S, P.S.Dptr, P.Dval, P.S.Lptr, P.Lval, X, Y, tX, P.uplo, Val(:L))
end

function LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, A::HermOrSymTri{UPLO, T}) where {UPLO, T}
    @assert size(A, 1) == size(X, 2) == size(Y, 2)
    @assert !(T <: Complex) || A isa Hermitian
    @assert A.uplo == char(parent(A).uplo)
    P = parent(A)
    X, tX = unwrap(X)
    return symm_impl!(P.S, P.S.Dptr, P.Dval, P.S.Lptr, P.Lval, X, Y, tX, P.uplo, Val(:R))
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

# ================================ mul!! =================================

# C is a workspace. Returns B.
function mul!!(C, F::AbstractFactorization, B)
    ldiv!(B, F.P, lmul!(NaturalFactorization(F), mul!(C, F.P, B)))
end

# C is a workspace. Returns B.
function mul!!(C, B, F::AbstractFactorization)
    mul!(B, rmul!(rdiv!(C, B, F.P), NaturalFactorization(F)), F.P)
end

# ============================= symm_impl! ===============================

function symm_impl!(
        S::ChordalSymbolic{I},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        X::AbstractVecOrMat,
        Y::AbstractVecOrMat,
        tX::Val{TX},
        uplo::Val{UPLO},
        side::Val{SIDE},
    ) where {T, I <: Integer, TX, UPLO, SIDE}

    res = S.res
    rel = S.rel
    chd = S.chd

    nMptr = S.nMptr
    nNval = S.nNval
    nFval = S.nFval

    if X isa AbstractVector
        nrhs = one(I)
    elseif SIDE === :L
        nrhs = convert(I, size(X, 2))
    else
        nrhs = convert(I, size(X, 1))
    end

    Mptr = FVector{I}(undef, nMptr)
    Mval = FVector{T}(undef, nNval * nrhs)
    Fval = FVector{T}(undef, nFval * nrhs)

    # Initialize Y to zero
    zerorec!(Y)

    ns = zero(I); Mptr[one(I)] = one(I)

    # Forward pass: lower triangle, y-updates upward
    for j in vertices(res)
        ns = symm_fwd_loop!(X, Y, Mptr, Mval, Dptr, Dval, Lptr, Lval, Fval, res, rel, chd, ns, j, tX, uplo, side)
    end

    # Backward pass: upper triangle, x-values downward
    for j in reverse(vertices(res))
        ns = symm_bwd_loop!(X, Y, Mptr, Mval, Lptr, Lval, Fval, res, rel, chd, ns, j, tX, uplo, side)
    end

    return Y
end

function symm_fwd_loop!(
        X::AbstractVecOrMat{T},
        Y::AbstractVecOrMat{T},
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
        tX::Val{TX},
        uplo::Val{UPLO},
        side::Val{SIDE},
    ) where {T, I <: Integer, TX, UPLO, SIDE}
    #
    # nrhs is the number of right-hand sides
    #
    if X isa AbstractVector
        nrhs = one(I)
    elseif SIDE === :L
        nrhs = convert(I, size(X, 2))
    else
        nrhs = convert(I, size(X, 1))
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
    if X isa AbstractVector
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
    # A₁₁ is the symmetric diagonal block
    # A₂₁ is the off-diagonal block
    #
    #        res(j)
    #   A = [ A₁₁ ] res(j)
    #       [ A₂₁ ] sep(j)
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    A₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        A₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        A₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    # X₁ is the input at res(j)
    # Y₁ is the output at res(j)
    #
    if X isa AbstractVector
        X₁ = view(X, neighbors(res, j))
        Y₁ = view(Y, neighbors(res, j))
    elseif SIDE === :L
        X₁ = view(X, neighbors(res, j), oneto(nrhs))
        Y₁ = view(Y, neighbors(res, j), oneto(nrhs))
    else
        X₁ = view(X, oneto(nrhs), neighbors(res, j))
        Y₁ = view(Y, oneto(nrhs), neighbors(res, j))
    end
    #
    #     F ← 0
    #
    zerorec!(F)

    for i in Iterators.reverse(neighbors(chd, j))
        #
        # assemble child message into F
        #
        #     F[rel(i)] += M(i)
        #
        div_fwd_update!(F, Mptr, Mval, rel, ns, i, side)
        ns -= one(I)
    end
    #
    #     Y₁ += A₁₁ * X₁ (symmetric)
    #
    if X isa AbstractVector
        symv!(uplo, one(T), A₁₁, X₁, one(T), Y₁)
    else
        symm!(side, uplo, one(T), A₁₁, X₁, one(T), Y₁)
    end
    #
    #     Y₁ += F₁ (assembled lower-triangle contributions from subtree)
    #
    addrec!(Y₁, F₁)

    if ispositive(na)
        ns += one(I)
        #
        # M₂ is the message to ancestor
        #
        #     M₂ = A₂₁ * X₁ + F₂
        #
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * nrhs

        if X isa AbstractVector
            M₂ = view(Mval, strt:stop - one(I))
        elseif SIDE === :L
            M₂ = reshape(view(Mval, strt:stop - one(I)), na, nrhs)
        else
            M₂ = reshape(view(Mval, strt:stop - one(I)), nrhs, na)
        end
        #
        #     M₂ ← F₂
        #
        copyrec!(M₂, F₂)
        #
        #     M₂ += A₂₁ * X₁
        #
        if X isa AbstractVector
            if UPLO === :L
                gemv!(Val(:N), one(T), A₂₁, X₁, one(T), M₂)
            else
                gemv!(Val(:C), one(T), A₂₁, X₁, one(T), M₂)
            end
        elseif SIDE === :L
            if UPLO === :L
                gemm!(Val(:N), tX, one(T), A₂₁, X₁, one(T), M₂)
            else
                gemm!(Val(:C), tX, one(T), A₂₁, X₁, one(T), M₂)
            end
        else
            if UPLO === :L
                gemm!(tX, Val(:C), one(T), X₁, A₂₁, one(T), M₂)
            else
                gemm!(tX, Val(:N), one(T), X₁, A₂₁, one(T), M₂)
            end
        end
    end

    return ns
end

function symm_bwd_loop!(
        X::AbstractVecOrMat{T},
        Y::AbstractVecOrMat{T},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        tX::Val{TX},
        uplo::Val{UPLO},
        side::Val{SIDE},
    ) where {T, I <: Integer, TX, UPLO, SIDE}
    #
    # nrhs is the number of right-hand sides
    #
    if X isa AbstractVector
        nrhs = one(I)
    elseif SIDE === :L
        nrhs = convert(I, size(X, 2))
    else
        nrhs = convert(I, size(X, 1))
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
    # A₂₁ is the off-diagonal block
    #
    Lp = Lptr[j]

    if UPLO === :L
        A₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        A₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    # X₁ is the input at res(j)
    # Y₁ is the output at res(j)
    #
    if X isa AbstractVector
        X₁ = view(X, neighbors(res, j))
        Y₁ = view(Y, neighbors(res, j))
    elseif SIDE === :L
        X₁ = view(X, neighbors(res, j), oneto(nrhs))
        Y₁ = view(Y, neighbors(res, j), oneto(nrhs))
    else
        X₁ = view(X, oneto(nrhs), neighbors(res, j))
        Y₁ = view(Y, oneto(nrhs), neighbors(res, j))
    end
    #
    # F is used to assemble x_bag = [X₁; N]
    #
    #        nrhs
    #   F = [ F₁ ] nn   ← X₁
    #       [ F₂ ] na   ← N (from parent)
    #
    if X isa AbstractVector
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
    #     F₁ ← X₁
    #
    copyrec!(F₁, X₁)
    #
    # Receive N from parent (x values at sep(j))
    #
    if ispositive(na)
        strt = Mptr[ns]

        if X isa AbstractVector
            N = view(Mval, strt:strt + na - one(I))
        elseif SIDE === :L
            N = reshape(view(Mval, strt:strt + na * nrhs - one(I)), na, nrhs)
        else
            N = reshape(view(Mval, strt:strt + na * nrhs - one(I)), nrhs, na)
        end

        ns -= one(I)
        #
        #     Y₁ += A₂₁ᴴ * N (upper triangle contribution)
        #
        if X isa AbstractVector
            if UPLO === :L
                gemv!(Val(:C), one(T), A₂₁, N, one(T), Y₁)
            else
                gemv!(Val(:N), one(T), A₂₁, N, one(T), Y₁)
            end
        elseif SIDE === :L
            if UPLO === :L
                gemm!(Val(:C), Val(:N), one(T), A₂₁, N, one(T), Y₁)
            else
                gemm!(Val(:N), Val(:N), one(T), A₂₁, N, one(T), Y₁)
            end
        else
            if UPLO === :L
                gemm!(Val(:N), Val(:N), one(T), N, A₂₁, one(T), Y₁)
            else
                gemm!(Val(:N), Val(:C), one(T), N, A₂₁, one(T), Y₁)
            end
        end
        #
        #     F₂ ← N (for x_bag assembly)
        #
        copyrec!(F₂, N)
    end
    #
    # Pass N(i) = x_bag[rel(i)] to each child
    #
    for i in neighbors(chd, j)
        ns += one(I)
        div_bwd_update!(F, Mptr, Mval, rel, ns, i, side)
    end

    return ns
end
