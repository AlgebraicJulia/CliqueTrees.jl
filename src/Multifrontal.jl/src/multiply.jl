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

function LinearAlgebra.lmul!(A::MaybeAdjOrTransTri, B::AbstractVecOrMat)
    @assert size(A, 1) == size(B, 1)
    A, tA = unwrap(A)
    B, tB = unwrap(B)
    return mul_impl!(A.S, A.S.Dptr, A.Dval, A.S.Lptr, A.Lval, B, tA, tB, A.uplo, Val(:L), A.diag)
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

function mul_impl!(
        S::ChordalSymbolic{I},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        B::AbstractVecOrMat{R},
        tA::Val{TA},
        tB::Val{TB},
        uplo::Val{UPLO},
        side::Val{SIDE},
        diag::Val{DIAG},
    ) where {T, R, I <: Integer, TA, TB, UPLO, SIDE, DIAG}

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
    Mval = FVector{R}(undef, nNval * nrhs)
    Fval = FVector{R}(undef, nFval * nrhs)

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
        C::AbstractVecOrMat{R},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{R},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Fval::AbstractVector{R},
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
    ) where {T, R, I <: Integer, TA, TB, UPLO, SIDE, DIAG}
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
                gemv!(Val(:N), 1, L₂₁, C₁, 1, M₂)
            else
                gemv!(tA, 1, L₂₁, C₁, 1, M₂)
            end
        elseif SIDE === :L
            if UPLO === :L
                gemm!(Val(:N), tB, 1, L₂₁, C₁, 1, M₂)
            else
                gemm!(tA, tB, 1, L₂₁, C₁, 1, M₂)
            end
        else
            if UPLO === :L
                gemm!(tB, tA, 1, C₁, L₂₁, 1, M₂)
            else
                gemm!(tB, Val(:N), 1, C₁, L₂₁, 1, M₂)
            end
        end
    end
    #
    #     C₁ ← D₁₁ C₁
    #
    if C isa AbstractVector
        trmv!(uplo, tA, diag, D₁₁, C₁)
    else
        trmm!(side, uplo, tA, diag, 1, D₁₁, C₁)
    end
    #
    #     C₁ ← C₁ + F₁
    #
    addrec!(C₁, F₁)

    return ns
end

function mul_bwd_loop!(
        C::AbstractVecOrMat{R},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{R},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Fval::AbstractVector{R},
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
    ) where {T, R, I <: Integer, TA, TB, UPLO, SIDE, DIAG}
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
        trmm!(side, uplo, tA, diag, 1, D₁₁, C₁)
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
                gemv!(tA, 1, L₂₁, M₂, 1, C₁)
            else
                gemv!(Val(:N), 1, L₂₁, M₂, 1, C₁)
            end
        elseif SIDE === :L
            if UPLO === :L
                gemm!(tA, Val(:N), 1, L₂₁, M₂, 1, C₁)
            else
                gemm!(Val(:N), Val(:N), 1, L₂₁, M₂, 1, C₁)
            end
        else
            if UPLO === :L
                gemm!(Val(:N), Val(:N), 1, M₂, L₂₁, 1, C₁)
            else
                gemm!(Val(:N), tA, 1, M₂, L₂₁, 1, C₁)
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
function mul!!(C::AbstractVecOrMat, F::AbstractFactorization, B::AbstractVecOrMat)
    ldiv!(B, F.P, lmul!(NaturalFactorization(F), mul!(C, F.P, B)))
end

# C is a workspace. Returns B.
function mul!!(C::AbstractVecOrMat, B::AbstractVecOrMat, F::AbstractFactorization)
    mul!(B, rmul!(rdiv!(C, B, F.P), NaturalFactorization(F)), F.P)
end
