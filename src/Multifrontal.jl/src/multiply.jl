# ================================== dot ==================================

function LinearAlgebra.dot(A::AbstractVector, X::HermOrSymTri, B::AbstractVector)
    @assert size(X, 1) == size(A, 1) == size(B, 1)
    @assert checksymtri(X)
    P = parent(X)
    return symdot_impl!(P.S, P.S.Dptr, P.Dval, P.S.Lptr, P.Lval, A, B, P.uplo)
end

function LinearAlgebra.dot(A::AbstractMatrix, X::HermOrSymTri, B::AbstractMatrix)
    @assert size(X, 1) == size(A, 1) == size(B, 1)
    @assert size(A) == size(B)
    @assert checksymtri(X)
    P = parent(X)
    return symdot_impl!(P.S, P.S.Dptr, P.Dval, P.S.Lptr, P.Lval, A, B, P.uplo)
end

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

# --- HermOrSymTri ---

function LinearAlgebra.mul!(Y::AbstractVecOrMat, A::HermOrSymTri, X::AbstractVecOrMat)
    @assert size(A, 1) == size(X, 1) == size(Y, 1)
    @assert checksymtri(A)
    P = parent(A)
    X, tX = unwrap(X)
    return symm_impl!(P.S, P.S.Dptr, P.Dval, P.S.Lptr, P.Lval, X, Y, tX, P.uplo, Val(:L))
end

function LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, A::HermOrSymTri)
    @assert size(A, 1) == size(X, 2) == size(Y, 2)
    @assert checksymtri(A)
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

    R = promote_eltype(T, X, Y)

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
    Mval = FVector{R}(undef, nNval * nrhs)
    Fval = FVector{R}(undef, nFval * nrhs)

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
        X::AbstractVecOrMat,
        Y::AbstractVecOrMat,
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
        tX::Val{TX},
        uplo::Val{UPLO},
        side::Val{SIDE},
    ) where {T, R, I <: Integer, TX, UPLO, SIDE}
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
        symv!(uplo, 1, A₁₁, X₁, 1, Y₁)
    else
        symm!(side, uplo, 1, A₁₁, X₁, 1, Y₁)
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
                gemv!(Val(:N), 1, A₂₁, X₁, 1, M₂)
            else
                gemv!(Val(:C), 1, A₂₁, X₁, 1, M₂)
            end
        elseif SIDE === :L
            if UPLO === :L
                gemm!(Val(:N), tX, 1, A₂₁, X₁, 1, M₂)
            else
                gemm!(Val(:C), tX, 1, A₂₁, X₁, 1, M₂)
            end
        else
            if UPLO === :L
                gemm!(tX, Val(:C), 1, X₁, A₂₁, 1, M₂)
            else
                gemm!(tX, Val(:N), 1, X₁, A₂₁, 1, M₂)
            end
        end
    end

    return ns
end

function symm_bwd_loop!(
        X::AbstractVecOrMat,
        Y::AbstractVecOrMat,
        Mptr::AbstractVector{I},
        Mval::AbstractVector{R},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Fval::AbstractVector{R},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        tX::Val{TX},
        uplo::Val{UPLO},
        side::Val{SIDE},
    ) where {T, R, I <: Integer, TX, UPLO, SIDE}
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
                gemv!(Val(:C), 1, A₂₁, N, 1, Y₁)
            else
                gemv!(Val(:N), 1, A₂₁, N, 1, Y₁)
            end
        elseif SIDE === :L
            if UPLO === :L
                gemm!(Val(:C), Val(:N), 1, A₂₁, N, 1, Y₁)
            else
                gemm!(Val(:N), Val(:N), 1, A₂₁, N, 1, Y₁)
            end
        else
            if UPLO === :L
                gemm!(Val(:N), Val(:N), 1, N, A₂₁, 1, Y₁)
            else
                gemm!(Val(:N), Val(:C), 1, N, A₂₁, 1, Y₁)
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

# ============================= symdot_impl! ==============================

function symdot_impl!(
        S::ChordalSymbolic{I},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        A::AbstractVecOrMat,
        B::AbstractVecOrMat,
        uplo::Val{UPLO},
    ) where {T, I <: Integer, UPLO}

    R = promote_eltype(T, A, B)

    res = S.res
    rel = S.rel
    chd = S.chd

    nMptr = S.nMptr
    nNval = S.nNval
    nFval = S.nFval

    if B isa AbstractVector
        nrhs = one(I)
    else
        nrhs = convert(I, size(B, 2))
    end

    Mptr = FVector{I}(undef, nMptr)
    Mval = FVector{R}(undef, nNval * nrhs)
    Fval = FVector{R}(undef, nFval * nrhs)

    out = zero(R)

    ns = zero(I); Mptr[one(I)] = one(I)

    # Forward pass: lower triangle, B-messages upward
    for j in vertices(res)
        loc, ns = symdot_fwd_loop!(A, B, Mptr, Mval, Dptr, Dval, Lptr, Lval, Fval, res, rel, chd, ns, j, uplo)
        out += loc
    end

    # Backward pass: upper triangle, B-values downward
    for j in reverse(vertices(res))
        loc, ns = symdot_bwd_loop!(A, B, Mptr, Mval, Lptr, Lval, Fval, res, rel, chd, ns, j, uplo)
        out += loc
    end

    return out
end

function symdot_fwd_loop!(
        A::AbstractVecOrMat,
        B::AbstractVecOrMat,
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
        uplo::Val{UPLO},
    ) where {T, R, I <: Integer, UPLO}
    #
    # nrhs is the number of right-hand sides
    #
    if A isa AbstractVector
        nrhs = one(I)
    else
        nrhs = convert(I, size(A, 2))
    end
    #
    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    nn = eltypedegree(res, j)
    #
    # na is the size of the separator at node j
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(rel, j)
    #
    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na
    #
    # F is the frontal workspace
    #
    #         nrhs
    #     F = [ F₁ ] nn
    #         [ F₂ ] na
    #
    if A isa AbstractVector
        F = view(Fval, oneto(nj))
        F₁ = view(F, oneto(nn))
        F₂ = view(F, nn + one(I):nj)
    else
        F = reshape(view(Fval, oneto(nj * nrhs)), nj, nrhs)
        F₁ = view(F, oneto(nn), oneto(nrhs))
        F₂ = view(F, nn + one(I):nj, oneto(nrhs))
    end
    #
    # D₁₁ is the symmetric diagonal block
    # L₂₁ is the off-diagonal block
    #
    #         res(j)
    #     X = [ D₁₁ ] res(j)
    #         [ L₂₁ ] sep(j)
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
    # A₁, B₁ are the inputs at res(j)
    #
    if A isa AbstractVector
        A₁ = view(A, neighbors(res, j))
        B₁ = view(B, neighbors(res, j))
    else
        A₁ = view(A, neighbors(res, j), oneto(nrhs))
        B₁ = view(B, neighbors(res, j), oneto(nrhs))
    end
    #
    #     loc ← A₁ᴴ D₁₁ B₁
    #
    if A isa AbstractVector
        symv!(uplo, 1, D₁₁, B₁, 0, F₁)
    else
        symm!(Val(:L), uplo, 1, D₁₁, B₁, 0, F₁)
    end

    loc = dot(A₁, F₁)
    #
    #     F ← 0
    #
    zerorec!(F)

    for i in Iterators.reverse(neighbors(chd, j))
        #
        # assemble child message into F
        #
        #     F ← F + Rᵢ Mᵢ
        #
        div_fwd_update!(F, Mptr, Mval, rel, ns, i, Val(:L))
        ns -= one(I)
    end
    #
    #     loc ← loc + A₁ᴴ F₁
    #
    loc += dot(A₁, F₁)
    #
    # M₂ is the message to ancestor
    #
    if ispositive(na)
        ns += one(I)

        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * nrhs

        if A isa AbstractVector
            M₂ = view(Mval, strt:stop - one(I))
        else
            M₂ = reshape(view(Mval, strt:stop - one(I)), na, nrhs)
        end
        #
        #     M₂ ← F₂ + L₂₁ B₁
        #
        copyrec!(M₂, F₂)

        if A isa AbstractVector
            if UPLO === :L
                gemv!(Val(:N), 1, L₂₁, B₁, 1, M₂)
            else
                gemv!(Val(:C), 1, L₂₁, B₁, 1, M₂)
            end
        else
            if UPLO === :L
                gemm!(Val(:N), Val(:N), 1, L₂₁, B₁, 1, M₂)
            else
                gemm!(Val(:C), Val(:N), 1, L₂₁, B₁, 1, M₂)
            end
        end
    end

    return loc, ns
end

function symdot_bwd_loop!(
        A::AbstractVecOrMat,
        B::AbstractVecOrMat,
        Mptr::AbstractVector{I},
        Mval::AbstractVector{R},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        Fval::AbstractVector{R},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        uplo::Val{UPLO},
    ) where {T, R, I <: Integer, UPLO}
    #
    # nrhs is the number of right-hand sides
    #
    if A isa AbstractVector
        nrhs = one(I)
    else
        nrhs = convert(I, size(A, 2))
    end
    #
    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    nn = eltypedegree(res, j)
    #
    # na is the size of the separator at node j
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(rel, j)
    #
    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na
    #
    # L₂₁ is the off-diagonal block
    #
    Lp = Lptr[j]

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    # A₁, B₁ are the inputs at res(j)
    #
    if A isa AbstractVector
        A₁ = view(A, neighbors(res, j))
        B₁ = view(B, neighbors(res, j))
    else
        A₁ = view(A, neighbors(res, j), oneto(nrhs))
        B₁ = view(B, neighbors(res, j), oneto(nrhs))
    end
    #
    # F is used to assemble b_bag = [B₁; N]
    #
    #         nrhs
    #     F = [ F₁ ] nn   ← B₁
    #         [ F₂ ] na   ← N
    #
    if A isa AbstractVector
        F = view(Fval, oneto(nj))
        F₁ = view(F, oneto(nn))
        F₂ = view(F, nn + one(I):nj)
    else
        F = reshape(view(Fval, oneto(nj * nrhs)), nj, nrhs)
        F₁ = view(F, oneto(nn), oneto(nrhs))
        F₂ = view(F, nn + one(I):nj, oneto(nrhs))
    end
    #
    #     F₁ ← B₁
    #
    copyrec!(F₁, B₁)

    loc = zero(R)
    #
    # N is the message from ancestor
    #
    if ispositive(na)
        strt = Mptr[ns]

        if A isa AbstractVector
            N = view(Mval, strt:strt + na - one(I))
        else
            N = reshape(view(Mval, strt:strt + na * nrhs - one(I)), na, nrhs)
        end

        ns -= one(I)
        #
        #     loc ← A₁ᴴ L₂₁ᴴ N
        #
        if A isa AbstractVector
            if UPLO === :L
                gemv!(Val(:N), 1, L₂₁, A₁, 0, F₂)
            else
                gemv!(Val(:C), 1, L₂₁, A₁, 0, F₂)
            end
        else
            if UPLO === :L
                gemm!(Val(:N), Val(:N), 1, L₂₁, A₁, 0, F₂)
            else
                gemm!(Val(:C), Val(:N), 1, L₂₁, A₁, 0, F₂)
            end
        end

        loc = dot(F₂, N)
        #
        #     F₂ ← N
        #
        copyrec!(F₂, N)
    end

    for i in neighbors(chd, j)
        #
        # send message to child i
        #
        #     Mᵢ ← Rᵢᵀ F
        #
        ns += one(I)
        div_bwd_update!(F, Mptr, Mval, rel, ns, i, Val(:L))
    end

    return loc, ns
end
