# ================================== \ ==================================

# --- Permutation ---

function Base.:\(A::Permutation, b::AbstractVector)
    return ldiv!(Array(b), A, b)
end

function Base.:\(A::AdjOrTransPerm, b::AbstractVector)
    return ldiv!(Array(b), A, b)
end

function Base.:\(A::Permutation, B::SparseMatrixCSC)
    return permute(B, invperm(A.perm), axes(B, 2))
end

function Base.:\(A::AdjOrTransPerm, B::SparseMatrixCSC)
    return parent(A) * B
end

# --- ChordalTriangular ---

function Base.:\(A::MaybeAdjOrTransTri{DIAG, UPLO, T}, B::AbstractVecOrMat) where {DIAG, UPLO, T}
    return ldiv!(A, Array{T}(B))
end

# ================================== / ==================================

# --- Permutation ---

function Base.:/(A::AbstractMatrix, B::MaybeAdjOrTransPerm)
    return rdiv!(Array(A), A, B)
end

function Base.:/(A::Transpose{<:Any, <:AbstractVector}, B::MaybeAdjOrTransPerm)
    return transpose(transpose(B) \ parent(A))
end

function Base.:/(A::Transpose{<:Any, <:AbstractVector}, B::Adjoint{Bool, <:Permutation})
    return transpose(transpose(B) \ parent(A))
end

function Base.:/(A::Adjoint{<:Any, <:AbstractVector}, B::MaybeAdjOrTransPerm)
    return adjoint(adjoint(B) \ parent(A))
end

function Base.:/(A::Adjoint{<:Any, <:AbstractVector}, B::Transpose{Bool, <:Permutation})
    return adjoint(adjoint(B) \ parent(A))
end

function Base.:/(A::SparseMatrixCSC, B::Permutation)
    return permute(A, axes(A, 1), B.perm)
end

function Base.:/(A::SparseMatrixCSC, B::AdjOrTransPerm)
    return A * parent(B)
end

# --- ChordalTriangular ---

function Base.:/(B::AbstractMatrix, A::MaybeAdjOrTransTri{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    return rdiv!(Matrix{T}(B), A)
end

function Base.:/(A::Transpose{<:Any, <:AbstractVector}, B::MaybeAdjOrTransTri{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    return transpose(transpose(B) \ parent(A))
end

function Base.:/(A::Adjoint{<:Any, <:AbstractVector}, B::MaybeAdjOrTransTri{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    return adjoint(adjoint(B) \ parent(A))
end

# ================================ ldiv! ================================

# --- Permutation ---

function LinearAlgebra.ldiv!(C::AbstractVecOrMat, A::Permutation, B::AbstractVecOrMat)
    @boundscheck size(C, 1) == size(B, 1) == size(A, 1) || throw(DimensionMismatch())

    @inbounds for j in axes(C, 2)
        for i in axes(C, 1)
            C[A.perm[i], j] = B[i, j]
        end
    end

    return C
end

function LinearAlgebra.ldiv!(C::Permutation, A::Permutation, B::Permutation)
    @boundscheck size(C, 1) == size(B, 1) == size(A, 1) || throw(DimensionMismatch())

    @inbounds for i in axes(C, 1)
        C.perm[A.perm[i]] = B.perm[i]
    end

    return C
end

function LinearAlgebra.ldiv!(C::AbstractVecOrMat, A::AdjOrTransPerm, B::AbstractVecOrMat)
    return mul!(C, parent(A), B)
end

# --- ChordalFactorization ---

function LinearAlgebra.ldiv!(F::ChordalFactorization{DIAG, UPLO, T}, B::AbstractVecOrMat) where {DIAG, UPLO, T}
    @assert size(F, 1) == size(B, 1)
    C = FArray{T}(undef, size(B))

    if DIAG === :N
        return ldiv!(B, F.P, ldiv!(F.U, ldiv!(F.L, mul!(C, F.P, B))))
    else
        return ldiv!(B, F.P, ldiv!(F.U, ldiv!(F.D, ldiv!(F.L, mul!(C, F.P, B)))))
    end
end

# --- ChordalTriangular ---

function LinearAlgebra.ldiv!(A::MaybeAdjOrTransTri{DIAG, UPLO}, B::AbstractVecOrMat) where {DIAG, UPLO}
    @assert size(A, 1) == size(B, 1)
    A, tA = unwrap(A)
    B, tB = unwrap(B)
    return div_impl!(A.S, A.S.Dptr, A.Dval, A.S.Lptr, A.Lval, B, tA, tB, Val(UPLO), Val(:L), Val(DIAG))
end

# ================================ rdiv! ================================

# --- Permutation ---

function LinearAlgebra.rdiv!(C::AbstractMatrix, A::AbstractMatrix, B::Permutation)
    @boundscheck size(C, 2) == size(A, 2) == size(B, 1) || throw(DimensionMismatch())

    @inbounds for j in axes(C, 2)
        for i in axes(C, 1)
            C[i, j] = A[i, B.perm[j]]
        end
    end

    return C
end

function LinearAlgebra.rdiv!(C::Permutation, A::Permutation, B::Permutation)
    @boundscheck size(C, 2) == size(A, 2) == size(B, 1) || throw(DimensionMismatch())

    @inbounds for i in axes(C, 1)
        C.perm[B.perm[i]] = A.perm[i]
    end

    return C
end

function LinearAlgebra.rdiv!(C::AbstractMatrix, A::AbstractMatrix, B::AdjOrTransPerm)
    return mul!(C, A, parent(B))
end

# --- ChordalFactorization ---

function LinearAlgebra.rdiv!(B::AbstractMatrix, F::ChordalFactorization{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    @assert size(F, 1) == size(B, 2)
    C = FMatrix{T}(undef, size(B))

    if DIAG === :N
        return mul!(B, rdiv!(rdiv!(rdiv!(C, B, F.P), F.U), F.L), F.P)
    else
        return mul!(B, rdiv!(rdiv!(rdiv!(rdiv!(C, B, F.P), F.U), F.D), F.L), F.P)
    end
end

# --- ChordalTriangular ---

function LinearAlgebra.rdiv!(B::AbstractMatrix, A::MaybeAdjOrTransTri{DIAG, UPLO}) where {DIAG, UPLO}
    @assert size(A, 1) == size(B, 2)
    A, tA = unwrap(A)
    B, tB = unwrap(B)
    return div_impl!(A.S, A.S.Dptr, A.Dval, A.S.Lptr, A.Lval, B, tA, tB, Val(UPLO), Val(:R), Val(DIAG))
end

# ============================== div_impl! ==============================

function div_impl!(
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

    if isforward(UPLO, TA, SIDE)
        for j in vertices(res)
            ns = div_fwd_loop!(B, Mptr, Mval, Dptr, Dval, Lptr, Lval, Fval, res, rel, chd, ns, j, tA, tB, uplo, side, diag)
        end
    else
        for j in reverse(vertices(res))
            ns = div_bwd_loop!(B, Mptr, Mval, Dptr, Dval, Lptr, Lval, Fval, res, rel, chd, ns, j, tA, tB, uplo, side, diag)
        end
    end

    return B
end

function div_fwd_loop!(
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
    #
    #     C₁ ← C₁ + F₁
    #
    addrec!(C₁, F₁)
    #
    #     C₁ ← D₁₁⁻¹ C₁
    #
    if C isa AbstractVector
        trsv!(uplo, tA, diag, D₁₁, C₁)
    else
        trsm!(side, uplo, tA, diag, one(T), D₁₁, C₁)
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
        #     M₂ ← F₂ - L₂₁ C₁
        #
        copyrec!(M₂, F₂)

        if C isa AbstractVector
            if UPLO === :L
                gemv!(Val(:N), -one(T), L₂₁, C₁, one(T), M₂)
            else
                gemv!(tA, -one(T), L₂₁, C₁, one(T), M₂)
            end
        elseif SIDE === :L
            if UPLO === :L
                gemm!(Val(:N), tB, -one(T), L₂₁, C₁, one(T), M₂)
            else
                gemm!(tA, tB, -one(T), L₂₁, C₁, one(T), M₂)
            end
        else
            if UPLO === :L
                gemm!(tB, tA, -one(T), C₁, L₂₁, one(T), M₂)
            else
                gemm!(tB, Val(:N), -one(T), C₁, L₂₁, one(T), M₂)
            end
        end
    end

    return ns
end

function div_bwd_loop!(
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
    # subtract the update matrix from ancestor
    #
    #     C₁ ← C₁ - L₂₁ᴴ M₂
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
                gemv!(tA, -one(T), L₂₁, M₂, one(T), C₁)
            else
                gemv!(Val(:N), -one(T), L₂₁, M₂, one(T), C₁)
            end
        elseif SIDE === :L
            if UPLO === :L
                gemm!(tA, Val(:N), -one(T), L₂₁, M₂, one(T), C₁)
            else
                gemm!(Val(:N), Val(:N), -one(T), L₂₁, M₂, one(T), C₁)
            end
        else
            if UPLO === :L
                gemm!(Val(:N), Val(:N), -one(T), M₂, L₂₁, one(T), C₁)
            else
                gemm!(Val(:N), tA, -one(T), M₂, L₂₁, one(T), C₁)
            end
        end
    end
    #
    #     C₁ ← D₁₁⁻ᴴ C₁
    #
    if C isa AbstractVector
        trsv!(uplo, tA, diag, D₁₁, C₁)
    else
        trsm!(side, uplo, tA, diag, one(T), D₁₁, C₁)
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
    #     F₂ ← M₂
    #
    if ispositive(na)
        strt = Mptr[ns + one(I)]
        if C isa AbstractVector
            M₂ = view(Mval, strt:strt + na - one(I))
        elseif SIDE === :L
            M₂ = reshape(view(Mval, strt:strt + na * nrhs - one(I)), na, nrhs)
        else
            M₂ = reshape(view(Mval, strt:strt + na * nrhs - one(I)), nrhs, na)
        end
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

function div_fwd_update!(
        F::AbstractVecOrMat{T},
        ptr::AbstractVector{I},
        val::AbstractVector{T},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
        side::Val{SIDE},
    ) where {T, I <: Integer, SIDE}

    if F isa AbstractVector
        nrhs = one(I)
    elseif SIDE === :L
        nrhs = convert(I, size(F, 2))
    else
        nrhs = convert(I, size(F, 1))
    end
    #
    #     na = |sep(i)|
    #
    na = eltypedegree(rel, i)
    #
    #     inj: sep(i) → bag(parent(i))
    #
    inj = neighbors(rel, i)
    #
    #     F ← F + Rᵢ Mᵢ
    #
    strt = ptr[ns]

    if F isa AbstractVector
        M = view(val, strt:strt + na - one(I))
        addrec!(F, M, inj)
    elseif SIDE === :L
        M = reshape(view(val, strt:strt + na * nrhs - one(I)), na, nrhs)
        addrec!(F, M, inj, oneto(nrhs))
    else
        M = reshape(view(val, strt:strt + na * nrhs - one(I)), nrhs, na)
        addrec!(F, M, oneto(nrhs), inj)
    end

    return
end

function div_bwd_update!(
        F::AbstractVecOrMat{T},
        ptr::AbstractVector{I},
        val::AbstractVector{T},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
        side::Val{SIDE},
    ) where {T, I <: Integer, SIDE}

    if F isa AbstractVector
        nrhs = one(I)
    elseif SIDE === :L
        nrhs = convert(I, size(F, 2))
    else
        nrhs = convert(I, size(F, 1))
    end
    #
    #     na = |sep(i)|
    #
    na = eltypedegree(rel, i)
    #
    #     inj: sep(i) → bag(parent(i))
    #
    inj = neighbors(rel, i)
    #
    #     Mᵢ ← Rᵢᵀ F
    #
    strt = ptr[ns]
    stop = ptr[ns + one(I)] = strt + na * nrhs

    if F isa AbstractVector
        M = view(val, strt:stop - one(I))
        copyrec!(M, F, inj)
    elseif SIDE === :L
        M = reshape(view(val, strt:stop - one(I)), na, nrhs)
        copyrec!(M, F, inj, oneto(nrhs))
    else
        M = reshape(view(val, strt:stop - one(I)), nrhs, na)
        copyrec!(M, F, oneto(nrhs), inj)
    end

    return
end
