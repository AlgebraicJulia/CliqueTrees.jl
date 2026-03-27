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

function LinearAlgebra.ldiv!(A::MaybeAdjOrTransTri, B::AbstractVecOrMat)
    @assert size(A, 1) == size(B, 1)
    A, tA = unwrap(A)
    B, tB = unwrap(B)
    return div_impl!(A.S, A.S.Dptr, A.Dval, A.S.Lptr, A.Lval, B, tA, tB, A.uplo, Val(:L), A.diag)
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

function LinearAlgebra.ldiv!(F::NaturalFactorization{DIAG}, B::AbstractVecOrMat) where {DIAG}
    @assert size(F, 1) == size(B, 1)

    if DIAG === :N
        return ldiv!(F.U, ldiv!(F.L, B))
    else
        return ldiv!(F.U, ldiv!(F.D, ldiv!(F.L, B)))
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

function LinearAlgebra.rdiv!(B::AbstractMatrix, A::MaybeAdjOrTransTri)
    @assert size(A, 1) == size(B, 2)
    A, tA = unwrap(A)
    B, tB = unwrap(B)
    return div_impl!(A.S, A.S.Dptr, A.Dval, A.S.Lptr, A.Lval, B, tA, tB, A.uplo, Val(:R), A.diag)
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

function LinearAlgebra.rdiv!(B::AbstractMatrix, F::NaturalFactorization{DIAG}) where {DIAG}
    @assert size(F, 1) == size(B, 2)

    if DIAG === :N
        return rdiv!(rdiv!(B, F.U), F.L)
    else
        return rdiv!(rdiv!(rdiv!(B, F.U), F.D), F.L)
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

# ============================== div_impl! ==============================

function div_impl!(
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
        trsm!(side, uplo, tA, diag, 1, D₁₁, C₁)
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
                gemv!(Val(:N), -1, L₂₁, C₁, 1, M₂)
            else
                gemv!(tA, -1, L₂₁, C₁, 1, M₂)
            end
        elseif SIDE === :L
            if UPLO === :L
                gemm!(Val(:N), tB, -1, L₂₁, C₁, 1, M₂)
            else
                gemm!(tA, tB, -1, L₂₁, C₁, 1, M₂)
            end
        else
            if UPLO === :L
                gemm!(tB, tA, -1, C₁, L₂₁, 1, M₂)
            else
                gemm!(tB, Val(:N), -1, C₁, L₂₁, 1, M₂)
            end
        end
    end

    return ns
end

function div_bwd_loop!(
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
                gemv!(tA, -1, L₂₁, M₂, 1, C₁)
            else
                gemv!(Val(:N), -1, L₂₁, M₂, 1, C₁)
            end
        elseif SIDE === :L
            if UPLO === :L
                gemm!(tA, Val(:N), -1, L₂₁, M₂, 1, C₁)
            else
                gemm!(Val(:N), Val(:N), -1, L₂₁, M₂, 1, C₁)
            end
        else
            if UPLO === :L
                gemm!(Val(:N), Val(:N), -1, M₂, L₂₁, 1, C₁)
            else
                gemm!(Val(:N), tA, -1, M₂, L₂₁, 1, C₁)
            end
        end
    end
    #
    #     C₁ ← D₁₁⁻ᴴ C₁
    #
    if C isa AbstractVector
        trsv!(uplo, tA, diag, D₁₁, C₁)
    else
        trsm!(side, uplo, tA, diag, 1, D₁₁, C₁)
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
        F::AbstractVecOrMat{R},
        ptr::AbstractVector{I},
        val::AbstractVector{R},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
        side::Val{SIDE},
    ) where {R, I <: Integer, SIDE}

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
        addscatterrec!(F, M, inj, Val(:L))
    elseif SIDE === :L
        M = reshape(view(val, strt:strt + na * nrhs - one(I)), na, nrhs)
        addscatterrec!(F, M, inj, Val(:L))
    else
        M = reshape(view(val, strt:strt + na * nrhs - one(I)), nrhs, na)
        addscatterrec!(F, M, inj, Val(:R))
    end

    return
end

function div_bwd_update!(
        F::AbstractVecOrMat{R},
        ptr::AbstractVector{I},
        val::AbstractVector{R},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
        side::Val{SIDE},
    ) where {R, I <: Integer, SIDE}

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
        copygatherrec!(M, F, inj, Val(:L))
    elseif SIDE === :L
        M = reshape(view(val, strt:stop - one(I)), na, nrhs)
        copygatherrec!(M, F, inj, Val(:L))
    else
        M = reshape(view(val, strt:stop - one(I)), nrhs, na)
        copygatherrec!(M, F, inj, Val(:R))
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
