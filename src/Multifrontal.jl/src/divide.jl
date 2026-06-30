# ===== DivisionWorkspace =====

abstract type AbstractDivisionWorkspace end

struct DivisionWorkspace{T, I <: Integer} <: AbstractDivisionWorkspace
    Mptr::FVector{I}
    Mval::FVector{T}
    Fval::FVector{T}
end

struct DenseDivisionWorkspace <: AbstractDivisionWorkspace end

function DivisionWorkspace{T}(S::ChordalSymbolic{I}, nrhs::Integer) where {T, I <: Integer}
    Mptr = FVector{I}(undef, S.nMptr)
    Mval = FVector{T}(undef, S.nNval * nrhs)
    Fval = FVector{T}(undef, S.nFval * nrhs)
    return DivisionWorkspace{T, I}(Mptr, Mval, Fval)
end

function DivisionWorkspace(L::ChordalTriangular{DIAG, UPLO, T}, nrhs::Integer) where {DIAG, UPLO, T}
    return DivisionWorkspace{T}(L.S, nrhs)
end

function DivisionWorkspace(F::AbstractFactorization, nrhs::Integer)
    return DivisionWorkspace(triangular(F), nrhs)
end

function DivisionWorkspace(A::AdjOrTransTri, nrhs::Integer)
    return DivisionWorkspace(parent(A), nrhs)
end

function DivisionWorkspace(::AbstractTriangular, nrhs::Integer)
    return DenseDivisionWorkspace()
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

function LinearAlgebra.ldiv!(A::MaybeAdjOrTransTri, B::AbstractVecOrMat)
    W = DivisionWorkspace(A, size(B, 2))
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

function LinearAlgebra.ldiv!(F::NaturalFactorization{DIAG}, B::AbstractVecOrMat) where {DIAG}
    W = DivisionWorkspace(F, size(B, 2))
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

function LinearAlgebra.rdiv!(B::AbstractMatrix, A::MaybeAdjOrTransTri)
    W = DivisionWorkspace(A, size(B, 1))
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

function LinearAlgebra.rdiv!(B::AbstractMatrix, F::NaturalFactorization{DIAG}) where {DIAG}
    W = DivisionWorkspace(F, size(B, 1))
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

# ============================== div_impl! ==============================

function div_impl!(
        B::AbstractVecOrMat{R},
        W::DivisionWorkspace{R, I},
        L::ChordalTriangular{DIAG, UPLO, T, I},
        tA::Val{TA},
        tB::Val{TB},
        side::Val{SIDE},
    ) where {T, R, I <: Integer, DIAG, UPLO, TA, TB, SIDE}
    return div_impl!(B, W.Mptr, W.Mval, W.Fval, L, tA, tB, side)
end

function div_impl!(
        B::AbstractVecOrMat{R},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{R},
        Fval::AbstractVector{R},
        L::ChordalTriangular{DIAG, UPLO, T, I},
        tA::Val{TA},
        tB::Val{TB},
        side::Val{SIDE},
        args...,
    ) where {T, R, I <: Integer, DIAG, UPLO, TA, TB, SIDE}
    return div_impl!(
        B, Mptr, Mval,
        L.S.Dptr, L.Dval,
        L.S.Lptr, L.Lval,
        Fval,
        L.S.res, L.S.rel, L.S.chd,
        tA, tB, L.uplo, side, L.diag,
        args...)
end

function mt_div_impl!(
        B::AbstractMatrix{R},
        L::ChordalTriangular{DIAG, UPLO, T, I},
        tA::Val{TA},
        tB::Val{TB},
        side::Val{SIDE},
        bs::I,
        nt::I,
    ) where {T, R, I <: Integer, DIAG, UPLO, TA, TB, SIDE}
    if SIDE === :L
        nrhs = convert(I, size(B, 2))
    else
        nrhs = convert(I, size(B, 1))
    end

    Mptr = FVector{I}(undef, L.S.nMptr * nt)
    Mval = FVector{R}(undef, L.S.nNval * bs * nt)
    Fval = FVector{R}(undef, L.S.nFval * bs * nt)

    return mt_div_impl!(B, Mptr, Mval, Fval, L, tA, tB, side, bs, nt)
end

function mt_div_impl!(
        B::AbstractMatrix{R},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{R},
        Fval::AbstractVector{R},
        L::ChordalTriangular{DIAG, UPLO, T, I},
        tA::Val{TA},
        tB::Val{TB},
        side::Val{SIDE},
        bs::I,
        nt::I,
    ) where {T, R, I <: Integer, DIAG, UPLO, TA, TB, SIDE}
    if SIDE === :L
        nrhs = convert(I, size(B, 2))
    else
        nrhs = convert(I, size(B, 1))
    end

    nMptr = L.S.nMptr
    nNval = L.S.nNval
    nFval = L.S.nFval

    pool = Channel{I}(nt)

    for t in oneto(nt)
        put!(pool, t)
    end

    @threads for strt in one(I):bs:nrhs
        size = min(bs, nrhs - strt + one(I))
        stop =         strt + size - one(I)

        t = take!(pool)

        Mptroff = (t - one(I)) * nMptr
        Mvaloff = (t - one(I)) * nNval * bs
        Fvaloff = (t - one(I)) * nFval * bs

        Mptrblk = view(Mptr, Mptroff + one(I):Mptroff + nMptr)
        Mvalblk = view(Mval, Mvaloff + one(I):Mvaloff + nNval * size)
        Fvalblk = view(Fval, Fvaloff + one(I):Fvaloff + nFval * size)

        if SIDE === :L
            Bblk = view(B, :, strt:stop)
        else
            Bblk = view(B, strt:stop, :)
        end

        div_impl!(Bblk, Mptrblk, Mvalblk, Fvalblk, L, tA, tB, side)

        put!(pool, t)
    end

    close(pool)
    return B
end

function div_impl!(
        B::AbstractVecOrMat{R},
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
        tA::Val{TA},
        tB::Val{TB},
        uplo::Val{UPLO},
        side::Val{SIDE},
        diag::Val{DIAG},
        rng::AbstractRange{I} = vertices(res),
    ) where {T, R, I <: Integer, TA, TB, UPLO, SIDE, DIAG}

    ns = zero(I); Mptr[one(I)] = one(I)

    if isforward(UPLO, TA, SIDE)
        for j in rng
            ns = div_fwd_loop!(B, Mptr, Mval, Dptr, Dval, Lptr, Lval, Fval, res, rel, chd, ns, j, tA, tB, uplo, side, diag)
        end
    else
        for j in reverse(rng)
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
    nn = eltypedegree(res, j)

    if T <: Real && isone(nn)
        return div_fwd_loop_nod!(
            C, Mptr, Mval,
            Dptr, Dval,
            Lptr, Lval,
            Fval,
            res, rel, chd, ns, j, tA, tB, uplo, side, diag
        )
    else
        return div_fwd_loop_snd!(
            C, Mptr, Mval,
            Dptr, Dval,
            Lptr, Lval,
            Fval,
            res, rel, chd, ns, nn, j, tA, tB, uplo, side, diag
        )
    end
end

function div_fwd_loop_snd!(
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
        nn::I,
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

# Fast path for nn = 1 and real element types
function div_fwd_loop_nod!(
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
    # nn = 1 (the size of the residual at node j)
    #
    nn = one(I)
    #
    # na is the size of the separator at node j
    #
    #     na = |sep(j)|
    #
    na = eltypedegree(rel, j)
    #
    # nj is the size of the bag at node j
    #
    #     nj = |bag(j)| = 1 + na
    #
    nj = nn + na
    #
    # F is the frontal matrix at node j
    #
    #        nrhs
    #   F = [ f₁ ] 1
    #       [ F₂ ] na
    #
    if C isa AbstractVector
        F = view(Fval, oneto(nj))
        F₂ = view(F, nn + one(I):nj)
    elseif SIDE === :L
        F = reshape(view(Fval, oneto(nj * nrhs)), nj, nrhs)
        f₁ = view(F, one(I), oneto(nrhs))
        F₂ = view(F, nn + one(I):nj, oneto(nrhs))
    else
        F = reshape(view(Fval, oneto(nj * nrhs)), nrhs, nj)
        f₁ = view(F, oneto(nrhs), one(I))
        F₂ = view(F, oneto(nrhs), nn + one(I):nj)
    end
    #
    # L is part of the factor (d₁₁ is scalar, l₂₁ is vector)
    #
    #        res(j)
    #   L = [ d₁₁ ] res(j)
    #       [ l₂₁ ] sep(j)
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    Rp = pointers(res)[j]
    d₁₁ = Dval[Dp]
    l₂₁ = view(Lval, Lp:Lp + na - one(I))
    #
    # c₁ is part of the right-hand side
    #
    #        nrhs
    #   C = [ c₁ ] res(j)
    #
    if C isa AbstractVector
        c₁ = C[Rp]
    elseif SIDE === :L
        c₁ = view(C, Rp, oneto(nrhs))
    else
        c₁ = view(C, oneto(nrhs), Rp)
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
    #     c₁ ← c₁ + f₁
    #
    if C isa AbstractVector
        c₁ += Fval[one(I)]
    else
        addrec!(c₁, f₁)
    end
    #
    #     c₁ ← d₁₁⁻¹ c₁ (scalar division)
    #
    if DIAG === :N
        if C isa AbstractVector
            c₁ /= d₁₁
        else
            rdiv!(c₁, d₁₁)
        end
    end
    #
    # Write back scalar for vector case
    #
    if C isa AbstractVector
        C[Rp] = c₁
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
        #     M₂ ← F₂ - l₂₁ c₁
        #
        copyrec!(M₂, F₂)

        if C isa AbstractVector
            axpy!(-c₁, l₂₁, M₂)
        elseif SIDE === :L
            ger!(-1, l₂₁, c₁, M₂)
        else
            ger!(-1, c₁, l₂₁, M₂)
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
    nn = eltypedegree(res, j)

    if T <: Real && isone(nn)
        return div_bwd_loop_nod!(
            C, Mptr, Mval,
            Dptr, Dval,
            Lptr, Lval,
            Fval,
            res, rel, chd, ns, j, tA, tB, uplo, side, diag
        )
    else
        return div_bwd_loop_snd!(
            C, Mptr, Mval,
            Dptr, Dval,
            Lptr, Lval,
            Fval,
            res, rel, chd, ns, nn, j, tA, tB, uplo, side, diag
        )
    end
end

function div_bwd_loop_snd!(
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
        nn::I,
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
                gemv!(    tA,  -1, L₂₁, M₂, 1, C₁)
            else
                gemv!(Val(:N), -1, L₂₁, M₂, 1, C₁)
            end
        elseif SIDE === :L
            if UPLO === :L
                gemm!(    tA,  Val(:N), -1, L₂₁, M₂, 1, C₁)
            else
                gemm!(Val(:N), Val(:N), -1, L₂₁, M₂, 1, C₁)
            end
        else
            if UPLO === :L
                gemm!(Val(:N), Val(:N), -1, M₂, L₂₁, 1, C₁)
            else
                gemm!(Val(:N),     tA,  -1, M₂, L₂₁, 1, C₁)
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

# Fast path for nn = 1 and real element types
function div_bwd_loop_nod!(
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
    # nn = 1 (the size of the residual at node j)
    #
    nn = one(I)
    #
    # na is the size of the separator at node j
    #
    #     na = |sep(j)|
    #
    na = eltypedegree(rel, j)
    #
    # nj is the size of the bag at node j
    #
    #     nj = |bag(j)| = 1 + na
    #
    nj = nn + na
    #
    # L is part of the factor (d₁₁ is scalar, l₂₁ is vector)
    #
    #        res(j)
    #   L = [ d₁₁ ] res(j)
    #       [ l₂₁ ] sep(j)
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    Rp = pointers(res)[j]
    d₁₁ = Dval[Dp]
    l₂₁ = view(Lval, Lp:Lp + na - one(I))
    #
    # c₁ is part of the right-hand side
    #
    #        nrhs
    #   C = [ c₁ ] res(j)
    #
    if C isa AbstractVector
        c₁ = C[Rp]
    elseif SIDE === :L
        c₁ = view(C, Rp, oneto(nrhs))
    else
        c₁ = view(C, oneto(nrhs), Rp)
    end
    #
    # subtract the update matrix from ancestor
    #
    #     c₁ ← c₁ - l₂₁ᵀ M₂
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
            c₁ -= dot(l₂₁, M₂)
        elseif SIDE === :L
            gemv!(Val(:T), -1, M₂, l₂₁, 1, c₁)
        else
            gemv!(Val(:N), -1, M₂, l₂₁, 1, c₁)
        end
    end
    #
    #     c₁ ← d₁₁⁻¹ c₁ (scalar division)
    #
    if DIAG === :N
        if C isa AbstractVector
            c₁ /= d₁₁
        else
            rdiv!(c₁, d₁₁)
        end
    end
    #
    # Write back scalar for vector case
    #
    if C isa AbstractVector
        C[Rp] = c₁
    end
    #
    # F is the frontal matrix at node j
    #
    #        nrhs
    #   F = [ f₁ ] 1
    #       [ F₂ ] na
    #
    if C isa AbstractVector
        F = view(Fval, oneto(nj))
        F₂ = view(F, nn + one(I):nj)
        Fval[one(I)] = c₁
    elseif SIDE === :L
        F = reshape(view(Fval, oneto(nj * nrhs)), nj, nrhs)
        f₁ = view(F, one(I), oneto(nrhs))
        F₂ = view(F, nn + one(I):nj, oneto(nrhs))
        copyrec!(f₁, c₁)
    else
        F = reshape(view(Fval, oneto(nj * nrhs)), nrhs, nj)
        f₁ = view(F, oneto(nrhs), one(I))
        F₂ = view(F, oneto(nrhs), nn + one(I):nj)
        copyrec!(f₁, c₁)
    end
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

# ============================= div_piv_impl! =============================

function div_piv_impl!(
        B::AbstractVecOrMat{R},
        L::ChordalTriangular{DIAG, UPLO, T, I},
        tA::Val{TA},
        tB::Val{TB},
        side::Val{SIDE},
    ) where {T, R, I <: Integer, DIAG, UPLO, TA, TB, SIDE}
    if B isa AbstractVector
        nrhs = one(I)
    elseif SIDE === :L
        nrhs = convert(I, size(B, 2))
    else
        nrhs = convert(I, size(B, 1))
    end

    Mptr = FVector{I}(undef, L.S.nMptr)
    Mval = FVector{R}(undef, L.S.nNval * nrhs)
    Fval = FVector{R}(undef, L.S.nFval * nrhs)

    return div_piv_impl!(B, Mptr, Mval, Fval, L, tA, tB, side)
end

function div_piv_impl!(
        B::AbstractVecOrMat{R},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{R},
        Fval::AbstractVector{R},
        L::ChordalTriangular{DIAG, UPLO, T, I},
        tA::Val{TA},
        tB::Val{TB},
        side::Val{SIDE},
    ) where {T, R, I <: Integer, DIAG, UPLO, TA, TB, SIDE}
    return div_piv_impl!(
        B, Mptr, Mval,
        L.S.Dptr, L.Dval,
        L.S.Lptr, L.Lval,
        Fval,
        L.S.res, L.S.rel, L.S.chd,
        tA, tB, L.uplo, side, L.diag
    )
end

function div_piv_impl!(
        B::AbstractVecOrMat{R},
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
        tA::Val{TA},
        tB::Val{TB},
        uplo::Val{UPLO},
        side::Val{SIDE},
        diag::Val{DIAG},
        rng::AbstractRange{I} = vertices(res),
    ) where {T, R, I <: Integer, TA, TB, UPLO, SIDE, DIAG}

    ns = zero(I); Mptr[one(I)] = one(I)

    if isforward(UPLO, TA, SIDE)
        for j in rng
            ns = div_piv_fwd_loop!(B, Mptr, Mval, Dptr, Dval, Lptr, Lval, Fval, res, rel, chd, ns, j, tA, tB, uplo, side, diag)
        end
    else
        for j in reverse(rng)
            ns = div_piv_bwd_loop!(B, Mptr, Mval, Dptr, Dval, Lptr, Lval, Fval, res, rel, chd, ns, j, tA, tB, uplo, side, diag)
        end
    end

    return B
end

function div_piv_fwd_loop!(
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
    #
    # Compute rank by scanning diagonal of D₁₁
    #
    rank = zero(I)

    @inbounds while rank < nn && ispositive(D₁₁[rank + one(I), rank + one(I)])
        rank += one(I)
    end

    rD₁₁ = view(D₁₁, oneto(rank), oneto(rank))

    if UPLO === :L
         L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
        rL₂₁ = view(L₂₁, oneto(na), oneto(rank))
    else
         L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
        rL₂₁ = view(L₂₁, oneto(rank), oneto(na))
    end
    #
    # C₁ is part of the right-hand side
    #
    #        nrhs
    #   C = [ C₁ ] res(j)
    #
    if C isa AbstractVector
         C₁ = view(C,  neighbors(res, j))
        rC₁ = view(C₁, oneto(rank))
        nC₁ = view(C₁, rank + one(I):nn)
    elseif SIDE === :L
         C₁ = view(C,  neighbors(res, j), oneto(nrhs))
        rC₁ = view(C₁, oneto(rank),       oneto(nrhs))
        nC₁ = view(C₁, rank + one(I):nn,  oneto(nrhs))
    else
         C₁ = view(C,  oneto(nrhs), neighbors(res, j))
        rC₁ = view(C₁, oneto(nrhs), oneto(rank))
        nC₁ = view(C₁, oneto(nrhs), rank + one(I):nn)
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
    # Zero out null-space components
    #
    zerorec!(nC₁)
    #
    #     C₁ ← D₁₁⁻¹ C₁  (only the full-rank part)
    #
    if ispositive(rank)
        if C isa AbstractVector
            trsv!(uplo, tA, diag, rD₁₁, rC₁)
        else
            trsm!(side, uplo, tA, diag, 1, rD₁₁, rC₁)
        end
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
        #     M₂ ← F₂ - L₂₁ C₁  (only the full-rank part)
        #
        copyrec!(M₂, F₂)

        if isforward(UPLO, :N, SIDE)
            tL = Val(:N)
        else
            tL = tA
        end

        if ispositive(rank)
            if C isa AbstractVector
                gemv!(tL, -1, rL₂₁, rC₁, 1, M₂)
            elseif SIDE === :L
                gemm!(tL, tB, -1, rL₂₁, rC₁, 1, M₂)
            else
                gemm!(tB, tL, -1, rC₁, rL₂₁, 1, M₂)
            end
        end
    end

    return ns
end

function div_piv_bwd_loop!(
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
    #
    # Compute rank by scanning diagonal of D₁₁
    #
    rank = zero(I)

    @inbounds while rank < nn && ispositive(D₁₁[rank + one(I), rank + one(I)])
        rank += one(I)
    end

    rD₁₁ = view(D₁₁, oneto(rank), oneto(rank))

    if UPLO === :L
         L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
        rL₂₁ = view(L₂₁, oneto(na), oneto(rank))
    else
         L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
        rL₂₁ = view(L₂₁, oneto(rank), oneto(na))
    end
    #
    # C₁ is part of the right-hand side
    #
    #        nrhs
    #   C = [ C₁ ] res(j)
    #
    if C isa AbstractVector
         C₁ = view(C,  neighbors(res, j))
        rC₁ = view(C₁, oneto(rank))
        nC₁ = view(C₁, rank + one(I):nn)
    elseif SIDE === :L
         C₁ = view(C,  neighbors(res, j), oneto(nrhs))
        rC₁ = view(C₁, oneto(rank),       oneto(nrhs))
        nC₁ = view(C₁, rank + one(I):nn,  oneto(nrhs))
    else
         C₁ = view(C,  oneto(nrhs), neighbors(res, j))
        rC₁ = view(C₁, oneto(nrhs), oneto(rank))
        nC₁ = view(C₁, oneto(nrhs), rank + one(I):nn)
    end
    #
    # subtract the update matrix from ancestor
    #
    #     C₁ ← C₁ - L₂₁ᴴ M₂  (only the full-rank part)
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

        if isforward(UPLO, :N, SIDE)
            tL = tA
        else
            tL = Val(:N)
        end

        if ispositive(rank)
            if C isa AbstractVector
                gemv!(tL, -1, rL₂₁, M₂, 1, rC₁)
            elseif SIDE === :L
                gemm!(tL, Val(:N), -1, rL₂₁, M₂, 1, rC₁)
            else
                gemm!(Val(:N), tL, -1, M₂, rL₂₁, 1, rC₁)
            end
        end
    end
    #
    # Zero out null-space components
    #
    zerorec!(nC₁)
    #
    #     C₁ ← D₁₁⁻ᴴ C₁  (only the full-rank part)
    #
    if ispositive(rank)
        if C isa AbstractVector
            trsv!(uplo, tA, diag, rD₁₁, rC₁)
        else
            trsm!(side, uplo, tA, diag, 1, rD₁₁, rC₁)
        end
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
