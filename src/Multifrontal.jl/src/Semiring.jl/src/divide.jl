# ===== sldiv! =====

function sldiv!(F::MaybeAdjOrTransSLU{S, T}, B::AbstractVecOrMat) where {S, T}
    P = parent(F)

    if B isa AbstractVector
        nrhs = 1
    else
        nrhs = size(B, 2)
    end

    W = DivisionWorkspace{T}(P.S, nrhs)
    return sldiv!(W, F, B)
end

function sldiv!(W::DivisionWorkspace, F::MaybeAdjOrTransSLU{S, T}, B::AbstractVecOrMat) where {S, T}
    P = parent(F)
    C = FArray{T}(undef, size(B))

    if F isa TransSLU
        mul!(C, P.Q, B)
        sgetrs!(P.s, Val(:L), Val(:T), W, P.L, P.U, C)
        ldiv!(B, P.P, C)
    elseif F isa AdjSLU
        mul!(C, P.Q, B)
        sgetrs!(P.s, Val(:L), Val(:C), W, P.L, P.U, C)
        ldiv!(B, P.P, C)
    else
        mul!(C, P.P, B)
        sgetrs!(P.s, Val(:L), Val(:N), W, P.L, P.U, C)
        ldiv!(B, P.Q, C)
    end

    return B
end

function ldiv!(F::MaybeAdjOrTransSLU, B::AbstractVecOrMat)
    return sldiv!(F, B)
end

function ldiv!(W::DivisionWorkspace, F::MaybeAdjOrTransSLU, B::AbstractVecOrMat)
    return sldiv!(W, F, B)
end

function Base.:\(F::MaybeAdjOrTransSLU, B::AbstractVecOrMat)
    return sldiv!(F, copy(B))
end

function sldiv!(
        s::AbstractSemiring,
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        B::AbstractVecOrMat,
    ) where {T, I}
    if B isa AbstractVector
        nrhs = one(I)
    else
        nrhs = convert(I, size(B, 2))
    end

    W = DivisionWorkspace{T}(L.S, nrhs)
    return sldiv!(s, W, L, U, B)
end

function sldiv!(
        s::AbstractSemiring,
        W::DivisionWorkspace{T},
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        B::AbstractVecOrMat,
    ) where {T, I}
    return sgetrs!(s, Val(:L), Val(:N), W, L, U, B)
end

# ===== srdiv! =====

function srdiv!(B::AbstractMatrix, F::MaybeAdjOrTransSLU{S, T}) where {S, T}
    P = parent(F)
    W = DivisionWorkspace{T}(P.S, size(B, 1))
    return srdiv!(W, B, F)
end

function srdiv!(W::DivisionWorkspace, B::AbstractMatrix, F::MaybeAdjOrTransSLU{S, T}) where {S, T}
    P = parent(F)
    C = FMatrix{T}(undef, size(B))

    if F isa TransSLU
        rdiv!(C, B, P.P)
        sgetrs!(P.s, Val(:R), Val(:T), W, P.L, P.U, C)
        mul!(B, C, P.Q)
    elseif F isa AdjSLU
        rdiv!(C, B, P.P)
        sgetrs!(P.s, Val(:R), Val(:C), W, P.L, P.U, C)
        mul!(B, C, P.Q)
    else
        rdiv!(C, B, P.Q)
        sgetrs!(P.s, Val(:R), Val(:N), W, P.L, P.U, C)
        mul!(B, C, P.P)
    end

    return B
end

function rdiv!(B::AbstractMatrix, F::MaybeAdjOrTransSLU)
    return srdiv!(B, F)
end

function rdiv!(W::DivisionWorkspace, B::AbstractMatrix, F::MaybeAdjOrTransSLU)
    return srdiv!(W, B, F)
end

function Base.:/(B::AbstractMatrix, F::MaybeAdjOrTransSLU)
    return srdiv!(copy(B), F)
end

function srdiv!(
        s::AbstractSemiring,
        B::AbstractMatrix,
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
    ) where {T, I}
    nrhs = convert(I, size(B, 1))
    W = DivisionWorkspace{T}(L.S, nrhs)
    return srdiv!(s, W, B, L, U)
end

function srdiv!(
        s::AbstractSemiring,
        W::DivisionWorkspace{T},
        B::AbstractMatrix,
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
    ) where {T, I}
    return sgetrs!(s, Val(:R), Val(:N), W, L, U, B)
end

# ===== sgetrs! =====

function sgetrs!(
        s::AbstractSemiring,
        ::Val{:L},
        trans::Val{TRANS},
        W::DivisionWorkspace{T},
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        B::AbstractVecOrMat,
    ) where {T, I, TRANS}
    S = L.S

    nt = nthreads()

    if B isa AbstractVector
        nrhs = one(I)
        pool = nothing
    else
        nrhs = convert(I, size(B, 2))
        pool = spool(T, nt)
    end

    if isforward(:L, TRANS, :L)
        for j in vertices(S.res)
            sldiv_fwd!(s, B, W.Mval, L.Dval, L.Lval, S.Dptr, S.Lptr, S.res, S.sep, pool, nt, nrhs, j, trans, Val(:L))
        end

        for j in reverse(vertices(S.res))
            sldiv_bwd!(s, B, W.Mval, U.Dval, U.Lval, S.Dptr, S.Lptr, S.res, S.sep, pool, nt, nrhs, j, trans, Val(:U))
        end
    else
        for j in vertices(S.res)
            sldiv_fwd!(s, B, W.Mval, U.Dval, U.Lval, S.Dptr, S.Lptr, S.res, S.sep, pool, nt, nrhs, j, trans, Val(:U))
        end

        for j in reverse(vertices(S.res))
            sldiv_bwd!(s, B, W.Mval, L.Dval, L.Lval, S.Dptr, S.Lptr, S.res, S.sep, pool, nt, nrhs, j, trans, Val(:L))
        end
    end

    return B
end

function sldiv_fwd!(
        s::AbstractSemiring,
        C::AbstractVecOrMat{T},
        Mval::AbstractVector{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        pool,
        nt::Integer,
        nrhs::I,
        j::I,
        trans::Val{TRANS},
        uplo::Val{UPLO},
    ) where {T, I, TRANS, UPLO}
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
    na = eltypedegree(sep, j)
    Dp = Dptr[j]
    Lp = Lptr[j]
    #
    #          res(j)
    #     L = [ D₁₁ ] res(j)
    #         [ L₂₁ ] sep(j)
    #
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    #   C = [ C₁ ] res(j)
    #       [ C₂ ] sep(j)
    #
    #   C₁ ← L₁₁* C₁
    #
    if C isa AbstractVector
        C₁ = view(C, neighbors(res, j))
        strsx!(s, Val(:L), trans, uplo, D₁₁, C₁)
    else
        C₁ = view(C, neighbors(res, j), oneto(nrhs))
        strsx_mt!(s, Val(:L), trans, uplo, D₁₁, C₁, pool, nt)
    end

    if ispositive(na)
        #
        #   M₂ ← L₂₁ C₁
        #
        #   C₂ ← C₂ + M₂
        #
        if C isa AbstractVector
            M₂ = view(Mval, oneto(na))
            fill!(M₂, szero(s, T, trans))
            sgemv!(s, trans, Val(:N), M₂, L₂₁, C₁)
            sscatteradd!(s, trans, C, M₂, neighbors(sep, j))
        else
            M₂ = reshape(view(Mval, oneto(na * nrhs)), na, nrhs)
            fill!(M₂, szero(s, T, trans))
            sgemx_mt!(s, trans, Val(:N), M₂, L₂₁, C₁, pool, ceil(Int, log2(nt)) + 1)
            sscatteradd!(s, trans, C, M₂, neighbors(sep, j), Val(:L))
        end
    end

    return
end

function sldiv_bwd!(
        s::AbstractSemiring,
        C::AbstractVecOrMat{T},
        Mval::AbstractVector{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        pool,
        nt::Integer,
        nrhs::I,
        j::I,
        trans::Val{TRANS},
        uplo::Val{UPLO},
    ) where {T, I, TRANS, UPLO}
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
    na = eltypedegree(sep, j)
    Dp = Dptr[j]
    Lp = Lptr[j]
    #
    #          res(j) sep(j)
    #     U = [ D₁₁    U₁₂ ] res(j)
    #
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :U
        U₁₂ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    else
        U₁₂ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    end
    #
    #   C = [ C₁ ] res(j)
    #       [ C₂ ] sep(j)
    #
    if C isa AbstractVector
        C₁ = view(C, neighbors(res, j))
    else
        C₁ = view(C, neighbors(res, j), oneto(nrhs))
    end

    if ispositive(na)
        #
        #   M₂ ← C₂
        #
        #   C₁ ← U₁₂ M₂ + C₁
        #
        if C isa AbstractVector
            M₂ = view(Mval, oneto(na))
            copygatherrec!(M₂, C, neighbors(sep, j))
            sgemv!(s, trans, Val(:N), C₁, U₁₂, M₂)
        else
            M₂ = reshape(view(Mval, oneto(na * nrhs)), na, nrhs)
            copygatherrec!(M₂, C, neighbors(sep, j), Val(:L))
            sgemx_mt!(s, trans, Val(:N), C₁, U₁₂, M₂, pool, ceil(Int, log2(nt)) + 1)
        end
    end
    #
    #   C₁ ← U₁₁* C₁
    #
    if C isa AbstractVector
        strsx!(s, Val(:L), trans, uplo, D₁₁, C₁)
    else
        strsx_mt!(s, Val(:L), trans, uplo, D₁₁, C₁, pool, nt)
    end

    return
end

function sgetrs!(
        s::AbstractSemiring,
        ::Val{:R},
        trans::Val{TRANS},
        W::DivisionWorkspace{T},
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        B::AbstractMatrix,
    ) where {T, I, TRANS}
    S = L.S

    nt = nthreads()
    nrhs = convert(I, size(B, 1))
    pool = spool(T, nt)

    if isforward(:L, TRANS, :R)
        for j in vertices(S.res)
            srdiv_fwd!(s, B, W.Mval, L.Dval, L.Lval, S.Dptr, S.Lptr, S.res, S.sep, pool, nt, nrhs, j, trans, Val(:L))
        end

        for j in reverse(vertices(S.res))
            srdiv_bwd!(s, B, W.Mval, U.Dval, U.Lval, S.Dptr, S.Lptr, S.res, S.sep, pool, nt, nrhs, j, trans, Val(:U))
        end
    else
        for j in vertices(S.res)
            srdiv_fwd!(s, B, W.Mval, U.Dval, U.Lval, S.Dptr, S.Lptr, S.res, S.sep, pool, nt, nrhs, j, trans, Val(:U))
        end

        for j in reverse(vertices(S.res))
            srdiv_bwd!(s, B, W.Mval, L.Dval, L.Lval, S.Dptr, S.Lptr, S.res, S.sep, pool, nt, nrhs, j, trans, Val(:L))
        end
    end

    return B
end

function srdiv_fwd!(
        s::AbstractSemiring,
        C::AbstractMatrix{T},
        Mval::AbstractVector{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        pool,
        nt::Integer,
        nrhs::I,
        j::I,
        trans::Val{TRANS},
        uplo::Val{UPLO},
    ) where {T, I, TRANS, UPLO}
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
    na = eltypedegree(sep, j)
    Dp = Dptr[j]
    Lp = Lptr[j]
    #
    #          res(j) sep(j)
    #     U = [ D₁₁    U₁₂ ] res(j)
    #
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :U
        U₁₂ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    else
        U₁₂ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    end
    #
    #          res(j) sep(j)
    #     C = [  C₁     C₂ ]
    #
    #   C₁ ← C₁ U₁₁*
    #
    C₁ = view(C, oneto(nrhs), neighbors(res, j))
    strsx_mt!(s, Val(:R), trans, uplo, D₁₁, C₁, pool, nt)

    if ispositive(na)
        #
        #   M₂ ← C₁ U₁₂
        #
        #   C₂ ← C₂ + M₂
        #
        M₂ = reshape(view(Mval, oneto(na * nrhs)), nrhs, na)
        fill!(M₂, szero(s, T, trans))
        sgemx_mt!(s, Val(:N), trans, M₂, C₁, U₁₂, pool, ceil(Int, log2(nt)) + 1)
        sscatteradd!(s, trans, C, M₂, neighbors(sep, j), Val(:R))
    end

    return
end

function srdiv_bwd!(
        s::AbstractSemiring,
        C::AbstractMatrix{T},
        Mval::AbstractVector{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        pool,
        nt::Integer,
        nrhs::I,
        j::I,
        trans::Val{TRANS},
        uplo::Val{UPLO},
    ) where {T, I, TRANS, UPLO}
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
    na = eltypedegree(sep, j)
    Dp = Dptr[j]
    Lp = Lptr[j]
    #
    #          res(j)
    #     L = [ D₁₁ ] res(j)
    #         [ L₂₁ ] sep(j)
    #
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    #          res(j) sep(j)
    #     C = [  C₁     C₂ ]
    #
    C₁ = view(C, oneto(nrhs), neighbors(res, j))

    if ispositive(na)
        #
        #   M₂ ← C₂
        #
        #   C₁ ← M₂ L₂₁ + C₁
        #
        M₂ = reshape(view(Mval, oneto(na * nrhs)), nrhs, na)
        copygatherrec!(M₂, C, neighbors(sep, j), Val(:R))
        sgemx_mt!(s, Val(:N), trans, C₁, M₂, L₂₁, pool, ceil(Int, log2(nt)) + 1)
    end
    #
    #   C₁ ← C₁ L₁₁*
    #
    strsx_mt!(s, Val(:R), trans, uplo, D₁₁, C₁, pool, nt)

    return
end
