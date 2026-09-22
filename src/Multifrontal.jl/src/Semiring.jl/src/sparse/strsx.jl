# ===== strsx! =====

function strsx!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        A::ChordalTriangular{<:Any, UPLO, T, I},
        B::AbstractVecOrMat;
        nt::Integer = nthreads(),
    ) where {SIDE, TRANS, UPLO, T, I}
    S = A.S

    if B isa AbstractVector
        nrhs = one(I)
        pool = nothing
    elseif SIDE === :L
        nrhs = convert(I, size(B, 2))
        pool = spool_mt(T, nt)
    else
        nrhs = convert(I, size(B, 1))
        pool = spool_mt(T, nt)
    end

    W = DivisionWorkspace{T}(S, nrhs)
    return strsx_mt!(s, side, trans, A, B, W, pool, nt)
end

function strsx_mt!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        A::ChordalTriangular{<:Any, UPLO, T, I},
        B::AbstractVecOrMat,
        W::DivisionWorkspace{T},
        pool,
        nt::Integer,
    ) where {SIDE, TRANS, UPLO, T, I}
    S = A.S

    if B isa AbstractVector
        nrhs = one(I)
    elseif SIDE === :L
        nrhs = convert(I, size(B, 2))
    else
        nrhs = convert(I, size(B, 1))
    end

    if isforward(UPLO, TRANS, SIDE)
        for j in vertices(S.res)
            strsx_fwd!(s, B, W.Mval, A.Dval, A.Lval, S.Dptr, S.Lptr, S.res, S.sep, pool, nt, nrhs, j, trans, A.uplo, side)
        end
    else
        for j in reverse(vertices(S.res))
            strsx_bwd!(s, B, W.Mval, A.Dval, A.Lval, S.Dptr, S.Lptr, S.res, S.sep, pool, nt, nrhs, j, trans, A.uplo, side)
        end
    end

    return B
end

# ===== strsx_fwd! =====

function strsx_fwd!(
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
        side::Val{SIDE},
    ) where {T, I, TRANS, UPLO, SIDE}
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
    if C isa AbstractVector
        C₁ = view(C, neighbors(res, j))
    elseif SIDE === :L
        C₁ = view(C, neighbors(res, j), oneto(nrhs))
    else
        C₁ = view(C, oneto(nrhs), neighbors(res, j))
    end
    #
    #   C₁ ← L₁₁* C₁
    #
    if C isa AbstractVector
        strsx!(s, side, trans, uplo, D₁₁, C₁)
    else
        strsx_mt!(s, side, trans, uplo, D₁₁, C₁, pool, nt)
    end

    if ispositive(na)
        if C isa AbstractVector
            M₂ = view(Mval, oneto(na))
        elseif SIDE === :L
            M₂ = reshape(view(Mval, oneto(na * nrhs)), na, nrhs)
        else
            M₂ = reshape(view(Mval, oneto(na * nrhs)), nrhs, na)
        end
        #
        #   M₂ ← L₂₁ C₁
        #
        szerorec!(s, M₂, trans)

        if C isa AbstractVector
            if SIDE === :L
                sgemx!(s, trans, Val(:N), M₂, L₂₁, C₁)
            else
                sgemx!(s, Val(:N), trans, M₂, C₁, L₂₁)
            end
        elseif SIDE === :L
            sgemx_mt!(s, trans, Val(:N), M₂, L₂₁, C₁, pool, nt)
        else
            sgemx_mt!(s, Val(:N), trans, M₂, C₁, L₂₁, pool, nt)
        end
        #
        #   C₂ ← C₂ + M₂
        #
        if C isa AbstractVector
            sscatteradd!(s, trans, C, M₂, neighbors(sep, j))
        else
            sscatteradd!(s, trans, C, M₂, neighbors(sep, j), side)
        end
    end

    return
end

# ===== strsx_bwd! =====

function strsx_bwd!(
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
        side::Val{SIDE},
    ) where {T, I, TRANS, UPLO, SIDE}
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
    elseif SIDE === :L
        C₁ = view(C, neighbors(res, j), oneto(nrhs))
    else
        C₁ = view(C, oneto(nrhs), neighbors(res, j))
    end

    if ispositive(na)
        if C isa AbstractVector
            M₂ = view(Mval, oneto(na))
        elseif SIDE === :L
            M₂ = reshape(view(Mval, oneto(na * nrhs)), na, nrhs)
        else
            M₂ = reshape(view(Mval, oneto(na * nrhs)), nrhs, na)
        end
        #
        #   M₂ ← C₂
        #
        if C isa AbstractVector
            copygatherrec!(M₂, C, neighbors(sep, j))
        else
            copygatherrec!(M₂, C, neighbors(sep, j), side)
        end
        #
        #   C₁ ← U₁₂ M₂ + C₁
        #
        if C isa AbstractVector
            if SIDE === :L
                sgemx!(s, trans, Val(:N), C₁, U₁₂, M₂)
            else
                sgemx!(s, Val(:N), trans, C₁, M₂, U₁₂)
            end
        elseif SIDE === :L
            sgemx_mt!(s, trans, Val(:N), C₁, U₁₂, M₂, pool, nt)
        else
            sgemx_mt!(s, Val(:N), trans, C₁, M₂, U₁₂, pool, nt)
        end
    end
    #
    #   C₁ ← U₁₁* C₁
    #
    if C isa AbstractVector
        strsx!(s, side, trans, uplo, D₁₁, C₁)
    else
        strsx_mt!(s, side, trans, uplo, D₁₁, C₁, pool, nt)
    end

    return
end
