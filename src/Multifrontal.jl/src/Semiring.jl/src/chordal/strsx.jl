const STRSX_1_NB = 12

# ===== strsx! =====

function strsx!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        diag::Val,
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
    return strsx_mt!(s, side, trans, diag, A, B, W, pool, nt)
end

function strsx_mt!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        diag::Val,
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
        for f in vertices(S.res)
            nn = eltypedegree(S.res, f)

            if isone(nn)
                strsx_fwd_1!(s, B, A.Dval, A.Lval, S.Dptr, S.Lptr, S.res, S.sep, nrhs, f, trans, A.uplo, diag, side)
            else
                strsx_fwd!(s, B, W.Mval, A.Dval, A.Lval, S.Dptr, S.Lptr, S.res, S.sep, pool, nt, nrhs, f, trans, A.uplo, diag, side)
            end
        end
    else
        for f in reverse(vertices(S.res))
            nn = eltypedegree(S.res, f)

            if isone(nn)
                strsx_bwd_1!(s, B, A.Dval, A.Lval, S.Dptr, S.Lptr, S.res, S.sep, nrhs, f, trans, A.uplo, diag, side)
            else
                strsx_bwd!(s, B, W.Mval, A.Dval, A.Lval, S.Dptr, S.Lptr, S.res, S.sep, pool, nt, nrhs, f, trans, A.uplo, diag, side)
            end
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
        f::I,
        trans::Val{TRANS},
        uplo::Val{UPLO},
        diag::Val,
        side::Val{SIDE},
    ) where {T, I, TRANS, UPLO, SIDE}
    #
    # nn is the size of the residual at node f
    #
    #     nn = | res(f) |
    #
    nn = eltypedegree(res, f)
    #
    # na is the size of the separator at node f
    #
    #     na = | sep(f) |
    #
    na = eltypedegree(sep, f)
    Dp = Dptr[f]
    Lp = Lptr[f]
    #
    #          res(f)
    #     L = [ D₁₁ ] res(f)
    #         [ L₂₁ ] sep(f)
    #
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end

    strsx_fwd_upd!(s, C, Mval, D₁₁, L₂₁, res, sep, na, nrhs, pool, nt, f, trans, uplo, diag, side)

    return
end

function strsx_fwd_upd!(
        s::AbstractSemiring,
        C::AbstractVecOrMat{T},
        Mval::AbstractVector{T},
        D₁₁::AbstractMatrix{T},
        L₂₁::AbstractMatrix{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        na::I,
        nrhs::I,
        pool,
        nt::Integer,
        f::I,
        trans::Val,
        uplo::Val,
        diag::Val,
        side::Val{SIDE},
    ) where {T, I, SIDE}
    #
    #   C = [ C₁ ] res(f)
    #       [ C₂ ] sep(f)
    #
    #
    # fres is the residual at node f
    #
    #     fres = res(f)
    #
    fres = neighbors(res, f)
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)

    if C isa AbstractVector
        C₁ = view(C, fres)
    elseif SIDE === :L
        C₁ = view(C, fres, oneto(nrhs))
    else
        C₁ = view(C, oneto(nrhs), fres)
    end
    #
    #   C₁ ← L₁₁* C₁
    #
    if C isa AbstractVector
        strsx!(s, side, trans, uplo, diag, D₁₁, C₁)
    else
        strsx_mt!(s, side, trans, uplo, diag, D₁₁, C₁, pool, nt)
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
            sscatteradd!(s, trans, C, M₂, fsep)
        else
            sscatteradd!(s, trans, C, M₂, fsep, side)
        end
    end

    return
end

# ===== strsx_fwd_1! =====

function strsx_fwd_1!(
        s::AbstractSemiring,
        C::AbstractVecOrMat{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nrhs::I,
        f::I,
        trans::Val{TRANS},
        uplo::Val{UPLO},
        diag::Val{DIAG},
        side::Val{SIDE},
    ) where {T, I, TRANS, UPLO, SIDE, DIAG}
    #
    # na is the size of the separator at node f
    #
    #     na = | sep(f) |
    #
    na = eltypedegree(sep, f)
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)
    Dp = Dptr[f]
    Lp = Lptr[f]
    Rp = pointers(res)[f]
    #
    #          res(f)
    #     L = [ d₁₁ ] res(f)
    #         [ l₂₁ ] sep(f)
    #
    d₁₁ = Dval[Dp]
    l₂₁ = view(Lval, Lp:Lp + na - one(I))
    #
    #   c₁ ← d₁₁* c₁
    #
    if !isintegral(s) && DIAG === :N
        v = sstar(s, d₁₁)

        if C isa AbstractVector
            if SIDE === :L
                C[Rp] = sprod(s, v, C[Rp], trans, Val(:N))
            else
                C[Rp] = sprod(s, C[Rp], v, Val(:N), trans)
            end
        else
            @inbounds for k in oneto(nrhs)
                if SIDE === :L
                    C[Rp, k] = sprod(s, v, C[Rp, k], trans, Val(:N))
                else
                    C[k, Rp] = sprod(s, C[k, Rp], v, Val(:N), trans)
                end
            end
        end
    end

    if ispositive(na)
        #
        #   M₂ ← l₂₁ c₁       C₂ ← C₂ + M₂
        #
        if C isa AbstractVector
            v = C[Rp]

            @inbounds for i in oneto(na)
                if SIDE === :L
                    C[fsep[i]] = smuladd(s, l₂₁[i], v, C[fsep[i]], trans, Val(:N))
                else
                    C[fsep[i]] = smuladd(s, v, l₂₁[i], C[fsep[i]], Val(:N), trans)
                end
            end
        else
            if SIDE === :L
                @inbounds for k in oneto(nrhs)
                    v = C[Rp, k]

                    for i in oneto(na)
                        C[fsep[i], k] = smuladd(s, l₂₁[i], v, C[fsep[i], k], trans, Val(:N))
                    end
                end
            else
                Z = sizeof(T)
                sC = stride(C, 2)

                @preserve C begin
                    pC = pointer(C)
                    pr = pC + (Rp - one(I)) * sC * Z

                    @inbounds for i in oneto(na)
                        saxpy_kern!(s, Val(:N), trans, Val(:R), pC + (fsep[i] - one(I)) * sC * Z, pr, l₂₁[i], nrhs)
                    end
                end
            end
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
        f::I,
        trans::Val{TRANS},
        uplo::Val{UPLO},
        diag::Val,
        side::Val{SIDE},
    ) where {T, I, TRANS, UPLO, SIDE}
    #
    # nn is the size of the residual at node f
    #
    #     nn = | res(f) |
    #
    nn = eltypedegree(res, f)
    #
    # na is the size of the separator at node f
    #
    #     na = | sep(f) |
    #
    na = eltypedegree(sep, f)
    Dp = Dptr[f]
    Lp = Lptr[f]
    #
    #          res(f) sep(f)
    #     U = [ D₁₁    U₁₂ ] res(f)
    #
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :U
        U₁₂ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    else
        U₁₂ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    end

    strsx_bwd_upd!(s, C, Mval, D₁₁, U₁₂, res, sep, na, nrhs, pool, nt, f, trans, uplo, diag, side)

    return
end

function strsx_bwd_upd!(
        s::AbstractSemiring,
        C::AbstractVecOrMat{T},
        Mval::AbstractVector{T},
        D₁₁::AbstractMatrix{T},
        U₁₂::AbstractMatrix{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        na::I,
        nrhs::I,
        pool,
        nt::Integer,
        f::I,
        trans::Val,
        uplo::Val,
        diag::Val,
        side::Val{SIDE},
    ) where {T, I, SIDE}
    #
    #   C = [ C₁ ] res(f)
    #       [ C₂ ] sep(f)
    #
    #
    # fres is the residual at node f
    #
    #     fres = res(f)
    #
    fres = neighbors(res, f)
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)

    if C isa AbstractVector
        C₁ = view(C, fres)
    elseif SIDE === :L
        C₁ = view(C, fres, oneto(nrhs))
    else
        C₁ = view(C, oneto(nrhs), fres)
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
            copygatherrec!(M₂, C, fsep)
        else
            copygatherrec!(M₂, C, fsep, side)
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
        strsx!(s, side, trans, uplo, diag, D₁₁, C₁)
    else
        strsx_mt!(s, side, trans, uplo, diag, D₁₁, C₁, pool, nt)
    end

    return
end


# ===== strsx_bwd_1! =====

function strsx_bwd_1!(
        s::AbstractSemiring,
        C::AbstractVecOrMat{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nrhs::I,
        f::I,
        trans::Val{TRANS},
        uplo::Val{UPLO},
        diag::Val{DIAG},
        side::Val{SIDE},
    ) where {T, I, TRANS, UPLO, SIDE, DIAG}
    #
    # na is the size of the separator at node f
    #
    #     na = | sep(f) |
    #
    na = eltypedegree(sep, f)
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)
    Dp = Dptr[f]
    Lp = Lptr[f]
    Rp = pointers(res)[f]
    #
    #          res(f) sep(f)
    #     U = [ d₁₁    u₁₂ ] res(f)
    #
    d₁₁ = Dval[Dp]
    u₁₂ = view(Lval, Lp:Lp + na - one(I))

    if ispositive(na)
        #
        #   M₂ ← C₂       c₁ ← u₁₂ M₂ + c₁
        #
        if C isa AbstractVector
            v = C[Rp]

            @inbounds for i in oneto(na)
                if SIDE === :L
                    v = smuladd(s, u₁₂[i], C[fsep[i]], v, trans, Val(:N))
                else
                    v = smuladd(s, C[fsep[i]], u₁₂[i], v, Val(:N), trans)
                end
            end

            C[Rp] = v
        else
            if SIDE === :L
                @inbounds for k in oneto(nrhs)
                    v = C[Rp, k]

                    for i in oneto(na)
                        v = smuladd(s, u₁₂[i], C[fsep[i], k], v, trans, Val(:N))
                    end

                    C[Rp, k] = v
                end
            elseif nrhs <= STRSX_1_NB
                @inbounds for k in oneto(nrhs)
                    v = C[k, Rp]

                    for i in oneto(na)
                        v = smuladd(s, C[k, fsep[i]], u₁₂[i], v, Val(:N), trans)
                    end

                    C[k, Rp] = v
                end
            else
                Z = sizeof(T)
                sC = stride(C, 2)

                @preserve C begin
                    pC = pointer(C)
                    pr = pC + (Rp - one(I)) * sC * Z

                    @inbounds for i in oneto(na)
                        saxpy_kern!(s, Val(:N), trans, Val(:R), pr, pC + (fsep[i] - one(I)) * sC * Z, u₁₂[i], nrhs)
                    end
                end
            end
        end
    end
    #
    #   c₁ ← d₁₁* c₁
    #
    if !isintegral(s) && DIAG === :N
        v = sstar(s, d₁₁)

        if C isa AbstractVector
            if SIDE === :L
                C[Rp] = sprod(s, v, C[Rp], trans, Val(:N))
            else
                C[Rp] = sprod(s, C[Rp], v, Val(:N), trans)
            end
        else
            @inbounds for k in oneto(nrhs)
                if SIDE === :L
                    C[Rp, k] = sprod(s, v, C[Rp, k], trans, Val(:N))
                else
                    C[k, Rp] = sprod(s, C[k, Rp], v, Val(:N), trans)
                end
            end
        end
    end

    return
end
