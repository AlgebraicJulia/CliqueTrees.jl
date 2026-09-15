const RowVector{T, V <: AbstractVector{T}} = Transpose{T, V}

# ===== ldiv! =====

function ldiv!(F::SemiringLU{Sem, T}, B::AbstractVecOrMat) where {Sem, T}
    if B isa AbstractVector
        nrhs = 1
    else
        nrhs = size(B, 2)
    end

    W = DivisionWorkspace{T}(F.S, nrhs)
    return ldiv!(W, F, B)
end

function ldiv!(W::DivisionWorkspace, F::SemiringLU{Sem, T}, B::AbstractVecOrMat) where {Sem, T}
    C = FArray{T}(undef, size(B))
    mul!(C, F.P, B)
    sldiv!(F.s, W, F.L, F.U, C)
    ldiv!(B, F.Q, C)
    return B
end

# ===== rdiv! =====

function rdiv!(B::AbstractMatrix, F::SemiringLU{Sem, T}) where {Sem, T}
    W = DivisionWorkspace{T}(F.S, size(B, 1))
    return rdiv!(W, B, F)
end

function rdiv!(W::DivisionWorkspace, B::AbstractMatrix, F::SemiringLU{Sem, T}) where {Sem, T}
    C = FMatrix{T}(undef, size(B))
    rdiv!(C, B, F.Q)
    srdiv!(F.s, W, C, F.L, F.U)
    mul!(B, C, F.P)
    return B
end

function rdiv!(W::DivisionWorkspace, bt::RowVector, F::SemiringLU{Sem, T}) where {Sem, T}
    b = parent(bt)
    c = FVector{T}(undef, length(b))
    mul!(c, F.Q, b)
    srdiv!(F.s, W, c, F.L, F.U)
    ldiv!(b, F.P, c)
    return bt
end

function Base.:/(bt::RowVector, F::SemiringLU)
    return rdiv!(transpose(copy(parent(bt))), F)
end

# ===== sldiv! =====

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
    S = L.S

    if B isa AbstractVector
        nrhs = one(I)
    else
        nrhs = convert(I, size(B, 2))
    end
    #
    #   B ← L* B
    #
    for j in vertices(S.res)
        sldiv_fwd!(s, B, W.Mval, L.Dval, L.Lval, S.Dptr, S.Lptr, S.res, S.sep, nrhs, j)
    end
    #
    #   B ← U* B
    #
    for j in reverse(vertices(S.res))
        sldiv_bwd!(s, B, W.Mval, U.Dval, U.Lval, S.Dptr, S.Lptr, S.res, S.sep, nrhs, j)
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
        nrhs::I,
        j::I,
    ) where {T, I}
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
    L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    #
    #   C = [ C₁ ] res(j)
    #       [ C₂ ] sep(j)
    #
    if C isa AbstractVector
        C₁ = view(C, neighbors(res, j))
    else
        C₁ = view(C, neighbors(res, j), oneto(nrhs))
    end
    #
    #   C₁ ← L₁₁* C₁
    #
    strsx!(s, Val(:L), Val(:L), D₁₁, C₁)

    if ispositive(na)
        if C isa AbstractVector
            M₂ = view(Mval, oneto(na))
        else
            M₂ = reshape(view(Mval, oneto(na * nrhs)), na, nrhs)
        end
        #
        #   M₂ ← L₂₁ C₁
        #
        fill!(M₂, szero(s, T))
        sgemx!(s, M₂, L₂₁, C₁)
        #
        #   C₂ ← C₂ + M₂
        #
        if C isa AbstractVector
            sscatteradd!(s, C, M₂, neighbors(sep, j))
        else
            sscatteradd!(s, C, M₂, neighbors(sep, j), Val(:L))
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
        nrhs::I,
        j::I,
    ) where {T, I}
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
    U₁₂ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
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
        if C isa AbstractVector
            M₂ = view(Mval, oneto(na))
        else
            M₂ = reshape(view(Mval, oneto(na * nrhs)), na, nrhs)
        end
        #
        #   M₂ ← C₂
        #
        if C isa AbstractVector
            copygatherrec!(M₂, C, neighbors(sep, j))
        else
            copygatherrec!(M₂, C, neighbors(sep, j), Val(:L))
        end
        #
        #   C₁ ← U₁₂ M₂ + C₁
        #
        sgemx!(s, C₁, U₁₂, M₂)
    end
    #
    #   C₁ ← U₁₁* C₁
    #
    strsx!(s, Val(:L), Val(:U), D₁₁, C₁)

    return
end

# ===== srdiv! =====

function srdiv!(
        s::AbstractSemiring,
        B::AbstractVecOrMat,
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
    ) where {T, I}
    if B isa AbstractVector
        nrhs = one(I)
    else
        nrhs = convert(I, size(B, 1))
    end

    W = DivisionWorkspace{T}(L.S, nrhs)
    return srdiv!(s, W, B, L, U)
end

function srdiv!(
        s::AbstractSemiring,
        W::DivisionWorkspace{T},
        B::AbstractVecOrMat,
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
    ) where {T, I}
    S = L.S

    if B isa AbstractVector
        nrhs = one(I)
    else
        nrhs = convert(I, size(B, 1))
    end
    #
    #   B ← B U*
    #
    for j in vertices(S.res)
        srdiv_fwd!(s, B, W.Mval, U.Dval, U.Lval, S.Dptr, S.Lptr, S.res, S.sep, nrhs, j)
    end
    #
    #   B ← B L*
    #
    for j in reverse(vertices(S.res))
        srdiv_bwd!(s, B, W.Mval, L.Dval, L.Lval, S.Dptr, S.Lptr, S.res, S.sep, nrhs, j)
    end

    return B
end

function srdiv_fwd!(
        s::AbstractSemiring,
        C::AbstractVecOrMat{T},
        Mval::AbstractVector{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nrhs::I,
        j::I,
    ) where {T, I}
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
    U₁₂ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    #
    #          res(j) sep(j)
    #     C = [  C₁     C₂ ]
    #
    if C isa AbstractVector
        C₁ = view(C, neighbors(res, j))
    else
        C₁ = view(C, oneto(nrhs), neighbors(res, j))
    end
    #
    #   C₁ ← C₁ U₁₁*
    #
    strsx!(s, Val(:R), Val(:U), D₁₁, C₁)

    if ispositive(na)
        if C isa AbstractVector
            M₂ = view(Mval, oneto(na))
        else
            M₂ = reshape(view(Mval, oneto(na * nrhs)), nrhs, na)
        end
        #
        #   M₂ ← C₁ U₁₂
        #
        fill!(M₂, szero(s, T))
        sgemx!(s, M₂, C₁, U₁₂)
        #
        #   C₂ ← C₂ + M₂
        #
        if C isa AbstractVector
            sscatteradd!(s, C, M₂, neighbors(sep, j))
        else
            sscatteradd!(s, C, M₂, neighbors(sep, j), Val(:R))
        end
    end

    return
end

function srdiv_bwd!(
        s::AbstractSemiring,
        C::AbstractVecOrMat{T},
        Mval::AbstractVector{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nrhs::I,
        j::I,
    ) where {T, I}
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
    L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    #
    #          res(j) sep(j)
    #     C = [  C₁     C₂ ]
    #
    if C isa AbstractVector
        C₁ = view(C, neighbors(res, j))
    else
        C₁ = view(C, oneto(nrhs), neighbors(res, j))
    end

    if ispositive(na)
        if C isa AbstractVector
            M₂ = view(Mval, oneto(na))
        else
            M₂ = reshape(view(Mval, oneto(na * nrhs)), nrhs, na)
        end
        #
        #   M₂ ← C₂
        #
        if C isa AbstractVector
            copygatherrec!(M₂, C, neighbors(sep, j))
        else
            copygatherrec!(M₂, C, neighbors(sep, j), Val(:R))
        end
        #
        #   C₁ ← M₂ L₂₁ + C₁
        #
        sgemx!(s, C₁, M₂, L₂₁)
    end
    #
    #   C₁ ← C₁ L₁₁*
    #
    strsx!(s, Val(:R), Val(:L), D₁₁, C₁)

    return
end
