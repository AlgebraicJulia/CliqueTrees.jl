# ===== scopyto! =====

function scopyto!(s::AbstractSemiring, A::ChordalTriangular{<:Any, <:Any, T}, B::SparseMatrixCSC) where {T}
    fill!(A, szero(s, T))
    return copy_scatter!(A, B)
end

# ===== slu! =====

function slu!(s::AbstractSemiring, L::ChordalTriangular{<:Any, :L, T, I}, U::ChordalTriangular{<:Any, :U, T, I}) where {T, I}
    S = L.S

    Fval = FVector{T}(undef, S.nFval * S.nFval)
    Mptr = FVector{I}(undef, S.nMptr)
    Mval = FVector{T}(undef, S.nMval)

    res = S.res
    rel = S.rel
    chd = S.chd

    ns = zero(I); Mptr[one(I)] = one(I)

    for j in vertices(res)
        ns = slu_loop!(s, L.Dval, U.Dval, L.Lval, U.Lval, S.Dptr, S.Lptr, Mptr, Mval, Fval, res, rel, chd, ns, j)
    end

    return L, U
end

function slu_loop!(
        s::AbstractSemiring,
        LDval::AbstractVector{T},
        UDval::AbstractVector{T},
        LLval::AbstractVector{T},
        ULval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
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
    na = eltypedegree(rel, j)
    #
    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na
    #
    # F is the frontal matrix at node j
    #
    #           nn  na
    #     F = [ F₁₁ F₁₂ ] nn
    #         [ F₂₁ F₂₂ ] na
    #
    F = reshape(view(Fval, oneto(nj * nj)), nj, nj)
    F₁₁ = view(F, oneto(nn),      oneto(nn))
    F₂₁ = view(F, nn + one(I):nj, oneto(nn))
    F₁₂ = view(F, oneto(nn),      nn + one(I):nj)
    F₂₂ = view(F, nn + one(I):nj, nn + one(I):nj)

    Dp = Dptr[j]
    Lp = Lptr[j]
    LD = reshape(view(LDval, Dp:Dp + nn * nn - one(I)), nn, nn)
    UD = reshape(view(UDval, Dp:Dp + nn * nn - one(I)), nn, nn)
    LL = reshape(view(LLval, Lp:Lp + nn * na - one(I)), na, nn)
    UL = reshape(view(ULval, Lp:Lp + nn * na - one(I)), nn, na)
    #
    #     F₁₁ ← L₁₁ + U₁₁
    #     F₂₁ ← L₂₁
    #     F₁₂ ← U₁₂
    #     F₂₂ ← 0
    #
    slu_gather!(F₁₁, LD, UD)
    copyrec!(F₂₁, LL)
    copyrec!(F₁₂, UL)
    fill!(F₂₂, szero(s, T))

    for i in Iterators.reverse(neighbors(chd, j))
        slu_send!(s, F, Mptr, Mval, rel, ns, i)
        ns -= one(I)
    end
    #
    #     F₁₁ ← L₁₁ + U₁₁       (F₁₁* = U₁₁* L₁₁*)
    #
    slu!(s, F₁₁)

    if ispositive(na)
        #
        #     F₂₁ ← F₂₁ U₁₁*
        #     F₁₂ ← L₁₁* F₁₂
        #
        strsx!(s, Val(:R), Val(:U), F₁₁, F₂₁)
        strsx!(s, Val(:L), Val(:L), F₁₁, F₁₂)
        #
        #     M₂₂ ← F₂₂
        #     M₂₂ ← F₂₁ F₁₂ + M₂₂
        #
        ns += one(I)
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * na
        M₂₂ = reshape(view(Mval, strt:stop - one(I)), na, na)
        copyrec!(M₂₂, F₂₂)
        sgemx!(s, M₂₂, F₂₁, F₁₂)
    end
    #
    #     L₁₁ ← F₁₁    U₁₁ ← F₁₁
    #     L₂₁ ← F₂₁    U₁₂ ← F₁₂
    #
    copyrec!(LD, F₁₁)
    copyrec!(UD, F₁₁)
    copyrec!(LL, F₂₁)
    copyrec!(UL, F₁₂)

    return ns
end

function slu_gather!(F::AbstractMatrix, LD::AbstractMatrix, UD::AbstractMatrix)
    n = size(F, 1)

    @inbounds for j in 1:n
        for i in 1:n
            if i > j
                F[i, j] = LD[i, j]
            else
                F[i, j] = UD[i, j]
            end
        end
    end

    return F
end

function slu_send!(s::AbstractSemiring, F::AbstractMatrix, Mptr::AbstractVector{I}, Mval::AbstractVector, rel::AbstractGraph{I}, ns::I, i::I) where {I}
    #
    # na is the size of the separator at node i
    #
    #     na = | sep(i) |
    #
    na = eltypedegree(rel, i)
    #
    # inj is the subset inclusion
    #
    #     inj: sep(i) → bag(parent(i))
    #
    inj = neighbors(rel, i)
    #
    # M is the update matrix from child i
    #
    strt = Mptr[ns]
    M = reshape(view(Mval, strt:strt + na * na - one(I)), na, na)
    #
    # add M to F
    #
    #     F ← F + inj M injᵀ
    #
    @inbounds for w in oneto(na)
        iw = inj[w]

        for v in oneto(na)
            iv = inj[v]
            F[iv, iw] = splus(s, F[iv, iw], M[v, w])
        end
    end

    return
end
