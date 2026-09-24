function sgetrp!(F::ChordalSLU)
    sgetrp!(F.s, lowertriangular(F), uppertriangular(F))
    return F
end

function sgetrp!(
        s::AbstractSemiring,
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I};
        nt::Integer = nthreads(),
    ) where {T, I <: Integer}
    Mptr = FVector{I}(undef, L.S.nMptr)
    Mval = FVector{T}(undef, L.S.nMval)
    Fval = FVector{T}(undef, L.S.nFval * L.S.nFval)
    pool = spool_mt(T, nt)

    return sgetrp_mt!(s, L, U, Mptr, Mval, Fval, pool, nt)
end

function sgetrp_mt!(
        s::AbstractSemiring,
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Fval::AbstractVector{T},
        pool::Channel,
        nt::Integer,
    ) where {T, I <: Integer}
    S = L.S

    LDval = L.Dval
    UDval = U.Dval
    LLval = L.Lval
    ULval = U.Lval

    Dptr = S.Dptr
    Lptr = S.Lptr

    res = S.res
    rel = S.rel
    chd = S.chd

    ns = zero(I); Mptr[one(I)] = one(I)

    for j in reverse(vertices(res))
        nn = eltypedegree(res, j)

        if isone(nn)
            ns = sgetrp_loop_1!(s, Mptr, Mval, Dptr, LDval, UDval, Lptr, LLval, ULval, Fval, res, rel, chd, ns, j)
        else
            ns = sgetrp_loop!(s, Mptr, Mval, Dptr, LDval, UDval, Lptr, LLval, ULval, Fval, res, rel, chd, pool, nt, ns, j)
        end
    end

    return L, U
end

function sgetrp_loop!(
        s::AbstractSemiring,
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Dptr::AbstractVector{I},
        LDval::AbstractVector{T},
        UDval::AbstractVector{T},
        Lptr::AbstractVector{I},
        LLval::AbstractVector{T},
        ULval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        pool::Channel,
        nt::Integer,
        ns::I,
        j::I,
    ) where {T, I <: Integer}
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
    F₁₂ = view(F, oneto(nn),      nn + one(I):nj)
    F₂₁ = view(F, nn + one(I):nj, oneto(nn))
    F₂₂ = view(F, nn + one(I):nj, nn + one(I):nj)
    #
    # L and U are parts of the triangular factors
    #
    #          res(j)                    res(j) sep(j)
    #     L = [ L₁₁  ] res(j)       U = [ U₁₁   U₁₂ ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    L₁₁ = reshape(view(LDval, Dp:Dp + nn * nn - one(I)), nn, nn)
    U₁₁ = reshape(view(UDval, Dp:Dp + nn * nn - one(I)), nn, nn)
    L₂₁ = reshape(view(LLval, Lp:Lp + nn * na - one(I)), na, nn)
    U₁₂ = reshape(view(ULval, Lp:Lp + nn * na - one(I)), nn, na)

    if ispositive(na)
        #
        # M₂₂ is the update matrix from the parent of node j
        #
        strt = Mptr[ns]
        M₂₂ = reshape(view(Mval, strt:strt + na * na - one(I)), na, na)
        ns -= one(I)
        #
        #     F₂₂ ← M₂₂
        #
        copyrec!(F₂₂, M₂₂)
        #
        #     F₂₁ ← (F₂₂ L₂₁) L₁₁*
        #
        szerorec!(s, F₂₁, Val(:N))
        sgemx_mt!(s, Val(:N), Val(:N), F₂₁, M₂₂, L₂₁, pool, nt)
        strsx_mt!(s, Val(:R), Val(:N), Val(:L), Val(:U), L₁₁, F₂₁, pool, nt)
        #
        #     F₁₂ ← U₁₁* (U₁₂ F₂₂)
        #
        szerorec!(s, F₁₂, Val(:N))
        sgemx_mt!(s, Val(:N), Val(:N), F₁₂, U₁₂, M₂₂, pool, nt)
        strsx_mt!(s, Val(:L), Val(:N), Val(:U), Val(:N), U₁₁, F₁₂, pool, nt)
    end
    #
    #     F₁₁ ← L₁₁*
    #
    strtri_mt!(s, Val(:L), Val(:U), L₁₁, pool, nt)
    szerorec!(s, F₁₁, Val(:N))

    @inbounds for k in oneto(nn)
        F₁₁[k, k] = sone(s, T, Val(:N))

        for i in k + one(I):nn
            F₁₁[i, k] = L₁₁[i, k]
        end
    end
    #
    #     F₁₁ ← U₁₁* (L₁₁* + U₁₂ F₂₁)
    #
    if ispositive(na)
        sgemx_mt!(s, Val(:N), Val(:N), F₁₁, U₁₂, F₂₁, pool, nt)
    end

    strsx_mt!(s, Val(:L), Val(:N), Val(:U), Val(:N), U₁₁, F₁₁, pool, nt)
    #
    #     L₁₁ ← F₁₁, U₁₁ ← F₁₁
    #
    copyrec!(L₁₁, F₁₁)
    copyrec!(U₁₁, F₁₁)
    #
    #     L₂₁ ← F₂₁, U₁₂ ← F₁₂
    #
    copyrec!(L₂₁, F₂₁)
    copyrec!(U₁₂, F₁₂)

    for i in neighbors(chd, j)
        #
        # send update matrix to child i
        #
        #     Mᵢ ← Rᵢᵀ F Rᵢ
        #
        ns += one(I)
        sgetrp_send!(F, Mptr, Mval, rel, ns, i)
    end

    return ns
end

function sgetrp_loop_1!(
        s::AbstractSemiring,
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Dptr::AbstractVector{I},
        LDval::AbstractVector{T},
        UDval::AbstractVector{T},
        Lptr::AbstractVector{I},
        LLval::AbstractVector{T},
        ULval::AbstractVector{T},
        Fval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
    ) where {T, I <: Integer}
    #
    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    nn = one(I)
    #
    # na is the size of the separator at node j
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(rel, j)
    #
    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) | = 1 + na
    #
    nj = nn + na
    #
    # F is the frontal matrix at node j
    #
    #           1   na
    #     F = [ f₁₁ f₁₂ ] 1
    #         [ f₂₁ F₂₂ ] na
    #
    F = reshape(view(Fval, oneto(nj * nj)), nj, nj)

    f₁₂ = view(F, one(I), nn + one(I):nj)
    f₂₁ = view(F, nn + one(I):nj, one(I))
    F₂₂ = view(F, nn + one(I):nj, nn + one(I):nj)
    #
    # L and U are parts of the triangular factors
    #
    #          1                         1    na
    #     L = [ l₁₁ ] 1            U = [ u₁₁   u₁₂ ] 1
    #         [ l₂₁ ] na
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    u₁₁ = sstar(s, UDval[Dp])
    l₂₁ = view(LLval, Lp:Lp + na - one(I))
    u₁₂ = view(ULval, Lp:Lp + na - one(I))
    #
    #     z₁₁ ← 1
    #
    z₁₁ = sone(s, T, Val(:N))

    if ispositive(na)
        #
        # M₂₂ is the update matrix from the parent of node j
        #
        strt = Mptr[ns]
        M₂₂ = reshape(view(Mval, strt:strt + na * na - one(I)), na, na)
        ns -= one(I)
        #
        #     F₂₂ ← M₂₂
        #
        copyrec!(F₂₂, M₂₂)
        #
        #     z₂₁ ← F₂₂ l₂₁
        #
        szerorec!(s, f₂₁, Val(:N))
        sgemx!(s, Val(:N), Val(:N), f₂₁, M₂₂, l₂₁)
        #
        #     z₁₂ ← u₁₁* (u₁₂ F₂₂)
        #
        szerorec!(s, f₁₂, Val(:N))
        sgemx!(s, Val(:N), Val(:N), f₁₂, u₁₂, M₂₂)

        @inbounds for k in oneto(na)
            f₁₂[k] = sprod(s, u₁₁, f₁₂[k], Val(:N), Val(:N))
        end
        #
        #     z₁₁ ← 1 + u₁₂ z₂₁
        #
        z₁₁ = splus(s, z₁₁, sdot(s, Val(:N), Val(:N), u₁₂, f₂₁), Val(:N))
    end
    #
    #     z₁₁ ← u₁₁* z₁₁
    #
    z₁₁ = sprod(s, u₁₁, z₁₁, Val(:N), Val(:N))
    #
    #     l₁₁ ← z₁₁, u₁₁ ← z₁₁
    #
    LDval[Dp] = z₁₁
    UDval[Dp] = z₁₁
    F[one(I)] = z₁₁
    #
    #     l₂₁ ← z₂₁, u₁₂ ← z₁₂
    #
    copyrec!(l₂₁, f₂₁)
    copyrec!(u₁₂, f₁₂)

    for i in neighbors(chd, j)
        #
        # send update matrix to child i
        #
        #     Mᵢ ← Rᵢᵀ F Rᵢ
        #
        ns += one(I)
        sgetrp_send!(F, Mptr, Mval, rel, ns, i)
    end

    return ns
end

function sgetrp_send!(
        F::AbstractMatrix{T},
        ptr::AbstractVector{I},
        val::AbstractVector{T},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
    ) where {T, I <: Integer}
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
    # M is the update matrix for child i
    #
    strt = ptr[ns]
    stop = ptr[ns + one(I)] = strt + na * na
    M = reshape(view(val, strt:stop - one(I)), na, na)
    #
    # copy F into M
    #
    #     M ← Rᵢᵀ F Rᵢ
    #
    @inbounds for b in oneto(na), a in oneto(na)
        M[a, b] = F[inj[a], inj[b]]
    end

    return
end
