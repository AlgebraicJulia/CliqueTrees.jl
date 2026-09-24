# ===== sgetrf! =====

function sgetrf!(s::AbstractSemiring, L::ChordalTriangular{:N, :L, T, I}, U::ChordalTriangular{:N, :U, T, I}; nt::Integer = nthreads()) where {T, I}
    W = FactorizationWorkspace(L)
    pool = spool_mt(T, nt)
    return sgetrf_mt!(s, L, U, W, pool, nt)
end

function sgetrf_mt!(s::AbstractSemiring, L::ChordalTriangular{:N, :L, T, I}, U::ChordalTriangular{:N, :U, T, I}, W::FactorizationWorkspace{T, I}, pool, nt::Integer) where {T, I}
    S = L.S

    Fval = W.Fval
    Mptr = W.Mptr
    Mval = W.Mval

    res = S.res
    rel = S.rel
    chd = S.chd

    ns = zero(I); Mptr[one(I)] = one(I)

    for j in vertices(res)
        nn = eltypedegree(res, j)

        if isone(nn)
            ns = sgetrf_loop_1!(s, L.Dval, U.Dval, L.Lval, U.Lval, S.Dptr, S.Lptr, Mptr, Mval, Fval, res, rel, chd, ns, j)
        else
            ns = sgetrf_loop!(s, L.Dval, U.Dval, L.Lval, U.Lval, S.Dptr, S.Lptr, Mptr, Mval, Fval, res, rel, chd, pool, nt, ns, j)
        end
    end

    return L, U
end

function sgetrf_loop!(
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
        pool::Channel,
        nt::Integer,
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
    #     F ← 0
    #
    szerorec!(s, F, Val(:N))

    for i in Iterators.reverse(neighbors(chd, j))
        sgetrf_send!(s, F, Mptr, Mval, rel, ns, i)
        ns -= one(I)
    end
    #
    #     L₁₁ ← L₁₁ + U₁₁ + F₁₁       (L₁₁* = U₁₁* L₁₁*)
    #     U₁₁ ← L₁₁
    #
    @inbounds for j in oneto(nn)
        for i in oneto(nn)
            if i > j
                LD[i, j] = splus(s, F₁₁[i, j], LD[i, j], Val(:N))
            else
                LD[i, j] = splus(s, F₁₁[i, j], UD[i, j], Val(:N))
            end
        end
    end

    sgetrf_mt!(s, LD, pool, nt)
    copytri!(UD, LD, Val(:U))

    if ispositive(na)
        #
        #     L₂₁ ← L₂₁ + F₂₁    U₁₂ ← U₁₂ + F₁₂
        #
        @inbounds for c in oneto(nn)
            for r in oneto(na)
                LL[r, c] = splus(s, LL[r, c], F₂₁[r, c], Val(:N))
            end
        end

        @inbounds for c in oneto(na)
            for r in oneto(nn)
                UL[r, c] = splus(s, UL[r, c], F₁₂[r, c], Val(:N))
            end
        end
        #
        #     L₂₁ ← L₂₁ U₁₁*
        #     U₁₂ ← L₁₁* U₁₂
        #
        strsx_mt!(s, Val(:R), Val(:N), Val(:U), Val(:N), LD, LL, pool, nt)
        strsx_mt!(s, Val(:L), Val(:N), Val(:L), Val(:U), LD, UL, pool, nt)
        #
        #     M₂₂ ← F₂₂
        #     M₂₂ ← L₂₁ U₁₂ + M₂₂
        #
        ns += one(I)
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * na
        M₂₂ = reshape(view(Mval, strt:stop - one(I)), na, na)
        copyrec!(M₂₂, F₂₂)
        sgemx_mt!(s, Val(:N), Val(:N), M₂₂, LL, UL, pool, nt)
    end

    return ns
end

function sgetrf_loop_1!(
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
    # nn = 1 (the size of the residual at node j)
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
    f₂₁ = view(F, nn + one(I):nj, one(I))
    f₁₂ = view(F, one(I),         nn + one(I):nj)
    F₂₂ = view(F, nn + one(I):nj, nn + one(I):nj)

    Dp = Dptr[j]
    Lp = Lptr[j]
    l₂₁ = view(LLval, Lp:Lp + na - one(I))
    u₁₂ = view(ULval, Lp:Lp + na - one(I))
    #
    #     F ← 0
    #
    szerorec!(s, F, Val(:N))

    for i in Iterators.reverse(neighbors(chd, j))
        sgetrf_send!(s, F, Mptr, Mval, rel, ns, i)
        ns -= one(I)
    end
    #
    #     f₁₁ ← u₁₁ + f₁₁
    #
    d₁₁ = splus(s, UDval[Dp], F[one(I)], Val(:N))
    #
    #     l₁₁ ← f₁₁    u₁₁ ← f₁₁
    #
    LDval[Dp] = d₁₁
    UDval[Dp] = d₁₁

    if ispositive(na)
        #
        #     l₂₁ ← l₂₁ + f₂₁    u₁₂ ← u₁₂ + f₁₂
        #
        @inbounds for i in oneto(na)
            l₂₁[i] = splus(s, l₂₁[i], f₂₁[i], Val(:N))
            u₁₂[i] = splus(s, u₁₂[i], f₁₂[i], Val(:N))
        end
        #
        #     l₂₁ ← l₂₁ u₁₁*
        #
        if !isintegral(s)
            ds = sstar(s, d₁₁)

            @inbounds for i in oneto(na)
                l₂₁[i] = sprod(s, l₂₁[i], ds, Val(:N), Val(:N))
            end
        end
        #
        #     M₂₂ ← F₂₂ + l₂₁ u₁₂
        #
        ns += one(I)
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * na
        M₂₂ = reshape(view(Mval, strt:stop - one(I)), na, na)

        copyto!(M₂₂, F₂₂)
        sger!(s, Val(:N), Val(:N), Val(:R), l₂₁, u₁₂, M₂₂)
    end

    return ns
end

function sgetrf_send!(s::AbstractSemiring, F::AbstractMatrix, Mptr::AbstractVector{I}, Mval::AbstractVector, rel::AbstractGraph{I}, ns::I, i::I) where {I}
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
            F[iv, iw] = splus(s, F[iv, iw], M[v, w], Val(:N))
        end
    end

    return
end
