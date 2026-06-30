function LinearAlgebra.nullspace(F::AbstractFactorization{DIAG, UPLO, T}; atol::Real = 0, rtol::Real = atol > 0 ? 0 : ncl(F) * eps(real(T))) where {DIAG, UPLO, T}
    null_impl(triangular(F), F.d, F.perm, convert(T, atol), convert(T, rtol))
end

function null_impl(L::ChordalTriangular{DIAG, UPLO, T, I}, d::AbstractVector{T}, perm::AbstractVector{I}, atol::T, rtol::T) where {DIAG, UPLO, T, I}
    S = L.S

    maxdiag = zero(T)

    if DIAG === :N
        for j in vertices(S.res)
            D, _ = diagblock(L, j)

            for i in axes(D, 1)
                maxdiag = max(maxdiag, abs(D[i, i]))
            end
        end
    else
        for j in axes(L, 1)
            maxdiag = max(maxdiag, abs(d[j]))
        end
    end

    tol = max(atol, rtol * maxdiag)

    mrkptr = FVector{I}(undef, nfr(S) + 1)
    mrktgt = FVector{I}(undef, ncl(S))
    nanc = FVector{I}(undef, nfr(S))
    anc = FVector{I}(undef, nfr(S))
    Cptr = FVector{I}(undef, nfr(S) + 1)

    mrk, nXval, nFval, nMval = null_symb!(mrkptr, mrktgt, nanc, anc, Cptr, L, d, tol)

    Xval = FVector{T}(undef, nXval)
    Mptr = FVector{I}(undef, S.nMptr)
    Mval = FVector{T}(undef, nMval)
    Fval = FVector{T}(undef, nFval)

    null_impl!(Xval, mrk, nanc, Cptr, Mptr, Mval, Fval, L, L.uplo, L.diag)

    N = Matrix{T}(undef, ncl(S), ne(mrk))

    return null_half!(N, Xval, mrk, nanc, anc, Cptr, S.res, S.idx, perm)
end


function null_symb!(
        mrkptr::AbstractVector{I},
        mrktgt::AbstractVector{I},
        nanc::AbstractVector{I},
        anc::AbstractVector{I},
        Cptr::AbstractVector{I},
        L::ChordalTriangular{DIAG, UPLO, T, I},
        d::AbstractVector{T},
        tol::T,
    ) where {DIAG, UPLO, T, I}
    p = zero(I)

    for j in vertices(L.S.res)
        mrkptr[j] = p + one(I)
        nn = eltypedegree(L.S.res, j)
        rng = neighbors(L.S.res, j)

        if DIAG === :N
            D, _ = diagblock(L, j)
            dj = view(D, diagind(D))
        else
            dj = view(d, rng)
        end

        for i in oneto(nn)
            if -tol <= dj[i] <= tol
                p += one(I); mrktgt[p] = rng[i]
            end
        end
    end

    mrkptr[nv(L.S.res) + one(I)] = p + one(I)

    mrk = BipartiteGraph(nov(L.S.res), nv(L.S.res), p, mrkptr, mrktgt)

    for i in reverse(vertices(L.S.res))
        j = L.S.pnt[i]

        if iszero(j)
            nanc[i] = zero(I)
            anc[i] = zero(I)
        else
            nanc[i] = eltypedegree(mrk, j) + nanc[j]

            if ispositive(eltypedegree(mrk, j))
                anc[i] = j
            else
                anc[i] = anc[j]
            end
        end
    end

    p = ns = nFval = nMval = zero(I)

    for j in vertices(L.S.res)
        nn = eltypedegree(L.S.res, j)
        na = eltypedegree(L.S.sep, j)
        nj = nn + na

        mn = eltypedegree(mrk, j)
        ma = nanc[j]
        mj = mn + ma

        ln = nn - mn
        Cptr[j] = p + one(I); p += ln * mj

        for i in neighbors(L.S.chd, j)
            ns -= nanc[i] * eltypedegree(L.S.sep, i)
        end

        ns += ma * na

        nFval = max(nFval, (ln + na) * mj)
        nMval = max(nMval, ns)
    end

    Cptr[nv(L.S.res) + one(I)] = p + one(I)

    nXval = p

    return mrk, nXval, nFval, nMval
end

function null_impl!(
        Xval::AbstractVector{T},
        mrk::AbstractGraph{I},
        nanc::AbstractVector{I},
        Cptr::AbstractVector{I},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Fval::AbstractVector{T},
        L::ChordalTriangular{DIAG, UPLO, T, I},
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {DIAG, UPLO, T, I}
    S = L.S

    res = S.res
    rel = S.rel
    chd = S.chd
    Dptr = S.Dptr
    Dval = L.Dval
    Lptr = S.Lptr
    Lval = L.Lval

    ns = zero(I)
    Mptr[one(I)] = one(I)

    for j in reverse(vertices(res))
        ns = null_loop!(Xval, mrk, nanc, Cptr, Mptr, Mval, Fval,
                             Dptr, Dval, Lptr, Lval, res, rel, chd,
                             ns, convert(I, j), uplo, diag)
    end

    return Xval
end

function null_loop!(
        Xval::AbstractVector{T},
        mrk::AbstractGraph{I},
        nanc::AbstractVector{I},
        Cptr::AbstractVector{I},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Fval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {DIAG, UPLO, T, I}
    nn = eltypedegree(res, j)
    return null_loop_snd!(Xval, mrk, nanc, Cptr, Mptr, Mval, Fval,
                          Dptr, Dval, Lptr, Lval, res, rel, chd,
                          ns, nn, j, uplo, diag)
end

function null_loop_snd!(
        Xval::AbstractVector{T},
        mrk::AbstractGraph{I},
        nanc::AbstractVector{I},
        Cptr::AbstractVector{I},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Fval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        nn::I,
        j::I,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {DIAG, UPLO, T, I}
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
    # mj is the number of marked rows at node j
    #
    #     mj = mn + ma
    #
    # mn is the number of marked rows owned by node j
    #
    # ma is the number of marked rows inherited from ancestors
    #
    mn = eltypedegree(mrk, j)
    ma = nanc[j]
    mj = mn + ma
    #
    # ln is the number of leading (non-null) columns
    #
    #     ln = nn - mn
    #
    ln = nn - mn
    #
    # L factor blocks at front j
    #
    # Block indices: 1 = leading (ln), 2 = null (mn), 3 = separator (na)
    #
    #        n
    #   L = [ Dₙₙ ] n  (residual)
    #       [ L₃ₙ ] 3  (separator)
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    Dₙₙ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₃ₙ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        L₃ₙ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    # Dₙₙ block views:
    #
    #          1     2                      1     2
    #   Dₙₙ = [ D₁₁  0 ] 1  (lower)  Dₙₙ = [ D₁₁  D₁₂ ] 1  (upper)
    #         [ D₂₁  I ] 2                 [ 0    I   ] 2
    #
    D₁₁ = view(Dₙₙ, one(I):ln, one(I):ln)

    if UPLO === :L
        D₂₁ = view(Dₙₙ, ln + one(I):nn, one(I):ln)
    else
        D₁₂ = view(Dₙₙ, one(I):ln, ln + one(I):nn)
    end
    #
    # L₃ₙ block views (L₃₂ = 0):
    #
    if UPLO === :L
        L₃₁ = view(L₃ₙ, one(I):na, one(I):ln)
    else
        L₃₁ = view(L₃ₙ, one(I):ln, one(I):na)
    end
    #
    # C₁ is the compressed output block at front j (only left columns stored)
    #
    # Row indices: 2 = owned (mn), 3 = inherited (ma)
    #
    #        1
    #   C₁ = [ C₂₁ ] 2 (owned)
    #        [ C₃₁ ] 3 (inherited)
    #
    # The right block [I; 0] is reconstructed on-the-fly in null_half!
    #
    Cp = Cptr[j]
    C₁ = reshape(view(Xval, Cp:Cp + mj * ln - one(I)), mj, ln)
    C₂₁ = view(C₁, one(I):mn, one(I):ln)
    C₃₁ = view(C₁, mn + one(I):mj, one(I):ln)
    #
    # F is the compact frontal matrix at node j (right block omitted)
    #
    #        1       3
    #   F = [ F₁    F₃ ] mj
    #
    # The right block [I; 0] is handled on-the-fly in null_update!
    #
    fj = ln + na
    F = reshape(view(Fval, one(I):mj * fj), mj, fj)
    F₁ = view(F, one(I):mj, one(I):ln)
    F₃ = view(F, one(I):mj, ln + one(I):fj)
    F₂₃ = view(F₃, one(I):mn, one(I):na)
    F₃₃ = view(F₃, mn + one(I):mj, one(I):na)
    #
    # ancestor coupling + F₃ assembly
    #
    #     C₃₁ ← −M₃₃ L₃₁
    #     F₃ ← [ 0 ; M₃₃ ]
    #
    if ispositive(na)
        strt = Mptr[ns]; ns -= one(I)

        if ispositive(ma)
            M₃₃ = reshape(view(Mval, strt:strt + ma * na - one(I)), ma, na)

            if UPLO === :L
                gemm!(Val(:N), Val(:N), -one(T), M₃₃, L₃₁, zero(T), C₃₁)
            else
                gemm!(Val(:N), Val(:T), -one(T), M₃₃, L₃₁, zero(T), C₃₁)
            end

            copyrec!(F₃₃, M₃₃)
        end

        zerorec!(F₂₃)
    else
        zerorec!(C₃₁)
    end
    #
    # triangular solve (exploiting block structure)
    #
    #          1     2                   1              2
    #   Dₙₙ = [ D₁₁  0 ] 1     Dₙₙ⁻¹ = [ D₁₁⁻¹         0 ] 1
    #         [ D₂₁  I ] 2             [ -D₂₁ D₁₁⁻¹    I ] 2
    #
    #                   1             2
    #   Cₙ Dₙₙ⁻¹ = [ -D₂₁ D₁₁⁻¹  |  I ] 2 (mn)
    #              [  C₃₁ D₁₁⁻¹  |  0 ] 3 (ma)
    #
    # (C₃₂ = 0, so C₃₂ D₂₁ = 0)
    #
    if ispositive(ln)
        if UPLO === :L
            @inbounds for j in axes(D₂₁, 2)
                for i in axes(D₂₁, 1)
                    C₂₁[i, j] = -D₂₁[i, j]
                end
            end
        else
            @inbounds for j in axes(D₁₂, 1)
                for i in axes(D₁₂, 2)
                    C₂₁[i, j] = -D₁₂[j, i]
                end
            end
        end
        #
        # C₁ ← C₁ D₁₁⁻¹  (lower)
        # C₁ ← C₁ D₁₁⁻ᵀ  (upper)
        #
        if ispositive(mj)
            if UPLO === :L
                trsm!(Val(:R), uplo, Val(:N), diag, one(T), D₁₁, C₁)
            else
                trsm!(Val(:R), uplo, Val(:T), diag, one(T), D₁₁, C₁)
            end
        end
    end
    #
    # assemble F₁
    #
    copyrec!(F₁, C₁)
    #
    # push F restricted to sep(i) to each child i
    #
    #     Mᵢ ← Rᵢᵀ F
    #
    for i in neighbors(chd, j)
        ns += one(I)
        null_update!(F, Mptr, Mval, rel, ns, ln, nn, mj, i)
    end

    return ns
end

function null_update!(
        F::AbstractMatrix{T},
        ptr::AbstractVector{I},
        val::AbstractVector{T},
        rel::AbstractGraph{I},
        ns::I,
        ln::I,
        nn::I,
        mj::I,
        i::I,
    ) where {T, I}
    #
    # na = |sep(i)|
    #
    na = eltypedegree(rel, i)
    #
    # inj: sep(i) → bag(parent(i))
    #
    inj = neighbors(rel, i)
    #
    # Mᵢ ← Rᵢᵀ F
    #
    strt = ptr[ns]
    stop = ptr[ns + one(I)] = strt + mj * na

    M = reshape(view(val, strt:stop - one(I)), mj, na)

    @inbounds for k in oneto(na)
        col = inj[k]

        if col <= ln
            #
            # left block: read from F₁
            #
            for r in oneto(mj)
                M[r, k] = F[r, col]
            end
        elseif col <= nn
            #
            # right block: [I; 0] where I is mn × mn
            #
            for r in oneto(mj)
                M[r, k] = zero(T)
            end

            M[col - ln, k] = one(T)
        else
            #
            # separator block: read from F₃ (at shifted index)
            #
            fcol = ln + (col - nn)

            for r in oneto(mj)
                M[r, k] = F[r, fcol]
            end
        end
    end

    return
end

function null_half!(
        N::AbstractMatrix{T},
        Xval::AbstractVector{T},
        mrk::AbstractGraph{I},
        nanc::AbstractVector{I},
        anc::AbstractVector{I},
        Cptr::AbstractVector{I},
        res::AbstractGraph{I},
        idx::AbstractVector{I},
        perm::AbstractVector{I},
    ) where {T, I}
    fill!(N, zero(T))

    for j in vertices(res)
        nn = eltypedegree(res, j)
        mn = eltypedegree(mrk, j)
        ln = nn - mn
        mj = mn + nanc[j]
        vstrt = pointers(res)[j]
        Cp = Cptr[j]
        #
        # left block: walk ancestors with marks, write all positions
        #
        i = j
        k = zero(I)

        while !iszero(i)
            mi = eltypedegree(mrk, i)
            wstrt = pointers(mrk)[i]

            for w in one(I):mi
                c = wstrt + w - one(I)

                for loc in one(I):ln
                    p = vstrt + loc - one(I)
                    N[perm[p], c] = Xval[Cp + (loc - one(I)) * mj + k + w - one(I)]
                end
            end

            k += mi
            i = anc[i]
        end
        #
        # right block: identity for owned marks
        #
        wstrt = pointers(mrk)[j]

        for w in one(I):mn
            c = wstrt + w - one(I)
            loc = ln + w
            p = vstrt + loc - one(I)
            N[perm[p], c] = one(T)
        end
    end

    return N
end
