const STRTRI_SPLIT = 4
const STRTRI_MINBAND = 256

# ===== strtri! =====

function strtri!(
        s::AbstractSemiring,
        diag::Val,
        A::ChordalTriangular{<:Any, UPLO, T, I},
        X::AbstractMatrix;
        nt::Integer = nthreads(),
    ) where {UPLO, T, I}
    pool = spool_mt(s, T, nt)

    return strtri_mt!(s, diag, A, X, pool, nt)
end

# ===== strtri_mt! =====

function strtri_mt!(
        s::AbstractSemiring,
        diag::Val,
        A::ChordalTriangular{<:Any, UPLO, T, I},
        X::AbstractMatrix,
        pool::AbstractVector,
        nt::Integer,
    ) where {UPLO, T, I}
    @assert nt >= 1

    S = A.S
    n = convert(I, ncl(S))
    #
    # fdesc: F → F maps each front f ∈ F to its
    # first descendant fdesc(f) ∈ F.
    #
    fdesc = FVector{I}(undef, nfr(S))

    @inbounds for f in vertices(S.res)
        fdesc[f] = f
    end

    @inbounds for f in vertices(S.res)
        p = S.pnt[f]

        if ispositive(p)
            fdesc[p] = min(fdesc[p], fdesc[f])
        end
    end

    nw = min(nt, max(1, ncl(S) ÷ STRTRI_MINBAND))

    if nw <= 1
        #
        # one band: the serial algorithm, with
        # fine-grain parallelism in the dense kernels
        #
        Tval = FVector{T}(undef, max(S.nFval * S.nFval, one(I)))
        Mval = FVector{T}(undef, max(S.nFval * n, one(I)))
        strtri_band!(s, diag, A, X, fdesc, Tval, Mval, pool, nt, one(I), n)
    else
        #
        # nb bands of (nearly) equal height, dealt to
        # nw workers cyclically: worker w gets bands
        # w, w + nw, w + 2nw, ...
        #
        nb = min(STRTRI_SPLIT * nw, ncl(S) ÷ STRTRI_MINBAND)
        bsize = convert(I, cld(n, nb))
        nb = convert(Int, cld(n, bsize))

        @threads for w in 1:nw
            strtri_worker!(s, diag, A, X, fdesc, pool, w, nw, nb, bsize)
        end
    end

    return X
end

function strtri_worker!(
        s::AbstractSemiring,
        diag::Val,
        A::ChordalTriangular{<:Any, <:Any, T, I},
        X::AbstractMatrix,
        fdesc::AbstractVector{I},
        pool::AbstractVector,
        w::Int,
        nw::Int,
        nb::Int,
        bsize::I,
    ) where {T, I}
    S = A.S
    n = convert(I, ncl(S))

    Tval = FVector{T}(undef, max(S.nFval * S.nFval, one(I)))
    Mval = FVector{T}(undef, max(S.nFval * bsize, one(I)))
    poolw = view(pool, w:w)

    for k in w:nw:nb
        bstrt = convert(I, k - 1) * bsize + one(I)
        bstop = min(convert(I, k) * bsize, n)
        strtri_band!(s, diag, A, X, fdesc, Tval, Mval, poolw, 1, bstrt, bstop)
    end

    return
end

# ===== strtri_band! =====

function strtri_band!(
        s::AbstractSemiring,
        diag::Val,
        A::ChordalTriangular{<:Any, UPLO, T, I},
        X::AbstractMatrix,
        fdesc::AbstractVector{I},
        Tval::AbstractVector{T},
        Mval::AbstractVector{T},
        pool::AbstractVector,
        nt::Integer,
        bstrt::I,
        bstop::I,
    ) where {UPLO, T, I}
    S = A.S
    res = S.res
    #
    #   X[band, :] ← 0
    #
    if UPLO === :L
        szerorec!(s, view(X, axes(X, 1), bstrt:bstop), Val(:N))
    else
        szerorec!(s, view(X, bstrt:bstop, axes(X, 2)), Val(:N))
    end

    for f in vertices(res)
        #
        # the descendant rows of f are
        #
        #     dsc(f) = Qp:Rq
        #
        Qp = pointers(res)[fdesc[f]]
        Rq = pointers(res)[f + one(I)] - one(I)

        if Qp <= bstop && bstrt <= Rq
            strtri_fwd!(s, X, Mval, Tval, A.Dval, A.Lval, S.Dptr, S.Lptr, res, S.sep, pool, nt, f, A.uplo, diag, max(Qp, bstrt), min(Rq, bstop))
        end
    end

    return X
end

# ===== strtri_fwd! =====

function strtri_fwd!(
        s::AbstractSemiring,
        X::AbstractMatrix{T},
        Mval::AbstractVector{T},
        Tval::AbstractVector{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        pool::AbstractVector,
        nt::Integer,
        f::I,
        uplo::Val{UPLO},
        diag::Val,
        rstrt::I,
        rstop::I,
    ) where {T, I, UPLO}
    if UPLO === :L
        nrhs = convert(I, size(X, 1))
    else
        nrhs = convert(I, size(X, 2))
    end
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

    Dp = Dptr[f]
    Lp = Lptr[f]
    Rp = pointers(res)[f]
    #
    # fdsc is the set of descendant rows of f in the band
    #
    #     fdsc = dsc(f) ∩ band
    #
    fdsc = rstrt:rstop
    #
    # nr is the number of those rows
    #
    #     nr = | fdsc |
    #
    nr = rstop - rstrt + one(I)
    #
    #          res(f) sep(f)
    #     U = [ D₁₁    U₁₂ ] res(f)
    #
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        U₁₂ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        U₁₂ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    #          res(f) sep(f)
    #     X = [ X₀₁    X₀₂ ] dsc(f) ∖ res(f)
    #         [ X₁₁    X₁₂ ] res(f)
    #
    # restricted to the rows fdsc
    #
    if UPLO === :L
        X₁ = view(X, fres, fdsc)
    else
        X₁ = view(X, fdsc, fres)
    end
    istrt = max(Rp, rstrt) - Rp + one(I)
    istop = rstop - Rp + one(I)

    if istrt <= istop
        Y₁₁ = reshape(view(Tval, oneto(nn * nn)), nn, nn)
        #
        #   Y₁₁ ← D₁₁
        #
        copytri!(Y₁₁, D₁₁, uplo)
        #
        #   Y₁₁ ← Y₁₁*
        #
        strtri!(s, uplo, diag, Y₁₁; nt)
        #
        #   X₁₁ ← Y₁₁
        #
        copyscattertri!(X, Y₁₁, fres, istrt, istop, uplo)

        if diag === Val(:U)
            @inbounds for i in istrt:istop
                v = fres[i]
                X[v, v] = sone(s, T, Val(:N))
            end
        end
    end
    #
    #   X₀₁ ← X₀₁ D₁₁*
    #
    qstop = min(Rp - one(I), rstop)

    if rstrt <= qstop
        if UPLO === :L
            X₀₁ = view(X, fres, rstrt:qstop)
            strsx_mt!(s, Val(:L), Val(:N), uplo, diag, D₁₁, X₀₁, pool, nt)
        else
            X₀₁ = view(X, rstrt:qstop, fres)
            strsx_mt!(s, Val(:R), Val(:N), uplo, diag, D₁₁, X₀₁, pool, nt)
        end
    end

    if ispositive(na)
        if UPLO === :L
            M₂ = reshape(view(Mval, oneto(nr * na)), na, nr)
        else
            M₂ = reshape(view(Mval, oneto(nr * na)), nr, na)
        end
        #
        #   M₂ ← 0
        #
        szerorec!(s, M₂, Val(:N))
        #
        #   M₂ ← X₁ U₁₂
        #
        if UPLO === :L
            sgemx_mt!(s, Val(:N), Val(:N), M₂, U₁₂, X₁, pool, nt)
        else
            sgemx_mt!(s, Val(:N), Val(:N), M₂, X₁, U₁₂, pool, nt)
        end
        #
        #   X₂ ← X₂ + M₂
        #
        if UPLO === :L
            sscatteradd!(s, view(X, oneto(nrhs), fdsc), M₂, fsep, Val(:L))
        else
            sscatteradd!(s, view(X, fdsc, oneto(nrhs)), M₂, fsep, Val(:R))
        end
    end

    return
end
