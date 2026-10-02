const STRTRI_SPLIT = 4

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

    nf = nfr(S)
    nc = ncl(S)
    bs = cld(nc, STRTRI_SPLIT * nt)

    fdsc = FVector{I}(undef, nf)
    bptr = FVector{I}(undef, nf + 1)
    #
    # fdsc: F → F maps each front f ∈ F to its
    # first descendant fdsc(f) ∈ F.
    #
    @inbounds for f in vertices(S.res)
        fdsc[f] = f
    end

    @inbounds for f in vertices(S.res)
        p = S.pnt[f]

        if ispositive(p)
            fdsc[p] = min(fdsc[p], fdsc[f])
        end
    end
    #
    # the cth cut contains the fronts
    #
    #   bptr[c] ... bptr[c + 1] - 1 ⊆ F
    #
    hmax = zero(I)
    n = 1; bptr[1] = one(I); strt = zero(I)

    for f in vertices(S.res)
        stop = pointers(S.res)[f + one(I)] - one(I)

        if stop - strt >= bs
            hmax = max(hmax, stop - strt)
            n += 1; bptr[n] = f + one(I); strt = stop
        end
    end

    if strt < nc
        hmax = max(hmax, nc - strt)
        n += 1; bptr[n] = nf + one(I)
    end

    nb = n - 1
    nw = min(nt, nb)

    if nw <= 1
        Mval = FVector{T}(undef, max(S.nFval * nc, one(I)))
        strtri_band!(s, diag, A, X, fdsc, Mval, pool, nt, one(I), nf)
    else
        @threads for w in 1:nw
            tstrt = fld(nt * (w - 1), nw) + 1
            tstop = fld(nt *  w,      nw)
            poolw = view(pool, tstrt:tstop)
            strtri_task!(s, diag, A, X, fdsc, poolw, w, nw, tstop - tstrt + 1, bptr, nb, hmax)
        end
    end

    return X
end

function strtri_task!(
        s::AbstractSemiring,
        diag::Val,
        A::ChordalTriangular{<:Any, <:Any, T, I},
        X::AbstractMatrix,
        fdsc::AbstractVector{I},
        pool::AbstractVector,
        w::Int,
        nw::Int,
        nt::Int,
        bptr::AbstractVector{I},
        nb::Int,
        hmax::I,
    ) where {T, I}
    S = A.S

    Mval = FVector{T}(undef, max(S.nFval * hmax, one(I)))

    for k in w:nw:nb
        strtri_band!(s, diag, A, X, fdsc, Mval, pool, nt, bptr[k], bptr[k + 1] - one(I))
    end

    return
end

# ===== strtri_band! =====

function strtri_band!(
        s::AbstractSemiring,
        diag::Val,
        A::ChordalTriangular{<:Any, UPLO, T, I},
        X::AbstractMatrix,
        fdsc::AbstractVector{I},
        Mval::AbstractVector{T},
        pool::AbstractVector,
        nt::Integer,
        fstrt::I,
        fstop::I,
    ) where {UPLO, T, I}
    S = A.S
    res = S.res
    bstrt = pointers(res)[fstrt]
    bstop = pointers(res)[fstop + one(I)] - one(I)

    if UPLO === :L
        nrhs = convert(I, size(X, 1))
    else
        nrhs = convert(I, size(X, 2))
    end
    #
    #   X ← 0
    #
    if UPLO === :L
        szerorec!(s, view(X, oneto(nrhs), bstrt:bstop), Val(:N))
    else
        szerorec!(s, view(X, bstrt:bstop, oneto(nrhs)), Val(:N))
    end

    for f in fstrt:nv(res)
        #
        # the descendants of f in the band
        #
        #     rstrt ... rstop = dsc(f) ∩ band
        #
        rstrt = pointers(res)[fdsc[f]]
        rstop = pointers(res)[f + one(I)] - one(I)

        rstrt = max(rstrt, bstrt)
        rstop = min(rstop, bstop)

        if rstrt ≤ rstop
            strtri_fwd!(s, X, Mval, A.Dval, A.Lval, S.Dptr, S.Lptr, res, S.sep, pool, nt, f, A.uplo, diag, rstrt, rstop)
        end
    end

    return X
end

# ===== strtri_fwd! =====

function strtri_fwd!(
        s::AbstractSemiring,
        X::AbstractMatrix{T},
        Mval::AbstractVector{T},
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
        diag::Val{DIAG},
        rstrt::I,
        rstop::I,
    ) where {T, I, UPLO, DIAG}
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
    # fdsc is the descendants of f in the band
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

    if Rp <= rstop
        X₁₁ = view(X, fres, fres)
        #
        #   X₁₁ ← D₁₁
        #
        copytri!(X₁₁, D₁₁, uplo)
        #
        #   X₁₁ ← X₁₁*
        #
        strtri_mt!(s, uplo, diag, X₁₁, pool, nt)

        if DIAG === :U
            @inbounds for v in fres
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
