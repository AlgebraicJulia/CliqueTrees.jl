# ===== strtri! =====

function strtri!(
        s::AbstractSemiring,
        U::ChordalTriangular{<:Any, :U, T, I},
        X::AbstractMatrix;
        nt::Integer = nthreads(),
    ) where {T, I}
    S = U.S

    fdesc = FVector{I}(undef, nfr(S))
    Tval = FVector{T}(undef, S.nFval * S.nFval)
    Mval = FVector{T}(undef, S.nFval * ncl(S))
    pool = spool_mt(T, nt)

    return strtri_mt!(s, U, X, fdesc, Tval, Mval, pool, nt)
end

function strtri_mt!(
        s::AbstractSemiring,
        U::ChordalTriangular{<:Any, :U, T, I},
        X::AbstractMatrix,
        fdesc::AbstractVector{I},
        Tval::AbstractVector{T},
        Mval::AbstractVector{T},
        pool,
        nt::Integer,
    ) where {T, I}
    S = U.S
    #
    # fdesc: F → F maps each front f ∈ F to its
    # first descendant fdesc(f) ∈ F.
    #
    for f in vertices(S.res)
        fdesc[f] = f
    end

    for f in vertices(S.res)
        p = S.pnt[f]

        if ispositive(p)
            fdesc[p] = min(fdesc[p], fdesc[f])
        end
    end

    szerorec!(s, X, Val(:N))

    for f in vertices(S.res)
        strtri_fwd!(s, X, Mval, Tval, U.Dval, U.Lval, S.Dptr, S.Lptr, S.res, S.sep, fdesc, pool, nt, f)
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
        fdesc::AbstractVector{I},
        pool,
        nt::Integer,
        f::I,
    ) where {T, I}
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
    Qp = pointers(res)[fdesc[f]]
    #
    # fdsc are the descendants at node f
    #
    #     fdsc = dsc(f)
    #
    fdsc = Qp:Rp + nn - one(I)
    #
    # nd is the number of descendants at node f
    #
    #     nd = | dsc(f) |
    #
    nd = length(fdsc)
    #
    #          res(f) sep(f)
    #     U = [ D₁₁    U₁₂ ] res(f)
    #
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)
    U₁₂ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    #
    #          res(f) sep(f)
    #     X = [ X₀₁    X₀₂ ] dsc(f)
    #         [ X₁₁    X₁₂ ] res(f)
    #
    X₁ = view(X, fdsc, fres)

    Y₁₁ = reshape(view(Tval, oneto(nn * nn)), nn, nn)
    #
    #   Y₁₁ ← D₁₁
    #
    copytri!(Y₁₁, D₁₁, Val(:U))
    #
    #   Y₁₁ ← Y₁₁*
    #
    strtri!(s, Val(:U), Val(:N), Y₁₁; nt)
    #
    #   X₁₁ ← Y₁₁
    #
    copyscattertri!(X, Y₁₁, fres, Val(:U))
    #
    #   X₀₁ ← X₀₁ D₁₁*
    #
    if Qp < Rp
        X₀₁ = view(X, Qp:Rp - one(I), fres)
        strsx_mt!(s, Val(:R), Val(:N), Val(:U), Val(:N), D₁₁, X₀₁, pool, nt)
    end

    if ispositive(na)
        M₂ = reshape(view(Mval, oneto(nd * na)), nd, na)
        #
        #   M₂ ← 0
        #
        szerorec!(s, M₂, Val(:N))
        #
        #   M₂ ← X₁ U₁₂
        #
        sgemx_mt!(s, Val(:N), Val(:N), M₂, X₁, U₁₂, pool, nt)
        #
        #   X₂ ← X₂ + M₂
        #
        sscatteradd!(s, view(X, fdsc, axes(X, 2)), M₂, fsep, Val(:R))
    end

    return
end
