struct NullspaceWorkspace{T, I}
    idx::FVector{I}
    head::FVector{I}
    next::FVector{I}
    Fval::FVector{T}
end

function NullspaceWorkspace{T}(S::ChordalSymbolic{I}, nrhs::Integer) where {T, I <: Integer}
    idx  = FVector{I}(undef, ncl(S))
    head = FVector{I}(undef, nfr(S))
    next = FVector{I}(undef, max(convert(I, nrhs), one(I)))
    Fval = FVector{T}(undef, max(S.nFval * convert(I, nrhs), one(I)))
    return NullspaceWorkspace{T, I}(idx, head, next, Fval)
end

function NullspaceWorkspace{T}(L::ChordalTriangular, nrhs::Integer) where {T}
    return NullspaceWorkspace{T}(L.S, nrhs)
end

function LinearAlgebra.nullspace(F::AbstractFactorization{DIAG, UPLO, T}; kw...) where {DIAG, UPLO, T}
    L = triangular(F)
    d = F.d

    k = nullity(L, d, nulltol(L, d; kw...))
    W = NullspaceWorkspace{T}(L, k)
    C = Matrix{T}(undef, ncl(L), k)
    nullspace!(C, W, L, d; kw...)

    return F.P \ C
end

function nullspace!(C::AbstractMatrix{T}, W::NullspaceWorkspace{T, I}, L::ChordalTriangular{DIAG, UPLO, T, I}; kw...) where {DIAG, UPLO, T, I}
    d = Ones{T}(ncl(L))
    return nullspace!(C, W, L, d; kw...)
end

function nullspace!(C::AbstractMatrix{T}, W::NullspaceWorkspace{T, I}, L::ChordalTriangular{DIAG, UPLO, T, I}, d::AbstractVector{T}; kw...) where {DIAG, UPLO, T, I}
    tol = nulltol(L, d; kw...)
    k = nullsym!(W.idx, W.head, W.next, L, d, tol)
    null_impl!(C, W, L, L.uplo, L.diag, k)
    return k
end

function null_impl!(C, W, L, uplo, diag, nrhs)
    S = L.S
    @assert size(C, 1) >= ncl(S)
    @assert size(C, 2) >= nrhs
    @assert length(W.next) >= nrhs
    @assert length(W.Fval) >= S.nFval * nrhs

    res = S.res
    sep = S.sep
    Dptr = S.Dptr
    Dval = L.Dval
    Lptr = S.Lptr
    Lval = L.Lval

    idx = W.idx
    head = W.head
    next = W.next
    Fval = W.Fval

    fill!(C, false)

    if ispositive(nrhs)
        for f in reverse(vertices(res))
            null_loop!(C, Fval, Dptr, Dval, Lptr, Lval, res, sep,
                       idx, head, next, f, uplo, diag)
        end
    end

    return C
end

function null_loop!(
        C::AbstractMatrix{T},
        Fval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        idx::AbstractVector{I},
        head::AbstractVector{I},
        next::AbstractVector{I},
        f::I,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {T, I, UPLO, DIAG}
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
    # nanc is the number of columns of C that are
    # ancestors of f
    #
    nanc = zero(I)
    #
    # rank is the rank of Dₙₙ
    #
    rank = nn

    jhed = head[f]
    janc = zero(I)
    j = jhed

    @inbounds while !iszero(j)
        nanc += one(I)

        if idx[j] == f
            rank -= one(I)
        elseif iszero(janc)
            janc = j
        end

        j = next[j]
    end

    if ispositive(nanc)
        if isone(nn)
            null_loop_nod!(C, Dptr, Dval, Lptr, Lval, res, sep, next,
                           f, na, rank, jhed, janc, uplo, diag)
        else
            null_loop_snd!(C, Fval, Dptr, Dval, Lptr, Lval, res, sep, next,
                           f, nn, na, nanc, rank, jhed, janc, uplo, diag)
        end
    end

    return
end

function null_loop_snd!(
        C::AbstractMatrix{T},
        Fval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        next::AbstractVector{I},
        f::I,
        nn::I,
        na::I,
        nanc::I,
        rank::I,
        jhed::I,
        janc::I,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {T, I, UPLO, DIAG}
    if UPLO === :L
        trans = Val(:C)
    else
        trans = Val(:N)
    end

    Rp = pointers(res)[f]
    Sp = pointers(sep)[f]
    Dp = Dptr[f]
    Lp = Lptr[f]
    #
    # L is part of the factor
    #
    #           rank
    #     L = [ D₁₁ ] rank
    #         [ D₂₁ ] null
    #         [ L₃₁ ] sep
    #
    Dₙₙ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)
    D₁₁ = view(Dₙₙ, oneto(rank), oneto(rank))

    if UPLO === :L
        L₃ₙ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
        D₂₁ = view(Dₙₙ, rank + one(I):nn, oneto(rank))
        L₃₁ = view(L₃ₙ,        oneto(na), oneto(rank))
    else
        L₃ₙ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
        D₂₁ = view(Dₙₙ, oneto(rank), rank + one(I):nn)
        L₃₁ = view(L₃ₙ, oneto(rank),        oneto(na))
    end
    #
    # F is the frontal matrix at node f, packed into Fval:
    #
    #            null      anc - res
    #   F = [    F₁₂         F₁₃    ] rank
    #       [     I           0     ] null
    #       [     0          F₃₃    ] sep
    #
    F₁  = reshape(view(Fval, oneto(rank * nanc)), rank, nanc)
    F₁₂ = view(F₁, oneto(rank),        oneto(nn - rank))
    F₁₃ = view(F₁, oneto(rank), nn - rank + one(I):nanc)

    if ispositive(nanc + rank - nn) && ispositive(na)
        F₃₃ = reshape(view(Fval, rank * nanc + one(I):rank * nanc + na * (nanc + rank - nn)), na, nanc + rank - nn)
        #
        #     F₃₃ ← C₃₃
        #
        jloc = zero(I)
        j = janc

        @inbounds while !iszero(j)
            jloc += one(I)

            for iloc in oneto(na)
                i = targets(sep)[Sp + iloc - one(I)]

                F₃₃[iloc, jloc] = C[i, j]
            end

            j = next[j]
        end
        #
        #     F₁₃ ← -L₃₁ᴴ F₃₃
        #
        gemm!(trans, Val(:N), -one(T), L₃₁, F₃₃, zero(T), F₁₃)
    end
    #
    #     F₁₂ ← - D₂₁ᴴ
    #
    if ispositive(rank)
        @inbounds for jloc in oneto(nn - rank)
            for iloc in oneto(rank)
                if UPLO === :L
                    F₁₂[iloc, jloc] = -conj(D₂₁[jloc, iloc])
                else
                    F₁₂[iloc, jloc] = -D₂₁[iloc, jloc]
                end
            end
        end
        #
        #     F₁ ← D₁₁⁻ᴴ F₁
        #
        trsm!(Val(:L), uplo, trans, diag, one(T), D₁₁, F₁)
    end
    #
    # scatter F into C
    #
    jloc = zero(I)
    j = jhed

    @inbounds while !iszero(j)
        jloc += one(I)

        for iloc in oneto(rank)
            i = Rp + iloc - one(I)

            C[i, j] = F₁[iloc, jloc]
        end

        j = next[j]
    end

    i = Rp + rank - one(I)
    j = jhed

    @inbounds while !iszero(j) && j != janc
        i += one(I)

        C[i, j] = one(T)

        j = next[j]
    end

    return
end

function null_loop_nod!(
        C::AbstractMatrix{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        next::AbstractVector{I},
        f::I,
        na::I,
        rank::I,
        jhed::I,
        janc::I,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {T, I, UPLO, DIAG}

    Rp = pointers(res)[f]
    Sp = pointers(sep)[f]
    Dp = Dptr[f]
    Lp = Lptr[f]

    if iszero(rank)
        @inbounds C[Rp, jhed] = one(T)
    else
        d₁₁ = Dval[Dp]
        j = janc

        @inbounds while !iszero(j)
            acc = zero(T)

            for s in oneto(na)
                i = targets(sep)[Sp + s - one(I)]

                if UPLO === :L
                    acc += conj(Lval[Lp + s - one(I)]) * C[i, j]
                else
                    acc += Lval[Lp + s - one(I)] * C[i, j]
                end
            end

            if DIAG === :N
                if UPLO === :L
                    acc /= conj(d₁₁)
                else
                    acc /= d₁₁
                end
            end

            C[Rp, j] = -acc
            j = next[j]
        end
    end

    return
end

function nulltol(L::ChordalTriangular{DIAG}, d::AbstractVector; kw...) where {DIAG}
    if DIAG === :N
        return nulltol(L; kw...)
    else
        return nulltol(Diagonal(d); kw...)
    end
end

function nulltol(L::ChordalTriangular{DIAG, UPLO, T, I}; atol::Real = 0, rtol::Real = atol > 0 ? 0 : ncl(L.S) * eps(real(T))) where {DIAG, UPLO, T, I}
    S = L.S
    maxdiag = zero(real(T))

    for f in vertices(S.res)
        D, _ = diagblock(L, f)

        for i in axes(D, 1)
            maxdiag = max(maxdiag, abs(D[i, i]))
        end
    end

    return convert(T, max(atol, rtol * maxdiag * maxdiag))
end

function nulltol(D::Diagonal{T}; atol::Real = 0, rtol::Real = atol > 0 ? 0 : size(D, 1) * eps(real(T))) where {T}
    maxdiag = zero(real(T))

    for i in axes(D, 1)
        maxdiag = max(maxdiag, abs(D[i, i]))
    end

    return convert(T, max(atol, rtol * maxdiag))
end

function nullity(L::ChordalTriangular{DIAG}, d::AbstractVector, tol::Real) where {DIAG}
    if DIAG === :N
        return nullity(L, tol)
    else
        return nullity(Diagonal(d), tol)
    end
end

function nullity(L::ChordalTriangular, tol::Real)
    S = L.S
    nrhs = 0

    for f in vertices(S.res)
        D, _ = diagblock(L, f)

        for i in diagind(D)
            if abs2(D[i]) <= tol
                nrhs += 1
            end
        end
    end

    return nrhs
end

function nullity(D::Diagonal, tol::Real)
    nrhs = 0

    for i in axes(D, 1)
        if -tol <= D[i, i] <= tol
            nrhs += 1
        end
    end

    return nrhs
end

function nullsym!(
        idx::AbstractVector{I},
        head::AbstractVector{I},
        next::AbstractVector{I},
        L::ChordalTriangular{DIAG, UPLO, T, I},
        d::AbstractVector{T},
        tol::T,
    ) where {DIAG, UPLO, T, I}
    S = L.S; nrhs = zero(I)

    @inbounds for f in vertices(S.res)
        if DIAG === :N
            D, _ = diagblock(L, f)

            for i in diagind(D)
                if abs2(D[i]) <= tol
                    nrhs += one(I); idx[nrhs] = f
                end
            end
        else
            for i in neighbors(S.res, f)
                if -tol <= d[i] <= tol
                    nrhs += one(I); idx[nrhs] = f
                end
            end
        end
    end

    j = nrhs

    @inbounds for f in reverse(vertices(S.res))
        g = S.pnt[f]

        if iszero(g)
            k = zero(I)
        else
            k = head[g]
        end

        while ispositive(j) && idx[j] == f
            next[j] = k; k = j; j -= one(I)
        end

        head[f] = k
    end

    return nrhs
end
