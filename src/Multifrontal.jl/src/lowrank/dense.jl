function LinearAlgebra.lowrankupdate!(F::ChordalFactorization, v::AbstractVector, w::Real=true)
    lowrankupdate!!(triangular(F), F.d, mul!(similar(v, promote_eltype(F, v)), F.P, v), w)
    return F
end

function LinearAlgebra.lowrankdowndate!(F::ChordalFactorization, v::AbstractVector, w::Real=true)
    lowrankupdate!(F, v, -w)
    return F
end

function lowrankupdate!!(F::ChordalTriangular{:N, UPLO, T, I}, v::AbstractVector{T}, w::Real) where {UPLO, T, I}
    lowrankupdate!!(F, Ones{T}(ncl(F)), v, w)
    return zero(I)
end

function lowrankupdate!!(F::ChordalTriangular{DIAG, UPLO, T, I}, d::AbstractVector{T}, v::AbstractVector{T}, w::Real) where {DIAG, UPLO, T, I}
    return lowrank!!(F, d, v, convert(real(T), w))
end

function lowrankdowndate!!(F::ChordalTriangular{DIAG, UPLO, T, I}, v::AbstractVector{T}, w::Real=true) where {DIAG, UPLO, T, I}
    return lowrankupdate!!(F, v, -w)
end

function lowrank!!(F::ChordalTriangular{DIAG, UPLO, T, I}, d::AbstractVector{T}, v::AbstractVector{T}, w::Real) where {DIAG, UPLO, T, I}
    @assert !iszero(w)
    @assert length(v) == ncl(F)

    Pval = FVector{I}(undef, nfr(F))
    Kval = FVector{T}(undef, ne(F.S.rel))
    Sval = FVector{T}(undef, 3 * F.S.nFval)
    Fval = FVector{T}(undef, F.S.nFval + F.S.nNval)
    Mval = FVector{T}(undef, F.S.nNval)

    j = findbag(F.S, v); n = zero(I)

    while !iszero(j)
        n += one(I); Pval[n] = j; j = F.S.pnt[j]
    end

    path = view(Pval, oneto(n))

    lowrank_copy!(Kval, path, v, F.S.res, F.S.rel)
    lowrank_impl!(Kval, Fval, Mval, Sval, path, v, d, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, F.S.res, F.S.rel, convert(T, w), F.uplo, F.diag)

    return zero(I)
end

function lowrank_copy!(
        Kval::AbstractVector{T},
        path::AbstractVector{I},
        Cval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
    ) where {T, I <: Integer}

    j = zero(I)

    for i in Iterators.reverse(path)
        if !iszero(j)
            na = eltypedegree(rel, i)
            nn = eltypedegree(res, j)

            inj = neighbors(rel, i)
            cj = view(Cval, neighbors(res, j))
            ki = view(Kval, incident(rel, i))
            kj = view(Kval, incident(rel, j))

            for s in oneto(na)
                is = inj[s]

                if is <= nn
                    ki[s] = cj[is]
                else
                    ki[s] = kj[is - nn]
                end
            end
        end

        j = i
    end

    return
end

function lowrank_impl!(
        Kval::AbstractVector{T},
        Fval::AbstractVector{T},
        Mval::AbstractVector{T},
        Sval::AbstractVector{T},
        path::AbstractVector{I},
        Cval::AbstractVector{T},
        d::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        w::T,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {T, I <: Integer, UPLO, DIAG}
    #
    # the running scale α is the reciprocal of the update weight
    #
    #   L D Lᴴ + w v vᴴ,    α₀ = 1 / w
    #
    α = inv(w)

    for k in oneto(I(length(path)))
        α = lowrank_loop!(Kval, Fval, Mval, Sval, path, k, Cval, d, Dptr, Dval, Lptr, Lval, res, rel, α, uplo, diag)
    end

    return
end

function lowrank_loop!(
        Kval::AbstractVector{T},
        Fval::AbstractVector{T},
        Mval::AbstractVector{T},
        Sval::AbstractVector{T},
        path::AbstractVector{I},
        k::I,
        Cval::AbstractVector{T},
        d::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        α::T,
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {T, I <: Integer, UPLO, DIAG}

    j = path[k]
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
    # c is part of the rank-1 vector
    #
    #     c := [ c₁ ] res(j)
    #          [ k₂ ] sep(j)
    #
    c₁ = view(Cval, neighbors(res, j))
    k₂ = view(Kval, incident(rel, j))
    #
    # d₁ is the diagonal for res(j)
    #
    d₁ = view(d, neighbors(res, j))
    #
    # D₁₁ and L₂₁ are the diagonal and off-diagonal blocks
    #
    #     L := [ D₁₁ ] res(j)
    #          [ L₂₁ ] sep(j)
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + na * nn - one(I)), na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    # f is the frontal vector at node j
    #
    #     f := [ f₁ ] nn
    #          [ f₂ ] na
    #
    p = zero(I)
    f  = view(Fval, p + one(I):p + nn + na)
    f₁ = view(Fval, p + one(I):p + nn); p += nn
    f₂ = view(Fval, p + one(I):p + na); p += na
    #
    #     f₁ ← c₁
    #
    copyrec!(f₁, c₁)
    #
    #     f₂ ← 0
    #
    zerorec!(f₂)

    if !isone(k)
        #
        # i is the child preceding j on the path
        #
        i = path[k - one(I)]
        #
        # m is the update vector from child i
        #
        m = view(Mval, oneto(eltypedegree(rel, i)))
        #
        # add m into f
        #
        #     f ← f + Rᵢ m
        #
        addscatterrec!(f, m, neighbors(rel, i), Val(:L))
    end
    #
    # m₂ is the update vector from j
    #
    m₂ = view(Mval, oneto(na))
    #
    #     c₁ ← f₁
    #
    copyrec!(c₁, f₁)
    #
    #     m₂ ← f₂
    #
    copyrec!(m₂, f₂)
    #
    # update D₁₁, L₂₁, d₁, and m₂
    #
    return lowrank_kernel!(D₁₁, L₂₁, d₁, c₁, k₂, m₂, Sval, α, uplo, diag)
end

#
# Given a unit-lower-triangular matrix D, a matrix L, and
# vectors d, c, k, and m, compute E, F, and e such that
#
#     [ D ] diag(d) [ Dᵀ Lᵀ ] ± [  c  ] [ cᵀ kᵀ+mᵀ ] = [ E ] diag(e) [ Eᵀ Fᵀ ]
#     [ L ]                     [ k+m ]                [ F ]
#
# over-writing D with E, L with F, and d with e. Additionally, the
# vector m is over-written with the difference m ← m - L D⁻¹ c.
function lowrank_kernel!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        d::AbstractVector{T},
        c::AbstractVector{T},
        k::AbstractVector{T},
        m::AbstractVector{T},
        ::AbstractVector{T},
        α::T,
        uplo::Val{:L},
        diag::Val{DIAG},
    ) where {T, DIAG}
    @assert size(D, 1) == size(D, 2) == size(L, 2) == length(c)
    @assert size(L, 1) == length(k) == length(m)

    @inbounds for j in axes(L, 2)
        cj = c[j]

        if DIAG === :U
            dj = d[j]
        else
            rj = D[j, j]
            dj = rj^2
        end

        β = α * dj + cj^2

        @assert !iszero(β / α)

        if DIAG === :U
            d[j] = β / α
            p = cj
            h = cj / β
            α = β / dj
        else
            sj = D[j, j] = sqrt(β / α)
            h = sj * cj / β

            if iszero(rj)
                p = zero(T)
                g = zero(T)
            else
                p = cj / rj
                g = sj / rj
                α = β / dj
            end
        end

        for i in j + 1:size(L, 2)
            ci = c[i] -= p * D[i, j]

            if DIAG === :U
                D[i, j] += h * ci
            else
                D[i, j] = g * D[i, j] + h * ci
            end
        end

        for i in axes(L, 1)
            mi = m[i] -= p * L[i, j]

            if DIAG === :U
                L[i, j] += h * (k[i] + mi)
            else
                L[i, j] = g * L[i, j] + h * (k[i] + mi)
            end
        end
    end

    return α
end

function lowrank_kernel!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        d::AbstractVector{T},
        c::AbstractVector{T},
        k::AbstractVector{T},
        m::AbstractVector{T},
        Sval::AbstractVector{T},
        α::T,
        uplo::Val{:U},
        diag::Val{DIAG},
    ) where {T, DIAG}
    @assert size(D, 1) == size(D, 2) == size(L, 1) == length(c)
    @assert size(L, 2) == length(k) == length(m)

    n = size(D, 1)
    p = view(Sval,         1:n)
    g = view(Sval,     n + 1:2n)
    h = view(Sval, 2n + 1:3n)

    @inbounds for i in axes(D, 2)
        ci = c[i]

        for j in 1:i - 1
            ci -= p[j] * D[j, i]

            if DIAG === :U
                D[j, i] += h[j] * ci
            else
                D[j, i] = g[j] * D[j, i] + h[j] * ci
            end
        end

        c[i] = ci

        if DIAG === :U
            di = d[i]
        else
            ri = D[i, i]
            di = ri^2
        end

        β = α * di + ci^2

        @assert !iszero(β / α)

        if DIAG === :U
            d[i] = β / α
            p[i] = ci
            h[i] = ci / β
            α = β / di
        else
            si = D[i, i] = sqrt(β / α)
            h[i] = si * ci / β

            if iszero(ri)
                p[i] = zero(T)
                g[i] = zero(T)
            else
                p[i] = ci / ri
                g[i] = si / ri
                α = β / di
            end
        end
    end

    @inbounds for i in axes(L, 2)
        mi = m[i]
        ki = k[i]

        for j in axes(L, 1)
            mi -= p[j] * L[j, i]

            if DIAG === :U
                L[j, i] += h[j] * (ki + mi)
            else
                L[j, i] = g[j] * L[j, i] + h[j] * (ki + mi)
            end
        end

        m[i] = mi
    end

    return α
end

function findbag(S::ChordalSymbolic{I}, w::AbstractVector{T}) where {T, I}
    for j in vertices(S.res)
        for i in neighbors(S.res, j)
            if !iszero(w[i])
                res = neighbors(S.res, j)
                sep = neighbors(S.sep, j)

                for k in i + one(I):nov(S.res)
                    @assert iszero(w[k]) || k in res || insorted(k, sep)
                end

                return j
            end
        end
    end

    return zero(I)
end
