function LinearAlgebra.lowrankupdate!(F::ChordalLDLt{UPLO, T, I}, v::AbstractVector{T}) where {UPLO, T, I}
    return _lowrankupdate!(F, v, one(T))
end

function LinearAlgebra.lowrankupdate!(F::ChordalCholesky{UPLO, T, I}, v::AbstractVector{T}) where {UPLO, T, I}
    return _lowrankupdate!(F, v, one(T))
end

function LinearAlgebra.lowrankdowndate!(F::ChordalLDLt{UPLO, T, I}, v::AbstractVector{T}) where {UPLO, T, I}
    return _lowrankupdate!(F, v, -one(T))
end

function LinearAlgebra.lowrankdowndate!(F::ChordalCholesky{UPLO, T, I}, v::AbstractVector{T}) where {UPLO, T, I}
    return _lowrankupdate!(F, v, -one(T))
end

function _lowrankupdate!(F::ChordalLDLt{UPLO, T, I}, v::AbstractVector{T}, σ::T) where {UPLO, T, I}
    @assert length(v) == nov(F.S.res)

    Pval = FVector{I}(undef, nv(F.S.rel))
    Cval = FVector{T}(undef, nov(F.S.res))
    Kval = FVector{T}(undef, ne(F.S.rel))
    Fval = FVector{T}(undef, F.S.nFval + F.S.nNval)
    Mval = FVector{T}(undef, F.S.nNval)

    for i in outvertices(F.S.res)
        Cval[i] = v[F.perm[i]]
    end

    j = findbag(F.S, Cval); n = zero(I)

    while !iszero(j)
        n += one(I); Pval[n] = j; j = F.S.pnt[j]
    end

    path = view(Pval, oneto(n))

    lowrank_copy!(Kval, path, Cval, F.S.res, F.S.rel)
    ldlt_lowrank_impl!(Kval, Fval, Mval, path, Cval, F.d, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, F.S.res, F.S.rel, σ, Val{UPLO}())

    return F
end

function _lowrankupdate!(F::ChordalCholesky{UPLO, T, I}, v::AbstractVector{T}, σ::T) where {UPLO, T, I}
    @assert length(v) == nov(F.S.res)

    Pval = FVector{I}(undef, nv(F.S.rel))
    Cval = FVector{T}(undef, nov(F.S.res))
    Kval = FVector{T}(undef, ne(F.S.rel))
    Fval = FVector{T}(undef, F.S.nFval + F.S.nNval)
    Mval = FVector{T}(undef, F.S.nNval)

    for i in outvertices(F.S.res)
        Cval[i] = v[F.perm[i]]
    end

    j = findbag(F.S, Cval); n = zero(I)

    while !iszero(j)
        n += one(I); Pval[n] = j; j = F.S.pnt[j]
    end

    path = view(Pval, oneto(n))

    lowrank_copy!(Kval, path, Cval, F.S.res, F.S.rel)
    chol_lowrank_impl!(Kval, Fval, Mval, path, Cval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, F.S.res, F.S.rel, σ, Val{UPLO}())

    return F
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

function ldlt_lowrank_impl!(
        Kval::AbstractVector{T},
        Fval::AbstractVector{T},
        Mval::AbstractVector{T},
        path::AbstractVector{I},
        Cval::AbstractVector{T},
        d::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        σ::T,
        uplo::Val{UPLO},
    ) where {T, I <: Integer, UPLO}

    path_len = length(path)
    α = one(T)

    for k in oneto(I(path_len))
        α = ldlt_lowrank_loop!(Kval, Fval, Mval, path, k, Cval, d, Dptr, Dval, Lptr, Lval, res, rel, α, σ, uplo)
    end

    return
end

function chol_lowrank_impl!(
        Kval::AbstractVector{T},
        Fval::AbstractVector{T},
        Mval::AbstractVector{T},
        path::AbstractVector{I},
        Cval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        σ::T,
        uplo::Val{UPLO},
    ) where {T, I <: Integer, UPLO}

    path_len = length(path)
    α = one(T)

    for k in oneto(I(path_len))
        α = chol_lowrank_loop!(Kval, Fval, Mval, path, k, Cval, Dptr, Dval, Lptr, Lval, res, rel, α, σ, uplo)
    end

    return
end

function ldlt_lowrank_loop!(
        Kval::AbstractVector{T},
        Fval::AbstractVector{T},
        Mval::AbstractVector{T},
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
        σ::T,
        uplo::Val{UPLO},
    ) where {T, I <: Integer, UPLO}

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
        addrec!(f, m, neighbors(rel, i))
    end
    #
    # m₂ is the update vector from j
    #
    m₂ = view(Mval, oneto(na))
    #
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
    return ldlt_davis_hager!(D₁₁, L₂₁, d₁, c₁, k₂, m₂, α, σ, uplo)
end

function chol_lowrank_loop!(
        Kval::AbstractVector{T},
        Fval::AbstractVector{T},
        Mval::AbstractVector{T},
        path::AbstractVector{I},
        k::I,
        Cval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        α::T,
        σ::T,
        uplo::Val{UPLO},
    ) where {T, I <: Integer, UPLO}

    j = path[k]
    nn = eltypedegree(res, j)
    na = eltypedegree(rel, j)

    c₁ = view(Cval, neighbors(res, j))
    k₂ = view(Kval, incident(rel, j))

    Dp = Dptr[j]
    Lp = Lptr[j]
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + na * nn - one(I)), na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end

    p = zero(I)
    f  = view(Fval, p + one(I):p + nn + na)
    f₁ = view(Fval, p + one(I):p + nn); p += nn
    f₂ = view(Fval, p + one(I):p + na); p += na

    copyrec!(f₁, c₁)
    zerorec!(f₂)

    if !isone(k)
        i = path[k - one(I)]
        m = view(Mval, oneto(eltypedegree(rel, i)))
        addrec!(f, m, neighbors(rel, i))
    end

    m₂ = view(Mval, oneto(na))

    copyrec!(c₁, f₁)
    copyrec!(m₂, f₂)

    return chol_davis_hager!(D₁₁, L₂₁, c₁, k₂, m₂, α, σ, uplo)
end

#
# Given a unit-lower-triangular matrix D, a matrix L, and
# vectors d, c, k, and m, compute E, F, and e such that
#
#     [ D ] diag(d) [ Dᵀ Lᵀ ] + σ [  c  ] [ cᵀ kᵀ+mᵀ ] = [ E ] diag(e) [ Eᵀ Fᵀ ]
#     [ L ]                       [ k+m ]                [ F ]
#
# over-writing D with E, L with F, and d with e. Additionally, the
# vector m is over-written with the difference m ← m - L D⁻¹ c.
function ldlt_davis_hager!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        d::AbstractVector{T},
        c::AbstractVector{T},
        k::AbstractVector{T},
        m::AbstractVector{T},
        α::T,
        σ::T,
        uplo::Val{:L},
    ) where {T}

    @inbounds for j in eachindex(d)
        cj = c[j]
        dj = d[j]

        β = α * dj + σ * cj^2

        if ispositive(β)
            d[j] = β / α

            α = β / dj
            δ = σ * cj / β

            for i in j + 1:length(d)
                ci = c[i] -= cj * D[i, j]
                D[i, j] += δ * ci
            end

            for i in eachindex(k)
                mi = m[i] -= cj * L[i, j]
                L[i, j] += δ * (k[i] + mi)
            end
        else
            error()
        end
    end

    return α
end

# TODO: bad memory access pattern
function ldlt_davis_hager!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        d::AbstractVector{T},
        c::AbstractVector{T},
        k::AbstractVector{T},
        m::AbstractVector{T},
        α::T,
        σ::T,
        uplo::Val{:U},
    ) where {T}

    @inbounds for j in eachindex(d)
        cj = c[j]
        dj = d[j]

        β = α * dj + σ * cj^2

        if ispositive(β)
            d[j] = β / α

            α = β / dj
            δ = σ * cj / β

            for i in j + 1:length(d)
                ci = c[i] -= cj * D[j, i]
                D[j, i] += δ * ci
            end

            for i in eachindex(k)
                mi = m[i] -= cj * L[j, i]
                L[j, i] += δ * (k[i] + mi)
            end
        else
            error()
        end
    end

    return α
end

function chol_davis_hager!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        c::AbstractVector{T},
        k::AbstractVector{T},
        m::AbstractVector{T},
        α::T,
        σ::T,
        uplo::Val{:L},
    ) where {T}

    n = size(D, 1)

    @inbounds for j in 1:n
        Djj = D[j, j]
        p = c[j] / Djj
        β = α + σ * p^2

        if ispositive(β)
            D[j, j] = Djj * sqrt(β / α)
            γ = σ * p / β

            for i in j + 1:n
                ci = c[i] -= D[i, j] * p
                D[i, j] += γ * ci
            end

            for i in eachindex(k)
                mi = m[i] -= L[i, j] * p
                L[i, j] += γ * (k[i] + mi)
            end

            α = β
        else
            error()
        end
    end

    return α
end

# TODO: bad memory access pattern
function chol_davis_hager!(
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        c::AbstractVector{T},
        k::AbstractVector{T},
        m::AbstractVector{T},
        α::T,
        σ::T,
        uplo::Val{:U},
    ) where {T}

    n = size(D, 1)

    @inbounds for j in 1:n
        Djj = D[j, j]
        p = c[j] / Djj
        β = α + σ * p^2

        if ispositive(β)
            D[j, j] = Djj * sqrt(β / α)
            γ = σ * p / β

            for i in j + 1:n
                ci = c[i] -= D[j, i] * p
                D[j, i] += γ * ci
            end

            for i in eachindex(k)
                mi = m[i] -= L[j, i] * p
                L[j, i] += γ * (k[i] + mi)
            end

            α = β
        else
            error()
        end
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
