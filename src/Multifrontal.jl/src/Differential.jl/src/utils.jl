function softplus(x::T) where {T}
    if x >= zero(T)
        return x + log1p(exp(-x))
    else
        return log1p(exp(x))
    end
end

function sigmoid(x::T) where {T}
    if x >= zero(T)
        return inv(one(T) + exp(-x))
    else
        e = exp(x)
        return e / (one(T) + e)
    end
end

function triview(::Val{DIAG}, ::Val{UPLO}, S::ChordalSymbolic, x::AbstractVector) where {DIAG, UPLO}
    nd = ndz(S)
    nl = nlz(S)
    Dval = view(x,      1:nd)
    Lval = view(x, nd + 1:nd + nl)
    return ChordalTriangular{DIAG, UPLO}(S, Dval, Lval)
end

function fflat(f, L::ChordalTriangular)
    x = vcat(L.Dval, L.Lval)
    X = triview(L.diag, L.uplo, L.S, x)
    f(X)
    return x
end

function fflat(f, H::HermOrSymTri)
    return fflat(f, parent(H))
end

function clean!(L::ChordalTriangular{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    @inbounds for f in fronts(L.S)
        D, _ = diagblock(L, f)

        if UPLO === :L
            tril!(parent(D))
        else
            triu!(parent(D))
        end
    end

    return L
end

function scale!(L::ChordalTriangular{DIAG, UPLO}) where {DIAG, UPLO}
    @inbounds for f in fronts(L.S)
        D, _ = diagblock(L, f)

        if UPLO === :L
            for j in axes(D, 1)
                for i in j + 1:size(D, 1)
                    D[i, j] *= 2
                end
            end
        else
            for j in axes(D, 1)
                for i in 1:j - 1
                    D[i, j] *= 2
                end
            end
        end
    end

    rmul!(L.Lval, 2)
    return L
end

function unscale!(L::ChordalTriangular{DIAG, UPLO}) where {DIAG, UPLO}
    @inbounds for f in fronts(L.S)
        D, _ = diagblock(L, f)

        if UPLO === :L
            for j in axes(D, 1)
                for i in j + 1:size(D, 1)
                    D[i, j] /= 2
                end
            end
        else
            for j in axes(D, 1)
                for i in 1:j - 1
                    D[i, j] /= 2
                end
            end
        end
    end

    rdiv!(L.Lval, 2)
    return L
end
