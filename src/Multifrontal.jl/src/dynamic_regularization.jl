abstract type AbstractRegularization end

const MaybeRegularization = Union{Nothing, AbstractRegularization}

@kwdef struct DynamicRegularization{T} <: AbstractRegularization
    delta::T
    epsilon::T
end

function DynamicRegularization(F::ChordalLDLt; delta::Real=dynm_delta(F), epsilon::Real=dynm_epsilon(F))
    return DynamicRegularization(; delta, epsilon)
end

function dynm_delta(F::ChordalLDLt{UPLO, T}) where {UPLO, T}
    return cbrt(eps(real(T)))
end

function dynm_epsilon(F::ChordalLDLt{UPLO, T}) where {UPLO, T}
    return sqrt(eps(real(T)))
end

@kwdef struct GMW81{T} <: AbstractRegularization
    beta::T
    delta::T
end

function GMW81(F::ChordalLDLt; beta::Real=gmw81_beta(F), delta::Real=gmw81_delta(F))
    return GMW81(; beta, delta)
end

function gmw81_beta(F::ChordalLDLt{UPLO, T, I}) where {UPLO, T, I}
    A = ChordalTriangular(F)
    n = size(A, 1)
    gamma = zero(real(T))
    xi = zero(real(T))

    @inbounds for j in vertices(A.S.res)
        nn = eltypedegree(A.S.res, j)
        na = eltypedegree(A.S.rel, j)

        Dp = A.S.Dptr[j]
        Lp = A.S.Lptr[j]

        D = reshape(view(A.Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

        if UPLO === :L
            L = reshape(view(A.Lval, Lp:Lp + nn * na - one(I)), na, nn)
        else
            L = reshape(view(A.Lval, Lp:Lp + nn * na - one(I)), nn, na)
        end

        for i in oneto(nn)
            gamma = max(gamma, abs(D[i, i]))
        end

        if UPLO === :L
            for col in oneto(nn)
                for row in col + one(I):nn
                    xi = max(xi, abs(D[row, col]))
                end
            end
        else
            for col in oneto(nn)
                for row in oneto(col - one(I))
                    xi = max(xi, abs(D[row, col]))
                end
            end
        end

        for v in L
            xi = max(xi, abs(v))
        end
    end

    return max(gamma, xi / sqrt(n^2 - 1), eps(real(T)))
end

function gmw81_delta(F::ChordalLDLt{UPLO, T}) where {UPLO, T}
    return cbrt(eps(real(T)))
end

# ===== regularize functions =====

function regularize(::Nothing, S::AbstractVector, Djj, j::Integer)
    if S[j] * Djj <= zero(Djj)
        return zero(Djj)
    else
        return Djj
    end
end

function regularize(R::DynamicRegularization, S::AbstractVector, Djj, j::Integer)
    if S[j] * Djj < R.epsilon
        return R.delta * S[j]
    else
        return Djj
    end
end

function regularize(
        R::GMW81,
        S::AbstractVector,
        D::AbstractMatrix{T},
        L::AbstractMatrix{T},
        Djj::T,
        j::Integer,
        uplo::Val{UPLO},
    ) where {T, UPLO}
    Aimax = zero(real(T))

    if UPLO === :L
        for i in j + 1:size(D, 1)
            Aimax = max(Aimax, abs(D[i, j]))
        end

        for i in axes(L, 1)
            Aimax = max(Aimax, abs(L[i, j]))
        end
    else
        for i in j + 1:size(D, 2)
            Aimax = max(Aimax, abs(D[j, i]))
        end

        for i in axes(L, 2)
            Aimax = max(Aimax, abs(L[j, i]))
        end
    end

    return S[j] * max(abs(Djj), Aimax^2 / R.beta, R.delta)
end
