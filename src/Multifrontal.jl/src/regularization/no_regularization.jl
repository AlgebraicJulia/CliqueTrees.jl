struct NoRegularization <: AbstractRegularization end

function initialize(::ChordalFactorization, S::AbstractVector, R::NoRegularization)
    return R
end

function checksigns(S::AbstractVector, ::NoRegularization)
    for s in S
        if !iszero(s) && !isone(s) && !isone(-s)
            return false
        end
    end

    return true
end

function checksigns(S::AbstractVector, ::AbstractRegularization)
    for s in S
        if !isone(s) && !isone(-s)
            return false
        end
    end

    return true
end

function regularize(::NoRegularization, S::AbstractVector, Djj, j::Integer)
    if !iszero(S[j]) && !ispositive(real(S[j]) * Djj)
        return zero(Djj)
    else
        return Djj
    end
end
