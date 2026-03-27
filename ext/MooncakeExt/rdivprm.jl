# Mooncake rule for X / P where P is a Permutation

@is_primitive MinimalCtx Tuple{typeof(/), AbstractVecOrMat, Permutation}

function Mooncake.rrule!!(
    ::CoDual{typeof(/)},
    X::CoDual{<:AbstractVecOrMat},
    P::CoDual{<:Permutation}
)
    Xval = primal(X)
    Pval = primal(P)
    dX = tangent(X)

    Y = Xval / Pval
    dY = zero(Y)

    function rdivprm_pb!!(::NoRData)
        # ΔX = ΔY * P
        axpy!(1, dY * Pval, dX)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(Y, dY), rdivprm_pb!!
end
