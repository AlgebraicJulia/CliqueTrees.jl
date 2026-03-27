# Mooncake rule for P \ X where P is a Permutation

@is_primitive MinimalCtx Tuple{typeof(\), Permutation, AbstractVecOrMat}

function Mooncake.rrule!!(
    ::CoDual{typeof(\)},
    P::CoDual{<:Permutation},
    X::CoDual{<:AbstractVecOrMat}
)
    Pval = primal(P)
    Xval = primal(X)
    dX = tangent(X)

    Y = Pval \ Xval
    dY = zero(Y)

    function ldivprm_pb!!(::NoRData)
        # ΔX = P * ΔY (apply permutation)
        axpy!(1, Pval * dY, dX)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(Y, dY), ldivprm_pb!!
end
