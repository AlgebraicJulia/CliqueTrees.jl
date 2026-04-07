function ldiv_rrule_impl!(Y::AbstractVecOrMat, F::ChordalCholesky, X::AbstractVecOrMat, dX::AbstractVecOrMat)
    ldiv!(F, dX)
    copyto!(X, Y)
    return dX
end

function ldiv_frule_impl!(F::ChordalCholesky, X::AbstractVecOrMat, dX::AbstractVecOrMat)
    ldiv!(F,  X)
    ldiv!(F, dX)
    return X, dX
end

function rdiv_rrule_impl!(Y::AbstractMatrix, F::ChordalCholesky, X::AbstractMatrix, dX::AbstractMatrix)
    rdiv!(dX, F)
    copyto!(X, Y)
    return dX
end

function rdiv_frule_impl!(F::ChordalCholesky, X::AbstractMatrix, dX::AbstractMatrix)
    rdiv!( X, F)
    rdiv!(dX, F)
    return X, dX
end
