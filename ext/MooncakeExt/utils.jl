function tangentdata(x::Tangent)
    return x.fields
end

function tangentdata(x::FData)
    return x.data
end

function toarray(X::AbstractArray, dX)
    return dX
end

function toarray(X::SparseMatrixCSC, dX)
    dnzval = tangentdata(dX).nzval
    return SparseMatrixCSC(X.m, X.n, X.colptr, X.rowval, dnzval)
end

function toarray(X::Hermitian, dX)
    dY = toarray(parent(X), tangentdata(dX).data)
    return Hermitian(dY, Symbol(X.uplo))
end

function toarray(X::Symmetric, dX)
    dY = toarray(parent(X), tangentdata(dX).data)
    return Symmetric(dY, Symbol(X.uplo))
end

function totangent(Y::Mat, dY) where {Mat <: SparseMatrixCSC}
    colptr = fill(NoTangent(), length(Y.colptr))
    rowval = fill(NoTangent(), length(Y.rowval))
    return build_tangent(Mat, NoTangent(), NoTangent(), colptr, rowval, dY.nzval)
end

function totangent(Y::Mat, dY) where {Mat <: HermOrSym}
    return build_tangent(Mat, totangent(parent(Y), parent(dY)), NoTangent())
end

function tofdata(Y, dY)
    return fdata(totangent(Y, dY))
end

function primaltangent(x::Union{CoDual, Dual})
    return (primal(x), tangent(x))
end

function primaltangent(x::Union{CoDual{<:AbstractArray}, Dual{<:AbstractArray}})
    X = primal(x)
    dX = toarray(X, tangent(x))
    return (X, dX)
end
