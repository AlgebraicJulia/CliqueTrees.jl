# wraptri: wrap B in the same wrapper type as A
function wraptri(::ChordalTriangular{:N, UPLO}, B::ChordalTriangular{:N, UPLO}) where {UPLO}
    return B
end

function wraptri(::HermTri{UPLO}, B::ChordalTriangular{:N, UPLO}) where {UPLO}
    return Hermitian(B, UPLO)
end

function wraptri(::SymTri{UPLO}, B::ChordalTriangular{:N, UPLO}) where {UPLO}
    return Symmetric(B, UPLO)
end

function wraptri(::AdjTri{:N, UPLO}, B::ChordalTriangular{:N, UPLO}) where {UPLO}
    return adjoint(B)
end

function wraptri(::TransTri{:N, UPLO}, B::ChordalTriangular{:N, UPLO}) where {UPLO}
    return transpose(B)
end

function copylike(A::ChordalTriangular, B::ChordalTriangular)
    return copy(B)
end

function copylike(A::ChordalTriangular, B::HermOrSymTri)
    return copy(parent(B))
end

function copylike(A::ChordalTriangular, B::Union{UniformScaling, Diagonal})
    return copyto!(similar(A), B)
end

function copylike(A::HermOrSymTri, B::ChordalTriangular)
    return wraptri(A, copy(B))
end

function copylike(A::HermOrSymTri, B::HermOrSymTri)
    return wraptri(A, copy(parent(B)))
end

function copylike(A::HermOrSymTri, B::Union{UniformScaling, Diagonal})
    return copyto!(similar(A), B)
end

# fwrap: unwrap, apply f, rewrap
function fwrap(f, A)
    return f(A)
end

function fwrap(f, A::Adjoint)
    return adjoint(fwrap(f, parent(A)))
end

function fwrap(f, A::Transpose)
    return transpose(fwrap(f, parent(A)))
end

