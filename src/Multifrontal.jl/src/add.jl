# ================================= axpy! =================================

function LinearAlgebra.axpy!(α::Number, X::ChordalTriangular{DIAG, UPLO}, Y::ChordalTriangular{DIAG, UPLO}) where {DIAG, UPLO}
    @assert checksymbolic(X, Y)
    axpy!(α, X.Dval, Y.Dval)
    axpy!(α, X.Lval, Y.Lval)
    return Y
end

function LinearAlgebra.axpy!(α::Number, X::AdjTri{DIAG, UPLO}, Y::AdjTri{DIAG, UPLO}) where {DIAG, UPLO}
    axpy!(α, parent(X), parent(Y))
    return Y
end

function LinearAlgebra.axpy!(α::Number, X::TransTri{DIAG, UPLO}, Y::TransTri{DIAG, UPLO}) where {DIAG, UPLO}
    axpy!(α, parent(X), parent(Y))
    return Y
end

function LinearAlgebra.axpy!(α::Number, X::HermTri{UPLO, T}, Y::HermTri{UPLO, T}) where {UPLO, T}
    @assert checksymtri(X) && checksymtri(Y)
    axpy!(α, parent(X), parent(Y))
    return Y
end

function LinearAlgebra.axpy!(α::Number, X::SymTri{UPLO, T}, Y::SymTri{UPLO, T}) where {UPLO, T}
    @assert checksymtri(X) && checksymtri(Y)
    axpy!(α, parent(X), parent(Y))
    return Y
end

function LinearAlgebra.axpy!(α, J::UniformScaling, A::ChordalTriangular{:N})
    @inbounds for f in fronts(A)
        D, _ = diagblock(A, f)

        for i in diagind(D)
            D[i] += α * J.λ
        end
    end

    return A
end

function LinearAlgebra.axpy!(α, J::Diagonal, A::ChordalTriangular{:N})
    @inbounds for f in fronts(A)
        D, res = diagblock(A, f)

        for (i, j) in enumerate(diagind(D))
            D[j] += α * J.diag[res[i]]
        end
    end

    return A
end

# =================================== + ===================================

function Base.:+(A::ChordalTriangular{:N, UPLO, T}, B::ChordalTriangular{:N, UPLO, T}) where {UPLO, T}
    return axpy!(one(T), A, copy(B))
end

function Base.:+(A::HermTri{UPLO}, B::HermTri{UPLO}) where {UPLO}
    return Hermitian(parent(A) + parent(B), UPLO)
end

function Base.:+(A::SymTri{UPLO}, B::SymTri{UPLO}) where {UPLO}
    return Symmetric(parent(A) + parent(B), UPLO)
end

function Base.:+(A::AdjTri{:N, UPLO}, B::AdjTri{:N, UPLO}) where UPLO
    return adjoint(parent(A) + parent(B))
end

function Base.:+(A::TransTri{:N, UPLO}, B::TransTri{:N, UPLO}) where UPLO
    return transpose(parent(A) + parent(B))
end

function Base.:+(A::ChordalTriangular{:N}, J::UniformScaling)
    B = similar(A, promote_eltype(A, J))
    copyto!(B, A)
    axpy!(true, J, B)
    return B
end

function Base.:+(J::UniformScaling, A::ChordalTriangular{:N})
    return A + J
end

function Base.:+(A::HermTri{UPLO}, J::UniformScaling) where {UPLO}
    return Hermitian(parent(A) + J, UPLO)
end

function Base.:+(A::HermTri{UPLO}, J::UniformScaling{<:Complex}) where {UPLO}
    @assert iszero(imag(J.λ))
    return Hermitian(parent(A) + real(J.λ) * I, UPLO)
end

function Base.:+(J::UniformScaling, A::HermTri)
    return A + J
end

function Base.:+(J::UniformScaling{<:Complex}, A::HermTri)
    return A + J
end

function Base.:+(A::SymTri{UPLO}, J::UniformScaling) where {UPLO}
    return Symmetric(parent(A) + J, UPLO)
end

function Base.:+(J::UniformScaling, A::SymTri)
    return A + J
end

function Base.:+(A::ChordalTriangular{:N}, J::Diagonal)
    B = similar(A, promote_eltype(A, J))
    copyto!(B, A)
    axpy!(true, J, B)
    return B
end

function Base.:+(J::Diagonal, A::ChordalTriangular{:N})
    return A + J
end

function Base.:+(A::HermTri{UPLO}, D::Diagonal) where {UPLO}
    return Hermitian(parent(A) + D, UPLO)
end

function Base.:+(A::HermTri{UPLO}, D::Diagonal{<:Real}) where {UPLO}
    return Hermitian(parent(A) + D, UPLO)
end

function Base.:+(D::Diagonal, A::HermTri)
    return A + D
end

function Base.:+(D::Diagonal{<:Real}, A::HermTri)
    return A + D
end

function Base.:+(A::SymTri{UPLO}, D::Diagonal) where {UPLO}
    return Symmetric(parent(A) + D, UPLO)
end

function Base.:+(A::SymTri{UPLO}, D::Diagonal{<:Number}) where {UPLO}
    return Symmetric(parent(A) + D, UPLO)
end

function Base.:+(D::Diagonal, A::SymTri)
    return A + D
end

function Base.:+(D::Diagonal{<:Number}, A::SymTri)
    return A + D
end

# =================================== - ===================================

function Base.:-(X::ChordalTriangular{:N})
    return -1 * X
end

function Base.:-(X::HermOrSymTri)
    return -1 * X
end

function Base.:-(X::AdjTri{:N})
    return -1 * X
end

function Base.:-(X::TransTri{:N})
    return -1 * X
end

function Base.:-(A::ChordalTriangular{:N, UPLO, T}, B::ChordalTriangular{:N, UPLO, T}) where {UPLO, T}
    return axpy!(-one(T), B, copy(A))
end

function Base.:-(A::HermTri{UPLO}, B::HermTri{UPLO}) where {UPLO}
    return Hermitian(parent(A) - parent(B), UPLO)
end

function Base.:-(A::SymTri{UPLO}, B::SymTri{UPLO}) where {UPLO}
    return Symmetric(parent(A) - parent(B), UPLO)
end

function Base.:-(A::AdjTri{:N, UPLO}, B::AdjTri{:N, UPLO}) where UPLO
    return adjoint(parent(A) - parent(B))
end

function Base.:-(A::TransTri{:N, UPLO}, B::TransTri{:N, UPLO}) where UPLO
    return transpose(parent(A) - parent(B))
end

function Base.:-(A::ChordalTriangular{:N}, J::UniformScaling)
    return A + (-J)
end

function Base.:-(A::HermTri, J::UniformScaling)
    return A + (-J)
end

function Base.:-(A::SymTri, J::UniformScaling)
    return A + (-J)
end

function Base.:-(A::ChordalTriangular{:N}, D::Diagonal)
    return A + (-D)
end

function Base.:-(A::HermTri, D::Diagonal)
    return A + (-D)
end

function Base.:-(A::HermTri, D::Diagonal{<:Real})
    return A + (-D)
end

function Base.:-(A::SymTri, D::Diagonal)
    return A + (-D)
end

function Base.:-(A::SymTri, D::Diagonal{<:Number})
    return A + (-D)
end

function Base.:-(J::UniformScaling, A::ChordalTriangular{:N})
    return J + (-A)
end

function Base.:-(J::UniformScaling, A::HermTri)
    return J + (-A)
end

function Base.:-(J::UniformScaling{<:Complex}, A::HermTri)
    return J + (-A)
end

function Base.:-(J::UniformScaling, A::SymTri)
    return J + (-A)
end

function Base.:-(D::Diagonal, A::ChordalTriangular{:N})
    return D + (-A)
end

function Base.:-(D::Diagonal, A::HermTri)
    return D + (-A)
end

function Base.:-(D::Diagonal{<:Real}, A::HermTri)
    return D + (-A)
end

function Base.:-(D::Diagonal, A::SymTri)
    return D + (-A)
end

function Base.:-(D::Diagonal{<:Number}, A::SymTri)
    return D + (-A)
end

# ================================== zero ==================================

function Base.zero(A::ChordalTriangular{:N})
    return fill!(similar(A), 0)
end

function Base.zero(A::HermTri{UPLO}) where {UPLO}
    return Hermitian(zero(parent(A)), UPLO)
end

function Base.zero(A::SymTri{UPLO}) where {UPLO}
    return Symmetric(zero(parent(A)), UPLO)
end

function Base.zero(A::AdjTri{:N})
    return adjoint(zero(parent(A)))
end

function Base.zero(A::TransTri{:N})
    return transpose(zero(parent(A)))
end
