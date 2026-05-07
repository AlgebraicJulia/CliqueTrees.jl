# ================================= selaxpby! =================================

function selaxpy!(α::Number, A::AbstractMatrix, B::AbstractMatrix)
    return selaxpby!(α, A, true, B)
end

function selaxpby!(α::Number, A::HermOrSym, β::Number, B::HermOrSym)
    @assert A.uplo == B.uplo
    selaxpby!(α, parent(A), β, parent(B), A.uplo)
    return A
end

function selaxpby!(β::Number, B::SparseMatrixCSC{TB, IB}, α::Number, A::SparseMatrixCSC{TA, IA}, uplo::Char) where {TA, IA, TB, IB}
    if iszero(α)
        fill!(nonzeros(A), α)
    elseif !isone(α)
        rmul!(nonzeros(A), α)
    end

    @inbounds for j in axes(A, 2)
        pa, pastop = A.colptr[j], A.colptr[j + 1] - one(IA)
        pb, pbstop = B.colptr[j], B.colptr[j + 1] - one(IB)

        while pa <= pastop && pb <= pbstop
            ia = rowvals(A)[pa]
            ib = rowvals(B)[pb]

            if ia == ib
                if (uplo == 'L' && ia >= j) || (uplo == 'U' && ia <= j)
                    nonzeros(A)[pa] += β * nonzeros(B)[pb]
                end

                pa += one(IA)
                pb += one(IB)
            elseif ia < ib
                pa += one(IA)
            else
                pb += one(IB)
            end
        end
    end

    return A
end

function selaxpby!(β::Number, B::ChordalTriangular, α::Number, A::SparseMatrixCSC, uplo::Char)
    @assert char(B.uplo) == uplo
    return selaxpby!(β, B, α, A)
end

function selaxpby!(β::Number, B::ChordalTriangular{DIAG, :L, T, I}, α::Number, A::SparseMatrixCSC) where {DIAG, T, I}
    if iszero(α)
        fill!(nonzeros(A), α)
    elseif !isone(α)
        rmul!(nonzeros(A), α)
    end

    @inbounds for f in fronts(B)
        D, res = diagblock(B, f)
        L, sep = offdblock(B, f)

        rlo = first(res)
        rhi = last(res)

        if !isempty(sep)
            slo = first(sep)
            shi = last(sep)
        end

        for j in eachindex(res)
            i = one(I)

            for p in nzrange(A, res[j])
                row = rowvals(A)[p]

                if row <= rhi
                    row < rlo && continue
                    nonzeros(A)[p] += β * parent(D)[row - rlo + one(I), j]
                elseif !isempty(sep)
                    row < slo && continue
                    row > shi && break

                    while sep[i] < row
                        i += one(I)
                    end

                    nonzeros(A)[p] += β * L[i, j]
                    i += one(I)
                end
            end
        end
    end

    return A
end

function selaxpby!(β::Number, B::ChordalTriangular{DIAG, :U, T, I}, α::Number, A::SparseMatrixCSC) where {DIAG, T, I}
    if iszero(α)
        fill!(nonzeros(A), α)
    elseif !isone(α)
        rmul!(nonzeros(A), α)
    end

    @inbounds for i in fronts(B)
        D, res = diagblock(B, i)
        L, sep = offdblock(B, i)

        rlo = first(res)
        rhi = last(res)

        for j in eachindex(res)
            for p in nzrange(A, res[j])
                row = rowvals(A)[p]
                row < rlo && continue
                row > rhi && break
                nonzeros(A)[p] += β * parent(D)[row - rlo + one(I), j]
            end
        end

        for j in eachindex(sep)
            for p in nzrange(A, sep[j])
                row = rowvals(A)[p]
                row < rlo && continue
                row > rhi && break
                nonzeros(A)[p] += β * L[row - rlo + one(I), j]
            end
        end
    end

    return A
end

# ================================= axpby! =================================

function LinearAlgebra.axpby!(α::Number, X::ChordalTriangular{DIAG, UPLO}, β::Number, Y::ChordalTriangular{DIAG, UPLO}) where {DIAG, UPLO}
    @assert checksymbolic(X, Y)
    axpby!(α, X.Dval, β, Y.Dval)
    axpby!(α, X.Lval, β, Y.Lval)
    return Y
end

function LinearAlgebra.axpby!(α::Number, X::AdjTri{DIAG, UPLO}, β::Number, Y::AdjTri{DIAG, UPLO}) where {DIAG, UPLO}
    axpby!(α, parent(X), β, parent(Y))
    return Y
end

function LinearAlgebra.axpby!(α::Number, X::TransTri{DIAG, UPLO}, β::Number, Y::TransTri{DIAG, UPLO}) where {DIAG, UPLO}
    axpby!(α, parent(X), β, parent(Y))
    return Y
end

function LinearAlgebra.axpby!(α::Number, X::HermTri{UPLO, T}, β::Number, Y::HermTri{UPLO, T}) where {UPLO, T}
    @assert checksymtri(X) && checksymtri(Y)
    axpby!(α, parent(X), β, parent(Y))
    return Y
end

function LinearAlgebra.axpby!(α::Number, X::SymTri{UPLO, T}, β::Number, Y::SymTri{UPLO, T}) where {UPLO, T}
    @assert checksymtri(X) && checksymtri(Y)
    axpby!(α, parent(X), β, parent(Y))
    return Y
end

function LinearAlgebra.axpby!(α::Number, J::UniformScaling, β::Number, A::ChordalTriangular{:N})
    @inbounds for f in fronts(A)
        D, _ = diagblock(A, f)

        for i in diagind(D)
            if iszero(β)
                D[i] = α * J.λ
            else
                D[i] = α * J.λ + β * D[i]
            end
        end
    end

    return A
end

function LinearAlgebra.axpby!(α::Number, J::Diagonal, β::Number, A::ChordalTriangular{:N})
    @inbounds for f in fronts(A)
        D, res = diagblock(A, f)

        for (i, j) in zip(diagind(D), res)
            Jj = J.diag[j]

            if iszero(β)
                D[i] = α * Jj
            else
                D[i] = α * Jj + β * D[i]
            end
        end
    end

    return A
end

function LinearAlgebra.axpby!(α::Number, J::UniformScaling, β::Number, A::HermOrSymTri)
    axpby!(α, J, β, parent(A))
    return A
end

function LinearAlgebra.axpby!(α::Number, J::UniformScaling, β::Number, A::AdjOrTransTri{:N})
    axpby!(α, J, β, parent(A))
    return A
end

function LinearAlgebra.axpby!(α::Number, J::Diagonal, β::Number, A::HermOrSymTri)
    axpby!(α, J, β, parent(A))
    return A
end

function LinearAlgebra.axpby!(α::Number, J::Diagonal, β::Number, A::AdjOrTransTri{:N})
    axpby!(α, J, β, parent(A))
    return A
end

function LinearAlgebra.axpby!(α::Number, X::SparseMatrixCSC, β::Number, Y::ChordalTriangular{:N, :L, T, I}) where {T, I}
    if iszero(β)
        fill!(Y, β)
    elseif !isone(β)
        rmul!(Y, β)
    end

    @inbounds for f in fronts(Y)
        D, res = diagblock(Y, f)
        L, sep = offdblock(Y, f)

        rlo = first(res)
        rhi = last(res)

        if !isempty(sep)
            slo = first(sep)
            shi = last(sep)
        end

        for jloc in eachindex(res)
            j = res[jloc]
            k = one(I)

            for p in nzrange(X, j)
                i = rowvals(X)[p]

                if i <= rhi
                    i < rlo && continue
                    D[i - rlo + one(I), jloc] += α * nonzeros(X)[p]
                elseif !isempty(sep)
                    i < slo && continue
                    i > shi && break

                    while sep[k] < i
                        k += one(I)
                    end

                    L[k, jloc] += α * nonzeros(X)[p]
                    k += one(I)
                end
            end
        end
    end

    return Y
end

function LinearAlgebra.axpby!(α::Number, X::SparseMatrixCSC, β::Number, Y::ChordalTriangular{:N, :U, T, I}) where {T, I}
    if !isone(β)
        if iszero(β)
            fill!(Y, zero(T))
        else
            rmul!(Y, β)
        end
    end

    @inbounds for f in fronts(Y)
        D, res = diagblock(Y, f)
        L, sep = offdblock(Y, f)

        rlo = first(res)
        rhi = last(res)

        for jloc in eachindex(res)
            j = res[jloc]

            for p in nzrange(X, j)
                i = rowvals(X)[p]
                i < rlo && continue
                i > rhi && break

                D[i - rlo + one(I), jloc] += α * nonzeros(X)[p]
            end
        end

        for jloc in eachindex(sep)
            j = sep[jloc]

            for p in nzrange(X, j)
                i = rowvals(X)[p]
                i < rlo && continue
                i > rhi && break

                L[i - rlo + one(I), jloc] += α * nonzeros(X)[p]
            end
        end
    end

    return Y
end

function LinearAlgebra.axpby!(α::Number, X::HermOrSymSparse, β::Number, Y::HermOrSymTri)
    axpby!(α, parent(X), β, parent(Y))
    return Y
end


# ================================= axpy! =================================

function LinearAlgebra.axpy!(α::Number, X::ChordalTriangular{DIAG, UPLO}, Y::ChordalTriangular{DIAG, UPLO}) where {DIAG, UPLO}
    return axpby!(α, X, true, Y)
end

function LinearAlgebra.axpy!(α::Number, X::AdjTri{DIAG, UPLO}, Y::AdjTri{DIAG, UPLO}) where {DIAG, UPLO}
    return axpby!(α, X, true, Y)
end

function LinearAlgebra.axpy!(α::Number, X::TransTri{DIAG, UPLO}, Y::TransTri{DIAG, UPLO}) where {DIAG, UPLO}
    return axpby!(α, X, true, Y)
end

function LinearAlgebra.axpy!(α::Number, X::HermTri{UPLO, T}, Y::HermTri{UPLO, T}) where {UPLO, T}
    return axpby!(α, X, true, Y)
end

function LinearAlgebra.axpy!(α::Number, X::SymTri{UPLO, T}, Y::SymTri{UPLO, T}) where {UPLO, T}
    return axpby!(α, X, true, Y)
end

function LinearAlgebra.axpy!(α, J::UniformScaling, A::ChordalTriangular{:N})
    return axpby!(α, J, true, A)
end

function LinearAlgebra.axpy!(α, J::Diagonal, A::ChordalTriangular{:N})
    return axpby!(α, J, true, A)
end

function LinearAlgebra.axpy!(α, J::UniformScaling, A::HermOrSymTri)
    return axpby!(α, J, true, A)
end

function LinearAlgebra.axpy!(α, J::UniformScaling, A::AdjOrTransTri{:N})
    return axpby!(α, J, true, A)
end

function LinearAlgebra.axpy!(α, J::Diagonal, A::HermOrSymTri)
    return axpby!(α, J, true, A)
end

function LinearAlgebra.axpy!(α, J::Diagonal, A::AdjOrTransTri{:N})
    return axpby!(α, J, true, A)
end

function LinearAlgebra.axpy!(α::Number, X::SparseMatrixCSC, Y::ChordalTriangular{:N})
    return axpby!(α, X, true, Y)
end

function LinearAlgebra.axpy!(α::Number, X::HermOrSymSparse, Y::HermOrSymTri)
    return axpby!(α, X, true, Y)
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
