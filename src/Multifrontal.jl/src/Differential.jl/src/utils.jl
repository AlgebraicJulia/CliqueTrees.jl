function scldia!(A::ChordalTriangular, α)
    @inbounds for j in fronts(A)
        D, _ = diagblock(A, j)

        for i in diagind(D)
            D[i] *= α
        end
    end

    return A
end

function scldia!(A::SparseMatrixCSC, α)
    for i in diagind(A)
        A[i] *= α
    end

    return A
end

function scldia!(A::HermOrSym, α)
    scldia!(parent(A), α)
    return A
end

function scldia!(F::ChordalCholesky, α)
    scldia!(triangular(F), α)
    return F
end

function selaxpby!(A::SparseMatrixCSC{TA, IA}, B::SparseMatrixCSC{TB, IB}, α, β, uplo::Char) where {TA, IA, TB, IB}
    @inbounds for j in axes(A, 2)
        pa, pastop = A.colptr[j], A.colptr[j + 1] - one(IA)
        pb, pbstop = B.colptr[j], B.colptr[j + 1] - one(IB)

        while pa <= pastop && pb <= pbstop
            ia, ib = rowvals(A)[pa], rowvals(B)[pb]

            if ia == ib
                if (uplo == 'L' && ia >= j) || (uplo == 'U' && ia <= j)
                    if iszero(β)
                        nonzeros(A)[pa] = α * nonzeros(B)[pb]
                    else
                        nonzeros(A)[pa] = α * nonzeros(B)[pb] + β * nonzeros(A)[pa]
                    end
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

function selaxpby!(α, B::HermOrSymSparse, β, A::HermOrSymSparse)
    selaxpby!(parent(A), parent(B), α, β, A.uplo)
    return A
end

function selaxpy!(α, B::HermOrSymSparse, A::HermOrSymSparse)
    return selaxpby!(α, B, true, A)
end

function selaxpy!(α, B::HermOrSymTri, P::Permutation, A::HermOrSymSparse)
    return selaxpy!(α, project(A, B, P), A)
end

function symdot(A::HermSparse, B::HermSparse)
    @assert A.uplo == B.uplo
    return symdot_impl(parent(A), parent(B), Val(:C), A.uplo)
end

function symdot(A::SymSparse, B::SymSparse)
    @assert A.uplo == B.uplo
    return symdot_impl(parent(A), parent(B), Val(:N), A.uplo)
end

function symdot_impl(A::SparseMatrixCSC{TA, IA}, B::SparseMatrixCSC{TB, IB}, tA::Val{T}, uplo::Char) where {TA, IA, TB, IB, T}
    if T === :C
        out = real(zero(TA) * zero(TB))
    else
        out = zero(TA) * zero(TB)
    end

    @inbounds for j in axes(A, 2)
        pa, pastop = A.colptr[j], A.colptr[j + 1] - one(IA)
        pb, pbstop = B.colptr[j], B.colptr[j + 1] - one(IB)

        while pa <= pastop && pb <= pbstop
            ia, ib = rowvals(A)[pa], rowvals(B)[pb]
            Aij = nonzeros(A)[pa]
            Bij = nonzeros(B)[pb]

            if ia == ib
                if (uplo == 'L' && ia >= j) || (uplo == 'U' && ia <= j)
                    if ia == j
                        if T === :C
                            out += real(Aij) * real(Bij)
                        else
                            out += Aij * Bij
                        end
                    else
                        if T === :C
                            out += 2 * real(conj(Aij) * Bij)
                        else
                            out += 2 * Aij * Bij
                        end
                    end
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

    return out
end
