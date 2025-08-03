# copy the lower triangular part of B to A
function lacpy!(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    m = size(B, 1)

    @inbounds for j in axes(B, 2)
        for i in j:m
            A[i, j] = B[i, j]
        end
    end

    return
end

@static if VERSION >= v"1.11"

function lacpy!(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}}
    LAPACK.lacpy!(A, B, 'L')
    return
end

end

# factorize A as
#
#     A = L Lᴴ
#
# and store A ← L
function potrf!(A::AbstractMatrix)
    F = LinearAlgebra.cholesky!(Hermitian(A, :L), NoPivot())
    status = iszero(F.info)
    return status
end

# compute the lower triangular part of the difference
#
#     C = L - A * Aᴴ
#
# and store L ← C
function syrk!(A::AbstractMatrix{T}, L::AbstractMatrix{T}) where {T}
    m = size(A, 1)

    @inbounds for j in axes(A, 1), k in axes(A, 2)
        Ajk = A[j, k]

        for i in j:m
            L[i, j] -= A[i, k] * Ajk
        end
    end

    return
end

function syrk!(A::AbstractMatrix{T}, L::AbstractMatrix{T}) where {T <: Complex}
    m = size(A, 1)

    @inbounds for j in axes(A, 1), k in axes(A, 2)
        Ajk = conj(A[j, k])

        for i in j:m
            L[i, j] -= A[i, k] * Ajk
        end
    end

    return
end

function syrk!(A::AbstractMatrix{T}, L::AbstractMatrix{T}) where {T <: Union{Float32, Float64}}
    BLAS.syrk!('L', 'N', -one(T), A, one(T), L)
    return
end

function syrk!(A::AbstractMatrix{T}, L::AbstractMatrix{T}) where {T <: Union{ComplexF32, ComplexF64}}
    BLAS.herk!('L', 'N', -one(real(T)), A, one(real(T)), L)
    return
end

# solve for x in
#
#     L x = b
#
# and store b ← x
function trsv!(L::AbstractMatrix{T}, b::AbstractVector{T}, tL::Val{false}) where {T}
    ldiv!(LowerTriangular(L), b)
    return
end

# solve for x in
#
#     Lᴴ x = b
#
# and store b ← x
function trsv!(L::AbstractMatrix{T}, b::AbstractVector{T}, tL::Val{true}) where {T}
    ldiv!(LowerTriangular(L) |> adjoint, b)
    return
end

# solve for X in
#
#     L X = B
#
# and store B ← X
function trsm!(L::AbstractMatrix{T}, B::AbstractMatrix{T}, tL::Val{false}, side::Val{false}) where {T}
    ldiv!(LowerTriangular(L), B)
    return
end

# solve for X in
#
#     Lᴴ X = B
#
# and store B ← X
function trsm!(L::AbstractMatrix{T}, B::AbstractMatrix{T}, tL::Val{true}, side::Val{false}) where {T}
    ldiv!(LowerTriangular(L) |> adjoint, B)
    return
end

# solve for X in
#
#     X Lᴴ = B
#
# and store B ← X
function trsm!(L::AbstractMatrix{T}, B::AbstractMatrix{T}, tL::Val{true}, side::Val{true}) where {T}
    rdiv!(B, LowerTriangular(L) |> adjoint)
    return
end

# compute the difference
#
#     z = y - A x
#
# and store y ← z
function gemv!(A::AbstractMatrix{T}, x::AbstractVector{T}, y::AbstractVector{T}, tA::Val{false}) where {T}
    mul!(y, A, x, -one(T), one(T))
    return
end

# compute the difference
#
#     z = y - Aᴴ x
#
# and store y ← z
function gemv!(A::AbstractMatrix{T}, x::AbstractVector{T}, y::AbstractVector{T}, tA::Val{true}) where {T}
    mul!(y, A |> adjoint, x, -one(T), one(T))
    return
end

# compute the difference
#
#     Z = Y - A X
#
# and store Y ← Z
function gemm!(A::AbstractMatrix{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}, tA::Val{false}) where {T}
    mul!(Y, A, X, -one(T), one(T))
    return
end

# compute the difference
#
#     Z = Y - Aᴴ X
#
# and store Y ← Z
function gemm!(A::AbstractMatrix{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}, tA::Val{true}) where {T}
    mul!(Y, A |> adjoint, X, -one(T), one(T))
    return
end
