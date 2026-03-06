# ===== trtri! =====

function trtri!(uplo::Val, diag::Val, A::AbstractMatrix{T}) where {T <: BlasFloat}
    LAPACK.trtri!(char(uplo), char(diag), A)
    return
end

function trtri!(uplo::Val, diag::Val, A::AbstractMatrix{T}) where {T}
    copyto!(A, inv(tri(uplo, diag, A)))
    return
end
