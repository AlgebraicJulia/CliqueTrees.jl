function uncholesky!(F::DenseCholeskyPivoted{UPLO, T}) where {UPLO, T}
    n = size(F, 1)
    W = FVector{T}(undef, n * n)
    unchol_dense!(F.M, F.d, W, F.uplo, F.diag)
    return F
end

function unldlt!(F::DenseLDLtPivoted{UPLO, T}) where {UPLO, T}
    n = size(F, 1)
    W = FVector{T}(undef, n * n)
    unchol_dense!(F.M, F.d, W, F.uplo, F.diag)
    return F
end

function unchol_dense!(
        A::AbstractMatrix{T},
        d::AbstractVector{T},
        Wval::AbstractVector{T},
        uplo::Val{UPLO},
        diag::Val{DIAG},
    ) where {UPLO, DIAG, T}
    W = reshape(view(Wval, 1:length(A)), size(A))
    fill!(W, zero(T))
    copytri!(W, A, uplo)

    if UPLO === :L
        side = Val(:R)
    else
        side = Val(:L)
    end

    if DIAG === :U
        @inbounds for i in diagind(W)
            W[i] = one(T)
        end
    end

    cmul!(side, diag, W, d)
    trmm!(side, uplo, Val(:C), diag, one(T), A, W)

    copytri!(A, W, uplo)
    return
end
