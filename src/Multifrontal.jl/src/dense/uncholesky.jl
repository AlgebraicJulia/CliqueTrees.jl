function unfactorize!(L::AbstractTriangular{T}, d::AbstractVector) where {T}
    M = parent(L)
    W = FVector{T}(undef, length(L))

    if L isa LowerTriangular || L isa UnitLowerTriangular
        uplo = Val(:L)
    else
        uplo = Val(:U)
    end

    if L isa LowerTriangular || L isa UpperTriangular
        diag = Val(:N)
    else
        diag = Val(:U)
    end

    unchol_dense!(M, d, W, uplo, diag)
    return 0
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
