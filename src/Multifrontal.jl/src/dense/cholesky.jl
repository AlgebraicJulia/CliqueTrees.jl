function chol!(
        F::DenseFactorization{DIAG, UPLO, T},
        ::NoPivot,
        signs::AbstractVector,
        reg::AbstractRegularization,
        check::Bool,
        tol::Real,
    ) where {DIAG, UPLO, T}
    S = permuteto(T, signs, eachindex(signs))
    R = initialize(triangular(F), S, reg)

    if DIAG === :N
        W = Ones{T}(length(F))
    else
        W = FVector{T}(undef, length(F))
    end

    if R isa SE99
        if DIAG === :N
            d = diag(F.M)
        else
            d = F.d .= view(F.M, diagind(F.M))
        end
    else
        d = F.d
    end

    F.info[] = potrf!(F.uplo, W, F.M, d, S, R, F.diag)

    if ispositive(F.info[]) && check
        if DIAG === :N
            throw(PosDefException(F.info[]))
        else
            throw(ZeroPivotException(F.info[]))
        end
    end

    return F
end

function chol!(
        F::DenseFactorization{DIAG, UPLO, T},
        ::RowMaximum,
        signs::AbstractVector,
        reg::AbstractRegularization,
        check::Bool,
        tol::Real,
    ) where {DIAG, UPLO, T}
    S = permuteto(T, signs, eachindex(signs))
    R = initialize(triangular(F), S, reg)
    W = FVector{T}(undef, length(F))

    if R isa SE99 || R isa GMW81
        if DIAG === :N
            d = diag(F.M)
        else
            d = F.d .= view(F.M, diagind(F.M))
        end
    else
        d = F.d
    end

    F.info[], rank = pstrf!(F.uplo, W, F.M, d, F.invp, S, R, tol, F.diag)

    for i in eachindex(F.perm)
        F.perm[i] = F.invp[F.perm[i]]
    end

    for i in eachindex(F.perm)
        F.invp[F.perm[i]] = i
    end

    return F
end
