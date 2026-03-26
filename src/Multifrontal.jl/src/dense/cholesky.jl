# ===== factorize! Level 2: Dense triangular with kwargs =====

function factorize!(
        L::AbstractTriangular{T},
        d::AbstractVector,
        pivot::NoPivot;
        signs::AbstractVector,
        reg::AbstractRegularization,
        check::Bool,
        tol::Real,
    ) where {T}
    @assert checksigns(signs, reg)

    info = factorize!(L, d, pivot, signs, reg, tol)

    if ispositive(info) && check
        if L isa LowerTriangular || L isa UpperTriangular
            throw(PosDefException(info))
        else
            throw(ZeroPivotException(info))
        end
    end

    return info
end

function factorize!(
        L::AbstractTriangular{T},
        d::AbstractVector,
        pivot::RowMaximum,
        perm::AbstractVector,
        invp::AbstractVector;
        signs::AbstractVector,
        reg::AbstractRegularization,
        check::Bool,
        tol::Real,
    ) where {T}
    @assert checksigns(signs, reg)

    info = factorize!(L, d, pivot, perm, invp, signs, reg, tol)

    if ispositive(info) && check
        if L isa LowerTriangular || L isa UpperTriangular
            throw(PosDefException(info))
        else
            throw(ZeroPivotException(info))
        end
    end

    return info
end

# ===== factorize! Level 3: Dense triangular all positional =====

function factorize!(
        L::AbstractTriangular{T},
        d::AbstractVector,
        pivot::NoPivot,
        signs::AbstractVector,
        reg::AbstractRegularization,
        tol::Real,
    ) where {T}
    S = permuteto(T, signs, eachindex(signs))
    R = initialize(L, S, reg)
    M = parent(L)

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

    if diag === Val(:N)
        W = Ones{T}(length(L))
    else
        W = FVector{T}(undef, length(L))
    end

    if R isa SE99
        if diag === Val(:N)
            e = LinearAlgebra.diag(M)
        else
            e = d
            e .= view(M, diagind(M))
        end
    else
        e = d
    end

    return potrf!(uplo, W, M, e, S, R, diag)
end

function factorize!(
        L::AbstractTriangular{T},
        d::AbstractVector,
        pivot::RowMaximum,
        perm::AbstractVector,
        invp::AbstractVector,
        signs::AbstractVector,
        reg::AbstractRegularization,
        tol::Real,
    ) where {T}
    S = permuteto(T, signs, eachindex(signs))
    R = initialize(L, S, reg)
    M = parent(L)

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

    W = FVector{T}(undef, length(L))

    if R isa SE99 || R isa GMW81
        if diag === Val(:N)
            e = LinearAlgebra.diag(M)
        else
            e = d .= view(M, diagind(M))
        end
    else
        e = d
    end

    info, rank = pstrf!(uplo, W, M, e, invp, S, R, tol, diag)

    for i in eachindex(perm)
        perm[i] = invp[perm[i]]
    end

    for i in eachindex(perm)
        invp[perm[i]] = i
    end

    return info
end
