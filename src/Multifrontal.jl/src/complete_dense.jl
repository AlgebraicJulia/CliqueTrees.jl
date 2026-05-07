function complete_dense!(
        X::AbstractMatrix{T},
        L::ChordalTriangular{:N, UPLO, T, I};
        check::Bool=true,
    ) where {UPLO, T, I <: Integer}
    n = ncl(L)

    mark  = FVector{I}(undef, n)
    Wval = FVector{T}(undef, n * L.S.nFval)

    info = complete_dense_impl!(X, Wval, mark, L)
    check && checkinfo(info, L.diag)

    return X
end

function complete_dense_impl!(
        X::AbstractMatrix{T},
        Wval::AbstractVector{T},
        mark::AbstractVector{I},
        L::ChordalTriangular{:N, UPLO, T, I},
    ) where {UPLO, T, I <: Integer}
    info = complete_dense_impl!(
        X, Wval, mark,
        L.S.Dptr, L.Dval,
        L.S.Lptr, L.Lval,
        L.S.res, L.S.sep, L.S.pnt,
        L.uplo)

    return info
end

function complete_dense_impl!(
        X::AbstractMatrix{T},
        Wval::AbstractVector{T},
        mark::AbstractVector{I},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        pnt::AbstractVector{I},
        uplo::Val{UPLO},
    ) where {UPLO, T, I <: Integer}
    fill!(X, zero(T))
    fill!(mark, zero(I))
    #
    # pre-pass: compute nc (last index of subtree root's residual)
    # stored as -nc in mark[j]
    #
    for j in reverse(vertices(res))
        na = eltypedegree(sep, j)

        if iszero(na)
            mark[j] = one(I) - pointers(res)[j + one(I)]
        else
            mark[j] = mark[pnt[j]]
        end
    end

    for j in reverse(vertices(res))
        info = complete_dense_loop!(X, Wval, mark, Dptr, Dval, Lptr, Lval, res, sep, j, uplo)
        ispositive(info) && return info
    end

    return zero(I)
end

function complete_dense_loop!(
        X::AbstractMatrix{T},
        Wval::AbstractVector{T},
        mark::AbstractVector{I},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        j::I,
        uplo::Val{UPLO},
    ) where {UPLO, T, I <: Integer}
    #
    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    nn = eltypedegree(res, j)
    #
    # na is the size of the separator at node j
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(sep, j)
    #
    # nm is the last index of the residual at node j
    #
    nm = last(neighbors(res, j))
    #
    # nc is the last index of the residual at the root of the subtree
    #
    nc = -mark[j]
    #
    # nb is the number of indices after the front within the subtree
    #
    nb = nc - nm - na
    #
    # mark separator indices with current front
    #
    for i in neighbors(sep, j)
        mark[i] = j
    end
    #
    # D₁₁ is the diagonal block
    #
    #          res(j)
    #     D = [ D₁₁ ] res(j)
    #
    Dp = Dptr[j]
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)
    #
    # L₂₁ is the off-diagonal block
    #
    #          res(j)
    #     L = [ L₂₁ ] sep(j)
    #
    Lp = Lptr[j]

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    # W₂₁ and W₂₂ are views into the workspace for separator rows
    # W₃₁ and W₃₂ are views into the workspace for remaining rows
    #
    p = one(I)
    if UPLO === :L
        W₂₁ = reshape(view(Wval, p:p + na * nn - one(I)), na, nn); p += na * nn
        W₂₂ = reshape(view(Wval, p:p + na * na - one(I)), na, na); p += na * na
        W₃₁ = reshape(view(Wval, p:p + nb * nn - one(I)), nb, nn); p += nb * nn
        W₃₂ = reshape(view(Wval, p:p + nb * na - one(I)), nb, na)
    else
        W₂₁ = reshape(view(Wval, p:p + nn * na - one(I)), nn, na); p += nn * na
        W₂₂ = reshape(view(Wval, p:p + na * na - one(I)), na, na); p += na * na
        W₃₁ = reshape(view(Wval, p:p + nn * nb - one(I)), nn, nb); p += nn * nb
        W₃₂ = reshape(view(Wval, p:p + na * nb - one(I)), na, nb)
    end
    #
    # gather X entries into W₂₂ and W₃₂
    #
    complete_dense_gather!(X, W₂₂, W₃₂, mark, res, sep, j, uplo)

    if ispositive(na)
        #
        # W₂₁ ← L₂₁
        #
        copyrec!(W₂₁, L₂₁)
        #
        # W₂₂ ← chol(W₂₂)
        #
        info = potrf!(uplo, W₂₂)
        ispositive(info) && return info
        #
        # W₂₁ ← W₂₂⁻¹ W₂₁
        #
        if UPLO === :L
            trsm!(Val(:L), Val(:L), Val(:N), Val(:N), one(T), W₂₂, W₂₁)
            trsm!(Val(:L), Val(:L), Val(:T), Val(:N), one(T), W₂₂, W₂₁)
        else
            trsm!(Val(:R), Val(:U), Val(:N), Val(:N), one(T), W₂₂, W₂₁)
            trsm!(Val(:R), Val(:U), Val(:T), Val(:N), one(T), W₂₂, W₂₁)
        end
        #
        # W₃₁ ← W₃₂ W₂₁  (for :U, transposed: W₃₁ ← W₂₁ W₃₂)
        #
        if UPLO === :L
            gemm!(Val(:N), Val(:N), one(T), W₃₂, W₂₁, zero(T), W₃₁)
        else
            gemm!(Val(:N), Val(:N), one(T), W₂₁, W₃₂, zero(T), W₃₁)
        end
    end
    #
    # write diagonal block to X
    #
    copyscattertri!(X, D₁₁, neighbors(res, j), uplo)
    #
    # write off-diagonal results back to X
    #
    complete_dense_scatter!(X, L₂₁, W₃₁, mark, res, j, uplo)

    return zero(I)
end

function complete_dense_gather!(
        X::AbstractMatrix{T},
        L₂₂::AbstractMatrix{T},
        L₃₂::AbstractMatrix{T},
        mark::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        f::I,
        ::Val{:L},
    ) where {T, I <: Integer}
    nm = last(neighbors(res, f))
    nc = -mark[f]

    j₁ = 0

    for j in neighbors(sep, f)
        j₁ += 1

        i₂ = 0
        i₃ = 0

        for i in nm + one(I):nc
            if mark[i] == f
                i₂ += 1

                if i₂ ≥ j₁
                    L₂₂[i₂, j₁] = X[i, j]
                end
            else
                i₃ += 1

                if i > j
                    L₃₂[i₃, j₁] = X[i, j]
                else
                    L₃₂[i₃, j₁] = X[j, i]
                end
            end
        end
    end

    return
end

function complete_dense_gather!(
        X::AbstractMatrix{T},
        L₂₂::AbstractMatrix{T},
        L₃₂::AbstractMatrix{T},
        mark::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        f::I,
        ::Val{:U},
    ) where {T, I <: Integer}
    nm = last(neighbors(res, f))
    nc = -mark[f]

    i₂ = 0
    i₃ = 0

    for i in nm + one(I):nc
        if mark[i] == f
            i₂ += 1

            j₁ = 0

            for j in neighbors(sep, f)
                j₁ += 1

                if i₂ ≤ j₁
                    L₂₂[i₂, j₁] = X[i, j]
                end
            end
        else
            i₃ += 1

            j₁ = 0

            for j in neighbors(sep, f)
                j₁ += 1

                if i < j
                    L₃₂[j₁, i₃] = X[i, j]
                else
                    L₃₂[j₁, i₃] = X[j, i]
                end
            end
        end
    end

    return
end

function complete_dense_scatter!(
        X::AbstractMatrix{T},
        L₂₁::AbstractMatrix{T},
        W₃₁::AbstractMatrix{T},
        mark::AbstractVector{I},
        res::AbstractGraph{I},
        f::I,
        ::Val{:L},
    ) where {T, I <: Integer}
    nm = last(neighbors(res, f))
    nc = -mark[f]

    j₁ = 0

    for j in neighbors(res, f)
        j₁ += 1

        i₂ = 0
        i₃ = 0

        for i in nm + one(I):nc
            if mark[i] == f
                i₂ += 1
                X[i, j] = L₂₁[i₂, j₁]
            else
                i₃ += 1
                X[i, j] = W₃₁[i₃, j₁]
            end
        end
    end

    return
end

function complete_dense_scatter!(
        X::AbstractMatrix{T},
        L₂₁::AbstractMatrix{T},
        W₃₁::AbstractMatrix{T},
        mark::AbstractVector{I},
        res::AbstractGraph{I},
        f::I,
        ::Val{:U},
    ) where {T, I <: Integer}
    nm = last(neighbors(res, f))
    nc = -mark[f]

    i₂ = 0
    i₃ = 0

    for i in nm + one(I):nc
        if mark[i] == f
            i₂ += 1

            j₁ = 0

            for j in neighbors(res, f)
                j₁ += 1
                X[j, i] = L₂₁[j₁, i₂]
            end
        else
            i₃ += 1

            j₁ = 0

            for j in neighbors(res, f)
                j₁ += 1
                X[j, i] = W₃₁[j₁, i₃]
            end
        end
    end

    return
end
