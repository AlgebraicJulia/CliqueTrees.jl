function dfcholesky!(
        Y::AbstractCholesky,
        F::AbstractCholesky;
        adj::Bool=false,
        inv::Bool=false,
    )
    Y.info[] = dfcholesky!(triangular(Y), triangular(F); adj, inv)
    return Y
end

function dfcholesky!(
        Y::ChordalTriangular{:N, UPLO, T, I},
        F::ChordalTriangular{:N, UPLO, T, I};
        adj::Bool=false,
        inv::Bool=false,
    ) where {UPLO, T, I <: Integer}
    @assert checksymbolic(Y, F)

    Mptr = FVector{I}(undef, F.S.nMptr)
    Mval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)

    if adj && inv
        chol_diff_impl!(Mptr, Mval, Fval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, Y.S.Dptr, Y.Dval, Y.S.Lptr, Y.Lval, F.S.res, F.S.rel, F.S.chd, Val(UPLO), Val(true), Val(true))
    elseif adj
        chol_diff_impl!(Mptr, Mval, Fval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, Y.S.Dptr, Y.Dval, Y.S.Lptr, Y.Lval, F.S.res, F.S.rel, F.S.chd, Val(UPLO), Val(true), Val(false))
    elseif inv
        chol_diff_impl!(Mptr, Mval, Fval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, Y.S.Dptr, Y.Dval, Y.S.Lptr, Y.Lval, F.S.res, F.S.rel, F.S.chd, Val(UPLO), Val(false), Val(true))
    else
        chol_diff_impl!(Mptr, Mval, Fval, F.S.Dptr, F.Dval, F.S.Lptr, F.Lval, Y.S.Dptr, Y.Dval, Y.S.Lptr, Y.Lval, F.S.res, F.S.rel, F.S.chd, Val(UPLO), Val(false), Val(false))
    end

    return zero(I)
end

function chol_diff_impl!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Fval::AbstractVector{T},
        LDptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLptr::AbstractVector{I},
        LLval::AbstractVector{T},
        YDptr::AbstractVector{I},
        YDval::AbstractVector{T},
        YLptr::AbstractVector{I},
        YLval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        uplo::Val{UPLO},
        adj::Val{ADJ},
        inv::Val{INV},
    ) where {UPLO, T, I <: Integer, ADJ, INV}

    ns = zero(I); Mptr[one(I)] = one(I)

    iter = ADJ ? reverse(vertices(res)) : vertices(res)

    for j in iter
        ns = chol_diff_loop!(
            Mptr, Mval, Fval,
            LDptr, LDval, LLptr, LLval,
            YDptr, YDval, YLptr, YLval,
            res, rel, chd, ns, j, uplo, adj, inv
        )
    end

    return
end

# adj=true, inv=false: Jᵀ (reverse mode)
function chol_diff_loop!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Fval::AbstractVector{T},
        LDptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLptr::AbstractVector{I},
        LLval::AbstractVector{T},
        YDptr::AbstractVector{I},
        YDval::AbstractVector{T},
        YLptr::AbstractVector{I},
        YLval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        uplo::Val{UPLO},
        ::Val{true},
        ::Val{false},
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
    na = eltypedegree(rel, j)
    #
    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na
    #
    # F is the frontal matrix at node j
    #
    #           nn  na
    #     F = [ F₁₁     ] nn
    #         [ F₂₁ F₂₂ ] na
    #
    F = reshape(view(Fval, oneto(nj * nj)), nj, nj)

    F₁₁ = view(F, oneto(nn), oneto(nn))
    F₂₂ = view(F, nn + one(I):nj, nn + one(I):nj)

    if UPLO === :L
        F₂₁ = view(F, nn + one(I):nj, oneto(nn))
    else
        F₂₁ = view(F, oneto(nn), nn + one(I):nj)
    end
    #
    # L is the Cholesky factor at node j
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    LDp = LDptr[j]
    LLp = LLptr[j]
    L₁₁ = reshape(view(LDval, LDp:LDp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(LLval, LLp:LLp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(LLval, LLp:LLp + nn * na - one(I)), nn, na)
    end
    #
    # Y is the sensitivity matrix at node j
    #
    #          res(j)
    #     Y = [ Y₁₁  ] res(j)
    #         [ Y₂₁  ] sep(j)
    #
    YDp = YDptr[j]
    YLp = YLptr[j]
    Y₁₁ = reshape(view(YDval, YDp:YDp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        Y₂₁ = reshape(view(YLval, YLp:YLp + nn * na - one(I)), na, nn)
    else
        Y₂₁ = reshape(view(YLval, YLp:YLp + nn * na - one(I)), nn, na)
    end
    #
    # receive update matrix from parent
    #
    if ispositive(na)
        strt = Mptr[ns]
        M₂₂ = reshape(view(Mval, strt:strt + na * na - one(I)), na, na)
        ns -= one(I)
        #
        #     F₂₂ ← M₂₂
        #
        copytri!(F₂₂, M₂₂, uplo)
        #
        #     Y₁₁ += tril(-L₁₁⁻ᵀ Y₂₁ᵀ L₂₁)
        #
        if UPLO === :L
            gemm!(Val(:C), Val(:N), one(T), Y₂₁, L₂₁, zero(T), F₁₁)
            trsm!(Val(:L), uplo, Val(:C), Val(:N), -one(T), L₁₁, F₁₁)
        else
            gemm!(Val(:N), Val(:C), one(T), L₂₁, Y₂₁, zero(T), F₁₁)
            trsm!(Val(:R), uplo, Val(:C), Val(:N), -one(T), L₁₁, F₁₁)
        end

        addtri!(Y₁₁, F₁₁, uplo)
        #
        #     Y₁₁ += tril(2 L₁₁⁻ᵀ L₂₁ᵀ M L₂₁)
        #
        if UPLO === :L
            symm!(Val(:L), uplo, one(T), M₂₂, L₂₁, zero(T), F₂₁)
            gemm!(Val(:C), Val(:N), one(T), L₂₁, F₂₁, zero(T), F₁₁)
            trsm!(Val(:L), uplo, Val(:C), Val(:N), two(T), L₁₁, F₁₁)
        else
            symm!(Val(:R), uplo, one(T), M₂₂, L₂₁, zero(T), F₂₁)
            gemm!(Val(:N), Val(:C), one(T), F₂₁, L₂₁, zero(T), F₁₁)
            trsm!(Val(:R), uplo, Val(:C), Val(:N), two(T), L₁₁, F₁₁)
        end

        addtri!(Y₁₁, F₁₁, uplo)
        #
        #     Y₂₁ ← (Y₂₁ - 2 M L₂₁) L₁₁⁻¹ / 2
        #
        if UPLO === :L
            symm!(Val(:L), uplo, -two(T), M₂₂, L₂₁, one(T), Y₂₁)
            trsm!(Val(:R), uplo, Val(:N), Val(:N), one(T), L₁₁, Y₂₁)
        else
            symm!(Val(:R), uplo, -two(T), M₂₂, L₂₁, one(T), Y₂₁)
            trsm!(Val(:L), uplo, Val(:N), Val(:N), one(T), L₁₁, Y₂₁)
        end

        rdiv!(Y₂₁, two(T))
    end
    #
    #     Σ̄₁₁ = Φ(L⁻ᵀ (Φ(Lᵀ L̄) + Φ(Lᵀ L̄)ᵀ) L⁻¹)
    #
    chol_diff_factor!(Y₁₁, L₁₁, uplo, Val(true), Val(false))
    #
    # build frontal matrix and send update to children
    #
    copytri!(F₁₁, Y₁₁, uplo)
    copyrec!(F₂₁, Y₂₁)

    for i in neighbors(chd, j)
        ns += one(I)
        chol_diff_send!(F, Mptr, Mval, rel, ns, i, uplo)
    end

    return ns
end

function chol_diff_factor!(
        Y::AbstractMatrix{T},
        L::AbstractMatrix{T},
        uplo::Val{UPLO},
        ::Val{true},
        ::Val{false},
    ) where {T, UPLO}
    #
    #     Y ← Lᵀ tril(Y)
    #
    if UPLO === :L
        tril!(Y)
        trmm!(Val(:L), uplo, Val(:C), Val(:N), one(T), L, Y)
    else
        triu!(Y)
        trmm!(Val(:R), uplo, Val(:C), Val(:N), one(T), L, Y)
    end
    #
    #     Y ← Y + Yᵀ
    #
    symmtri!(Y, uplo)
    #
    #     Y ← L⁻ᵀ Y L⁻¹
    #
    if UPLO === :L
        trsm!(Val(:L), uplo, Val(:C), Val(:N), one(T), L, Y)
        trsm!(Val(:R), uplo, Val(:N), Val(:N), one(T), L, Y)
    else
        trsm!(Val(:R), uplo, Val(:C), Val(:N), one(T), L, Y)
        trsm!(Val(:L), uplo, Val(:N), Val(:N), one(T), L, Y)
    end
    #
    #     Y ← tril(Y) / 2
    #
    rdivtri!(Y, two(T), uplo)

    return
end

function chol_diff_send!(
        F::AbstractMatrix{T},
        ptr::AbstractVector{I},
        val::AbstractVector{T},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
        uplo::Val{UPLO},
    ) where {UPLO, T, I <: Integer}
    #
    # na is the size of the separator at node i
    #
    #     na = | sep(i) |
    #
    na = eltypedegree(rel, i)
    #
    # inj is the subset inclusion
    #
    #     inj: sep(i) → bag(parent(i))
    #
    inj = neighbors(rel, i)
    #
    # M is the update matrix for child i
    #
    strt = ptr[ns]
    stop = ptr[ns + one(I)] = strt + na * na
    M = reshape(view(val, strt:stop - one(I)), na, na)
    #
    #     M ← injᵀ F inj
    #
    copygathertri!(M, F, inj, uplo)

    return
end

# ===== Forward mode: J: dΣ → dL =====

# adj=false, inv=false: J (forward mode)
function chol_diff_loop!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Fval::AbstractVector{T},
        LDptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLptr::AbstractVector{I},
        LLval::AbstractVector{T},
        YDptr::AbstractVector{I},
        YDval::AbstractVector{T},
        YLptr::AbstractVector{I},
        YLval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        uplo::Val{UPLO},
        ::Val{false},
        ::Val{false},
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
    na = eltypedegree(rel, j)
    #
    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na
    #
    # F is the frontal perturbation matrix at node j
    #
    #            nn  na
    #     dF = [ dF₁₁      ] nn
    #          [ dF₂₁ dF₂₂ ] na
    #
    F = reshape(view(Fval, oneto(nj * nj)), nj, nj)

    F₁₁ = view(F, oneto(nn), oneto(nn))
    F₂₂ = view(F, nn + one(I):nj, nn + one(I):nj)

    if UPLO === :L
        F₂₁ = view(F, nn + one(I):nj, oneto(nn))
    else
        F₂₁ = view(F, oneto(nn), nn + one(I):nj)
    end
    #
    # L is the Cholesky factor at node j
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    LDp = LDptr[j]
    LLp = LLptr[j]
    L₁₁ = reshape(view(LDval, LDp:LDp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(LLval, LLp:LLp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(LLval, LLp:LLp + nn * na - one(I)), nn, na)
    end
    #
    # Y is the direction matrix at node j
    #
    #          res(j)
    #     Y = [ Y₁₁  ] res(j)
    #         [ Y₂₁  ] sep(j)
    #
    YDp = YDptr[j]
    YLp = YLptr[j]
    Y₁₁ = reshape(view(YDval, YDp:YDp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        Y₂₁ = reshape(view(YLval, YLp:YLp + nn * na - one(I)), na, nn)
    else
        Y₂₁ = reshape(view(YLval, YLp:YLp + nn * na - one(I)), nn, na)
    end
    #
    # receive update matrices from children
    #
    zerotri!(F₁₁, uplo)
    zerorec!(F₂₁)
    zerotri!(F₂₂, uplo)

    for i in Iterators.reverse(neighbors(chd, j))
        chol_diff_recv!(F, Mptr, Mval, rel, ns, i, uplo)
        ns -= one(I)
    end
    #
    #     F₁₁ ← F₁₁ + Y₁₁
    #     F₂₁ ← F₂₁ + Y₂₁
    #
    addtri!(F₁₁, Y₁₁, uplo)
    addrec!(F₂₁, Y₂₁)
    #
    #     dL₁₁ = L₁₁ Φ(L₁₁⁻¹ sym(dΣ₁₁) L₁₁⁻ᵀ)
    #
    chol_diff_factor!(F₁₁, L₁₁, uplo, Val(false), Val(false))
    #
    # send update matrix to parent
    #
    if ispositive(na)
        ns += one(I)
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * na
        M₂₂ = reshape(view(Mval, strt:stop - one(I)), na, na)
        #
        #     M₂₂ ← F₂₂
        #
        copytri!(M₂₂, F₂₂, uplo)
        #
        #     F₂₁ ← F₂₁ - L₂₁ F₁₁ᵀ
        #     F₂₁ ← F₂₁ L₁₁⁻ᵀ
        #     M₂₂ ← M₂₂ - F₂₁ L₂₁ᵀ - L₂₁ F₂₁ᵀ
        #
        if UPLO === :L
            gemm!(Val(:N), Val(:C), -one(T), L₂₁, F₁₁, one(T), F₂₁)
            trsm!(Val(:R), uplo, Val(:C), Val(:N), one(T), L₁₁, F₂₁)
            syr2k!(uplo, Val(:N), -one(T), F₂₁, L₂₁, one(T), M₂₂)
        else
            gemm!(Val(:C), Val(:N), -one(T), F₁₁, L₂₁, one(T), F₂₁)
            trsm!(Val(:L), uplo, Val(:C), Val(:N), one(T), L₁₁, F₂₁)
            syr2k!(uplo, Val(:C), -one(T), F₂₁, L₂₁, one(T), M₂₂)
        end
    end
    #
    # build frontal matrix
    #
    copytri!(Y₁₁, F₁₁, uplo)
    copyrec!(Y₂₁, F₂₁)

    return ns
end

function chol_diff_factor!(
        Y::AbstractMatrix{T},
        L::AbstractMatrix{T},
        uplo::Val{UPLO},
        ::Val{false},
        ::Val{false},
    ) where {T, UPLO}
    #
    #     Y ← sym(Y)
    #
    symmtri!(Y, uplo)
    #
    #     Y ← L⁻¹ Y L⁻ᵀ
    #
    if UPLO === :L
        trsm!(Val(:L), uplo, Val(:N), Val(:N), one(T), L, Y)
        trsm!(Val(:R), uplo, Val(:C), Val(:N), one(T), L, Y)
    else
        trsm!(Val(:R), uplo, Val(:N), Val(:N), one(T), L, Y)
        trsm!(Val(:L), uplo, Val(:C), Val(:N), one(T), L, Y)
    end
    #
    #     Y ← Φ(Y) = tril(Y) / 2
    #
    if UPLO === :L
        tril!(Y)
    else
        triu!(Y)
    end

    @inbounds for j in diagind(Y)
        Y[j] /= two(T)
    end
    #
    #     Y ← L Y
    #
    if UPLO === :L
        trmm!(Val(:L), uplo, Val(:N), Val(:N), one(T), L, Y)
    else
        trmm!(Val(:R), uplo, Val(:N), Val(:N), one(T), L, Y)
    end

    return
end

function chol_diff_recv!(
        F::AbstractMatrix{T},
        ptr::AbstractVector{I},
        val::AbstractVector{T},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
        uplo::Val{UPLO},
    ) where {UPLO, T, I <: Integer}
    #
    # na is the size of the separator at node i
    #
    #     na = | sep(i) |
    #
    na = eltypedegree(rel, i)
    #
    # inj is the subset inclusion
    #
    #     inj: sep(i) → bag(parent(i))
    #
    inj = neighbors(rel, i)
    #
    # M is the update matrix from child i
    #
    strt = ptr[ns]
    M = reshape(view(val, strt:strt + na * na - one(I)), na, na)
    #
    #     F ← F + inj M injᵀ
    #
    addscattertri!(F, M, inj, uplo)

    return
end

# ===== Inverse mode: J⁻¹: dL → dΣ =====

# adj=false, inv=true: J⁻¹ (inverse differential)
function chol_diff_loop!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Fval::AbstractVector{T},
        LDptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLptr::AbstractVector{I},
        LLval::AbstractVector{T},
        YDptr::AbstractVector{I},
        YDval::AbstractVector{T},
        YLptr::AbstractVector{I},
        YLval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        uplo::Val{UPLO},
        ::Val{false},
        ::Val{true},
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
    na = eltypedegree(rel, j)
    #
    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na
    #
    # F is the frontal matrix at node j
    #
    #            nn  na
    #     dF = [ dF₁₁      ] nn
    #          [ dF₂₁ dF₂₂ ] na
    #
    F = reshape(view(Fval, oneto(nj * nj)), nj, nj)

    F₁₁ = view(F, oneto(nn), oneto(nn))
    F₂₂ = view(F, nn + one(I):nj, nn + one(I):nj)

    if UPLO === :L
        F₂₁ = view(F, nn + one(I):nj, oneto(nn))
    else
        F₂₁ = view(F, oneto(nn), nn + one(I):nj)
    end
    #
    # L is the Cholesky factor at node j
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    LDp = LDptr[j]
    LLp = LLptr[j]
    L₁₁ = reshape(view(LDval, LDp:LDp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(LLval, LLp:LLp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(LLval, LLp:LLp + nn * na - one(I)), nn, na)
    end
    #
    # Y is the direction matrix at node j
    #
    #          res(j)
    #     Y = [ Y₁₁  ] res(j)
    #         [ Y₂₁  ] sep(j)
    #
    YDp = YDptr[j]
    YLp = YLptr[j]
    Y₁₁ = reshape(view(YDval, YDp:YDp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        Y₂₁ = reshape(view(YLval, YLp:YLp + nn * na - one(I)), na, nn)
    else
        Y₂₁ = reshape(view(YLval, YLp:YLp + nn * na - one(I)), nn, na)
    end
    #
    # receive update matrices from children
    #
    zerotri!(F₁₁, uplo)
    zerorec!(F₂₁)
    zerotri!(F₂₂, uplo)

    for i in Iterators.reverse(neighbors(chd, j))
        chol_diff_recv!(F, Mptr, Mval, rel, ns, i, uplo)
        ns -= one(I)
    end
    #
    # send update matrix to parent
    #
    if ispositive(na)
        ns += one(I)
        strt = Mptr[ns]
        stop = Mptr[ns + one(I)] = strt + na * na
        M₂₂ = reshape(view(Mval, strt:stop - one(I)), na, na)
        #
        #     M₂₂ ← F₂₂ + dL₂₁ L₂₁ᵀ + L₂₁ dL₂₁ᵀ
        #     F₂₁ ← F₂₁ + L₂₁ dL₁₁ᵀ
        #     Y₂₁ ← dL₂₁ L₁₁ᵀ + F₂₁
        #
        copytri!(M₂₂, F₂₂, uplo)

        if UPLO === :L
            syr2k!(uplo, Val(:N), one(T), Y₂₁, L₂₁, one(T), M₂₂)
            trmm!(Val(:R), uplo, Val(:C), Val(:N), one(T), L₁₁, Y₂₁)
        else
            syr2k!(uplo, Val(:C), one(T), Y₂₁, L₂₁, one(T), M₂₂)
            trmm!(Val(:L), uplo, Val(:C), Val(:N), one(T), L₁₁, Y₂₁)
        end

        addrec!(Y₂₁, F₂₁)
        copyrec!(F₂₁, L₂₁)

        if UPLO === :L
            trmm!(Val(:R), uplo, Val(:C), Val(:N), one(T), Y₁₁, F₂₁)
        else
            trmm!(Val(:L), uplo, Val(:C), Val(:N), one(T), Y₁₁, F₂₁)
        end

        addrec!(Y₂₁, F₂₁)
    end
    #
    #     dΣ₁₁ = sym(dL₁₁ L₁₁ᵀ)
    #
    chol_diff_factor!(Y₁₁, L₁₁, uplo, Val(false), Val(true))
    #
    #     Y₁₁ ← Y₁₁ + F₁₁
    #
    addtri!(Y₁₁, F₁₁, uplo)

    return ns
end

function chol_diff_factor!(
        Y::AbstractMatrix{T},
        L::AbstractMatrix{T},
        uplo::Val{UPLO},
        ::Val{false},
        ::Val{true},
    ) where {T, UPLO}
    #
    #     Y ← tril(Y)
    #
    if UPLO === :L
        tril!(Y)
    else
        triu!(Y)
    end
    #
    #     Y ← Y Lᵀ
    #
    if UPLO === :L
        trmm!(Val(:R), uplo, Val(:C), Val(:N), one(T), L, Y)
    else
        trmm!(Val(:L), uplo, Val(:C), Val(:N), one(T), L, Y)
    end
    #
    #     Y ← Y + Yᵀ
    #
    @inbounds for j in axes(Y, 1)
        if UPLO === :L
            irng = j:last(axes(Y, 1))
        else
            irng = first(axes(Y, 1)):j
        end

        for i in irng
            Y[i, j] += Y[j, i]
        end
    end

    return
end

# ===== Adjoint of inverse mode: J⁻ᵀ: Σ̄ → L̄ =====

# adj=true, inv=true: J⁻ᵀ (adjoint of inverse)
function chol_diff_loop!(
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Fval::AbstractVector{T},
        LDptr::AbstractVector{I},
        LDval::AbstractVector{T},
        LLptr::AbstractVector{I},
        LLval::AbstractVector{T},
        YDptr::AbstractVector{I},
        YDval::AbstractVector{T},
        YLptr::AbstractVector{I},
        YLval::AbstractVector{T},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        uplo::Val{UPLO},
        ::Val{true},
        ::Val{true},
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
    na = eltypedegree(rel, j)
    #
    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na
    #
    # F is the frontal matrix at node j
    #
    #           nn  na
    #     F = [ F₁₁     ] nn
    #         [ F₂₁ F₂₂ ] na
    #
    F = reshape(view(Fval, oneto(nj * nj)), nj, nj)

    F₁₁ = view(F, oneto(nn), oneto(nn))
    F₂₂ = view(F, nn + one(I):nj, nn + one(I):nj)

    if UPLO === :L
        F₂₁ = view(F, nn + one(I):nj, oneto(nn))
    else
        F₂₁ = view(F, oneto(nn), nn + one(I):nj)
    end
    #
    # L is the Cholesky factor at node j
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    LDp = LDptr[j]
    LLp = LLptr[j]
    L₁₁ = reshape(view(LDval, LDp:LDp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(LLval, LLp:LLp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(LLval, LLp:LLp + nn * na - one(I)), nn, na)
    end
    #
    # Y is the sensitivity matrix at node j
    #
    #          res(j)
    #     Y = [ Y₁₁  ] res(j)
    #         [ Y₂₁  ] sep(j)
    #
    YDp = YDptr[j]
    YLp = YLptr[j]
    Y₁₁ = reshape(view(YDval, YDp:YDp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        Y₂₁ = reshape(view(YLval, YLp:YLp + nn * na - one(I)), na, nn)
    else
        Y₂₁ = reshape(view(YLval, YLp:YLp + nn * na - one(I)), nn, na)
    end
    #
    #     F₁₁ ← Σ̄₁₁
    #     F₂₁ ← Σ̄₂₁
    #
    copytri!(F₁₁, Y₁₁, uplo)
    copyrec!(F₂₁, Y₂₁)
    #
    # receive update matrix from parent
    #
    #     L̄₁₁ = 2 tril(sym(Σ̄₁₁) L₁₁)
    #
    chol_diff_factor!(Y₁₁, L₁₁, uplo, Val(true), Val(true))
    #
    #     F₂₂ ← M₂₂
    #     L̄₂₁ = 2 Σ̄₂₁ L₁₁ + 2 M̄₂₂ L₂₁
    #     L̄₁₁ += 2 Σ̄₂₁ᵀ L₂₁
    #
    if ispositive(na)
        strt = Mptr[ns]
        M₂₂ = reshape(view(Mval, strt:strt + na * na - one(I)), na, na)
        ns -= one(I)

        copytri!(F₂₂, M₂₂, uplo)

        if UPLO === :L
            trmm!(Val(:R), uplo, Val(:N), Val(:N), two(T), L₁₁, Y₂₁)
            symm!(Val(:L), uplo, two(T), M₂₂, L₂₁, one(T), Y₂₁)
            gemmt!(uplo, Val(:C), Val(:N), two(T), F₂₁, L₂₁, one(T), Y₁₁)
        else
            trmm!(Val(:L), uplo, Val(:N), Val(:N), two(T), L₁₁, Y₂₁)
            symm!(Val(:R), uplo, two(T), M₂₂, L₂₁, one(T), Y₂₁)
            gemmt!(uplo, Val(:N), Val(:C), two(T), L₂₁, F₂₁, one(T), Y₁₁)
        end
    end
    #
    # send update to children
    #
    for i in neighbors(chd, j)
        ns += one(I)
        chol_diff_send!(F, Mptr, Mval, rel, ns, i, uplo)
    end

    return ns
end

function chol_diff_factor!(
        Y::AbstractMatrix{T},
        L::AbstractMatrix{T},
        uplo::Val{UPLO},
        ::Val{true},
        ::Val{true},
    ) where {T, UPLO}
    #
    #     Y ← sym(Y)
    #
    symmtri!(Y, uplo)
    #
    #     Y ← 2 Y L
    #
    if UPLO === :L
        trmm!(Val(:R), uplo, Val(:N), Val(:N), two(T), L, Y)
    else
        trmm!(Val(:L), uplo, Val(:N), Val(:N), two(T), L, Y)
    end

    return
end
