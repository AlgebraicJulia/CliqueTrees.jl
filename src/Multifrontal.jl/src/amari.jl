# Amari-Chentsov Tensor
# =====================
#
# Computes the third derivative of the log-det barrier:
#
#     T = -P(X⁻¹ Y X⁻¹ Z X⁻¹ + X⁻¹ Z X⁻¹ Y X⁻¹)P
#
# inv=false: fwd_1 → fwd_2 → scale → bwd
# inv=true:  H⁻¹ → fwd_1 → fwd_2 → scale → bwd → H⁻¹


"""
    amari!(T, Y, Z, L, S; inv=false, check=true)

Compute the Amari-Chentsov tensor (third derivative of log-det barrier).

Given the Cholesky factor `L` of a sparse PD matrix `X`, its selected inverse
`S = selinv(L)`, and two tangent directions `Y` and `Z`, computes:

- `inv=false`: T = ∇³f(X)[Y, Z] = -P(X⁻¹ Y X⁻¹ Z X⁻¹ + X⁻¹ Z X⁻¹ Y X⁻¹)
- `inv=true`:  T = ∇³f*(S)[Y, Z] = H⁻¹(∇³f(X)[H⁻¹(Y), H⁻¹(Z)])

where H = ∇²f(X) is the Hessian at X, and H⁻¹ = fisher!(inv=true).

Note: `Y` and `Z` are used as workspace and will be overwritten.
"""
function amari!(
        T::AbstractCholesky{UPLO, V},
        Y::AbstractCholesky{UPLO, V},
        Z::AbstractCholesky{UPLO, V},
        L::AbstractCholesky{UPLO, V},
        S::AbstractCholesky{UPLO, V};
        inv::Bool=false,
        check::Bool=true,
    ) where {UPLO, V}
    info = amari!(
        triangular(T),
        triangular(Y), triangular(Z),
        triangular(L), triangular(S);
        inv, check)
    T.info[] = info
    return T
end


function amari!(
        T::ChordalTriangular{:N, UPLO, V, I},
        Y::ChordalTriangular{:N, UPLO, V, I},
        Z::ChordalTriangular{:N, UPLO, V, I},
        L::ChordalTriangular{:N, UPLO, V, I},
        S::ChordalTriangular{:N, UPLO, V, I};
        inv::Bool=false,
        check::Bool=true,
    ) where {UPLO, V, I <: Integer}
    @assert checksymbolic(T, Y, Z, L, S)

    # Allocate workspace
    Uptr = FVector{I}(undef, L.S.nMptr)
    Uval = FVector{V}(undef, L.S.nMval)   # fwd stack, then reused as Σ
    Vval = FVector{V}(undef, L.S.nMval)   # σ stack, then reused as V (T₂₂)
    Fval = FVector{V}(undef, L.S.nFval * L.S.nFval)

    if inv
        info = amari_impl!(Uptr, Uval, Vval, Fval, T, Y, Z, L, S, Val(true))
    else
        info = amari_impl!(Uptr, Uval, Vval, Fval, T, Y, Z, L, S, Val(false))
    end

    checkinfo(info, L.diag, check)

    return info
end


function amari_impl!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{V},
        Vval::AbstractVector{V},
        Fval::AbstractVector{V},
        T::ChordalTriangular{:N, UPLO, V, I},
        Y::ChordalTriangular{:N, UPLO, V, I},
        Z::ChordalTriangular{:N, UPLO, V, I},
        L::ChordalTriangular{:N, UPLO, V, I},
        S::ChordalTriangular{:N, UPLO, V, I},
        inv::Val{INV},
    ) where {UPLO, V, I <: Integer, INV}
    #
    # fwd_1: linearize Cholesky once per direction
    #
    if INV
        fisherroot_bwd!(Uptr, Uval, Fval, L, Y, Val(true))
        info = fisher_scale!(Uptr, Uval, Fval, L, S, Y, Val(true))
        ispositive(info) && return info
    else
        fisherroot_fwd!(Uptr, Uval, Fval, L, Y, Val(false))
    end

    if Y !== Z
        if INV
            fisherroot_bwd!(Uptr, Uval, Fval, L, Z, Val(true))
            info = fisher_scale!(Uptr, Uval, Fval, L, S, Z, Val(true))
            ispositive(info) && return info
        else
            fisherroot_fwd!(Uptr, Uval, Fval, L, Z, Val(false))
        end
    end
    #
    # fwd: linearize Cholesky twice → T
    #
    amari_fwd!(Uptr, Uval, Fval, T, Y, Z, L)
    #
    # Phase 2: T₂₁ ← S₂₂ T₂₁ (via fisher_scale! with inv=false)
    # Uval now reused as Σ stack (S values)
    #
    fisher_scale!(Uptr, Uval, Fval, L, S, T, Val(false))
    #
    # Phase 3: Y direction (Uval=Σ, Vval=σ)
    #
    amari_bwd!(Uptr, Uval, Vval, Fval, T, Y, Z, L, S)
    #
    # Phase 4: Z direction (Uval=Σ, Vval=σ reused)
    #
    amari_bwd!(Uptr, Uval, Vval, Fval, T, Z, Y, L, S)
    #
    # bwd: phase 5 + L scaling of T + T-frontal emit
    #
    if INV
        # H⁻¹ pushforward: bwd(true) cancels bwd(false), leaving scale + fwd
        info = fisher_scale!(Uptr, Uval, Fval, L, S, T, Val(true))
        ispositive(info) && return info
        fisherroot_fwd!(Uptr, Uval, Fval, L, T, Val(true))
    else
        fisherroot_bwd!(Uptr, Vval, Fval, L, T, Val(false))
    end

    return zero(I)
end


# Forward Pass
# ============


function amari_fwd!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{V},
        Fval::AbstractVector{V},
        T::ChordalTriangular{:N, UPLO, V, I},
        Y::ChordalTriangular{:N, UPLO, V, I},
        Z::ChordalTriangular{:N, UPLO, V, I},
        L::ChordalTriangular{:N, UPLO, V, I},
    ) where {UPLO, V, I <: Integer}

    ns = zero(I); Uptr[one(I)] = one(I)

    for j in vertices(L.S.res)
        ns = amari_fwd_loop!(
            Uptr, Uval, Fval,
            L.S.Dptr, L.S.Lptr,
            T.Dval, T.Lval,
            Y.Dval, Y.Lval,
            Z.Dval, Z.Lval,
            L.Dval, L.Lval,
            L.S.res, L.S.rel, L.S.chd, ns, j, L.uplo)
    end

    return
end


function amari_fwd_loop!(
        Uptr::AbstractVector{I},
        Uval::AbstractVector{V},
        Fval::AbstractVector{V},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        TDval::AbstractVector{V},
        TLval::AbstractVector{V},
        YDval::AbstractVector{V},
        YLval::AbstractVector{V},
        ZDval::AbstractVector{V},
        ZLval::AbstractVector{V},
        LDval::AbstractVector{V},
        LLval::AbstractVector{V},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        uplo::Val{UPLO},
    ) where {UPLO, V, I <: Integer}
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
    # Dp and Lp are indices into the diagonal and off-diagonal blocks
    #
    Dp = Dptr[j]
    Lp = Lptr[j]
    T₁₁ = reshape(view(TDval, Dp:Dp + nn * nn - one(I)), nn, nn)
    Y₁₁ = reshape(view(YDval, Dp:Dp + nn * nn - one(I)), nn, nn)
    Z₁₁ = reshape(view(ZDval, Dp:Dp + nn * nn - one(I)), nn, nn)
    L₁₁ = reshape(view(LDval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        T₂₁ = reshape(view(TLval, Lp:Lp + nn * na - one(I)), na, nn)
        Y₂₁ = reshape(view(YLval, Lp:Lp + nn * na - one(I)), na, nn)
        Z₂₁ = reshape(view(ZLval, Lp:Lp + nn * na - one(I)), na, nn)
        L₂₁ = reshape(view(LLval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        T₂₁ = reshape(view(TLval, Lp:Lp + nn * na - one(I)), nn, na)
        Y₂₁ = reshape(view(YLval, Lp:Lp + nn * na - one(I)), nn, na)
        Z₂₁ = reshape(view(ZLval, Lp:Lp + nn * na - one(I)), nn, na)
        L₂₁ = reshape(view(LLval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    #     F ← 0
    #
    zerotri!(F₁₁, uplo)
    zerorec!(F₂₁)
    zerotri!(F₂₂, uplo)

    for i in Iterators.reverse(neighbors(chd, j))
        #
        # add the update matrix for child i to F
        #
        #     F ← F + Rᵢ Uᵢ Rᵢᵀ
        #
        amari_add_update!(F, Uptr, Uval, rel, ns, i, uplo)
        ns -= one(I)
    end
    #
    #     T₁₁ ← F₁₁
    #
    copytri!(T₁₁, F₁₁, uplo)

    if UPLO === :L
        sdR, sdL = Val(:R), Val(:L)
    else
        sdR, sdL = Val(:L), Val(:R)
    end
    #
    #     T₁₁ ← L₁₁⁻¹ (sym T₁₁) L₁₁⁻ᴴ
    #
    symmtri!(T₁₁, uplo)
    trsm!(sdL, uplo, Val(:N), Val(:N), one(V), L₁₁, T₁₁)
    trsm!(sdR, uplo, Val(:C), Val(:N), one(V), L₁₁, T₁₁)

    if ispositive(na)
        #
        # U₂₂ is the update matrix for node j
        #
        ns += one(I)
        strt = Uptr[ns]
        stop = Uptr[ns + one(I)] = strt + na * na
        U₂₂ = reshape(view(Uval, strt:stop - one(I)), na, na)
        #
        #     U₂₂ ← F₂₂
        #
        copytri!(U₂₂, F₂₂, uplo)
        #
        #     T₂₁ ← F₂₁ L₁₁⁻ᴴ
        #
        copyrec!(T₂₁, F₂₁)
        trsm!(sdR, uplo, Val(:C), Val(:N), one(V), L₁₁, T₂₁)
        #
        #     U₂₂ -= syr2k(Y₂₁, Z₂₁)
        #
        if UPLO === :L
            syr2k!(uplo, Val(:N), -one(real(V)), Y₂₁, Z₂₁, one(real(V)), U₂₂)
        else
            syr2k!(uplo, Val(:C), -one(real(V)), Z₂₁, Y₂₁, one(real(V)), U₂₂)
        end
        #
        #     U₂₂ -= syr2k(L₂₁, T₂₁)
        #     T₂₁ -= L₂₁ T₁₁
        #     U₂₂ -= syr2k(T₂₁, L₂₁)
        #
        if UPLO === :L
            gemmt!(uplo, Val(:N), Val(:C), -one(real(V)), L₂₁, T₂₁, one(real(V)), U₂₂)
        else
            gemmt!(uplo, Val(:C), Val(:N), -one(real(V)), T₂₁, L₂₁, one(real(V)), U₂₂)
        end

        symm!(sdR, uplo, -one(V), T₁₁, L₂₁, one(V), T₂₁)

        if UPLO === :L
            gemmt!(uplo, Val(:N), Val(:C), -one(real(V)), T₂₁, L₂₁, one(real(V)), U₂₂)
        else
            gemmt!(uplo, Val(:C), Val(:N), -one(real(V)), L₂₁, T₂₁, one(real(V)), U₂₂)
        end
        #
        #     T₂₁ -= Y₂₁ Z₁₁ + Z₂₁ Y₁₁
        #
        symm!(sdR, uplo, -one(V), Z₁₁, Y₂₁, one(V), T₂₁)
        symm!(sdR, uplo, -one(V), Y₁₁, Z₂₁, one(V), T₂₁)
    end
    #
    # Cross-coupling from second-order product rule (only writes uplo triangle;
    # scale's symmtri! will re-sync before un-sandwich)
    #
    #     T₁₁ -= syr2k(Y₁₁, Z₁₁)
    #
    symmtri!(Y₁₁, uplo)
    symmtri!(Z₁₁, uplo)
    syr2k!(uplo, Val(:N), -one(real(V)), Y₁₁, Z₁₁, one(real(V)), T₁₁)

    return ns
end


# amari_bwd!
# ============
#
# Top-level function for single direction backward pass.
# Reads S from Σ stack, reads σ from σ stack, does phase 3 or 4 work,
# emits σ to children, passes S through to children via Σ stack.


function amari_bwd!(
        Uptr::AbstractVector{I},
        ΣUval::AbstractVector{V},
        σUval::AbstractVector{V},
        Fval::AbstractVector{V},
        T::ChordalTriangular{:N, UPLO, V, I},
        Y::ChordalTriangular{:N, UPLO, V, I},
        Z::ChordalTriangular{:N, UPLO, V, I},
        L::ChordalTriangular{:N, UPLO, V, I},
        S::ChordalTriangular{:N, UPLO, V, I},
    ) where {UPLO, V, I <: Integer}

    Uptr[one(I)] = one(I)
    ns = zero(I)

    for j in reverse(vertices(L.S.res))
        ns = amari_bwd_loop!(
            Uptr, ΣUval, σUval, Fval,
            L.S.Dptr, L.S.Lptr,
            T.Dval, T.Lval,
            Y.Dval, Y.Lval,
            Z.Dval, Z.Lval,
            L.Dval, L.Lval,
            S.Dval, S.Lval,
            L.S.res, L.S.rel, L.S.chd,
            ns, j, L.uplo)
    end

    return
end


function amari_bwd_loop!(
        Uptr::AbstractVector{I},
        ΣUval::AbstractVector{V},
        σUval::AbstractVector{V},
        Fval::AbstractVector{V},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        TDval::AbstractVector{V},
        TLval::AbstractVector{V},
        YDval::AbstractVector{V},
        YLval::AbstractVector{V},
        ZDval::AbstractVector{V},
        ZLval::AbstractVector{V},
        LDval::AbstractVector{V},
        LLval::AbstractVector{V},
        SDval::AbstractVector{V},
        SLval::AbstractVector{V},
        res::AbstractGraph{I},
        rel::AbstractGraph{I},
        chd::AbstractGraph{I},
        ns::I,
        j::I,
        uplo::Val{UPLO},
    ) where {UPLO, V, I <: Integer}

    nn = eltypedegree(res, j)
    na = eltypedegree(rel, j)
    nj = nn + na

    F = reshape(view(Fval, oneto(nj * nj)), nj, nj)
    F₁₁ = view(F, oneto(nn), oneto(nn))
    F₂₂ = view(F, nn + one(I):nj, nn + one(I):nj)

    if UPLO === :L
        F₂₁ = view(F, nn + one(I):nj, oneto(nn))
        sdR, sdL = Val(:R), Val(:L)
    else
        F₂₁ = view(F, oneto(nn), nn + one(I):nj)
        sdR, sdL = Val(:L), Val(:R)
    end

    Dp = Dptr[j]
    Lp = Lptr[j]

    T₁₁ = reshape(view(TDval, Dp:Dp + nn * nn - one(I)), nn, nn)
    Y₁₁ = reshape(view(YDval, Dp:Dp + nn * nn - one(I)), nn, nn)
    L₁₁ = reshape(view(LDval, Dp:Dp + nn * nn - one(I)), nn, nn)
    S₁₁ = reshape(view(SDval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        T₂₁ = reshape(view(TLval, Lp:Lp + nn * na - one(I)), na, nn)
        Y₂₁ = reshape(view(YLval, Lp:Lp + nn * na - one(I)), na, nn)
        Z₂₁ = reshape(view(ZLval, Lp:Lp + nn * na - one(I)), na, nn)
        L₂₁ = reshape(view(LLval, Lp:Lp + nn * na - one(I)), na, nn)
        S₂₁ = reshape(view(SLval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        T₂₁ = reshape(view(TLval, Lp:Lp + nn * na - one(I)), nn, na)
        Y₂₁ = reshape(view(YLval, Lp:Lp + nn * na - one(I)), nn, na)
        Z₂₁ = reshape(view(ZLval, Lp:Lp + nn * na - one(I)), nn, na)
        L₂₁ = reshape(view(LLval, Lp:Lp + nn * na - one(I)), nn, na)
        S₂₁ = reshape(view(SLval, Lp:Lp + nn * na - one(I)), nn, na)
    end

    if UPLO === :L
        trN, trC = Val(:C), Val(:N)
    else
        trN, trC = Val(:N), Val(:C)
    end
    #
    #     F₁₁ ← Y₁₁ - L₂₁ᴴ S₂₂ Y₂₁ - Y₂₁ᴴ S₂₂ L₂₁ + L₂₁ᴴ σ₂₂ L₂₁
    #     F₂₁ ←                              S₂₂ Y₂₁ -      σ₂₂ L₂₁
    #     T₁₁ ← T₁₁                 - Z₂₁ᴴ S₂₂ Y₂₁
    #     T₂₁ ← T₂₁                                  -      σ₂₂ Z₂₁
    #
    copytri!(F₁₁, Y₁₁, uplo)

    if ispositive(na)
        #
        # Pull S₂₂ and σ₂₂ from parent's emits
        #
        strt = Uptr[ns]; ns -= one(I)
        S₂₂ = reshape(view(ΣUval, strt:strt + na * na - one(I)), na, na)
        σ₂₂ = reshape(view(σUval, strt:strt + na * na - one(I)), na, na)

        copytri!(F₂₂, σ₂₂, uplo)
        symm!(sdL, uplo,  one(V),  S₂₂, Y₂₁, zero(V), F₂₁)
        gemmt!(uplo, trN, trC, -one(real(V)), Z₂₁, F₂₁, one(real(V)), T₁₁)
        symm!(sdL, uplo, -one(V), σ₂₂, Z₂₁, one(V),  T₂₁)
        gemmt!(uplo, trN, trC, -one(real(V)), L₂₁, F₂₁, one(real(V)), F₁₁)
        symm!(sdL, uplo, -one(V), σ₂₂, L₂₁, one(V),  F₂₁)
        gemmt!(uplo, trN, trC, -one(real(V)), F₂₁, L₂₁, one(real(V)), F₁₁)
        trsm!(sdR, uplo, Val(:N), Val(:N), one(V), L₁₁, F₂₁)
    end
    #
    # Scale F₁₁
    #
    #     F₁₁ ← L₁₁⁻ᴴ sym(F₁₁) L₁₁⁻¹
    #
    symmtri!(F₁₁, uplo)
    trsm!(sdL, uplo, Val(:C), Val(:N), one(V), L₁₁, F₁₁)
    trsm!(sdR, uplo, Val(:N), Val(:N), one(V), L₁₁, F₁₁)

    # σ emit to children
    ns_start = ns

    for i in neighbors(chd, j)
        ns += one(I)
        amari_get_update!(F, Uptr, σUval, rel, ns, i, uplo)
    end

    # Σ pass-through: emit S to children
    #
    #     F ← S (at this node)
    #
    copytri!(F₁₁, S₁₁, uplo)

    if ispositive(na)
        copyrec!(F₂₁, S₂₁)
        copytri!(F₂₂, S₂₂, uplo)
    end

    ns_local = ns_start

    for i in neighbors(chd, j)
        ns_local += one(I)
        amari_get_update!(F, Uptr, ΣUval, rel, ns_local, i, uplo)
    end

    return ns
end


# Update Matrix Helpers
# =====================


function amari_add_update!(
        F::AbstractMatrix{V},
        ptr::AbstractVector{I},
        val::AbstractVector{V},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
        uplo::Val{UPLO},
    ) where {UPLO, V, I <: Integer}
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
    # U is the update matrix from child i
    #
    strt = ptr[ns]
    U = reshape(view(val, strt:strt + na * na - one(I)), na, na)
    #
    # add U to F
    #
    #     F ← F + inj U injᵀ
    #
    addscattertri!(F, U, inj, uplo)

    return
end


function amari_get_update!(
        F::AbstractMatrix{V},
        ptr::AbstractVector{I},
        val::AbstractVector{V},
        rel::AbstractGraph{I},
        ns::I,
        i::I,
        uplo::Val{UPLO},
    ) where {UPLO, V, I <: Integer}
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
    # U is the update matrix for child i
    #
    strt = ptr[ns]
    stop = ptr[ns + one(I)] = strt + na * na
    U = reshape(view(val, strt:stop - one(I)), na, na)
    #
    # copy F into U
    #
    #     U ← injᵀ F inj
    #
    copygathertri!(U, F, inj, uplo)
    return
end
