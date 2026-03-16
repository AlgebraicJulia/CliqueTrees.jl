struct CMPLHessian{UPLO, T, I}
    A::FChordalCholesky{UPLO, T, I}
    B::FChordalCholesky{UPLO, T, I}
    C::FChordalCholesky{UPLO, T, I}
    Mptr::FVector{I}
    Mval::FVector{T}
    Fval::FVector{T}
    fill::FVector{I}
end

function Base.size(H::CMPLHessian)
    return (length(H.fill), length(H.fill))
end

function Base.eltype(::CMPLHessian{UPLO, T}) where {UPLO, T}
    return T
end

function LinearAlgebra.mul!(c::AbstractVector{T}, H::CMPLHessian{UPLO, T, I}, b::AbstractVector{T}) where {UPLO, T, I}
    fill!(H.C, zero(T))

    for i in eachindex(H.fill)
        setflatindex!(H.C, b[i], H.fill[i])
    end

    info = fisher_impl!(
        H.Mptr, H.Mval, H.Fval,
        H.A.S.Dptr, H.B.Dval, H.A.S.Lptr, H.B.Lval,
        H.A.S.Dptr, H.A.Dval, H.A.S.Lptr, H.A.Lval,
        H.A.S.Dptr, H.C.Dval, H.A.S.Lptr, H.C.Lval,
        H.A.S.res, H.A.S.rel, H.A.S.chd, H.B.uplo, Val(true))
    ispositive(info) && throw(PosDefException(info))

    for i in eachindex(H.fill)
        c[i] = getflatindex(H.C, H.fill[i])
    end

    return c
end

function complete!(
        F::ChordalCholesky{UPLO, T, I},
        A::HermOrSym;
        alpha::Real = 0.01,
        beta::Real = 0.5,
        tol::Real = 1e-7,
        maxiter::Integer = 50,
        cgmaxiter::Integer = 100,
    ) where {UPLO, T, I}
    C = sympermute(parent(A), F.invp, A.uplo, char(F.uplo))
    copyto!(triangular(F), C)

    fill = FVector{I}(undef, half(ndz(F.S) + ncl(F.S)) + nlz(F.S) - nnz(C))
    complete_gen_fill!(fill, F.S, C, F.uplo)

    B = similar(F)
    C = similar(F)

    Mptr = FVector{I}(undef, F.S.nMptr)
    Mval = FVector{T}(undef, F.S.nMval)
    Fval = FVector{T}(undef, F.S.nFval * F.S.nFval)
    Wval = FVector{T}(undef, F.S.nFval * F.S.nFval)

    dual = FVector{T}(undef, length(fill))
    grad = FVector{T}(undef, length(fill))
    workspace = cgworkspace(T, length(fill))
    H = CMPLHessian(F, B, C, Mptr, Mval, Fval, fill)

    complete_gen_impl!(workspace, F, B, C, H, Mptr, Mval, Fval, Wval, fill, dual, grad,
        convert(Int, cgmaxiter), convert(Int, maxiter), convert(T, alpha), convert(T, beta), convert(T, tol))

    return F
end

#
# Newton-CG for maximum determinant matrix completion.
#
# Let Ω ⊆ [n] × [n] index the specified entries, and let
#
#     P: Rⁿˣⁿ → R^Ω    denote projection onto Ω
#     E: R^Ω → Rⁿˣⁿ    denote extension by zero
#
# We solve the dual problem
#
#     maximize    logdet(C + E(y))
#     subject to  C + E(y) ≻ 0
#
# where y ∈ R^m are the fill entries. Let X be the max-det completion
# of C + E(y), and let ψ(S) = -logdet(S) be the dual barrier. Then
#
#     ∇f(y)     = -P ∇ψ(X)
#     ∇²f(y)[z] = -P ∇²ψ(X)[E(z)]
#
#
function complete_gen_impl!(
        workspace,
        A::ChordalCholesky{UPLO, T, I},
        B::ChordalCholesky{UPLO, T, I},
        C::ChordalCholesky{UPLO, T, I},
        H::CMPLHessian{UPLO, T, I},
        Mptr::AbstractVector{I},
        Mval::AbstractVector{T},
        Fval::AbstractVector{T},
        Wval::AbstractVector{T},
        fill::AbstractVector{I},
        dual::AbstractVector{T},
        grad::AbstractVector{T},
        cgmaxiter::Int,
        maxiter::Int,
        alpha::T,
        beta::T,
        tol::T,
    ) where {UPLO, T, I}

    iter = 0; drop = typemax(T)
    #
    #     y ← 0
    #
    fill!(dual, zero(T))

    while iter <= maxiter && drop >= tol
        iter += 1
        #
        #     A ← C + E(y)
        #
        for i in eachindex(fill)
            setflatindex!(A, dual[i], fill[i])
        end
        #
        #     B ← chol(-∇ψ(X))
        #
        copyto!(B, A)

        info = complete_impl!(
            Mptr, Mval, A.S.Dptr, B.Dval, A.S.Lptr, B.Lval, Fval,
            A.S.res, A.S.rel, A.S.chd, B.uplo)
        ispositive(info) && throw(PosDefException(info))
        #
        #     C ← -∇ψ(X)
        #
        copyto!(C, B)

        unchol_impl!(
            Mptr, Mval, A.S.Dptr, C.Dval, A.S.Lptr, C.Lval, C.d, Fval, Wval,
            A.S.res, A.S.rel, A.S.chd, C.uplo, C.diag)
        #
        #     g ← ∇f(y) = -P ∇ψ(X)
        #
        for i in eachindex(fill)
            grad[i] = getflatindex(C, fill[i])
        end

        if norm(grad) > one(T)
            rtol = convert(T, 1e-3)
        else
            rtol = convert(T, 1e-8)
        end
        #
        # solve for descent direction d:
        #
        #     ∇²f(y)[d] = g
        #
        desc = cgsolution(cg!(workspace, H, grad, rtol, cgmaxiter))
        #
        # Newton decrement:
        #
        #     δ ← dᵀ g
        #
        drop = dot(desc, grad)
        cost = logdet(B)
        step = one(T)
        #
        #     A ← C + E(y + d)
        #
        for i in eachindex(fill)
            setflatindex!(A, desc[i] + dual[i], fill[i])
        end
        #
        #     B ← chol(-∇ψ(X))
        #
        copyto!(B, A)

        info = complete_impl!(
            Mptr, Mval, A.S.Dptr, B.Dval, A.S.Lptr, B.Lval, Fval,
            A.S.res, A.S.rel, A.S.chd, B.uplo)
        ispositive(info) && throw(PosDefException(info))
        #
        # Armijo backtracking: find t such that
        #
        #     f(y + t d) ≤ f(y) - α t δ
        #
        while logdet(B) > cost - alpha * step * drop
            step *= beta
            #
            #     A ← C + E(y + t d)
            #
            for i in eachindex(fill)
                setflatindex!(A, step * desc[i] + dual[i], fill[i])
            end
            #
            #     B ← chol(-∇ψ(X))
            #
            copyto!(B, A)

            info = complete_impl!(
                Mptr, Mval, A.S.Dptr, B.Dval, A.S.Lptr, B.Lval, Fval,
                A.S.res, A.S.rel, A.S.chd, B.uplo)
            ispositive(info) && throw(PosDefException(info))
        end
        #
        #     y ← y + t d
        #
        axpy!(step, desc, dual)
    end

    #
    #     A ← C + E(y)
    #
    for i in eachindex(fill)
        setflatindex!(A, dual[i], fill[i])
    end
    #
    #     A ← chol(-∇ψ(X))
    #
    info = complete_impl!(
        Mptr, Mval, A.S.Dptr, A.Dval, A.S.Lptr, A.Lval, Fval,
        A.S.res, A.S.rel, A.S.chd, A.uplo)
    ispositive(info) && throw(PosDefException(info))
    return
end

function complete_gen_fill!(fill::AbstractVector{I}, S::ChordalSymbolic{I}, C::SparseMatrixCSC, ::Val{:L}) where {I}
    fp = one(I)
    dp = one(I)
    lp = one(I) + convert(I, ndz(S))

    for f in fronts(S)
        res = neighbors(S.res, f)
        sep = neighbors(S.sep, f)

        for j in res
            cp = C.colptr[j]
            cpstop = C.colptr[j + one(I)]

            for i in res
                if cp < cpstop && i == rowvals(C)[cp]
                    cp += one(I)
                elseif i >= j
                    fill[fp] = dp; fp += one(I)
                end

                dp += one(I)
            end

            for i in sep
                if cp < cpstop && i == rowvals(C)[cp]
                    cp += one(I)
                else
                    fill[fp] = lp; fp += one(I)
                end

                lp += one(I)
            end
        end
    end

    return
end

function complete_gen_fill!(fill::AbstractVector{I}, S::ChordalSymbolic{I}, C::SparseMatrixCSC, ::Val{:U}) where {I}
    fp = one(I)
    dp = one(I)
    lp = one(I) + convert(I, ndz(S))

    for f in fronts(S)
        res = neighbors(S.res, f)
        sep = neighbors(S.sep, f)

        for j in res
            cp = C.colptr[j]
            cpstop = C.colptr[j + one(I)]

            for i in res
                while cp < cpstop && rowvals(C)[cp] < i
                    cp += one(I)
                end

                if cp < cpstop && i == rowvals(C)[cp]
                    cp += one(I)
                elseif i <= j
                    fill[fp] = dp; fp += one(I)
                end

                dp += one(I)
            end
        end

        for j in sep
            cp = C.colptr[j]
            cpstop = C.colptr[j + one(I)]

            for i in res
                while cp < cpstop && rowvals(C)[cp] < i
                    cp += one(I)
                end

                if cp < cpstop && i == rowvals(C)[cp]
                    cp += one(I)
                else
                    fill[fp] = lp; fp += one(I)
                end

                lp += one(I)
            end
        end
    end

    return
end
