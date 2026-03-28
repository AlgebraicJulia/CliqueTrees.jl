# Kernel functions for X * L, X * L', X * transpose(L)

# ===== frule_impl =====

function mul_frule_impl(X::AbstractMatrix, A::MaybeAdjOrTransTri{:N}, dX, dL)
    return X * A, dX * A + X * ProjectTo(A)(dL)
end

# ===== rrule_impl =====

function mul_rrule_impl(X::AbstractMatrix, A::MaybeAdjOrTransTri{:N}, Y::AbstractMatrix, ΔY::AbstractMatrix)
    ΔX = ΔY * A'

    ΔL = @thunk begin
        ΔA = ProjectTo(A)(similar(parent(A)))
        selupd!(ΔA, X', ΔY, 1, 0)
        parent(ΔA)
    end

    return ΔX, ΔL
end

# ===== frule / rrule =====

for T in (ChordalTriangular{:N}, AdjTri{:N}, TransTri{:N})
    @eval function ChainRulesCore.frule((_, dX, dL)::Tuple, ::typeof(*), X::AbstractMatrix, A::$T)
        return mul_frule_impl(X, A, dX, dL)
    end

    @eval function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix, A::$T)
        return mul_rrule(X, A)
    end

    # type ambiguity with ChainRules
    @eval function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix{<:RealOrComplex}, A::$T{<:Any, <:RealOrComplex})
        return mul_rrule(X, A)
    end
end
