# frules and rrules for adjoint and transpose of ChordalTriangular
# These prevent Zygote from falling back to dense matrix conversion

# Kernel functions for adjoint and transpose
function adjoint_frule_impl(L, dL)
    return adjoint(L), dL
end

function transpose_frule_impl(L, dL)
    return transpose(L), dL
end

function adjoint_rrule_impl(L, A, ΔA)
    return ΔA
end

function transpose_rrule_impl(L, A, ΔA)
    return ΔA
end

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(adjoint), L::Union{AdjTri, ChordalTriangular})
    return adjoint_frule_impl(L, dL)
end

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(transpose), L::Union{TransTri, ChordalTriangular})
    return transpose_frule_impl(L, dL)
end

function ChainRulesCore.rrule(::typeof(adjoint), L::Union{AdjTri, ChordalTriangular})
    A = adjoint(L)

    function pullback(ΔA)
        if ΔA isa ZeroTangent
            return NoTangent(), ZeroTangent()
        else
            return NoTangent(), adjoint_rrule_impl(L, A, ΔA)
        end
    end

    return A, pullback
end

function ChainRulesCore.rrule(::typeof(transpose), L::Union{TransTri, ChordalTriangular})
    A = transpose(L)

    function pullback(ΔA)
        if ΔA isa ZeroTangent
            return NoTangent(), ZeroTangent()
        else
            return NoTangent(), transpose_rrule_impl(L, A, ΔA)
        end
    end

    return A, pullback
end
