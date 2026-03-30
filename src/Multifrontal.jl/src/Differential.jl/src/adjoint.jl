# frules and rrules for adjoint and transpose of ChordalTriangular
# These prevent Zygote from falling back to dense matrix conversion

# Kernel functions for adjoint and transpose
function adjoint_frule_impl(L, dL)
    return adjoint(L), adjoint(dL)
end

function transpose_frule_impl(L, dL)
    return transpose(L), transpose(dL)
end

function adjoint_rrule_impl(L, A, ΔA)
    return ProjectTo(L)(adjoint(ΔA))
end

function transpose_rrule_impl(L, A, ΔA)
    return ProjectTo(L)(transpose(ΔA))
end

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(adjoint), L::Union{AdjTri{:N}, ChordalTriangular{:N}})
    return adjoint_frule_impl(L, dL)
end

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(transpose), L::Union{TransTri{:N}, ChordalTriangular{:N}})
    return transpose_frule_impl(L, dL)
end

function ChainRulesCore.rrule(::typeof(adjoint), L::Union{AdjTri{:N}, ChordalTriangular{:N}})
    A = adjoint(L)

    function pullback(ΔA)
        return NoTangent(), adjoint_rrule_impl(L, A, ΔA)
    end

    return A, pullback
end

function ChainRulesCore.rrule(::typeof(transpose), L::Union{TransTri{:N}, ChordalTriangular{:N}})
    A = transpose(L)

    function pullback(ΔA)
        return NoTangent(), transpose_rrule_impl(L, A, ΔA)
    end

    return A, pullback
end
