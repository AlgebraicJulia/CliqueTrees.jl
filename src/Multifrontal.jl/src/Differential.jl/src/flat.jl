function flat(L::ChordalTriangular{:N})
    x = fflat(clean!, L)
    return L.uplo, L.S, x
end

function flat(H::HermOrSymTri)
    L = parent(H)
    x = fflat(scale! ∘ clean!, L)
    return L.uplo, L.S, x
end

# Kernel functions for flat
function flat_frule_impl(L::ChordalTriangular{:N}, dL::ChordalTriangular{:N})
    @assert checksymbolic(L, dL)
    y = flat(L)
    _, _, dy = flat(dL)
    return y, (NoTangent(), NoTangent(), dy)
end

function flat_frule_impl(H::HermOrSymTri{UPLO}, dH::ChordalTriangular{:N}) where {UPLO}
    @assert checksymbolic(H, dH)
    y = flat(H)
    _, _, dy = flat(Hermitian(dH, UPLO))
    return y, (NoTangent(), NoTangent(), dy)
end

function flat_rrule_impl(L::ChordalTriangular{:N}, y, Δy::AbstractVector)
    return unflattri(L.uplo, L.S, Δy)
end

function flat_rrule_impl(H::HermOrSymTri, y, Δy::AbstractVector)
    return unflatsym(parent(H).uplo, parent(H).S, Δy)
end

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(flat), L::ChordalTriangular{:N})
    return flat_frule_impl(L, dL)
end

function ChainRulesCore.frule((_, dH)::Tuple, ::typeof(flat), H::HermOrSymTri{UPLO}) where {UPLO}
    return flat_frule_impl(H, dH)
end

function ChainRulesCore.rrule(::typeof(flat), L::ChordalTriangular{:N})
    y = flat(L)

    function pullback((_, _, Δy))
        if Δy isa ZeroTangent
            return NoTangent(), ZeroTangent()
        else
            return NoTangent(), flat_rrule_impl(L, y, Δy)
        end
    end

    return y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(flat), H::HermOrSymTri{UPLO}) where {UPLO}
    y = flat(H)

    function pullback((_, _, Δy))
        if Δy isa ZeroTangent
            return NoTangent(), ZeroTangent()
        else
            return NoTangent(), flat_rrule_impl(H, y, Δy)
        end
    end

    return y, pullback ∘ unthunk
end
