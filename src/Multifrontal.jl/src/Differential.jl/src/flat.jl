function flat(L::ChordalTriangular{:N})
    return fflat(clean!, L)
end

function flat(H::HermOrSymTri)
    L = parent(H)
    return fflat(scale! ∘ clean!, L)
end

# Kernel functions for flat
function flat_frule_impl(L::ChordalTriangular{:N}, dL)
    if dL isa ZeroTangent
        dy = ZeroTangent()
    else
        dy = flat(dL)
    end

    return flat(L), dy
end

function flat_frule_impl(H::HermOrSymTri, dH)
    if dH isa ZeroTangent
        dy = ZeroTangent()
    else
        dy = flat(ProjectTo(H)(dH))
    end

    return flat(H), dy
end

function flat_rrule_impl(L::ChordalTriangular{:N}, y, Δy::AbstractVector)
    return unflattri(Δy, L.S, L.uplo)
end

function flat_rrule_impl(H::HermOrSymTri, y, Δy::AbstractVector)
    return unflatsym(Δy, parent(H).S, parent(H).uplo)
end

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(flat), L::ChordalTriangular{:N})
    return flat_frule_impl(L, dL)
end

function ChainRulesCore.frule((_, dH)::Tuple, ::typeof(flat), H::HermOrSymTri{UPLO}) where {UPLO}
    return flat_frule_impl(H, dH)
end

function ChainRulesCore.rrule(::typeof(flat), L::ChordalTriangular{:N})
    y = flat(L)

    function pullback(Δy)
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

    function pullback(Δy)
        if Δy isa ZeroTangent
            return NoTangent(), ZeroTangent()
        else
            return NoTangent(), flat_rrule_impl(H, y, Δy)
        end
    end

    return y, pullback ∘ unthunk
end
