# Kernel functions for A + B and A - B

# ===== frule_impl =====

function add_frule_impl(A, B, dA, dB)
    return A + B, dA + dB
end

function sub_frule_impl(A, B, dA, dB)
    return A - B, dA - dB
end

# ===== frule =====

for T in CHORDAL_TYPES
    # T + T
    @eval function ChainRulesCore.frule((_, dA, dB)::Tuple, ::typeof(+), A::$T, B::$T)
        return add_frule_impl(A, B, dA, dB)
    end
    @eval function ChainRulesCore.frule((_, dA, dB)::Tuple, ::typeof(-), A::$T, B::$T)
        return sub_frule_impl(A, B, dA, dB)
    end
    # T ± Diagonal/UniformScaling
    for S in (Diagonal, UniformScaling)
        @eval function ChainRulesCore.frule((_, dA, dB)::Tuple, ::typeof(+), A::$T, B::$S)
            return add_frule_impl(A, B, dA, dB)
        end
        @eval function ChainRulesCore.frule((_, dA, dB)::Tuple, ::typeof(+), A::$S, B::$T)
            return add_frule_impl(A, B, dA, dB)
        end
        @eval function ChainRulesCore.frule((_, dA, dB)::Tuple, ::typeof(-), A::$T, B::$S)
            return sub_frule_impl(A, B, dA, dB)
        end
        @eval function ChainRulesCore.frule((_, dA, dB)::Tuple, ::typeof(-), A::$S, B::$T)
            return sub_frule_impl(A, B, dA, dB)
        end
    end
end

# ===== rrule_impl =====

function add_rrule_impl(A::MaybeHermOrSymTri{UPLO}, B::MaybeHermOrSymTri{UPLO}, Y::MaybeHermOrSymTri{UPLO}, ΔY::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, B, Y, ΔY)
    return ΔY, ΔY
end

function add_rrule_impl(A, B, Y, ΔY)
    ΔA = @thunk ProjectTo(A)(ΔY)
    ΔB = @thunk ProjectTo(B)(ΔY)
    return ΔA, ΔB
end

function sub_rrule_impl(A::MaybeHermOrSymTri{UPLO}, B::MaybeHermOrSymTri{UPLO}, Y::MaybeHermOrSymTri{UPLO}, ΔY::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, B, Y, ΔY)
    return ΔY, -ΔY
end

function sub_rrule_impl(A, B, Y, ΔY)
    ΔA = @thunk ProjectTo(A)(ΔY)
    ΔB = @thunk -ProjectTo(B)(ΔY)
    return ΔA, ΔB
end

# ===== rrule helper =====

function add_rrule(A, B)
    Y = A + B

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔA, ΔB = add_rrule_impl(A, B, Y, ΔY)
            return NoTangent(), ΔA, ΔB
        end
    end

    return Y, pullback ∘ unthunk
end

function sub_rrule(A, B)
    Y = A - B

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔA, ΔB = sub_rrule_impl(A, B, Y, ΔY)
            return NoTangent(), ΔA, ΔB
        end
    end

    return Y, pullback ∘ unthunk
end

# ===== rrule =====

for T in CHORDAL_TYPES
    # T + T
    @eval function ChainRulesCore.rrule(::typeof(+), A::$T, B::$T)
        return add_rrule(A, B)
    end
    @eval function ChainRulesCore.rrule(::typeof(-), A::$T, B::$T)
        return sub_rrule(A, B)
    end
    # T + Diagonal, Diagonal + T
    for S in (Diagonal, UniformScaling)
        @eval function ChainRulesCore.rrule(::typeof(+), A::$T, B::$S)
            return add_rrule(A, B)
        end
        @eval function ChainRulesCore.rrule(::typeof(+), A::$S, B::$T)
            return add_rrule(A, B)
        end
        @eval function ChainRulesCore.rrule(::typeof(-), A::$T, B::$S)
            return sub_rrule(A, B)
        end
        @eval function ChainRulesCore.rrule(::typeof(-), A::$S, B::$T)
            return sub_rrule(A, B)
        end
    end
end
