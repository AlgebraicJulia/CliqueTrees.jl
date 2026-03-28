for T1 in CHORDAL_TYPES
    for T2 in CHORDAL_TYPES
        @eval function ChainRulesCore.frule(_::Tuple, ::typeof(*), ::$T1, ::$T2)
            error()
        end

        @eval function ChainRulesCore.rrule(::typeof(*), ::$T1, ::$T2)
            error()
        end

        @eval function ChainRulesCore.rrule(::typeof(*), ::$T1{<:Any, <:RealOrComplex}, ::$T2{<:Any, <:RealOrComplex})
            error()
        end
    end
end

# Permutation × CHORDAL_TYPES disambiguations
for T in CHORDAL_TYPES
    # P * L
    @eval function ChainRulesCore.frule(_::Tuple, ::typeof(*), ::Permutation, ::$T)
        error()
    end

    @eval function ChainRulesCore.rrule(::typeof(*), ::Permutation, ::$T)
        error()
    end

    @eval function ChainRulesCore.rrule(::typeof(*), ::Permutation, ::$T{<:Any, <:RealOrComplex})
        error()
    end

    # L * P
    @eval function ChainRulesCore.frule(_::Tuple, ::typeof(*), ::$T, ::Permutation)
        error()
    end

    @eval function ChainRulesCore.rrule(::typeof(*), ::$T, ::Permutation)
        error()
    end

    @eval function ChainRulesCore.rrule(::typeof(*), ::$T{<:Any, <:RealOrComplex}, ::Permutation)
        error()
    end
end

# P * P
function ChainRulesCore.frule(_::Tuple, ::typeof(*), ::Permutation, ::Permutation)
    error()
end

function ChainRulesCore.rrule(::typeof(*), ::Permutation, ::Permutation)
    error()
end
