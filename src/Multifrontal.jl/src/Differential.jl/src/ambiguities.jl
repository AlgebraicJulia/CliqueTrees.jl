const ChordalTypes = (ChordalTriangular{:N}, AdjTri{:N}, TransTri{:N}, HermTri, SymTri)

for T1 in ChordalTypes
    for T2 in ChordalTypes
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
