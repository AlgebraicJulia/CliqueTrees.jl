#-----------------------#
# dissection algorithms #
#-----------------------#

package(::Type{<:METISND}) = "Metis"
package(::Type{<:KaHyParND}) = "KaHyPar"

#------------------------#
# elimination algorithms #
#------------------------#

package(::Type{<:METIS}) = "Metis"
package(::Type{<:AMD}) = "AMD"
package(::Type{<:SymAMD}) = "AMD"
package(::Type{<:Spectral}) = "Laplacians"
package(::Type{<:FlowCutter}) = "FlowCutterPACE17_jll"
package(::Type{<:BT}) = "TreeWidthSolver"

function package(::Type{<:ND{<:Any, <:Any, D}}) where D
    return package(D)
end
