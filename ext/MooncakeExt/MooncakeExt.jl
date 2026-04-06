module MooncakeExt

using CliqueTrees.Multifrontal: ChordalCholesky, HermOrSymSparse, HermSparse, SymSparse, project
using CliqueTrees.Multifrontal.Differential: selinv, selaxpy!, symdot
using CliqueTrees.Multifrontal.Differential: logdet_frule_impl, logdet_rrule_impl!, logdet_rrule_frule_impl!, logdet_rrule_rrule_impl!
using CliqueTrees.Multifrontal.Differential: selinv_frule_impl, selinv_rrule_impl!
using Base: AbstractVecOrMat
using LinearAlgebra: Hermitian, Symmetric, HermOrSym, logdet
using SparseArrays: SparseMatrixCSC, nonzeros
using Mooncake
using Mooncake: @is_primitive, CoDual, Dual, FData, MinimalCtx, NoFData, NoRData, NoTangent, Tangent, primal, tangent, build_tangent, fdata
using Mooncake: FriendlyTangentCache, AsRaw

function Mooncake.tangent_type(::Type{<:ChordalCholesky})
    return NoTangent
end

function Mooncake.friendly_tangent_cache(::ChordalCholesky)
    return FriendlyTangentCache{AsRaw}(nothing)
end

include("utils.jl")
include("logdet.jl")
include("selinv.jl")

end
