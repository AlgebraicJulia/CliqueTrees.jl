module Differential

using Base: promote_eltype
using ChainRulesCore
using ChainRulesCore: NoTangent, ZeroTangent, unthunk, @thunk
using LinearAlgebra
using LinearAlgebra: dot, tr, RealOrComplex, UniformScaling, axpy!

using ...Multifrontal
using ...Multifrontal: HermOrSymTri, MaybeHermOrSymTri, ChordalCholesky, ChordalTriangular, ChordalSymbolic
using ...Multifrontal: AdjTri, TransTri, AdjOrTransTri, HermTri, SymTri
using ...Multifrontal: cholesky!, complete!, dfcholesky!, fisher!, rmul!, selinv!, selupd!, uncholesky!
using ...Multifrontal: fronts, diagblock, diagind, ndz, nlz, nnz, triangular, checksymbolic

export selinv, complete, uncholesky, soft, flat, unflattri, unflatsym

include("utils.jl")
include("selinv.jl")
include("complete.jl")
include("cholesky.jl")
include("uncholesky.jl")
include("logdet.jl")
include("ldiv.jl")
include("ldivadj.jl")
include("rdiv.jl")
include("rdivadj.jl")
include("lmul.jl")
include("lmuladj.jl")
include("rmul.jl")
include("rmuladj.jl")
include("dot.jl")
include("quad.jl")
include("soft.jl")
include("unflat.jl")
include("flat.jl")
include("lmulsym.jl")
include("rmulsym.jl")
include("adjoint.jl")
include("trace.jl")
include("lmulnum.jl")
include("rmulnum.jl")
include("ldivnum.jl")
include("rdivnum.jl")
include("add.jl")
include("laddnum.jl")
include("raddnum.jl")
include("ambiguities.jl")

end
