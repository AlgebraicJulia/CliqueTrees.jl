module EnzymeExt

using CliqueTrees.Multifrontal: ChordalTriangular, ChordalSymbolic, HermOrSymTri,
    HermTri, SymTri, AdjTri, TransTri, Permutation, symbolic
using CliqueTrees.Multifrontal.Differential: selinv, complete, uncholesky, soft,
    flat, unflattri, unflatsym,
    cholesky_rrule_impl, cholesky_frule_impl,
    uncholesky_rrule_impl, uncholesky_frule_impl,
    selinv_rrule_impl, selinv_frule_impl,
    complete_rrule_impl, complete_frule_impl,
    dot_rrule_impl, dot_frule_impl,
    ldiv_rrule_impl, ldiv_frule_impl,
    tr_rrule_impl, tr_frule_impl,
    diag_rrule_impl, diag_frule_impl,
    logdet_rrule_impl, logdet_frule_impl,
    soft_rrule_impl, soft_frule_impl,
    rdiv_frule_impl, rdiv_rrule_impl,
    adjoint_frule_impl, adjoint_rrule_impl,
    transpose_frule_impl, transpose_rrule_impl,
    add_frule_impl,
    flat_frule_impl, flat_rrule_impl,
    unflattri_frule_impl, unflattri_rrule_impl,
    unflatsym_frule_impl, unflatsym_rrule_impl,
    mul_frule_impl, mul_rrule_impl,
    ldiv_frule_impl, rdiv_frule_impl
using ChainRulesCore: unthunk, ZeroTangent
using LinearAlgebra: dot, logdet, tr, diag, cholesky, adjoint, transpose, Hermitian, Symmetric, UniformScaling, I, axpy!

using Enzyme: Const, Active, Duplicated, Annotation
using Enzyme.EnzymeRules: EnzymeRules,
    RevConfigWidth, FwdConfigWidth, AugmentedReturn, needs_shadow, needs_primal, overwritten

# ChordalSymbolic is non-differentiable (structural/symbolic type)
function EnzymeRules.inactive_type(::Type{<:ChordalSymbolic})
    return true
end

# Permutation is non-differentiable (structural type)
function EnzymeRules.inactive_type(::Type{<:Permutation})
    return true
end

include("ldiv.jl")
include("lmul.jl")
include("ldivadj.jl")
include("lmuladj.jl")
include("lmulsym.jl")
include("rdiv.jl")
include("rmul.jl")
include("rdivadj.jl")
include("rmuladj.jl")
include("rmulsym.jl")
include("quad.jl")
include("logdet.jl")
include("cholesky.jl")
include("selinv.jl")
include("dot.jl")
include("uncholesky.jl")
include("soft.jl")
include("complete.jl")
include("adjoint.jl")
include("unflat.jl")
include("flat.jl")
include("trace.jl")
include("diag.jl")
include("rmulnum.jl")
include("rdivnum.jl")
include("add.jl")
include("raddnum.jl")

end
