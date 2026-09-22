module Semiring

using Base: oneto, promote_eltype, BitInteger, unsafe_convert, unsafe_rational
using Base.Checked: mul_with_overflow
using Base.GC: @preserve
using Base.Threads: @spawn, nthreads
using Graphs: AbstractGraph, neighbors, vertices
using LinearAlgebra: Factorization, Transpose, AdjointFactorization, TransposeFactorization, lu!, mul!, ldiv!, rdiv!, lmul!, rmul!
import LinearAlgebra
using SIMD: Vec, vload, vstore, vifelse, shufflevector
using SparseArrays: SparseMatrixCSC, permute

using ...Multifrontal: ChordalSymbolic, ChordalTriangular, DivisionWorkspace,
    FactorizationWorkspace, FChordalTriangular, FArray, FMatrix, FVector, Permutation, THRESHOLD,
    copy_scatter!, copygatherrec!, copyrec!, copytri!, eltypedegree, isforward, ispositive, symbolic,
    symmetric, unwrap

export AbstractSemiring, DualQuantale, NegativeQuantale, Lattice
export PlusProd, MinPlus, MaxPlus, MinProd, MaxProd, MinMax, MaxMin
export MinPlusLaw, MaxPlusLaw, MinProdLaw, MaxProdLaw, LawvereQuantale
export AndOr, OrAnd
export splus, sprod, sstar, szero, sone, smuladd
export slte, sgte, TropicalSemiring
export Pred, Succ, UnsafePred, UnsafeSucc
export RelProd

abstract type AbstractSemiring end

const N_OR_R = Union{Val{:N}, Val{:R}}
const T_OR_C = Union{Val{:T}, Val{:C}}

include("semiring/semiring.jl")

include("dense/dense.jl")
include("utils.jl")
include("abstract_slu.jl")
include("chordal_slu.jl")
include("sparse/sparse.jl")
include("dense_slu.jl")

end
