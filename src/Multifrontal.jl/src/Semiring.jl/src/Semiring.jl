module Semiring

using Base: oneto, promote_eltype, BitInteger, unsafe_convert, unsafe_rational
using Base.Checked: mul_with_overflow
using Base.GC: @preserve
using Base.Threads: @spawn, nthreads
using Graphs: AbstractGraph, neighbors, vertices
using LinearAlgebra: Factorization, Transpose, AdjointFactorization, TransposeFactorization, mul!
import LinearAlgebra: lu!, ldiv!, rdiv!
using SIMD: Vec, vload, vstore, vifelse, shufflevector
using SparseArrays: SparseMatrixCSC, permute

using ...Multifrontal: ChordalSymbolic, ChordalTriangular, DivisionWorkspace,
    FactorizationWorkspace, FChordalTriangular, FArray, FMatrix, FVector, Permutation, THRESHOLD,
    copy_scatter!, copygatherrec!, copyrec!, eltypedegree, isforward, ispositive, symbolic,
    symmetric

export AbstractSemiring, AbstractQuantale, DualQuantale, NegativeQuantale, Lattice
export PlusProd, MinPlus, MaxPlus, MinProd, MaxProd, MinMax, MaxMin
export MinPlusLaw, MaxPlusLaw, MinProdLaw, MaxProdLaw, LawvereQuantale
export AndOr, OrAnd
export splus, sprod, sstar, szero, sone, smuladd, sldiv!, srdiv!
export slte, sgte, TropicalSemiring
export Pred, Succ, UnsafePred, UnsafeSucc
export RelProd

abstract type AbstractSemiring end

abstract type AbstractQuantale <: AbstractSemiring end

include("semiring/semiring.jl")

include("blas/blas.jl")
include("utils.jl")
include("semiring_lu.jl")
include("slu.jl")
include("divide.jl")

end
