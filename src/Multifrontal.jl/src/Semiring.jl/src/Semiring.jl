module Semiring

using Base: oneto, promote_eltype, BitInteger, unsafe_convert, unsafe_rational
using Base.Checked: mul_with_overflow
using Base.GC: @preserve
using Base.Threads: @spawn, nthreads
using Graphs: AbstractGraph, neighbors, vertices
using LinearAlgebra: Factorization, Transpose, mul!
import LinearAlgebra: lu!, ldiv!, rdiv!
using SIMD: Vec, vload, vstore, vifelse, shufflevector
using SparseArrays: SparseMatrixCSC, permute

using ...Multifrontal: ChordalSymbolic, ChordalTriangular, DivisionWorkspace,
    FactorizationWorkspace, FChordalTriangular, FArray, FMatrix, FVector, Permutation, THRESHOLD,
    copy_scatter!, copygatherrec!, copyrec!, eltypedegree, isforward, ispositive, symbolic,
    symmetric

export AbstractSemiring, AbstractQuantale, IntegralQuantale, AbstractLattice, DualLattice
export PlusProd, MinPlus, MaxPlus, MinProd, MaxProd, MinMax, MaxMin
export MinPlusLaw, MaxPlusLaw, MinProdLaw, MaxProdLaw, LawvereQuantale
export AndOr, OrAnd
export splus, sprod, sstar, szero, sone, smuladd, sldiv!, srdiv!
export slte, sgte, TropicalSemiring
export Pred, Succ, UnsafePred, UnsafeSucc
export RelPlus, RelProd, RelationQuantale

abstract type AbstractSemiring end

abstract type AbstractQuantale <: AbstractSemiring end

abstract type IntegralQuantale <: AbstractQuantale end

abstract type AbstractLattice <: IntegralQuantale end

include("semiring/semiring.jl")

include("blas/sgemx.jl")
include("blas/strsx.jl")
include("blas/slu.jl")
include("utils.jl")
include("semiring_lu.jl")
include("slu.jl")
include("divide.jl")

end
