module Semiring

using Base: oneto, promote_eltype, BitInteger, unsafe_rational
using Base.Checked: mul_with_overflow
using Base.GC: @preserve
using Base.Threads: @spawn, nthreads
using Graphs: AbstractGraph, neighbors, vertices
using SIMD: Vec, vload, vstore, shufflevector
using SparseArrays: SparseMatrixCSC

using ...Multifrontal: ChordalTriangular, DivisionWorkspace, FVector, THRESHOLD,
    copy_scatter!, copygatherrec!, copyrec!, eltypedegree, isforward, ispositive

export AbstractSemiring, AbstractQuantale, IntegralQuantale, AbstractLattice, DualLattice
export PlusProd, MinPlus, MaxPlus, MinProd, MaxProd, MinMax, MaxMin
export GCDProd, LCMProd, GCDLCM, LCMGCD, AndOr, OrAnd, RelPlus, RelProd
export LAndPar, LOrTens
export splus, sprod, sstar, szero, sone, smuladd, slu!, sldiv!, srdiv!
export slte, sgte, TropicalSemiring
export Pred, Succ, Jet, Best, AffGCDProd

abstract type AbstractSemiring end

abstract type AbstractQuantale <: AbstractSemiring end

abstract type IntegralQuantale <: AbstractQuantale end

abstract type AbstractLattice <: IntegralQuantale end

include("semiring/semiring.jl")

include("blas/sgemx.jl")
include("blas/strsx.jl")
include("blas/slu.jl")
include("utils.jl")
include("slu.jl")
include("divide.jl")

end
