module Semiring

using Base: oneto, promote_eltype, BitInteger, unsafe_convert, unsafe_rational, IEEEFloat
using Base.BinaryPlatforms.CPUID: test_cpu_feature, JL_X86_avx512f, JL_X86_avx2, JL_X86_fma
using Base.Checked: mul_with_overflow
using Base.GC: @preserve
using Base.Threads: @spawn, @threads, nthreads
using Graphs: AbstractGraph, neighbors, vertices
using LinearAlgebra: Factorization, Transpose, AdjointFactorization, TransposeFactorization, lu!, mul!, ldiv!, rdiv!, lmul!, rmul!, tril!
import LinearAlgebra
using SIMD: Vec, vload, vstore, vifelse, shufflevector
using SparseArrays: SparseMatrixCSC, getcolptr, nonzeros, nzrange, permute, rowvals, sparse

using ...Multifrontal: ChordalSymbolic, ChordalTriangular, DivisionWorkspace,
    FactorizationWorkspace, FChordalTriangular, FArray, FMatrix, FVector, Permutation, THRESHOLD,
    copy_scatter!, copygatherrec!, copyrec!, copyscattertri!, copytri!, eltypedegree, four, isforward, ispositive, ncl, nfr, pointers, two,
    symbolic, symmetric, unwrap, DEFAULT_ELIMINATION_ALGORITHM, PermutationOrAlgorithm

export AbstractSemiring, DualQuantale, NegativeQuantale, Lattice
export PlusProd, MinPlus, MaxPlus, MinProd, MaxProd, MinMax, MaxMin
export MinPlusLaw, MaxPlusLaw, MinProdLaw, MaxProdLaw, LawvereQuantale
export AndOr, OrAnd
export splus, sprod, sstar, szero, sone, smuladd, sdot, saxpy!, sger!
export slte, sgte, TropicalSemiring
export Pred, Succ, UnsafePred, UnsafeSucc
export Relative

abstract type AbstractSemiring end

const N_OR_R = Union{Val{:N}, Val{:R}}
const N_OR_T = Union{Val{:N}, Val{:T}}
const N_OR_C = Union{Val{:N}, Val{:C}}
const T_OR_C = Union{Val{:T}, Val{:C}}
const R_OR_C = Union{Val{:R}, Val{:C}}

include("semiring/semiring.jl")

include("dense/dense.jl")
include("sparse/sparse.jl")
include("utils.jl")
include("abstract_slu.jl")
include("chordal_slu.jl")
include("chordal/chordal.jl")
include("dense_slu.jl")
include("bellman.jl")
include("dijkstra.jl")

end
