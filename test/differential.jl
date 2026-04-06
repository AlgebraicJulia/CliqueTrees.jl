# Tests for Mooncake-based AD rules
# Requires MooncakeSparse to be loaded for consistent gradient conventions
using Test
using LinearAlgebra
using SparseArrays
using SparseArrays: nonzeros
using Random: randn!

import Mooncake
using Mooncake: Config, prepare_derivative_cache, prepare_gradient_cache,
                zero_tangent, value_and_derivative!!, value_and_gradient!!

using MooncakeSparse
using MooncakeSparse: ldivwith

using CliqueTrees
using CliqueTrees.Multifrontal: ChordalCholesky, logdet, selinv

function randtangent(A::AbstractArray)
    dA = similar(A)
    randn!(dA)
    return dA
end

function randtangent(A::AbstractSparseArray)
    dA = similar(A)
    randn!(nonzeros(dA))
    return dA
end

function randtangent(A::Union{Hermitian, Symmetric})
    dA = similar(A)
    randn!(nonzeros(parent(dA)))
    return dA
end

function testadjoint(f, args...; rtol=1e-4)
    config = Config(friendly_tangents=true)

    fwd_cache = prepare_derivative_cache(f, args...; config)
    rev_cache = prepare_gradient_cache(f, args...; config)

    df = zero_tangent(f)
    tangents = map(randtangent, args)

    _, dy = value_and_derivative!!(fwd_cache, (f, df), zip(args, tangents)...)
    _, (_, gradients...) = value_and_gradient!!(rev_cache, f, args...)

    return isapprox(dy, sum(real ∘ splat(dot), zip(gradients, tangents)); rtol)
end

@testset "Mooncake AD rules" begin
    for T in (Float64,)
        @testset "$T" begin
            n = 50
            p = 0.1

            A = sprandn(T, n, n, p)
            Q = parent(Symmetric(A' * A + n * I))
            F = cholesky!(ChordalCholesky(Symmetric(Q)))

            @testset "logdet(Q, F)" begin
                @test testadjoint(Q -> logdet(Q, F), Q)
            end

            @testset "dot(selinv, selinv)" begin
                @test testadjoint(Q -> (Σ = selinv(Q, F); dot(Σ, Σ)), Q)
            end

            @testset "ldivwith(Q, F, B)" begin
                B = randn(T, n)
                @test testadjoint((Q, B) -> (X = ldivwith(Q, F, B); dot(X, X)), Q, B)
            end
        end
    end
end
