using Test
using LinearAlgebra
using MatrixMarket
using SparseArrays
using Random
using SuiteSparseMatrixCollection

using CliqueTrees
using CliqueTrees.Multifrontal: ChordalTriangular, DChordalCholesky, triangular, HermTri, SymTri, Permutation, ndz, selupd!
using CliqueTrees.Multifrontal.Differential: cholesky, selinv, uncholesky, soft, flat, unflattri, unflatsym

using ADTypes: AutoZygote, AutoMooncake
using DifferentiationInterface
using DifferentiationInterfaceTest
using Mooncake
using Zygote

if !@isdefined(SSMC)
    const SSMC = ssmc_db()
end

if !@isdefined(readmatrix)
    function readmatrix(name::String)
        path = joinpath(fetch_ssmc(SSMC[SSMC.name .== name, :]; format = "MM")[1], "$(name).mtx")
        return mmread(path)
    end
end

@testset "differentiation (backward)" begin
    # Set up test matrices
    M = SparseMatrixCSC{Float64}(readmatrix("685_bus"))

    F = DChordalCholesky{:L}(M)
    A = Hermitian(triangular(F), :L) + 2I
    LA = cholesky(A)
    CA = selinv(LA) + 2I
    n = size(LA, 1)

    uplo = parent(LA).uplo
    S = parent(LA).S
    P = F.P  # Permutation from original factorization

    # Construct B by adding rank-1 updates to A at clique positions
    cliques = [
        [183, 155, 156, 186, 157, 158],
        [185, 177, 176, 180, 182],
        [185, 179, 180, 176, 182],
        [185, 179, 189, 173, 187],
        [358, 359, 355, 357, 356],
        [276, 272, 273, 274, 275],
        [616, 615, 621, 617, 622],
        [616, 615, 621, 617, 553],
        [185, 179, 180, 173],
        [242, 245, 244, 248],
    ]

    B = copy(A)

    for c in cliques
        v = zeros(Float64, size(B, 1))
        v[c] .= 100.0
        selupd!(parent(B), v, v', 1, 1)
    end

    LB = cholesky(B)
    CB = selinv(LB) + 2I

    # Test functions - must be scalar-valued for gradient testing

    function loss_residual_prec(; x, b, pchol, qprec, uplo, S, P)
        LP = soft(unflattri(pchol, S, uplo))
        Q = unflatsym(qprec, S, uplo)
        r = b - P' * (Q * (P * x))
        z = P' * (LP' \ (LP \ (P * r)))
        return dot(z, z)
    end

    function loss_residual_chol(; x, b, pchol, qchol, uplo, S, P)
        LP = soft(unflattri(pchol, S, uplo))
        LQ = soft(unflattri(qchol, S, uplo))
        r = b - P' * (LQ * (LQ' * (P * x)))
        z = P' * (LP' \ (LP \ (P * r)))
        return dot(z, z)
    end

    function loss_entropy_prec(; qmean, qprec, pmean, pprec, uplo, S, P)
        Aprec = unflatsym(pprec, S, uplo)
        Bprec = unflatsym(qprec, S, uplo)
        LA = cholesky(Aprec)
        LB = cholesky(Bprec)
        return 0.5 * (dot(Aprec, selinv(Bprec, LB)) + dot(pmean - qmean, Aprec, pmean - qmean)) + logdet(Aprec, LA) - logdet(Bprec, LB)
    end

    function loss_entropy_chol(; qmean, qchol, pmean, pchol, uplo, S, P)
        LA = soft(unflattri(pchol, S, uplo))
        LB = soft(unflattri(qchol, S, uplo))
        Aprec = uncholesky(LA)
        return 0.5 * (dot(Aprec, selinv(LB)) + dot(pmean - qmean, Aprec, pmean - qmean)) + logdet(LA) - logdet(LB)
    end

    # Compute reference gradient using central differences (O(eps²) accurate)
    function finite_diff_gradient(f, x; eps=1e-5)
        grad = similar(x)
        for i in eachindex(x)
            x_plus = copy(x)
            x_minus = copy(x)
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
        end
        return grad
    end

    # Add scenarios for all variables in a loss function
    # NOTE: We provide explicit prep_args with the actual input value because:
    # - DifferentiationInterfaceTest defaults prep_args to zero(x)
    # - When preparing Mooncake with zeros, the loss functions fail because
    #   unflatsym creates matrices that are not positive definite (zero diagonal)
    # - By providing copy(val) as prep_args, we ensure preparation uses valid inputs
    function add_all_scenarios!(scenarios, name, f, vars::NamedTuple, fixed::NamedTuple)
        for (varname, val) in pairs(vars)
            g = v -> f(; merge(vars, NamedTuple{(varname,)}((v,)))..., fixed...)
            push!(scenarios, Scenario{:gradient, :out}(
                g, copy(val);
                res1 = finite_diff_gradient(g, copy(val)),
                prep_args = (; x = copy(val), contexts = ()),  # Use actual input for preparation, not zeros
                name = "$name ($varname)"
            ))
        end
    end

    # Create scenarios with reference gradients computed via finite differences
    function make_scenarios(LA, A, CA, LB, B, CB, uplo, S, P)
        n = size(LA, 1)

        # Get flat representations for A (prior)
        pchol = flat(LA)
        pprec = flat(A)
        pscov = flat(CA)

        # Get flat representations for B (variational)
        qchol = flat(LB)
        qprec = flat(B)
        qscov = flat(CB)

        # Random vectors
        x = randn(Float64, n)
        b = randn(Float64, n)
        pmean = randn(Float64, n)
        qmean = randn(Float64, n)

        # Fixed parameters (not differentiated)
        fixed = (; uplo, S, P)

        scenarios = Scenario[]

        add_all_scenarios!(scenarios, "residual_prec", loss_residual_prec,
            (; x, b, pchol, qprec), fixed)

        add_all_scenarios!(scenarios, "residual_chol", loss_residual_chol,
            (; x, b, pchol, qchol), fixed)

        add_all_scenarios!(scenarios, "entropy_prec", loss_entropy_prec,
            (; qmean, qprec, pmean, pprec), fixed)

        add_all_scenarios!(scenarios, "entropy_chol", loss_entropy_chol,
            (; qmean, qchol, pmean, pchol), fixed)

        return scenarios
    end

    scenarios = make_scenarios(LA, A, CA, LB, B, CB, uplo, S, P)

    @testset "Zygote" begin
        test_differentiation(
            AutoZygote(),
            scenarios;
            correctness = true,
            type_stability = :none,
            detailed = true,
            rtol = 1e-3,
        )
    end

    @testset "Mooncake" begin
        test_differentiation(
            AutoMooncake(; config=nothing),
            scenarios;
            correctness = true,
            type_stability = :none,
            detailed = true,
            rtol = 1e-3,
        )
    end

end
