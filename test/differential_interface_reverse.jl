using Test
using LinearAlgebra
using MatrixMarket
using SparseArrays
using Random
using SuiteSparseMatrixCollection

using CliqueTrees
using CliqueTrees.Multifrontal: ChordalTriangular, DChordalCholesky, triangular, HermTri, SymTri, ndz, selupd!
using CliqueTrees.Multifrontal.Differential: cholesky, selinv, complete, uncholesky, soft, flat, unflattri, unflatsym

using ADTypes: AutoZygote, AutoEnzyme, AutoMooncake
using DifferentiationInterface
using DifferentiationInterfaceTest
using Enzyme
using Mooncake
using Zygote

const SSMC = ssmc_db()

function readmatrix(name::String)
    path = joinpath(fetch_ssmc(SSMC[SSMC.name .== name, :]; format = "MM")[1], "$(name).mtx")
    return mmread(path)
end

@testset "differentiation (backward)" begin
    # Set up test matrices
    A = SparseMatrixCSC{Float64}(readmatrix("685_bus"))

    F = DChordalCholesky{:L}(A)
    P = Hermitian(triangular(F), :L) + 2I
    LP = cholesky(P)
    CP = selinv(LP) + 2I
    n = size(LP, 1)

    uplo, S, _ = flat(LP)

    # Construct Q by adding rank-1 updates to P at clique positions
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

    Q = copy(P)

    for c in cliques
        v = zeros(Float64, size(Q, 1))
        v[c] .= 100.0
        selupd!(parent(Q), v, v', 1, 1)
    end

    LQ = cholesky(Q)
    CQ = selinv(LQ) + 2I

    # Test functions - must be scalar-valued for gradient testing

    function loss_residual_prec(; x, b, pchol, qprec, uplo, S)
        LP = soft(unflattri(uplo, S, pchol))
        Q = unflatsym(uplo, S, qprec)
        r = b - Q * x
        z = adjoint(LP) \ (LP \ r)
        return dot(z, z)
    end

    function loss_residual_chol(; x, b, pchol, qchol, uplo, S)
        LP = soft(unflattri(uplo, S, pchol))
        LQ = soft(unflattri(uplo, S, qchol))
        r = b - LQ * (adjoint(LQ) * x)
        z = adjoint(LP) \ (LP \ r)
        return dot(z, z)
    end

    function loss_entropy_prec(; qmean, qprec, pmean, pprec, uplo, S)
        P = unflatsym(uplo, S, pprec)
        Q = unflatsym(uplo, S, qprec)
        LP = cholesky(P)
        LQ = cholesky(Q)
        return 0.5 * (dot(P, selinv(LQ)) + dot(pmean - qmean, P, pmean - qmean)) + logdet(LP) - logdet(LQ)
    end

    function loss_entropy_chol(; qmean, qchol, pmean, pchol, uplo, S)
        LP = soft(unflattri(uplo, S, pchol))
        LQ = soft(unflattri(uplo, S, qchol))
        P = uncholesky(LP)
        return 0.5 * (dot(P, selinv(LQ)) + dot(pmean - qmean, P, pmean - qmean)) + logdet(LP) - logdet(LQ)
    end

    function loss_entropy_dual(; qmean, qscov, pmean, pscov, uplo, S)
        LP = complete(unflatsym(uplo, S, pscov))
        LQ = complete(unflatsym(uplo, S, qscov))
        P = uncholesky(LP)
        Q = unflatsym(uplo, S, qscov)
        return 0.5 * (dot(P, Q) + dot(pmean - qmean, P, pmean - qmean)) + logdet(LP) - logdet(LQ)
    end

    # Test function using scalar operations: tr, *, /, \, +, + xI
    function loss_scalar_ops(; alpha, beta, pprec, qprec, uplo, S)
        a = alpha[1]
        b = beta[1]
        P = unflatsym(uplo, S, pprec)
        Q = unflatsym(uplo, S, qprec)

        # scalar multiplication
        aP = a * P
        Qb = Q * b

        # scalar division
        Pd = P / a
        dQ = b \ Q

        # matrix addition
        PQ = P + Q

        # identity scaling
        Pi = P + a * I
        iQ = b * I + Q

        # trace
        return tr(aP) + tr(Qb) + tr(Pd) + tr(dQ) + tr(PQ) + tr(Pi) + tr(iQ)
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
    function make_scenarios(LP, P, CP, LQ, Q, CQ, uplo, S)
        n = size(LP, 1)

        # Get flat representations for P (prior)
        _, _, pchol = flat(LP)
        _, _, pprec = flat(P)
        _, _, pscov = flat(CP)

        # Get flat representations for Q (variational)
        _, _, qchol = flat(LQ)
        _, _, qprec = flat(Q)
        _, _, qscov = flat(CQ)

        # Random vectors
        x = randn(Float64, n)
        b = randn(Float64, n)
        pmean = randn(Float64, n)
        qmean = randn(Float64, n)

        # Fixed parameters (not differentiated)
        fixed = (; uplo, S)

        scenarios = Scenario[]

        add_all_scenarios!(scenarios, "residual_prec", loss_residual_prec,
            (; x, b, pchol, qprec), fixed)

        add_all_scenarios!(scenarios, "residual_chol", loss_residual_chol,
            (; x, b, pchol, qchol), fixed)

        add_all_scenarios!(scenarios, "entropy_prec", loss_entropy_prec,
            (; qmean, qprec, pmean, pprec), fixed)

        add_all_scenarios!(scenarios, "entropy_chol", loss_entropy_chol,
            (; qmean, qchol, pmean, pchol), fixed)

        add_all_scenarios!(scenarios, "entropy_dual", loss_entropy_dual,
            (; qmean, qscov, pmean, pscov), fixed)

        # Scalars for scalar ops test (as single-element arrays for finite_diff_gradient)
        alpha = [2.5]
        beta = [1.5]

        add_all_scenarios!(scenarios, "scalar_ops", loss_scalar_ops,
            (; alpha, beta, pprec, qprec), fixed)

        return scenarios
    end

    scenarios = make_scenarios(LP, P, CP, LQ, Q, CQ, uplo, S)

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

    @testset "Enzyme" begin
        test_differentiation(
            AutoEnzyme(; function_annotation=Enzyme.Duplicated),
            scenarios;
            correctness = true,
            type_stability = :none,
            detailed = true,
            rtol = 1e-3,
        )
    end
end
