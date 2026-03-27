using Test
using LinearAlgebra
using MatrixMarket
using SparseArrays
using Random
using SuiteSparseMatrixCollection

using CliqueTrees
using CliqueTrees.Multifrontal: ChordalTriangular, ChordalCholesky, triangular, HermTri, SymTri, ndz, selupd!
using CliqueTrees.Multifrontal.Differential: cholesky, selinv, uncholesky, softmax, flat, unflattri, unflatsym

using ADTypes: AutoForwardDiff
using DifferentiationInterface
using DifferentiationInterfaceTest
using ForwardDiff

if !@isdefined(SSMC)
    const SSMC = ssmc_db()
end

if !@isdefined(readmatrix)
    function readmatrix(name::String)
        path = joinpath(fetch_ssmc(SSMC[SSMC.name .== name, :]; format = "MM")[1], "$(name).mtx")
        return mmread(path)
    end
end

@testset "differentiation (forward)" begin
    # Set up test matrices
    A = SparseMatrixCSC{Float64}(readmatrix("685_bus"))

    F = ChordalCholesky{:L}(A)
    P = Hermitian(triangular(F), :L) + 2I
    LP = cholesky(P)
    n = size(LP, 1)

    uplo = parent(LP).uplo
    S = parent(LP).S
    pprec = flat(P)

    # Build a second precision matrix Q
    Q = copy(P)

    cliques = [
        [183, 155, 156, 186, 157, 158],
        [185, 177, 176, 180, 182],
        [358, 359, 355, 357, 356],
        [276, 272, 273, 274, 275],
    ]

    for c in cliques
        v = zeros(Float64, n)
        v[c] .= 50.0
        selupd!(parent(Q), v, v', 1, 1)
    end
    qprec = flat(Q)

    # Random vectors for quadratic forms
    x = randn(Float64, n)
    y = randn(Float64, n)

    # ============================================================================
    # Forward-mode test functions: scalar -> scalar
    # These are realistic sensitivity analyses
    # ============================================================================

    # Regularization sensitivity: d/dλ logdet(P + λI)
    # Useful for: tuning regularization, computing degrees of freedom
    function logdet_regularization(λ::Real; pprec, uplo, S)
        P = unflatsym(pprec, S, uplo)
        Pλ = P + λ * I
        Lλ = cholesky(Pλ)
        return logdet(Lλ)
    end

    # Entropy along regularization path: H(P + λI) = (n/2)(1 + log(2π)) + (1/2)logdet(Σ)
    # where Σ = (P + λI)^{-1}
    function entropy_regularization(λ::Real; pprec, uplo, S)
        n = size(S, 1)
        P = unflatsym(pprec, S, uplo)
        Pλ = P + λ * I
        Lλ = cholesky(Pλ)
        # H = (n/2)(1 + log(2π)) - (1/2)logdet(P)
        return 0.5 * n * (1 + log(2π)) - 0.5 * logdet(Lλ)
    end

    # Interpolation: KL(P_t || Q) where P_t = (1-t)P + tQ
    # KL = (1/2)[tr(Q^{-1} P_t) - n + logdet(Q) - logdet(P_t)]
    function kl_interpolation(t::Real; pprec, qprec, uplo, S)
        n = size(S, 1)
        P = unflatsym(pprec, S, uplo)
        Q = unflatsym(qprec, S, uplo)
        Pt = (1 - t) * P + t * Q
        LPt = cholesky(Pt)
        LQ = cholesky(Q)
        Qinv = selinv(LQ)
        return 0.5 * (dot(Qinv, Pt) - n + logdet(LQ) - logdet(LPt))
    end

    # Mahalanobis distance sensitivity: d/dt x'(P + tI)^{-1}x
    # Useful for: outlier detection thresholds, confidence regions
    function mahalanobis_regularization(λ::Real; x, pprec, uplo, S)
        P = unflatsym(pprec, S, uplo)
        Pλ = P + λ * I
        Lλ = cholesky(Pλ)
        Σλ = selinv(Lλ)
        return dot(x, Σλ, x)
    end

    # Quadratic form interpolation: x'[(1-t)P + tQ]y
    function quadform_interpolation(t::Real; x, y, pprec, qprec, uplo, S)
        P = unflatsym(pprec, S, uplo)
        Q = unflatsym(qprec, S, uplo)
        Pt = (1 - t) * P + t * Q
        return dot(x, Pt, y)
    end

    # Trace of covariance along regularization: tr((P + λI)^{-1})
    # This equals sum of marginal variances
    function marginal_variance_sum(λ::Real; pprec, uplo, S)
        P = unflatsym(pprec, S, uplo)
        Pλ = P + λ * I
        Lλ = cholesky(Pλ)
        Σλ = selinv(Lλ)
        return tr(Σλ)
    end

    # ============================================================================
    # Compute reference derivative using finite differences
    # ============================================================================

    function finite_diff_derivative(f, t; eps=1e-7)
        return (f(t + eps) - f(t - eps)) / (2 * eps)
    end

    # ============================================================================
    # Create forward-mode scenarios (derivative tests)
    # ============================================================================

    # NOTE: We provide explicit prep_args with the actual input value because:
    # - DifferentiationInterfaceTest defaults prep_args to zero(x)
    # - When preparing with zeros, the loss functions fail because
    #   cholesky fails on non-positive-definite matrices (zero diagonal after P + 0*I)
    # - By providing the actual test value as prep_args, we ensure preparation uses valid inputs
    function make_forward_scenarios(pprec, qprec, x, y, uplo, S)
        scenarios = Scenario[]

        λ0 = 0.5  # regularization test point
        t0 = 0.3  # interpolation test point

        # logdet regularization
        f1 = λ -> logdet_regularization(λ; pprec, uplo, S)
        push!(scenarios, Scenario{:derivative, :out}(
            f1, λ0;
            res1 = finite_diff_derivative(f1, λ0),
            prep_args = (; x = λ0, contexts = ()),
            name = "logdet_regularization"
        ))

        # entropy regularization
        f2 = λ -> entropy_regularization(λ; pprec, uplo, S)
        push!(scenarios, Scenario{:derivative, :out}(
            f2, λ0;
            res1 = finite_diff_derivative(f2, λ0),
            prep_args = (; x = λ0, contexts = ()),
            name = "entropy_regularization"
        ))

        # KL interpolation
        f3 = t -> kl_interpolation(t; pprec, qprec, uplo, S)
        push!(scenarios, Scenario{:derivative, :out}(
            f3, t0;
            res1 = finite_diff_derivative(f3, t0),
            prep_args = (; x = t0, contexts = ()),
            name = "kl_interpolation"
        ))

        # Mahalanobis regularization
        f4 = λ -> mahalanobis_regularization(λ; x, pprec, uplo, S)
        push!(scenarios, Scenario{:derivative, :out}(
            f4, λ0;
            res1 = finite_diff_derivative(f4, λ0),
            prep_args = (; x = λ0, contexts = ()),
            name = "mahalanobis_regularization"
        ))

        # Quadform interpolation
        f5 = t -> quadform_interpolation(t; x, y, pprec, qprec, uplo, S)
        push!(scenarios, Scenario{:derivative, :out}(
            f5, t0;
            res1 = finite_diff_derivative(f5, t0),
            prep_args = (; x = t0, contexts = ()),
            name = "quadform_interpolation"
        ))

        # Marginal variance sum
        f6 = λ -> marginal_variance_sum(λ; pprec, uplo, S)
        push!(scenarios, Scenario{:derivative, :out}(
            f6, λ0;
            res1 = finite_diff_derivative(f6, λ0),
            prep_args = (; x = λ0, contexts = ()),
            name = "marginal_variance_sum"
        ))

        return scenarios
    end

    scenarios = make_forward_scenarios(pprec, qprec, x, y, uplo, S)

    @testset "ForwardDiff" begin
        test_differentiation(
            AutoForwardDiff(),
            scenarios;
            correctness = true,
            type_stability = :none,
            detailed = true,
            rtol = 1e-3,
        )
    end

end
