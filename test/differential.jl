using Test
using LinearAlgebra
using SparseArrays
using Random: randn!
using MatrixMarket
using SuiteSparseMatrixCollection
using CliqueTrees
using ChainRules
using ChainRulesCore: NoTangent, unthunk
using CliqueTrees.Multifrontal: ChordalTriangular, ChordalSymbolic, HermTri
using CliqueTrees.Multifrontal: chordal, symbolic, ndz, nlz
using CliqueTrees.Multifrontal.Differential
using CliqueTrees.Multifrontal.Differential: frule, rrule
using CliqueTrees.Multifrontal.Differential: cholesky, uncholesky, selinv, complete, soft
using CliqueTrees.Multifrontal.Differential: flat, unflattri, unflatsym

if !@isdefined(SSMC)
    const SSMC = ssmc_db()
end

if !@isdefined(readmatrix)
    function readmatrix(name::String)
        path = joinpath(fetch_ssmc(SSMC[SSMC.name .== name, :]; format="MM")[1], "$(name).mtx")
        return mmread(path)
    end
end

@testset "differentiation (manual)" begin
    A = SparseMatrixCSC{Float64}(readmatrix("685_bus"))

    rtol = 2e-3

    # Helper to create random tangent for ChordalTriangular
    function randtangent(L::ChordalTriangular)
        dL = zero(L)
        randn!(dL.Dval)
        randn!(dL.Lval)
        return dL
    end

    for UPLO in (:L, :U)
        perm, S = symbolic(A)
        H = chordal(A, perm, S, Val(UPLO))
        L = cholesky(H)
        n = size(L, 1)

        @testset "UPLO = $UPLO" begin
            @testset "cholesky" begin
                @testset "frule" begin
                    dH = randtangent(L)
                    L_out, dL = frule((NoTangent(), dH), cholesky, H)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    L2 = cholesky(H2)
                    dL_fd = (L2 - L_out) / 1e-7
                    @test isapprox(dL, dL_fd; rtol=rtol)
                end

                @testset "rrule" begin
                    L_out, pullback = rrule(cholesky, H)
                    ΔL = randtangent(L)
                    dH = randtangent(L)
                    _, ΔH = pullback(ΔL)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    L2 = cholesky(H2)
                    dL = (L2 - L_out) / 1e-7
                    lhs = dot(ΔL, dL)
                    rhs = dot(Hermitian(unthunk(ΔH), UPLO), Hermitian(dH, UPLO))
                    @test isapprox(lhs, rhs; rtol=rtol)
                end
            end

            @testset "uncholesky" begin
                @testset "frule" begin
                    dL = randtangent(L)
                    H_out, dH = frule((NoTangent(), dL), uncholesky, L)

                    L2 = L + 1e-7 * dL
                    H2 = uncholesky(L2)
                    dH_fd = (parent(H2) - parent(H_out)) / 1e-7
                    @test isapprox(dH, dH_fd; rtol=rtol)
                end

                @testset "rrule" begin
                    H_out, pullback = rrule(uncholesky, L)
                    ΔH = randtangent(L)
                    dL = randtangent(L)
                    _, ΔL = pullback(ΔH)

                    L2 = L + 1e-7 * dL
                    H2 = uncholesky(L2)
                    dH = (parent(H2) - parent(H_out)) / 1e-7
                    lhs = dot(Hermitian(ΔH, UPLO), Hermitian(dH, UPLO))
                    rhs = dot(unthunk(ΔL), dL)
                    @test isapprox(lhs, rhs; rtol=rtol)
                end
            end

            @testset "L \\ x" begin
                x = randn(n)

                @testset "frule" begin
                    dL = randtangent(L)
                    dx = randn(n)
                    y, dy = frule((NoTangent(), dL, dx), \, L, x)

                    L2 = L + 1e-7 * dL
                    y2 = L2 \ (x + 1e-7 * dx)
                    @test isapprox(dy, (y2 - y) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(\, L, x)
                    Δy = randn(n)
                    dx = randn(n)
                    _, ΔL, Δx = pullback(Δy)

                    y2 = L \ (x + 1e-7 * dx)
                    dy = (y2 - y) / 1e-7
                    @test isapprox(dot(Δy, dy), dot(unthunk(Δx), dx); rtol=rtol)
                end
            end

            @testset "L' \\ x" begin
                x = randn(n)

                @testset "frule" begin
                    dL = randtangent(L)
                    dx = randn(n)
                    y, dy = frule((NoTangent(), dL, dx), \, L', x)

                    L2 = L + 1e-7 * dL
                    y2 = L2' \ (x + 1e-7 * dx)
                    @test isapprox(dy, (y2 - y) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(\, L', x)
                    Δy = randn(n)
                    dx = randn(n)
                    _, ΔL, Δx = pullback(Δy)

                    y2 = L' \ (x + 1e-7 * dx)
                    dy = (y2 - y) / 1e-7
                    @test isapprox(dot(Δy, dy), dot(unthunk(Δx), dx); rtol=rtol)
                end
            end

            @testset "L * x" begin
                x = randn(n)

                @testset "frule" begin
                    dL = randtangent(L)
                    dx = randn(n)
                    y, dy = frule((NoTangent(), dL, dx), *, L, x)

                    L2 = L + 1e-7 * dL
                    y2 = L2 * (x + 1e-7 * dx)
                    @test isapprox(dy, (y2 - y) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(*, L, x)
                    Δy = randn(n)
                    dx = randn(n)
                    _, ΔL, Δx = pullback(Δy)

                    y2 = L * (x + 1e-7 * dx)
                    dy = (y2 - y) / 1e-7
                    @test isapprox(dot(Δy, dy), dot(unthunk(Δx), dx); rtol=rtol)
                end
            end

            @testset "L' * x" begin
                x = randn(n)

                @testset "frule" begin
                    dL = randtangent(L)
                    dx = randn(n)
                    y, dy = frule((NoTangent(), dL, dx), *, L', x)

                    L2 = L + 1e-7 * dL
                    y2 = L2' * (x + 1e-7 * dx)
                    @test isapprox(dy, (y2 - y) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(*, L', x)
                    Δy = randn(n)
                    dx = randn(n)
                    _, ΔL, Δx = pullback(Δy)

                    y2 = L' * (x + 1e-7 * dx)
                    dy = (y2 - y) / 1e-7
                    @test isapprox(dot(Δy, dy), dot(unthunk(Δx), dx); rtol=rtol)
                end
            end

            @testset "x / L" begin
                x = randn(1, n)

                @testset "frule" begin
                    dL = randtangent(L)
                    dx = randn(1, n)
                    y, dy = frule((NoTangent(), dx, dL), /, x, L)

                    L2 = L + 1e-7 * dL
                    y2 = (x + 1e-7 * dx) / L2
                    @test isapprox(dy, (y2 - y) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(/, x, L)
                    Δy = randn(1, n)
                    dx = randn(1, n)
                    _, Δx, ΔL = pullback(Δy)

                    y2 = (x + 1e-7 * dx) / L
                    dy = (y2 - y) / 1e-7
                    @test isapprox(dot(Δy, dy), dot(unthunk(Δx), dx); rtol=rtol)
                end
            end

            @testset "x / L'" begin
                x = randn(1, n)

                @testset "frule" begin
                    dL = randtangent(L)
                    dx = randn(1, n)
                    y, dy = frule((NoTangent(), dx, dL), /, x, L')

                    L2 = L + 1e-7 * dL
                    y2 = (x + 1e-7 * dx) / L2'
                    @test isapprox(dy, (y2 - y) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(/, x, L')
                    Δy = randn(1, n)
                    dx = randn(1, n)
                    _, Δx, ΔL = pullback(Δy)

                    y2 = (x + 1e-7 * dx) / L'
                    dy = (y2 - y) / 1e-7
                    @test isapprox(dot(Δy, dy), dot(unthunk(Δx), dx); rtol=rtol)
                end
            end

            @testset "x * L" begin
                x = randn(1, n)

                @testset "frule" begin
                    dL = randtangent(L)
                    dx = randn(1, n)
                    y, dy = frule((NoTangent(), dx, dL), *, x, L)

                    L2 = L + 1e-7 * dL
                    y2 = (x + 1e-7 * dx) * L2
                    @test isapprox(dy, (y2 - y) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(*, x, L)
                    Δy = randn(1, n)
                    dx = randn(1, n)
                    _, Δx, ΔL = pullback(Δy)

                    y2 = (x + 1e-7 * dx) * L
                    dy = (y2 - y) / 1e-7
                    @test isapprox(dot(Δy, dy), dot(unthunk(Δx), dx); rtol=rtol)
                end
            end

            @testset "x * L'" begin
                x = randn(1, n)

                @testset "frule" begin
                    dL = randtangent(L)
                    dx = randn(1, n)
                    y, dy = frule((NoTangent(), dx, dL), *, x, L')

                    L2 = L + 1e-7 * dL
                    y2 = (x + 1e-7 * dx) * L2'
                    @test isapprox(dy, (y2 - y) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(*, x, L')
                    Δy = randn(1, n)
                    dx = randn(1, n)
                    _, Δx, ΔL = pullback(Δy)

                    y2 = (x + 1e-7 * dx) * L'
                    dy = (y2 - y) / 1e-7
                    @test isapprox(dot(Δy, dy), dot(unthunk(Δx), dx); rtol=rtol)
                end
            end

            @testset "logdet" begin
                @testset "frule" begin
                    dL = randtangent(L)
                    y, dy = frule((NoTangent(), dL), logdet, L)

                    L2 = L + 1e-7 * dL
                    y2 = logdet(L2)
                    @test isapprox(dy, (y2 - y) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(logdet, L)
                    Δy = randn()
                    dL = randtangent(L)
                    _, ΔL = pullback(Δy)

                    L2 = L + 1e-7 * dL
                    dy = (logdet(L2) - y) / 1e-7
                    @test isapprox(Δy * dy, dot(unthunk(ΔL), dL); rtol=rtol)
                end
            end

            @testset "diag" begin
                @testset "frule" begin
                    dL = randtangent(L)
                    y, dy = frule((NoTangent(), dL), diag, L)

                    L2 = L + 1e-7 * dL
                    y2 = diag(L2)
                    @test isapprox(dy, (y2 - y) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(diag, L)
                    Δy = randn(size(L, 1))
                    dL = randtangent(L)
                    _, ΔL = pullback(Δy)

                    L2 = L + 1e-7 * dL
                    dy = (diag(L2) - y) / 1e-7
                    @test isapprox(dot(Δy, dy), dot(unthunk(ΔL), dL); rtol=rtol)
                end
            end

            @testset "selinv" begin
                @testset "frule" begin
                    dL = randtangent(L)
                    Y, dY = frule((NoTangent(), dL), selinv, L)

                    L2 = L + 1e-7 * dL
                    Y2 = selinv(L2)
                    dY_fd = (parent(Y2) - parent(Y)) / 1e-7
                    @test isapprox(dY, dY_fd; rtol=rtol)
                end

                @testset "rrule" begin
                    Y, pullback = rrule(selinv, L)
                    ΔY = randtangent(L)
                    dL = randtangent(L)
                    _, ΔL = pullback(ΔY)

                    L2 = L + 1e-7 * dL
                    Y2 = selinv(L2)
                    dY = (parent(Y2) - parent(Y)) / 1e-7
                    lhs = dot(Hermitian(ΔY, UPLO), Hermitian(dY, UPLO))
                    rhs = dot(unthunk(ΔL), dL)
                    @test isapprox(lhs, rhs; rtol=rtol)
                end
            end

            @testset "complete" begin
                Y = selinv(L)

                @testset "frule" begin
                    dY = randtangent(L)
                    L_out, dL = frule((NoTangent(), dY), complete, Y)

                    Y2 = Y + 1e-8 * Hermitian(dY, UPLO)
                    L2 = complete(Y2)
                    dL_fd = (L2 - L_out) / 1e-8
                    @test isapprox(dL, dL_fd; rtol=rtol)
                end

                @testset "rrule" begin
                    L_out, pullback = rrule(complete, Y)
                    ΔL = randtangent(L)
                    dY = randtangent(L)
                    _, ΔY = pullback(ΔL)

                    Y2 = Y + 1e-8 * Hermitian(dY, UPLO)
                    L2 = complete(Y2)
                    dL = (L2 - L_out) / 1e-8
                    lhs = dot(ΔL, dL)
                    rhs = dot(Hermitian(unthunk(ΔY), UPLO), Hermitian(dY, UPLO))
                    @test isapprox(lhs, rhs; rtol=rtol)
                end
            end

            @testset "soft" begin
                L_in = copy(L)
                randn!(L_in.Dval)
                randn!(L_in.Lval)

                @testset "frule" begin
                    dL = randtangent(L)
                    Y, dY = frule((NoTangent(), dL), soft, L_in)

                    L2 = L_in + 1e-7 * dL
                    Y2 = soft(L2)
                    dY_fd = (Y2 - Y) / 1e-7
                    @test isapprox(dY, dY_fd; rtol=rtol)
                end

                @testset "rrule" begin
                    Y, pullback = rrule(soft, L_in)
                    ΔY = randtangent(L)
                    dL = randtangent(L)
                    _, ΔL = pullback(ΔY)

                    L2 = L_in + 1e-7 * dL
                    Y2 = soft(L2)
                    dY = (Y2 - Y) / 1e-7
                    lhs = dot(ΔY, dY)
                    rhs = dot(unthunk(ΔL), dL)
                    @test isapprox(lhs, rhs; rtol=rtol)
                end
            end

            @testset "H * x" begin
                x = randn(n)

                @testset "frule" begin
                    dH = randtangent(L)
                    dx = randn(n)
                    y, dy = frule((NoTangent(), dH, dx), *, H, x)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    y2 = H2 * (x + 1e-7 * dx)
                    @test isapprox(dy, (y2 - y) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(*, H, x)
                    Δy = randn(n)
                    dx = randn(n)
                    _, ΔH, Δx = pullback(Δy)

                    y2 = H * (x + 1e-7 * dx)
                    dy = (y2 - y) / 1e-7
                    @test isapprox(dot(Δy, dy), dot(unthunk(Δx), dx); rtol=rtol)
                end
            end

            @testset "x * H" begin
                x = randn(1, n)

                @testset "frule" begin
                    dH = randtangent(L)
                    dx = randn(1, n)
                    y, dy = frule((NoTangent(), dx, dH), *, x, H)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    y2 = (x + 1e-7 * dx) * H2
                    @test isapprox(dy, (y2 - y) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(*, x, H)
                    Δy = randn(1, n)
                    dx = randn(1, n)
                    _, Δx, ΔH = pullback(Δy)

                    y2 = (x + 1e-7 * dx) * H
                    dy = (y2 - y) / 1e-7
                    @test isapprox(dot(Δy, dy), dot(unthunk(Δx), dx); rtol=rtol)
                end
            end

            @testset "dot(A, B)" begin
                A_parent = randtangent(L)
                B_parent = randtangent(L)
                A_herm = Hermitian(A_parent, UPLO)
                B_herm = Hermitian(B_parent, UPLO)

                @testset "frule" begin
                    dA = randtangent(L)
                    dB = randtangent(L)
                    y, dy = frule((NoTangent(), dA, dB), dot, A_herm, B_herm)

                    A2 = A_herm + 1e-7 * Hermitian(dA, UPLO)
                    B2 = B_herm + 1e-7 * Hermitian(dB, UPLO)
                    y2 = dot(A2, B2)
                    @test isapprox(dy, (y2 - y) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(dot, A_herm, B_herm)
                    Δy = randn()
                    dA = randtangent(L)
                    _, ΔA, ΔB = pullback(Δy)

                    A2 = A_herm + 1e-7 * Hermitian(dA, UPLO)
                    dy = (dot(A2, B_herm) - y) / 1e-7
                    rhs = dot(Hermitian(unthunk(ΔA), UPLO), Hermitian(dA, UPLO))
                    @test isapprox(Δy * dy, rhs; rtol=rtol)
                end
            end

            @testset "dot(x, A, y)" begin
                x = randn(n)
                z = randn(n)

                @testset "frule" begin
                    dH = randtangent(L)
                    dx = randn(n)
                    dz = randn(n)
                    y, dy = frule((NoTangent(), dx, dH, dz), dot, x, H, z)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    y2 = dot(x + 1e-7 * dx, H2, z + 1e-7 * dz)
                    @test isapprox(dy, (y2 - y) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(dot, x, H, z)
                    Δy = randn()
                    dH = randtangent(L)
                    dx = randn(n)
                    dz = randn(n)
                    _, Δx, ΔH, Δz = pullback(Δy)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    dy_H = (dot(x, H2, z) - y) / 1e-7
                    dy_x = (dot(x + 1e-7 * dx, H, z) - y) / 1e-7
                    dy_z = (dot(x, H, z + 1e-7 * dz) - y) / 1e-7
                    @test isapprox(Δy * dy_H, dot(Hermitian(unthunk(ΔH), UPLO), Hermitian(dH, UPLO)); rtol=rtol)
                    @test isapprox(Δy * dy_x, dot(unthunk(Δx), dx); rtol=rtol)
                    @test isapprox(Δy * dy_z, dot(unthunk(Δz), dz); rtol=rtol)
                end
            end

            @testset "flat/unflattri" begin
                @testset "flat frule" begin
                    dL = randtangent(L)
                    y, dy = frule((NoTangent(), dL), flat, L)

                    L2 = L + 1e-7 * dL
                    y2 = flat(L2)
                    @test isapprox(dy, (y2 - y) / 1e-7; rtol=rtol)
                end

                @testset "flat rrule" begin
                    y, pullback = rrule(flat, L)
                    Δflat = randn(ndz(S) + nlz(S))
                    dL = randtangent(L)
                    _, ΔL = pullback(Δflat)

                    L2 = L + 1e-7 * dL
                    y2 = flat(L2)
                    dy_flat = (y2 - y) / 1e-7
                    lhs = dot(Δflat, dy_flat)
                    rhs = dot(unthunk(ΔL), dL)
                    @test isapprox(lhs, rhs; rtol=rtol)
                end

                @testset "unflattri frule" begin
                    flatvec = flat(L)
                    dflat = randn(length(flatvec))
                    L_out, dL = frule((NoTangent(), dflat, NoTangent(), NoTangent()), unflattri, flatvec, S, L.uplo)

                    L2 = unflattri(flatvec + 1e-7 * dflat, S, L.uplo)
                    dL_fd = (L2 - L_out) / 1e-7
                    @test isapprox(dL, dL_fd; rtol=rtol)
                end

                @testset "unflattri rrule" begin
                    flatvec = flat(L)
                    L_out, pullback = rrule(unflattri, flatvec, S, L.uplo)
                    ΔL = randtangent(L)
                    dflat = randn(length(flatvec))
                    _, Δflat, _, _ = pullback(ΔL)

                    L2 = unflattri(flatvec + 1e-7 * dflat, S, L.uplo)
                    dL = (L2 - L_out) / 1e-7
                    lhs = dot(ΔL, dL)
                    rhs = dot(unthunk(Δflat), dflat)
                    @test isapprox(lhs, rhs; rtol=rtol)
                end
            end

            @testset "tr(H)" begin
                @testset "frule" begin
                    dH = randtangent(L)
                    y, dy = frule((NoTangent(), dH), tr, H)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    y2 = tr(H2)
                    @test isapprox(dy, (y2 - y) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(tr, H)
                    Δy = randn()
                    dH = randtangent(L)
                    _, ΔH = pullback(Δy)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    dy = (tr(H2) - y) / 1e-7
                    @test isapprox(Δy * dy, dot(Hermitian(unthunk(ΔH), UPLO), Hermitian(dH, UPLO)); rtol=rtol)
                end
            end

            @testset "diag(H)" begin
                @testset "frule" begin
                    dH = randtangent(L)
                    y, dy = frule((NoTangent(), dH), diag, H)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    y2 = diag(H2)
                    @test isapprox(dy, (y2 - y) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(diag, H)
                    Δy = randn(size(H, 1))
                    dH = randtangent(L)
                    _, ΔH = pullback(Δy)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    dy = (diag(H2) - y) / 1e-7
                    @test isapprox(dot(Δy, dy), dot(Hermitian(unthunk(ΔH), UPLO), Hermitian(dH, UPLO)); rtol=rtol)
                end
            end

            @testset "α * H" begin
                α = 2.5

                @testset "frule" begin
                    dα = 0.3
                    dH = randtangent(L)
                    y, dy = frule((NoTangent(), dα, dH), *, α, H)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    y2 = (α + 1e-7 * dα) * H2
                    @test isapprox(parent(dy), (parent(y2) - parent(y)) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(*, α, H)
                    ΔY = randtangent(L)
                    dH = randtangent(L)
                    _, Δα, ΔH = pullback(ΔY)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    dY = (parent(α * H2) - parent(y)) / 1e-7
                    @test isapprox(dot(Hermitian(ΔY, UPLO), Hermitian(dY, UPLO)), dot(Hermitian(unthunk(ΔH), UPLO), Hermitian(dH, UPLO)); rtol=rtol)
                end
            end

            @testset "H * α" begin
                α = 2.5

                @testset "frule" begin
                    dα = 0.3
                    dH = randtangent(L)
                    y, dy = frule((NoTangent(), dH, dα), *, H, α)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    y2 = H2 * (α + 1e-7 * dα)
                    @test isapprox(parent(dy), (parent(y2) - parent(y)) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(*, H, α)
                    ΔY = randtangent(L)
                    dH = randtangent(L)
                    _, ΔH, Δα = pullback(ΔY)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    dY = (parent(H2 * α) - parent(y)) / 1e-7
                    @test isapprox(dot(Hermitian(ΔY, UPLO), Hermitian(dY, UPLO)), dot(Hermitian(unthunk(ΔH), UPLO), Hermitian(dH, UPLO)); rtol=rtol)
                end
            end

            @testset "H / α" begin
                α = 2.5

                @testset "frule" begin
                    dα = 0.3
                    dH = randtangent(L)
                    y, dy = frule((NoTangent(), dH, dα), /, H, α)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    y2 = H2 / (α + 1e-7 * dα)
                    @test isapprox(parent(dy), (parent(y2) - parent(y)) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(/, H, α)
                    ΔY = randtangent(L)
                    dH = randtangent(L)
                    _, ΔH, Δα = pullback(ΔY)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    dY = (parent(H2 / α) - parent(y)) / 1e-7
                    @test isapprox(dot(Hermitian(ΔY, UPLO), Hermitian(dY, UPLO)), dot(Hermitian(unthunk(ΔH), UPLO), Hermitian(dH, UPLO)); rtol=rtol)
                end
            end

            @testset "α \\ H" begin
                α = 2.5

                @testset "frule" begin
                    dα = 0.3
                    dH = randtangent(L)
                    y, dy = frule((NoTangent(), dα, dH), \, α, H)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    y2 = (α + 1e-7 * dα) \ H2
                    @test isapprox(parent(dy), (parent(y2) - parent(y)) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(\, α, H)
                    ΔY = randtangent(L)
                    dH = randtangent(L)
                    _, Δα, ΔH = pullback(ΔY)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    dY = (parent(α \ H2) - parent(y)) / 1e-7
                    @test isapprox(dot(Hermitian(ΔY, UPLO), Hermitian(dY, UPLO)), dot(Hermitian(unthunk(ΔH), UPLO), Hermitian(dH, UPLO)); rtol=rtol)
                end
            end

            @testset "H + H" begin
                H2_base = Hermitian(randtangent(L) + 10 * parent(H), UPLO)

                @testset "frule" begin
                    dH1 = randtangent(L)
                    dH2 = randtangent(L)
                    y, dy = frule((NoTangent(), dH1, dH2), +, H, H2_base)

                    H1_pert = H + 1e-7 * Hermitian(dH1, UPLO)
                    H2_pert = H2_base + 1e-7 * Hermitian(dH2, UPLO)
                    y2 = H1_pert + H2_pert
                    @test isapprox(parent(dy), (parent(y2) - parent(y)) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(+, H, H2_base)
                    ΔY = randtangent(L)
                    dH1 = randtangent(L)
                    _, ΔH1, ΔH2 = pullback(ΔY)

                    H1_pert = H + 1e-7 * Hermitian(dH1, UPLO)
                    dY = (parent(H1_pert + H2_base) - parent(y)) / 1e-7
                    @test isapprox(dot(Hermitian(ΔY, UPLO), Hermitian(dY, UPLO)), dot(Hermitian(unthunk(ΔH1), UPLO), Hermitian(dH1, UPLO)); rtol=rtol)
                end
            end

            @testset "H + αI" begin
                α = 2.5

                @testset "frule" begin
                    dα = 0.3
                    dH = randtangent(L)
                    y, dy = frule((NoTangent(), dH, dα * I), +, H, α * I)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    y2 = H2 + (α + 1e-7 * dα) * I
                    @test isapprox(parent(dy), (parent(y2) - parent(y)) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(+, H, α * I)
                    ΔY = randtangent(L)
                    dH = randtangent(L)
                    _, ΔH, ΔαI = pullback(ΔY)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    dY = (parent(H2 + α * I) - parent(y)) / 1e-7
                    @test isapprox(dot(Hermitian(ΔY, UPLO), Hermitian(dY, UPLO)), dot(Hermitian(unthunk(ΔH), UPLO), Hermitian(dH, UPLO)); rtol=rtol)
                end
            end

            @testset "αI + H" begin
                α = 2.5

                @testset "frule" begin
                    dα = 0.3
                    dH = randtangent(L)
                    y, dy = frule((NoTangent(), dα * I, dH), +, α * I, H)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    y2 = (α + 1e-7 * dα) * I + H2
                    @test isapprox(parent(dy), (parent(y2) - parent(y)) / 1e-7; rtol=rtol)
                end

                @testset "rrule" begin
                    y, pullback = rrule(+, α * I, H)
                    ΔY = randtangent(L)
                    dH = randtangent(L)
                    _, ΔαI, ΔH = pullback(ΔY)

                    H2 = H + 1e-7 * Hermitian(dH, UPLO)
                    dY = (parent(α * I + H2) - parent(y)) / 1e-7
                    @test isapprox(dot(Hermitian(ΔY, UPLO), Hermitian(dY, UPLO)), dot(Hermitian(unthunk(ΔH), UPLO), Hermitian(dH, UPLO)); rtol=rtol)
                end
            end
        end
    end
end # differentiation (manual)
