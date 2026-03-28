using Test
using LinearAlgebra
using SparseArrays
using Random: randn!
using MatrixMarket
using SuiteSparseMatrixCollection
using CliqueTrees
using ChainRules
using ChainRulesCore: ProjectTo, NoTangent, ZeroTangent, unthunk
using CliqueTrees.Multifrontal: ChordalTriangular, ChordalSymbolic, HermTri
using CliqueTrees.Multifrontal: chordal, cong, symbolic, ndz, nlz
using CliqueTrees.Multifrontal.Differential
using CliqueTrees.Multifrontal.Differential: frule, rrule
using CliqueTrees.Multifrontal.Differential: cholesky, uncholesky, selinv, softmax, sigmoid
using CliqueTrees.Multifrontal.Differential: flat, unflattri, unflatsym
using CliqueTrees.Multifrontal.Differential: ldiv, rdiv

if !@isdefined(SSMC)
    const SSMC = ssmc_db()
end

if !@isdefined(readmatrix)
    function readmatrix(name::String)
        path = joinpath(fetch_ssmc(SSMC[SSMC.name .== name, :]; format="MM")[1], "$(name).mtx")
        return mmread(path)
    end
end

function udot(a, b)
    return dot(a, b)
end

function udot(a::UniformScaling, b::UniformScaling)
    return a.λ * b.λ
end

@testset "differentiation (manual)" begin
    M = readmatrix("685_bus")

    # Helper to create random tangent for ChordalTriangular
    function randtangent(L::ChordalTriangular)
        dL = zero(L)
        randn!(dL.Dval)
        randn!(dL.Lval)
        return dL
    end

    # Helper to create random tangent for SparseMatrixCSC
    function randtangent(A::SparseMatrixCSC)
        dA = copy(A)
        randn!(dA.nzval)
        return dA
    end

    # Helper to create random tangent for Hermitian sparse matrix
    function randtangent(A::Hermitian{<:Any, <:SparseMatrixCSC})
        return Hermitian(randtangent(parent(A)), Symbol(A.uplo))
    end

    function randtangent(D::Diagonal)
        return Diagonal(randn(eltype(D), size(D, 1)))
    end

    # Helper for comparing tangents (handles ZeroTangent)
    tangent_approx(a, b) = iszero(a) && iszero(b) || a ≈ b

    for T in (Float32, Float64, BigFloat)
        A = SparseMatrixCSC{T}(M)

        for UPLO in (:L, :U)
            P, S = symbolic(A)
            H = chordal(A, P, S, Val(UPLO))
            L = cholesky(H)
            B = Hermitian(A, UPLO)
            n = size(L, 1)

            @testset "T = $T, UPLO = $UPLO" begin
                @testset "cholesky" begin
                    PH = ProjectTo(H) ∘ unthunk
                    PL = ProjectTo(L) ∘ unthunk
                    for dH in (randtangent(L), ZeroTangent())
                        for ΔL in (randtangent(L), ZeroTangent())
                            _, dL = frule((NoTangent(), dH), cholesky, H)
                            _, pullback = rrule(cholesky, H)
                            _, ΔH = pullback(ΔL)

                            lhs = dot(PL(ΔL), PL(dL))
                            rhs = dot(PH(ΔH), PH(dH))
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "uncholesky" begin
                    PH = ProjectTo(H) ∘ unthunk
                    PL = ProjectTo(L) ∘ unthunk
                    for dL in (randtangent(L), ZeroTangent())
                        for ΔH in (randtangent(L), ZeroTangent())
                            _, dH = frule((NoTangent(), dL), uncholesky, L)
                            _, pullback = rrule(uncholesky, L)
                            _, ΔL = pullback(ΔH)

                            lhs = dot(PH(ΔH), PH(dH))
                            rhs = dot(PL(ΔL), PL(dL))
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "L \\ x" begin
                    x = randn(T, n)
                    PL = ProjectTo(L) ∘ unthunk
                    Px = ProjectTo(x) ∘ unthunk
                    for dL in (randtangent(L), ZeroTangent())
                        for dx in (randn(T, n), ZeroTangent())
                            for Δy in (randn(T, n), ZeroTangent())
                                _, dy = frule((NoTangent(), dL, dx), \, L, x)
                                _, pullback = rrule(\, L, x)
                                _, ΔL, Δx = pullback(Δy)

                                lhs = dot(Px(Δy), Px(dy))
                                rhs = dot(PL(ΔL), PL(dL)) + dot(Px(Δx), Px(dx))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "L' \\ x" begin
                    x = randn(T, n)
                    PL = ProjectTo(L) ∘ unthunk
                    Px = ProjectTo(x) ∘ unthunk
                    for dL in (randtangent(L), ZeroTangent())
                        for dx in (randn(T, n), ZeroTangent())
                            for Δy in (randn(T, n), ZeroTangent())
                                _, dy = frule((NoTangent(), dL, dx), \, L', x)
                                _, pullback = rrule(\, L', x)
                                _, ΔL, Δx = pullback(Δy)

                                lhs = dot(Px(Δy), Px(dy))
                                rhs = dot(PL(ΔL), PL(dL)) + dot(Px(Δx), Px(dx))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "L * x" begin
                    x = randn(T, n)
                    PL = ProjectTo(L) ∘ unthunk
                    Px = ProjectTo(x) ∘ unthunk
                    for dL in (randtangent(L), ZeroTangent())
                        for dx in (randn(T, n), ZeroTangent())
                            for Δy in (randn(T, n), ZeroTangent())
                                _, dy = frule((NoTangent(), dL, dx), *, L, x)
                                _, pullback = rrule(*, L, x)
                                _, ΔL, Δx = pullback(Δy)

                                lhs = dot(Px(Δy), Px(dy))
                                rhs = dot(PL(ΔL), PL(dL)) + dot(Px(Δx), Px(dx))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "L' * x" begin
                    x = randn(T, n)
                    PL = ProjectTo(L) ∘ unthunk
                    Px = ProjectTo(x) ∘ unthunk
                    for dL in (randtangent(L), ZeroTangent())
                        for dx in (randn(T, n), ZeroTangent())
                            for Δy in (randn(T, n), ZeroTangent())
                                _, dy = frule((NoTangent(), dL, dx), *, L', x)
                                _, pullback = rrule(*, L', x)
                                _, ΔL, Δx = pullback(Δy)

                                lhs = dot(Px(Δy), Px(dy))
                                rhs = dot(PL(ΔL), PL(dL)) + dot(Px(Δx), Px(dx))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "x / L" begin
                    x = randn(T, 1, n)
                    PL = ProjectTo(L) ∘ unthunk
                    Px = ProjectTo(x) ∘ unthunk
                    for dL in (randtangent(L), ZeroTangent())
                        for dx in (randn(T, 1, n), ZeroTangent())
                            for Δy in (randn(T, 1, n), ZeroTangent())
                                _, dy = frule((NoTangent(), dx, dL), /, x, L)
                                _, pullback = rrule(/, x, L)
                                _, Δx, ΔL = pullback(Δy)

                                lhs = dot(Px(Δy), Px(dy))
                                rhs = dot(Px(Δx), Px(dx)) + dot(PL(ΔL), PL(dL))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "x / L'" begin
                    x = randn(T, 1, n)
                    PL = ProjectTo(L) ∘ unthunk
                    Px = ProjectTo(x) ∘ unthunk
                    for dL in (randtangent(L), ZeroTangent())
                        for dx in (randn(T, 1, n), ZeroTangent())
                            for Δy in (randn(T, 1, n), ZeroTangent())
                                _, dy = frule((NoTangent(), dx, dL), /, x, L')
                                _, pullback = rrule(/, x, L')
                                _, Δx, ΔL = pullback(Δy)

                                lhs = dot(Px(Δy), Px(dy))
                                rhs = dot(Px(Δx), Px(dx)) + dot(PL(ΔL), PL(dL))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "x * L" begin
                    x = randn(T, 1, n)
                    PL = ProjectTo(L) ∘ unthunk
                    Px = ProjectTo(x) ∘ unthunk
                    for dL in (randtangent(L), ZeroTangent())
                        for dx in (randn(T, 1, n), ZeroTangent())
                            for Δy in (randn(T, 1, n), ZeroTangent())
                                _, dy = frule((NoTangent(), dx, dL), *, x, L)
                                _, pullback = rrule(*, x, L)
                                _, Δx, ΔL = pullback(Δy)

                                lhs = dot(Px(Δy), Px(dy))
                                rhs = dot(Px(Δx), Px(dx)) + dot(PL(ΔL), PL(dL))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "x * L'" begin
                    x = randn(T, 1, n)
                    PL = ProjectTo(L) ∘ unthunk
                    Px = ProjectTo(x) ∘ unthunk
                    for dL in (randtangent(L), ZeroTangent())
                        for dx in (randn(T, 1, n), ZeroTangent())
                            for Δy in (randn(T, 1, n), ZeroTangent())
                                _, dy = frule((NoTangent(), dx, dL), *, x, L')
                                _, pullback = rrule(*, x, L')
                                _, Δx, ΔL = pullback(Δy)

                                lhs = dot(Px(Δy), Px(dy))
                                rhs = dot(Px(Δx), Px(dx)) + dot(PL(ΔL), PL(dL))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "logdet" begin
                    PL = ProjectTo(L) ∘ unthunk
                    Py = ProjectTo(one(T)) ∘ unthunk
                    for dL in (randtangent(L), ZeroTangent())
                        for Δy in (randn(T), ZeroTangent())
                            _, dy = frule((NoTangent(), dL), logdet, L)
                            _, pullback = rrule(logdet, L)
                            _, ΔL = pullback(Δy)

                            lhs = dot(Py(Δy), Py(dy))
                            rhs = dot(PL(ΔL), PL(dL))
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "logdet(H, L)" begin
                    PH = ProjectTo(H) ∘ unthunk
                    Py = ProjectTo(one(T)) ∘ unthunk
                    for dH in (randtangent(L), ZeroTangent())
                        for Δy in (randn(T), ZeroTangent())
                            _, dy = frule((NoTangent(), dH, NoTangent()), logdet, H, L)
                            _, pullback = rrule(logdet, H, L)
                            _, ΔH, _ = pullback(Δy)

                            lhs = dot(Py(Δy), Py(dy))
                            rhs = dot(PH(ΔH), PH(dH))
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "diag" begin
                    PL = ProjectTo(L) ∘ unthunk
                    Py = ProjectTo(randn(T, n)) ∘ unthunk
                    for dL in (randtangent(L), ZeroTangent())
                        for Δy in (randn(T, n), ZeroTangent())
                            _, dy = frule((NoTangent(), dL), diag, L)
                            _, pullback = rrule(diag, L)
                            _, ΔL = pullback(Δy)

                            lhs = dot(Py(Δy), Py(dy))
                            rhs = dot(PL(ΔL), PL(dL))
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "selinv" begin
                    PL = ProjectTo(L) ∘ unthunk
                    PH = ProjectTo(H) ∘ unthunk
                    for dL in (randtangent(L), ZeroTangent())
                        for ΔY in (randtangent(L), ZeroTangent())
                            _, dY = frule((NoTangent(), dL), selinv, L)
                            _, pullback = rrule(selinv, L)
                            _, ΔL = pullback(ΔY)

                            lhs = dot(PH(ΔY), PH(dY))
                            rhs = dot(PL(ΔL), PL(dL))
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "selinv(H, L)" begin
                    PH = ProjectTo(H) ∘ unthunk
                    for dH in (randtangent(L), ZeroTangent())
                        for ΔY in (randtangent(L), ZeroTangent())
                            _, dY = frule((NoTangent(), dH, NoTangent()), selinv, H, L)
                            _, pullback = rrule(selinv, H, L)
                            _, ΔH, _ = pullback(ΔY)

                            lhs = dot(PH(ΔY), PH(dY))
                            rhs = dot(PH(ΔH), PH(dH))
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "ldiv(H, L, x)" begin
                    x = randn(T, n)
                    PH = ProjectTo(H) ∘ unthunk
                    Px = ProjectTo(x) ∘ unthunk
                    for dH in (randtangent(L), ZeroTangent())
                        for dx in (randn(T, n), ZeroTangent())
                            for Δy in (randn(T, n), ZeroTangent())
                                _, dy = frule((NoTangent(), dH, NoTangent(), dx), ldiv, H, L, x)
                                _, pullback = rrule(ldiv, H, L, x)
                                _, ΔH, _, Δx = pullback(Δy)

                                lhs = dot(Px(Δy), Px(dy))
                                rhs = dot(PH(ΔH), PH(dH)) + dot(Px(Δx), Px(dx))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "rdiv(x, H, L)" begin
                    x = randn(T, 1, n)
                    PH = ProjectTo(H) ∘ unthunk
                    Px = ProjectTo(x) ∘ unthunk
                    for dH in (randtangent(L), ZeroTangent())
                        for dx in (randn(T, 1, n), ZeroTangent())
                            for Δy in (randn(T, 1, n), ZeroTangent())
                                _, dy = frule((NoTangent(), dx, dH, NoTangent()), rdiv, x, H, L)
                                _, pullback = rrule(rdiv, x, H, L)
                                _, Δx, ΔH, _ = pullback(Δy)

                                lhs = dot(Px(Δy), Px(dy))
                                rhs = dot(Px(Δx), Px(dx)) + dot(PH(ΔH), PH(dH))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "softmax" begin
                    L_in = copy(L)
                    randn!(L_in.Dval)
                    randn!(L_in.Lval)
                    PL = ProjectTo(L) ∘ unthunk
                    for dL in (randtangent(L), ZeroTangent())
                        for ΔY in (randtangent(L), ZeroTangent())
                            _, dY = frule((NoTangent(), dL), softmax, L_in)
                            _, pullback = rrule(softmax, L_in)
                            _, ΔL = pullback(ΔY)

                            lhs = dot(PL(ΔY), PL(dY))
                            rhs = dot(PL(ΔL), PL(dL))
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "H * x" begin
                    x = randn(T, n)
                    PH = ProjectTo(H) ∘ unthunk
                    Px = ProjectTo(x) ∘ unthunk
                    for dH in (randtangent(L), ZeroTangent())
                        for dx in (randn(T, n), ZeroTangent())
                            for Δy in (randn(T, n), ZeroTangent())
                                _, dy = frule((NoTangent(), dH, dx), *, H, x)
                                _, pullback = rrule(*, H, x)
                                _, ΔH, Δx = pullback(Δy)

                                lhs = dot(Px(Δy), Px(dy))
                                rhs = dot(PH(ΔH), PH(dH)) + dot(Px(Δx), Px(dx))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "x * H" begin
                    x = randn(T, 1, n)
                    PH = ProjectTo(H) ∘ unthunk
                    Px = ProjectTo(x) ∘ unthunk
                    for dH in (randtangent(L), ZeroTangent())
                        for dx in (randn(T, 1, n), ZeroTangent())
                            for Δy in (randn(T, 1, n), ZeroTangent())
                                _, dy = frule((NoTangent(), dx, dH), *, x, H)
                                _, pullback = rrule(*, x, H)
                                _, Δx, ΔH = pullback(Δy)

                                lhs = dot(Px(Δy), Px(dy))
                                rhs = dot(Px(Δx), Px(dx)) + dot(PH(ΔH), PH(dH))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "dot(A, B)" begin
                    A_parent = randtangent(L)
                    B_parent = randtangent(L)
                    A_herm = Hermitian(A_parent, UPLO)
                    B_herm = Hermitian(B_parent, UPLO)
                    PA = ProjectTo(A_herm) ∘ unthunk
                    PB = ProjectTo(B_herm) ∘ unthunk
                    Py = ProjectTo(one(T)) ∘ unthunk
                    for dA in (randtangent(L), ZeroTangent())
                        for dB in (randtangent(L), ZeroTangent())
                            for Δy in (randn(T), ZeroTangent())
                                _, dy = frule((NoTangent(), dA, dB), dot, A_herm, B_herm)
                                _, pullback = rrule(dot, A_herm, B_herm)
                                _, ΔA, ΔB = pullback(Δy)

                                lhs = dot(Py(Δy), Py(dy))
                                rhs = dot(PA(ΔA), PA(dA)) + dot(PB(ΔB), PB(dB))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "dot(x, A, y)" begin
                    x = randn(T, n)
                    z = randn(T, n)
                    PH = ProjectTo(H) ∘ unthunk
                    Px = ProjectTo(x) ∘ unthunk
                    Pz = ProjectTo(z) ∘ unthunk
                    Py = ProjectTo(one(T)) ∘ unthunk
                    for dH in (randtangent(L), ZeroTangent())
                        for dx in (randn(T, n), ZeroTangent())
                            for dz in (randn(T, n), ZeroTangent())
                                for Δy in (randn(T), ZeroTangent())
                                    _, dy = frule((NoTangent(), dx, dH, dz), dot, x, H, z)
                                    _, pullback = rrule(dot, x, H, z)
                                    _, Δx, ΔH, Δz = pullback(Δy)

                                    lhs = dot(Py(Δy), Py(dy))
                                    rhs = dot(Px(Δx), Px(dx)) + dot(PH(ΔH), PH(dH)) + dot(Pz(Δz), Pz(dz))
                                    @test tangent_approx(lhs, rhs)
                                end
                            end
                        end
                    end
                end

                @testset "flat" begin
                    flatvec = flat(L)
                    PL = ProjectTo(L) ∘ unthunk
                    Pflat = ProjectTo(flatvec) ∘ unthunk
                    for dL in (randtangent(L), ZeroTangent())
                        for Δflat in (randn(T, ndz(S) + nlz(S)), ZeroTangent())
                            _, dy = frule((NoTangent(), dL), flat, L)
                            _, pullback = rrule(flat, L)
                            _, ΔL = pullback(Δflat)

                            lhs = dot(Pflat(Δflat), Pflat(dy))
                            rhs = dot(PL(ΔL), PL(dL))
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "unflattri" begin
                    flatvec = flat(L)
                    PL = ProjectTo(L) ∘ unthunk
                    Pflat = ProjectTo(flatvec) ∘ unthunk
                    for dflat in (randn(T, length(flatvec)), ZeroTangent())
                        for ΔL in (randtangent(L), ZeroTangent())
                            _, dL = frule((NoTangent(), dflat, NoTangent(), NoTangent()), unflattri, flatvec, S, L.uplo)
                            _, pullback = rrule(unflattri, flatvec, S, L.uplo)
                            _, Δflat, _, _ = pullback(ΔL)

                            lhs = dot(PL(ΔL), PL(dL))
                            rhs = dot(Pflat(Δflat), Pflat(dflat))
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "chordal" begin
                    PA = ProjectTo(B) ∘ unthunk
                    PH = ProjectTo(H) ∘ unthunk
                    for dA in (randtangent(B), ZeroTangent())
                        for ΔH in (randtangent(L), ZeroTangent())
                            _, dH = frule((NoTangent(), dA, NoTangent(), NoTangent(), NoTangent()), chordal, B, P, S, Val(UPLO))
                            _, pullback = rrule(chordal, B, P, S, Val(UPLO))
                            _, ΔA, _, _, _ = pullback(ΔH)

                            lhs = dot(PH(ΔH), PH(dH))
                            rhs = dot(PA(ΔA), PA(dA))
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "cong" begin
                    PA = unthunk
                    PH = ProjectTo(H) ∘ unthunk
                    for dH in (randtangent(L), ZeroTangent())
                        for ΔA in (randtangent(B), ZeroTangent())
                            _, dA = frule((NoTangent(), dH, NoTangent()), cong, H, P)
                            _, pullback = rrule(cong, H, P)
                            _, ΔH, _ = pullback(ΔA)

                            lhs = dot(PA(ΔA), PA(dA))
                            rhs = dot(PH(ΔH), PH(dH))
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "tr(H)" begin
                    PH = ProjectTo(H) ∘ unthunk
                    Py = ProjectTo(one(T)) ∘ unthunk
                    for dH in (randtangent(L), ZeroTangent())
                        for Δy in (randn(T), ZeroTangent())
                            _, dy = frule((NoTangent(), dH), tr, H)
                            _, pullback = rrule(tr, H)
                            _, ΔH = pullback(Δy)

                            lhs = dot(Py(Δy), Py(dy))
                            rhs = dot(PH(ΔH), PH(dH))
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "diag(H)" begin
                    PH = ProjectTo(H) ∘ unthunk
                    Py = ProjectTo(randn(T, n)) ∘ unthunk
                    for dH in (randtangent(L), ZeroTangent())
                        for Δy in (randn(T, n), ZeroTangent())
                            _, dy = frule((NoTangent(), dH), diag, H)
                            _, pullback = rrule(diag, H)
                            _, ΔH = pullback(Δy)

                            lhs = dot(Py(Δy), Py(dy))
                            rhs = dot(PH(ΔH), PH(dH))
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "α * H" begin
                    α = T(2.5)
                    PH = ProjectTo(H) ∘ unthunk
                    Pα = ProjectTo(α) ∘ unthunk
                    for dα in (T(0.3), ZeroTangent())
                        for dH in (randtangent(L), ZeroTangent())
                            for ΔY in (randtangent(L), ZeroTangent())
                                _, dy = frule((NoTangent(), dα, dH), *, α, H)
                                _, pullback = rrule(*, α, H)
                                _, Δα, ΔH = pullback(ΔY)

                                lhs = dot(PH(ΔY), PH(dy))
                                rhs = dot(Pα(Δα), Pα(dα)) + dot(PH(ΔH), PH(dH))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "H * α" begin
                    α = T(2.5)
                    PH = ProjectTo(H) ∘ unthunk
                    Pα = ProjectTo(α) ∘ unthunk
                    for dH in (randtangent(L), ZeroTangent())
                        for dα in (T(0.3), ZeroTangent())
                            for ΔY in (randtangent(L), ZeroTangent())
                                _, dy = frule((NoTangent(), dH, dα), *, H, α)
                                _, pullback = rrule(*, H, α)
                                _, ΔH, Δα = pullback(ΔY)

                                lhs = dot(PH(ΔY), PH(dy))
                                rhs = dot(PH(ΔH), PH(dH)) + dot(Pα(Δα), Pα(dα))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "H / α" begin
                    α = T(2.5)
                    PH = ProjectTo(H) ∘ unthunk
                    Pα = ProjectTo(α) ∘ unthunk
                    for dH in (randtangent(L), ZeroTangent())
                        for dα in (T(0.3), ZeroTangent())
                            for ΔY in (randtangent(L), ZeroTangent())
                                _, dy = frule((NoTangent(), dH, dα), /, H, α)
                                _, pullback = rrule(/, H, α)
                                _, ΔH, Δα = pullback(ΔY)

                                lhs = dot(PH(ΔY), PH(dy))
                                rhs = dot(PH(ΔH), PH(dH)) + dot(Pα(Δα), Pα(dα))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "α \\ H" begin
                    α = T(2.5)
                    PH = ProjectTo(H) ∘ unthunk
                    Pα = ProjectTo(α) ∘ unthunk
                    for dα in (T(0.3), ZeroTangent())
                        for dH in (randtangent(L), ZeroTangent())
                            for ΔY in (randtangent(L), ZeroTangent())
                                _, dy = frule((NoTangent(), dα, dH), \, α, H)
                                _, pullback = rrule(\, α, H)
                                _, Δα, ΔH = pullback(ΔY)

                                lhs = dot(PH(ΔY), PH(dy))
                                rhs = dot(Pα(Δα), Pα(dα)) + dot(PH(ΔH), PH(dH))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "H + H" begin
                    H2_base = Hermitian(randtangent(L) + T(10) * parent(H), UPLO)
                    PH = ProjectTo(H) ∘ unthunk
                    PH2 = ProjectTo(H2_base) ∘ unthunk
                    for dH1 in (randtangent(L), ZeroTangent())
                        for dH2 in (randtangent(L), ZeroTangent())
                            for ΔY in (randtangent(L), ZeroTangent())
                                _, dy = frule((NoTangent(), dH1, dH2), +, H, H2_base)
                                _, pullback = rrule(+, H, H2_base)
                                _, ΔH1, ΔH2 = pullback(ΔY)

                                lhs = dot(PH(ΔY), PH(dy))
                                rhs = dot(PH(ΔH1), PH(dH1)) + dot(PH2(ΔH2), PH2(dH2))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "H + αI" begin
                    α = T(2.5) * I
                    PH = ProjectTo(H) ∘ unthunk
                    Pα = ProjectTo(α) ∘ unthunk
                    for dH in (randtangent(L), ZeroTangent())
                        for dα in (T(0.3) * I, ZeroTangent())
                            for ΔY in (randtangent(L), ZeroTangent())
                                _, dy = frule((NoTangent(), dH, dα), +, H, α)
                                _, pullback = rrule(+, H, α)
                                _, ΔH, Δα = pullback(ΔY)

                                lhs = dot(PH(ΔY), PH(dy))
                                rhs = dot(PH(ΔH), PH(dH)) + udot(Pα(Δα), Pα(dα))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "αI + H" begin
                    α = T(2.5) * I
                    PH = ProjectTo(H) ∘ unthunk
                    Pα = ProjectTo(α) ∘ unthunk
                    for dα in (T(0.3) * I, ZeroTangent())
                        for dH in (randtangent(L), ZeroTangent())
                            for ΔY in (randtangent(L), ZeroTangent())
                                _, dy = frule((NoTangent(), dα, dH), +, α, H)
                                _, pullback = rrule(+, α, H)
                                _, Δα, ΔH = pullback(ΔY)

                                lhs = dot(PH(ΔY), PH(dy))
                                rhs = udot(Pα(Δα), Pα(dα)) + dot(PH(ΔH), PH(dH))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "H + D" begin
                    D = Diagonal(randn(T, n))
                    PH = ProjectTo(H) ∘ unthunk
                    PD = ProjectTo(D) ∘ unthunk
                    for dH in (randtangent(L), ZeroTangent())
                        for dD in (randtangent(D), ZeroTangent())
                            for ΔY in (randtangent(L), ZeroTangent())
                                _, dy = frule((NoTangent(), dH, dD), +, H, D)
                                _, pullback = rrule(+, H, D)
                                _, ΔH, ΔD = pullback(ΔY)

                                lhs = dot(PH(ΔY), PH(dy))
                                rhs = dot(PH(ΔH), PH(dH)) + dot(PD(ΔD), PD(dD))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "D + H" begin
                    D = Diagonal(randn(T, n))
                    PH = ProjectTo(H) ∘ unthunk
                    PD = ProjectTo(D) ∘ unthunk
                    for dD in (randtangent(D), ZeroTangent())
                        for dH in (randtangent(L), ZeroTangent())
                            for ΔY in (randtangent(L), ZeroTangent())
                                _, dy = frule((NoTangent(), dD, dH), +, D, H)
                                _, pullback = rrule(+, D, H)
                                _, ΔD, ΔH = pullback(ΔY)

                                lhs = dot(PH(ΔY), PH(dy))
                                rhs = dot(PD(ΔD), PD(dD)) + dot(PH(ΔH), PH(dH))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "H - D" begin
                    D = Diagonal(randn(T, n))
                    PH = ProjectTo(H) ∘ unthunk
                    PD = ProjectTo(D) ∘ unthunk
                    for dH in (randtangent(L), ZeroTangent())
                        for dD in (randtangent(D), ZeroTangent())
                            for ΔY in (randtangent(L), ZeroTangent())
                                _, dy = frule((NoTangent(), dH, dD), -, H, D)
                                _, pullback = rrule(-, H, D)
                                _, ΔH, ΔD = pullback(ΔY)

                                lhs = dot(PH(ΔY), PH(dy))
                                rhs = dot(PH(ΔH), PH(dH)) + dot(PD(ΔD), PD(dD))
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end
            end  # @testset "T = $T, UPLO = $UPLO"
        end  # for UPLO
    end  # for T
end  # @testset "differentiation (manual)"
