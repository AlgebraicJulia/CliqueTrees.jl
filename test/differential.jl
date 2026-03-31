using Test
using LinearAlgebra
using SparseArrays
using Random: randn!
using MatrixMarket
using SuiteSparseMatrixCollection
using CliqueTrees
using ChainRulesCore: NoTangent, ZeroTangent, unthunk
using CliqueTrees.Multifrontal: ChordalTriangular, ChordalSymbolic, HermTri, HermOrSymSparse
using CliqueTrees.Multifrontal: chordal, symbolic, ndz, nlz
using CliqueTrees.Multifrontal.Differential
using CliqueTrees.Multifrontal.Differential: frule, rrule
using CliqueTrees.Multifrontal.Differential: cholesky, uncholesky, selinv
using CliqueTrees.Multifrontal.Differential: ldivsym

if !@isdefined(SSMC)
    const SSMC = ssmc_db()
end

if !@isdefined(readmatrix)
    function readmatrix(name::String)
        path = joinpath(fetch_ssmc(SSMC[SSMC.name .== name, :]; format="MM")[1], "$(name).mtx")
        return mmread(path)
    end
end

# Helper for inner products that handles unthunking and UniformScaling
# When both args are UniformScaling, we need the dimension n for correct Frobenius inner product
function _mydot(a::UniformScaling, b::UniformScaling, n::Int)
    # ⟨λI, μI⟩ = λ * μ * n for n×n matrices
    return a.λ * b.λ * n
end

function _mydot(a, b, ::Int)
    return dot(a, b)
end

function mydot(a, b, n::Int)
    return _mydot(unthunk(a), unthunk(b), n)
end

# Without dimension - only works correctly when at least one arg is not UniformScaling
function mydot(a, b)
    au = unthunk(a)
    bu = unthunk(b)
    if au isa UniformScaling && bu isa UniformScaling
        error("mydot(UniformScaling, UniformScaling) requires dimension n - use mydot(a, b, n)")
    end
    return dot(au, bu)
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

    # Helper to create random tangent for HermOrSymSparse
    function randtangent(A::HermOrSymSparse)
        dA = copy(A)
        randn!(nonzeros(parent(dA)))
        return dA
    end

    function randtangent(D::Diagonal)
        return Diagonal(randn(eltype(D), size(D, 1)))
    end

    function randtangent(::UniformScaling{T}) where T
        return randn(T) * I
    end

    # Helper to generate all tangent types to test for a given primal
    function tangents(L::ChordalTriangular)
        T = eltype(L)
        n = size(L, 1)
        return (randtangent(L), Diagonal(randn(T, n)), randn(T) * I, ZeroTangent())
    end

    # For Hermitian, tangents must also be Hermitian (in the tangent space)
    function tangents(H::HermTri)
        L = parent(H)
        T = eltype(L)
        n = size(L, 1)
        uplo = Symbol(H.uplo)
        return (Hermitian(randtangent(L), uplo), Diagonal(randn(T, n)), randn(T) * I, ZeroTangent())
    end

    function tangents(x::AbstractArray{T}) where T
        return (randn(T, size(x)...), ZeroTangent())
    end

    function tangents(x::T) where T<:Number
        return (randn(T), ZeroTangent())
    end

    function tangents(::UniformScaling{T}) where T
        return (randn(T) * I, ZeroTangent())
    end

    function tangents(D::Diagonal)
        return (randtangent(D), ZeroTangent())
    end

    # Tangents for HermOrSymSparse
    function tangents(A::HermOrSymSparse)
        return (randtangent(A), ZeroTangent())
    end

    # Helper for comparing tangents (handles ZeroTangent)
    tangent_approx(a, b) = iszero(a) && iszero(b) || a ≈ b

    for T in (Float32, Float64, BigFloat)
        for UPLO in (:L, :U)
            A = Hermitian(SparseMatrixCSC{T}(M), UPLO)
            P, S = symbolic(A)
            H = chordal(A, P, S, Val(UPLO))
            L = cholesky(H)
            n = size(L, 1)

            @testset "T = $T, UPLO = $UPLO" begin
                @testset "cholesky" begin
                    # Input: H (Hermitian), Output: L (ChordalTriangular)
                    # dH must be in tangent space of H (Hermitian)
                    for dH in tangents(H)
                        for ΔL in tangents(L)
                            _, dL = frule((NoTangent(), dH), cholesky, H)
                            _, pullback = rrule(cholesky, H)
                            _, ΔH = pullback(ΔL)

                            lhs = mydot(ΔL, dL)
                            rhs = mydot(ΔH, dH)
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "uncholesky" begin
                    # Input: L (ChordalTriangular), Output: H (Hermitian)
                    # ΔH must be in cotangent space of H (Hermitian)
                    for dL in tangents(L)
                        for ΔH in tangents(H)
                            _, dH = frule((NoTangent(), dL), uncholesky, L)
                            _, pullback = rrule(uncholesky, L)
                            _, ΔL = pullback(ΔH)

                            lhs = mydot(ΔH, dH)
                            rhs = mydot(ΔL, dL)
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "L \\ x" begin
                    x = randn(T, n)
                    for dL in tangents(L)
                        for dx in tangents(x)
                            for Δy in tangents(x)
                                _, dy = frule((NoTangent(), dL, dx), \, L, x)
                                _, pullback = rrule(\, L, x)
                                _, ΔL, Δx = pullback(Δy)

                                lhs = mydot(Δy, dy)
                                rhs = mydot(ΔL, dL) + mydot(Δx, dx)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "L' \\ x" begin
                    x = randn(T, n)
                    # Tangent for L' is dL' where dL is tangent of L
                    for dL in tangents(L)
                        dL_adj = dL isa ZeroTangent ? dL : dL'
                        for dx in tangents(x)
                            for Δy in tangents(x)
                                _, dy = frule((NoTangent(), dL_adj, dx), \, L', x)
                                _, pullback = rrule(\, L', x)
                                _, ΔL_adj, Δx = pullback(Δy)

                                lhs = mydot(Δy, dy)
                                rhs = mydot(ΔL_adj, dL_adj) + mydot(Δx, dx)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "L * x" begin
                    x = randn(T, n)
                    for dL in tangents(L)
                        for dx in tangents(x)
                            for Δy in tangents(x)
                                _, dy = frule((NoTangent(), dL, dx), *, L, x)
                                _, pullback = rrule(*, L, x)
                                _, ΔL, Δx = pullback(Δy)

                                lhs = mydot(Δy, dy)
                                rhs = mydot(ΔL, dL) + mydot(Δx, dx)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "L' * x" begin
                    x = randn(T, n)
                    # Tangent for L' is dL' where dL is tangent of L
                    for dL in tangents(L)
                        dL_adj = dL isa ZeroTangent ? dL : dL'
                        for dx in tangents(x)
                            for Δy in tangents(x)
                                _, dy = frule((NoTangent(), dL_adj, dx), *, L', x)
                                _, pullback = rrule(*, L', x)
                                _, ΔL_adj, Δx = pullback(Δy)

                                lhs = mydot(Δy, dy)
                                rhs = mydot(ΔL_adj, dL_adj) + mydot(Δx, dx)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "x / L" begin
                    x = randn(T, 1, n)
                    for dL in tangents(L)
                        for dx in tangents(x)
                            for Δy in tangents(x)
                                _, dy = frule((NoTangent(), dx, dL), /, x, L)
                                _, pullback = rrule(/, x, L)
                                _, Δx, ΔL = pullback(Δy)

                                lhs = mydot(Δy, dy)
                                rhs = mydot(Δx, dx) + mydot(ΔL, dL)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "x / L'" begin
                    x = randn(T, 1, n)
                    # Tangent for L' is dL' where dL is tangent of L
                    for dL in tangents(L)
                        dL_adj = dL isa ZeroTangent ? dL : dL'
                        for dx in tangents(x)
                            for Δy in tangents(x)
                                _, dy = frule((NoTangent(), dx, dL_adj), /, x, L')
                                _, pullback = rrule(/, x, L')
                                _, Δx, ΔL_adj = pullback(Δy)

                                lhs = mydot(Δy, dy)
                                rhs = mydot(Δx, dx) + mydot(ΔL_adj, dL_adj)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "x * L" begin
                    x = randn(T, 1, n)
                    for dL in tangents(L)
                        for dx in tangents(x)
                            for Δy in tangents(x)
                                _, dy = frule((NoTangent(), dx, dL), *, x, L)
                                _, pullback = rrule(*, x, L)
                                _, Δx, ΔL = pullback(Δy)

                                lhs = mydot(Δy, dy)
                                rhs = mydot(Δx, dx) + mydot(ΔL, dL)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "x * L'" begin
                    x = randn(T, 1, n)
                    # Tangent for L' is dL' where dL is tangent of L
                    for dL in tangents(L)
                        dL_adj = dL isa ZeroTangent ? dL : dL'
                        for dx in tangents(x)
                            for Δy in tangents(x)
                                _, dy = frule((NoTangent(), dx, dL_adj), *, x, L')
                                _, pullback = rrule(*, x, L')
                                _, Δx, ΔL_adj = pullback(Δy)

                                lhs = mydot(Δy, dy)
                                rhs = mydot(Δx, dx) + mydot(ΔL_adj, dL_adj)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "logdet" begin
                    for dL in tangents(L)
                        for Δy in tangents(one(T))
                            _, dy = frule((NoTangent(), dL), logdet, L)
                            _, pullback = rrule(logdet, L)
                            _, ΔL = pullback(Δy)

                            lhs = mydot(Δy, dy)
                            rhs = mydot(ΔL, dL)
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "logdet(A::HermOrSymSparse, L, P)" begin
                    for dA in tangents(A)
                        for Δy in tangents(one(T))
                            _, dy = frule((NoTangent(), dA, NoTangent(), NoTangent()), logdet, A, L, P)
                            _, pullback = rrule(logdet, A, L, P)
                            _, ΔA, _, _ = pullback(Δy)

                            lhs = mydot(Δy, dy)
                            rhs = mydot(ΔA, dA)
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "diag" begin
                    y = randn(T, n)
                    for dL in tangents(L)
                        for Δy in tangents(y)
                            _, dy = frule((NoTangent(), dL), diag, L)
                            _, pullback = rrule(diag, L)
                            _, ΔL = pullback(Δy)

                            lhs = mydot(Δy, dy)
                            rhs = mydot(ΔL, dL)
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "selinv" begin
                    # Input: L (ChordalTriangular), Output: Y (Hermitian)
                    # ΔY must be in cotangent space of Y (Hermitian)
                    for dL in tangents(L)
                        for ΔY in tangents(H)
                            _, dY = frule((NoTangent(), dL), selinv, L)
                            _, pullback = rrule(selinv, L)
                            _, ΔL = pullback(ΔY)

                            lhs = mydot(ΔY, dY)
                            rhs = mydot(ΔL, dL)
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "selinv(A::HermOrSymSparse, L, P)" begin
                    for dA in tangents(A)
                        for ΔY in tangents(A)
                            _, dY = frule((NoTangent(), dA, NoTangent(), NoTangent()), selinv, A, L, P)
                            _, pullback = rrule(selinv, A, L, P)
                            _, ΔA, _, _ = pullback(ΔY)

                            lhs = mydot(ΔY, dY)
                            rhs = mydot(ΔA, dA)
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end


                @testset "ldivsym(A::HermOrSymSparse, L, P, x)" begin
                    x = randn(T, n)
                    for dA in tangents(A)
                        for dx in tangents(x)
                            for Δy in tangents(x)
                                _, dy = frule((NoTangent(), dA, NoTangent(), NoTangent(), dx), ldivsym, A, L, P, x)
                                _, pullback = rrule(ldivsym, A, L, P, x)
                                _, ΔA, _, _, Δx = pullback(Δy)

                                lhs = mydot(Δy, dy)
                                rhs = mydot(ΔA, dA) + mydot(Δx, dx)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "H * x" begin
                    x = randn(T, n)
                    # Input: H (Hermitian), so dH must be in tangents(H)
                    for dH in tangents(H)
                        for dx in tangents(x)
                            for Δy in tangents(x)
                                _, dy = frule((NoTangent(), dH, dx), *, H, x)
                                _, pullback = rrule(*, H, x)
                                _, ΔH, Δx = pullback(Δy)

                                lhs = mydot(Δy, dy)
                                rhs = mydot(ΔH, dH) + mydot(Δx, dx)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "x * H" begin
                    x = randn(T, 1, n)
                    # Input: H (Hermitian), so dH must be in tangents(H)
                    for dH in tangents(H)
                        for dx in tangents(x)
                            for Δy in tangents(x)
                                _, dy = frule((NoTangent(), dx, dH), *, x, H)
                                _, pullback = rrule(*, x, H)
                                _, Δx, ΔH = pullback(Δy)

                                lhs = mydot(Δy, dy)
                                rhs = mydot(Δx, dx) + mydot(ΔH, dH)
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
                    # Input: A_herm, B_herm (Hermitian), so dA, dB must be in tangents(H)
                    for dA in tangents(H)
                        for dB in tangents(H)
                            for Δy in tangents(one(T))
                                _, dy = frule((NoTangent(), dA, dB), dot, A_herm, B_herm)
                                _, pullback = rrule(dot, A_herm, B_herm)
                                _, ΔA, ΔB = pullback(Δy)

                                lhs = mydot(Δy, dy)
                                rhs = mydot(ΔA, dA) + mydot(ΔB, dB)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "dot(x, A, y)" begin
                    x = randn(T, n)
                    z = randn(T, n)
                    # Input: H (Hermitian), so dH must be in tangents(H)
                    for dH in tangents(H)
                        for dx in tangents(x)
                            for dz in tangents(z)
                                for Δy in tangents(one(T))
                                    _, dy = frule((NoTangent(), dx, dH, dz), dot, x, H, z)
                                    _, pullback = rrule(dot, x, H, z)
                                    _, Δx, ΔH, Δz = pullback(Δy)

                                    lhs = mydot(Δy, dy)
                                    rhs = mydot(Δx, dx) + mydot(ΔH, dH) + mydot(Δz, dz)
                                    @test tangent_approx(lhs, rhs)
                                end
                            end
                        end
                    end
                end

                @testset "chordal" begin
                    # Output: H (Hermitian), so ΔH must be in tangents(H) - need dimension for UniformScaling
                    for dA in (randtangent(A), ZeroTangent())
                        for ΔH in tangents(H)
                            _, dH = frule((NoTangent(), dA, NoTangent(), NoTangent(), NoTangent()), chordal, A, P, S, Val(UPLO))
                            _, pullback = rrule(chordal, A, P, S, Val(UPLO))
                            _, ΔA, _, _, _ = pullback(ΔH)

                            lhs = mydot(ΔH, dH, n)
                            rhs = mydot(ΔA, dA)
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "tr(H)" begin
                    # Input: H (Hermitian), so dH must be in tangents(H)
                    # ΔH is UniformScaling, so need dimension for mydot
                    for dH in tangents(H)
                        for Δy in tangents(one(T))
                            _, dy = frule((NoTangent(), dH), tr, H)
                            _, pullback = rrule(tr, H)
                            _, ΔH = pullback(Δy)

                            lhs = mydot(Δy, dy)
                            rhs = mydot(ΔH, dH, n)
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "diag(H)" begin
                    y = randn(T, n)
                    # Input: H (Hermitian), so dH must be in tangents(H)
                    for dH in tangents(H)
                        for Δy in tangents(y)
                            _, dy = frule((NoTangent(), dH), diag, H)
                            _, pullback = rrule(diag, H)
                            _, ΔH = pullback(Δy)

                            lhs = mydot(Δy, dy)
                            rhs = mydot(ΔH, dH)
                            @test tangent_approx(lhs, rhs)
                        end
                    end
                end

                @testset "α * H" begin
                    α = T(2.5)
                    # Input/output: H (Hermitian) - need dimension for UniformScaling
                    for dα in tangents(α)
                        for dH in tangents(H)
                            for ΔY in tangents(H)
                                _, dy = frule((NoTangent(), dα, dH), *, α, H)
                                _, pullback = rrule(*, α, H)
                                _, Δα, ΔH = pullback(ΔY)

                                lhs = mydot(ΔY, dy, n)
                                rhs = mydot(Δα, dα) + mydot(ΔH, dH, n)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "H * α" begin
                    α = T(2.5)
                    # Input/output: H (Hermitian) - need dimension for UniformScaling
                    for dH in tangents(H)
                        for dα in tangents(α)
                            for ΔY in tangents(H)
                                _, dy = frule((NoTangent(), dH, dα), *, H, α)
                                _, pullback = rrule(*, H, α)
                                _, ΔH, Δα = pullback(ΔY)

                                lhs = mydot(ΔY, dy, n)
                                rhs = mydot(ΔH, dH, n) + mydot(Δα, dα)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "H / α" begin
                    α = T(2.5)
                    # Input/output: H (Hermitian) - need dimension for UniformScaling
                    for dH in tangents(H)
                        for dα in tangents(α)
                            for ΔY in tangents(H)
                                _, dy = frule((NoTangent(), dH, dα), /, H, α)
                                _, pullback = rrule(/, H, α)
                                _, ΔH, Δα = pullback(ΔY)

                                lhs = mydot(ΔY, dy, n)
                                rhs = mydot(ΔH, dH, n) + mydot(Δα, dα)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "α \\ H" begin
                    α = T(2.5)
                    # Input/output: H (Hermitian) - need dimension for UniformScaling
                    for dα in tangents(α)
                        for dH in tangents(H)
                            for ΔY in tangents(H)
                                _, dy = frule((NoTangent(), dα, dH), \, α, H)
                                _, pullback = rrule(\, α, H)
                                _, Δα, ΔH = pullback(ΔY)

                                lhs = mydot(ΔY, dy, n)
                                rhs = mydot(Δα, dα) + mydot(ΔH, dH, n)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "H + H" begin
                    H2_base = Hermitian(randtangent(L) + T(10) * parent(H), UPLO)
                    # Input/output: H (Hermitian) - need dimension for UniformScaling
                    for dH1 in tangents(H)
                        for dH2 in tangents(H)
                            for ΔY in tangents(H)
                                _, dy = frule((NoTangent(), dH1, dH2), +, H, H2_base)
                                _, pullback = rrule(+, H, H2_base)
                                _, ΔH1, ΔH2 = pullback(ΔY)

                                lhs = mydot(ΔY, dy, n)
                                rhs = mydot(ΔH1, dH1, n) + mydot(ΔH2, dH2, n)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "H + αI" begin
                    α = T(2.5) * I
                    # Input/output: H (Hermitian) - need dimension for UniformScaling
                    for dH in tangents(H)
                        for dα in tangents(α)
                            for ΔY in tangents(H)
                                _, dy = frule((NoTangent(), dH, dα), +, H, α)
                                _, pullback = rrule(+, H, α)
                                _, ΔH, Δα = pullback(ΔY)

                                lhs = mydot(ΔY, dy, n)
                                rhs = mydot(ΔH, dH, n) + mydot(Δα, dα, n)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "αI + H" begin
                    α = T(2.5) * I
                    # Input/output: H (Hermitian) - need dimension for UniformScaling
                    for dα in tangents(α)
                        for dH in tangents(H)
                            for ΔY in tangents(H)
                                _, dy = frule((NoTangent(), dα, dH), +, α, H)
                                _, pullback = rrule(+, α, H)
                                _, Δα, ΔH = pullback(ΔY)

                                lhs = mydot(ΔY, dy, n)
                                rhs = mydot(Δα, dα, n) + mydot(ΔH, dH, n)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "H + D" begin
                    D = Diagonal(randn(T, n))
                    # Input/output: H (Hermitian) - need dimension for UniformScaling
                    for dH in tangents(H)
                        for dD in tangents(D)
                            for ΔY in tangents(H)
                                _, dy = frule((NoTangent(), dH, dD), +, H, D)
                                _, pullback = rrule(+, H, D)
                                _, ΔH, ΔD = pullback(ΔY)

                                lhs = mydot(ΔY, dy, n)
                                rhs = mydot(ΔH, dH, n) + mydot(ΔD, dD)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "D + H" begin
                    D = Diagonal(randn(T, n))
                    # Input/output: H (Hermitian) - need dimension for UniformScaling
                    for dD in tangents(D)
                        for dH in tangents(H)
                            for ΔY in tangents(H)
                                _, dy = frule((NoTangent(), dD, dH), +, D, H)
                                _, pullback = rrule(+, D, H)
                                _, ΔD, ΔH = pullback(ΔY)

                                lhs = mydot(ΔY, dy, n)
                                rhs = mydot(ΔD, dD) + mydot(ΔH, dH, n)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end

                @testset "H - D" begin
                    D = Diagonal(randn(T, n))
                    # Input/output: H (Hermitian) - need dimension for UniformScaling
                    for dH in tangents(H)
                        for dD in tangents(D)
                            for ΔY in tangents(H)
                                _, dy = frule((NoTangent(), dH, dD), -, H, D)
                                _, pullback = rrule(-, H, D)
                                _, ΔH, ΔD = pullback(ΔY)

                                lhs = mydot(ΔY, dy, n)
                                rhs = mydot(ΔH, dH, n) + mydot(ΔD, dD)
                                @test tangent_approx(lhs, rhs)
                            end
                        end
                    end
                end
            end  # @testset "T = $T, UPLO = $UPLO"
        end  # for UPLO
    end  # for T
end  # @testset "differentiation (manual)"
