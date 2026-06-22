using LinearAlgebra: nullspace

@testset "nullspace" begin
    # Sheaf Laplacian L = D'D where D is a coboundary operator (double integrator dynamics)
    # This matrix has a 6-dimensional null space
    h = 0.25
    ρ = [1 h h^2/2; 0 1 h]      # dynamics restriction map [A_d | B_d]
    σ = [-1 0 0; 0 -1 0]        # state projection map -[I | 0]
    D = spzeros(18, 24)
    D[1:2,   1:2]   = -I(2)                         # boundary edge
    D[3:4,   1:3]   = ρ;   D[3:4,   4:6]   = σ      # dynamics edges
    D[5:6,   4:6]   = ρ;   D[5:6,   7:9]   = σ
    D[7:8,   7:9]   = ρ;   D[7:8,   10:12] = σ
    D[9:10,  10:12] = ρ;   D[9:10,  13:15] = σ
    D[11:12, 13:15] = ρ;   D[11:12, 16:18] = σ
    D[13:14, 16:18] = ρ;   D[13:14, 19:21] = σ
    D[15:16, 19:21] = ρ;   D[15:16, 22:23] = σ[:, 1:2]
    D[17:18, 22:24] = ρ                              # last edge (no target projection)
    L = D' * D

    # Verify it's semidefinite with non-trivial null space
    eigs = eigvals(Matrix(L))
    nullity = count(e -> abs(e) < 1e-10, eigs)
    @test minimum(eigs) ≈ 0 atol=1e-10
    @test nullity == 6

    # Note: nullspace currently only supports UPLO=:L
    UPLO = :L

    # Compute pivoted LDLt factorization
    F = ldlt!(ChordalLDLt{UPLO}(L), RowMaximum(); check=false)

    # Compute nullspace basis
    N = nullspace(F)

    # Check dimensions
    @test size(N, 1) == size(L, 1)
    @test size(N, 2) == nullity

    # Check that columns are in the nullspace: L * N ≈ 0
    @test norm(L * N) < 1e-10

    # Check orthonormality (columns should be linearly independent)
    @test rank(Matrix(N)) == nullity

    # Check that output is a Matrix
    @test isa(N, Matrix)

    # Test with a simple rank-1 matrix (full rank minus 1)
    n = 10
    v = randn(n)
    A = sparse(v * v')  # rank 1, nullity n-1

    F = ldlt!(ChordalLDLt{UPLO}(Symmetric(A, UPLO)), RowMaximum(); check=false)
    N = nullspace(F)

    @test size(N, 2) == n - 1
    @test norm(A * N) < 1e-10
    @test rank(Matrix(N)) == n - 1

    # Test with a full-rank positive definite matrix (empty nullspace)
    M = readmatrix("nos4")

    F = ldlt!(ChordalLDLt{UPLO}(M), RowMaximum())
    N = nullspace(F)

    @test size(N, 2) == 0
    @test isa(N, Matrix)

    # Test rtol/atol parameters
    F = ldlt!(ChordalLDLt{UPLO}(L), RowMaximum(); check=false)

    # With very tight tolerance, should get same result
    N1 = nullspace(F)
    N2 = nullspace(F; rtol=eps(Float64))
    @test size(N1) == size(N2)

    # With absolute tolerance
    N3 = nullspace(F; atol=1e-8)
    @test size(N3, 2) == nullity
end
