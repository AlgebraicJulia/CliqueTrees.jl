using CliqueTrees.Multifrontal: flatindices, setflatindex!, triangular, selinv!, complete!, fisherroot!, fisher!, amari!

@testset "cholesky" begin
    matrices = ("685_bus", "Trefethen_500", "bcsstk26", "bcsstk13", "mhd1280b")

    for UPLO in (:L, :U), pivot in (NoPivot(), RowMaximum())
        for name in matrices
            M = readmatrix(name); n = size(M, 2)
            F0 = cholesky(M)

            b = rand(n)
            x = rand(n)
            B = rand(n, 4)
            X = rand(n, 4)
            C = rand(5, n)
            Y = rand(5, n)

            F = cholesky!(ChordalCholesky{UPLO}(M), pivot)
            @test isa(repr("text/plain", F), String)
            @test issuccess(F) == issuccess(F0)
            @test logdet(F) ≈ logdet(F0)
            @test det(F) ≈ det(F0)
            @test isapprox(b, M * (F \ b); rtol=1e-6, atol=1e-14)
            @test isapprox(B, M * (F \ B); rtol=1e-6, atol=1e-14)
            @test isapprox(C, (C / F) * M; rtol=1e-6, atol=1e-14)
            @test isapprox(M * x, F * x; rtol=1e-6, atol=1e-14)
            @test isapprox(M * X, F * X; rtol=1e-6, atol=1e-14)
            @test isapprox(Y * M, Y * F; rtol=1e-6, atol=1e-14)

            F = ldlt!(ChordalLDLt{UPLO}(M), pivot)
            @test isa(repr("text/plain", F), String)
            @test issuccess(F) == issuccess(F0)
            @test logdet(F) ≈ logdet(F0)
            @test det(F) ≈ det(F0)
            @test isapprox(b, M * (F \ b); rtol=1e-6, atol=1e-14)
            @test isapprox(B, M * (F \ B); rtol=1e-6, atol=1e-14)
            @test isapprox(C, (C / F) * M; rtol=1e-6, atol=1e-14)
            @test isapprox(M * x, F * x; rtol=1e-6, atol=1e-14)
            @test isapprox(M * X, F * X; rtol=1e-6, atol=1e-14)
            @test isapprox(Y * M, Y * F; rtol=1e-6, atol=1e-14)
        end

        M = SparseMatrixCSC{BigFloat}(readmatrix("nos4")); n = size(M, 2)

        b = rand(BigFloat, n)
        x = rand(BigFloat, n)
        B = rand(BigFloat, n, 4)
        X = rand(BigFloat, n, 4)
        C = rand(BigFloat, 5, n)
        Y = rand(BigFloat, 5, n)

        F = cholesky!(ChordalCholesky{UPLO}(M), pivot)
        @test isapprox(b, M * (F \ b); rtol=1e-6, atol=1e-14)
        @test isapprox(B, M * (F \ B); rtol=1e-6, atol=1e-14)
        @test isapprox(C, (C / F) * M; rtol=1e-6, atol=1e-14)
        @test isapprox(M * x, F * x; rtol=1e-6, atol=1e-14)
        @test isapprox(M * X, F * X; rtol=1e-6, atol=1e-14)
        @test isapprox(Y * M, Y * F; rtol=1e-6, atol=1e-14)

        F = ldlt!(ChordalLDLt{UPLO}(M), pivot)
        @test isapprox(b, M * (F \ b); rtol=1e-6, atol=1e-14)
        @test isapprox(B, M * (F \ B); rtol=1e-6, atol=1e-14)
        @test isapprox(C, (C / F) * M; rtol=1e-6, atol=1e-14)
        @test isapprox(M * x, F * x; rtol=1e-6, atol=1e-14)
        @test isapprox(M * X, F * X; rtol=1e-6, atol=1e-14)
        @test isapprox(Y * M, Y * F; rtol=1e-6, atol=1e-14)

        M = SparseMatrixCSC{Float64}(readmatrix("685_bus"))
        @inferred cholesky!(ChordalCholesky{UPLO}(M), pivot)
        @inferred ldlt!(ChordalLDLt{UPLO}(M), pivot)
        @test_call target_modules = (CliqueTrees,) cholesky!(ChordalCholesky{UPLO}(M), pivot)
        @test_call target_modules = (CliqueTrees,) ldlt!(ChordalLDLt{UPLO}(M), pivot)
        @test_opt target_modules = (CliqueTrees,) cholesky!(ChordalCholesky{UPLO}(M), pivot)
        @test_opt target_modules = (CliqueTrees,) ldlt!(ChordalLDLt{UPLO}(M), pivot)

        for name in ("685_bus", "bcsstk26")
            M = readmatrix(name); n = size(M, 2)
            b = rand(n)

            for A in (Symmetric(M, :L), M)
                F = cholesky!(ChordalCholesky{UPLO}(A), pivot)
                P = flatindices(F, A)

                fill!(F, 0)
                cholesky!(F; check=false)
                @test !issuccess(F)

                for (i, p) in enumerate(P)
                    iszero(p) && continue
                    setflatindex!(F, nonzeros(M)[i], p)
                end

                cholesky!(F)
                @test issuccess(F)
                @test isapprox(b, M * (F \ b); rtol=1e-6, atol=1e-14)

                F = ldlt!(ChordalLDLt{UPLO}(A), pivot)
                P = flatindices(F, A)

                fill!(F, 0)
                ldlt!(F; check=false)
                @test !issuccess(F)

                for (i, p) in enumerate(P)
                    iszero(p) && continue
                    setflatindex!(F, nonzeros(M)[i], p)
                end

                ldlt!(F)
                @test issuccess(F)
                @test isapprox(b, M * (F \ b); rtol=1e-6, atol=1e-14)
            end
        end
    end
end

@testset "cholesky (cholmod)" begin
    matrices = ("685_bus", "Trefethen_500", "bcsstk26", "bcsstk13", "mhd1280b")

    for name in matrices
        M = readmatrix(name); n = size(M, 2)
        F0 = cholesky(M)

        b = rand(n)
        B = rand(n, 4)

        F = ChordalCholesky(F0)
        @test isapprox(b, M * (F \ b); rtol=1e-6, atol=1e-14)
        @test isapprox(B, M * (F \ B); rtol=1e-6, atol=1e-14)
    end
end

@testset "cholesky (dense)" begin
    for UPLO in (:L, :U), pivot in (NoPivot(), RowMaximum())
        for n in (10, 50, 100)
            A = rand(n, n)
            M = A' * A + I
            F0 = cholesky(M)

            b = rand(n)
            B = rand(n, 4)

            F = cholesky!(DenseCholeskyPivoted{UPLO}(copy(M)), pivot)
            @test isa(repr("text/plain", F), String)
            @test issuccess(F) == issuccess(F0)
            @test logdet(F) ≈ logdet(F0)
            @test det(F) ≈ det(F0)
            @test isapprox(b, M * (F \ b); rtol=1e-6, atol=1e-14)
            @test isapprox(B, M * (F \ B); rtol=1e-6, atol=1e-14)

            F = ldlt!(DenseLDLtPivoted{UPLO}(copy(M)), pivot)
            @test isa(repr("text/plain", F), String)
            @test issuccess(F) == issuccess(F0)
            @test logdet(F) ≈ logdet(F0)
            @test det(F) ≈ det(F0)
            @test isapprox(b, M * (F \ b); rtol=1e-6, atol=1e-14)
            @test isapprox(B, M * (F \ B); rtol=1e-6, atol=1e-14)
        end
    end
end

@testset "regularization (dense)" begin
    A = rand(10, 10)
    A = A' * A
    A -= 1.0001 * minimum(eigvals(A)) * I
    signs = ones(size(A, 1))

    for UPLO in (:L, :U)
        for R in (GMW81, SE99)
            F = DenseCholesky{UPLO}(copy(A))
            cholesky!(F; reg=R())
            @test A ≈ Matrix(F) rtol=1e-3

            F = DenseCholeskyPivoted{UPLO}(copy(A))
            cholesky!(F, RowMaximum(); reg=R())
            @test A ≈ Matrix(F) rtol=1e-3

            F = DenseLDLt{UPLO}(copy(A))
            ldlt!(F; signs, reg=R())
            @test A ≈ Matrix(F) rtol=1e-3

            F = DenseLDLtPivoted{UPLO}(copy(A))
            ldlt!(F, RowMaximum(); signs, reg=R())
            @test A ≈ Matrix(F) rtol=1e-3
        end
    end

    A = Float64[
        1 1 2
        1 1 3
        2 3 1
    ]

    signs = [1, 1, 1]

    for UPLO in (:L, :U)
        F = DenseCholesky{UPLO}(copy(A))
        cholesky!(F; reg=GMW81())
        @test diag(F) ≈ [3.771, 5.750, 1.121] rtol=1e-3

        F = DenseLDLt{UPLO}(copy(A))
        ldlt!(F; signs, reg=GMW81())
        @test diag(F) ≈ [3.771, 5.750, 1.121] rtol=1e-3
    end
end

@testset "regularization (sparse)" begin
    A = readmatrix("msc01440")
    A -= 1.0001 * minimum(eigvals(Matrix(A))) * I
    signs = ones(size(A, 1))

    for UPLO in (:L, :U)
        for P in (NoPivot, RowMaximum)
            for R in (GMW81, SE99)
                F = ChordalCholesky{UPLO}(A)
                cholesky!(F, P(); reg=R())
                @test A ≈ Matrix(F) rtol=1e-3

                F = ChordalLDLt{UPLO}(A)
                ldlt!(F, P(); signs, reg=R())
                @test A ≈ Matrix(F) rtol=1e-3
            end
        end
    end

    C = readmatrix("662_bus")
    C -= 1.0001 * minimum(eigvals(Matrix(C))) * I
    B = sprand(size(A, 1), size(C, 1), .001)

    P = randperm(size(A, 1) + size(C, 1))
    A = [A B; B' -C][P, P]
    signs = [signs; -signs][P]

    for UPLO in (:L, :U)
        for P in (NoPivot, RowMaximum)
            for R in (GMW81, SE99)
                F = ChordalLDLt{UPLO}(A)
                ldlt!(F, P(); signs, reg=R())
                @test A ≈ Matrix(F) rtol=1e-3
            end
        end
    end

    A = [
        1 1 2
        1 1 3
        2 3 1
    ]

    signs = [1, 1, 1]

    for UPLO in (:L, :U)
        F = ChordalCholesky{UPLO}(A)
        cholesky!(F; reg=GMW81())
        @test diag(F) ≈ [3.771, 5.750, 1.121] rtol=1e-3

        F = ChordalLDLt{UPLO}(A)
        ldlt!(F; signs, reg=GMW81())
        @test diag(F) ≈ [3.771, 5.750, 1.121] rtol=1e-3
    end
end

@testset "condition number" begin
    A = readmatrix("bcsstk26")
    F = cholesky!(ChordalCholesky(A))

    for p in (1, Inf)
        @test cond(F, p) ≈ cond(A, p) rtol=1
        @test cond(F.L, p) ≈ cond(sparse(F.L), p) rtol=1
        @test cond(F.U, p) ≈ cond(sparse(F.U), p) rtol=1
    end

    F = ldlt!(ChordalLDLt(A))

    for p in (1, Inf)
        @test cond(F, p) ≈ cond(A, p) rtol=1
        @test cond(F.L, p) ≈ cond(sparse(F.L), p) rtol=1
        @test cond(F.U, p) ≈ cond(sparse(F.U), p) rtol=1
    end
end

@testset "selinv" begin
    matrices = ("nos4", "mesh3e1", "494_bus", "mhdb416", "685_bus")

    for name in matrices
        M = readmatrix(name); n = size(M, 2)

        for UPLO in (:L, :U)
            F = cholesky!(ChordalCholesky{UPLO}(M))
            G = copy(F)
            N = F.P * inv(Matrix(M)) * F.P'

            L = triangular(F)
            selinv!(G)
            S = triangular(G)

            @test all(isapprox(N[i, j], v; rtol=1e-6, atol=1e-14) for (i, j, v) in zip(findnz(S)...))

            complete!(G)

            @test all(isapprox(L[i, j], v; rtol=1e-6, atol=1e-14) for (i, j, v) in zip(findnz(S)...))
        end
    end

    M = SparseMatrixCSC{BigFloat}(readmatrix("nos4")); n = size(M, 2)

    for UPLO in (:L, :U)
        F = cholesky!(ChordalCholesky{UPLO}(M))
        G = copy(F)
        N = F.P * inv(Matrix(M)) * F.P'

        L = triangular(F)
        selinv!(G)
        S = triangular(G)

        @test all(isapprox(N[i, j], v; rtol=1e-6, atol=1e-14) for (i, j, v) in zip(findnz(S)...))

        complete!(G)

        @test all(isapprox(L[i, j], v; rtol=1e-6, atol=1e-14) for (i, j, v) in zip(findnz(S)...))
    end

    M = SparseMatrixCSC{Float64}(readmatrix("nos4"))
    L = cholesky!(ChordalCholesky{:L}(M))
    U = cholesky!(ChordalCholesky{:U}(M))
    @inferred selinv!(L)
    @inferred selinv!(U)
    @test_call target_modules = (CliqueTrees,) selinv!(L)
    @test_call target_modules = (CliqueTrees,) selinv!(U)
    @test_opt target_modules = (CliqueTrees,) selinv!(L)
    @test_opt target_modules = (CliqueTrees,) selinv!(U)
end

@testset "complete" begin
    matrices = ("nos4", "mesh3e1", "494_bus", "mhdb416", "685_bus")

    for name in matrices
        M = readmatrix(name)

        for UPLO in (:L, :U)
            F = cholesky!(ChordalCholesky{UPLO}(M))
            G = copy(F)

            selinv!(G)
            complete!(G)

            @test triangular(F) ≈ triangular(G)

            selinv!(G)
            complete!(G)

            @test triangular(F) ≈ triangular(G)
        end
    end

    M = SparseMatrixCSC{BigFloat}(readmatrix("nos4"))

    for UPLO in (:L, :U)
        F = cholesky!(ChordalCholesky{UPLO}(M))
        G = copy(F)

        selinv!(G)
        complete!(G)

        @test triangular(F) ≈ triangular(G)

        selinv!(G)
        complete!(G)

        @test triangular(F) ≈ triangular(G)
    end

    M = SparseMatrixCSC{Float64}(readmatrix("nos4"))
    L = selinv!(cholesky!(ChordalCholesky{:L}(M)))
    U = selinv!(cholesky!(ChordalCholesky{:U}(M)))
    @inferred complete!(L)
    @inferred complete!(U)
    @test_call target_modules = (CliqueTrees,) complete!(L)
    @test_call target_modules = (CliqueTrees,) complete!(U)
    @test_opt target_modules = (CliqueTrees,) complete!(L)
    @test_opt target_modules = (CliqueTrees,) complete!(U)
end

@testset "fisherroot (inv=false)" begin
    matrices = ("nos4", "mesh3e1", "494_bus", "mhdb416", "685_bus")

    for name in matrices
        M = readmatrix(name)

        for UPLO in (:L, :U)
            F = cholesky!(ChordalCholesky{UPLO}(M))
            S = selinv!(copy(F))

            Y = similar(F)
            rand!(Y.Dval)
            rand!(Y.Lval)
            Z = copy(Y)

            fisherroot!(Y, F, S; adj=false, inv=false)
            fisherroot!(Y, F, S; adj=true, inv=false)

            fisher!(Z, F, S; inv=false)

            @test triangular(Y) ≈ triangular(Z)
        end
    end

    M = SparseMatrixCSC{BigFloat}(readmatrix("nos4"))

    for UPLO in (:L, :U)
        F = cholesky!(ChordalCholesky{UPLO}(M))
        S = selinv!(copy(F))

        Y = similar(F)
        rand!(Y.Dval)
        rand!(Y.Lval)
        Z = copy(Y)

        fisherroot!(Y, F, S; adj=false, inv=false)
        fisherroot!(Y, F, S; adj=true, inv=false)

        fisher!(Z, F, S; inv=false)

        @test triangular(Y) ≈ triangular(Z)
    end

    M = SparseMatrixCSC{Float64}(readmatrix("nos4"))
    F = cholesky!(ChordalCholesky(M))
    S = selinv!(copy(F))
    Y = copy(F)
    @inferred fisherroot!(Y, F, S; adj=false, inv=false)
    @inferred fisherroot!(Y, F, S; adj=true, inv=false)
    @test_call target_modules = (CliqueTrees,) fisherroot!(Y, F, S; adj=false, inv=false)
    @test_call target_modules = (CliqueTrees,) fisherroot!(Y, F, S; adj=true, inv=false)
    @test_opt target_modules = (CliqueTrees,) fisherroot!(Y, F, S; adj=false, inv=false)
    @test_opt target_modules = (CliqueTrees,) fisherroot!(Y, F, S; adj=true, inv=false)
end

@testset "fisherroot (inv=true)" begin
    matrices = ("nos4", "mesh3e1", "494_bus", "mhdb416", "685_bus")

    for name in matrices
        M = readmatrix(name)

        for UPLO in (:L, :U)
            F = cholesky!(ChordalCholesky{UPLO}(M))
            S = selinv!(copy(F))

            Y = similar(F)
            rand!(Y.Dval)
            rand!(Y.Lval)
            Z = copy(Y)

            fisherroot!(Y, F, S; adj=true, inv=true)
            fisherroot!(Y, F, S; adj=false, inv=true)

            fisher!(Z, F, S; inv=true)

            @test triangular(Y) ≈ triangular(Z)
        end
    end

    M = SparseMatrixCSC{BigFloat}(readmatrix("nos4"))

    for UPLO in (:L, :U)
        F = cholesky!(ChordalCholesky{UPLO}(M))
        S = selinv!(copy(F))

        Y = similar(F)
        rand!(Y.Dval)
        rand!(Y.Lval)
        Z = copy(Y)

        fisherroot!(Y, F, S; adj=true, inv=true)
        fisherroot!(Y, F, S; adj=false, inv=true)

        fisher!(Z, F, S; inv=true)

        @test triangular(Y) ≈ triangular(Z)
    end

    M = SparseMatrixCSC{Float64}(readmatrix("nos4"))
    F = cholesky!(ChordalCholesky(M))
    S = selinv!(copy(F))
    Y = copy(F)
    @inferred fisherroot!(Y, F, S; adj=true, inv=true)
    @inferred fisherroot!(Y, F, S; adj=false, inv=true)
    @test_call target_modules = (CliqueTrees,) fisherroot!(Y, F, S; adj=true, inv=true)
    @test_call target_modules = (CliqueTrees,) fisherroot!(Y, F, S; adj=false, inv=true)
    @test_opt target_modules = (CliqueTrees,) fisherroot!(Y, F, S; adj=true, inv=true)
    @test_opt target_modules = (CliqueTrees,) fisherroot!(Y, F, S; adj=false, inv=true)
end

@testset "fisher (inv=false)" begin
    matrices = ("nos4", "mesh3e1", "494_bus", "mhdb416", "685_bus")

    for name in matrices
        M = readmatrix(name)

        for UPLO in (:L, :U)
            F = cholesky!(ChordalCholesky{UPLO}(M))
            S = selinv!(copy(F))
            Sinv = F.P * inv(Matrix(M)) * F.P'

            Y = similar(F)
            rand!(Y.Dval)
            rand!(Y.Lval)
            Ym = Symmetric(Matrix(triangular(Y)), UPLO)
            fisher!(Y, F, S; inv=false)
            Yexact = Sinv * Ym * Sinv

            YT = triangular(Y)
            @test all(isapprox(Yexact[i, j], v; rtol=1e-6, atol=1e-14) for (i, j, v) in zip(findnz(YT)...))
        end
    end

    M = SparseMatrixCSC{BigFloat}(readmatrix("nos4"))

    for UPLO in (:L, :U)
        F = cholesky!(ChordalCholesky{UPLO}(M))
        S = selinv!(copy(F))
        Sinv = F.P * inv(Matrix(M)) * F.P'

        Y = similar(F)
        rand!(Y.Dval)
        rand!(Y.Lval)
        Ym = Symmetric(Matrix(triangular(Y)), UPLO)
        fisher!(Y, F, S; inv=false)
        Yexact = Sinv * Ym * Sinv

        YT = triangular(Y)
        @test all(isapprox(Yexact[i, j], v; rtol=1e-6, atol=1e-14) for (i, j, v) in zip(findnz(YT)...))
    end

    M = SparseMatrixCSC{Float64}(readmatrix("nos4"))
    F = cholesky!(ChordalCholesky(M))
    S = selinv!(copy(F))
    Y = similar(F)
    rand!(Y.Dval)
    rand!(Y.Lval)
    @inferred fisher!(Y, F, S; inv=false)
    @test_call target_modules = (CliqueTrees,) fisher!(Y, F, S; inv=false)
    @test_opt target_modules = (CliqueTrees,) fisher!(Y, F, S; inv=false)
end

@testset "fisher (inv=true)" begin
    matrices = ("nos4", "mesh3e1", "494_bus", "mhdb416", "685_bus")

    for name in matrices
        M = readmatrix(name)

        for UPLO in (:L, :U)
            F = cholesky!(ChordalCholesky{UPLO}(M))
            S = selinv!(copy(F))

            Y = similar(F)
            rand!(Y.Dval)
            rand!(Y.Lval)
            Z = copy(Y)

            fisher!(Y, F, S; inv=false)
            fisher!(Y, F, S; inv=true)

            @test triangular(Y) ≈ triangular(Z) rtol=1e-5 atol=1e-14
        end
    end

    M = SparseMatrixCSC{BigFloat}(readmatrix("nos4"))

    for UPLO in (:L, :U)
        F = cholesky!(ChordalCholesky{UPLO}(M))
        S = selinv!(copy(F))

        Y = similar(F)
        rand!(Y.Dval)
        rand!(Y.Lval)
        Z = copy(Y)

        fisher!(Y, F, S; inv=false)
        fisher!(Y, F, S; inv=true)

        @test triangular(Y) ≈ triangular(Z) rtol=1e-5 atol=1e-14
    end

    M = SparseMatrixCSC{Float64}(readmatrix("nos4"))
    F = cholesky!(ChordalCholesky(M))
    S = selinv!(copy(F))
    Y = copy(F)
    @inferred fisher!(Y, F, S; inv=true)
    @test_call target_modules = (CliqueTrees,) fisher!(Y, F, S; inv=true)
    @test_opt target_modules = (CliqueTrees,) fisher!(Y, F, S; inv=true)
end

@testset "amari (inv=false)" begin
    matrices = ("nos4", "mesh3e1", "494_bus", "mhdb416", "685_bus")

    for name in matrices
        M = readmatrix(name)

        for UPLO in (:L, :U)
            F = cholesky!(ChordalCholesky{UPLO}(M))
            S = selinv!(copy(F))
            Sinv = F.P * inv(Matrix(M)) * F.P'

            Y = similar(F)
            rand!(Y.Dval)
            rand!(Y.Lval)
  
            Z = similar(F)
            rand!(Z.Dval)
            rand!(Z.Lval)

            T = similar(F)

            Ym = Symmetric(Matrix(triangular(Y)), UPLO)
            Zm = Symmetric(Matrix(triangular(Z)), UPLO)

            amari!(T, Y, Z, F, S; inv=false)

            Texact = -(Sinv * Ym * Sinv * Zm * Sinv + Sinv * Zm * Sinv * Ym * Sinv)

            TT = triangular(T)
            @test all(isapprox(Texact[i, j], v; rtol=1e-6, atol=1e-14) for (i, j, v) in zip(findnz(TT)...))
        end
    end

    M = SparseMatrixCSC{BigFloat}(readmatrix("nos4"))

    for UPLO in (:L, :U)
        F = cholesky!(ChordalCholesky{UPLO}(M))
        S = selinv!(copy(F))
        Sinv = F.P * inv(Matrix(M)) * F.P'

        Y = similar(F)
        rand!(Y.Dval)
        rand!(Y.Lval)

        Z = similar(F)
        rand!(Z.Dval)
        rand!(Z.Lval)

        T = similar(F)

        Ym = Symmetric(Matrix(triangular(Y)), UPLO)
        Zm = Symmetric(Matrix(triangular(Z)), UPLO)

        amari!(T, Y, Z, F, S; inv=false)

        Texact = -(Sinv * Ym * Sinv * Zm * Sinv + Sinv * Zm * Sinv * Ym * Sinv)

        TT = triangular(T)
        @test all(isapprox(Texact[i, j], v; rtol=1e-6, atol=1e-14) for (i, j, v) in zip(findnz(TT)...))
    end

    M = SparseMatrixCSC{Float64}(readmatrix("nos4"))
    F = cholesky!(ChordalCholesky(M))
    S = selinv!(copy(F))
    Y = similar(F)
    rand!(Y.Dval)
    rand!(Y.Lval)
    Z = similar(F)
    rand!(Z.Dval)
    rand!(Z.Lval)
    T = similar(F)
    @inferred amari!(T, Y, Z, F, S; inv=false)
    # @test_call target_modules = (CliqueTrees,) amari!(T, Y, Z, F, S; inv=false)
    # @test_opt target_modules = (CliqueTrees,) amari!(T, Y, Z, F, S; inv=false)
end

@testset "amari (inv=true)" begin
    matrices = ("nos4", "mesh3e1", "494_bus", "mhdb416", "685_bus")

    for name in matrices
        M = readmatrix(name)

        for UPLO in (:L, :U)
            F = cholesky!(ChordalCholesky{UPLO}(M))
            S = selinv!(copy(F))

            Y = similar(F)
            rand!(Y.Dval)
            rand!(Y.Lval)

            Z = similar(F)
            rand!(Z.Dval)
            rand!(Z.Lval)

            # Direct inv=true
            Y1 = copy(Y)
            Z1 = copy(Z)
            T1 = similar(F)
            amari!(T1, Y1, Z1, F, S; inv=true)

            # Roundtrip: fisher inv=true on inputs, amari inv=false, fisher inv=true on output, negate
            Y2 = copy(Y)
            Z2 = copy(Z)
            T2 = similar(F)
            fisher!(Y2, F, S; inv=true)
            fisher!(Z2, F, S; inv=true)
            amari!(T2, Y2, Z2, F, S; inv=false)
            fisher!(T2, F, S; inv=true)

            @test triangular(T1) ≈ triangular(T2)
        end
    end

    M = SparseMatrixCSC{BigFloat}(readmatrix("nos4"))

    for UPLO in (:L, :U)
        F = cholesky!(ChordalCholesky{UPLO}(M))
        S = selinv!(copy(F))

        Y = similar(F)
        rand!(Y.Dval)
        rand!(Y.Lval)

        Z = similar(F)
        rand!(Z.Dval)
        rand!(Z.Lval)

        # Direct inv=true
        Y1 = copy(Y)
        Z1 = copy(Z)
        T1 = similar(F)
        amari!(T1, Y1, Z1, F, S; inv=true)

        # Roundtrip
        Y2 = copy(Y)
        Z2 = copy(Z)
        T2 = similar(F)
        fisher!(Y2, F, S; inv=true)
        fisher!(Z2, F, S; inv=true)
        amari!(T2, Y2, Z2, F, S; inv=false)
        fisher!(T2, F, S; inv=true)

        @test triangular(T1) ≈ triangular(T2)
    end

    M = SparseMatrixCSC{Float64}(readmatrix("nos4"))
    F = cholesky!(ChordalCholesky(M))
    S = selinv!(copy(F))
    Y = copy(F)
    Z = copy(F)
    T = similar(F)
    @inferred amari!(T, Y, Z, F, S; inv=true)
    # @test_call target_modules = (CliqueTrees,) amari!(T, Y, Z, F, S; inv=true)
    # @test_opt target_modules = (CliqueTrees,) amari!(T, Y, Z, F, S; inv=true)
end
