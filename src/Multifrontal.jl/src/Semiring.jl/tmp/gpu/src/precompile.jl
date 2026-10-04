# ===== precompilation =====
#
# Included only when SemiringGPU is loaded as a package (PACKAGED). While the package precompiles,
# the solver runs on small graphs, so that the host code and the GPU kernels' inference results go
# into the package image. GPUCompiler can then also cache each kernel's machine code on disk after its
# first compilation, when its disk cache is enabled in the environment:
#
#   using GPUCompiler; GPUCompiler.enable_disk_cache!()      (once; a preference of the environment)
#
# and later sessions load the kernels instead of compiling them. The workload needs a working GPU at
# precompile time; without one only the host code is precompiled.

using PrecompileTools: @setup_workload, @compile_workload

# a 3D grid graph with integer weights (deterministic)
function precompile_grid(nx::Integer, ::Type{T}) where {T}
    id(i, j, l) = i + (j - 1) * nx + (l - 1) * nx^2
    I = Int[]; J = Int[]; V = T[]

    for l in 1:nx, j in 1:nx, i in 1:nx, d in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        a, b, c = i + d[1], j + d[2], l + d[3]
        (a <= nx && b <= nx && c <= nx) || continue
        w = T(1 + (7 * i + 11 * j + 13 * l + 3 * d[2] + 5 * d[3]) % 97); u = id(i, j, l); v = id(a, b, c)
        append!(I, (u, v)); append!(J, (v, u)); append!(V, (w, w))
    end

    return SparseArrays.sparse(I, J, V, nx^3, nx^3)
end

function precompile_workload()
    s = Semiring.MinPlus()

    for T in (Float32, Float64)
        A = precompile_grid(8, T)
        # the public API, and the pipeline it runs on graphs of production size: small thresholds make
        # this graph's top fronts take the GPU factorization, the dense solve paths and the operators
        apsp_gpu(A; semiring = s)
        apsp_gpu(A, [1, 5, 9]; semiring = s)
        # the orderings AutoOrder picks on larger graphs: BFSND (lattices) and HubAMF (graphs with hubs)
        L = precompile_grid(12, T); m = size(L, 1)
        Semiring.ssymbolic(L; alg = CliqueTrees.BFSND())
        H = L + SparseArrays.sparse(vcat(fill(1, m - 1), 2:m), vcat(2:m, fill(1, m - 1)), ones(T, 2m - 2), m, m)
        Semiring.ssymbolic(H; alg = CliqueTrees.AutoOrder())
        F = Semiring.ChordalSLU(s, A); copyto!(F, A)
        P = FactorPlan(F; large = 16, graph = false, nstreams = 8); factorize!(P)
        G = GPUSLU(P; large = 64); precompute_ops!(G)
        n = size(A, 1); D = CuMatrix{T}(undef, n, n)
        closure_gpu!(D, G)
        with_config(() -> closure_gpu!(D, G); layered_min_rows = 1)
        sssp_gpu!(CuMatrix{T}(undef, 3, n), G, CuVector([1, 2, 3]))
    end

    # every GEMM kernel the autotuner may pick, for min-plus Float32
    V = Float32
    A = CUDA.ones(V, 300, 200); B = CUDA.ones(V, 200, 200)

    for n in (16, 32, 64, 128, 200), ow in (false, true)
        C = CUDA.zeros(V, 300, n)

        for c in gemm_candidates(n, false, min3_ok(s, V))
            launch!(s, C, A, view(B, :, 1:n), c, Val(ow))
        end
    end

    CUDA.synchronize()
    return
end

@setup_workload begin
    @compile_workload begin
        CUDA.functional() && with_config(precompile_workload; gemm_tune = false)
    end

    CUDA.functional() && CUDA.synchronize()
    reset_caches!()                  # no device buffers or profiles in the image
end
