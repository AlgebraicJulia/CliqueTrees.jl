# apsp_gpu (one call, original labels) against the CPU closure Matrix(F) (original labels).
#
#   julia --project=. test/test_api.jl
#
# Every output (:device, :host), the sources variant, several GPUs (one device repeated) and tiny
# relabelling buffers (many blocks, a ragged last one), on grids, a random graph and a directed graph,
# for min-plus (Float32, Float64), max-plus, max-min and plus-times; then the input errors.
haskey(ENV, "SEMIRINGGPU_TUNE_FILE") || (ENV["SEMIRINGGPU_TUNE_FILE"] = tempname())    # keep tuning on a shared GPU out of the depot
include(joinpath(@__DIR__, "..", "src", "SemiringGPU.jl"))
include(joinpath(@__DIR__, "graphs.jl"))
using .SemiringGPU
using .SemiringGPU: Semiring
using .Semiring: MinPlus, MaxPlus, MaxMin, PlusProd, ChordalSLU, mlu
using CUDA, LinearAlgebra, SparseArrays, Random, Test
Random.seed!(7)

# undirected, average degree d: several components and isolated vertices (unreachable pairs)
function randgraph(n, d, ::Type{T}) where {T}
    m = round(Int, d * n / 2)
    I = rand(1:n, m); J = rand(1:n, m); keep = I .!= J
    I = I[keep]; J = J[keep]; V = T.(rand(1:100, length(I)))
    return sparse(vcat(I, J), vcat(J, I), vcat(V, V), n, n, min)
end

# directed and strongly connected: a directed cycle, plus chords with a different weight each way
function digraph(n, ::Type{T}) where {T}
    I = collect(1:n); J = [mod1(i + 1, n) for i in 1:n]; V = T.(rand(1:100, n))
    for _ in 1:n
        a, b = rand(1:n, 2)
        a == b || (append!(I, (a, b)); append!(J, (b, a)); append!(V, T.(rand(1:100, 2))))
    end
    return sparse(I, J, V, n, n, min)
end

closure(s, A) = Matrix(mlu(s, A))                               # CPU: A*[i, j] in the labels of A
same(s, x, y) = s isa PlusProd ? isapprox(x, y; rtol = 1e-10) : isequal(x, y)
errorof(f) = try f(); nothing catch e; e end

const CASES = [
    ("grid 30×30 MinPlus Float32", MinPlus(), grid(30, 30, Float32)),
    ("grid3 8³ MinPlus Float64", MinPlus(), grid3(8, Float64)),
    ("random n=1201 MinPlus Float32", MinPlus(), randgraph(1201, 3.0, Float32)),
    ("directed n=601 MinPlus Float32", MinPlus(), digraph(601, Float32)),
    ("grid 25×36 MaxPlus Float32 (weights < 0)", MaxPlus(), -grid(25, 36, Float32)),
    ("grid3 9³ MaxMin Float32", MaxMin(), grid3(9, Float32)),
    ("grid 20×20 MaxMin Int32", MaxMin(), SparseMatrixCSC{Int32, Int}(grid(20, 20, Float32))),
    ("grid 30×30 PlusProd Float64 (weights / 500)", PlusProd(), grid(30, 30, Float64) ./ 500),
]

@testset "apsp_gpu" begin
    d = CUDA.device()

    @testset "$name" for (name, s, A) in CASES
        n = size(A, 1); T = eltype(A)
        C = closure(s, A)
        @test ChordalSLU(s, A).rperm != 1:n                     # the labels really are permuted

        # whole closure, both outputs
        D = apsp_gpu(A; semiring = s)
        @test D isa CuMatrix{T} && size(D) == (n, n)
        @test same(s, Array(D), C)
        H = apsp_gpu(A; semiring = s, output = :host)
        @test H isa Matrix{T}
        @test same(s, H, C)

        # tiny buffers: 1 row / column per block, and 7 per block (ragged last block)
        for b in (1, 7n + 3)
            @test same(s, Array(apsp_gpu(A; semiring = s, buffer = b)), C)
            @test same(s, apsp_gpu(A; semiring = s, output = :host, buffer = b), C)
        end

        # rows for some sources: unsorted, repeated
        src = [n, 3, 17, 3, 1, n ÷ 2, 17]
        X = apsp_gpu(A, src; semiring = s)
        @test X isa CuMatrix{T} && size(X) == (length(src), n)
        @test same(s, Array(X), C[src, :])
        @test same(s, apsp_gpu(A, src; semiring = s, output = :host), C[src, :])
        @test same(s, apsp_gpu(A, [5]; semiring = s, output = :host), C[5:5, :])

        # several GPUs (one device three times; n is not a multiple of 3 for most cases)
        @test same(s, apsp_gpu(A; semiring = s, devices = [d, d, d], output = :host), C)
        @test same(s, apsp_gpu(A; semiring = s, devices = [d, d, d], output = :host, buffer = 5), C)

        for b in (0, 5n + 1)
            blocks = apsp_gpu(A; semiring = s, devices = [d, d, d], buffer = b)
            @test length(blocks) == 3
            @test sort(reduce(vcat, first.(blocks))) == 1:n         # every source exactly once
            @test all(B isa CuMatrix{T} && size(B) == (length(sg), n) for (sg, B) in blocks)
            @test all(same(s, Array(B), C[sg, :]) for (sg, B) in blocks)
        end
    end

    @testset "label convention: D[i, j] = A*[i, j], the path i → j" begin
        # arcs 1 → 2 (1), 2 → 3 (2), 3 → 1 (4)
        A = sparse([1, 2, 3], [2, 3, 1], Float32[1, 2, 4], 3, 3)
        D = apsp_gpu(A; output = :host)
        @test D == Float32[0 1 3; 6 0 2; 4 5 0]
        @test apsp_gpu(A, [3, 1]; output = :host) == Float32[4 5 0; 0 1 3]
        @test apsp_gpu(A; devices = [d, d], output = :host) == D
        A = digraph(301, Float32); C = closure(MinPlus(), A)
        @test C != C'                                            # not symmetric, so the orientation is tested
        @test apsp_gpu(A; output = :host) == C
    end

    @testset "settings and edge cases" begin
        A = grid(30, 30, Float32); C = closure(MinPlus(), A)
        @test with_config(() -> apsp_gpu(A; output = :host); merge = 1, skip_fill = false, layered_min_rows = 1) == C
        @test with_config(() -> apsp_gpu(A, [9, 2]; output = :host); merge = 1) == C[[9, 2], :]
        @test size(apsp_gpu(A, Int[])) == (0, 900)
        @test size(apsp_gpu(A, Int[]; output = :host)) == (0, 900)
        @test apsp_gpu(A, 1:900; output = :host) == C                       # a range of sources

        E = spzeros(Float32, 0, 0)
        @test size(apsp_gpu(E)) == (0, 0) && size(apsp_gpu(E; output = :host)) == (0, 0)
        @test isempty(apsp_gpu(E; devices = [d, d]))
        V = spzeros(Float32, 1, 1)                                          # one vertex
        @test apsp_gpu(V; output = :host) == fill(0f0, 1, 1)
        P2 = sparse([1, 2], [2, 1], Float32[3, 3], 2, 2)                    # two vertices, three blocks (one empty)
        @test apsp_gpu(P2; devices = [d, d, d], output = :host) == Float32[0 3; 3 0]
        blocks = apsp_gpu(P2; devices = [d, d, d])
        @test sum(length ∘ first, blocks) == 2 && any(isempty ∘ first, blocks)
    end

    @testset "errors" begin
        A = grid(10, 10, Float32)
        R = sprand(Float32, 3, 4, 0.5)
        @test_throws ArgumentError apsp_gpu(R)                              # not square
        @test_throws ArgumentError apsp_gpu(R, [1])
        @test_throws ArgumentError apsp_gpu(R; devices = [d, d])
        for I in (Int32, Int64)                                              # min-plus: the integer infinity overflows
            Ai = SparseMatrixCSC{I, Int}(A)
            @test_throws ArgumentError apsp_gpu(Ai)
            @test_throws ArgumentError apsp_gpu(Ai, [1])
            @test_throws ArgumentError apsp_gpu(Ai; devices = [d, d])
            @test occursin("overflows", sprint(showerror, errorof(() -> apsp_gpu(Ai))))
        end
        @test_throws ArgumentError apsp_gpu(SparseMatrixCSC{Float16, Int}(A))  # 2-byte elements
        @test_throws ArgumentError apsp_gpu(A; output = :disk)
        @test_throws ArgumentError apsp_gpu(A, [1]; output = :disk)
        @test_throws ArgumentError apsp_gpu(A; devices = CuDevice[])
        @test_throws ArgumentError apsp_gpu(A; devices = [0])
        @test_throws ArgumentError apsp_gpu(A, [0])
        @test_throws ArgumentError apsp_gpu(A, [1, 101])
        # a directed path: its strongly connected components reach one another
        Pd = sparse([1, 2], [2, 3], Float32[1, 2], 3, 3)
        e = errorof(() -> apsp_gpu(Pd))
        @test e isa ArgumentError && occursin("strongly connected", e.msg)
        @test_throws ArgumentError apsp_gpu(Pd, [1])
        @test_throws ArgumentError apsp_gpu(Pd; devices = [d, d])
        # an n × n result larger than the GPU: refused before any work, with advice
        n = isqrt(CUDA.total_memory() ÷ 4) + 1000
        L = spdiagm(1 => ones(Float32, n - 1)); L = L + L'
        e = errorof(() -> apsp_gpu(L))
        @test e isa ErrorException && occursin("not enough GPU memory", e.msg) && occursin("devices", e.msg) && occursin("sources", e.msg)
        e = errorof(() -> apsp_gpu(L; devices = [d, d]))                    # both blocks on the same device
        @test e isa ErrorException && occursin("not enough GPU memory", e.msg)
    end
end
