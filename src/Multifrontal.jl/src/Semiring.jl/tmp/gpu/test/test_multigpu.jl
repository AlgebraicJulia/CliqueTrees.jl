# Multi-GPU closure (src/multigpu.jl): row_blocks, and closure_multigpu! against closure_gpu! with one
# device repeated (several blocks on one GPU, as test/sanitize_small.jl and bench/multigpu.jl DEVICES=0,0
# do), preallocated `out` blocks, and the timer. With more than one GPU, also over all of them.
#
#   julia --project=. -t auto test/test_multigpu.jl
haskey(ENV, "SEMIRINGGPU_TUNE_FILE") || (ENV["SEMIRINGGPU_TUNE_FILE"] = tempname())
include(joinpath(@__DIR__, "..", "src", "SemiringGPU.jl"))
include(joinpath(@__DIR__, "graphs.jl"))
using .SemiringGPU
using .SemiringGPU: Semiring, row_blocks
using .Semiring: MinPlus, ChordalSLU, mlu
using CUDA, SparseArrays, Random, Test

Random.seed!(23)

# the error inside a failed task of an @sync block
rootcause(e) = e isa CompositeException ? rootcause(first(e.exceptions)) : e isa TaskFailedException ? rootcause(e.task.exception) : e

# the contract of row_blocks(n, ng): ng contiguous ranges, in order, covering 1:n exactly, lengths
# differing by at most one with the longer ones first; for n < ng the last ng - n are empty
function blocks_ok(n, ng)
    R = row_blocks(n, ng)
    lens = length.(R)
    ok = R isa Vector{UnitRange{Int}} && length(R) == ng &&
         first(R[1]) == 1 && last(R[end]) == n &&
         all(first(R[g + 1]) == last(R[g]) + 1 for g in 1:(ng - 1)) &&
         vcat(collect.(R)...) == collect(1:n) &&
         sum(lens) == n && maximum(lens) - minimum(lens) <= 1 && issorted(lens; rev = true) &&
         lens == [div(n, ng) + (g <= rem(n, ng)) for g in 1:ng]
    ok || println("  row_blocks($n, $ng) = $R")
    return ok
end

# disconnected, with isolated vertices (unreachable pairs)
function pieces(n, ::Type{T}) where {T}
    I = Int[]; J = Int[]
    for _ in 1:2n
        u = rand(1:n); v = clamp(u + rand(-10:10), 1, n)
        u != v && rand() < 0.6 && (push!(I, u); push!(J, v))
    end
    W = T.(rand(0:100, length(I)))
    return sparse(vcat(I, J), vcat(J, I), vcat(W, W), n, n, min)
end

path(n, ::Type{T}) where {T} = (I = collect(1:(n - 1)); J = I .+ 1; W = T.(rand(1:100, n - 1));
                                sparse(vcat(I, J), vcat(J, I), vcat(W, W), n, n))

s = MinPlus()
d = CUDA.device()

@testset "test_multigpu" begin

@testset "row_blocks" begin
    @test all(blocks_ok(n, ng) for n in 0:50, ng in 1:9)
    @test all(blocks_ok(n, ng) for (n, ng) in ((1, 64), (63, 64), (65, 64), (10^6, 7), (10^6 + 3, 8), (12_345, 1)))
    @test row_blocks(10, 3) == [1:4, 5:7, 8:10]
    @test row_blocks(2, 3) == [1:1, 2:2, 3:2]          # n < ng: trailing empty blocks
    @test row_blocks(0, 2) == [1:0, 1:0]
    @test row_blocks(7, 1) == [1:7]
    @test_throws Exception row_blocks(5, 0)            # at least one block
end

graphs = [("grid3 9³", grid3(9, Float32)), ("grid 30×30", grid(30, 30, Float32)), ("pieces 500", pieces(500, Float32)),
          ("path 2", path(2, Float32)), ("single vertex", spzeros(Float32, 1, 1))]

@testset "closure_multigpu! == closure_gpu! rows: $name" for (name, A) in graphs
    n = size(A, 1)
    F = ChordalSLU(s, A); copyto!(F, A)
    P = FactorPlan(F; large = 16, graph = false); factorize!(P)
    G = GPUSLU(P; large = 64); precompute_ops!(G)
    D = Array(closure_gpu(G))                          # elimination coordinates: D[i, j] = A*[p[i], p[j]]
    p = Array(G.rperm)
    H = similar(D); H[p, p] = D
    @test isequal(H, Matrix(mlu(s, A)))                # the single-GPU reference itself

    for devs in ([d], [d, d], [d, d, d])
        ng = length(devs)
        rows = row_blocks(n, ng)
        MG = MultiGPUSLU(P; devices = devs, large = 64)
        @test MG.n == n && MG.devices == devs && length(MG.parts) == ng

        blocks = closure_multigpu!(MG)
        @test length(blocks) == ng
        @test [b[1] for b in blocks] == rows
        @test all(b[2] isa CuMatrix{Float32} && size(b[2]) == (length(b[1]), n) for b in blocks)
        @test all(CUDA.device(blocks[g][2]) == devs[g] for g in 1:ng)
        @test all(isequal(Array(X), D[r, :]) for (r, X) in blocks)
        @test isequal(vcat([Array(X) for (_, X) in blocks]...), D)      # the blocks are the whole closure

        # without precomputed operators, and from a host factor (MultiGPUSLU(F, factor))
        MG2 = MultiGPUSLU(P; devices = devs, large = 64, ops = false)
        @test all(isequal(Array(X), D[r, :]) for (r, X) in closure_multigpu!(MG2))
        hfactor = map(Array, (P.LD, P.LL, P.UD, P.UL))
        MG3 = MultiGPUSLU(P.F, hfactor; devices = devs, large = 64)
        @test all(isequal(Array(X), D[r, :]) for (r, X) in closure_multigpu!(MG3))

        # preallocated blocks, filled with garbage first; also with the layered sweep, which does not
        # fill the whole result (skip_fill)
        for kw in ((;), (layered_min_rows = 1,), (layered_min_rows = 1, skip_fill = false))
            out = [CUDA.fill(NaN32, length(r), n) for r in rows]
            got = with_config(() -> closure_multigpu!(MG; out); kw...)
            @test all(got[g][2] === out[g] for g in 1:ng)              # computed in place
            @test [b[1] for b in got] == rows
            @test all(isequal(Array(out[g]), D[rows[g], :]) for g in 1:ng)
        end
        again = closure_multigpu!(MG; out = [b[2] for b in blocks])     # reuse the blocks of an earlier call
        @test all(isequal(Array(X), D[r, :]) for (r, X) in again)

        # blocks of the wrong size are rejected
        bad = [CUDA.zeros(Float32, length(r) + 1, n) for r in rows]
        err = try closure_multigpu!(MG; out = bad); nothing catch e; rootcause(e) end
        @test err isa AssertionError

        # the timer: device g's wall time
        timer = fill(-1.0, ng)
        closure_multigpu!(MG; timer)
        @test all(t -> isfinite(t) && t >= 0, timer)
        @test all(timer[g] > 0 for g in 1:ng if !isempty(rows[g]))
        td = Dict{Int, Float64}()
        res = closure_multigpu!(MG; timer = td, out = [CUDA.fill(NaN32, length(r), n) for r in rows])
        @test sort(collect(keys(td))) == 1:ng
        @test all(isequal(Array(X), D[r, :]) for (r, X) in res)
    end
end

@testset "all GPUs" begin
    devs = collect(CUDA.devices())
    if length(devs) < 2
        println("  one GPU: the multi-device case was run with the device repeated only")
    else
        A = grid3(10, Float32); n = size(A, 1)
        F = ChordalSLU(s, A); copyto!(F, A)
        P = FactorPlan(F; large = 16, graph = false); factorize!(P)
        G = GPUSLU(P; large = 64); D = Array(closure_gpu(G))
        for ds in (devs, vcat(devs, devs[1:1]))
            MG = MultiGPUSLU(P; devices = ds, large = 64)
            blocks = closure_multigpu!(MG)
            @test all(CUDA.device(blocks[g][2]) == ds[g] for g in eachindex(ds))
            @test all(isequal(Array(X), D[r, :]) for (r, X) in blocks)
            CUDA.device!(d)
        end
    end
end

end # test_multigpu
