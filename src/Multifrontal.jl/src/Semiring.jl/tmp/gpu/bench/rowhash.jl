# Position-sensitive reference hashes of the all-pairs distance matrix, per row, for checking other codes'
# output entry by entry (the sum-and-count checksum does not see where an entry is).
#
#   OUTDIR=dir julia --project=. -t 16 bench/rowhash.jl graph ...
#
# With 1-based original labels i (row, source) and j (column, target), all arithmetic in UInt64 modulo 2^64:
#   v(i, j)    = d(i, j) as an integer if finite (all distances are integers), else 0xFFFFFFFF
#   rowhash(i) = Σ_j v(i, j) * (j * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019)
#   fullhash   = Σ_i rowhash(i) * (i * 0xD6E8FEB86659FD93 + 1)
# Writes dir/<graph>.rowhash (line i: rowhash(i) in hex) and prints fullhash and the sum-and-count checksum.
include(joinpath(@__DIR__, "bench_apsp.jl"))
using CUDA, Printf

const S = SemiringGPU
const OUTDIR = get(ENV, "OUTDIR", "rowhash")
const K1 = 0x9E3779B97F4A7C15; const K2 = 0x632BE59BD9B4E019; const K3 = 0xD6E8FEB86659FD93
mkpath(OUTDIR)
val(x) = isfinite(x) ? UInt64(round(x)) : UInt64(0xFFFFFFFF)

for name in filter(a -> !startswith(a, "--"), ARGS)
    A = read_mtx(joinpath(MTX, name * ".mtx")); n = size(A, 1)
    D = S.apsp_gpu(A); CUDA.synchronize()
    h = CUDA.zeros(UInt64, n)
    b = max(1, 2^29 ÷ n)                                  # columns per chunk (≤ 4 GB of UInt64)

    for j0 in 1:b:n
        J = j0:min(n, j0 + b - 1)
        w = CuVector{UInt64}(UInt64.(J) .* K1 .+ K2)
        h .+= vec(sum(val.(view(D, :, J)) .* w'; dims = 2))
    end

    rh = Array(h)
    full = sum(rh[i] * (UInt64(i) * K3 + one(UInt64)) for i in 1:n)
    chk = (mapreduce(x -> isfinite(x) ? Float64(x) : 0.0, +, D), count(!isfinite, D))
    open(joinpath(OUTDIR, name * ".rowhash"), "w") do io
        for x in rh; println(io, string(x; base = 16, pad = 16)); end
    end
    @printf("%-18s n=%7d fullhash %016x checksum %.6e/%d\n", name, n, full, chk...)
    CUDA.unsafe_free!(D); GC.gc(); CUDA.reclaim()
end
