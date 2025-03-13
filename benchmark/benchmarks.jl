using BenchmarkTools
using CliqueTrees
using MatrixMarket
using SparseArrays
using SuiteSparseMatrixCollection

const SUITE = BenchmarkGroup()

names = ("mycielskian4", "dwt_59", "can_292", "lshp3466", "wing", "144", "333SP")
ssmc = ssmc_db()

for name in names
    # download graph
    matrix::SparseMatrixCSC{Float64, Int32} = mmread(
        joinpath(fetch_ssmc(ssmc[ssmc.name .== name, :]; format = "MM")[1], "$(name).mtx")
    )

    # construct permutation
    order, index = permutation(matrix; alg = NodeND())

    for alg in (BFS, MCS, LexBFS, RCM)
        SUITE[name]["permutation"][string(alg)] = @benchmarkable permutation(
            $matrix; alg = $(alg())
        )
    end

    order, index = permutation(matrix; alg = NodeND())

    SUITE["eliminationtree"] = @benchmarkable eliminationtree($matrix; alg = $order)

    for snd in (Nodal, Fundamental, Maximal)
        SUITE[name]["supernodetree"][string(snd)] = @benchmarkable supernodetree(
            $matrix; alg = $order, snd = $(snd())
        )
        SUITE[name]["cliquetree"][string(snd)] = @benchmarkable cliquetree(
            $matrix; alg = $order, snd = $(snd())
        )
        SUITE[name]["eliminationgraph"][string(snd)] = @benchmarkable eliminationgraph(
            $matrix; alg = $order, snd = $(snd())
        )
    end
end
