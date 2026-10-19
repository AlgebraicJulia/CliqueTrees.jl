# test/septw.jl, test/sepenum.jl, test/sepdp.jl, test/rtw.jl of CliqueTrees.jl-gsep
using Test
const T = "/home/itaykadosh/Projects/GATAS/CliqueTrees.jl-gsep/test"
@testset "gsep t$(Threads.nthreads())" begin
    for f in ("septw.jl", "sepenum.jl", "sepdp.jl", "rtw.jl")
        t = @elapsed @testset "$f" begin
            include(joinpath(T, f))
        end
        println("== $f done in $(round(t; digits = 1)) s"); flush(stdout)
    end
end
