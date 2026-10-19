using Test

@static if VERSION >= v"1.12"
    @testset "Code Quality (Aqua.jl)" begin
        include("aqua.jl")
    end
end

#=
@testset "Code Quality (JET.jl)" begin
    include("jet.jl")
end
=#

@testset "Core" begin
    include("core.jl")
end

@testset "BT dynamic program" begin
    include("btdp.jl")
end

@testset "Separator dynamic program" begin
    include("sepdp.jl")
    include("sepenum.jl")
    include("pmcenum.jl")
    include("septw.jl")
end

@testset "Contraction-recursive treewidth" begin
    include("rtw.jl")
end

@testset "Branch decompositions" begin
    include("branch.jl")
end

@testset "Multifrontal" begin
    include("multifrontal.jl")
end

@testset "Differential" begin
    include("differential.jl")
end
