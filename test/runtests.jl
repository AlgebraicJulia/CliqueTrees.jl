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

@testset "Multifrontal" begin
    include("multifrontal.jl")
end

@testset "Differential" begin
    include("differential.jl")
end
