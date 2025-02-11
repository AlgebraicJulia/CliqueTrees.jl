using Test

@testset "Code Quality (Aqua.jl)" begin
    include("aqua.jl")
end

@testset "Code Quality (JET.jl)" begin
    include("jet.jl")
end

@testset "Core" begin
    include("core.jl")
end
