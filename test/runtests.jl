using Test
using Random
using Distributions
using LinearAlgebra
using Statistics
using ForwardDiff
using ConditionalExpectiles
using ConditionalExpectiles.GARCHModels
using ConditionalExpectiles.Expectiles
using ConditionalExpectiles.GaoSongRisk

@testset "GARCHModels" begin
    include("test_types.jl")
    include("test_recursion.jl")
    include("test_simulation.jl")
    include("test_likelihood.jl")
    include("test_estimation.jl")
    include("test_inference.jl")
    include("test_expectiles.jl")
    include("test_gao_song_risk.jl")
end
