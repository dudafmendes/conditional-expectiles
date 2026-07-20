@testset "Types and basic interface" begin
    model = GARCHModel(0.1, [0.2], [0.7], nothing, Normal())

    @test model.ω == 0.1
    @test model.α == [0.2]
    @test model.β == [0.7]
    @test isnothing(model.γ)
    @test model.dist isa Normal

    v = convert(Vector{Float64}, model)
    @test v == [0.1, 0.2, 0.7]

    model_gjr = GARCHModel(0.1, [0.1], [0.8], 0.05, TDist(10))
    v2 = convert(Vector{Float64}, model_gjr)
    @test v2 == [0.1, 0.1, 0.8, 0.05]
end
