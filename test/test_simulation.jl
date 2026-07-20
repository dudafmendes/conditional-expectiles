@testset "Simulation" begin
    Random.seed!(1234)

    model = GARCHModel(0.1, [0.05], [0.9], nothing, Normal())
    y, σ2 = simulate(model, 500)

    @test length(y) == 500
    @test length(σ2) == 500
    @test all(σ2 .> 0)

    # standardized residuals should have variance near 1 in large samples
    η = y ./ sqrt.(σ2)
    @test abs(mean(η)) < 0.15
    @test abs(var(η) - 1.0) < 0.20
end

@testset "Simulation with Student-t" begin
    Random.seed!(1234)

    model = GARCHModel(0.1, [0.05], [0.9], nothing, TDist(10))
    y, σ2 = simulate(model, 500)

    @test length(y) == 500
    @test all(σ2 .> 0)

    η = y ./ sqrt.(σ2)
    @test isfinite(mean(η))
    @test isfinite(var(η))
end
