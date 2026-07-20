@testset "Estimation returns valid model" begin
    Random.seed!(202)

    true_model = GARCHModel(0.1, [0.05], [0.9], nothing, Normal())
    y, _ = simulate(true_model, 800)

    fit = GARCHModels.estimate(y, 1, 1, gjr=false, dist=Normal())

    @test fit isa GARCHModel
    @test fit.ω > 0
    @test length(fit.α) == 1
    @test length(fit.β) == 1
    @test fit.α[1] ≥ 0
    @test fit.β[1] ≥ 0
end


@testset "estimate and estimate_fixed agree" begin
    Random.seed!(303)

    true_model = GARCHModel(0.1, [0.08], [0.85], nothing, Normal())
    y, _ = simulate(true_model, 1000)

    fit1 = GARCHModels.estimate(y, 1, 1, gjr=false, dist=Normal())
    fit2 = GARCHModels.estimate_fixed(y, 1, 1, gjr=false, dist=Normal(); hot_start=true)

    @test isapprox(fit1.ω,    fit2.ω;    atol=1e-2, rtol=1e-1)
    @test isapprox(fit1.α[1], fit2.α[1]; atol=1e-2, rtol=1e-1)
    @test isapprox(fit1.β[1], fit2.β[1]; atol=1e-2, rtol=1e-1)
end

@testset "Parameter recovery in moderate sample" begin
    Random.seed!(404)

    true_model = GARCHModel(0.1, [0.05], [0.9], nothing, Normal())
    y, _ = simulate(true_model, 2000)

    fit = GARCHModels.estimate(y, 1, 1, gjr=false, dist=Normal())

    @test abs(fit.ω - 0.1) < 0.2
    # @test abs(fit.α[1] - 0.05) < 0.10
    # @test abs(fit.β[1] - 0.9) < 0.10
    @test abs(fit.α[1] + fit.β[1] - 0.95) < 0.1

end

@testset "GJR estimation sanity" begin
    Random.seed!(505)

    true_model = GARCHModel(0.1, [0.05], [0.85], 0.05, Normal())
    y, _ = simulate(true_model, 2500)

    fit = GARCHModels.estimate(y, 1, 1, gjr=true, dist=Normal())

    @test fit.ω > 0
    @test fit.α[1] ≥ 0
    @test fit.β[1] ≥ 0
    @test fit.γ ≥ 0
end
