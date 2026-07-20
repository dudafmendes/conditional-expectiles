@testset "Likelihood" begin
    Random.seed!(123)

    true_model = GARCHModel(0.1, [0.05], [0.9], nothing, Normal())
    y, _ = simulate(true_model, 1000)

    θ_true = [0.1, 0.05, 0.9]
    θ_bad  = [0.5, 0.4, 0.1]

    nll_true = garch_negloglik(θ_true, y, 1, 1, false, Normal())
    nll_bad  = garch_negloglik(θ_bad, y, 1, 1, false, Normal())

    @test isfinite(nll_true)
    @test isfinite(nll_bad)
    @test nll_true < nll_bad
end

@testset "Likelihood is ForwardDiff-compatible" begin
    Random.seed!(1)
    y = randn(50)
    θ = [0.1, 0.1, 0.8]

    f(x) = garch_negloglik(x, y, 1, 1, false, Normal())
    g = ForwardDiff.gradient(f, θ)

    @test length(g) == 3
    @test all(isfinite, g)
end
