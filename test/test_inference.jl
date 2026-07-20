@testset "Parameter variance / inference" begin
    Random.seed!(606)

    true_model = GARCHModel(0.1, [0.05], [0.9], nothing, Normal())
    y, _ = simulate(true_model, 1500)
    fit = GARCHModels.estimate(y, 1, 1, gjr=false, dist=Normal())

    V = GARCHModels.garch_parameter_variance(y, fit)

    @test size(V) == (3, 3)
    @test issymmetric(V)
    @test all(isfinite, V)

    eigvals_V = eigvals(Symmetric(V))
    @test minimum(eigvals_V) > -1e-8
end

@testset "Parameter variance scaling" begin
    Random.seed!(707)

    true_model = GARCHModel(0.1, [0.05], [0.9], nothing, Normal())
    y, _ = simulate(true_model, 1200)
    fit = GARCHModels.estimate(y, 1, 1, gjr=false, dist=Normal())

    V1 = GARCHModels.garch_parameter_variance(y, fit)
    V2 = GARCHModels.garch_parameter_variance(y, fit; rootn=true)

    @test size(V1) == size(V2)
    @test norm(V2 / length(y) - V1) < 1e-6 || norm(V2 / length(y) - V1) / max(norm(V1), 1e-12) < 1e-6
end
