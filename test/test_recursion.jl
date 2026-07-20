@testset "Variance recursion and residuals" begin
    y = [1.0, -2.0, 0.5]
    model = GARCHModel(0.1, [0.2], [0.3], nothing, Normal())

    σ2 = garch_variance(y, model)

    # manual recursion:
    # h1 = 0.1
    # h2 = 0.1 + 0.2*y1^2 + 0.3*h1 = 0.1 + 0.2*1 + 0.3*0.1 = 0.33
    # h3 = 0.1 + 0.2*y2^2 + 0.3*h2 = 0.1 + 0.8 + 0.099 = 0.999
    @test isapprox(σ2[1], 0.1; atol=1e-10)
    @test isapprox(σ2[2], 0.33; atol=1e-10)
    @test isapprox(σ2[3], 0.999; atol=1e-10)

    ε = residuals(y, model)
    @test isapprox(ε[1], y[1] / sqrt(σ2[1]); atol=1e-10)
    @test isapprox(ε[2], y[2] / sqrt(σ2[2]); atol=1e-10)
    @test isapprox(ε[3], y[3] / sqrt(σ2[3]); atol=1e-10)
end

@testset "GJR asymmetry" begin
    y = [1.0, -2.0, 0.5]
    model = GARCHModel(0.1, [0.2], [0.3], 0.4, Normal())

    σ2 = garch_variance(y, model)

    # h1 = 0.1
    # h2 = 0.1 + 0.2*1^2 + 0.3*0.1 + 0     = 0.33   (lagged y1 positive)
    # h3 = 0.1 + 0.2*(-2)^2 + 0.3*0.33 + 0.4*4 = 2.599
    @test isapprox(σ2[1], 0.1; atol=1e-10)
    @test isapprox(σ2[2], 0.33; atol=1e-10)
    @test isapprox(σ2[3], 2.599; atol=1e-10)
end

@testset "GJR asymmetry" begin
    y = [1.0, -2.0, 0.5]
    model = GARCHModel(0.1, [0.2], [0.3], 0.4, Normal())

    σ2 = garch_variance(y, model)

    # h1 = 0.1
    # h2 = 0.1 + 0.2*1^2 + 0.3*0.1 + 0     = 0.33   (lagged y1 positive)
    # h3 = 0.1 + 0.2*(-2)^2 + 0.3*0.33 + 0.4*4 = 2.499
    @test isapprox(σ2[1], 0.1; atol=1e-10)
    @test isapprox(σ2[2], 0.33; atol=1e-10)
    @test isapprox(σ2[3], 2.599; atol=1e-10)
end
