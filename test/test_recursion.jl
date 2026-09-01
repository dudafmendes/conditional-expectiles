@testset "Variance recursion and residuals" begin
    y = [1.0, -2.0, 0.5]
    model = GARCHModel(0.1, [0.2], [0.3], nothing, Normal())

    σ2 = garch_variance(y, model)

    # Gao--Song equation (3.19) retains the intercept tail ω/(1-β).
    intercept_tail = 0.1 / (1 - 0.3)
    # h1 = intercept_tail
    # h2 = 0.1 + 0.2*y1^2 + 0.3*h1
    # h3 = 0.1 + 0.2*y2^2 + 0.3*h2
    @test isapprox(σ2[1], intercept_tail; atol=1e-10)
    @test isapprox(σ2[2], 0.1 + 0.2*y[1]^2 + 0.3*σ2[1]; atol=1e-10)
    @test isapprox(σ2[3], 0.1 + 0.2*y[2]^2 + 0.3*σ2[2]; atol=1e-10)

    ε = residuals(y, model)
    @test isapprox(ε[1], y[1] / sqrt(σ2[1]); atol=1e-10)
    @test isapprox(ε[2], y[2] / sqrt(σ2[2]); atol=1e-10)
    @test isapprox(ε[3], y[3] / sqrt(σ2[3]); atol=1e-10)
end

@testset "GJR asymmetry" begin
    y = [1.0, -2.0, 0.5]
    model = GARCHModel(0.1, [0.2], [0.3], 0.4, Normal())

    σ2 = garch_variance(y, model)

    intercept_tail = 0.1 / (1 - 0.3)
    @test isapprox(σ2[1], intercept_tail; atol=1e-10)
    @test isapprox(σ2[2], 0.1 + 0.2*y[1]^2 + 0.3*σ2[1]; atol=1e-10)
    @test isapprox(
        σ2[3], 0.1 + 0.2*y[2]^2 + 0.3*σ2[2] + 0.4*y[2]^2; atol=1e-10,
    )
end

@testset "GJR asymmetry" begin
    y = [1.0, -2.0, 0.5]
    model = GARCHModel(0.1, [0.2], [0.3], 0.4, Normal())

    σ2 = garch_variance(y, model)

    intercept_tail = 0.1 / (1 - 0.3)
    @test isapprox(σ2[1], intercept_tail; atol=1e-10)
    @test isapprox(σ2[2], 0.1 + 0.2*y[1]^2 + 0.3*σ2[1]; atol=1e-10)
    @test isapprox(
        σ2[3], 0.1 + 0.2*y[2]^2 + 0.3*σ2[2] + 0.4*y[2]^2; atol=1e-10,
    )
end
