

@testset "Expectiles" begin

    # ------------------------------------------------------------
    # 1. Pure expectile solver
    # ------------------------------------------------------------
    @testset "expectile solver" begin
        x = [-2.0, -1.0, 0.0, 1.0, 2.0]

        # τ = 0.5 should reproduce the mean
        e50 = expectile(x, 0.5)
        @test isapprox(e50, mean(x); atol=1e-10)

        # monotonicity in τ
        e25 = expectile(x, 0.25)
        e75 = expectile(x, 0.75)
        @test e25 < e50 < e75

        # location equivariance
        c = 3.7
        @test isapprox(expectile(x .+ c, 0.8), expectile(x, 0.8) + c; atol=1e-10)

        # positive scale equivariance
        a = 2.5
        @test isapprox(expectile(a .* x, 0.8), a * expectile(x, 0.8); atol=1e-10)

        # constant vector
        xc = fill(2.3, 10)
        @test isapprox(expectile(xc, 0.2), 2.3; atol=1e-12)
        @test isapprox(expectile(xc, 0.8), 2.3; atol=1e-12)
    end

    # ------------------------------------------------------------
    # 2. Expectile first-order condition
    # ------------------------------------------------------------
    @testset "expectile first-order condition" begin
        x = [-3.0, -1.0, 0.5, 2.0, 4.0]
        τ = 0.8
        eτ = expectile(x, τ)

        ψ = 0.0
        for xi in x
            ψ += (xi - eτ) * abs((xi < eτ) - τ)
        end

        @test abs(ψ) < 1e-8
    end

    # ------------------------------------------------------------
    # 3. Auxiliary asymptotic quantities
    # ------------------------------------------------------------
    @testset "aux_expectile_var" begin
        ε = [-1.0, 0.0, 2.0]
        τ = 0.75
        eτ = 1.0

        s2, dψ = aux_expectile_var(ε, eτ, τ)

        # manual calculation using the same formula as in the module
        weights = [abs((ε[i] < eτ) - τ) for i in eachindex(ε)]
        s2_manual = mean(((ε .- eτ).^2) .* (weights .^ 2))
        dψ_manual = mean(weights)

        @test isapprox(s2, s2_manual; atol=1e-12)
        @test isapprox(dψ, dψ_manual; atol=1e-12)

        @test s2 ≥ 0
        @test dψ > 0
    end

    # ------------------------------------------------------------
    # 4. residual_expectile wrappers
    # ------------------------------------------------------------
    @testset "residual_expectile wrappers" begin
        y = [1.0, -2.0, 0.5, 1.5]
        σ2 = [1.0, 4.0, 0.25, 2.25]
        τ = 0.7

        ε = y ./ sqrt.(σ2)

        e1 = residual_expectile(y, σ2, τ)
        e2 = expectile(ε, τ)

        @test isapprox(e1, e2; atol=1e-12)
    end

    # ------------------------------------------------------------
    # 5. known-scale expectile variance
    # ------------------------------------------------------------
    @testset "expectile_var with known scale" begin
        y = [1.0, -2.0, 0.5, 1.5, -0.25]
        σ2 = [1.0, 4.0, 0.25, 2.25, 0.0625]
        τ = 0.8

        ε = y ./ sqrt.(σ2)
        eτ = expectile(ε, τ)
        s2, dψ = aux_expectile_var(ε, eτ, τ)
        v_manual = s2 / dψ^2

        v_fun = expectile_var(y, σ2, τ)

        @test isapprox(v_fun, v_manual; atol=1e-12)
        @test v_fun ≥ 0
    end

    # ------------------------------------------------------------
    # 6. kernel derivative helper sanity
    # ------------------------------------------------------------
    @testset "kernel derivatives" begin
        ε = [-1.0, 0.0, 1.0, 2.0]

        g1 = zeros(length(ε))
        g2 = zeros(length(ε))
        GARCHModels._kernel_derivatives!(g1, g2, ε, Normal())

        @test all(isfinite, g1)
        @test all(isfinite, g2)

        # For Normal and x = 0, g(x,s) = -log s - 0.5(x/s)^2
        # g1(0;1) = -1, g2(0;1) = 1
        @test isapprox(g1[2], -1.0; atol=1e-8)
        @test isapprox(g2[2],  1.0; atol=1e-8)
    end

    # ------------------------------------------------------------
    # 7. GARCH bridge: residual expectile from model
    # ------------------------------------------------------------
    @testset "residual_expectile with fitted model" begin
        Random.seed!(1234)

        true_model = GARCHModel(0.1, [0.05], [0.9], nothing, Normal())
        y, σ2 = simulate(true_model, 1000)

        e1 = residual_expectile(y, true_model, 0.8)
        e2 = residual_expectile(y, garch_variance(y, true_model), 0.8)

        @test isapprox(e1, e2; atol=1e-8)
    end

    # ------------------------------------------------------------
    # 8. expectile_var with fitted model
    # ------------------------------------------------------------
    @testset "expectile_var with fitted model" begin
        Random.seed!(4321)

        true_model = GARCHModel(0.1, [0.05], [0.9], nothing, Normal())
        y, _ = simulate(true_model, 1200)

        fit = GARCHModels.estimate(y, 1, 1, gjr=false, dist=Normal())
        v = expectile_var(y, fit, 0.8)

        @test isfinite(v)
        @test v ≥ 0
    end

    # ------------------------------------------------------------
    # 9. conditional_expectile matches forecast sigma times residual expectile
    # ------------------------------------------------------------
    @testset "conditional_expectile identity" begin
        Random.seed!(2026)

        model = GARCHModel(0.1, [0.05], [0.9], nothing, Normal())
        y, _ = simulate(model, 800)

        τ = 0.8
        ξ = residual_expectile(y, model, τ)
        σ2_next = forecast_variance(y, model)

        ce = conditional_expectile(y, model, τ)

        @test isapprox(ce, sqrt(σ2_next) * ξ; atol=1e-10)
    end

    # ------------------------------------------------------------
    # 10. conditional expectile variance
    # ------------------------------------------------------------
    @testset "conditional_expectile_var" begin
        Random.seed!(999)

        model = GARCHModel(0.1, [0.05], [0.9], nothing, Normal())
        y, _ = simulate(model, 1500)

        fit = GARCHModels.estimate(y, 1, 1, gjr=false, dist=Normal())

        v = conditional_expectile_var(y, fit, 0.8)

        @test isfinite(v)
        @test v ≥ 0
    end

    # ------------------------------------------------------------
    # 11. GJR path sanity
    # ------------------------------------------------------------
    @testset "GJR-GARCH expectile sanity" begin
        Random.seed!(321)

        model = GARCHModel(0.1, [0.06], [0.85], 0.05, Normal())
        y, _ = simulate(model, 1200)

        fit = GARCHModels.estimate(y, 1, 1, gjr=true, dist=Normal())

        e = residual_expectile(y, fit, 0.8)
        v = expectile_var(y, fit, 0.8)
        cv = conditional_expectile_var(y, fit, 0.8)

        @test isfinite(e)
        @test isfinite(v)
        @test isfinite(cv)

        @test v ≥ 0
        @test cv ≥ 0
    end

    # ------------------------------------------------------------
    # 12. left-tail / right-tail ordering
    # ------------------------------------------------------------
    @testset "expectile ordering across τ" begin
        Random.seed!(111)

        x = randn(2000)

        e1 = expectile(x, 0.2)
        e2 = expectile(x, 0.5)
        e3 = expectile(x, 0.8)

        @test e1 < e2 < e3
        @test abs(e2 - mean(x)) < 1e-2
    end
end
