@testset "Gao--Song two-step VaR and ES inference" begin
    @test GaoSongRisk._empirical_quantile([1.0, 2.0, 3.0, 4.0], 0.50) == 2.0
    @test GaoSongRisk._empirical_quantile([1.0, 2.0, 3.0, 4.0], 0.51) == 3.0

    density_data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    density_point = 0.3
    scott_bandwidth = 1.06 * std(density_data) * length(density_data)^(-1 / 5)
    scott_density = mean(
        pdf(Normal(), (density_point - value) / scott_bandwidth) / scott_bandwidth
        for value in density_data
    )
    @test isapprox(
        GaoSongRisk._kernel_density(density_data, density_point),
        scott_density;
        atol=1e-14,
    )

    path_model = GARCHModel(0.08, [0.15], [0.80], nothing, Normal())
    path_data = [0.2, -0.4, 0.1]
    h, dh, h_next, dh_next = GaoSongRisk._variance_path_gradient(path_data, path_model)
    shared_h, shared_dh, shared_h_next, shared_dh_next =
        GARCHModels._risk_variance_path_gradient(path_data, path_model)
    legacy_dh, legacy_h, legacy_dh_next, legacy_h_next =
        GARCHModels._garch_variance_gradient(path_data, path_model)
    @test h == shared_h == legacy_h == garch_variance(path_data, path_model)
    @test dh == shared_dh == legacy_dh
    @test h_next == shared_h_next == legacy_h_next == forecast_variance(path_data, path_model)
    @test dh_next == shared_dh_next == legacy_dh_next
    intercept_tail = path_model.ω / (1 - path_model.β[1])
    @test isapprox(h[1], intercept_tail; atol=1e-12)
    @test isapprox(h[2], intercept_tail + path_model.α[1] * path_data[1]^2; atol=1e-12)
    @test isapprox(dh[1, 1], 1 / (1 - path_model.β[1]); atol=1e-12)
    @test isapprox(dh[1, 2], 0.0; atol=1e-12)
    @test isapprox(
        dh[1, 3],
        path_model.ω / (1 - path_model.β[1])^2;
        atol=1e-12,
    )
    @test isapprox(
        h_next,
        path_model.ω + path_model.α[1] * path_data[end]^2 + path_model.β[1] * h[end];
        atol=1e-12,
    )
    @test all(isfinite, dh_next)

    gjr_parameters = [0.08, 0.12, 0.75, 0.08]
    gjr_model(parameters) = GARCHModel(
        parameters[1], [parameters[2]], [parameters[3]], parameters[4], Normal(),
    )
    gjr_h, gjr_dh, gjr_h_next, gjr_dh_next = GaoSongRisk._variance_path_gradient(
        path_data, gjr_model(gjr_parameters),
    )
    step = 1e-6
    for parameter_index in eachindex(gjr_parameters)
        upper = copy(gjr_parameters)
        lower = copy(gjr_parameters)
        upper[parameter_index] += step
        lower[parameter_index] -= step
        upper_h, _, upper_next, _ = GaoSongRisk._variance_path_gradient(
            path_data, gjr_model(upper),
        )
        lower_h, _, lower_next, _ = GaoSongRisk._variance_path_gradient(
            path_data, gjr_model(lower),
        )
        numerical_path_gradient = (upper_h - lower_h) / (2 * step)
        numerical_next_gradient = (upper_next - lower_next) / (2 * step)
        @test isapprox(
            gjr_dh[:, parameter_index], numerical_path_gradient; atol=1e-8,
        )
        @test isapprox(
            gjr_dh_next[parameter_index], numerical_next_gradient; atol=1e-8,
        )
    end

    rng = MersenneTwister(7719)
    model = GARCHModel(0.08, [0.08], [0.88], nothing, Normal())
    y, _ = GARCHModels.simulate(rng, model, 1200; burnout=500)
    fit = GARCHModels.estimate(y, 1, 1; dist=Normal())

    results = Dict{Float64,Any}()
    for level in (0.01, 0.05)
        result = gao_song_fhs_intervals(y, fit, level; ci_level=0.90)
        results[level] = result
        @test result.diagnostics.tail_count > 0
        @test isfinite(result.density) && result.density > 0
        @test result.var.interval.valid
        @test result.es.interval.valid
        @test result.var.rootn_variance >= 0
        @test result.es.rootn_variance >= 0
        @test isapprox(
            result.var.interval.length,
            result.var.interval.upper - result.var.interval.lower;
            atol=1e-12,
        )
        @test isapprox(
            result.es.interval.length,
            result.es.interval.upper - result.es.interval.lower;
            atol=1e-12,
        )
    end
    result = results[0.05]

    threshold = -1.0
    alpha = mean(result.residuals .<= threshold)
    matched = gao_song_fhs_intervals(
        y, fit, alpha; threshold=threshold, ci_level=0.90,
    )
    @test matched.threshold == threshold
    @test isapprox(matched.var.estimate, matched.sigma_next * threshold; atol=1e-12)
    @test matched.var.interval.valid
    @test matched.es.interval.valid
end
