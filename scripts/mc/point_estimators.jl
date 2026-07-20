using Statistics

function summarize_point_estimators(results::Vector, spec::MCSpec, true_xi::Float64)
    ω = [r.ω_hat for r in results]
    α1 = [r.α_hat[1] for r in results]
    β1 = [r.β_hat[1] for r in results]

    ξ = [r.xi_hat for r in results]
    ce_err = [r.ce_hat - r.ce_true for r in results]
    xi_err = [r.xi_hat - true_xi for r in results]
    sigma_err = [r.sigma_hat_next - r.sigma_true_next for r in results]
    persistence_hat = [r.fitted_persistence for r in results]
    information_condition = [r.information_condition for r in results]
    max_variance = [r.max_fitted_variance for r in results]

    out = Dict{String,Any}()

    out["omega_bias"] = mean(ω) - spec.model.ω
    out["omega_rmse"] = sqrt(mean((ω .- spec.model.ω).^2))
    # calculate bias and RMSE standard errors
    out["omega_se_bias"] = std(ω) / sqrt(length(ω))
    out["omega_se_rmse"] = std((ω .- spec.model.ω).^2) / sqrt(length(ω))

    out["alpha1_bias"] = mean(α1) - spec.model.α[1]
    out["alpha1_rmse"] = sqrt(mean((α1 .- spec.model.α[1]).^2))
    # calculate standard errors
    out["alpha1_se_bias"] = std(α1) / sqrt(length(α1))
    out["alpha1_se_rmse"] = std((α1 .- spec.model.α[1]).^2) / sqrt(length(α1))

    out["beta1_bias"] = mean(β1) - spec.model.β[1]
    out["beta1_rmse"] = sqrt(mean((β1 .- spec.model.β[1]).^2))
    # calculate standard errors
    out["beta1_se_bias"] = std(β1) / sqrt(length(β1))
    out["beta1_se_rmse"] = std((β1 .- spec.model.β[1]).^2) / sqrt(length(β1))

    out["alphabeta_bias"] = mean(α1 .+ β1) - (spec.model.α[1] + spec.model.β[1])
    out["alphabeta_rmse"] = sqrt(mean(((α1 .+ β1) .- (spec.model.α[1] + spec.model.β[1])).^2))
    # calculate standard errors
    out["alphabeta_se_bias"] = std(α1 .+ β1) / sqrt(length(α1 .+ β1))
    out["alphabeta_se_rmse"] = std(((α1 .+ β1) .- (spec.model.α[1] + spec.model.β[1])).^2) / sqrt(length(α1 .+ β1))

    out["xi_bias"] = mean(ξ) - true_xi
    out["xi_rmse"] = sqrt(mean((ξ .- true_xi).^2))
    # calculate standard errors
    out["xi_se_bias"] = std(ξ) / sqrt(length(ξ))
    out["xi_se_rmse"] = std((ξ .- true_xi).^2) / sqrt(length(ξ))

    out["ce_bias"] = mean(ce_err)
    out["ce_rmse"] = sqrt(mean(ce_err.^2))
    # calculate standard errors
    out["ce_se_bias"] = std(ce_err) / sqrt(length(ce_err))
    out["ce_se_rmse"] = std(ce_err.^2) / sqrt(length(ce_err))
    out["sigma_error_sd"] = std(sigma_err)
    out["xi_sigma_error_corr"] = cor(xi_err, sigma_err)
    out["persistence_hat_mean"] = mean(persistence_hat)
    out["persistence_hat_p99"] = quantile(persistence_hat, 0.99)
    out["near_stationarity_fraction"] = mean(persistence_hat .>= 0.995)
    out["information_condition_median"] = median(information_condition)
    out["information_condition_p99"] = quantile(information_condition, 0.99)
    out["max_variance_p99"] = quantile(max_variance, 0.99)
    out["abs_xi_error_p99"] = quantile(abs.(xi_err), 0.99)
    out["abs_ce_error_p99"] = quantile(abs.(ce_err), 0.99)

    if has_gjr(spec)
        γ = [r.γ_hat for r in results if r.γ_hat !== nothing]
        out["gamma_bias"] = mean(γ) - spec.model.γ
        out["gamma_rmse"] = sqrt(mean((γ .- spec.model.γ).^2))
        # calculate standard errors
        out["gamma_se_bias"] = std(γ) / sqrt(length(γ))
        out["gamma_se_rmse"] = std((γ .- spec.model.γ).^2) / sqrt(length(γ))
    end

    return out
end
